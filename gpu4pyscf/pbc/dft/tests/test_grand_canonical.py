from types import SimpleNamespace

import cupy as cp
import numpy as np
import pytest

from pyscf.pbc import gto

from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.dft import grand_canonical as gc
from gpu4pyscf.pbc.dft import grand_canonical_cg as gc_cg
from gpu4pyscf.pbc.dft.grand_canonical import GrandCanonicalKRKS


class _MockCell:
    precision = 1e-12
    nelectron = 2

    @staticmethod
    def get_scaled_kpts(kpts):
        return kpts


class _FixedFockKRKS:
    def __init__(self, fock, kpts=None):
        self.cell = _MockCell()
        self.kpts = (np.zeros((len(fock), 3)) if kpts is None
                     else np.asarray(kpts, dtype=float))
        self._fock = cp.stack(fock)
        self.verbose = 0
        self.stdout = None
        self.max_memory = 0
        self.scf_summary = {}
        self.veff_calls = 0
        self.energy_veff = None

    def get_ovlp(self, cell, kpts):
        return cp.stack([cp.eye(x.shape[0], dtype=x.dtype)
                         for x in self._fock])

    def get_hcore(self, cell, kpts):
        return self._fock

    def check_linear_dependency(self, overlap, **kwargs):
        return cp.stack([cp.eye(x.shape[0], dtype=x.dtype) for x in overlap])

    def get_init_guess(self, cell, kpts=None):
        return cp.stack([cp.eye(x.shape[0], dtype=x.dtype)
                         for x in self._fock])

    def get_veff(self, cell, dm, **kwargs):
        self.veff_calls += 1
        self.last_dm = dm
        return cp.zeros_like(dm)

    def energy_elec(self, dm, hcore, veff):
        self.energy_veff = veff
        energy = cp.einsum('kij,kji->', hcore, dm).real / len(dm)
        return energy, 0.

    @staticmethod
    def energy_nuc():
        return 0.


class _LinearFockKRKS(_FixedFockKRKS):
    def __init__(self, hcore, coupling=.2):
        super().__init__(hcore)
        self.coupling = coupling

    def get_veff(self, cell, dm, **kwargs):
        self.veff_calls += 1
        self.last_dm = dm
        return self.coupling * dm

    def energy_elec(self, dm, hcore, veff):
        self.energy_veff = veff
        nkpts = len(dm)
        e1 = cp.einsum('kij,kji->', hcore, dm).real / nkpts
        e2 = .5 * cp.einsum('kij,kji->', veff, dm).real / nkpts
        return e1+e2, e2


class _CountingSetupKRKS(_FixedFockKRKS):
    def __init__(self, fock):
        super().__init__(fock)
        self.setup_calls = dict(build=0, overlap=0, hcore=0, orth=0, enuc=0)

    def build(self):
        self.setup_calls['build'] += 1

    def get_ovlp(self, cell, kpts):
        self.setup_calls['overlap'] += 1
        return super().get_ovlp(cell, kpts)

    def get_hcore(self, cell, kpts):
        self.setup_calls['hcore'] += 1
        return super().get_hcore(cell, kpts)

    def check_linear_dependency(self, overlap, **kwargs):
        self.setup_calls['orth'] += 1
        return super().check_linear_dependency(overlap, **kwargs)

    def energy_nuc(self):
        self.setup_calls['enuc'] += 1
        return 0.


class _TaggedSolventKRKS(_FixedFockKRKS):
    def __init__(self, hcore, solvent):
        super().__init__(hcore)
        self.solvent = cp.stack(solvent)

    def get_veff(self, cell, dm, **kwargs):
        self.veff_calls += 1
        self.last_dm = dm
        return tag_array(cp.zeros_like(dm), v_solvent=self.solvent,
                         e_solvent=0.)

    def get_fock(self, h1e=None, vhf=None, dm=None, cycle=-1, diis=None,
                 level_shift_factor=None, damp_factor=None):
        assert cycle == -1 and diis is None
        return h1e + vhf + vhf.v_solvent

    def energy_elec(self, dm, hcore, veff):
        self.energy_veff = veff
        fock = hcore + veff.v_solvent
        energy = cp.einsum('kij,kji->', fock, dm).real / len(dm)
        return energy, 0.


def _fock():
    return cp.asarray(
        [[-0.7, 0.12j], [-0.12j, 0.3]], dtype=cp.complex128)


def _solver(mf=None, mu=None, nelec=None, sigma=.15):
    if mf is None:
        mf = _FixedFockKRKS([_fock()])
    solver = GrandCanonicalKRKS(mf, mu=mu, sigma=sigma, nelec=nelec)
    return mf, solver


def _line_sample(alpha, value, slope):
    return gc_cg._LineSample(
        alpha, None, value, None, slope, 'test')


def _scripted_state(h, fock, residual_rms, value, exact_gradient):
    return SimpleNamespace(
        h=[cp.asarray([[h]], dtype=cp.float64)],
        fock=[cp.asarray([[fock]], dtype=cp.float64)],
        residual_rms=residual_rms,
        free_energy=value,
        grand_potential=value,
        exact_gradient=[cp.asarray([[exact_gradient]], dtype=cp.float64)],
        mu=0.,
        nelec=1.,
    )


class _ScriptedNLCGSolver:
    nelec = None
    mu = 0.
    sigma = .1
    conv_tol = 1e-8
    max_cycle = 3
    nlcg_initial_step = 1.
    nlcg_max_line_search_evaluations = 6
    verbose = 0
    stdout = None

    def __init__(self, initial):
        self.initial = initial

    def build(self):
        return self

    def calculate_cycle(self, unused_h, nelec=None, mu=None):
        return self.initial

    @staticmethod
    def _inner(left, right):
        return sum(float(cp.vdot(x, y).real.item())
                   for x, y in zip(left, right))

    def _finalize(self, state, converged):
        self.converged = converged
        self.residual_rms = state.residual_rms
        self.grand_potential = state.grand_potential
        return state.grand_potential


def test_fermi_functions_are_stable():
    gamma = cp.asarray([-1000., -50., 0., 50., 1000.])
    occ = gc._fermi_occ(gamma)
    entropy = gc._fermi_entropy(occ)
    assert bool(cp.all(cp.isfinite(occ)))
    assert np.isfinite(float(entropy))
    assert float(occ.min()) >= 0.
    assert float(occ.max()) <= 1.
    assert abs(float((occ[0]+occ[-1]).item())-1.) < 1e-14


def test_nlcg_fermi_divided_difference_is_stable():
    gamma = cp.asarray([-1000., .4, .4, .4+1e-13, 1000.])
    occ = gc_cg._fermi_occ(gamma)
    divided = gc_cg.fermi_divided_difference(gamma, occ)
    expected_diagonal = -occ * (1.-occ)
    assert bool(cp.all(cp.isfinite(occ)))
    assert bool(cp.all(cp.isfinite(divided)))
    assert float(cp.max(cp.abs(divided-divided.T)).item()) < 1e-14
    assert float(cp.max(divided).item()) <= 0.
    assert float(cp.max(cp.abs(
        cp.diag(divided)-expected_diagonal)).item()) < 1e-14
    assert abs(float((divided[1, 2]-expected_diagonal[1]).item())) < 1e-14
    assert abs(float((divided[1, 3]-expected_diagonal[1]).item())) < 1e-10


def test_nlcg_harmonic_step_finds_positive_curvature_derivative_root():
    alpha, curvature, interval = gc_cg._harmonic_step([
        _line_sample(0., 4., -4.),
        _line_sample(3., 1., 2.),
    ])
    assert alpha == pytest.approx(2.)
    assert curvature == pytest.approx(2.)
    assert interval == (0., 3.)


def test_nlcg_quadratic_step_finds_convex_minimum():
    alphas = np.linspace(0., 3., 3)
    values = (alphas-1.25)**2 + 7.
    alpha, curvature = gc_cg._convex_quadratic_step(
        alphas, values, (0., 3.))
    assert alpha == pytest.approx(1.25, abs=1e-8)
    assert curvature > 0.


def test_nlcg_quadratic_step_rejects_concave_stationary_point():
    alphas = np.asarray([0., 1., 2.])
    values = -(alphas-1.)**2
    assert gc_cg._convex_quadratic_step(
        alphas, values, (0., 2.)) == (None, None)


def test_nlcg_hermite_step_finds_convex_minimum():
    left = _line_sample(0., 1.25**2, -2.5)
    right = _line_sample(3., (3.-1.25)**2, 3.5)
    alpha, curvature, value = gc_cg._hermite_step(left, right)
    assert alpha == pytest.approx(1.25)
    assert curvature > 0.
    assert value == pytest.approx(0.)


def test_nlcg_hermite_step_rejects_concave_minimum():
    left = _line_sample(0., -1., 2.)
    right = _line_sample(2., -1., -2.)
    assert gc_cg._hermite_step(left, right) == (None, None, None)


def test_nlcg_orbital_rotation_is_absolute_and_preserves_spectrum():
    state = SimpleNamespace(
        eig=[cp.asarray([-1., 2.])],
        coeff=[cp.eye(2, dtype=cp.complex128)],
    )
    generator = cp.asarray(
        [[0., .2+.1j], [-.2+.1j, 0.]], dtype=cp.complex128)
    first = gc_cg._orbital_hamiltonians(state, [generator], .3)[0]
    repeated = gc_cg._orbital_hamiltonians(state, [generator], .3)[0]
    farther = gc_cg._orbital_hamiltonians(state, [generator], .6)[0]
    assert cp.allclose(first, repeated)
    assert not cp.allclose(first, farther)
    assert cp.allclose(cp.linalg.eigvalsh(first), state.eig[0])
    assert cp.allclose(cp.linalg.eigvalsh(farther), state.eig[0])


def test_nlcg_orbital_quadratic_step_requires_a_convex_bracket():
    convex = [
        _line_sample(1., .09, np.nan),
        _line_sample(2., .49, np.nan),
    ]
    assert gc_cg._orbital_quadratic_step(
        1.69, convex, convex[0]) == pytest.approx(1.3)
    concave = [
        _line_sample(1., -.09, np.nan),
        _line_sample(2., -.49, np.nan),
    ]
    assert gc_cg._orbital_quadratic_step(
        -1.69, concave, concave[0]) is None


def test_nlcg_orbital_lbfgs_history_keeps_a_descent_direction():
    solver = SimpleNamespace()
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))

    def state(off_diagonal):
        return SimpleNamespace(
            eig=[cp.asarray([-1., 1.])],
            coeff=[cp.eye(2)],
            occ=[cp.asarray([1., 0.])],
            fock=[cp.asarray(
                [[0., off_diagonal], [off_diagonal, 0.]])],
        )

    history = gc_cg._OrbitalHistory()
    first = gc_cg._orbital_rotation_data(solver, state(.4), history)
    assert first['method'] == 'orbital-steepest'
    history.previous_gradient = [
        value.copy() for value in first['true_gradient']]
    history.previous_step = [
        .1*value for value in first['direction']]
    second = gc_cg._orbital_rotation_data(solver, state(.2), history)
    assert second['method'] == 'orbital-lbfgs-1'
    assert len(history.pairs) == 1
    assert second['direction_metric'] > 0.


def test_nlcg_orbital_line_search_caches_absolute_samples(monkeypatch):
    solver = SimpleNamespace(
        nlcg_max_line_search_evaluations=10,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])],
        value=.0025**2,
        residual_rms=1.,
    )
    history = gc_cg._OrbitalHistory()
    seen = []
    data = {
        'gradients': [cp.asarray([[1.]])],
        'true_gradient': [cp.asarray([[-1.]])],
        'direction': [cp.asarray([[1.]])],
        'generators': [cp.asarray([[1.]])],
        'norm': 1.,
        'gradient_metric': 1.,
        'direction_metric': 1.,
        'method': 'orbital-steepest',
    }

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        seen.append(alpha)
        return SimpleNamespace(
            h=h,
            value=(alpha-.0025)**2,
            residual_rms=abs(alpha-.0025),
        )

    monkeypatch.setattr(
        gc_cg, '_orbital_rotation_data',
        lambda unused_solver, unused_origin, unused_history: data)
    monkeypatch.setattr(
        gc_cg, '_orbital_hamiltonians',
        lambda state, unused_generators, alpha: [
            state.h[0]+cp.asarray([[alpha]])])
    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, unused_state, unused_fixed_n: [
            cp.asarray([[0.]])])
    result = gc_cg._orbital_line_search(
        solver, origin, evaluate, lambda state: state.value, history)
    assert result.sample is not None
    assert result.resolved
    assert result.sample.alpha == pytest.approx(.0025)
    assert seen == pytest.approx([.001, .002, .004, .0025])
    assert len(seen) == len(set(seen))


def test_nlcg_line_consistency_detects_energy_slope_disagreement():
    origin_value = -10.
    exact = [
        _line_sample(0., 1., -2.),
        _line_sample(2., 1., 2.),
    ]
    inconsistent = [
        exact[0],
        _line_sample(2., 1.+1e-6, 2.),
    ]
    floor = gc_cg._line_roundoff(origin_value)
    assert gc_cg._line_consistency(
        exact, (0., 2.), origin_value) == pytest.approx(floor)
    assert gc_cg._line_consistency(
        inconsistent, (0., 2.), origin_value) == pytest.approx(1e-6)


def test_nlcg_strict_line_metrics_use_bracket_width_and_normalized_slope():
    origin = _line_sample(0., 1., -2.)
    candidate = _line_sample(1., 0., .01)
    samples = [origin, candidate, _line_sample(2., 1., 2.)]
    selected, interval, alpha_uncertainty, normalized_slope = (
        gc_cg._strict_line_metrics(
            samples, origin, consistency=1e-8, direction_norm=2.))
    assert selected is candidate
    assert interval == (0., 1.)
    assert alpha_uncertainty == pytest.approx(1.)
    assert normalized_slope == pytest.approx(.005)


def test_nlcg_strict_line_metrics_are_direction_scale_invariant():
    scale = 1e6
    origin = _line_sample(0., 1., -2.)
    candidate = _line_sample(1., 0., .01)
    base = gc_cg._strict_line_metrics(
        [origin, candidate], origin, consistency=1e-8,
        direction_norm=1.)
    scaled_origin = _line_sample(0., 1., -2.*scale)
    scaled_candidate = _line_sample(1./scale, 0., .01*scale)
    scaled = gc_cg._strict_line_metrics(
        [scaled_origin, scaled_candidate], scaled_origin,
        consistency=1e-8, direction_norm=scale)
    assert base[2] == pytest.approx(scaled[2])
    assert base[3] == pytest.approx(scaled[3])


def test_nlcg_strict_relative_alpha_is_infinite_near_zero():
    origin = _line_sample(0., 1., -1.)
    candidate = _line_sample(np.nextafter(0., 1.), 0., 1.)
    selected, interval, alpha_uncertainty, unused_slope = (
        gc_cg._strict_line_metrics(
            [origin, candidate], origin, consistency=1e-8,
            direction_norm=1.))
    assert selected is None
    assert interval is None
    assert np.isinf(alpha_uncertainty)


def test_nlcg_objective_equivalence_is_fixed_at_one_e_minus_eight():
    assert gc_cg._line_objective_band(-825., 1.) == pytest.approx(1e-8)
    assert gc_cg._line_objective_band(-825., 1e-10) == pytest.approx(1e-8)


def test_nlcg_origin_contractions_do_not_count_as_minimum_bracket():
    samples = [
        _line_sample(0., 0., -1.),
        _line_sample(.5, 1., -1.),
        _line_sample(1., 2., -1.),
    ]
    assert gc_cg._active_line_interval(samples) == (0., .5)
    assert gc_cg._minimum_line_interval(samples) is None


def test_nlcg_origin_best_contracts_before_remote_slope_interpolation():
    samples = [
        _line_sample(0., 0., -1.),
        _line_sample(.25, 1., -.5),
        _line_sample(1., 2., 1.),
    ]
    alpha, method = gc_cg._line_search_proposal(
        samples, [], initial_step=1., consistency=1., flat=True)
    assert alpha == pytest.approx(.125)
    assert method == 'contract'


def test_nlcg_functional_duplicate_uses_consistency_band():
    samples = [_line_sample(0., 0., -1.)]
    assert not gc_cg._line_candidate_is_new(
        1e-4, samples, consistency=1e-3)
    assert gc_cg._line_candidate_is_new(
        1e-2, samples, consistency=1e-3)


def test_nlcg_flat_bracket_probes_harmonic_root_to_refine_consistency():
    samples = [
        _line_sample(0., 0., -1.),
        _line_sample(1., -.1, 1.),
    ]
    alpha, method = gc_cg._line_search_proposal(
        samples, [], initial_step=1., consistency=1., flat=True)
    assert alpha == pytest.approx(.5)
    assert method == 'consistency-harmonic'


def test_nlcg_residual_line_keeps_refining_while_slope_improves(monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-12,
        nlcg_max_line_search_evaluations=5,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=0., residual_rms=1.,
        gradient=[cp.asarray([[-1.]])])
    proposals = iter([
        (1., 'initial'),
        (.5, 'hermite'),
        (.25, 'hermite'),
        (.2, 'consistency-harmonic'),
        (.19, 'harmonic'),
    ])
    slopes = iter([1., .8, .6, .1, .01])
    seen = []

    def evaluate(h):
        seen.append(float(h[0][0, 0].item()))
        return SimpleNamespace(
            h=h, value=0., residual_rms=1.,
            gradient=[cp.asarray([[next(slopes)]])])

    monkeypatch.setattr(
        gc_cg, '_line_search_proposal',
        lambda *unused_args, **unused_kwargs: next(proposals))
    monkeypatch.setattr(
        gc_cg, '_active_line_interval', lambda *unused_args: (0., 2.))
    monkeypatch.setattr(
        gc_cg, '_minimum_line_interval', lambda *unused_args: (0., 2.))
    monkeypatch.setattr(
        gc_cg, '_line_consistency', lambda *unused_args: 1.)
    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, origin.gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1., allow_restoration=True)
    assert result.sample is None
    assert result.evaluations == 5
    assert seen == pytest.approx([1., .5, .25, .2, .19])


def test_nlcg_strict_line_uses_full_budget_after_consistency_probe(
        monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-12,
        nlcg_max_line_search_evaluations=5,
        nlcg_line_search_alpha_rtol=1e-12,
        nlcg_line_search_slope_atol=1e-12,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=1., residual_rms=1.,
        gradient=[cp.asarray([[-1.]])])
    proposals = iter([
        (1., 'initial'),
        (.5, 'consistency-harmonic'),
        (.25, 'consistency-bisect'),
        (.2, 'consistency-bisect'),
        (.19, 'consistency-bisect'),
    ])
    slopes = iter([1., .8, .6, .4, .2])
    seen = []

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        seen.append(alpha)
        return SimpleNamespace(
            h=h, value=0., residual_rms=1.,
            gradient=[cp.asarray([[next(slopes)]])])

    monkeypatch.setattr(
        gc_cg, '_line_search_proposal',
        lambda *unused_args, **unused_kwargs: next(proposals))
    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, origin.gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.)
    assert result.sample is not None
    assert not result.resolved
    assert result.evaluations == 5
    assert result.reason == 'line-search evaluation limit'
    assert seen == pytest.approx([1., .5, .25, .2, .19])


def test_nlcg_strict_line_requires_alpha_and_slope_targets(monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-12,
        nlcg_max_line_search_evaluations=2,
        nlcg_line_search_alpha_rtol=2.,
        nlcg_line_search_slope_atol=.1,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=1., residual_rms=1.,
        gradient=[cp.asarray([[-1.]])])

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        return SimpleNamespace(
            h=h, value=(alpha-1.)**2, residual_rms=1.,
            gradient=[cp.asarray([[.2 if alpha >= .99 else -.2]])])

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, origin.gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.)
    assert not result.resolved
    assert result.alpha_relative_uncertainty <= 2.
    assert result.normalized_slope == pytest.approx(.2)
    assert result.evaluations == 2


def test_nlcg_strict_line_resolves_when_both_targets_are_met(monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-12,
        nlcg_max_line_search_evaluations=6,
        nlcg_line_search_alpha_rtol=.25,
        nlcg_line_search_slope_atol=1e-12,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=1.3**2, residual_rms=1.,
        gradient=[cp.asarray([[-2.6]])])

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        return SimpleNamespace(
            h=h, value=(alpha-1.3)**2, residual_rms=1.,
            gradient=[cp.asarray([[2.*(alpha-1.3)]])])

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, origin.gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.)
    assert result.resolved
    assert result.reason == 'resolved strict line minimum'
    assert result.sample.alpha == pytest.approx(1.3)
    assert result.alpha_relative_uncertainty == pytest.approx(.3/1.3)
    assert result.normalized_slope <= 1e-12
    assert result.evaluations == 3


def test_nlcg_restoration_selection_uses_origin_energy_cap():
    origin = _line_sample(0., 0., -1.)
    origin.state = SimpleNamespace(residual_rms=1.)
    objective_best = _line_sample(1., -9e-7, 0.)
    objective_best.state = SimpleNamespace(residual_rms=.9)
    residual_best = _line_sample(2., 5e-7, 0.)
    residual_best.state = SimpleNamespace(residual_rms=.5)
    samples = [origin, objective_best, residual_best]
    assert gc_cg._lexicographic_line_sample(
        samples, origin, 1e-6) is objective_best
    assert gc_cg._restoration_line_sample(
        samples, origin, 1e-6) is residual_best


def test_nlcg_inexact_selection_leaves_residual_growth_to_outer_retry():
    origin = _line_sample(0., 0., -1.)
    origin.state = SimpleNamespace(residual_rms=1.)
    unsafe_minimum = _line_sample(1., -2., 0.)
    unsafe_minimum.state = SimpleNamespace(residual_rms=1.6)
    safe_fallback = _line_sample(2., -1., 0.)
    safe_fallback.state = SimpleNamespace(residual_rms=1.4)
    assert gc_cg._inexact_line_sample(
        [origin, unsafe_minimum, safe_fallback], origin,
        consistency=.1) is unsafe_minimum


def test_nlcg_pareto_selection_requires_objective_and_residual_progress():
    origin = _line_sample(0., 0., -1.)
    origin.state = SimpleNamespace(residual_rms=1.)
    objective_only = _line_sample(1., -2., 0.)
    objective_only.state = SimpleNamespace(residual_rms=1.1)
    residual_only = _line_sample(2., 1., 0.)
    residual_only.state = SimpleNamespace(residual_rms=.5)
    pareto = _line_sample(3., -1., 0.)
    pareto.state = SimpleNamespace(residual_rms=.8)
    assert gc_cg._pareto_line_sample(
        [origin, objective_only, residual_only, pareto], origin) is pareto


def test_nlcg_pareto_progress_accepts_subpercent_residual_reduction():
    origin = _line_sample(0., 0., -1.)
    origin.state = SimpleNamespace(residual_rms=1.)
    trial = _line_sample(1e-3, -1., 0.)
    trial.state = SimpleNamespace(residual_rms=.995)
    assert gc_cg._pareto_improves(trial, origin)


@pytest.mark.parametrize(
    'objective_scale, expected_alpha, expected_resolved', [
        (1., 2., True),
        (1e-9, 1., False),
    ])
def test_nlcg_pareto_only_breaks_objective_ties(
        monkeypatch, objective_scale, expected_alpha, expected_resolved):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-12,
        nlcg_max_line_search_evaluations=6,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=4.*objective_scale,
        residual_rms=.01,
        gradient=[cp.asarray([[-4.*objective_scale]])])

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        residual = .0099 if alpha < 1.5 else .015
        return SimpleNamespace(
            h=h, value=objective_scale*(alpha-2.)**2,
            residual_rms=residual,
            gradient=[cp.asarray(
                [[2.*objective_scale*(alpha-2.)]])])

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, origin.gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1., allow_restoration=True)
    assert result.sample.alpha == pytest.approx(expected_alpha)
    assert result.evaluations == 2
    assert result.resolved is expected_resolved
    assert not result.restoration
    if expected_resolved:
        assert result.reason == 'resolved line minimum'
    else:
        assert result.reason == 'residual-descent truncation'


def test_nlcg_pareto_truncation_waits_for_near_solution_regime(monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-12,
        nlcg_max_line_search_evaluations=6,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=9., residual_rms=1.,
        gradient=[cp.asarray([[-6.]])])

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        residual = .99 if alpha < 1.5 else 1.5
        return SimpleNamespace(
            h=h, value=(alpha-3.)**2, residual_rms=residual,
            gradient=[cp.asarray([[2.*(alpha-3.)]])])

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, origin.gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1., allow_restoration=True)
    assert result.evaluations > 2


def test_nlcg_pulay_line_stops_after_two_nonimproving_residual_trials(
        monkeypatch):
    objective_scale = 1e-9
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-12,
        nlcg_max_line_search_evaluations=6,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=4.*objective_scale,
        residual_rms=1.,
        gradient=[cp.asarray([[-4.*objective_scale]])])
    seen = []

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        seen.append(alpha)
        return SimpleNamespace(
            h=h, value=objective_scale*(alpha-2.)**2,
            residual_rms=1.+.1*alpha,
            gradient=[cp.asarray(
                [[2.*objective_scale*(alpha-2.)]])])

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, origin.gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.,
        allow_restoration=True, require_residual_improvement=True)
    assert result.sample is None
    assert result.evaluations == 2
    assert seen == pytest.approx([1., 2.])
    assert result.reason == 'Pulay direction did not reduce the residual'


def test_nlcg_pulay_line_keeps_significant_objective_descent(monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-12,
        nlcg_max_line_search_evaluations=6,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=4., residual_rms=1.,
        gradient=[cp.asarray([[-4.]])])

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        return SimpleNamespace(
            h=h, value=(alpha-2.)**2,
            residual_rms=1.+.1*alpha,
            gradient=[cp.asarray([[2.*(alpha-2.)]])])

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, origin.gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.,
        allow_restoration=True, require_residual_improvement=True)
    assert result.sample.alpha == pytest.approx(2.)
    assert result.sample.state.residual_rms == pytest.approx(1.2)
    assert result.resolved
    assert result.reason == 'resolved line minimum'
    assert result.evaluations == 2


def test_nlcg_residual_secant_predicts_vector_least_squares_root():
    solver = SimpleNamespace()
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin = _line_sample(0., 0., 0.)
    origin.state = SimpleNamespace(residual=[cp.asarray([[1.]])])
    sample = _line_sample(1., 0., 0.)
    sample.state = SimpleNamespace(residual=[cp.asarray([[.75]])])
    assert gc_cg._residual_secant_step(
        solver, origin, sample) == pytest.approx(4.)


def test_nlcg_line_search_uses_absolute_unique_slots(monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-12,
        nlcg_max_line_search_evaluations=6,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin_h = 10.
    origin = SimpleNamespace(
        h=[cp.asarray([[origin_h]])],
        value=1.3**2,
        residual_rms=1.,
        gradient=[cp.asarray([[-2.6]])],
    )
    seen = []

    def evaluate(h):
        alpha = float(h[0][0, 0].item())-origin_h
        seen.append(alpha)
        return SimpleNamespace(
            h=h,
            value=(alpha-1.3)**2,
            residual_rms=1.,
            gradient=[cp.asarray([[2.*(alpha-1.3)]])],
        )

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, origin.gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.)
    assert result.resolved
    assert result.evaluations == 3
    assert seen[:2] == [1., 2.]
    assert seen[2] == pytest.approx(1.3)
    assert len(seen) == len(set(seen))


def test_nlcg_line_search_honors_evaluation_limit(monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-12,
        nlcg_max_line_search_evaluations=6,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=0., residual_rms=1.,
        gradient=[cp.asarray([[-1.]])])
    seen = []

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        seen.append(alpha)
        return SimpleNamespace(
            h=h, value=-alpha, residual_rms=1.,
            gradient=[cp.asarray([[-1.]])])

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, origin.gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.)
    assert not result.resolved
    assert result.evaluations == 6
    assert result.reason == 'line-search evaluation limit'
    assert seen == [1., 2., 4., 8., 16., 32.]


def test_nlcg_measured_inconsistency_does_not_hide_significant_gain(
        monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-12,
        nlcg_max_line_search_evaluations=2,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=0., residual_rms=1.,
        gradient=[cp.asarray([[-1e-3]])])

    def evaluate(h):
        return SimpleNamespace(
            h=h, value=-4e-7, residual_rms=.98,
            gradient=[cp.asarray([[1e-3]])])

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, origin.gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.)
    assert result.sample is not None
    assert not result.resolved
    assert result.sample.value == pytest.approx(-4e-7)
    assert result.consistency == pytest.approx(
        gc_cg.LINE_SEARCH_OBJECTIVE_FLAT)


def test_nlcg_significant_inexact_gain_allows_small_residual_growth(
        monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-12,
        nlcg_max_line_search_evaluations=2,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=0., residual_rms=1.,
        gradient=[cp.asarray([[-1e-3]])])

    def evaluate(h):
        return SimpleNamespace(
            h=h, value=-4e-7, residual_rms=1.001,
            gradient=[cp.asarray([[1e-3]])])

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, origin.gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.)
    assert result.sample is not None
    assert not result.resolved
    assert not result.restoration
    assert result.sample.value == pytest.approx(-4e-7)
    assert result.sample.state.residual_rms == pytest.approx(1.001)
    assert result.consistency == pytest.approx(
        gc_cg.LINE_SEARCH_OBJECTIVE_FLAT)


def test_nlcg_numerical_flat_inexact_gain_requires_meaningful_progress(
        monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-12,
        nlcg_max_line_search_evaluations=2,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=0., residual_rms=1.,
        gradient=[cp.asarray([[-1e-12]])])

    def evaluate(h):
        return SimpleNamespace(
            h=h, value=-1e-13, residual_rms=.999,
            gradient=[cp.asarray([[1e-12]])])

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, origin.gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.)
    assert result.sample is None


def test_nlcg_inexact_fallback_returns_growth_for_outer_retry(monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-12,
        nlcg_max_line_search_evaluations=2,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=0., residual_rms=1.,
        gradient=[cp.asarray([[-1e-3]])])

    def evaluate(h):
        return SimpleNamespace(
            h=h, value=-4e-7, residual_rms=1.6,
            gradient=[cp.asarray([[1e-3]])])

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, origin.gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.)
    assert result.sample is not None
    assert result.sample.state.residual_rms == pytest.approx(1.6)


def test_nlcg_canonical_charge_proposal_brackets_and_scales():
    proposal, bracket = gc_cg._canonical_charge_proposal(
        [(32.9, -.01), (33.1, .01)], 33.1, .01, 100.)
    assert proposal == pytest.approx(33.)
    assert bracket == pytest.approx((32.9, 33.1))

    proposal, bracket = gc_cg._canonical_charge_proposal(
        [(33., -1e-4)], 33., -1e-4, 100.)
    assert proposal == pytest.approx(33.001)
    assert bracket is None

    proposal, bracket = gc_cg._canonical_charge_proposal(
        [(33., -1.)], 33., -1., 100.)
    assert proposal == pytest.approx(
        33.+gc_cg.NLCG_CANONICAL_RESTORATION_NELEC_STEP)
    assert bracket is None

    proposal, bracket = gc_cg._canonical_charge_proposal(
        [(32.99825, -1.7e-4),
         (32.99994, -2e-6),
         (32.999945, -6e-6)],
        32.999945, -6e-6, 100.)
    assert 32.999945 < proposal < 33.0001
    assert bracket is None




def test_nlcg_reflected_residual_restoration_handles_flat_objective(
        monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-8,
        nlcg_max_line_search_evaluations=6,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    zero_gradient = [cp.asarray([[0.]])]
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=0., residual_rms=1.,
        gradient=zero_gradient)

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        return SimpleNamespace(
            h=h, value=0., residual_rms=abs(1.+alpha),
            gradient=zero_gradient)

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, zero_gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.,
        allow_restoration=True)
    assert result.sample is not None
    assert result.restoration
    assert result.sample.alpha == pytest.approx(-1.)
    assert result.sample.state.residual_rms == pytest.approx(0.)
    assert result.sample.value <= origin.value+result.consistency


def test_nlcg_reflection_does_not_require_an_energy_admissible_positive_slot(
        monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-8,
        nlcg_max_line_search_evaluations=6,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    zero_gradient = [cp.asarray([[0.]])]
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=0., residual_rms=1.,
        gradient=zero_gradient)
    proposals = iter([
        (1., 'initial'), (.5, 'contract'), (.25, 'contract'),
        (None, None),
    ])

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        return SimpleNamespace(
            h=h, value=(1e-3 if alpha > 0. else 0.),
            residual_rms=abs(1.+alpha), gradient=zero_gradient)

    monkeypatch.setattr(
        gc_cg, '_line_search_proposal',
        lambda *unused_args, **unused_kwargs: next(proposals))
    monkeypatch.setattr(
        gc_cg, '_active_line_interval', lambda *unused_args: (0., 1.))
    monkeypatch.setattr(
        gc_cg, '_minimum_line_interval', lambda *unused_args: (0., 1.))
    monkeypatch.setattr(
        gc_cg, '_line_consistency', lambda *unused_args: 1e-4)
    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, zero_gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.,
        allow_restoration=True)
    assert result.sample is not None
    assert result.restoration
    assert result.sample.alpha == pytest.approx(-1.)
    assert result.sample.state.residual_rms == pytest.approx(0.)
    assert result.sample.value <= origin.value+result.consistency


def test_nlcg_positive_residual_restoration_expands_to_secant_root(
        monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-8,
        nlcg_max_line_search_evaluations=8,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    zero_gradient = [cp.asarray([[0.]])]
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=0., residual_rms=1.,
        residual=[cp.asarray([[1.]])], gradient=zero_gradient)

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        residual = 1.-alpha/4.
        return SimpleNamespace(
            h=h, value=0., residual_rms=abs(residual),
            residual=[cp.asarray([[residual]])], gradient=zero_gradient)

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, zero_gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.,
        allow_restoration=True)
    assert result.sample is not None
    assert result.restoration
    assert result.reason == 'positive residual restoration'
    assert result.sample.alpha == pytest.approx(4.)
    assert result.sample.state.residual_rms == pytest.approx(0.)


def test_nlcg_positive_residual_secant_relinearizes_before_fallback(
        monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-8,
        nlcg_max_line_search_evaluations=8,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    zero_gradient = [cp.asarray([[0.]])]
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=0., residual_rms=1.,
        residual=[cp.asarray([1., 0.])], gradient=zero_gradient)

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        if np.isclose(alpha, 1.):
            residual = cp.asarray([.99, .1])
        elif .9 < alpha < 1.:
            residual = cp.asarray([.99, .08])
        elif 1. < alpha < 1.8:
            residual = cp.asarray([.98, 0.])
        else:
            residual = cp.asarray([1.1, 0.])
        return SimpleNamespace(
            h=h, value=0.,
            residual_rms=float(cp.linalg.norm(residual).item()),
            residual=[residual], gradient=zero_gradient)

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, zero_gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.,
        allow_restoration=True)
    assert result.sample is not None
    assert result.restoration
    assert result.reason == 'positive residual restoration'
    assert result.sample.state.residual_rms == pytest.approx(.98)
    assert result.sample.method == 'residual-secant'


def test_nlcg_reflected_restoration_rejects_excess_objective_increase(
        monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-8,
        nlcg_max_line_search_evaluations=6,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    zero_gradient = [cp.asarray([[0.]])]
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=0., residual_rms=1.,
        gradient=zero_gradient)

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        return SimpleNamespace(
            h=h, value=(1e-3 if alpha < 0. else 0.),
            residual_rms=abs(1.+alpha), gradient=zero_gradient)

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, zero_gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.,
        allow_restoration=True)
    assert result.sample is None
    assert result.consistency == pytest.approx(
        gc_cg._line_objective_band(
            origin.value, gc_cg._line_roundoff(origin.value)))


def test_nlcg_physical_step_transport_and_exact_gradient_scaling():
    solver = SimpleNamespace()
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    exact = [cp.asarray([[1e6]])]
    residual = [cp.asarray([[-1.]])]
    direction = gc_cg._scaled_exact_direction(solver, exact, residual)
    assert gc_cg._direction_norm(solver, direction) == pytest.approx(1.)
    assert gc_cg._step_from_displacement(1e6, 2.) == pytest.approx(2e-6)
    assert gc_cg._step_from_displacement(1e-6, 2.) == pytest.approx(2e6)


def test_nlcg_occupation_preconditioner_projects_null_and_damps_active_modes():
    solver = SimpleNamespace(
        beta=1e4,
        nelec=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    state = SimpleNamespace(
        eig=[cp.asarray([-1., 0.])],
        coeff=[cp.eye(2)],
        occ=[cp.asarray([1., .5])],
        residual=[cp.eye(2)],
        residual_rms=2.,
        mu=0.,
    )
    direction = gc_cg._occupation_preconditioned_direction(solver, state)
    assert direction is not None
    assert float(direction[0][0, 0].item()) == pytest.approx(0.)
    assert float(direction[0][1, 1].item()) < 5e-4
    assert cp.allclose(direction[0], direction[0].conj().T)


def test_nlcg_fermi_null_direction_excludes_fractional_mode():
    solver = SimpleNamespace(
        beta=1e4,
        nelec=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    state = SimpleNamespace(
        eig=[cp.asarray([-1., 0.])],
        coeff=[cp.eye(2)],
        occ=[cp.asarray([1., .5])],
        residual=[cp.eye(2)],
        residual_rms=2.,
        mu=0.,
    )
    direction = gc_cg._fermi_null_residual_direction(solver, state)
    assert direction is not None
    assert float(direction[0][0, 0].item()) == pytest.approx(1.)
    assert float(direction[0][1, 1].item()) == pytest.approx(0.)
    assert gc_cg._fermi_active_residual_rms(
        solver, state) == pytest.approx(np.sqrt(2.))


def test_nlcg_fermi_guard_limits_active_and_full_residual_growth(
        monkeypatch):
    solver = SimpleNamespace()
    origin = SimpleNamespace(active_residual=.01, residual_rms=.02)
    active_growth = SimpleNamespace(active_residual=.012,
                                    residual_rms=.021)
    null_growth = SimpleNamespace(active_residual=.0104,
                                  residual_rms=.04)
    monkeypatch.setattr(
        gc_cg, '_fermi_active_residual_rms',
        lambda unused_solver, state: state.active_residual)

    assert gc_cg._fermi_guarded_residual_rms(
        solver, origin, active_growth) == pytest.approx(.012)
    assert gc_cg._fermi_guarded_residual_rms(
        solver, origin, null_growth) == pytest.approx(.02)


def test_nlcg_occupation_direction_is_proactive_for_low_residual_fixed_n():
    direction = [cp.asarray([[1.]])]
    assert (gc_cg.NLCG_OCCUPATION_ACTIVE_GROWTH ==
            gc_cg.LINE_SEARCH_RESIDUAL_GROWTH_RESTART)
    assert gc_cg._occupation_direction_is_proactive(
        True, False, direction,
        gc_cg.LINE_SEARCH_PARETO_ACTIVE_RESIDUAL, True)
    assert not gc_cg._occupation_direction_is_proactive(
        False, False, direction, 1e-3, True)
    assert not gc_cg._occupation_direction_is_proactive(
        True, False, direction,
        2.*gc_cg.LINE_SEARCH_PARETO_ACTIVE_RESIDUAL, True)
    assert gc_cg._occupation_direction_is_proactive(
        True, True, direction,
        2.*gc_cg.LINE_SEARCH_PARETO_ACTIVE_RESIDUAL, False)


def test_nlcg_null_restoration_accepts_fixed_flat_energy_band(monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-12,
        nlcg_max_line_search_evaluations=6,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    origin = SimpleNamespace(
        h=[cp.zeros(2)], value=0., residual_rms=1.,
        gradient=[cp.zeros(2)])

    def evaluate(h):
        unused_x, y = (float(value.item()) for value in h[0])
        return SimpleNamespace(
            h=h,
            value=5e-9*y*y,
            residual_rms=abs(1.-.02*y),
            gradient=[cp.asarray([0., 1e-8*y])],
        )

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, origin.gradient, [cp.asarray([0., 1.])], evaluate,
        lambda state: state.value, initial_step=1.,
        allow_restoration=True, accept_bounded_residual=True)
    assert result.sample is not None
    assert result.restoration
    assert result.sample.method == 'initial'
    assert result.sample.state.residual_rms == pytest.approx(.98)
    assert result.sample.value == pytest.approx(5e-9)
    assert result.consistency == pytest.approx(1e-8)
    assert result.evaluations == 1


def test_nlcg_null_restoration_uses_energy_tangent_correction(monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-8,
        nlcg_max_line_search_evaluations=6,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    gradient = [cp.asarray([10., 0.])]
    origin = SimpleNamespace(
        h=[cp.zeros(2)], value=0., residual_rms=1., gradient=gradient)

    def evaluate(h):
        x, y = (float(value.item()) for value in h[0])
        return SimpleNamespace(
            h=h,
            value=10.*x+1e-3*y*y,
            residual_rms=abs(1.-.2*y),
            gradient=[cp.asarray([10., 2e-3*y])],
        )

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, gradient, [cp.asarray([0., 1.])], evaluate,
        lambda state: state.value, initial_step=1.,
        allow_restoration=True, accept_bounded_residual=True)
    assert result.sample is not None
    assert result.restoration
    assert result.sample.method == 'null-tangent'
    assert result.sample.state.residual_rms == pytest.approx(.8)
    assert result.sample.value <= origin.value+result.consistency
    assert result.evaluations == 2


def test_nlcg_null_restoration_refines_tangent_correction(monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-8,
        nlcg_max_line_search_evaluations=6,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    gradient = [cp.asarray([1., .1])]
    origin = SimpleNamespace(
        h=[cp.zeros(2)], value=0., residual_rms=1., gradient=gradient)

    def evaluate(h):
        x, y = (float(value.item()) for value in h[0])
        return SimpleNamespace(
            h=h, value=x+.1*y+2.*x*x, residual_rms=.98,
            gradient=[cp.asarray([1.+4.*x, .1])])

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, gradient, [cp.asarray([0., 1.])], evaluate,
        lambda state: state.value, initial_step=1.,
        allow_restoration=True, accept_bounded_residual=True)
    assert result.sample is not None
    assert result.sample.method == 'null-tangent-refine'
    assert result.sample.state.residual_rms == pytest.approx(.98)
    assert result.sample.value <= origin.value+result.consistency


def test_nlcg_null_probe_expands_before_tangent_correction(monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-8,
        nlcg_max_line_search_evaluations=6,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    gradient = [cp.asarray([10., 0.])]
    origin = SimpleNamespace(
        h=[cp.zeros(2)], value=0., residual_rms=1., gradient=gradient)
    probed_y = []

    def evaluate(h):
        x, y = (float(value.item()) for value in h[0])
        probed_y.append(y)
        return SimpleNamespace(
            h=h,
            value=10.*x+1e-8*y*y,
            residual_rms=abs(1.-.007*y),
            gradient=[cp.asarray([10., 2e-8*y])],
        )

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, gradient, [cp.asarray([0., 1.])], evaluate,
        lambda state: state.value, initial_step=1.,
        allow_restoration=True, accept_bounded_residual=True)
    assert probed_y[:2] == pytest.approx([1., 2.])
    assert result.sample is not None
    assert result.sample.method == 'null-tangent'
    assert result.sample.state.residual_rms == pytest.approx(.986)
    assert result.sample.value <= origin.value+result.consistency


def test_nlcg_null_restoration_cannot_inflate_its_energy_band(monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-8,
        nlcg_max_line_search_evaluations=6,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    gradient = [cp.asarray([[0.]])]
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=0., residual_rms=1.,
        gradient=gradient)

    def evaluate(h):
        return SimpleNamespace(
            h=h, value=.5, residual_rms=.5, gradient=gradient)

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.,
        allow_restoration=True, accept_bounded_residual=True)
    assert result.sample is None
    assert result.consistency == pytest.approx(
        gc_cg.LINE_SEARCH_OBJECTIVE_FLAT)


def test_nlcg_restoration_does_not_reclassify_objective_as_flat():
    significant_change = -2e-5
    restoration = gc_cg._LineSearchResult(
        _line_sample(1., 5e-9, 0.), False, 1,
        'bounded restoration', restoration=True,
        consistency=gc_cg.LINE_SEARCH_OBJECTIVE_FLAT)
    assert gc_cg._nonrestoration_objective_change(
        significant_change, 0., restoration) == significant_change

    objective_search = gc_cg._LineSearchResult(
        _line_sample(1., -5e-9, 0.), False, 1,
        'flat objective search', restoration=False,
        consistency=gc_cg.LINE_SEARCH_OBJECTIVE_FLAT)
    assert gc_cg._nonrestoration_objective_change(
        significant_change, 0., objective_search) == pytest.approx(-5e-9)


def test_nlcg_proactive_null_cleanup_waits_for_flat_objective(monkeypatch):
    initial = _scripted_state(0., 2., 1., 0., -1.)
    downhill = _scripted_state(1., 3., 2., -1e-4, -1.)
    converged = _scripted_state(2., 2., 0., -2e-4, 0.)
    for state, residual in ((initial, 1.), (downhill, 1.),
                            (converged, 0.)):
        state.residual = [cp.asarray([[residual]])]

    solver = _ScriptedNLCGSolver(initial)
    solver.nelec = 1.
    solver.sigma = 5e-3
    solver.max_cycle = 2
    directions = []

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.exact_gradient)
    monkeypatch.setattr(
        gc_cg, '_occupation_preconditioned_direction',
        lambda *unused_args: [cp.asarray([[2.]])])
    monkeypatch.setattr(
        gc_cg, '_occupation_direction_is_proactive',
        lambda *unused_args: True)
    monkeypatch.setattr(
        gc_cg, '_fermi_null_residual_direction',
        lambda *unused_args: [cp.asarray([[1.]])])

    def scripted_line_search(
            solver, unused_origin, unused_origin_gradient, direction,
            unused_evaluate, unused_objective, unused_initial_step,
            **unused_kwargs):
        directions.append(float(direction[0][0, 0].item()))
        state = downhill if len(directions) == 1 else converged
        sample = gc_cg._LineSample(
            1., state, state.free_energy, state.exact_gradient,
            solver._inner(state.exact_gradient, direction), 'scripted')
        return gc_cg._LineSearchResult(
            sample, True, 1,
            ('converged line sample' if state is converged else
             'resolved line minimum'))

    monkeypatch.setattr(gc_cg, '_line_search', scripted_line_search)
    gc_cg.nlcg(solver, h=initial.h)
    assert solver.converged
    assert directions == pytest.approx([2., 2.])


def test_nlcg_inconsistent_null_overshoot_contracts_without_tangent(
        monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-8,
        nlcg_max_line_search_evaluations=6,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    gradient = [cp.asarray([1., 0.])]
    origin = SimpleNamespace(
        h=[cp.zeros(2)], value=0., residual_rms=1.,
        gradient=gradient)
    evaluated = []

    def evaluate(h):
        x, y = (float(value.item()) for value in h[0])
        evaluated.append((x, y))
        return SimpleNamespace(
            h=h, value=x+(.1 if y > .75 else 0.),
            residual_rms=.5,
            gradient=gradient)

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, gradient, [cp.asarray([0., 1.])], evaluate,
        lambda state: state.value, initial_step=1.,
        allow_restoration=True, accept_bounded_residual=True)
    assert result.sample is not None
    assert result.restoration
    assert result.sample.alpha == pytest.approx(.5)
    assert result.evaluations == 2
    assert evaluated == pytest.approx([(0., 1.), (0., .5)])


def test_nlcg_null_restoration_stops_after_two_stagnant_trials(monkeypatch):
    solver = SimpleNamespace(
        nelec=None,
        conv_tol=1e-8,
        nlcg_max_line_search_evaluations=20,
        nlcg_line_search_alpha_rtol=1e-12,
        nlcg_line_search_slope_atol=1e-12,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    gradient = [cp.asarray([[0.]])]
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=0., residual_rms=1.,
        gradient=gradient)

    def evaluate(h):
        return SimpleNamespace(
            h=h, value=0., residual_rms=.995, gradient=gradient)

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.,
        allow_restoration=True, accept_bounded_residual=True)
    assert result.sample is None
    assert result.evaluations == 2


@pytest.mark.parametrize('sigma', [1e-4, 5e-3])
def test_nlcg_occupation_line_keeps_significant_objective_descent(
        monkeypatch, sigma):
    solver = SimpleNamespace(
        nelec=1.,
        sigma=sigma,
        conv_tol=1e-8,
        nlcg_max_line_search_evaluations=6,
        verbose=0,
        stdout=None,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    gradient = [cp.asarray([[-4.]])]
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=4., residual_rms=.01,
        gradient=gradient)

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        return SimpleNamespace(
            h=h, value=(alpha-2.)**2,
            residual_rms=.01-.003*alpha+.0035*alpha**2,
            gradient=[cp.asarray([[2.*(alpha-2.)]])])

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.,
        bound_residual_growth=True)
    assert result.sample is not None
    assert result.sample.alpha == pytest.approx(2.)
    assert result.sample.state.residual_rms == pytest.approx(.018)
    assert result.reason == 'resolved line minimum'
    assert result.resolved
    assert result.evaluations == 2


def test_nlcg_sharp_sigma_can_guard_an_active_residual_metric(monkeypatch):
    solver = SimpleNamespace(
        nelec=1., sigma=1e-4, conv_tol=1e-8,
        nlcg_max_line_search_evaluations=6, verbose=0, stdout=None)
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    gradient = [cp.asarray([[-1.]])]
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=0., residual_rms=.01,
        active_residual=.01, gradient=gradient)

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        return SimpleNamespace(
            h=h, value=-.25e-8*alpha, residual_rms=.1,
            active_residual=(.0104 if alpha <= .5 else .012),
            gradient=gradient)

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.,
        bound_residual_growth=True,
        residual_metric=lambda state: state.active_residual,
        residual_growth_limit=1.05)
    assert result.sample is not None
    assert result.sample.alpha == pytest.approx(.5)
    assert result.sample.state.residual_rms == pytest.approx(.1)
    assert gc_cg._sample_residual(result.sample) == pytest.approx(.0104)


def test_nlcg_sharp_sigma_contracts_to_residual_growth_boundary(
        monkeypatch):
    solver = SimpleNamespace(
        nelec=1., sigma=1e-4, conv_tol=1e-8,
        nlcg_max_line_search_evaluations=6, verbose=0, stdout=None)
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    gradient = [cp.asarray([[-1.]])]
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=0., residual_rms=.01,
        gradient=gradient)

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        residual = .012 if alpha <= .5 else .02
        return SimpleNamespace(
            h=h, value=-.25e-8*alpha, residual_rms=residual,
            gradient=gradient)

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=1.,
        bound_residual_growth=True)
    assert result.sample is not None
    assert result.sample.alpha == pytest.approx(.5)
    assert result.sample.state.residual_rms == pytest.approx(.012)
    assert result.reason == 'occupation residual-growth contraction'
    assert result.evaluations == 2


def test_nlcg_residual_contraction_yields_to_significant_objective_gain(
        monkeypatch):
    solver = SimpleNamespace(
        nelec=1., sigma=5e-3, conv_tol=1e-8,
        nlcg_max_line_search_evaluations=6, verbose=0, stdout=None)
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    gradient = [cp.asarray([[-3.6]])]
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=1.8**2, residual_rms=.01,
        gradient=gradient)
    evaluated = []

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        evaluated.append(alpha)
        return SimpleNamespace(
            h=h, value=(alpha-1.8)**2,
            residual_rms=(.012 if alpha <= 1. else .025),
            gradient=[cp.asarray([[2.*(alpha-1.8)]])])

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=4.,
        bound_residual_growth=True)
    assert result.sample is not None
    assert result.sample.alpha == pytest.approx(1.8)
    assert result.sample.state.residual_rms == pytest.approx(.025)
    assert result.resolved
    assert result.reason == 'resolved line minimum'
    assert evaluated == pytest.approx([4., 2., 1.8])


def test_nlcg_refines_smooth_residual_growth_boundary(monkeypatch):
    solver = SimpleNamespace(
        nelec=1., sigma=5e-3, conv_tol=1e-8,
        nlcg_max_line_search_evaluations=6, verbose=0, stdout=None)
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    gradient = [cp.asarray([[-1.]])]
    origin = SimpleNamespace(
        h=[cp.asarray([[0.]])], value=0., residual_rms=.01,
        gradient=gradient)

    def evaluate(h):
        alpha = float(h[0][0, 0].item())
        return SimpleNamespace(
            h=h, value=-.25e-8*alpha,
            residual_rms=.01-.003*alpha+.0035*alpha**2,
            gradient=gradient)

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.gradient)
    result = gc_cg._line_search(
        solver, origin, gradient, [cp.asarray([[1.]])], evaluate,
        lambda state: state.value, initial_step=2.,
        bound_residual_growth=True)

    expected = 1.+gc_cg.LINE_SEARCH_RESIDUAL_BOUNDARY_SAFETY*.6
    assert result.sample is not None
    assert result.sample.alpha == pytest.approx(expected)
    assert result.sample.state.residual_rms < 1.5*origin.residual_rms
    assert result.reason == (
        'occupation residual-growth boundary refinement')
    assert result.evaluations == 3


def test_nlcg_canonical_restoration_honors_evaluation_limit():
    unused, solver = _solver(mu=-.1, sigma=1e-4)
    solver.nlcg_max_line_search_evaluations = 2
    solver.build()
    h = [_fock()+cp.asarray([[.02, .01], [.01, -.01]])]
    origin = solver.calculate_cycle(h, mu=solver.mu)
    starting_nfev = solver.nfev
    result = gc_cg._canonical_restoration(
        solver, origin, solver.mu,
        lambda state: state.grand_potential,
        consistency=1., charge_history=[])
    assert result.evaluations <= 2
    assert solver.nfev-starting_nfev == result.evaluations


def test_nlcg_canonical_restoration_damping_is_scale_aware():
    solver = SimpleNamespace(conv_tol_coarse=1e-4)
    coarse = SimpleNamespace(residual_rms=1e-5)
    sharp_sigma_plateau = SimpleNamespace(residual_rms=1e-6)
    assert gc_cg._canonical_restoration_damp(
        solver, coarse) == gc_cg.NLCG_CANONICAL_RESTORATION_DAMP
    assert gc_cg._canonical_restoration_damp(
        solver, sharp_sigma_plateau) == 1.


def test_nlcg_repeated_weak_restoration_escalates_canonical(monkeypatch):
    initial = _scripted_state(0., 2., 1., 0., 0.)
    first = _scripted_state(1., 3., .95, 0., 0.)
    second = _scripted_state(2., 4., .91, 0., 0.)
    converged = _scripted_state(3., 3., 0., 0., 0.)
    solver = _ScriptedNLCGSolver(initial)
    solver.conv_tol_coarse = 2.
    weak_states = iter((first, second))
    line_calls = []
    canonical_calls = []

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.exact_gradient)

    def scripted_line_search(
            solver, unused_origin, unused_origin_gradient, direction,
            unused_evaluate, unused_objective, unused_initial_step,
            allow_restoration=False):
        line_calls.append(allow_restoration)
        state = next(weak_states)
        sample = gc_cg._LineSample(
            1., state, state.grand_potential, state.exact_gradient,
            solver._inner(state.exact_gradient, direction), 'scripted')
        return gc_cg._LineSearchResult(
            sample, False, 1, 'weak restoration',
            restoration=True, consistency=1.)

    def scripted_canonical(
            solver, origin, unused_mu, unused_objective,
            consistency, unused_history):
        canonical_calls.append((origin, consistency))
        if len(canonical_calls) == 1:
            return gc_cg._LineSearchResult(
                None, False, 1, 'charge bracket established',
                restoration=True, consistency=consistency)
        sample = gc_cg._LineSample(
            1., converged, converged.grand_potential,
            converged.exact_gradient, 0., 'canonical-restoration')
        return gc_cg._LineSearchResult(
            sample, False, 1, 'bounded canonical restoration',
            restoration=True, consistency=consistency)

    monkeypatch.setattr(gc_cg, '_line_search', scripted_line_search)
    monkeypatch.setattr(gc_cg, '_canonical_restoration', scripted_canonical)
    gc_cg.nlcg(solver, h=initial.h)
    assert solver.converged
    assert line_calls == [True, True]
    assert len(canonical_calls) == 2
    assert canonical_calls[0][0] is second
    assert canonical_calls[0][1] == pytest.approx(1.)


def test_nlcg_descent_check_rejects_nearly_orthogonal_direction():
    solver = SimpleNamespace()
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    gradient = [cp.asarray([1., 0.])]
    good = [cp.asarray([-1., 0.])]
    weak = [cp.asarray([-1e-10, 1.])]
    assert gc_cg._descent_metrics(solver, gradient, good)[0]
    assert not gc_cg._descent_metrics(solver, gradient, weak)[0]


@pytest.mark.parametrize('predicted_change,restoration', [
    (0.5*gc_cg.LINE_SEARCH_OBJECTIVE_FLAT, True),
    (gc_cg.LINE_SEARCH_OBJECTIVE_FLAT, False),
    (2.0*gc_cg.LINE_SEARCH_OBJECTIVE_FLAT, False),
])
def test_nlcg_flat_direction_gate_uses_fixed_objective_band(
        monkeypatch, predicted_change, restoration):
    initial = _scripted_state(0., 2., 1., 0., predicted_change)
    converged = _scripted_state(1., 1., 0., predicted_change, 0.)
    solver = _ScriptedNLCGSolver(initial)
    solver.max_cycle = 1
    calls = []

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.exact_gradient)

    def scripted_line_search(
            solver, unused_origin, unused_origin_gradient, direction,
            unused_evaluate, unused_objective, unused_initial_step,
            allow_restoration=False, **unused_options):
        calls.append((
            float(direction[0][0, 0].item()), allow_restoration))
        sample = gc_cg._LineSample(
            1., converged, converged.grand_potential,
            converged.exact_gradient, 0., 'scripted')
        return gc_cg._LineSearchResult(
            sample, True, 1, 'converged line sample',
            restoration=allow_restoration,
            consistency=gc_cg.LINE_SEARCH_OBJECTIVE_FLAT)

    monkeypatch.setattr(gc_cg, '_line_search', scripted_line_search)
    gc_cg.nlcg(solver, h=initial.h)

    assert solver.converged
    assert len(calls) == 1
    assert calls[0][1] is restoration
    assert (calls[0][0] > 0.) is restoration


def test_nlcg_flat_residual_uses_natural_step_not_transport(monkeypatch):
    predicted_change = 0.5*gc_cg.LINE_SEARCH_OBJECTIVE_FLAT
    initial = _scripted_state(0., 2., 1., 0., predicted_change)
    converged = _scripted_state(1., 1., 0., predicted_change, 0.)
    solver = _ScriptedNLCGSolver(initial)
    solver.max_cycle = 1
    solver.nlcg_initial_step = 1e6
    calls = []

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.exact_gradient)

    def scripted_line_search(
            solver, unused_origin, unused_origin_gradient, direction,
            unused_evaluate, unused_objective, initial_step,
            allow_restoration=False, **unused_options):
        calls.append((initial_step, allow_restoration))
        sample = gc_cg._LineSample(
            1., converged, converged.grand_potential,
            converged.exact_gradient, 0., 'scripted')
        return gc_cg._LineSearchResult(
            sample, True, 1, 'converged line sample',
            restoration=True,
            consistency=gc_cg.LINE_SEARCH_OBJECTIVE_FLAT)

    monkeypatch.setattr(gc_cg, '_line_search', scripted_line_search)
    gc_cg.nlcg(solver, h=initial.h)

    assert solver.converged
    assert len(calls) == 1
    assert calls[0][0] == pytest.approx(1.)
    assert calls[0][1] is True


def test_nlcg_uphill_pulay_direction_is_reflected_to_descent():
    solver = SimpleNamespace()
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    gradient = [cp.asarray([1., 0.])]
    uphill = [cp.asarray([2., 0.])]
    direction, metrics, reflected = gc_cg._orient_pulay_direction(
        solver, gradient, uphill)
    assert reflected
    assert metrics[0]
    assert metrics[1] < 0.
    assert cp.allclose(direction[0], -uphill[0])


def test_nlcg_pulay_direction_uses_accepted_fixed_point_history():
    class FakeDIIS:
        def __init__(self):
            self.errors = [cp.asarray([1.]), cp.asarray([-1.])]
            self.vectors = [cp.asarray([2.]), cp.asarray([6.])]
            self._bookkeep = [0, 1]
            self.pending_error = None

        def push_err_vec(self, error):
            self.pending_error = error

        def push_vec(self, vector):
            assert cp.allclose(vector, cp.asarray([3.]))
            assert cp.allclose(self.pending_error, cp.asarray([2.]))
            self.errors.append(self.pending_error)
            self.vectors.append(vector)
            self._bookkeep.append(2)

        def get_num_vec(self):
            return len(self._bookkeep)

        def get_err_vec(self, index):
            return self.errors[index]

        def get_vec(self, index):
            return self.vectors[index]

        def clear(self):
            pytest.fail('valid Pulay history should not be cleared')

    solver = SimpleNamespace(
        diis_pack=lambda blocks, weight_errors=False: cp.concatenate(
            [block.ravel() for block in blocks]),
        diis_unpack=lambda vector, unused_template: [vector.reshape(1, 1)],
        _sanitize_h=lambda blocks: blocks,
    )
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    state = SimpleNamespace(
        h=[cp.asarray([[1.]])],
        fock=[cp.asarray([[103.]])],
        residual=[cp.asarray([[2.]])],
    )
    direction, reason = gc_cg._pulay_direction(
        solver, state, FakeDIIS())
    assert direction is not None
    assert np.isfinite(float(direction[0][0, 0]))
    assert reason.startswith(
        'regularized Pulay history has %d vectors' %
        gc_cg.NLCG_PULAY_MIN_VECTORS)


def test_nlcg_preconditioned_pr_plus_and_fr_clipping():
    solver = SimpleNamespace()
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    old_exact = [cp.asarray([1., 0.])]
    old_residual = [cp.asarray([-1., 0.])]
    new_exact = [cp.asarray([0., 1.])]
    new_residual = [cp.asarray([1., -1.])]
    beta, reason = gc_cg._preconditioned_pr_plus(
        solver, old_exact, new_exact, old_residual, new_residual)
    assert beta == pytest.approx(1.)
    assert reason == 'preconditioned PR+'


def test_nlcg_preconditioned_pr_powell_restart():
    solver = SimpleNamespace()
    solver._inner = lambda left, right: sum(
        float(cp.vdot(x, y).real.item()) for x, y in zip(left, right))
    old_exact = [cp.asarray([1., 0.])]
    old_residual = [cp.asarray([-1., 0.])]
    new_exact = [cp.asarray([.5, 1.])]
    new_residual = [cp.asarray([-.5, -1.])]
    beta, reason = gc_cg._preconditioned_pr_plus(
        solver, old_exact, new_exact, old_residual, new_residual)
    assert beta == 0.
    assert reason == 'Powell restart'


def test_nlcg_resolved_occupation_step_tries_pr_before_plain_occupation(
        monkeypatch):
    initial = _scripted_state(0., 2., 1., 0., -1.)
    first = _scripted_state(1., 5., 1., -1., -1.)
    converged = _scripted_state(2., 2., 0., -2., 0.)
    solver = _ScriptedNLCGSolver(initial)
    solver.nelec = 1.
    solver.sigma = 5e-3
    solver.max_cycle = 2
    directions = []
    pr_descent_vectors = []

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.exact_gradient)
    monkeypatch.setattr(
        gc_cg, '_occupation_preconditioned_direction',
        lambda unused_solver, state, unused_scale: [
            cp.asarray([[2.+2.*float(state.h[0][0, 0].item())]])])
    monkeypatch.setattr(
        gc_cg, '_occupation_direction_is_proactive',
        lambda *unused_args: True)

    def scripted_pr(
            unused_solver, unused_old_exact, unused_new_exact,
            old_descent, new_descent):
        pr_descent_vectors.append((
            float(old_descent[0][0, 0].item()),
            float(new_descent[0][0, 0].item())))
        return .5, 'scripted occupation PR+'

    def scripted_line_search(
            solver, unused_origin, unused_origin_gradient, direction,
            unused_evaluate, unused_objective, unused_initial_step,
            **unused_kwargs):
        directions.append(float(direction[0][0, 0].item()))
        state = first if len(directions) == 1 else converged
        sample = gc_cg._LineSample(
            1., state, state.free_energy, state.exact_gradient,
            solver._inner(state.exact_gradient, direction), 'scripted')
        return gc_cg._LineSearchResult(
            sample, True, 1,
            ('converged line sample' if state is converged else
             'resolved line minimum'))

    monkeypatch.setattr(gc_cg, '_preconditioned_pr_plus', scripted_pr)
    monkeypatch.setattr(gc_cg, '_line_search', scripted_line_search)
    gc_cg.nlcg(solver, h=initial.h)

    assert solver.converged
    assert pr_descent_vectors == pytest.approx([(2., 4.)])
    assert directions == pytest.approx([2., 5.])


def test_nlcg_failed_occupation_pr_retries_plain_occupation(monkeypatch):
    initial = _scripted_state(0., 2., 1., 0., -1.)
    first = _scripted_state(1., 5., 1., -1., -1.)
    converged = _scripted_state(2., 2., 0., -2., 0.)
    solver = _ScriptedNLCGSolver(initial)
    solver.nelec = 1.
    solver.sigma = 5e-3
    solver.max_cycle = 2
    directions = []

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.exact_gradient)
    monkeypatch.setattr(
        gc_cg, '_occupation_preconditioned_direction',
        lambda unused_solver, state, unused_scale: [
            cp.asarray([[2.+2.*float(state.h[0][0, 0].item())]])])
    monkeypatch.setattr(
        gc_cg, '_occupation_direction_is_proactive',
        lambda *unused_args: True)
    monkeypatch.setattr(
        gc_cg, '_preconditioned_pr_plus',
        lambda *unused_args: (.5, 'scripted occupation PR+'))

    def scripted_line_search(
            solver, unused_origin, unused_origin_gradient, direction,
            unused_evaluate, unused_objective, unused_initial_step,
            **unused_kwargs):
        directions.append(float(direction[0][0, 0].item()))
        if len(directions) == 2:
            return gc_cg._LineSearchResult(
                None, False, 1, 'no lower objective sample')
        state = first if len(directions) == 1 else converged
        sample = gc_cg._LineSample(
            1., state, state.free_energy, state.exact_gradient,
            solver._inner(state.exact_gradient, direction), 'scripted')
        return gc_cg._LineSearchResult(
            sample, True, 1,
            ('converged line sample' if state is converged else
             'resolved line minimum'))

    monkeypatch.setattr(gc_cg, '_line_search', scripted_line_search)
    gc_cg.nlcg(solver, h=initial.h)

    assert solver.converged
    assert directions == pytest.approx([2., 5., 4.])


def test_nlcg_occupation_pr_benchmark_switch_preserves_plain_direction(
        monkeypatch):
    initial = _scripted_state(0., 2., 1., 0., -1.)
    first = _scripted_state(1., 5., 1., -1., -1.)
    converged = _scripted_state(2., 2., 0., -2., 0.)
    solver = _ScriptedNLCGSolver(initial)
    solver.nelec = 1.
    solver.sigma = 5e-3
    solver.max_cycle = 2
    directions = []

    monkeypatch.setattr(gc_cg, 'NLCG_OCCUPATION_PR_ENABLED', False)
    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.exact_gradient)
    monkeypatch.setattr(
        gc_cg, '_occupation_preconditioned_direction',
        lambda unused_solver, state, unused_scale: [
            cp.asarray([[2.+2.*float(state.h[0][0, 0].item())]])])
    monkeypatch.setattr(
        gc_cg, '_occupation_direction_is_proactive',
        lambda *unused_args: True)
    monkeypatch.setattr(
        gc_cg, '_preconditioned_pr_plus',
        lambda *unused_args: pytest.fail('occupation PR+ must be disabled'))

    def scripted_line_search(
            solver, unused_origin, unused_origin_gradient, direction,
            unused_evaluate, unused_objective, unused_initial_step,
            **unused_kwargs):
        directions.append(float(direction[0][0, 0].item()))
        state = first if len(directions) == 1 else converged
        sample = gc_cg._LineSample(
            1., state, state.free_energy, state.exact_gradient,
            solver._inner(state.exact_gradient, direction), 'scripted')
        return gc_cg._LineSearchResult(
            sample, True, 1,
            ('converged line sample' if state is converged else
             'resolved line minimum'))

    monkeypatch.setattr(gc_cg, '_line_search', scripted_line_search)
    gc_cg.nlcg(solver, h=initial.h)

    assert solver.converged
    assert directions == pytest.approx([2., 4.])


def test_nlcg_failed_conjugate_search_retries_residual_then_exact_gradient(
        monkeypatch):
    initial = _scripted_state(0., 2., 1., 0., -1.)
    first = _scripted_state(1., 5., 1., -1., -1.)
    converged = _scripted_state(2., 2., 0., -2., 0.)
    solver = _ScriptedNLCGSolver(initial)
    directions = []

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.exact_gradient)

    def scripted_line_search(
            solver, unused_origin, unused_origin_gradient, direction,
            unused_evaluate, unused_objective, unused_initial_step,
            allow_restoration=False):
        directions.append(float(direction[0][0, 0].item()))
        if len(directions) == 1:
            sample = gc_cg._LineSample(
                1., first, first.grand_potential, first.exact_gradient,
                solver._inner(first.exact_gradient, direction), 'scripted')
            return gc_cg._LineSearchResult(
                sample, True, 1, 'resolved line minimum')
        if len(directions) in (2, 3):
            return gc_cg._LineSearchResult(
                None, False, 1, 'no lower objective sample')
        sample = gc_cg._LineSample(
            1., converged, converged.grand_potential,
            converged.exact_gradient, 0., 'scripted')
        return gc_cg._LineSearchResult(
            sample, True, 1, 'converged line sample')

    monkeypatch.setattr(gc_cg, '_line_search', scripted_line_search)
    monkeypatch.setattr(
        gc_cg, '_preconditioned_pr_plus',
        lambda *unused_args: (1., 'scripted PR'))
    gc_cg.nlcg(solver, h=initial.h)
    assert solver.converged
    assert solver.cycles == 2
    assert directions == pytest.approx([1., 3., 2., 2.])


def test_nlcg_exact_retry_does_not_collapse_displacement_transport(
        monkeypatch):
    initial = _scripted_state(0., 2., 1., 0., -1.)
    first = _scripted_state(1., 5., 1., -1., -1.)
    second = _scripted_state(2., 6., .5, -2., -1.)
    converged = _scripted_state(3., 3., 0., -3., 0.)
    solver = _ScriptedNLCGSolver(initial)
    initial_steps = []

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.exact_gradient)

    def scripted_line_search(
            solver, unused_origin, unused_origin_gradient, direction,
            unused_evaluate, unused_objective, initial_step,
            allow_restoration=False):
        initial_steps.append(initial_step)
        call = len(initial_steps)
        if call == 1:
            state, alpha = first, 1.
        elif call in (2, 3):
            return gc_cg._LineSearchResult(
                None, False, 1, 'no lower objective sample')
        elif call == 4:
            state, alpha = second, 1e-6
        else:
            state, alpha = converged, 1.
        sample = gc_cg._LineSample(
            alpha, state, state.grand_potential, state.exact_gradient,
            solver._inner(state.exact_gradient, direction), 'scripted')
        return gc_cg._LineSearchResult(
            sample, True, 1,
            ('converged line sample' if state is converged
             else 'resolved line minimum'))

    monkeypatch.setattr(gc_cg, '_line_search', scripted_line_search)
    monkeypatch.setattr(
        gc_cg, '_preconditioned_pr_plus',
        lambda *unused_args: (1., 'scripted PR'))
    gc_cg.nlcg(solver, h=initial.h)
    assert solver.converged
    assert initial_steps == pytest.approx([1., 1./3., .5, .5, .5])


def test_nlcg_inexact_line_minimum_restarts_conjugacy(monkeypatch):
    initial = _scripted_state(0., 2., 1., 0., -1.)
    first = _scripted_state(1., 5., 1., -1., -1.)
    converged = _scripted_state(2., 2., 0., -2., 0.)
    solver = _ScriptedNLCGSolver(initial)
    directions = []

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.exact_gradient)

    def scripted_line_search(
            solver, unused_origin, unused_origin_gradient, direction,
            unused_evaluate, unused_objective, unused_initial_step,
            allow_restoration=False):
        directions.append(float(direction[0][0, 0].item()))
        state = first if len(directions) == 1 else converged
        sample = gc_cg._LineSample(
            1., state, state.grand_potential, state.exact_gradient,
            solver._inner(state.exact_gradient, direction), 'scripted')
        return gc_cg._LineSearchResult(
            sample, len(directions) > 1, 1,
            ('line-search evaluation limit' if len(directions) == 1
             else 'converged line sample'))

    monkeypatch.setattr(gc_cg, '_line_search', scripted_line_search)
    gc_cg.nlcg(solver, h=initial.h)
    assert solver.converged
    assert directions == pytest.approx([1., 2.])


def test_nlcg_significant_inexact_descent_skips_residual_retries(
        monkeypatch):
    initial = _scripted_state(0., 2., 1., 0., -1.)
    worse = _scripted_state(1., 5., 2., -1., -1.)
    converged = _scripted_state(2., 2., 0., -2., 0.)
    solver = _ScriptedNLCGSolver(initial)
    solver.max_cycle = 1
    directions = []

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.exact_gradient)

    def scripted_line_search(
            solver, unused_origin, unused_origin_gradient, direction,
            unused_evaluate, unused_objective, unused_initial_step,
            allow_restoration=False):
        directions.append(float(direction[0][0, 0].item()))
        state = worse if len(directions) == 1 else converged
        sample = gc_cg._LineSample(
            1., state, state.grand_potential, state.exact_gradient,
            solver._inner(state.exact_gradient, direction), 'scripted')
        return gc_cg._LineSearchResult(
            sample, len(directions) > 1, 1,
            ('unresolved objective descent' if len(directions) == 1
             else 'converged line sample'))

    monkeypatch.setattr(gc_cg, '_line_search', scripted_line_search)
    monkeypatch.setattr(
        gc_cg, '_preconditioned_pr_plus',
        lambda *unused_args: pytest.fail('PR+ should be restarted'))
    gc_cg.nlcg(solver, h=initial.h)
    assert not solver.converged
    assert solver.cycles == 1
    assert solver.grand_potential == pytest.approx(-1.)
    assert solver.residual_rms == pytest.approx(2.)
    assert directions == pytest.approx([1.])


def test_nlcg_deferred_growth_ranks_significant_objective_gain_first(
        monkeypatch):
    initial = _scripted_state(0., 2., 1., 0., -1.)
    objective_best = _scripted_state(1., 5., 2., -2., -1.)
    residual_best = _scripted_state(2., 5.2, 1.6, -1., -1.)
    solver = _ScriptedNLCGSolver(initial)
    solver.max_cycle = 1
    calls = []

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.exact_gradient)

    def scripted_line_search(
            solver, unused_origin, unused_origin_gradient, direction,
            unused_evaluate, unused_objective, unused_initial_step,
            **unused_kwargs):
        calls.append(float(direction[0][0, 0].item()))
        state = objective_best if len(calls) == 1 else residual_best
        sample = gc_cg._LineSample(
            1., state, state.grand_potential, state.exact_gradient,
            solver._inner(state.exact_gradient, direction), 'scripted')
        return gc_cg._LineSearchResult(
            sample, len(calls) > 1, 1,
            ('unresolved objective descent' if len(calls) == 1
             else 'resolved line minimum'))

    monkeypatch.setattr(gc_cg, '_line_search', scripted_line_search)
    gc_cg.nlcg(solver, h=initial.h)
    assert solver.grand_potential == pytest.approx(-2.)
    assert solver.residual_rms == pytest.approx(2.)
    assert calls == pytest.approx([1.])


def test_nlcg_deferred_gain_beats_later_safe_higher_objective(monkeypatch):
    initial = _scripted_state(0., 2., 1., 0., -1.)
    objective_best = _scripted_state(1., 5., 2., -2., -1.)
    safe_higher = _scripted_state(2., 4.8, 1.4, -1., -1.)
    solver = _ScriptedNLCGSolver(initial)
    solver.max_cycle = 1
    calls = []

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.exact_gradient)

    def scripted_line_search(
            solver, unused_origin, unused_origin_gradient, direction,
            unused_evaluate, unused_objective, unused_initial_step,
            **unused_kwargs):
        calls.append(float(direction[0][0, 0].item()))
        state = objective_best if len(calls) == 1 else safe_higher
        sample = gc_cg._LineSample(
            1., state, state.grand_potential, state.exact_gradient,
            solver._inner(state.exact_gradient, direction), 'scripted')
        return gc_cg._LineSearchResult(
            sample, len(calls) > 1, 1,
            ('unresolved objective descent' if len(calls) == 1
             else 'resolved line minimum'))

    monkeypatch.setattr(gc_cg, '_line_search', scripted_line_search)
    gc_cg.nlcg(solver, h=initial.h)
    assert solver.grand_potential == pytest.approx(-2.)
    assert solver.residual_rms == pytest.approx(2.)
    assert calls == pytest.approx([1.])


def test_nlcg_resolved_significant_descent_skips_residual_retries(
        monkeypatch):
    initial = _scripted_state(0., 2., 1., 0., -1.)
    objective_step = _scripted_state(1., 5., 2., -2., -1.)
    solver = _ScriptedNLCGSolver(initial)
    solver.max_cycle = 1
    calls = []

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.exact_gradient)

    def scripted_line_search(
            solver, unused_origin, unused_origin_gradient, direction,
            unused_evaluate, unused_objective, unused_initial_step,
            **unused_kwargs):
        calls.append(float(direction[0][0, 0].item()))
        sample = gc_cg._LineSample(
            1., objective_step, objective_step.grand_potential,
            objective_step.exact_gradient,
            solver._inner(objective_step.exact_gradient, direction),
            'scripted')
        return gc_cg._LineSearchResult(
            sample, True, 1, 'resolved line minimum')

    monkeypatch.setattr(gc_cg, '_line_search', scripted_line_search)
    gc_cg.nlcg(solver, h=initial.h)
    assert solver.grand_potential == pytest.approx(-2.)
    assert solver.residual_rms == pytest.approx(2.)
    assert calls == pytest.approx([1.])


def test_nlcg_pulay_retry_transports_active_objective_step(monkeypatch):
    initial = _scripted_state(0., 2., 1., 0., -1.)
    converged = _scripted_state(4., 4., 0., -1., 0.)
    solver = _ScriptedNLCGSolver(initial)
    solver.nelec = 1.
    initial_steps = []
    directions = []

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.exact_gradient)
    monkeypatch.setattr(
        gc_cg, '_pulay_direction',
        lambda *unused_args: ([cp.asarray([[4.]])], 'scripted Pulay'))

    def scripted_line_search(
            solver, unused_origin, unused_origin_gradient, direction,
            unused_evaluate, unused_objective, initial_step, **unused_kwargs):
        initial_steps.append(initial_step)
        directions.append(float(direction[0][0, 0].item()))
        if len(directions) == 1:
            return gc_cg._LineSearchResult(
                None, False, 1, 'no lower objective sample')
        sample = gc_cg._LineSample(
            1., converged, converged.free_energy,
            converged.exact_gradient, 0., 'scripted')
        return gc_cg._LineSearchResult(
            sample, True, 1, 'converged line sample')

    monkeypatch.setattr(gc_cg, '_line_search', scripted_line_search)
    gc_cg.nlcg(solver, h=initial.h)
    assert solver.converged
    assert directions == pytest.approx([1., 4.])
    assert initial_steps == pytest.approx([1., .25])


def test_nlcg_probes_transported_pulay_before_flat_objective(monkeypatch):
    initial = _scripted_state(0., 2., 1., 0., -1.)
    converged = _scripted_state(1., 1., 0., -1., 0.)
    solver = _ScriptedNLCGSolver(initial)
    solver.nelec = 1.
    solver.max_cycle = 1
    calls = []

    monkeypatch.setattr(
        gc_cg, 'objective_gradient',
        lambda unused_solver, state, unused_fixed_n: state.exact_gradient)
    monkeypatch.setattr(
        gc_cg, '_pulay_direction',
        lambda *unused_args: ([cp.asarray([[4.]])], 'scripted Pulay'))
    monkeypatch.setattr(gc_cg, 'NLCG_PULAY_STAGNATION_STEPS', 0)

    def scripted_line_search(
            unused_solver, unused_origin, unused_origin_gradient, direction,
            unused_evaluate, unused_objective, initial_step,
            **unused_kwargs):
        calls.append((float(direction[0][0, 0].item()), initial_step))
        sample = gc_cg._LineSample(
            initial_step, converged, converged.free_energy,
            converged.exact_gradient, 0., 'scripted')
        return gc_cg._LineSearchResult(
            sample, True, 1, 'converged line sample')

    monkeypatch.setattr(gc_cg, '_line_search', scripted_line_search)
    gc_cg.nlcg(solver, h=initial.h)

    assert solver.converged
    assert calls == pytest.approx([(4., .25)])


@pytest.mark.parametrize('direction', [
    cp.asarray([[.15, .04], [.04, -.07]], dtype=cp.complex128),
    cp.asarray([[0., .06j], [-.06j, 0.]], dtype=cp.complex128),
])
def test_fixed_mu_nlcg_gradient_matches_complex_finite_difference(direction):
    unused, solver = _solver(mu=-.1)
    solver.build()
    h = [cp.asarray(
        [[-.3, .08+.03j], [.08-.03j, .2]], dtype=cp.complex128)]
    state = solver.calculate_cycle(h, mu=solver.mu)
    gradient = gc_cg.objective_gradient(solver, state, fixed_n=False)
    epsilon = 1e-5
    plus = solver.calculate_cycle(
        [state.h[0]+epsilon*direction], mu=solver.mu)
    minus = solver.calculate_cycle(
        [state.h[0]-epsilon*direction], mu=solver.mu)
    finite_difference = (
        plus.grand_potential-minus.grand_potential) / (2.*epsilon)
    analytic = solver._inner(gradient, [direction])
    assert abs(finite_difference-analytic) < 2e-6


def test_fixed_n_nlcg_gradients_match_multik_finite_difference():
    target = 1.3
    fock = [
        cp.asarray([[-.7, .12j], [-.12j, .3]], dtype=cp.complex128),
        cp.asarray([[-.55, .03+.04j], [.03-.04j, .42]],
                   dtype=cp.complex128),
    ]
    solver = GrandCanonicalKRKS(
        _FixedFockKRKS(fock), sigma=.15, nelec=target)
    solver.build()
    h = [
        cp.asarray([[-.3, .08+.03j], [.08-.03j, .2]],
                   dtype=cp.complex128),
        cp.asarray([[-.2, -.05+.02j], [-.05-.02j, .35]],
                   dtype=cp.complex128),
    ]
    direction = [
        cp.asarray([[.15, .04j], [-.04j, -.07]],
                   dtype=cp.complex128),
        cp.asarray([[-.11, .02-.01j], [.02+.01j, .08]],
                   dtype=cp.complex128),
    ]
    state = solver.calculate_cycle(h, nelec=target)
    mu_gradient = gc_cg.mu_gradient_wrt_h(
        state.coeff, state.occ, solver.weight)
    free_energy_gradient = gc_cg.objective_gradient(
        solver, state, fixed_n=True)
    epsilon = 1e-5
    plus_h = [x+epsilon*d for x, d in zip(state.h, direction)]
    minus_h = [x-epsilon*d for x, d in zip(state.h, direction)]

    mu_plus = solver._solve_mu(
        [cp.linalg.eigvalsh(x) for x in plus_h], target)
    mu_minus = solver._solve_mu(
        [cp.linalg.eigvalsh(x) for x in minus_h], target)
    finite_difference_mu = (mu_plus-mu_minus) / (2.*epsilon)
    assert abs(
        finite_difference_mu-solver._inner(mu_gradient, direction)) < 2e-6
    assert abs(solver._inner(mu_gradient, solver.identity)-1.) < 1e-12

    plus = solver.calculate_cycle(plus_h, nelec=target)
    minus = solver.calculate_cycle(minus_h, nelec=target)
    finite_difference_free_energy = (
        plus.free_energy-minus.free_energy) / (2.*epsilon)
    assert abs(finite_difference_free_energy-solver._inner(
        free_energy_gradient, direction)) < 2e-6
    assert abs(solver._inner(
        free_energy_gradient, solver.identity)) < 1e-12
    assert abs(state.nelec-target) < 1e-10
    assert abs(plus.nelec-target) < 1e-10
    assert abs(minus.nelec-target) < 1e-10


def test_fixed_n_mu_gradient_rejects_singular_fermi_response():
    with pytest.raises(RuntimeError, match='numerically singular'):
        gc_cg.mu_gradient_wrt_h(
            [cp.eye(2)], [cp.asarray([0., 1.])], weight=1.)


def test_constructor_and_stream_object_configuration():
    mf = _FixedFockKRKS([_fock()])
    with pytest.raises(TypeError, match='sigma'):
        GrandCanonicalKRKS(mf, mu=-.1)
    with pytest.raises(TypeError, match='mu'):
        GrandCanonicalKRKS(mf, sigma=.1)
    with pytest.raises(ValueError, match='positive'):
        GrandCanonicalKRKS(mf, mu=-.1, sigma=0.)
    with pytest.raises(TypeError, match='different ensembles'):
        GrandCanonicalKRKS(mf, mu=-.1, sigma=.1, nelec=1.2)
    solver = GrandCanonicalKRKS(mf, mu=-.1, sigma=.1)
    assert solver.nlcg_line_search_alpha_rtol is None
    assert solver.nlcg_line_search_slope_atol is None
    solver.set(
        conv_tol=1e-7, diis_space=4, tighten_mu_threshold=2e-3,
        nlcg_initial_step=.75, nlcg_max_line_search_evaluations=5,
        nlcg_line_search_alpha_rtol=1e-3,
        nlcg_line_search_slope_atol=2e-6)
    assert solver.conv_tol == 1e-7
    assert solver.diis_space == 4
    assert solver.tighten_mu_threshold == 2e-3
    assert solver.nlcg_initial_step == .75
    assert solver.nlcg_max_line_search_evaluations == 5
    assert solver.nlcg_line_search_alpha_rtol == 1e-3
    assert solver.nlcg_line_search_slope_atol == 2e-6

    solver.nlcg_initial_step = 0.
    with pytest.raises(ValueError, match='nlcg_initial_step'):
        solver.check_sanity()
    solver.nlcg_initial_step = 1.
    solver.nlcg_max_line_search_evaluations = 1
    with pytest.raises(ValueError, match='nlcg_max_line_search_evaluations'):
        solver.check_sanity()
    solver.nlcg_max_line_search_evaluations = 5
    solver.nlcg_line_search_slope_atol = None
    with pytest.raises(ValueError, match='must both be set'):
        solver.check_sanity()
    solver.nlcg_line_search_slope_atol = 2e-6
    solver.nlcg_line_search_alpha_rtol = 0.
    with pytest.raises(ValueError, match='nlcg_line_search_alpha_rtol'):
        solver.check_sanity()
    solver.nlcg_line_search_alpha_rtol = 1e-3
    solver.nlcg_line_search_slope_atol = np.inf
    with pytest.raises(ValueError, match='nlcg_line_search_slope_atol'):
        solver.check_sanity()


def test_build_caches_mean_field_setup():
    mf = _CountingSetupKRKS([_fock()])
    solver = GrandCanonicalKRKS(mf, mu=-.1, sigma=.1)
    solver.build()
    solver.build()
    assert mf.setup_calls == dict(
        build=1, overlap=1, hcore=1, orth=1, enuc=1)


def test_fixed_n_mu_uses_pyscf_smearing_convention(monkeypatch):
    fock = [cp.diag(cp.asarray([-.7, .3])) for unused in range(3)]
    solver = GrandCanonicalKRKS(
        _FixedFockKRKS(fock), sigma=.15, nelec=1.3)
    solver.build()
    energies = [cp.asarray([-.8, .1]), cp.asarray([-.6, .2]),
                cp.asarray([-.4, .5])]
    original = gc._smearing_optimize
    seen = {}

    def recording_optimizer(f_occ, mo_energy, nocc, sigma):
        seen.update(f_occ=f_occ, mo_energy=mo_energy.copy(),
                    nocc=nocc, sigma=sigma)
        return original(f_occ, mo_energy, nocc, sigma)

    monkeypatch.setattr(gc, '_smearing_optimize', recording_optimizer)
    mu = solver._solve_mu(energies, 1.3)
    assert seen['f_occ'] is gc._fermi_smearing_occ
    assert np.array_equal(
        seen['mo_energy'], np.concatenate([cp.asnumpy(x) for x in energies]))
    assert seen['nocc'] == 1.3 * 3 / 2
    assert seen['sigma'] == .15
    assert abs(solver.nelec_from_eig(energies, mu)-1.3) < 1e-10


def test_fixed_n_state_has_target_charge_and_gauge_free_residual():
    target = 1.3
    unused, solver = _solver(nelec=target)
    solver.build()
    h = [cp.asarray([[-.3, .08+.03j], [.08-.03j, .2]])]
    state = solver.calculate_cycle(h, nelec=target)
    assert abs(state.nelec-target) < 1e-10
    assert abs(solver._trace_mean(state.residual)) < 1e-12
    assert abs(state.free_energy-(state.e_tot+state.entropy_energy)) < 1e-13
    assert np.isfinite(state.mu)


def test_fixed_n_state_removes_scalar_gauge_without_another_fock():
    target = 1.3
    unused, solver = _solver(nelec=target)
    solver.build()
    h = [_fock() + .37*cp.eye(2)]
    state = solver.calculate_cycle(h, nelec=target)
    assert solver.nfev == 1
    assert solver._rms(
        [x-y for x, y in zip(state.h, state.fock)]) < 1e-12
    assert abs(solver.nelec_from_eig(state.eig, state.mu)-target) < 1e-10


def test_tagged_solvent_fock_and_energy_use_same_veff():
    hcore = [cp.asarray([[-.6, 0.], [0., .2]])]
    solvent = [cp.asarray([[.1, .03j], [-.03j, -.04]])]
    mf = _TaggedSolventKRKS(hcore, solvent)
    solver = GrandCanonicalKRKS(mf, sigma=.15, nelec=1.2)
    solver.build()
    state = solver.calculate_cycle([cp.asarray([[-.2, .1j], [-.1j, .1]])],
                                   nelec=1.2)
    expected = hcore[0]+solvent[0]
    assert float(cp.max(cp.abs(state.fock[0]-expected)).item()) < 1e-13
    assert mf.energy_veff is not None
    assert getattr(mf.energy_veff, 'v_solvent', None) is not None


def test_fixed_mu_nlcg_converges_complex_fixed_fock_problem():
    mf, solver = _solver(mu=-.1)
    solver.conv_tol = 1e-8
    solver.max_cycle = 20
    solver.build()
    displacement = cp.asarray(
        [[.075, .02+.01j], [.02-.01j, -.035]],
        dtype=cp.complex128)
    h0 = [_fock()+displacement]
    initial = solver.calculate_cycle(h0, mu=solver.mu)
    e_tot = solver.nlcg(h=h0)
    assert solver.converged, solver.message
    assert solver.residual_rms <= solver.conv_tol
    assert solver.grand_potential < initial.grand_potential
    assert e_tot == solver.e_tot == mf.e_tot
    assert solver.mu == pytest.approx(-.1)
    assert solver.mo_coeff is mf.mo_coeff


def test_fixed_n_nlcg_converges_complex_fixed_fock_problem():
    target = 1.25
    mf, solver = _solver(nelec=target)
    solver.conv_tol = 1e-8
    solver.max_cycle = 20
    solver.build()
    h0 = [cp.asarray(
        [[-.1, .19-.11j], [.19+.11j, .6]], dtype=cp.complex128)]
    initial = solver.calculate_cycle(h0, nelec=target)
    e_tot = solver.nlcg(h=h0)
    assert solver.converged, solver.message
    assert solver.residual_rms <= solver.conv_tol
    assert solver.free_energy < initial.free_energy
    assert abs(solver.electron_number-target) < 1e-10
    assert np.isfinite(solver.mu)
    assert e_tot == solver.e_tot == mf.e_tot
    reconstructed = (
        (solver.mo_coeff * solver.mo_occ[:, None, :])
        @ solver.mo_coeff.conj().transpose(0, 2, 1))
    assert float(cp.max(cp.abs(
        reconstructed-solver._cycle_data.dm)).item()) < 1e-10


@pytest.mark.parametrize('solver_kwargs,coupling,objective_name', [
    ({'nelec': 1.3}, .3, 'free_energy'),
    ({'mu': -.1}, .15, 'grand_potential'),
])
def test_nlcg_interpolation_converges_nonlinear_problem(
        monkeypatch, solver_kwargs, coupling, objective_name):
    hcore = [cp.asarray(
        [[-.7, .08j], [-.08j, .3]], dtype=cp.complex128)]
    h0 = [cp.asarray(
        [[-.2, .18-.07j], [.18+.07j, .5]], dtype=cp.complex128)]
    mf = _LinearFockKRKS(hcore, coupling=coupling)
    solver = GrandCanonicalKRKS(mf, sigma=.15, **solver_kwargs)
    solver.conv_tol = 1e-8
    solver.max_cycle = 30
    solver.build()
    if 'nelec' in solver_kwargs:
        initial = solver.calculate_cycle(h0, nelec=solver.nelec)
    else:
        initial = solver.calculate_cycle(h0, mu=solver.mu)

    original_line_search = gc_cg._line_search
    accepted_values = []

    def recording_line_search(*args, **kwargs):
        result = original_line_search(*args, **kwargs)
        if result.sample is not None:
            accepted_values.append(result.sample.value)
        return result

    monkeypatch.setattr(gc_cg, '_line_search', recording_line_search)
    solver.nlcg(h=h0)

    values = [getattr(initial, objective_name)] + accepted_values
    assert solver.converged, solver.message
    assert solver.residual_rms <= solver.conv_tol
    assert solver.nfev < 61
    assert all(new <= old+1e-12 for old, new in zip(values, values[1:]))


@pytest.mark.parametrize('solver_kwargs,coupling,max_nfev', [
    ({'mu': -.1}, .15, 50),
    ({'nelec': 1.3}, .3, 60),
])
def test_nlcg_converges_sharp_sigma_nonlinear_problem(
        monkeypatch, solver_kwargs, coupling, max_nfev):
    hcore = [cp.asarray(
        [[-.7, .08j], [-.08j, .3]], dtype=cp.complex128)]
    h0 = [cp.asarray(
        [[-.2, .18-.07j], [.18+.07j, .5]], dtype=cp.complex128)]
    mf = _LinearFockKRKS(hcore, coupling=coupling)
    solver = GrandCanonicalKRKS(mf, sigma=1e-4, **solver_kwargs)
    solver.conv_tol = 1e-8
    solver.max_cycle = 30
    solver.build()
    accepted = []
    original_line_search = gc_cg._line_search
    original_canonical_restoration = gc_cg._canonical_restoration

    def recording_line_search(
            solver, origin, origin_gradient, direction, evaluate,
            objective, initial_step, allow_restoration=False, **kwargs):
        result = original_line_search(
            solver, origin, origin_gradient, direction, evaluate,
            objective, initial_step,
            allow_restoration=allow_restoration, **kwargs)
        if result.sample is not None:
            accepted.append((
                result.sample.value, objective(origin), result.consistency))
        return result

    def recording_canonical_restoration(
            solver, origin, target_mu, objective, consistency,
            charge_history):
        result = original_canonical_restoration(
            solver, origin, target_mu, objective, consistency,
            charge_history)
        if result.sample is not None:
            accepted.append((
                result.sample.value, objective(origin), result.consistency))
        return result

    monkeypatch.setattr(gc_cg, '_line_search', recording_line_search)
    monkeypatch.setattr(
        gc_cg, '_canonical_restoration',
        recording_canonical_restoration)
    solver.nlcg(h=h0)

    assert solver.converged, solver.message
    assert solver.cycles <= 30
    assert solver.nfev <= max_nfev
    assert solver.residual_rms <= solver.conv_tol
    assert all(value <= origin+band for value, origin, band in accepted)


def test_residual_diis_converges_complex_fixed_n_problem():
    hcore = [cp.asarray([[-.7, .08j], [-.08j, .3]], dtype=cp.complex128)]
    mf = _LinearFockKRKS(hcore, coupling=.15)
    solver = GrandCanonicalKRKS(mf, sigma=.15, nelec=1.3)
    solver.conv_tol = 1e-8
    solver.build()
    h0 = [cp.asarray([[-.2, .18-.07j], [.18+.07j, .5]])]
    fixed_n_calc = solver.start_fixed_n_calc(h0, 1.3)
    solver.fixed_n_subproblem(fixed_n_calc, solver.conv_tol)
    assert fixed_n_calc.converged, fixed_n_calc.message
    assert fixed_n_calc.cycle_data.residual_rms <= solver.conv_tol
    assert abs(fixed_n_calc.cycle_data.nelec-1.3) < 1e-10
    mismatch = [h-f for h, f in zip(
        fixed_n_calc.cycle_data.h, fixed_n_calc.cycle_data.fock)]
    assert solver._rms(mismatch) <= solver.conv_tol
    assert fixed_n_calc.cycles > 0


def test_same_n_refinement_preserves_diis_session():
    hcore = [cp.asarray([[-.7, .08j], [-.08j, .3]], dtype=cp.complex128)]
    solver = GrandCanonicalKRKS(
        _LinearFockKRKS(hcore, coupling=.15), sigma=.15, nelec=1.3)
    solver.build()
    h0 = [cp.asarray([[-.2, .18-.07j], [.18+.07j, .5]])]
    fixed_n_calc = solver.start_fixed_n_calc(h0, 1.3)
    adiis = fixed_n_calc.diis
    solver.fixed_n_subproblem(fixed_n_calc, 1e-3)
    coarse_cycles = fixed_n_calc.cycles
    coarse_nfev = solver.nfev
    assert fixed_n_calc.converged
    solver.fixed_n_subproblem(fixed_n_calc, 1e-8)
    assert fixed_n_calc.diis is adiis
    assert fixed_n_calc.converged
    assert fixed_n_calc.cycles >= coarse_cycles
    assert solver.nfev >= coarse_nfev
    assert fixed_n_calc.cycle_data.residual_rms <= 1e-8


def test_secant_proposals_and_neutral_charge_cap():
    unused, solver = _solver(mu=-.1)
    solver.build()
    state = solver.calculate_cycle([_fock()], nelec=1.3)
    samples = [
        gc.MuSample(SimpleNamespace(nelec=1.), -.1, None),
        gc.MuSample(SimpleNamespace(nelec=2.), .1, None),
    ]
    assert solver.secant_proposal(samples, state.h) == pytest.approx(1.8)

    proposal = solver.secant_proposal(samples[:1], state.h)
    assert abs(proposal-1.) <= solver.initial_nelec_step+1e-14


def test_public_fixed_n_kernel_publishes_standard_attributes():
    mf, solver = _solver(nelec=1.25)
    e_tot = solver.kernel()
    assert solver.converged, solver.message
    assert e_tot == solver.e_tot == mf.e_tot
    assert abs(solver.electron_number-1.25) < 1e-10
    assert solver.nfev == 2
    assert mf.mo_coeff is solver.mo_coeff


def test_public_fixed_mu_kernel_uses_fixed_n_root():
    mf, solver = _solver(mu=-.16)
    solver.conv_tol = 1e-9
    e_tot = solver.kernel()
    assert solver.converged, solver.message
    assert e_tot == mf.e_tot
    assert abs(solver.mu+.16) < solver.conv_tol_mu
    assert solver.outer_cycles == 4
    assert solver.nfev == 1 + solver.outer_cycles


def test_fock_evaluation_count_includes_initial_density_build():
    mf, solver = _solver(nelec=1.2)
    solver.kernel()
    assert solver.nfev == mf.veff_calls
    assert solver.nfev == 2


def test_fixed_mu_starts_from_initial_density_electron_number():
    mf, solver = _solver(mu=-.16)
    solver.build()
    dm0 = cp.asarray([[[.7, .1j], [-.1j, .2]]])
    unused_h, nelec = solver._initial_h(dm0)
    assert nelec == pytest.approx(.9)
    assert nelec != pytest.approx(solver.nelec_at_mu(unused_h, solver.mu))


def test_fixed_mu_clips_an_overcapacity_initial_density():
    mf = _FixedFockKRKS([cp.asarray([[-.7]])])
    mf.get_init_guess = lambda cell, kpts=None: cp.asarray([[[2.1]]])
    solver = GrandCanonicalKRKS(mf, mu=-.16, sigma=.15)
    solver.max_outer_cycle = 10
    solver.kernel()
    assert solver.converged, solver.message
    assert abs(solver.mu+.16) < solver.conv_tol_mu


def _small_periodic_cell():
    cell = gto.Cell()
    cell.a = [[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]]
    cell.atom = 'He 0 0 0'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [15, 15, 15]
    cell.verbose = 0
    return cell.build()


def test_real_multik_fixed_n_krks():
    cell = _small_periodic_cell()
    mf = cell.KRKS(kpts=cell.make_kpts([3, 1, 1])).to_gpu()
    mf.xc = 'LDA,VWN'
    solver = GrandCanonicalKRKS(mf, sigma=.08, nelec=1.6)
    solver.conv_tol = 1e-6
    solver.max_cycle = 30
    solver.kernel()
    assert solver.converged, solver.message
    assert abs(solver.electron_number-1.6) < 1e-9
    assert np.isfinite(solver.mu)
    dm = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    assert float(cp.max(cp.abs(dm-solver._cycle_data.dm)).item()) < 1e-8


def test_real_multik_fixed_n_nlcg():
    cell = _small_periodic_cell()
    mf = cell.KRKS(kpts=cell.make_kpts([3, 1, 1])).to_gpu()
    mf.xc = 'LDA,VWN'
    solver = GrandCanonicalKRKS(mf, sigma=.08, nelec=1.6)
    solver.conv_tol = 1e-6
    solver.max_cycle = 30
    solver.nlcg()
    assert solver.converged, solver.message
    assert solver.residual_rms <= solver.conv_tol
    assert abs(solver.electron_number-1.6) < 1e-9
    assert np.isfinite(solver.mu)
    dm = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    assert float(cp.max(cp.abs(dm-solver._cycle_data.dm)).item()) < 1e-8


def test_real_multik_fixed_mu_krks():
    cell = _small_periodic_cell()
    mf = cell.KRKS(kpts=cell.make_kpts([3, 1, 1])).to_gpu()
    mf.xc = 'LDA,VWN'
    solver = GrandCanonicalKRKS(mf, mu=-.4, sigma=.08)
    solver.conv_tol = 1e-6
    solver.conv_tol_coarse = 1e-5
    solver.conv_tol_mu = 1e-5
    solver.max_cycle = 30
    solver.kernel()
    assert solver.converged, solver.message
    assert abs(solver.mu+.4) < solver.conv_tol_mu
    assert solver.residual_rms <= 1e-6
