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


@pytest.mark.parametrize('count', [3, 4, 5])
def test_nlcg_polynomial_step_finds_convex_minimum(count):
    alphas = np.linspace(0., 3., count)
    values = (alphas-1.25)**2 + 7.
    alpha, curvature = gc_cg._polynomial_step(
        alphas, values, (0., 3.))
    assert alpha == pytest.approx(1.25, abs=1e-8)
    assert curvature > 0.


def test_nlcg_polynomial_step_rejects_concave_stationary_point():
    alphas = np.asarray([0., 1., 2.])
    values = -(alphas-1.)**2
    assert gc_cg._polynomial_step(
        alphas, values, (0., 2.)) == (None, None)


def test_nlcg_spline_step_finds_convex_minimum():
    alphas = np.arange(6, dtype=float)
    values = (alphas-2.)**2
    alpha, curvature = gc_cg._spline_step(
        alphas, values, (1., 3.))
    assert alpha == pytest.approx(2.)
    assert curvature > 0.


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
            unused_evaluate, unused_objective, unused_initial_step):
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
    gc_cg.nlcg(solver, h=initial.h)
    assert solver.converged
    assert solver.cycles == 2
    assert directions == pytest.approx([1., 4., 2., 1.])


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
            unused_evaluate, unused_objective, unused_initial_step):
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
    solver = GrandCanonicalKRKS(mf, mu=-.1, sigma=.1).set(
        conv_tol=1e-7, diis_space=4, tighten_mu_threshold=2e-3,
        nlcg_initial_step=.75, nlcg_max_line_search_evaluations=5)
    assert solver.conv_tol == 1e-7
    assert solver.diis_space == 4
    assert solver.tighten_mu_threshold == 2e-3
    assert solver.nlcg_initial_step == .75
    assert solver.nlcg_max_line_search_evaluations == 5

    solver.nlcg_initial_step = 0.
    with pytest.raises(ValueError, match='nlcg_initial_step'):
        solver.check_sanity()
    solver.nlcg_initial_step = 1.
    solver.nlcg_max_line_search_evaluations = 1
    with pytest.raises(ValueError, match='nlcg_max_line_search_evaluations'):
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
