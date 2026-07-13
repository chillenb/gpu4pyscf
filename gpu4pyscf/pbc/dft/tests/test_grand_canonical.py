import cupy as cp
import numpy as np
import pytest
from dataclasses import FrozenInstanceError, replace
from pyscf.pbc import gto

from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.dft.grand_canonical import (
    GrandCanonicalConfig, GrandCanonicalKRKS, _CanonicalSample,
    _CanonicalWork, _DIISItem, _FixedNSession, _LineSearchResult,
    fermi_divided_difference, fermi_entropy, fermi_occupations,
)


class _MockCell:
    precision = 1.0e-12
    nelectron = 2

    @staticmethod
    def get_scaled_kpts(kpts):
        return kpts


class _FixedFockKRKS:
    """Small GPU evaluator used to test the optimiser independently of grids."""

    def __init__(self, fock):
        self.cell = _MockCell()
        self.kpts = np.zeros((len(fock), 3))
        self._fock = cp.stack(fock)
        self.scf_summary = {}
        self.verbose = 0
        self.veff_seen = None
        self.energy_veff_seen = None
        self.veff_calls = 0

    def get_ovlp(self, cell, kpts):
        return cp.stack([cp.eye(f.shape[0], dtype=f.dtype) for f in self._fock])

    def get_hcore(self, cell, kpts):
        return self._fock

    def check_linear_dependency(self, overlap, **kwargs):
        return cp.stack([cp.eye(s.shape[0], dtype=s.dtype) for s in overlap])

    def get_init_guess(self, cell, kpts=None):
        return cp.stack([cp.eye(f.shape[0], dtype=f.dtype) for f in self._fock])

    def get_veff(self, cell, dm, **kwargs):
        self.veff_calls += 1
        self.last_dm = dm
        self.veff_seen = cp.zeros_like(dm)
        return self.veff_seen

    def energy_elec(self, dm_kpts, h1e_kpts, vhf):
        self.energy_veff_seen = vhf
        energy = cp.einsum('kij,kji->', h1e_kpts, dm_kpts).real / len(dm_kpts)
        return energy, 0.0

    @staticmethod
    def energy_nuc():
        return 0.0


class _CountingSetupKRKS(_FixedFockKRKS):
    """Record construction-time mean-field work separately from Fock calls."""

    def __init__(self, fock):
        self.setup_calls = {
            'build': 0,
            'get_ovlp': 0,
            'get_hcore': 0,
            'check_linear_dependency': 0,
            'energy_nuc': 0,
        }
        super().__init__(fock)

    def build(self):
        self.setup_calls['build'] += 1
        return self

    def get_ovlp(self, cell, kpts):
        self.setup_calls['get_ovlp'] += 1
        return super().get_ovlp(cell, kpts)

    def get_hcore(self, cell, kpts):
        self.setup_calls['get_hcore'] += 1
        return super().get_hcore(cell, kpts)

    def check_linear_dependency(self, overlap, **kwargs):
        self.setup_calls['check_linear_dependency'] += 1
        return super().check_linear_dependency(overlap, **kwargs)

    def energy_nuc(self):
        self.setup_calls['energy_nuc'] += 1
        return super().energy_nuc()


class _TaggedSolventKRKS(_FixedFockKRKS):
    """Mimic LPBE's tagged response potential and KSCF method signatures."""

    def __init__(self, hcore, solvent_potential):
        super().__init__(hcore)
        self._solvent_potential = cp.stack(solvent_potential)

    def get_veff(self, cell, dm, **kwargs):
        self.last_dm = dm
        self.veff_seen = tag_array(
            cp.zeros_like(dm), v_solvent=self._solvent_potential,
            e_solvent=0.0)
        return self.veff_seen

    def get_fock(self, h1e=None, vhf=None, dm=None, cycle=-1, diis=None,
                 level_shift_factor=None, damp_factor=None, **kwargs):
        assert cycle == -1 and diis is None
        return h1e + vhf + vhf.v_solvent

    def energy_elec(self, dm_kpts, h1e_kpts, vhf_kpts):
        self.energy_veff_seen = vhf_kpts
        fock = h1e_kpts + vhf_kpts.v_solvent
        energy = cp.einsum('kij,kji->', fock, dm_kpts).real / len(dm_kpts)
        return energy, 0.0


def _solver(mu=-0.1, checkpoint_path=None, initial_electron_number=None,
            cg_update='fletcher-reeves', cg_beta_max=5.0,
            electron_number=None, diis_switch_residual_rms=None,
            diis_max_objective_increase=1.0e-5,
            diis_max_delta_nelec=5.0e-2,
            line_search_nelec_guard_residual_rms=1.0e-2,
            line_search_max_delta_nelec=1.0,
            line_search_nelec_guard_max_delta_nelec=5.0e-2,
            canonical_continuation=False,
            canonical_continuation_max_delta_nelec=None,
            canonical_continuation_verification_residual_tol=1.0e-6,
            canonical_continuation_verification_density_tol=1.0e-9,
            line_search_method='strong-wolfe',
            line_search_c2=0.1,
            line_search_max_evals=12,
            line_search_max_trials=64,
            line_search_nelec_feasible_alpha=True,
            hager_zhang_objective_noise=1.0e-10,
            hager_zhang_max_evals=20,
            nlcg_exact_gradient_blend=True,
            nlcg_exact_gradient_polish=True,
            nlcg_residual_filter_rms=None,
            nlcg_residual_filter_max_relative_increase=0.0,
            nlcg_residual_filter_min_relative_reduction=2.0e-2,
            nlcg_residual_filter_objective_noise=1.0e-10,
            nlcg_residual_filter_warm_start=False,
            nlcg_residual_filter_initial_alpha=0.1,
            nlcg_residual_filter_alpha_min=0.02,
            nlcg_residual_filter_alpha_max=0.2,
            nlcg_residual_filter_max_evals=None,
            cg_restart_interval=20):
    f0 = cp.asarray([[[-0.7, 0.12j], [-0.12j, 0.3]]], dtype=cp.complex128)
    mf = _FixedFockKRKS(f0)
    config = GrandCanonicalConfig(
        max_cycle=50, required_consecutive_conv=1,
        conv_tol_omega=1.0e-10, conv_tol_grad_rms=1.0e-8,
        conv_tol_residual_rms=1.0e-7, conv_tol_density_rms=1.0e-9,
        conv_tol_nelec=1.0e-9, check_time_reversal=False,
        checkpoint_path=checkpoint_path,
        initial_electron_number=initial_electron_number,
        cg_update=cg_update,
        cg_beta_max=cg_beta_max,
        cg_restart_interval=cg_restart_interval,
        diis_switch_residual_rms=diis_switch_residual_rms,
        diis_max_objective_increase=diis_max_objective_increase,
        diis_max_delta_nelec=diis_max_delta_nelec,
        line_search_nelec_guard_residual_rms=(
            line_search_nelec_guard_residual_rms),
        line_search_max_delta_nelec=line_search_max_delta_nelec,
        line_search_nelec_guard_max_delta_nelec=(
            line_search_nelec_guard_max_delta_nelec),
        canonical_continuation=canonical_continuation,
        canonical_continuation_max_delta_nelec=(
            canonical_continuation_max_delta_nelec),
        canonical_continuation_verification_residual_tol=(
            canonical_continuation_verification_residual_tol),
        canonical_continuation_verification_density_tol=(
            canonical_continuation_verification_density_tol),
        line_search_method=line_search_method,
        line_search_c2=line_search_c2,
        line_search_max_evals=line_search_max_evals,
        line_search_max_trials=line_search_max_trials,
        line_search_nelec_feasible_alpha=(
            line_search_nelec_feasible_alpha),
        hager_zhang_objective_noise=hager_zhang_objective_noise,
        hager_zhang_max_evals=hager_zhang_max_evals,
        nlcg_exact_gradient_blend=nlcg_exact_gradient_blend,
        nlcg_exact_gradient_polish=nlcg_exact_gradient_polish,
        nlcg_residual_filter_rms=nlcg_residual_filter_rms,
        nlcg_residual_filter_max_relative_increase=(
            nlcg_residual_filter_max_relative_increase),
        nlcg_residual_filter_min_relative_reduction=(
            nlcg_residual_filter_min_relative_reduction),
        nlcg_residual_filter_objective_noise=(
            nlcg_residual_filter_objective_noise),
        nlcg_residual_filter_warm_start=(
            nlcg_residual_filter_warm_start),
        nlcg_residual_filter_initial_alpha=(
            nlcg_residual_filter_initial_alpha),
        nlcg_residual_filter_alpha_min=(
            nlcg_residual_filter_alpha_min),
        nlcg_residual_filter_alpha_max=(
            nlcg_residual_filter_alpha_max),
        nlcg_residual_filter_max_evals=(
            nlcg_residual_filter_max_evals),
    )
    return mf, GrandCanonicalKRKS(
        mf, mu=mu, sigma=0.15, config=config,
        electron_number=electron_number)


def test_stable_fermi_scalars_and_divided_difference():
    gamma = cp.asarray([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    q = fermi_occupations(gamma)
    entropy = fermi_entropy(gamma, q)
    assert bool(cp.all(cp.isfinite(q)))
    assert bool(cp.all(cp.isfinite(entropy)))
    assert float(q.min()) >= 0.0
    assert float(q.max()) <= 1.0
    assert abs(float((q[0] + q[-1]).item()) - 1.0) < 1.0e-14
    assert float(abs(entropy[0]).item()) == 0.0
    assert float(abs(entropy[-1]).item()) == 0.0

    degenerate = cp.asarray([0.4, 0.4 + 1.0e-13, -0.6])
    qd = fermi_occupations(degenerate)
    divided = fermi_divided_difference(degenerate, qd)
    assert bool(cp.all(cp.isfinite(divided)))
    assert float(cp.max(cp.abs(divided - divided.T)).item()) < 1.0e-13
    assert float(cp.max(divided).item()) <= 0.0
    assert abs(float((divided[0, 1] + qd[0] * (1.0 - qd[0])).item())) < 1.0e-10


def test_fixed_fock_gradient_and_evaluator_pairing():
    mf, solver = _solver()
    h = [cp.asarray([[-0.3, 0.08 + 0.03j], [0.08 - 0.03j, 0.2]])]
    direction = [cp.asarray([[0.15, 0.04j], [-0.04j, -0.07]])]
    state = solver.evaluate(h)
    epsilon = 1.0e-5
    plus = solver.evaluate(solver.axpy(epsilon, direction, h))
    minus = solver.evaluate(solver.axpy(-epsilon, direction, h))
    finite_difference = (plus.grand_potential - minus.grand_potential) / (2.0 * epsilon)
    analytic = solver.inner(state.gradient, direction)
    assert abs(finite_difference - analytic) < 2.0e-6
    assert solver.inner(state.gradient, state.residual) <= 1.0e-12
    assert mf.energy_veff_seen is mf.veff_seen


def test_cheap_fixed_mu_electron_number_matches_full_evaluation():
    _, solver = _solver()
    h = [cp.asarray([[-0.3, 0.08 + 0.03j],
                     [0.08 - 0.03j, 0.2]])]
    direction = [cp.asarray([[0.02, 0.01j], [-0.01j, -0.015]])]
    candidate = solver._sanitize_h(solver.axpy(0.2, direction, h))
    cheap = solver._cheap_fixed_mu_electron_number(candidate)
    full = solver.evaluate(candidate)
    assert abs(cheap - full.electron_number) < 1.0e-12


@pytest.mark.parametrize('alias, expected', [
    ('strong', 'strong-wolfe'),
    ('strong_wolfe', 'strong-wolfe'),
    ('HZ', 'hager-zhang'),
    ('cg descent', 'hager-zhang'),
])
def test_line_search_method_aliases(alias, expected):
    _, solver = _solver(line_search_method=alias)
    assert solver.config.line_search_method == expected


def test_hager_zhang_configuration_validation():
    with pytest.raises(ValueError, match='line_search_method'):
        _solver(line_search_method='golden-section')
    with pytest.raises(ValueError, match='Hager-Zhang constants'):
        config = GrandCanonicalConfig(hager_zhang_delta=0.6)
        GrandCanonicalKRKS(
            _FixedFockKRKS([cp.eye(2)]), mu=-0.1, sigma=0.15,
            config=config)
    with pytest.raises(ValueError, match='objective_noise'):
        _solver(hager_zhang_objective_noise=-1.0)
    with pytest.raises(ValueError, match='residual filter requires'):
        _solver(nlcg_residual_filter_rms=1.0e-3)
    with pytest.raises(ValueError, match='must be positive'):
        _solver(
            line_search_method='hager-zhang',
            nlcg_residual_filter_rms=0.0)
    with pytest.raises(ValueError, match='relative_reduction'):
        _solver(
            line_search_method='hager-zhang',
            nlcg_residual_filter_rms=1.0e-3,
            nlcg_residual_filter_min_relative_reduction=0.0)


def test_electron_number_prescreen_rejects_without_fock_build():
    mf, solver = _solver(
        line_search_nelec_guard_residual_rms=None,
        line_search_max_delta_nelec=1.0e-3,
        line_search_nelec_guard_max_delta_nelec=1.0e-3)
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    direction = [cp.eye(2, dtype=cp.complex128)]
    nfev_before = solver.nfev
    veff_before = mf.veff_calls
    trial = solver._trial(state, direction, 1.0)
    assert trial is None
    assert solver.nfev == nfev_before
    assert mf.veff_calls == veff_before
    assert solver.ncheap_nelec_reject == 1


def test_electron_number_prescreen_is_bypassed_at_fixed_n():
    mf, solver = _solver(
        electron_number=1.25,
        line_search_nelec_guard_residual_rms=1.0,
        line_search_max_delta_nelec=1.0e-12,
        line_search_nelec_guard_max_delta_nelec=1.0e-12)
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    direction = [cp.eye(2, dtype=cp.complex128)]
    nfev_before = solver.nfev
    veff_before = mf.veff_calls
    trial = solver._trial(state, direction, 1.0)
    assert trial is not None
    assert solver.nfev == nfev_before + 1
    assert mf.veff_calls == veff_before + 1
    assert solver.ncheap_nelec_reject == 0


def test_cheap_charge_bracketing_does_not_consume_fock_budget():
    mf, solver = _solver(
        line_search_nelec_guard_residual_rms=None,
        line_search_max_delta_nelec=1.0e-3,
        line_search_nelec_guard_max_delta_nelec=1.0e-3,
        line_search_max_evals=1)
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    identity = [cp.eye(2, dtype=cp.complex128)]
    slope = solver.inner(state.gradient, identity)
    direction = solver.scale_blocks(-np.copysign(1.0, slope), identity)
    nfev_before = solver.nfev
    veff_before = mf.veff_calls

    result = solver._armijo_fallback(state, direction)

    assert result.success, result.message
    assert result.nfev == 1
    assert solver.nfev == nfev_before + 1
    assert mf.veff_calls == veff_before + 1
    assert result.cheap_nelec_evaluations > 1
    assert result.cheap_nelec_alpha_reductions > 0


def test_fallback_line_search_work_is_combined():
    primary = _LineSearchResult(
        False, None, nfev=3, message='no bracket',
        line_search_method='hager-zhang', cheap_nelec_evaluations=7,
        cheap_nelec_alpha_reductions=2)
    fallback = _LineSearchResult(
        True, None, nfev=2, message='Armijo',
        line_search_method='armijo', cheap_nelec_evaluations=5,
        cheap_nelec_alpha_reductions=4)

    combined = GrandCanonicalKRKS._combine_line_search_work(
        primary, fallback)

    assert combined.nfev == 5
    assert combined.cheap_nelec_evaluations == 12
    assert combined.cheap_nelec_alpha_reductions == 6
    assert combined.line_search_method == 'armijo'
    assert 'Hager-Zhang'.lower() in combined.message.lower()


def test_charge_guard_checks_interior_of_endpoint_safe_line():
    h = cp.diag(cp.asarray([0.2, -0.6], dtype=cp.float64))
    solver = GrandCanonicalKRKS(
        _FixedFockKRKS([h]), mu=0.0, sigma=0.01,
        config=GrandCanonicalConfig(
            check_time_reversal=False,
            line_search_nelec_guard_residual_rms=None,
            line_search_max_delta_nelec=0.1,
            line_search_nelec_guard_max_delta_nelec=0.1))
    state = solver.evaluate([h])
    direction = [cp.diag(cp.asarray([-0.8, 0.8]))]

    cap, restricted = solver._charge_feasible_alpha_cap(
        state, direction, 1.0, maximum=0.1)
    midpoint = solver._trial(state, direction, 0.5)

    assert cap == pytest.approx(1.0)
    assert not restricted
    assert midpoint is None
    assert solver._last_trial_rejected_by_nelec


def test_electron_number_prescreen_configuration_validation():
    with pytest.raises(ValueError, match='guard_residual_rms'):
        _solver(line_search_nelec_guard_residual_rms=0.0)
    with pytest.raises(ValueError, match='max_delta_nelec'):
        _solver(line_search_max_delta_nelec=0.0)
    with pytest.raises(ValueError, match='may not exceed'):
        _solver(
            line_search_max_delta_nelec=0.5,
            line_search_nelec_guard_max_delta_nelec=0.6)


def test_zoom_accepts_armijo_point_at_electron_number_trust_boundary():
    _, solver = _solver()
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    direction = solver.scale_blocks(-1.0, state.gradient)
    dphi0 = solver.inner(state.gradient, direction)
    calls = []

    def trial_at_boundary(unused_state, unused_direction, alpha,
                          **unused_kwargs):
        calls.append(alpha)
        if len(calls) == 1:
            solver._last_trial_rejected_by_nelec = False
            solver.nfev += 1
            return replace(state, objective=state.objective - 1.0)
        solver._last_trial_rejected_by_nelec = True
        return None

    solver._trial = trial_at_boundary
    result = solver._zoom(
        state, direction, state.objective, dphi0,
        0.0, state, 1.0, None, None, 0)
    assert result.success
    assert result.alpha == pytest.approx(0.5)
    assert result.nfev == 1
    assert result.trust_boundary
    assert result.force_restart
    assert 'trust boundary' in result.message


def test_zoom_cheap_rejections_do_not_consume_expensive_budget(monkeypatch):
    _, solver = _solver()
    solver.config.line_search_zoom_evals = 1
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    direction = solver.scale_blocks(-1.0, state.gradient)
    dphi0 = solver.inner(state.gradient, direction)
    calls = []

    def delayed_feasible(unused_state, unused_direction, alpha,
                         **unused_kwargs):
        calls.append(alpha)
        if len(calls) <= 5:
            solver._last_trial_rejected_by_nelec = True
            return None
        solver._last_trial_rejected_by_nelec = False
        solver.nfev += 1
        return replace(state, objective=state.objective - 1.0)

    monkeypatch.setattr(solver, '_trial', delayed_feasible)
    result = solver._zoom(
        state, direction, state.objective, dphi0,
        0.0, state, 1.0, None, None, 0)

    assert result.success
    assert result.nfev == 1
    assert len(calls) == 6


def test_hager_zhang_weak_wolfe_caches_complex_fixed_fock_trials(
        monkeypatch):
    _, solver = _solver(
        electron_number=1.2, line_search_method='hager-zhang',
        hager_zhang_objective_noise=0.0,
        line_search_nelec_feasible_alpha=False)
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    direction = solver.copy_blocks(state.residual)
    original_trial = solver._trial
    calls = []

    def counted_trial(current, trial_direction, alpha):
        calls.append(alpha)
        return original_trial(current, trial_direction, alpha)

    monkeypatch.setattr(solver, '_trial', counted_trial)
    nfev_before = solver.nfev
    result = solver._line_search(state, direction)

    assert result.success, result.message
    assert result.weak_wolfe
    assert result.curvature_qualified
    assert not result.approximate_wolfe
    assert result.line_search_method == 'hager-zhang'
    assert result.nfev == solver.nfev - nfev_before
    assert len(calls) == len(set(calls))
    solver._verify_accepted_step(
        state, result.state, direction, result,
        solver.inner(state.gradient, direction))


@pytest.mark.parametrize('phi0', [-1.0, -1.0e6])
def test_hager_zhang_approximate_wolfe_uses_absolute_noise(
        monkeypatch, phi0):
    noise = 1.0e-8
    _, solver = _solver(
        electron_number=1.2, line_search_method='hager-zhang',
        hager_zhang_objective_noise=noise,
        line_search_nelec_feasible_alpha=False)
    evaluated = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    state = replace(evaluated, objective=phi0)
    direction = solver.scale_blocks(-1.0e-3, state.gradient)
    calls = []

    def approximate_trial(current, trial_direction, alpha):
        calls.append(alpha)
        solver.nfev += 1
        solver._last_trial_rejected_by_nelec = False
        return replace(
            current,
            h_orth=solver._sanitize_h(
                solver.axpy(alpha, trial_direction, current.h_orth)),
            objective=phi0 + 0.5 * noise,
            gradient=solver.scale_blocks(0.0, current.gradient))

    monkeypatch.setattr(solver, '_trial', approximate_trial)
    result = solver._line_search(state, direction)

    assert result.success, result.message
    assert result.approximate_wolfe
    assert result.weak_wolfe
    assert result.curvature_qualified
    assert result.objective_allowance == noise
    assert result.state.objective - state.objective == pytest.approx(
        0.5 * noise, abs=1.0e-11)
    assert calls == [pytest.approx(1.0)]
    solver._verify_accepted_step(
        state, result.state, direction, result,
        solver.inner(state.gradient, direction))


def test_hager_zhang_residual_filter_vetoes_wolfe_and_accepts_override(
        monkeypatch):
    filter_noise = 1.0e-10
    _, solver = _solver(
        electron_number=1.2, line_search_method='hager-zhang',
        hager_zhang_objective_noise=0.0,
        line_search_nelec_feasible_alpha=False,
        nlcg_residual_filter_rms=1.0e-3,
        nlcg_residual_filter_objective_noise=filter_noise)
    evaluated = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    state = replace(evaluated, residual_rms=1.0e-4)
    direction = solver.scale_blocks(-1.0e-3, state.gradient)
    dphi0 = solver.inner(state.gradient, direction)
    dd = solver.inner(direction, direction)
    calls = []

    def filtered_trial(current, trial_direction, alpha):
        calls.append(alpha)
        solver.nfev += 1
        solver._last_trial_rejected_by_nelec = False
        h = solver._sanitize_h(
            solver.axpy(alpha, trial_direction, current.h_orth))
        if len(calls) == 1:
            # Ordinary weak Wolfe, but a 50% residual increase: veto it.
            objective = current.objective + 0.2 * alpha * dphi0
            slope = 0.0
            residual_rms = 1.5 * current.residual_rms
        else:
            # Fails Wolfe curvature but lowers the residual by 20% while
            # staying inside the absolute objective allowance.
            objective = current.objective + 0.5 * filter_noise
            slope = dphi0
            residual_rms = 0.8 * current.residual_rms
        gradient = solver.scale_blocks(slope / dd, direction)
        return replace(
            current, h_orth=h, objective=objective,
            gradient=gradient, residual_rms=residual_rms)

    monkeypatch.setattr(solver, '_trial', filtered_trial)
    result = solver._line_search(state, direction)

    assert result.success, result.message
    assert calls == pytest.approx([1.0, 0.5])
    assert result.nfev == 2
    assert result.residual_filter_active
    assert result.residual_filter_qualified
    assert result.residual_filter_rejections == 1
    assert result.residual_filter_ratio == pytest.approx(0.8)
    assert result.force_restart
    assert not result.curvature_qualified
    assert solver.nresidual_filter_acceptances == 1
    solver._verify_accepted_step(
        state, result.state, direction, result, dphi0)


def test_hager_zhang_bounded_wolfe_retains_curvature_with_filter(
        monkeypatch):
    _, solver = _solver(
        electron_number=1.2, line_search_method='hager-zhang',
        hager_zhang_objective_noise=0.0,
        line_search_nelec_feasible_alpha=False,
        nlcg_residual_filter_rms=1.0e-3)
    evaluated = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    state = replace(evaluated, residual_rms=1.0e-4)
    direction = solver.scale_blocks(-1.0e-3, state.gradient)
    dphi0 = solver.inner(state.gradient, direction)

    def bounded_wolfe(current, trial_direction, alpha):
        solver.nfev += 1
        return replace(
            current,
            h_orth=solver._sanitize_h(
                solver.axpy(alpha, trial_direction, current.h_orth)),
            objective=current.objective + 0.2 * alpha * dphi0,
            gradient=solver.scale_blocks(0.0, current.gradient),
            residual_rms=0.9 * current.residual_rms)

    monkeypatch.setattr(solver, '_trial', bounded_wolfe)
    result = solver._line_search(state, direction)

    assert result.success
    assert result.weak_wolfe
    assert result.curvature_qualified
    assert result.residual_filter_active
    assert not result.residual_filter_qualified
    assert result.residual_filter_rejections == 0


def test_hager_zhang_residual_warm_start_reuses_clipped_accepted_alpha(
        monkeypatch):
    _, solver = _solver(
        line_search_method='hager-zhang',
        line_search_nelec_feasible_alpha=False,
        nlcg_residual_filter_rms=1.0e6,
        nlcg_residual_filter_warm_start=True,
        nlcg_residual_filter_initial_alpha=0.1,
        nlcg_residual_filter_alpha_min=0.02,
        nlcg_residual_filter_alpha_max=0.2)
    solver.config.max_cycle = 3
    h0 = [cp.asarray([[-0.1, 0.19 - 0.11j],
                      [0.19 + 0.11j, 0.6]])]
    starts = []
    accepted_alphas = iter([0.5, 0.005, 0.05])

    def accepted_line_search(current, unused_direction, *, alpha_init=None,
                             **unused_kwargs):
        starts.append(alpha_init)
        alpha = next(accepted_alphas)
        new = replace(
            current,
            objective=current.objective - 1.0e-3,
            grand_potential=current.grand_potential - 1.0e-3)
        return _LineSearchResult(
            True, new, alpha=alpha, force_restart=True,
            message='injected accepted step')

    monkeypatch.setattr(solver, '_line_search', accepted_line_search)
    monkeypatch.setattr(
        solver, '_verify_accepted_step', lambda *args, **kwargs: None)
    monkeypatch.setattr(solver, '_record', lambda *args, **kwargs: None)

    result = solver.kernel(h0=h0)

    assert result.niter == 3
    # First use the configured residual-mode alpha.  The accepted 0.5 is
    # clipped to 0.2 on the next cycle; accepted 0.005 is clipped to 0.02.
    assert starts == pytest.approx([0.1, 0.2, 0.02])
    assert solver._nlcg_residual_previous_alpha == pytest.approx(0.05)


@pytest.mark.parametrize('warm_start, residual_rms', [
    (False, 1.0e-4),
    (True, 2.0e-3),
])
def test_hager_zhang_residual_warm_start_inactive_uses_default_alpha(
        monkeypatch, warm_start, residual_rms):
    assert not GrandCanonicalConfig().nlcg_residual_filter_warm_start
    _, solver = _solver(
        line_search_method='hager-zhang',
        hager_zhang_objective_noise=0.0,
        line_search_nelec_feasible_alpha=False,
        nlcg_residual_filter_rms=1.0e-3,
        nlcg_residual_filter_warm_start=warm_start,
        nlcg_residual_filter_initial_alpha=0.1)
    evaluated = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    state = replace(evaluated, residual_rms=residual_rms)
    direction = solver.scale_blocks(-1.0e-3, state.gradient)
    dphi0 = solver.inner(state.gradient, direction)
    calls = []

    def ordinary_wolfe(current, trial_direction, alpha):
        calls.append(alpha)
        solver.nfev += 1
        return replace(
            current,
            h_orth=solver._sanitize_h(
                solver.axpy(alpha, trial_direction, current.h_orth)),
            objective=current.objective + 0.2 * alpha * dphi0,
            gradient=solver.scale_blocks(0.0, current.gradient),
            residual_rms=0.9 * current.residual_rms)

    monkeypatch.setattr(solver, '_trial', ordinary_wolfe)
    # A stale residual-mode alpha must not affect a search when either the
    # feature is disabled or the residual filter is not active.
    solver._nlcg_residual_previous_alpha = 0.05
    result = solver._line_search(state, direction)

    assert result.success
    assert calls == pytest.approx([1.0])
    assert result.alpha == pytest.approx(1.0)


def test_hager_zhang_residual_warm_start_limits_active_fock_evals(
        monkeypatch):
    _, solver = _solver(
        line_search_method='hager-zhang',
        hager_zhang_objective_noise=0.0,
        hager_zhang_max_evals=7,
        line_search_nelec_feasible_alpha=False,
        nlcg_residual_filter_rms=1.0e-3,
        nlcg_residual_filter_warm_start=True,
        nlcg_residual_filter_initial_alpha=0.1,
        nlcg_residual_filter_max_evals=2)
    evaluated = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    direction = solver.scale_blocks(-1.0e-3, evaluated.gradient)
    calls = []

    def failed_trial(current, trial_direction, alpha):
        del current, trial_direction
        calls.append(alpha)
        solver.nfev += 1
        solver._last_trial_rejected_by_nelec = False
        return None

    monkeypatch.setattr(solver, '_trial', failed_trial)

    active = replace(evaluated, residual_rms=1.0e-4)
    active_result = solver._line_search(
        active, direction,
        alpha_init=solver._nlcg_residual_alpha_init(active))
    assert not active_result.success
    assert active_result.nfev == 2
    assert calls == pytest.approx([0.1, 0.05])

    calls.clear()
    inactive = replace(evaluated, residual_rms=2.0e-3)
    inactive_result = solver._line_search(
        inactive, direction,
        alpha_init=solver._nlcg_residual_alpha_init(inactive))
    assert not inactive_result.success
    assert inactive_result.nfev == 7
    assert calls[0] == pytest.approx(1.0)
    assert len(calls) == 7


@pytest.mark.parametrize(
    'filter_rms, primary_evals, expected_override, fallback_evals', [
        (1.0e6, 1, [1], 1),
        (1.0e6, 2, [], 0),
        (1.0e-30, 2, [None], 3),
    ])
def test_kernel_residual_hz_and_fallback_share_eval_budget(
        monkeypatch, filter_rms, primary_evals, expected_override,
        fallback_evals):
    _, solver = _solver(
        line_search_method='hager-zhang',
        hager_zhang_max_evals=7,
        line_search_max_evals=3,
        line_search_nelec_feasible_alpha=False,
        nlcg_residual_filter_rms=filter_rms,
        nlcg_residual_filter_warm_start=True,
        nlcg_residual_filter_max_evals=2)
    solver.config.max_cycle = 1
    h0 = [cp.asarray([[-0.1, 0.19 - 0.11j],
                      [0.19 + 0.11j, 0.6]])]
    fallback_overrides = []
    fallback_trials = []

    def failed_hz(*unused_args, **unused_kwargs):
        solver.nfev += primary_evals
        return _LineSearchResult(
            False, None, nfev=primary_evals,
            line_search_method='hager-zhang',
            message='injected Hager-Zhang failure')

    def failed_fallback_trial(current, direction, alpha):
        del current, direction
        fallback_trials.append(alpha)
        solver.nfev += 1
        solver._last_trial_rejected_by_nelec = False
        return None

    original_fallback = solver._armijo_fallback

    def counted_fallback(*args, **kwargs):
        fallback_overrides.append(kwargs.get('max_evals_override'))
        return original_fallback(*args, **kwargs)

    monkeypatch.setattr(solver, '_line_search', failed_hz)
    monkeypatch.setattr(solver, '_trial', failed_fallback_trial)
    monkeypatch.setattr(solver, '_armijo_fallback', counted_fallback)

    result = solver.kernel(h0=h0)

    assert fallback_overrides == expected_override
    assert len(fallback_trials) == fallback_evals
    assert result.nfev == 1 + primary_evals + fallback_evals
    if filter_rms > 1.0:
        assert primary_evals + fallback_evals <= 2


def test_hager_zhang_residual_warm_start_resets_between_kernel_runs(
        monkeypatch):
    _, solver = _solver(
        line_search_method='hager-zhang',
        line_search_nelec_feasible_alpha=False,
        nlcg_residual_filter_rms=1.0,
        nlcg_residual_filter_warm_start=True)
    solver.config.max_cycle = 1
    h0 = [cp.asarray([[-0.1, 0.19 - 0.11j],
                      [0.19 + 0.11j, 0.6]])]
    previous_alphas = []

    def failed_line_search(*unused_args, **unused_kwargs):
        previous_alphas.append(solver._nlcg_residual_previous_alpha)
        solver._nlcg_residual_previous_alpha = 0.15
        return _LineSearchResult(False, None, message='injected failure')

    monkeypatch.setattr(solver, '_line_search', failed_line_search)
    monkeypatch.setattr(
        solver, '_armijo_fallback',
        lambda *args, **kwargs: _LineSearchResult(
            False, None, message='injected fallback failure'))

    solver.kernel(h0=h0)
    solver.kernel(h0=h0)

    assert previous_alphas == [None, None]


def test_fixed_electron_number_gradient_and_mu_constraint():
    target = 1.3
    _, solver = _solver(electron_number=target)
    h = [cp.asarray([[-0.3, 0.08 + 0.03j], [0.08 - 0.03j, 0.2]])]
    direction = [cp.asarray([[0.15, 0.04j], [-0.04j, -0.07]])]
    state = solver.evaluate(h)
    epsilon = 1.0e-5
    plus = solver.evaluate(solver.axpy(epsilon, direction, h))
    minus = solver.evaluate(solver.axpy(-epsilon, direction, h))
    finite_difference = (plus.free_energy - minus.free_energy) / (2.0 * epsilon)
    analytic = solver.inner(state.gradient, direction)
    assert abs(finite_difference - analytic) < 2.0e-6
    assert abs(state.electron_number - target) < solver.config.mu_electron_number_tol
    assert abs(plus.electron_number - target) < solver.config.mu_electron_number_tol
    assert abs(minus.electron_number - target) < solver.config.mu_electron_number_tol
    assert abs(solver.inner(state.gradient, solver.identity)) < 1.0e-12


def test_fixed_electron_number_minimisation_and_physical_mu():
    target = 1.25
    f0 = cp.asarray([[[-0.7, 0.12j], [-0.12j, 0.3]]], dtype=cp.complex128)
    mf = _FixedFockKRKS(f0)
    config = GrandCanonicalConfig(
        max_cycle=50, required_consecutive_conv=1,
        conv_tol_omega=1.0e-10, conv_tol_grad_rms=1.0e-8,
        conv_tol_residual_rms=1.0e-7, conv_tol_density_rms=1.0e-9,
        conv_tol_nelec=1.0e-9, check_time_reversal=False,
    )
    solver = GrandCanonicalKRKS(
        mf, sigma=0.15, config=config, electron_number=target)
    h0 = [cp.asarray([[-0.1, 0.19 - 0.11j], [0.19 + 0.11j, 0.6]])]
    result = solver.kernel(h0=h0)
    assert result.converged, result.message
    assert result.fixed_electron_number
    assert result.target_electron_number == target
    assert abs(result.electron_number - target) < config.mu_electron_number_tol
    assert abs(result.free_energy -
               (result.dft_total_energy + result.entropy_energy)) < 1.0e-13
    expected_mu = solver._solve_chemical_potential(
        [cp.linalg.eigvalsh(f0[0])])
    assert abs(result.mu - expected_mu) < 1.0e-7
    assert all(b.objective <= a.objective + 1.0e-11
               for a, b in zip(result.history, result.history[1:]))
    reconstructed = (mf.mo_coeff * mf.mo_occ[:, None, :]) @ mf.mo_coeff.conj().transpose(0, 2, 1)
    assert float(cp.max(cp.abs(reconstructed - result.dm_ao)).item()) < 1.0e-10


def test_fixed_electron_number_mu_increases_with_target():
    h0 = [cp.asarray([[-0.1, 0.19 - 0.11j], [0.19 + 0.11j, 0.6]])]
    _, low_solver = _solver(electron_number=0.8)
    _, high_solver = _solver(electron_number=1.4)
    low = low_solver.kernel(h0=h0)
    high = high_solver.kernel(h0=h0)
    assert low.converged, low.message
    assert high.converged, high.message
    assert high.mu > low.mu


def test_tagged_solvent_potential_is_included_in_fock_and_gradient():
    hcore = [cp.asarray([[-0.6, 0.0], [0.0, 0.2]])]
    v_solvent = [cp.asarray([[0.15, 0.04j], [-0.04j, -0.1]])]
    mf = _TaggedSolventKRKS(hcore, v_solvent)
    solver = GrandCanonicalKRKS(
        mf, mu=-0.05, sigma=0.2,
        config=GrandCanonicalConfig(check_time_reversal=False))
    h = [cp.asarray([[-0.25, 0.03], [0.03, 0.1]])]
    direction = [cp.asarray([[0.1, 0.02j], [-0.02j, -0.08]])]
    state = solver.evaluate(h)
    expected_fock = hcore[0] + v_solvent[0]
    assert float(cp.max(cp.abs(state.fock_ao[0] - expected_fock)).item()) < 1.0e-13
    epsilon = 1.0e-5
    plus = solver.evaluate(solver.axpy(epsilon, direction, h))
    minus = solver.evaluate(solver.axpy(-epsilon, direction, h))
    finite_difference = (plus.grand_potential - minus.grand_potential) / (2.0 * epsilon)
    assert abs(finite_difference - solver.inner(state.gradient, direction)) < 2.0e-6
    assert mf.energy_veff_seen is mf.veff_seen


def test_fixed_fock_direct_minimisation_and_final_density():
    mf, solver = _solver()
    h0 = [cp.asarray([[-0.1, 0.19 - 0.11j], [0.19 + 0.11j, 0.6]])]
    result = solver.kernel(h0=h0)
    assert result.converged, result.message
    assert result.grand_potential <= solver.history[0].grand_potential
    assert all(b.grand_potential <= a.grand_potential + 1.0e-11
               for a, b in zip(solver.history, solver.history[1:]))
    reconstructed = (mf.mo_coeff * mf.mo_occ[:, None, :]) @ mf.mo_coeff.conj().transpose(0, 2, 1)
    assert float(cp.max(cp.abs(reconstructed - result.dm_ao)).item()) < 1.0e-10


def test_cg_update_formulas_aliases_and_validation():
    aliases = {
        'FR': 'fletcher-reeves',
        'polak_ribiere': 'polak-ribiere',
        'Hestenes Stiefel': 'hestenes-stiefel',
        'Hager Zhang': 'hager-zhang',
    }
    h = [cp.asarray([[-0.3, 0.08 + 0.03j], [0.08 - 0.03j, 0.2]])]
    for alias, canonical in aliases.items():
        _, solver = _solver(cg_update=alias, cg_beta_max=1.0e6)
        assert solver.config.cg_update == canonical
        old = solver.evaluate(h)
        old_direction = solver.copy_blocks(old.residual)
        new = solver.evaluate(solver.axpy(0.1, old_direction, h))
        beta, reason = solver._cg_beta(old, new, old_direction)
        assert reason == ''

        if canonical == 'fletcher-reeves':
            numerator = solver.inner(new.gradient, new.z)
            denominator = solver.inner(old.gradient, old.z)
        elif canonical == 'hager-zhang':
            delta_gradient = solver.axpy(
                -1.0, old.gradient, new.gradient)
            delta_z = solver.axpy(-1.0, old.z, new.z)
            denominator = solver.inner(old_direction, delta_gradient)
            expected = (
                solver.inner(delta_gradient, new.z) / denominator -
                solver.config.hager_zhang_theta *
                solver.inner(delta_gradient, delta_z) *
                solver.inner(old_direction, new.gradient) /
                denominator**2)
        else:
            delta_z = solver.axpy(-1.0, old.z, new.z)
            numerator = solver.inner(new.gradient, delta_z)
            if canonical == 'polak-ribiere':
                denominator = solver.inner(old.gradient, old.z)
            else:
                delta_gradient = solver.axpy(-1.0, old.gradient, new.gradient)
                denominator = solver.inner(old_direction, delta_gradient)
        if canonical != 'hager-zhang':
            expected = numerator / denominator
        assert abs(beta - expected) < 1.0e-13

    with pytest.raises(ValueError, match='unsupported cg_update'):
        _solver(cg_update='not-a-cg-update')


def test_all_cg_updates_converge_fixed_fock_problem():
    h0 = [cp.asarray([[-0.1, 0.19 - 0.11j], [0.19 + 0.11j, 0.6]])]
    for update in ('fletcher-reeves', 'polak-ribiere', 'hestenes-stiefel',
                   'hager-zhang'):
        _, solver = _solver(cg_update=update)
        result = solver.kernel(h0=h0)
        assert result.converged, f'{update}: {result.message}'


def test_line_search_dispatch_can_disable_hz_residual_filter(monkeypatch):
    _, solver = _solver(
        line_search_method='hager-zhang',
        nlcg_residual_filter_rms=10.0)
    h0 = [cp.asarray([[-0.1, 0.19 - 0.11j],
                      [0.19 + 0.11j, 0.6]])]
    state = solver.evaluate(solver._initial_h(None, h0))

    def unexpected_filter(*unused_args, **unused_kwargs):
        pytest.fail('disabled NLCG residual filter was evaluated')

    monkeypatch.setattr(
        solver, '_residual_filter_metrics', unexpected_filter)
    result = solver._line_search(
        state, state.residual, method_override='hager-zhang',
        residual_filter_enabled=False)

    assert not result.residual_filter_active
    assert result.residual_filter_rejections == 0


def test_residual_diis_configuration_and_pulay_coefficients():
    _, solver = _solver(diis_switch_residual_rms=1.0e-3)
    assert solver.config.diis_switch_residual_rms == 1.0e-3
    with pytest.raises(ValueError, match='may not be smaller'):
        _solver(diis_switch_residual_rms=1.0e-8)

    fock = [cp.zeros((2, 2), dtype=cp.complex128)]
    history = [
        _DIISItem(fock, [cp.diag(cp.asarray([1.0, 0.0]))]),
        _DIISItem(fock, [cp.diag(cp.asarray([0.0, 1.0]))]),
    ]
    coefficients, condition, coefficient_l1, action = (
        solver._diis_coefficients(history))
    assert action == ''
    assert condition == pytest.approx(1.0)
    assert coefficient_l1 == pytest.approx(1.0)
    assert np.allclose(coefficients, [0.5, 0.5])


def test_residual_diis_prunes_unsafe_condition_and_coefficients():
    zero = [cp.zeros((2, 2), dtype=cp.complex128)]
    duplicate = [cp.diag(cp.asarray([1.0, 0.0]))]
    _, solver = _solver()
    solver.config.diis_max_condition = 1.0e6
    history = [
        _DIISItem(zero, duplicate),
        _DIISItem(zero, duplicate),
    ]
    _, condition, _, action = solver._diis_coefficients(history)
    assert condition > solver.config.diis_max_condition
    assert len(history) == 1
    assert action == 'reset ill-conditioned DIIS history'

    # These nearly parallel residuals have a tolerable Gram condition number,
    # but their cancellation requires coefficients with a large L1 norm.
    _, solver = _solver()
    solver.config.diis_max_condition = 1.0e12
    solver.config.diis_max_coefficient_l1 = 10.0
    history = [
        _DIISItem(zero, [cp.diag(cp.asarray([1.0, 0.0]))]),
        _DIISItem(
            zero, [cp.diag(cp.asarray([1.01, 0.01]))]),
    ]
    _, condition, _, action = solver._diis_coefficients(history)
    assert condition < solver.config.diis_max_condition
    assert len(history) == 1
    assert action == 'reset ill-conditioned DIIS history'


def test_residual_diis_trust_ratio_updates_next_damping():
    _, solver = _solver(diis_switch_residual_rms=1.0e-3)
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    damping = 0.25
    predicted = solver._diis_predicted_residual_rms(state, damping)
    assert predicted == pytest.approx(0.75 * state.residual_rms)

    good = replace(state, residual_rms=0.5 * state.residual_rms)
    next_damping, ratio = solver._next_diis_damping(
        state, good, predicted, damping, damping)
    assert ratio == pytest.approx(2.0)
    assert next_damping == pytest.approx(0.5)

    poor = replace(state, residual_rms=0.96 * state.residual_rms)
    next_damping, ratio = solver._next_diis_damping(
        state, poor, predicted, damping, damping)
    assert ratio == pytest.approx(0.16)
    assert next_damping == pytest.approx(0.125)

    # A high agreement ratio alone must not expand a trust radius that made
    # negligible absolute progress.
    small_damping = 0.005
    small_prediction = solver._diis_predicted_residual_rms(
        state, small_damping)
    stagnant = replace(state, residual_rms=0.99 * state.residual_rms)
    next_damping, ratio = solver._next_diis_damping(
        state, stagnant, small_prediction, small_damping, small_damping)
    assert ratio == pytest.approx(2.0)
    assert next_damping == pytest.approx(small_damping)

    # Experiments may opt into the standard ratio-only expansion rule while
    # the default retains the extra 2% progress floor above.
    solver.config.diis_trust_expand_min_relative_reduction = 0.0
    next_damping, ratio = solver._next_diis_damping(
        state, stagnant, small_prediction, small_damping, small_damping)
    assert ratio == pytest.approx(2.0)
    assert next_damping == pytest.approx(2.0 * small_damping)

    with pytest.raises(ValueError, match='trust ratios'):
        config = GrandCanonicalConfig(
            diis_trust_shrink_ratio=0.8,
            diis_trust_expand_ratio=0.5)
        GrandCanonicalKRKS(_FixedFockKRKS([
            cp.eye(2, dtype=cp.complex128)]), mu=-0.1, sigma=0.15,
            config=config)


def test_diis_backtracks_until_trial_reduces_residual():
    _, solver = _solver(diis_switch_residual_rms=1.0e-3)
    solver.config.diis_max_backtracks = 8
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    target = solver.axpy(
        0.1, [cp.eye(2, dtype=cp.complex128)], state.h_orth)
    factors = iter((3.0, 2.0, 0.99))
    calls = []

    def trial(current, direction, alpha):
        calls.append(alpha)
        return replace(
            current, residual_rms=next(factors) * current.residual_rms)

    solver._trial = trial
    accepted, damping, reason = solver._try_diis_target(
        state, target, starting_damping=0.1)
    assert accepted is not None
    assert damping == pytest.approx(0.025)
    assert reason == ''
    assert calls == pytest.approx([0.1, 0.05, 0.025])


def test_diis_falls_back_to_latest_fock_after_rejected_pulay_model():
    _, solver = _solver(diis_switch_residual_rms=1.0e-3)
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    zero = [cp.zeros((2, 2), dtype=cp.complex128)]
    history = [
        _DIISItem([cp.eye(2)], [cp.diag(cp.asarray([1.0, 0.0]))]),
        _DIISItem([3.0 * cp.eye(2)],
                  [cp.diag(cp.asarray([0.0, 1.0]))]),
    ]
    accepted = replace(state, residual_rms=0.5 * state.residual_rms)
    calls = []

    solver._diis_coefficients = lambda items: (
        np.full(len(items), 1.0 / len(items)), 1.0, 1.0, '')

    def try_target(unused_state, target, starting_damping,
                   max_backtracks=None):
        calls.append((solver.copy_blocks(target), starting_damping,
                      max_backtracks))
        if len(calls) == 1:
            return None, 0.0, 'rejected test Pulay model'
        return accepted, 0.125, ''

    solver._try_diis_target = try_target
    result = solver._diis_step(
        state, history, starting_damping=0.5)

    assert result.step.success
    assert result.step.state is accepted
    assert result.step.alpha == pytest.approx(0.125)
    assert [call[2] for call in calls] == [2, solver.config.diis_max_backtracks]
    assert all(call[1] == pytest.approx(0.5) for call in calls)
    assert float(cp.max(cp.abs(
        calls[0][0][0] - 2.0 * cp.eye(2))).item()) < 1.0e-14
    assert float(cp.max(cp.abs(
        calls[1][0][0] - history[-1].fock[0])).item()) < 1.0e-14
    assert 'latest-Fock fallback' in result.history_action


def test_diis_context_resumes_without_seed_evaluation_and_keeps_memory():
    mf, solver = _solver(electron_number=1.0)
    solver.config.diis_initial_damping = 0.25
    state = solver.evaluate([
        cp.asarray([[0.3, 0.19 + 0.04j],
                    [0.19 - 0.04j, -0.2]])])
    initial_residual = state.residual_rms
    context = solver._new_diis_context(
        state, None, niter=0, cycle_start=0)
    history = context.history

    coarse = solver._advance_diis(
        context, residual_tolerance=0.8 * initial_residual)
    assert coarse.converged
    assert context.niter == 1
    assert context.next_cycle == 1
    assert context.history is history
    assert len(history) == 1
    assert context.damping_hint == pytest.approx(0.5)

    calls_before = mf.veff_calls
    niter_before = context.niter
    tighter = solver._advance_diis(context, residual_tolerance=1.0e-8)
    assert tighter.converged
    assert context.history is history
    assert context.niter > niter_before
    assert context.next_cycle == context.niter
    # Every resumed Fock is a DIIS trial; there is no repeated seed build.
    assert mf.veff_calls - calls_before == context.niter - niter_before
    assert solver.nfev == mf.veff_calls


def test_canonical_session_refinement_preserves_history_and_accounts_incrementally(
        monkeypatch):
    mf, solver = _solver(electron_number=1.0)
    solver.config.diis_initial_damping = 0.25
    state = solver.evaluate([
        cp.asarray([[0.3, 0.19 + 0.04j],
                    [0.19 - 0.04j, -0.2]])])
    context = solver._new_diis_context(
        state, None, niter=0, cycle_start=0)
    session = _FixedNSession(1.0, solver, context)
    work = _CanonicalWork(0, [])

    coarse = solver._advance_canonical_session(
        session, 0.8 * state.residual_rms, work)
    assert coarse.converged
    assert context.history
    retained = tuple(context.history)
    damping = context.damping_hint

    # Exhausting the cumulative cycle budget proves that refinement does not
    # spend another seed evaluation merely to restart the DIIS driver.
    solver.config.max_cycle = context.next_cycle
    published = (
        solver.nfev, context.niter, len(solver.history),
        work.fixed_n_nfev, work.fixed_n_niter, len(work.history),
        mf.veff_calls)
    stopped = solver._advance_canonical_session(session, 1.0e-12, work)
    assert not stopped.converged
    assert published == (
        solver.nfev, context.niter, len(solver.history),
        work.fixed_n_nfev, work.fixed_n_niter, len(work.history),
        mf.veff_calls)

    seen = []
    original_step = solver._diis_step

    def inspect_step(current, history, starting_damping, **kwargs):
        seen.append((tuple(history[:len(retained)]), starting_damping))
        return original_step(
            current, history, starting_damping, **kwargs)

    monkeypatch.setattr(solver, '_diis_step', inspect_step)
    solver.config.max_cycle = context.next_cycle + 1
    before_nfev = solver.nfev
    before_niter = context.niter
    before_history = len(work.history)
    before_veff = mf.veff_calls
    solver._advance_canonical_session(session, 1.0e-12, work)

    assert len(seen) == 1
    assert all(actual is expected
               for actual, expected in zip(seen[0][0], retained))
    assert seen[0][1] == pytest.approx(damping)
    assert context.next_cycle == solver.config.max_cycle
    assert context.niter == before_niter + 1
    assert solver.nfev - before_nfev == mf.veff_calls - before_veff
    assert work.fixed_n_nfev == solver.nfev
    assert work.fixed_n_niter == context.niter
    assert len(work.history) - before_history == 1
    assert [record.cycle for record in work.history] == list(
        range(context.niter))
    assert [record.fock_evaluations for record in work.history] == sorted(
        record.fock_evaluations for record in work.history)


def test_repeated_secant_sample_uses_its_retained_session(monkeypatch):
    _, solver = _solver(mu=-0.1, canonical_continuation=True)
    monkeypatch.setattr(solver, '_canonical_bracket', lambda samples: None)
    first_nelec = []

    def repeat_first_sample(samples, physical_fock, current_nelec):
        if not first_nelec:
            first_nelec.append(samples[0].electron_number)
            return current_nelec + 0.02
        return first_nelec[0]

    monkeypatch.setattr(
        solver, '_canonical_continuation_proposal', repeat_first_sample)
    original_advance = solver._advance_canonical_session

    class ReusedSession(RuntimeError):
        pass

    def detect_resume(session, residual_tolerance, work):
        if session.published_nfev:
            raise ReusedSession
        return original_advance(session, residual_tolerance, work)

    monkeypatch.setattr(solver, '_advance_canonical_session', detect_resume)
    h0 = [cp.asarray([
        [-0.25, 0.2 + 0.05j],
        [0.2 - 0.05j, 0.1]], dtype=cp.complex128)]

    with pytest.raises(ReusedSession):
        solver._kernel_canonical_continuation(h0=h0)


def test_canonical_session_pruning_keeps_latest_and_bracket_endpoints():
    _, solver = _solver(canonical_continuation=True)
    sessions = [object() for _ in range(5)]
    samples = [
        _CanonicalSample(
            float(index), -1.0 + 0.5 * index, 1.0e-8,
            solver.copy_blocks(solver.hcore_ao), sessions[index])
        for index in range(5)
    ]
    bracket = (samples[0], samples[2])

    solver._prune_canonical_sessions(samples, bracket)

    assert samples[0].session is sessions[0]
    assert samples[1].session is None
    assert samples[2].session is sessions[2]
    assert samples[3].session is sessions[3]
    assert samples[4].session is sessions[4]


def test_residual_diis_accepts_noise_scale_objective_change_but_not_charge_jump():
    _, solver = _solver(
        diis_switch_residual_rms=1.0e-3,
        diis_max_objective_increase=1.0e-5,
        diis_max_delta_nelec=5.0e-2)
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    trial = replace(
        state, residual_rms=0.5 * state.residual_rms,
        objective=state.objective + 5.0e-6,
        electron_number=state.electron_number + 1.0e-2)
    acceptable, reason = solver._diis_trial_acceptable(state, trial)
    assert acceptable
    assert reason == ''

    large_charge_change = replace(
        trial, electron_number=state.electron_number + 1.0e-1)
    acceptable, reason = solver._diis_trial_acceptable(
        state, large_charge_change)
    assert not acceptable
    assert 'electron-number' in reason

    large_objective_change = replace(
        trial, objective=state.objective + 2.0e-5)
    acceptable, reason = solver._diis_trial_acceptable(
        state, large_objective_change)
    assert not acceptable
    assert 'objective increase' in reason


def test_residual_diis_polishes_fixed_fock():
    h0 = [cp.asarray([[-0.1, 0.19 - 0.11j],
                      [0.19 + 0.11j, 0.6]])]
    _, solver = _solver(
        diis_switch_residual_rms=1.0, diis_max_delta_nelec=2.0)
    result = solver.kernel(h0=h0)
    assert result.converged, result.message
    assert result.message == 'converged residual-DIIS fixed point'
    assert result.niter == 1
    assert result.history[0].optimizer == 'diis'
    assert result.history[0].search_direction_source == 'residual-diis'
    assert result.history[0].diis_history_size == 1
    assert result.history[0].diis_damping == 1.0
    assert result.residual_rms < solver.config.conv_tol_residual_rms


def test_workspace_prepares_static_mean_field_data_once_and_is_frozen():
    fock = [cp.asarray([[-0.7, 0.12j], [-0.12j, 0.3]],
                       dtype=cp.complex128)]
    mf = _CountingSetupKRKS(fock)

    solver = GrandCanonicalKRKS(
        mf, mu=-0.1, sigma=0.15,
        config=GrandCanonicalConfig(check_time_reversal=False))

    assert mf.setup_calls == {
        'build': 1,
        'get_ovlp': 1,
        'get_hcore': 1,
        'check_linear_dependency': 1,
        'energy_nuc': 1,
    }
    with pytest.raises(FrozenInstanceError):
        solver._workspace.nuclear_energy = 1.0


def test_canonical_inner_points_share_workspace_without_publishing(
        monkeypatch):
    fock = [cp.asarray([[-0.7, 0.12j], [-0.12j, 0.3]],
                       dtype=cp.complex128)]
    mf = _CountingSetupKRKS(fock)
    solver = GrandCanonicalKRKS(
        mf, mu=-0.1, sigma=0.15,
        config=GrandCanonicalConfig(
            check_time_reversal=False, canonical_continuation=True))
    expected_setup_calls = dict(mf.setup_calls)
    children = []
    advances = []
    original_spawn = solver._spawn_fixed_n

    def observed_spawn(electron_number, config):
        child = original_spawn(electron_number, config)
        children.append(child)
        original_advance_diis = child._advance_diis

        def unexpected_finalize(*args, **kwargs):
            raise AssertionError('canonical child published through _finalize')

        def observed_advance_diis(context, residual_tolerance=None):
            before = (child.nfev, context.niter, len(child.history))
            outcome = original_advance_diis(context, residual_tolerance)
            advances.append((child, context, before, outcome))
            return outcome

        monkeypatch.setattr(child, '_finalize', unexpected_finalize)
        monkeypatch.setattr(child, '_advance_diis', observed_advance_diis)
        return child

    monkeypatch.setattr(solver, '_spawn_fixed_n', observed_spawn)
    result = solver.kernel(h0=[
        cp.asarray([[0.25, 0.18 - 0.07j],
                    [0.18 + 0.07j, -0.15]])])

    assert result.converged, result.message
    assert children
    assert set(child for child, _, _, _ in advances) == set(children)
    assert all(child._workspace is solver._workspace for child in children)
    assert all(outcome.state is context.state
               for _, context, _, outcome in advances)
    assert sum(child.nfev for child in children) == (
        result.canonical_continuation_evaluations)
    assert sum(len(child.history) for child in children) == result.niter
    assert result.niter == len(result.history)
    assert mf.setup_calls == expected_setup_calls
    assert mf.converged
    assert mf.e_tot == pytest.approx(result.dft_total_energy)
    assert mf.grand_potential == pytest.approx(result.grand_potential)
    assert mf.mo_coeff is not None
    assert mf.mo_occ is not None


def test_automatic_canonical_continuation_finds_fixed_mu_electron_number():
    mf, solver = _solver(
        mu=-0.1, canonical_continuation=True)
    h0 = [cp.asarray([[0.25, 0.18 - 0.07j],
                      [0.18 + 0.07j, -0.15]])]
    expected_nelec = solver._electron_number_at_mu(solver.hcore_ao, solver.mu)
    result = solver.kernel(h0=h0)
    assert result.converged, result.message
    assert not result.fixed_electron_number
    assert result.canonical_continuation_steps >= 1
    assert result.canonical_continuation_evaluations >= 1
    assert np.isfinite(result.canonical_continuation_mu_error)
    assert abs(result.canonical_continuation_mu_error) <= (
        solver.config.canonical_continuation_handoff_delta_mu)
    assert abs(result.canonical_continuation_delta_nelec) <= (
        solver.config.canonical_continuation_handoff_delta_nelec)
    assert abs(result.electron_number - expected_nelec) <= (
        solver.config.canonical_continuation_handoff_delta_nelec)
    assert result.nfev == mf.veff_calls
    assert result.nfev > result.canonical_continuation_evaluations
    assert result.canonical_continuation_refinements >= 0
    assert solver.config.diis_max_coefficient_l1 == pytest.approx(10.0)


@pytest.mark.parametrize(
    'canonical_continuation,electron_number,expected_route', [
        (True, None, 'canonical'),
        (False, None, 'nlcg'),
        (True, 1.0, 'nlcg'),
    ])
def test_kernel_routes_canonical_and_direct_solves(
        canonical_continuation, electron_number, expected_route):
    _, solver = _solver(
        canonical_continuation=canonical_continuation,
        electron_number=electron_number)
    sentinel = object()
    calls = []

    def canonical_kernel(*args, **kwargs):
        calls.append(('canonical', args, kwargs))
        return sentinel

    def nlcg_kernel(*args, **kwargs):
        calls.append(('nlcg', args, kwargs))
        return sentinel

    solver._kernel_canonical_continuation = canonical_kernel
    solver._kernel_nlcg = nlcg_kernel
    assert solver.kernel(h0=solver.hcore_ao) is sentinel
    assert len(calls) == 1
    assert calls[0][0] == expected_route


def test_canonical_continuation_uses_plain_secant_after_first_proposal():
    _, solver = _solver(canonical_continuation=True)
    inner_config = solver._canonical_continuation_config(1.0e-4, 0.125)
    assert inner_config.diis_max_coefficient_l1 == pytest.approx(50.0)
    proposal = solver._canonical_continuation_proposal(
        [(1.0, -0.04), (1.03, -0.03)], solver.hcore_ao,
        current_nelec=1.03)
    assert proposal == pytest.approx(1.12)


def test_canonical_secant_cap_is_ten_percent_of_neutral_electrons():
    _, solver = _solver(canonical_continuation=True)
    assert solver.mf.cell.nelectron == 2
    assert solver.config.canonical_continuation_max_delta_nelec is None
    proposal = solver._canonical_continuation_proposal(
        [(1.0, -1.0), (1.01, -0.99)], solver.hcore_ao,
        current_nelec=1.01)
    assert proposal == pytest.approx(1.21)


def test_canonical_deprecated_absolute_secant_cap_remains_supported():
    _, solver = _solver(
        canonical_continuation=True,
        canonical_continuation_max_delta_nelec=0.075)
    proposal = solver._canonical_continuation_proposal(
        [(1.0, -1.0), (1.01, -0.99)], solver.hcore_ao,
        current_nelec=1.01)
    assert solver._canonical_secant_step_cap() == pytest.approx(0.075)
    assert proposal == pytest.approx(1.085)

    with pytest.raises(ValueError, match='max_delta_nelec'):
        _solver(
            canonical_continuation=True,
            canonical_continuation_max_delta_nelec=0.0)


def test_canonical_secant_uses_optimized_mu_error_across_a_bracket():
    _, solver = _solver(canonical_continuation=True)
    proposal = solver._canonical_continuation_proposal(
        [(1.0, -0.4), (1.1, 0.6)], solver.hcore_ao,
        current_nelec=1.1)
    assert proposal == pytest.approx(1.04)


def test_canonical_continuation_default_handoff_charge_is_conservative():
    _, solver = _solver(canonical_continuation=True)
    assert (solver.config.canonical_continuation_handoff_delta_nelec ==
            pytest.approx(2.0e-5))


def test_canonical_only_terminal_defaults_are_tight():
    _, solver = _solver(canonical_continuation=True)
    assert solver.config.canonical_continuation_handoff_delta_mu == pytest.approx(
        1.0e-6)
    assert (solver.config.canonical_continuation_handoff_delta_nelec ==
            pytest.approx(2.0e-5))
    assert (solver.config.canonical_continuation_unbracketed_handoff_delta_nelec ==
            pytest.approx(2.0e-5))
    assert (solver.config.canonical_continuation_verification_residual_tol ==
            pytest.approx(1.0e-6))
    assert (solver.config.canonical_continuation_verification_density_tol ==
            pytest.approx(1.0e-9))
    assert solver.config.canonical_continuation_root_nelec_tol == pytest.approx(
        1.0e-8)
    assert (solver.config.canonical_continuation_bracketed_residual_tol ==
            pytest.approx(1.0e-8))


def test_canonical_fixed_mu_candidate_is_gauge_exact_and_uses_fock_charge():
    fock0 = cp.asarray([[-0.7, 0.12 + 0.04j],
                       [0.12 - 0.04j, 0.3]], dtype=cp.complex128)
    mf = _FixedFockKRKS([fock0, fock0.conj()])
    mf.kpts = np.asarray([[0.25, 0.0, 0.0], [-0.25, 0.0, 0.0]])
    config = GrandCanonicalConfig(
        check_time_reversal=True, enforce_time_reversal=True)
    target_mu = -0.1
    fixed_mu_solver = GrandCanonicalKRKS(
        mf, mu=target_mu, sigma=0.15, config=replace(config))
    target_nelec = fixed_mu_solver._electron_number_at_mu(
        fixed_mu_solver.hcore_ao, target_mu)
    fixed_n_solver = GrandCanonicalKRKS(
        mf, mu=target_mu, sigma=0.15, config=replace(config),
        electron_number=target_nelec)

    gauge = 0.37
    h_fixed_n = [
        fock + gauge * identity
        for fock, identity in zip(
            fixed_n_solver.hcore_ao, fixed_n_solver.identity)]
    state = fixed_n_solver.evaluate(h_fixed_n)
    canonical = fixed_n_solver._finalize(
        state, True, 'synthetic converged fixed-N state', 0, 0.0)
    assert canonical.mu == pytest.approx(target_mu, abs=2.0e-12)

    nfev_before = fixed_mu_solver.nfev
    veff_before = mf.veff_calls
    candidate, measured_gauge, delta_nelec, predicted_residual = (
        fixed_mu_solver._canonical_fixed_mu_candidate(canonical))
    assert fixed_mu_solver.nfev == nfev_before
    assert mf.veff_calls == veff_before
    assert measured_gauge == pytest.approx(gauge, abs=2.0e-12)
    assert delta_nelec == pytest.approx(0.0, abs=2.0e-12)
    assert predicted_residual == pytest.approx(0.0, abs=2.0e-12)
    assert fixed_mu_solver.max_block_rms(fixed_mu_solver.axpy(
        -1.0, fixed_mu_solver.hcore_ao, candidate)) < 2.0e-12
    assert fixed_mu_solver._electron_number_at_mu(
        candidate, target_mu) == pytest.approx(target_nelec, abs=2.0e-12)

    # Using the canonical auxiliary H directly would make this gate depend on
    # its arbitrary scalar gauge and reject an otherwise exact physical Fock.
    wrong_delta_nelec = (
        fixed_mu_solver._electron_number_at_mu(
            canonical.h_orth, target_mu) - canonical.electron_number)
    assert abs(wrong_delta_nelec) > 1.0e-2

    extra_gauge = -0.23
    shifted = replace(canonical, h_orth=[
        h + extra_gauge * identity
        for h, identity in zip(canonical.h_orth, fixed_mu_solver.identity)])
    shifted_candidate, shifted_gauge, shifted_delta, shifted_residual = (
        fixed_mu_solver._canonical_fixed_mu_candidate(shifted))
    assert shifted_gauge == pytest.approx(
        measured_gauge + extra_gauge, abs=2.0e-12)
    assert shifted_delta == pytest.approx(delta_nelec, abs=2.0e-12)
    assert shifted_residual == pytest.approx(
        predicted_residual, abs=2.0e-12)
    assert fixed_mu_solver.max_block_rms(fixed_mu_solver.axpy(
        -1.0, candidate, shifted_candidate)) < 2.0e-12
    assert float(cp.max(cp.abs(
        shifted_candidate[1] - shifted_candidate[0].conj())).item()) < 2.0e-12


def test_canonical_continuation_uses_one_gauge_exact_verification_fock(
        monkeypatch):
    mf, solver = _solver(
        mu=-0.1, canonical_continuation=True)
    target_mu = solver.mu
    canonical_candidates = []
    original_candidate = solver._canonical_fixed_mu_candidate

    def observed_candidate(canonical_result):
        canonical_candidates.append(canonical_result)
        return original_candidate(canonical_result)

    monkeypatch.setattr(
        solver, '_canonical_fixed_mu_candidate', observed_candidate)

    verification_h = []
    original_evaluate = solver.evaluate
    reported_grad_rms = 100.0 * solver.config.conv_tol_grad_rms

    def observed_evaluate(h_orth):
        verification_h.append(solver.copy_blocks(h_orth))
        # Canonical/fixed-point termination is governed by the physical Fock
        # residual.  The exact direct-minimization gradient is retained as a
        # diagnostic but must not veto an otherwise verified canonical root.
        return replace(
            original_evaluate(h_orth), grad_rms=reported_grad_rms)

    monkeypatch.setattr(solver, 'evaluate', observed_evaluate)

    def unexpected_iterative_fixed_mu(*args, **kwargs):
        raise AssertionError(
            'canonical continuation entered an iterative fixed-mu solve')

    # These are instance attributes, so fixed-N child solvers retain their
    # ordinary class methods while an iterative fixed-mu solve is blocked.
    monkeypatch.setattr(solver, '_kernel_nlcg', unexpected_iterative_fixed_mu)
    monkeypatch.setattr(solver, '_kernel_diis', unexpected_iterative_fixed_mu)

    result = solver.kernel(h0=[
        cp.asarray([[0.25, 0.18 - 0.07j],
                    [0.18 + 0.07j, -0.15]])])

    assert result.converged, result.message
    assert not result.fixed_electron_number
    assert result.mu == pytest.approx(target_mu, abs=1.0e-14)
    assert result.canonical_verification_attempts == 1
    assert result.canonical_verification_evaluations == 1
    assert result.canonical_verification_failures == 0
    assert result.canonical_verification_residual_rms <= (
        solver.config.canonical_continuation_verification_residual_tol)
    assert result.canonical_verification_grad_rms == pytest.approx(
        reported_grad_rms)
    assert abs(result.canonical_verification_delta_nelec) <= (
        solver.config.canonical_continuation_handoff_delta_nelec)
    assert result.canonical_verification_density_rms <= (
        solver.config.canonical_continuation_verification_density_tol)
    assert result.canonical_terminal_mode == 'canonical-verification'
    assert len(verification_h) == 1
    assert canonical_candidates

    # A fixed-N result reports the physical optimized mu after removing its
    # scalar gauge g = mean(H-F).  The unique fixed-mu auxiliary Hamiltonian
    # that preserves its occupations is therefore
    # H_mu = H_N - g I + (mu_target - mu_opt) I.
    candidates = []
    for canonical in canonical_candidates:
        gauge = solver.trace_mean([
            h - f for h, f in zip(
                canonical.h_orth, canonical.fock_orth)])
        shifted = [
            h + (target_mu - canonical.mu - gauge) * identity
            for h, identity in zip(canonical.h_orth, solver.identity)]
        error = solver.max_block_rms(
            solver.axpy(-1.0, shifted, verification_h[0]))
        candidates.append((error, canonical))
    shift_error, source = min(candidates, key=lambda item: item[0])
    assert shift_error < 1.0e-12

    assert result.electron_number == pytest.approx(
        source.electron_number, abs=2.0e-10)
    assert float(cp.max(cp.abs(result.dm_ao - source.dm_ao)).item()) < 2.0e-10
    for actual, expected in zip(result.occupations, source.occupations):
        assert float(cp.max(cp.abs(actual - expected)).item()) < 2.0e-10
    assert result.grand_potential == pytest.approx(
        result.free_energy - target_mu * result.electron_number,
        abs=1.0e-12)

    # The verification is an evaluation, not an optimizer iteration.
    assert result.nfev == mf.veff_calls
    assert result.nfev == (
        result.canonical_continuation_evaluations +
        result.canonical_verification_evaluations)
    assert result.niter == len(result.history)
    fock_counts = [record.fock_evaluations for record in result.history]
    assert fock_counts == sorted(fock_counts)
    assert all(value < result.nfev for value in fock_counts)
    assert mf.canonical_verification_attempts_gc == 1
    assert mf.canonical_verification_evaluations_gc == 1
    assert mf.canonical_verification_failures_gc == 0
    assert mf.canonical_terminal_mode_gc == 'canonical-verification'
    for name in (
            'canonical_verification_attempts',
            'canonical_verification_evaluations',
            'canonical_verification_failures',
            'canonical_verification_residual_rms',
            'canonical_verification_grad_rms',
            'canonical_verification_delta_nelec',
            'canonical_verification_density_rms',
            'canonical_terminal_mode'):
        assert mf.scf_summary[name] == getattr(result, name)
    assert (mf.scf_summary['canonical_continuation_refinements'] ==
            result.canonical_continuation_refinements)
    assert (mf.canonical_continuation_refinements_gc ==
            result.canonical_continuation_refinements)
    assert mf.scf_summary['fock_evaluations_total'] == result.nfev


def test_failed_canonical_verification_resumes_fixed_n_continuation(
        monkeypatch):
    mf, solver = _solver(mu=-0.1, canonical_continuation=True)
    original_evaluate = solver.evaluate
    verification_calls = 0

    def fail_first_verification(h_orth):
        nonlocal verification_calls
        verification_calls += 1
        state = original_evaluate(h_orth)
        if verification_calls == 1:
            return replace(
                state,
                residual_rms=(
                    10.0 * solver.config.
                    canonical_continuation_verification_residual_tol))
        return state

    monkeypatch.setattr(solver, 'evaluate', fail_first_verification)
    original_start = solver._start_canonical_session
    sessions = []

    def tracked_start(*args, **kwargs):
        point, session = original_start(*args, **kwargs)
        sessions.append(session)
        return point, session

    monkeypatch.setattr(solver, '_start_canonical_session', tracked_start)

    def unexpected_iterative_fixed_mu(*args, **kwargs):
        raise AssertionError(
            'failed verification entered an iterative fixed-mu solve')

    monkeypatch.setattr(solver, '_kernel_nlcg', unexpected_iterative_fixed_mu)

    # Starting at the physical Fock makes the first fixed-N root exact.  The
    # injected verification failure must therefore trigger another fixed-N
    # refinement/verification, rather than being hidden by root-search error.
    result = solver._kernel_canonical_continuation(h0=solver.hcore_ao)

    assert result.converged, result.message
    assert verification_calls == 2
    assert result.canonical_verification_attempts == 2
    assert result.canonical_verification_evaluations == 2
    assert result.canonical_verification_failures == 1
    assert result.canonical_terminal_mode == 'canonical-verification'
    assert result.canonical_continuation_steps == 1
    assert result.canonical_continuation_refinements == 1
    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]
    assert result.nfev == mf.veff_calls
    assert result.nfev == (
        result.canonical_continuation_evaluations +
        result.canonical_verification_evaluations)


def test_failed_fixed_n_retry_starts_fresh(monkeypatch):
    _, solver = _solver(mu=-0.1, canonical_continuation=True)
    original_start = solver._start_canonical_session
    starts = []

    def fail_second_start(*args, **kwargs):
        point, session = original_start(*args, **kwargs)
        starts.append(session)
        if len(starts) == 2:
            point = replace(
                point, converged=False,
                message='injected fixed-N failure')
        return point, session

    monkeypatch.setattr(
        solver, '_start_canonical_session', fail_second_start)
    h0 = [cp.asarray([
        [-0.25, 0.2 + 0.05j],
        [0.2 - 0.05j, 0.1]], dtype=cp.complex128)]

    result = solver._kernel_canonical_continuation(h0=h0)

    assert result.converged, result.message
    assert len(starts) >= 3
    # Retrying the deliberately failed point creates a fresh child/context.
    assert starts[2] is not starts[1]


def test_fock_evaluation_count_includes_fresh_initial_guess_build():
    mf, solver = _solver(canonical_continuation=True)
    result = solver.kernel()
    assert result.converged, result.message
    assert result.nfev == mf.veff_calls
    assert result.nfev == (
        1 + result.canonical_continuation_evaluations + 1)


def test_initial_auxiliary_electron_number_selects_requested_basin():
    target = 1.25
    _, solver = _solver(initial_electron_number=target)
    h = solver._initial_h()
    state = solver.evaluate(h)
    assert abs(state.electron_number - target) < 1.0e-12

    unshifted = solver._initial_h_from_dm(solver.mf.get_init_guess(solver.mf.cell))
    differences = [shifted - original for shifted, original in zip(h, unshifted)]
    for difference, identity in zip(differences, solver.identity):
        scalar = cp.trace(difference).real / difference.shape[0]
        assert float(cp.max(cp.abs(difference - scalar * identity)).item()) < 1.0e-13


def test_low_temperature_restart_blends_residual_and_exact_gradient():
    mf = _FixedFockKRKS([cp.diag(cp.asarray([10.0, 0.01]))])
    solver = GrandCanonicalKRKS(
        mf, mu=0.0, sigma=0.001,
        config=GrandCanonicalConfig(check_time_reversal=False),
    )
    state = solver.evaluate([cp.diag(cp.asarray([-1.0, 0.0]))])
    cosine = -solver.inner(state.gradient, state.residual)
    cosine /= solver.norm(state.gradient) * solver.norm(state.residual)
    assert 0.0 < cosine < 0.05

    direction, reason = solver._restart_direction(state)
    assert reason == 'restarted with blended residual/exact gradient'
    new_cosine = -solver.inner(state.gradient, direction)
    new_cosine /= solver.norm(state.gradient) * solver.norm(direction)
    assert new_cosine > cosine
    # The exact gradient is saturated in the first orbital, so its residual
    # component must be retained by the blend rather than discarded.
    difference = direction[0][0, 0] - state.residual[0][0, 0]
    assert float(cp.abs(difference).item()) < 1.0e-13


def test_near_stationary_low_temperature_state_uses_exact_gradient():
    mf = _FixedFockKRKS([cp.diag(cp.asarray([-1.0, 1.0e-5]))])
    solver = GrandCanonicalKRKS(
        mf, mu=0.0, sigma=0.001,
        config=GrandCanonicalConfig(check_time_reversal=False),
    )
    state = solver.evaluate([cp.diag(cp.asarray([-1.0, 0.0]))])
    assert state.residual_rms < solver.config.exact_gradient_polish_residual_rms

    direction, restarted, reason = solver._ensure_descent(state, state.residual)
    assert restarted
    assert reason == 'exact-gradient final polishing'
    expected = solver.scale_blocks(-1.0, state.gradient)
    assert float(cp.max(cp.abs(direction[0] - expected[0])).item()) < 1.0e-13


def test_exact_gradient_nlcg_safeguards_can_be_disabled_independently():
    mf = _FixedFockKRKS([cp.diag(cp.asarray([10.0, 0.01]))])
    solver = GrandCanonicalKRKS(
        mf, mu=0.0, sigma=0.001,
        config=GrandCanonicalConfig(
            check_time_reversal=False,
            nlcg_exact_gradient_blend=False,
            nlcg_exact_gradient_polish=False),
    )
    state = solver.evaluate([cp.diag(cp.asarray([-1.0, 0.0]))])

    restarted, reason = solver._restart_direction(state)
    ensured, changed, ensure_reason = solver._ensure_descent(
        state, state.residual)

    assert reason == 'restarted with preconditioned residual'
    assert not changed
    assert ensure_reason == ''
    assert float(cp.max(cp.abs(
        restarted[0] - state.residual[0])).item()) < 1.0e-13
    assert float(cp.max(cp.abs(
        ensured[0] - state.residual[0])).item()) < 1.0e-13


def test_restart_and_chemical_potential_sign(tmp_path):
    checkpoint = str(tmp_path / 'gc.npz')
    _, solver = _solver(checkpoint_path=checkpoint)
    h0 = [cp.asarray([[-0.15, 0.11j], [-0.11j, 0.45]])]
    first = solver.kernel(h0=h0)
    assert first.converged
    assert (tmp_path / 'gc.npz').exists()
    _, resumed = _solver(checkpoint_path=checkpoint)
    second = resumed.kernel()
    assert second.converged
    assert abs(second.grand_potential - first.grand_potential) < 1.0e-10

    _, low_mu = _solver(mu=-0.25)
    _, high_mu = _solver(mu=0.05)
    low = low_mu.kernel(h0=h0)
    high = high_mu.kernel(h0=h0)
    assert high.electron_number > low.electron_number


def _small_periodic_cell():
    cell = gto.Cell()
    cell.a = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
    cell.atom = 'He 0 0 0'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [15, 15, 15]
    cell.verbose = 0
    return cell.build()


def test_real_multik_krks_evaluator_and_finalisation():
    cell = _small_periodic_cell()
    mf = cell.KRKS(kpts=cell.make_kpts([3, 1, 1])).to_gpu()
    mf.xc = 'LDA,VWN'
    config = GrandCanonicalConfig(
        max_cycle=25, required_consecutive_conv=1, conv_tol_omega=1.0e-7,
        conv_tol_grad_rms=1.0e-6, conv_tol_residual_rms=1.0e-5,
        conv_tol_density_rms=1.0e-7, conv_tol_nelec=1.0e-7,
        line_search_max_evals=10, line_search_zoom_evals=10,
    )
    solver = GrandCanonicalKRKS(mf, mu=-0.4, sigma=0.08, config=config)
    result = solver.kernel()
    assert result.converged, result.message
    assert solver._time_reversal_enabled
    assert all(b.grand_potential <= a.grand_potential + 1.0e-10
               for a, b in zip(result.history, result.history[1:]))
    reconstructed = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    assert float(cp.max(cp.abs(reconstructed - result.dm_ao)).item()) < 1.0e-9
    assert abs(mf.e_tot - result.dft_total_energy) < 1.0e-12
    assert abs(mf.grand_potential - result.grand_potential) < 1.0e-12


def test_real_multik_fixed_electron_number_minimisation():
    target = 1.6
    cell = _small_periodic_cell()
    mf = cell.KRKS(kpts=cell.make_kpts([3, 1, 1])).to_gpu()
    mf.xc = 'LDA,VWN'
    config = GrandCanonicalConfig(
        max_cycle=25, required_consecutive_conv=1, conv_tol_omega=1.0e-7,
        conv_tol_grad_rms=1.0e-6, conv_tol_residual_rms=1.0e-5,
        conv_tol_density_rms=1.0e-7, conv_tol_nelec=1.0e-7,
        line_search_max_evals=10, line_search_zoom_evals=10,
    )
    solver = GrandCanonicalKRKS(
        mf, sigma=0.08, config=config, electron_number=target)
    result = solver.kernel()
    assert result.converged, result.message
    assert result.fixed_electron_number
    assert abs(result.electron_number - target) < config.mu_electron_number_tol
    assert np.isfinite(result.mu)
    assert all(b.free_energy <= a.free_energy + 1.0e-10
               for a, b in zip(result.history, result.history[1:]))
    reconstructed = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    assert float(cp.max(cp.abs(reconstructed - result.dm_ao)).item()) < 1.0e-9
    assert abs(mf.free_energy - result.free_energy) < 1.0e-12


def test_real_gga_evaluator_is_supported():
    cell = _small_periodic_cell()
    mf = cell.KRKS(kpts=cell.make_kpts([1, 1, 1])).to_gpu()
    mf.xc = 'PBE'
    solver = GrandCanonicalKRKS(mf, mu=-0.4, sigma=0.08)
    state = solver.evaluate(solver._initial_h())
    assert np.isfinite(state.grand_potential)
    assert state.grad_rms >= 0.0
