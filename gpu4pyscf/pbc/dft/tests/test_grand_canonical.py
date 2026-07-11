import cupy as cp
import numpy as np
import pytest
from dataclasses import replace
from pyscf.pbc import gto

from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.dft.grand_canonical import (
    GrandCanonicalConfig, GrandCanonicalKRKS, _DIISItem, _LBFGSPair,
    _LineSearchResult, _TrialInfo, fermi_divided_difference, fermi_entropy,
    fermi_occupations,
)


class _MockCell:
    precision = 1.0e-12

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
            electron_number=None, optimizer='nlcg',
            lbfgs_initial_metric='fermi', lbfgs_history_size=5,
            lbfgs_line_search_c2=0.9, diis_switch_residual_rms=None,
            diis_max_objective_increase=1.0e-5,
            diis_max_delta_nelec=5.0e-2,
            line_search_nelec_guard_residual_rms=1.0e-2,
            line_search_max_delta_nelec=1.0,
            line_search_nelec_guard_max_delta_nelec=5.0e-2,
            line_search_nelec_guard_mode='reject',
            line_search_nelec_trust_initial=2.5e-1,
            line_search_nelec_trust_min=1.0e-3,
            line_search_nelec_trust_shrink=5.0e-1,
            line_search_nelec_trust_expand=2.0,
            canonical_continuation=False):
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
        optimizer=optimizer,
        lbfgs_initial_metric=lbfgs_initial_metric,
        lbfgs_history_size=lbfgs_history_size,
        lbfgs_line_search_c2=lbfgs_line_search_c2,
        diis_switch_residual_rms=diis_switch_residual_rms,
        diis_max_objective_increase=diis_max_objective_increase,
        diis_max_delta_nelec=diis_max_delta_nelec,
        line_search_nelec_guard_residual_rms=(
            line_search_nelec_guard_residual_rms),
        line_search_max_delta_nelec=line_search_max_delta_nelec,
        line_search_nelec_guard_max_delta_nelec=(
            line_search_nelec_guard_max_delta_nelec),
        line_search_nelec_guard_mode=line_search_nelec_guard_mode,
        line_search_nelec_trust_initial=line_search_nelec_trust_initial,
        line_search_nelec_trust_min=line_search_nelec_trust_min,
        line_search_nelec_trust_shrink=line_search_nelec_trust_shrink,
        line_search_nelec_trust_expand=line_search_nelec_trust_expand,
        canonical_continuation=canonical_continuation,
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
    ('reject', 'reject'),
    ('scalar', 'scalar-shift'),
    ('scalar_shift', 'scalar-shift'),
    ('scalar shift', 'scalar-shift'),
    ('fermi', 'fermi-response'),
    ('response', 'fermi-response'),
    ('fermi_response', 'fermi-response'),
])
def test_electron_number_guard_mode_aliases_and_default(alias, expected):
    assert GrandCanonicalConfig().line_search_nelec_guard_mode == 'reject'
    _, solver = _solver(line_search_nelec_guard_mode=alias)
    assert solver.config.line_search_nelec_guard_mode == expected


def test_electron_number_guard_mode_validation():
    with pytest.raises(TypeError, match='must be a string'):
        _solver(line_search_nelec_guard_mode=None)
    with pytest.raises(ValueError, match='reject, scalar-shift'):
        _solver(line_search_nelec_guard_mode='occupation-mixing')


def test_scalar_shift_projection_hits_charge_boundary_and_is_identity():
    _, solver = _solver(line_search_nelec_guard_mode='scalar-shift')
    mu_before = solver.mu
    candidate = [cp.asarray([
        [-0.25, 0.07 + 0.11j],
        [0.07 - 0.11j, 0.35],
    ], dtype=cp.complex128)]
    candidate = solver._sanitize_h(candidate)
    raw_nelec = solver._cheap_fixed_mu_electron_number(candidate)
    target_nelec = raw_nelec - 5.0e-2

    projected, parameter, fallback = (
        solver._project_trial_electron_number(candidate, target_nelec))

    assert not fallback
    assert solver.mu == mu_before
    assert abs(solver._cheap_fixed_mu_electron_number(projected) -
               target_nelec) <= 1.0e-10
    correction = projected[0] - candidate[0]
    expected = parameter * cp.eye(2, dtype=cp.complex128)
    assert float(cp.max(cp.abs(correction - expected)).item()) < 1.0e-12
    assert abs(complex(correction[0, 1].item())) < 1.0e-14
    assert abs(complex(projected[0][0, 1].item()) -
               complex(candidate[0][0, 1].item())) < 1.0e-14
    assert float(cp.max(cp.abs(
        projected[0] - projected[0].conj().T)).item()) < 1.0e-14


def test_fermi_response_projection_hits_boundary_and_targets_near_mu():
    levels = cp.asarray([-2.0, -0.05, 0.10, 2.0], dtype=cp.float64)
    fock = cp.diag(levels).astype(cp.complex128)
    mf = _FixedFockKRKS([fock])
    config = GrandCanonicalConfig(
        check_time_reversal=False,
        line_search_nelec_guard_mode='fermi-response')
    solver = GrandCanonicalKRKS(
        mf, mu=0.0, sigma=0.1, config=config)
    candidate = [fock.copy()]
    raw_nelec = solver._cheap_fixed_mu_electron_number(candidate)
    target_nelec = raw_nelec - 2.0e-2

    projected, _, fallback = solver._project_trial_electron_number(
        candidate, target_nelec)
    scalar, _ = solver._scalar_shift_to_nelec(candidate, target_nelec)

    assert not fallback
    assert abs(solver._cheap_fixed_mu_electron_number(projected) -
               target_nelec) <= 1.0e-10
    response_correction = cp.diag(projected[0] - candidate[0]).real
    scalar_correction = cp.diag(scalar[0] - candidate[0]).real
    near_mu = float(cp.max(cp.abs(response_correction[1:3])).item())
    far_from_mu = float(cp.max(cp.abs(
        response_correction[[0, 3]])).item())
    assert near_mu > 1.0e4 * max(far_from_mu, 1.0e-16)
    assert float(cp.max(cp.abs(
        response_correction - scalar_correction)).item()) > 1.0e-5
    assert float(cp.max(cp.abs(
        scalar_correction - scalar_correction[0])).item()) < 1.0e-12


def test_fermi_response_singular_response_falls_back_to_scalar_shift():
    levels = cp.asarray([-10.0, 10.0], dtype=cp.float64)
    fock = cp.diag(levels).astype(cp.complex128)
    mf = _FixedFockKRKS([fock])
    config = GrandCanonicalConfig(
        check_time_reversal=False,
        line_search_nelec_guard_mode='fermi-response')
    solver = GrandCanonicalKRKS(
        mf, mu=0.0, sigma=1.0e-3, config=config)
    candidate = [fock.copy()]
    target_nelec = (
        solver._cheap_fixed_mu_electron_number(candidate) - 1.0e-2)

    projected, parameter, fallback = (
        solver._project_trial_electron_number(candidate, target_nelec))

    assert fallback
    assert abs(solver._cheap_fixed_mu_electron_number(projected) -
               target_nelec) <= 1.0e-10
    correction = projected[0] - candidate[0]
    assert float(cp.max(cp.abs(
        correction - parameter * cp.eye(2))).item()) < 1.0e-12


def test_fermi_response_insufficient_charge_capacity_falls_back():
    # The frontier level has a well-resolved f(1-f), but changing that level
    # alone can remove only one electron.  Reaching this target also requires
    # moving the otherwise saturated occupied level.
    levels = cp.asarray([-10.0, 0.0, 10.0], dtype=cp.float64)
    fock = cp.diag(levels).astype(cp.complex128)
    mf = _FixedFockKRKS([fock])
    config = GrandCanonicalConfig(
        check_time_reversal=False,
        line_search_nelec_guard_mode='fermi-response')
    solver = GrandCanonicalKRKS(
        mf, mu=0.0, sigma=1.0e-2, config=config)
    candidate = [fock.copy()]
    occupations = fermi_occupations(
        solver.beta * (levels - solver.mu))
    response_max = float(cp.max(
        occupations * (1.0 - occupations)).item())
    assert response_max > 1.0e-14
    target_nelec = 1.5

    projected, parameter, fallback = (
        solver._project_trial_electron_number(candidate, target_nelec))

    assert fallback
    assert abs(solver._cheap_fixed_mu_electron_number(projected) -
               target_nelec) <= 1.0e-10
    correction = projected[0] - candidate[0]
    assert float(cp.max(cp.abs(
        correction - parameter * cp.eye(3))).item()) < 1.0e-12


@pytest.mark.parametrize('mode', ['scalar-shift', 'fermi-response'])
def test_projection_preserves_multik_time_reversal(mode):
    block = cp.asarray([
        [-0.3, 0.05 + 0.09j],
        [0.05 - 0.09j, 0.2],
    ], dtype=cp.complex128)
    mf = _FixedFockKRKS([block, block.conj()])
    mf.kpts = np.asarray([[0.25, 0.0, 0.0], [-0.25, 0.0, 0.0]])
    config = GrandCanonicalConfig(
        check_time_reversal=True, enforce_time_reversal=True,
        line_search_nelec_guard_mode=mode)
    solver = GrandCanonicalKRKS(
        mf, mu=-0.1, sigma=0.15, config=config)
    candidate = [block.copy(), block.conj()]
    target_nelec = (
        solver._cheap_fixed_mu_electron_number(candidate) - 2.0e-2)

    projected, _, _ = solver._project_trial_electron_number(
        candidate, target_nelec)

    assert solver._time_reversal_enabled
    assert abs(solver._cheap_fixed_mu_electron_number(projected) -
               target_nelec) <= 1.0e-10
    assert float(cp.max(cp.abs(
        projected[0] - projected[1].conj())).item()) < 1.0e-12
    assert all(float(cp.max(cp.abs(
        value - value.conj().T)).item()) < 1.0e-12
               for value in projected)


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


@pytest.mark.parametrize('mode', ['scalar-shift', 'fermi-response'])
def test_electron_number_projection_is_bypassed_at_fixed_n(mode):
    mf, solver = _solver(
        electron_number=1.25,
        line_search_nelec_guard_mode=mode,
        line_search_nelec_guard_residual_rms=None,
        line_search_nelec_trust_initial=1.0e-3)
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    nfev_before = solver.nfev
    trial = solver._trial(
        state, [cp.eye(2, dtype=cp.complex128)], 1.0)

    assert trial is not None
    assert solver.nfev == nfev_before + 1
    assert mf.veff_calls == nfev_before + 1
    assert not solver._last_trial_info.projected
    assert solver.nnelec_projection_attempts == 0
    assert solver.nnelec_projection_fallbacks == 0


def test_scalar_projection_trial_caps_delta_nelec_and_tracks_attempt():
    _, solver = _solver(
        line_search_nelec_guard_mode='scalar-shift',
        line_search_nelec_guard_residual_rms=None,
        line_search_nelec_trust_initial=1.0e-2)
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    identity = [cp.eye(2, dtype=cp.complex128)]
    identity_slope = solver.inner(state.gradient, identity)
    direction = solver.scale_blocks(
        -np.copysign(1.0, identity_slope), identity)

    trial = solver._trial(state, direction, 0.2)
    info = solver._last_trial_info

    assert trial is not None
    assert info.projected
    assert info.mode == 'scalar-shift'
    assert abs(info.raw_delta_nelec) > 1.0e-2
    assert abs(abs(info.projected_delta_nelec) - 1.0e-2) <= 1.0e-10
    assert abs(trial.electron_number - state.electron_number -
               info.projected_delta_nelec) < 1.0e-12
    assert info.actual_slope < 0.0
    assert solver.nnelec_projection_attempts == 1
    assert solver.ncheap_nelec_reject == 0
    assert solver.max_raw_delta_nelec == pytest.approx(
        abs(info.raw_delta_nelec))
    assert solver.max_nelec_projection_correction > 0.0


def test_adaptive_electron_number_trust_radius_shrinks_and_expands():
    _, solver = _solver(
        line_search_nelec_guard_mode='scalar-shift',
        line_search_max_delta_nelec=0.6,
        line_search_nelec_trust_initial=0.25,
        line_search_nelec_trust_min=0.01)
    old = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    info = _TrialInfo(
        actual_slope=-1.0, projected=True, trust_radius=0.05)
    poor = replace(old, objective=old.objective - 0.1)
    ratio = solver._accept_projected_trial(old, poor, info)
    assert ratio == pytest.approx(0.1)
    # Shrink from the effective boundary used by the trial, not from the
    # larger latent adaptive radius.
    assert solver._nelec_trust_radius == pytest.approx(0.025)

    solver._nelec_trust_radius = 0.4
    good = replace(old, objective=old.objective - 0.9)
    good_info = replace(info, trust_radius=0.3)
    ratio = solver._accept_projected_trial(old, good, good_info)
    assert ratio == pytest.approx(0.9)
    # Expansion is clamped by the global hard electron-number cap.
    assert solver._nelec_trust_radius == pytest.approx(0.6)

    solver._nelec_trust_radius = 0.012
    min_info = replace(info, trust_radius=0.012)
    solver._accept_projected_trial(old, poor, min_info)
    assert solver._nelec_trust_radius == pytest.approx(0.01)


def test_fixed_trust_radius_factors_leave_radius_unchanged():
    _, solver = _solver(
        line_search_nelec_guard_mode='fermi-response',
        line_search_nelec_trust_initial=0.25,
        line_search_nelec_trust_shrink=1.0,
        line_search_nelec_trust_expand=1.0)
    old = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    info = _TrialInfo(
        actual_slope=-1.0, projected=True, trust_radius=0.05)

    poor = replace(old, objective=old.objective - 0.1)
    solver._accept_projected_trial(old, poor, info)
    assert solver._nelec_trust_radius == pytest.approx(0.25)

    good = replace(old, objective=old.objective - 0.9)
    solver._accept_projected_trial(old, good, info)
    assert solver._nelec_trust_radius == pytest.approx(0.25)


def test_diis_trial_explicitly_disables_occupation_projection(monkeypatch):
    _, solver = _solver(
        line_search_nelec_guard_mode='fermi-response',
        line_search_nelec_trust_initial=1.0e-3)
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    target = solver.axpy(0.2, state.residual, state.h_orth)
    projection_flags = []

    def failed_trial(unused_state, unused_direction, unused_damping,
                     allow_nelec_projection=True):
        projection_flags.append(allow_nelec_projection)
        return None

    monkeypatch.setattr(solver, '_trial', failed_trial)
    trial, damping, reason, rejected = solver._try_diis_target(
        state, target, max_backtracks=0)

    assert trial is None
    assert damping == 0.0
    assert rejected is None
    assert reason == 'DIIS trial evaluation failed'
    assert projection_flags == [False]


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

    def trial_at_boundary(unused_state, unused_direction, alpha):
        calls.append(alpha)
        if len(calls) == 1:
            solver._last_trial_rejected_by_nelec = False
            return replace(state, objective=state.objective - 1.0)
        solver._last_trial_rejected_by_nelec = True
        return None

    solver._trial = trial_at_boundary
    result = solver._zoom(
        state, direction, state.objective, dphi0,
        0.0, state, 1.0, None, None, 0)
    assert result.success
    assert result.alpha == pytest.approx(0.5)
    assert result.nfev == 2
    assert result.trust_boundary
    assert result.force_restart
    assert 'trust boundary' in result.message


def test_projected_line_search_uses_actual_step_and_forces_restart(
        monkeypatch):
    _, solver = _solver(
        optimizer='lbfgs', line_search_nelec_guard_mode='scalar-shift')
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    direction = solver.copy_blocks(state.residual)
    dphi0 = solver.inner(state.gradient, direction)
    assert dphi0 < 0.0
    evaluated = {}

    def projected_trial(unused_state, unused_direction, alpha):
        actual_step = solver.scale_blocks(0.1 * alpha, direction)
        actual_slope = solver.inner(state.gradient, actual_step)
        original_slope = alpha * dphi0
        # This objective passes Armijo for the projected displacement but
        # deliberately fails Armijo for the raw alpha * direction step.
        objective = (state.objective + solver.config.line_search_c1 *
                     0.5 * (actual_slope + original_slope))
        trial_h = solver._sanitize_h(
            solver.axpy(1.0, actual_step, state.h_orth))
        trial = replace(
            state, h_orth=trial_h, objective=objective,
            grand_potential=objective)
        solver.nnelec_projection_attempts += 1
        solver._last_trial_info = _TrialInfo(
            actual_step=solver.copy_blocks(actual_step), projected=True,
            mode='scalar-shift', raw_delta_nelec=0.7,
            projected_delta_nelec=0.25, parameter=0.12,
            trust_radius=0.25, actual_slope=actual_slope,
            correction_rms=0.34)
        evaluated.update(
            alpha=alpha, actual_slope=actual_slope,
            original_slope=original_slope, trial=trial)
        return trial

    monkeypatch.setattr(solver, '_trial', projected_trial)
    result = solver._line_search(state, direction)

    assert result.success
    assert result.nelec_projection_applied
    assert not result.strong_wolfe
    assert result.force_restart
    assert result.actual_step is not None
    assert result.nelec_projection_correction_rms == pytest.approx(0.34)
    assert (evaluated['trial'].objective <= state.objective +
            solver.config.line_search_c1 * evaluated['actual_slope'])
    assert (evaluated['trial'].objective > state.objective +
            solver.config.line_search_c1 * evaluated['original_slope'])
    solver._verify_accepted_step(
        state, result.state, direction, result, dphi0)
    assert solver.nnelec_projection_attempts == 1
    assert solver.nnelec_projection_acceptances == 1
    assert np.isfinite(result.nelec_trust_ratio)

    history = [object()]
    pair_info = solver._update_lbfgs_history(
        history, state, result.state, result)
    assert history == []
    assert not pair_info['pair_added']
    assert 'occupation projection' in pair_info['action']

    solver._record(
        0, state, result.state, result, dphi0, 0.0,
        'occupation projection')
    record = solver.history[-1]
    assert record.nelec_projection_applied
    assert record.nelec_projection_mode == 'scalar-shift'
    assert record.raw_delta_nelec == pytest.approx(0.7)
    assert record.projected_delta_nelec == pytest.approx(0.25)
    assert record.nelec_projection_parameter == pytest.approx(0.12)
    assert record.nelec_trust_radius == pytest.approx(0.25)
    assert record.nelec_trust_ratio == pytest.approx(
        result.nelec_trust_ratio)
    assert record.nelec_projection_correction_rms == pytest.approx(0.34)


def test_nlcg_projected_acceptance_restarts_with_zero_beta(monkeypatch):
    _, solver = _solver(line_search_nelec_guard_mode='scalar-shift')
    solver.config.max_cycle = 2
    h0 = [cp.asarray([[-0.3, 0.08 + 0.03j],
                      [0.08 - 0.03j, 0.2]])]
    calls = []

    def accepted_step(state, direction):
        cycle = len(calls)
        if cycle == 0:
            alpha = 0.05
            step = solver.scale_blocks(alpha, direction)
            projected = False
        else:
            alpha = 1.0
            step = solver.scale_blocks(0.03, direction)
            projected = True
        slope = solver.inner(state.gradient, step)
        assert slope < 0.0
        new_h = solver._sanitize_h(
            solver.axpy(1.0, step, state.h_orth))
        objective = state.objective + 0.5 * slope
        new_state = replace(
            state, h_orth=new_h, objective=objective,
            grand_potential=objective)
        calls.append(cycle)
        if not projected:
            return _LineSearchResult(
                True, new_state, alpha, 1, True, False, 'strong Wolfe')
        return _LineSearchResult(
            True, new_state, alpha, 1, False, True,
            'accepted occupation-projected Armijo point', True,
            actual_step=solver.copy_blocks(step),
            nelec_projection_applied=True,
            nelec_projection_mode='scalar-shift',
            raw_delta_nelec=0.5, projected_delta_nelec=0.25,
            nelec_projection_parameter=0.1,
            nelec_trust_radius=0.25, nelec_trust_ratio=0.8,
            nelec_projection_correction_rms=0.2)

    monkeypatch.setattr(solver, '_line_search', accepted_step)
    monkeypatch.setattr(
        solver, '_ensure_descent',
        lambda unused_state, direction: (
            solver.copy_blocks(direction), False, ''))
    result = solver._kernel_nlcg(h0=h0)

    assert calls == [0, 1]
    assert len(result.history) == 2
    assert not result.history[0].nelec_projection_applied
    projected_record = result.history[1]
    assert projected_record.nelec_projection_applied
    assert projected_record.cg_beta == 0.0
    assert 'occupation-projected' in projected_record.restart_reason


def test_projection_diagnostics_are_exported_in_result_and_mean_field():
    mf, solver = _solver(line_search_nelec_guard_mode='fermi-response')
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    solver.nnelec_projection_attempts = 4
    solver.nnelec_projection_acceptances = 3
    solver.nnelec_projection_fallbacks = 2
    solver.max_raw_delta_nelec = 1.7
    solver.max_projected_delta_nelec = 0.73
    solver.max_nelec_projection_correction = 0.42
    solver._nelec_trust_radius = 0.03125
    solver.last_nelec_trust_ratio = 0.81

    result = solver._finalize(
        state, converged=False, message='diagnostic snapshot',
        niter=0, density_change=0.0)

    assert result.nelec_projection_attempts == 4
    assert result.nelec_projection_acceptances == 3
    assert result.nelec_projection_fallbacks == 2
    assert result.max_raw_delta_nelec == pytest.approx(1.7)
    assert result.max_projected_delta_nelec == pytest.approx(0.73)
    assert result.max_nelec_projection_correction == pytest.approx(0.42)
    assert result.final_nelec_trust_radius == pytest.approx(0.03125)
    assert result.last_nelec_trust_ratio == pytest.approx(0.81)
    assert mf.nelec_projection_attempts_gc == 4
    assert mf.nelec_projection_acceptances_gc == 3
    assert mf.max_projected_delta_nelec_gc == pytest.approx(0.73)
    assert mf.scf_summary['nelec_projection_fallbacks_gc'] == 2
    assert mf.scf_summary['max_raw_delta_nelec_gc'] == pytest.approx(1.7)
    assert (mf.scf_summary['max_projected_delta_nelec_gc'] ==
            pytest.approx(0.73))
    assert (mf.scf_summary['max_nelec_projection_correction_gc'] ==
            pytest.approx(0.42))
    assert (mf.scf_summary['final_nelec_trust_radius_gc'] ==
            pytest.approx(0.03125))
    assert (mf.scf_summary['last_nelec_trust_ratio_gc'] ==
            pytest.approx(0.81))


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
        else:
            delta_z = solver.axpy(-1.0, old.z, new.z)
            numerator = solver.inner(new.gradient, delta_z)
            if canonical == 'polak-ribiere':
                denominator = solver.inner(old.gradient, old.z)
            else:
                delta_gradient = solver.axpy(-1.0, old.gradient, new.gradient)
                denominator = solver.inner(old_direction, delta_gradient)
        assert abs(beta - numerator / denominator) < 1.0e-13

    with pytest.raises(ValueError, match='unsupported cg_update'):
        _solver(cg_update='not-a-cg-update')


def test_all_cg_updates_converge_fixed_fock_problem():
    h0 = [cp.asarray([[-0.1, 0.19 - 0.11j], [0.19 + 0.11j, 0.6]])]
    for update in ('fletcher-reeves', 'polak-ribiere', 'hestenes-stiefel'):
        _, solver = _solver(cg_update=update)
        result = solver.kernel(h0=h0)
        assert result.converged, f'{update}: {result.message}'


def test_lbfgs_configuration_aliases_validation_and_fixed_n_metric():
    _, solver = _solver(optimizer='L-BFGS', lbfgs_initial_metric='fermi_response')
    assert solver.config.optimizer == 'lbfgs'
    assert solver.config.lbfgs_initial_metric == 'fermi'

    _, canonical = _solver(
        optimizer='limited memory bfgs', electron_number=1.2)
    assert canonical.config.optimizer == 'lbfgs'
    assert canonical.config.lbfgs_initial_metric == 'scalar'

    with pytest.raises(ValueError, match='unsupported optimizer'):
        _solver(optimizer='not-an-optimizer')
    with pytest.raises(ValueError, match='lbfgs_history_size'):
        _solver(optimizer='lbfgs', lbfgs_history_size=-1)


def test_residual_diis_configuration_and_pulay_coefficients():
    _, solver = _solver(diis_switch_residual_rms=1.0e-3)
    assert solver.config.diis_switch_residual_rms == 1.0e-3
    assert not solver.config.diis_preserve_accepted_history
    with pytest.raises(ValueError, match='may not be smaller'):
        _solver(diis_switch_residual_rms=1.0e-8)

    fock = [cp.zeros((2, 2), dtype=cp.complex128)]
    history = [
        _DIISItem(fock, fock, [cp.diag(cp.asarray([1.0, 0.0]))]),
        _DIISItem(fock, fock, [cp.diag(cp.asarray([0.0, 1.0]))]),
    ]
    coefficients, condition, coefficient_l1, action = (
        solver._diis_coefficients(history))
    assert action == ''
    assert condition == pytest.approx(1.0)
    assert coefficient_l1 == pytest.approx(1.0)
    assert np.allclose(coefficients, [0.5, 0.5])


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


def test_diis_backtracking_stops_when_local_secant_cannot_reduce_residual():
    _, solver = _solver(diis_switch_residual_rms=1.0e-3)
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    target = solver.axpy(
        0.1, [cp.eye(2, dtype=cp.complex128)], state.h_orth)
    calls = []

    def trial(current, direction, alpha, allow_nelec_projection=True):
        assert not allow_nelec_projection
        calls.append(alpha)
        factor = 1.01 if len(calls) == 1 else 1.005
        return replace(current, residual_rms=factor * current.residual_rms)

    solver._trial = trial
    accepted, _, reason, rejected = solver._try_diis_target(
        state, target, starting_damping=0.1, max_backtracks=8)
    assert accepted is None
    assert len(calls) == 2
    assert 'secant predicts no acceptable residual' in reason
    assert rejected is not None


def test_diis_secant_does_not_extrapolate_grossly_nonlinear_trials():
    _, solver = _solver(diis_switch_residual_rms=1.0e-3)
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    target = solver.axpy(
        0.1, [cp.eye(2, dtype=cp.complex128)], state.h_orth)
    factors = iter((3.0, 2.0, 0.99))
    calls = []

    def trial(current, direction, alpha, allow_nelec_projection=True):
        assert not allow_nelec_projection
        calls.append(alpha)
        return replace(
            current, residual_rms=next(factors) * current.residual_rms)

    solver._trial = trial
    accepted, damping, reason, _ = solver._try_diis_target(
        state, target, starting_damping=0.1, max_backtracks=8)
    assert accepted is not None
    assert damping == pytest.approx(0.025)
    assert reason == ''
    assert calls == pytest.approx([0.1, 0.05, 0.025])


def test_residual_diis_repairs_rejected_model_by_pruning_oldest_vector():
    _, solver = _solver(diis_switch_residual_rms=1.0e-3)
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    zero = [cp.zeros((2, 2), dtype=cp.complex128)]
    history = [
        _DIISItem(zero, zero, [cp.diag(cp.asarray([1.0, 0.0]))]),
        _DIISItem(zero, zero, [cp.diag(cp.asarray([0.0, 1.0]))]),
        _DIISItem(zero, zero, [cp.asarray([[0.0, 1.0j], [-1.0j, 0.0]])]),
    ]
    calls = []

    def try_model(unused_state, unused_target, starting_damping,
                  max_backtracks, **unused_kwargs):
        calls.append((starting_damping, max_backtracks))
        if len(calls) == 1:
            return None, 0.0, 'rejected test model', None
        return replace(
            state, residual_rms=0.5 * state.residual_rms), 0.25, '', None

    solver._try_diis_target = try_model
    step, _, _, action, _ = solver._diis_step(
        state, history, starting_damping=0.5)
    assert step.success
    assert step.alpha == pytest.approx(0.25)
    assert calls == [(0.5, 2), (0.5, 2)]
    assert len(history) == 2
    assert 'dropped oldest DIIS vector' in action


def test_residual_diis_uses_rejected_trial_as_trust_interpolation_point():
    _, solver = _solver(diis_switch_residual_rms=1.0e-3)
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    zero = [cp.zeros((2, 2), dtype=cp.complex128)]
    history = [
        _DIISItem(zero, zero, [cp.diag(cp.asarray([1.0, 0.0]))]),
        _DIISItem(zero, zero, [cp.diag(cp.asarray([0.0, 1.0]))]),
    ]
    rejected = replace(state, residual_rms=1.1 * state.residual_rms)
    accepted = replace(state, residual_rms=0.5 * state.residual_rms)
    calls = []

    def coefficients(items):
        size = len(items)
        return np.full(size, 1.0 / size), 1.0, 1.0, ''

    def try_model(unused_state, unused_target, starting_damping,
                  max_backtracks, **unused_kwargs):
        calls.append((starting_damping, max_backtracks))
        if len(calls) == 1:
            return None, 0.0, 'rejected test model', rejected
        return accepted, 0.25, '', None

    solver._diis_coefficients = coefficients
    solver._try_diis_target = try_model
    step, _, _, action, _ = solver._diis_step(
        state, history, starting_damping=0.5)
    assert step.success
    assert calls == [(0.5, 2), (0.5, 2)]
    assert len(history) == 3
    assert 'augmented DIIS model with rejected trust trial' in action


def test_diis_preserves_accepted_history_after_temporary_pruning():
    _, solver = _solver(diis_switch_residual_rms=1.0e-3)
    solver.config.diis_preserve_accepted_history = True
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    zero = [cp.zeros((2, 2), dtype=cp.complex128)]
    history = [
        _DIISItem(zero, [cp.eye(2)], [cp.diag(cp.asarray([1.0, 0.0]))]),
        _DIISItem(zero, [2.0 * cp.eye(2)],
                  [cp.diag(cp.asarray([0.0, 1.0]))]),
        _DIISItem(zero, [3.0 * cp.eye(2)],
                  [cp.asarray([[0.0, 1.0j], [-1.0j, 0.0]])]),
    ]
    accepted_items = list(history)
    accepted = replace(state, residual_rms=0.5 * state.residual_rms)

    def coefficients(items):
        del items[0]
        return np.full(2, 0.5), 1.0, 1.0, 'temporary test pruning'

    solver._diis_coefficients = coefficients
    solver._try_diis_target = lambda *args, **kwargs: (
        accepted, 0.25, '', None)
    step, _, _, action, _ = solver._diis_step(
        state, history, starting_damping=0.5)

    assert step.success
    assert len(history) == len(accepted_items)
    assert all(actual is expected
               for actual, expected in zip(history, accepted_items))
    assert 'restored accepted DIIS history' in action


def test_diis_preserved_history_uses_latest_fock_fallback():
    _, solver = _solver(diis_switch_residual_rms=1.0e-3)
    solver.config.diis_preserve_accepted_history = True
    solver.config.diis_model_max_backtracks = 2
    solver.config.diis_max_backtracks = 7
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    zero = [cp.zeros((2, 2), dtype=cp.complex128)]
    history = [
        _DIISItem(zero, [cp.eye(2)], [cp.diag(cp.asarray([1.0, 0.0]))]),
        _DIISItem(zero, [3.0 * cp.eye(2)],
                  [cp.diag(cp.asarray([0.0, 1.0]))]),
    ]
    accepted_items = list(history)
    rejected = replace(state, residual_rms=1.1 * state.residual_rms)
    accepted = replace(state, residual_rms=0.5 * state.residual_rms)
    calls = []

    solver._diis_coefficients = lambda items: (
        np.full(len(items), 1.0 / len(items)), 1.0, 1.0, '')

    def try_target(unused_state, target, starting_damping,
                   max_backtracks, **unused_kwargs):
        calls.append((solver.copy_blocks(target), starting_damping,
                      max_backtracks))
        if len(calls) == 1:
            return None, 0.0, 'rejected test model', rejected
        return accepted, 0.125, '', None

    solver._try_diis_target = try_target
    step, _, _, action, _ = solver._diis_step(
        state, history, starting_damping=0.5)

    assert step.success
    assert step.state is accepted
    assert step.alpha == pytest.approx(0.125)
    assert [call[2] for call in calls] == [2, 7]
    assert all(call[1] == pytest.approx(0.5) for call in calls)
    assert float(cp.max(cp.abs(
        calls[1][0][0] - accepted_items[-1].fock[0])).item()) < 1.0e-14
    assert all(actual is expected
               for actual, expected in zip(history, accepted_items))
    assert rejected not in history
    assert 'latest-Fock fallback' in action


def test_residual_diis_allows_one_bounded_nonmonotone_restoration():
    _, solver = _solver(diis_switch_residual_rms=1.0e-3)
    solver.config.diis_max_restoration_residual_increase = 0.05
    state = solver.evaluate([
        cp.asarray([[-0.3, 0.08 + 0.03j],
                    [0.08 - 0.03j, 0.2]])])
    restoration = replace(
        state, residual_rms=1.02 * state.residual_rms,
        objective=state.objective - 1.0e-4)
    acceptable, reason = solver._diis_trial_acceptable(
        state, restoration, allow_restoration=True,
        best_residual_rms=state.residual_rms)
    assert acceptable
    assert reason == ''

    uphill = replace(restoration, objective=state.objective + 1.0e-8)
    acceptable, reason = solver._diis_trial_acceptable(
        state, uphill, allow_restoration=True,
        best_residual_rms=state.residual_rms)
    assert not acceptable
    assert 'trust envelope' in reason

    target = state.residual_rms * (
        1.0 - solver.config.diis_min_residual_reduction)
    not_recovered = replace(
        restoration, residual_rms=0.9995 * state.residual_rms)
    acceptable, _ = solver._diis_trial_acceptable(
        restoration, not_recovered, residual_target_rms=target)
    assert not acceptable
    recovered = replace(
        restoration, residual_rms=0.998 * state.residual_rms)
    acceptable, _ = solver._diis_trial_acceptable(
        restoration, recovered, residual_target_rms=target)
    assert acceptable


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


def test_residual_diis_polishes_fixed_fock_for_both_direct_optimizers():
    h0 = [cp.asarray([[-0.1, 0.19 - 0.11j],
                      [0.19 + 0.11j, 0.6]])]
    for optimizer in ('nlcg', 'lbfgs'):
        _, solver = _solver(
            optimizer=optimizer, diis_switch_residual_rms=1.0,
            diis_max_delta_nelec=2.0)
        result = solver.kernel(h0=h0)
        assert result.converged, f'{optimizer}: {result.message}'
        assert result.message == 'converged residual-DIIS fixed point'
        assert result.niter == 1
        assert result.history[0].optimizer == 'diis'
        assert result.history[0].search_direction_source == 'residual-diis'
        assert result.history[0].diis_history_size == 1
        assert result.history[0].diis_damping == 1.0
        assert result.residual_rms < solver.config.conv_tol_residual_rms
        if optimizer == 'lbfgs':
            assert solver._lbfgs_history == []


def test_automatic_canonical_continuation_finds_fixed_mu_electron_number():
    mf, solver = _solver(
        mu=-0.1, optimizer='lbfgs', lbfgs_initial_metric='scalar',
        canonical_continuation=True)
    h0 = [cp.asarray([[0.25, 0.18 - 0.07j],
                      [0.18 + 0.07j, -0.15]])]
    expected_nelec = solver._electron_number_at_mu(solver.hcore_ao, solver.mu)
    result = solver.kernel(h0=h0)
    assert result.converged, result.message
    assert not result.fixed_electron_number
    assert result.canonical_continuation_steps >= 1
    assert result.canonical_continuation_evaluations >= 1
    assert np.isfinite(result.canonical_continuation_mu_error)
    assert abs(result.canonical_continuation_delta_nelec) <= (
        solver.config.canonical_continuation_handoff_delta_nelec)
    assert abs(result.electron_number - expected_nelec) < 1.0e-10
    assert result.nfev == mf.veff_calls
    assert result.nfev > result.canonical_continuation_evaluations
    assert solver.config.diis_max_coefficient_l1 == pytest.approx(10.0)


def test_unbracketed_canonical_continuation_uses_screened_secant():
    _, solver = _solver(canonical_continuation=True)
    inner_config = solver._canonical_continuation_config(1.0e-4, 0.125)
    assert inner_config.diis_max_coefficient_l1 == pytest.approx(50.0)
    proposal = solver._canonical_continuation_proposal(
        [(1.0, -0.04), (1.03, -0.03)], solver.hcore_ao,
        current_nelec=1.03, maximum_step=0.1)
    assert proposal == pytest.approx(1.12)


def test_canonical_continuation_default_handoff_charge_is_conservative():
    _, solver = _solver(canonical_continuation=True)
    assert (solver.config.canonical_continuation_handoff_delta_nelec ==
            pytest.approx(0.05))
    assert (solver.config.canonical_continuation_interpolation_refine_width ==
            pytest.approx(0.05))


def test_fock_evaluation_count_includes_fresh_initial_guess_build():
    mf, solver = _solver(canonical_continuation=True)
    result = solver.kernel()
    assert result.converged, result.message
    assert result.nfev == mf.veff_calls
    assert result.nfev == (
        1 + result.canonical_continuation_evaluations + 1)


def test_fermi_inverse_metric_maps_exact_gradient_to_z():
    _, solver = _solver(optimizer='lbfgs')
    solver.config.lbfgs_inverse_metric_cap = 1.0e6
    h = [cp.asarray([[-0.3, 0.08 + 0.03j],
                     [0.08 - 0.03j, 0.2]])]
    state = solver.evaluate(h)
    metric_gradient = solver._apply_fermi_inverse_metric(
        state, state.gradient)
    assert float(cp.max(cp.abs(metric_gradient[0] - state.z[0])).item()) < 1.0e-11


def test_lbfgs_two_loop_matches_dense_inverse_bfgs_complex_blocks():
    _, solver = _solver(
        optimizer='lbfgs', lbfgs_initial_metric='scalar')

    def block(vector):
        off_diagonal = (vector[2] + 1j * vector[3]) / np.sqrt(2.0)
        return [cp.asarray([[vector[0], off_diagonal],
                            [off_diagonal.conjugate(), vector[1]]])]

    hessian = np.asarray([
        [3.0, 0.2, 0.1, 0.0],
        [0.2, 2.0, 0.0, -0.1],
        [0.1, 0.0, 1.5, 0.15],
        [0.0, -0.1, 0.15, 1.0],
    ])
    steps = [
        np.asarray([0.2, -0.1, 0.15, 0.05]),
        np.asarray([-0.05, 0.12, 0.08, -0.09]),
    ]
    pairs = []
    for step in steps:
        gradient_change = hessian @ step
        sy = float(step @ gradient_change)
        pairs.append(_LBFGSPair(
            block(step), block(gradient_change), 1.0 / sy, sy,
            float(np.linalg.norm(step)),
            float(np.linalg.norm(gradient_change)),
            sy / (np.linalg.norm(step) * np.linalg.norm(gradient_change))))

    gradient = np.asarray([0.3, -0.25, 0.11, 0.07])
    base = solver.evaluate([cp.diag(cp.asarray([-0.2, 0.1]))])
    state = replace(base, gradient=block(gradient))
    direction, used_history, reason = solver._lbfgs_direction(state, pairs)
    assert used_history
    assert reason == ''

    gamma = pairs[-1].sy / float(
        cp.asnumpy(cp.vdot(pairs[-1].y[0], pairs[-1].y[0]).real))
    gamma = np.clip(
        gamma, solver.config.lbfgs_scalar_h0_min,
        solver.config.lbfgs_scalar_h0_max)
    inverse = gamma * np.eye(4)
    for step, pair in zip(steps, pairs):
        gradient_change = hessian @ step
        transform = np.eye(4) - pair.rho * np.outer(step, gradient_change)
        inverse = (transform @ inverse @ transform.T +
                   pair.rho * np.outer(step, step))
    expected = block(-inverse @ gradient)
    assert float(cp.max(cp.abs(direction[0] - expected[0])).item()) < 1.0e-12


def test_lbfgs_fixed_fock_step_and_exact_gradient_pair():
    _, solver = _solver(
        optimizer='lbfgs', lbfgs_line_search_c2=0.1)
    h0 = [cp.asarray([[-0.1, 0.19 - 0.11j],
                      [0.19 + 0.11j, 0.6]])]
    initial = solver.evaluate(h0)
    result = solver.kernel(h0=h0)
    assert result.converged, result.message
    assert result.history[0].optimizer == 'lbfgs'
    assert result.history[0].search_direction_source == 'residual'
    assert result.history[0].strong_wolfe
    assert abs(result.history[0].alpha - 2.0) < 1.0e-10
    assert result.history[0].lbfgs_pair_added
    assert result.history[0].lbfgs_sy > 0.0
    assert len(solver._lbfgs_history) == 1

    pair = solver._lbfgs_history[0]
    final_state = solver.evaluate(result.h_orth)
    expected_y = solver.axpy(-1.0, initial.gradient, final_state.gradient)
    assert float(cp.max(cp.abs(pair.y[0] - expected_y[0])).item()) < 1.0e-12
    assert float(cp.max(cp.abs(result.h_orth[0] - result.fock_orth[0])).item()) < 1.0e-12


def test_lbfgs_non_wolfe_and_bad_direction_reset():
    _, solver = _solver(optimizer='lbfgs')
    h0 = [cp.asarray([[-0.1, 0.19 - 0.11j],
                      [0.19 + 0.11j, 0.6]])]
    old = solver.evaluate(h0)
    new = solver.evaluate(solver.axpy(0.5, old.residual, old.h_orth))
    history = []
    wolfe = _LineSearchResult(
        True, new, 0.5, 1, True, False, 'strong Wolfe')
    info = solver._update_lbfgs_history(history, old, new, wolfe)
    assert info['pair_added']
    assert len(history) == 1

    non_wolfe = _LineSearchResult(
        True, new, 0.5, 1, False, True, 'best Armijo')
    info = solver._update_lbfgs_history(history, old, new, non_wolfe)
    assert not history
    assert not info['pair_added']
    assert 'cleared' in info['action']

    uphill = solver.copy_blocks(old.gradient)
    restart, reset, reason = solver._ensure_lbfgs_descent(
        old, uphill, used_history=True)
    assert reset
    assert 'rejected L-BFGS direction' in reason
    assert solver._is_descent(old, restart)


def test_lbfgs_history_evicts_oldest_pair():
    _, solver = _solver(optimizer='lbfgs', lbfgs_history_size=2)
    state = solver.evaluate([
        cp.asarray([[-0.1, 0.19 - 0.11j],
                    [0.19 + 0.11j, 0.6]])])
    states = [state]
    for _ in range(3):
        state = solver.evaluate(
            solver.axpy(0.2, state.residual, state.h_orth))
        states.append(state)
    history = []
    wolfe = _LineSearchResult(
        True, states[1], 0.2, 1, True, False, 'strong Wolfe')
    for old, new in zip(states, states[1:]):
        info = solver._update_lbfgs_history(history, old, new, wolfe)
        assert info['pair_added']
    assert len(history) == 2
    expected_oldest = solver.axpy(-1.0, states[1].h_orth, states[2].h_orth)
    assert float(cp.max(cp.abs(history[0].s[0] - expected_oldest[0])).item()) < 1.0e-13
    actual_bytes = sum(value.nbytes for pair in history for value in pair.s + pair.y)
    assert actual_bytes == solver._lbfgs_history_allocation_bytes


def test_fixed_electron_lbfgs_uses_scalar_metric_and_converges():
    target = 1.25
    _, solver = _solver(electron_number=target, optimizer='lbfgs')
    assert solver.config.lbfgs_initial_metric == 'scalar'
    h0 = [cp.asarray([[-0.1, 0.19 - 0.11j],
                      [0.19 + 0.11j, 0.6]])]
    result = solver.kernel(h0=h0)
    assert result.converged, result.message
    assert abs(result.electron_number - target) < solver.config.mu_electron_number_tol
    assert all(record.optimizer == 'lbfgs' for record in result.history)
    assert all(b.objective <= a.objective + 1.0e-11
               for a, b in zip(result.history, result.history[1:]))


def test_lbfgs_checkpoint_restart_starts_with_empty_history(tmp_path):
    checkpoint = str(tmp_path / 'lbfgs.npz')
    _, solver = _solver(
        optimizer='lbfgs', checkpoint_path=checkpoint,
        lbfgs_line_search_c2=0.1)
    h0 = [cp.asarray([[-0.1, 0.19 - 0.11j],
                      [0.19 + 0.11j, 0.6]])]
    first = solver.kernel(h0=h0)
    assert first.converged, first.message
    assert solver._lbfgs_history

    _, resumed = _solver(
        optimizer='lbfgs', checkpoint_path=checkpoint,
        lbfgs_line_search_c2=0.1)
    assert resumed._lbfgs_history == []
    checkpoint_h = resumed._load_checkpoint_h()
    state = resumed.evaluate(checkpoint_h)
    direction, used_history, reason = resumed._lbfgs_direction(
        state, resumed._lbfgs_history)
    assert not used_history
    assert reason == 'empty L-BFGS history'
    assert float(cp.max(cp.abs(direction[0] - state.residual[0])).item()) < 1.0e-13

    second = resumed.kernel()
    assert second.converged, second.message
    assert resumed._lbfgs_history == []
    assert abs(second.grand_potential - first.grand_potential) < 1.0e-12


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


def test_real_multik_lbfgs_preserves_time_reversal_and_converges():
    cell = _small_periodic_cell()
    mf = cell.KRKS(kpts=cell.make_kpts([3, 1, 1])).to_gpu()
    mf.xc = 'LDA,VWN'
    config = GrandCanonicalConfig(
        optimizer='lbfgs', max_cycle=25, required_consecutive_conv=1,
        conv_tol_omega=1.0e-7, conv_tol_grad_rms=1.0e-6,
        conv_tol_residual_rms=1.0e-5, conv_tol_density_rms=1.0e-7,
        conv_tol_nelec=1.0e-7, line_search_max_evals=10,
        line_search_zoom_evals=10,
    )
    solver = GrandCanonicalKRKS(mf, mu=-0.4, sigma=0.08, config=config)
    result = solver.kernel()
    assert result.converged, result.message
    assert solver._time_reversal_enabled
    assert solver._lbfgs_history
    assert all(record.optimizer == 'lbfgs' for record in result.history)
    assert all(b.objective <= a.objective + 1.0e-10
               for a, b in zip(result.history, result.history[1:]))
    for pair in solver._lbfgs_history:
        for blocks in (pair.s, pair.y):
            assert all(float(cp.max(cp.abs(value - value.conj().T)).item()) < 1.0e-12
                       for value in blocks)
            for i, j in solver._tr_pairs:
                assert float(cp.max(cp.abs(blocks[i] - blocks[j].conj())).item()) < 1.0e-12


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
