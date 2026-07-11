import cupy as cp
import numpy as np
import pytest
from dataclasses import replace
from pyscf.pbc import gto

from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.dft.grand_canonical import (
    GrandCanonicalConfig, GrandCanonicalKRKS, _DIISItem, _LBFGSPair,
    _LineSearchResult, fermi_divided_difference, fermi_entropy,
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
            line_search_nelec_guard_max_delta_nelec=5.0e-2):
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

    with pytest.raises(ValueError, match='trust ratios'):
        config = GrandCanonicalConfig(
            diis_trust_shrink_ratio=0.8,
            diis_trust_expand_ratio=0.5)
        GrandCanonicalKRKS(_FixedFockKRKS([
            cp.eye(2, dtype=cp.complex128)]), mu=-0.1, sigma=0.15,
            config=config)


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
