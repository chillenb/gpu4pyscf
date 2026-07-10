import cupy as cp
import numpy as np
from pyscf.pbc import gto

from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.dft.grand_canonical import (
    GrandCanonicalConfig, GrandCanonicalKRKS, fermi_divided_difference,
    fermi_entropy, fermi_occupations,
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

    def get_ovlp(self, cell, kpts):
        return cp.stack([cp.eye(f.shape[0], dtype=f.dtype) for f in self._fock])

    def get_hcore(self, cell, kpts):
        return self._fock

    def check_linear_dependency(self, overlap, **kwargs):
        return cp.stack([cp.eye(s.shape[0], dtype=s.dtype) for s in overlap])

    def get_init_guess(self, cell, kpts=None):
        return cp.stack([cp.eye(f.shape[0], dtype=f.dtype) for f in self._fock])

    def get_veff(self, cell, dm, **kwargs):
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


def _solver(mu=-0.1, checkpoint_path=None, initial_electron_number=None):
    f0 = cp.asarray([[[-0.7, 0.12j], [-0.12j, 0.3]]], dtype=cp.complex128)
    mf = _FixedFockKRKS(f0)
    config = GrandCanonicalConfig(
        max_cycle=50, required_consecutive_conv=1,
        conv_tol_omega=1.0e-10, conv_tol_grad_rms=1.0e-8,
        conv_tol_residual_rms=1.0e-7, conv_tol_density_rms=1.0e-9,
        conv_tol_nelec=1.0e-9, check_time_reversal=False,
        checkpoint_path=checkpoint_path,
        initial_electron_number=initial_electron_number,
    )
    return mf, GrandCanonicalKRKS(mf, mu=mu, sigma=0.15, config=config)


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


def test_low_temperature_restart_keeps_preconditioned_residual():
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
    assert reason == 'restarted with preconditioned residual'
    assert float(cp.max(cp.abs(direction[0] - state.residual[0])).item()) < 1.0e-13


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


def test_real_gga_evaluator_is_supported():
    cell = _small_periodic_cell()
    mf = cell.KRKS(kpts=cell.make_kpts([1, 1, 1])).to_gpu()
    mf.xc = 'PBE'
    solver = GrandCanonicalKRKS(mf, mu=-0.4, sigma=0.08)
    state = solver.evaluate(solver._initial_h())
    assert np.isfinite(state.grand_potential)
    assert state.grad_rms >= 0.0
