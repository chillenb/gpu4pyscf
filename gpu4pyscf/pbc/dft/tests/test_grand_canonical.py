import cupy as cp
import numpy as np
import pytest

from pyscf.pbc import gto

from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.dft import grand_canonical as gc
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
    solver.enforce_time_reversal = False
    return mf, solver


def test_fermi_functions_are_stable():
    gamma = cp.asarray([-1000., -50., 0., 50., 1000.])
    occ = gc._fermi_occ(gamma)
    entropy = gc._fermi_entropy(gamma, occ)
    assert bool(cp.all(cp.isfinite(occ)))
    assert bool(cp.all(cp.isfinite(entropy)))
    assert float(occ.min()) >= 0.
    assert float(occ.max()) <= 1.
    assert abs(float((occ[0]+occ[-1]).item())-1.) < 1e-14
    assert float(abs(entropy[0]).item()) == 0.
    assert float(abs(entropy[-1]).item()) == 0.


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
        conv_tol=1e-7, diis_space=4, diis_backtrack=.4)
    assert solver.conv_tol == 1e-7
    assert solver.diis_space == 4
    assert solver.diis_backtrack == .4


def test_build_caches_mean_field_setup():
    mf = _CountingSetupKRKS([_fock()])
    solver = GrandCanonicalKRKS(mf, mu=-.1, sigma=.1)
    solver.enforce_time_reversal = False
    solver.build()
    solver.build()
    assert mf.setup_calls == dict(
        build=1, overlap=1, hcore=1, orth=1, enuc=1)


def test_fixed_n_mu_uses_pyscf_smearing_convention(monkeypatch):
    fock = [cp.diag(cp.asarray([-.7, .3])) for unused in range(3)]
    solver = GrandCanonicalKRKS(
        _FixedFockKRKS(fock), sigma=.15, nelec=1.3)
    solver.enforce_time_reversal = False
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
    assert abs(solver._nelec_from_eig(energies, mu)-1.3) < 1e-10


def test_fixed_n_state_has_target_charge_and_gauge_free_residual():
    target = 1.3
    unused, solver = _solver(nelec=target)
    solver.build()
    h = [cp.asarray([[-.3, .08+.03j], [.08-.03j, .2]])]
    state = solver._evaluate(h, nelec=target)
    assert abs(state.nelec-target) < 1e-10
    assert abs(solver._trace_mean(state.residual)) < 1e-12
    assert abs(state.free_energy-(state.e_tot+state.entropy_energy)) < 1e-13
    assert np.isfinite(state.mu)


def test_fixed_mu_candidate_preserves_occupations_and_physical_mu():
    unused, solver = _solver(mu=-.16)
    solver.build()
    h = [cp.asarray([[-.3, .08+.03j], [.08-.03j, .2]])]
    state = solver._evaluate(h, nelec=1.3)
    candidate, unused_delta = solver._fixed_mu_candidate(state)
    for new, old, eye in zip(candidate, state.h, solver.identity):
        assert float(cp.max(cp.abs(
            new-solver.mu*eye-(old-state.aux_mu*eye))).item()) < 1e-12
    assert abs(solver._nelec_at_mu(candidate, solver.mu)-state.nelec) < 1e-10


def test_tagged_solvent_fock_and_energy_use_same_veff():
    hcore = [cp.asarray([[-.6, 0.], [0., .2]])]
    solvent = [cp.asarray([[.1, .03j], [-.03j, -.04]])]
    mf = _TaggedSolventKRKS(hcore, solvent)
    solver = GrandCanonicalKRKS(mf, sigma=.15, nelec=1.2)
    solver.enforce_time_reversal = False
    solver.build()
    state = solver._evaluate([cp.asarray([[-.2, .1j], [-.1j, .1]])],
                             nelec=1.2)
    expected = hcore[0]+solvent[0]
    assert float(cp.max(cp.abs(state.fock[0]-expected)).item()) < 1e-13
    assert mf.energy_veff is not None
    assert getattr(mf.energy_veff, 'v_solvent', None) is not None


def test_residual_diis_converges_complex_fixed_n_problem():
    hcore = [cp.asarray([[-.7, .08j], [-.08j, .3]], dtype=cp.complex128)]
    mf = _LinearFockKRKS(hcore, coupling=.15)
    solver = GrandCanonicalKRKS(mf, sigma=.15, nelec=1.3)
    solver.enforce_time_reversal = False
    solver.conv_tol = 1e-8
    solver.build()
    h0 = [cp.asarray([[-.2, .18-.07j], [.18+.07j, .5]])]
    session = solver._new_session(h0, 1.3)
    solver._advance_session(session, solver.conv_tol)
    assert session.converged, session.message
    assert session.state.residual_rms <= solver.conv_tol
    assert abs(session.state.nelec-1.3) < 1e-10
    assert session.cycles > 0


def test_same_n_refinement_preserves_diis_session():
    hcore = [cp.asarray([[-.7, .08j], [-.08j, .3]], dtype=cp.complex128)]
    solver = GrandCanonicalKRKS(
        _LinearFockKRKS(hcore, coupling=.15), sigma=.15, nelec=1.3)
    solver.enforce_time_reversal = False
    solver.build()
    h0 = [cp.asarray([[-.2, .18-.07j], [.18+.07j, .5]])]
    session = solver._new_session(h0, 1.3)
    adiis = session.diis
    solver._advance_session(session, 1e-3)
    coarse_cycles = session.cycles
    coarse_nfev = solver.nfev
    assert session.converged
    solver._advance_session(session, 1e-8)
    assert session.diis is adiis
    assert session.converged
    assert session.cycles >= coarse_cycles
    assert solver.nfev >= coarse_nfev
    assert session.state.residual_rms <= 1e-8


def test_secant_proposals_and_neutral_charge_cap():
    unused, solver = _solver(mu=-.1)
    solver.build()
    state = solver._evaluate([_fock()], nelec=1.3)
    samples = [(1., -.1, state, None), (2., .1, state, None)]
    proposal = solver._secant_proposal(samples, state)
    assert proposal == pytest.approx(1.8)

    first = [(1., -.1, state, None)]
    proposal = solver._secant_proposal(first, state)
    assert abs(proposal-1.) <= .03+1e-14


def test_public_fixed_n_kernel_publishes_standard_attributes():
    mf, solver = _solver(nelec=1.25)
    e_tot = solver.kernel()
    assert solver.converged, solver.message
    assert e_tot == solver.e_tot == mf.e_tot
    assert abs(solver.electron_number-1.25) < 1e-10
    assert solver.nfev == 2
    assert mf.mo_coeff is solver.mo_coeff


def test_public_fixed_mu_kernel_uses_fixed_n_root_and_verification():
    mf, solver = _solver(mu=-.16)
    solver.conv_tol = 1e-9
    e_tot = solver.kernel()
    assert solver.converged, solver.message
    assert e_tot == mf.e_tot
    assert abs(solver.mu+.16) < solver.conv_tol_mu
    assert solver.outer_cycles == 4
    assert solver.verification_attempts == 1
    assert solver.nfev == 1 + solver.outer_cycles + solver.verification_attempts


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
    assert nelec != pytest.approx(solver._nelec_at_mu(unused_h, solver.mu))


def test_fixed_mu_clips_an_initial_density_at_capacity():
    mf = _FixedFockKRKS([cp.asarray([[-.7]])])
    mf.get_init_guess = lambda cell, kpts=None: cp.asarray([[[2.]]])
    solver = GrandCanonicalKRKS(mf, mu=-.16, sigma=.15)
    solver.enforce_time_reversal = False
    solver.max_outer_cycle = 10
    solver.kernel()
    assert solver.converged, solver.message
    assert abs(solver.mu+.16) < solver.conv_tol_mu


def test_time_reversal_projection_handles_complex_kpoints():
    kpts = np.asarray([[0., 0., 0.], [.25, 0., 0.], [-.25, 0., 0.]])
    f0 = _fock()
    mf = _FixedFockKRKS([f0, f0, f0.conj()], kpts=kpts)
    solver = GrandCanonicalKRKS(mf, sigma=.15, nelec=1.3)
    solver.build()
    assert solver._time_reversal
    blocks = [f0.copy(), f0+.03j*cp.eye(2), f0.conj()]
    projected = solver._project_time_reversal(blocks)
    assert float(cp.max(cp.abs(projected[2]-projected[1].conj())).item()) == 0.


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
    assert float(cp.max(cp.abs(dm-solver._state.dm)).item()) < 1e-8


def test_real_multik_fixed_mu_krks():
    cell = _small_periodic_cell()
    mf = cell.KRKS(kpts=cell.make_kpts([3, 1, 1])).to_gpu()
    mf.xc = 'LDA,VWN'
    solver = GrandCanonicalKRKS(mf, mu=-.4, sigma=.08)
    solver.conv_tol = 1e-6
    solver.conv_tol_coarse = 1e-5
    solver.conv_tol_mu = 1e-5
    solver.conv_tol_nelec = 1e-5
    solver.max_cycle = 30
    solver.kernel()
    assert solver.converged, solver.message
    assert abs(solver.mu+.4) < solver.conv_tol_mu
    assert solver.residual_rms <= 1e-6
