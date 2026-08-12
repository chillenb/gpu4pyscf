from types import SimpleNamespace

import cupy as cp
import numpy as np
import pytest

from pyscf.pbc import gto

from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.dft.grand_canonical import GrandCanonicalKRKS
from gpu4pyscf.pbc.dft import grand_canonical_potential as potential


class _PotentialNumInt:
    def __init__(self, cell, target_g):
        self.cell = cell
        self.mesh = tuple(cell.mesh)
        self.target_g = cp.asarray(target_g, dtype=cp.complex128)
        self.pot_guess = cp.asarray([99.0])
        self.calls = 0
        self.libxc = None

    def local_potential_to_ao(self, vlocal_g, kpts=None, hermi=1):
        vlocal_g = cp.asarray(vlocal_g).reshape(-1)
        value = (
            vlocal_g[0].real / self.cell.vol
            + 0.25 * (vlocal_g[1] + vlocal_g[-1]).real
            / self.cell.vol)
        return cp.asarray([[[value]]], dtype=cp.float64)


class _PotentialKRKS:
    def __init__(self, cell, target_g):
        self.cell = cell
        self.kpts = np.zeros((1, 3))
        self._numint = _PotentialNumInt(cell, target_g)
        self.verbose = 0
        self.stdout = None
        self.max_memory = 0
        self.scf_summary = {}
        self.xc = 'lda,'
        self.smearing_method = None
        self.sigma = 0.0
        self.converged = False

    def istype(self, name):
        return name == 'KRKS'

    def build(self):
        return self

    def get_ovlp(self, cell, kpts):
        return cp.ones((1, 1, 1))

    def get_hcore(self, cell, kpts):
        return cp.zeros((1, 1, 1))

    def check_linear_dependency(self, overlap, **kwargs):
        return cp.ones((1, 1, 1))

    def get_init_guess(self, cell, kpts=None, key=None):
        return cp.ones((1, 1, 1))

    def get_veff(self, cell, dm, **kwargs):
        ni = self._numint
        ni.calls += 1
        candidate = cp.asarray([float(ni.calls)])
        ni.pot_guess = candidate.copy()
        matrix = ni.local_potential_to_ao(ni.target_g, kpts=self.kpts)
        grid = SimpleNamespace(
            vlocal_g=ni.target_g.copy(),
            cavity_r=cp.zeros(ni.mesh),
            lpbe_pot_guess=candidate.copy(),
        )
        return tag_array(matrix, ecoul=0.0, exc=0.0, lpbe_grid=grid)

    def energy_elec(self, dm, hcore, veff):
        one_body = cp.einsum('kij,kji->', hcore, dm).real
        interaction = 0.5 * cp.einsum('kij,kji->', veff, dm).real
        return one_body + interaction, interaction

    @staticmethod
    def energy_nuc():
        return 0.0


@pytest.fixture
def setup():
    cell = gto.M(
        a=np.diag([5.0, 5.0, 5.0]),
        atom='He 0 0 0', basis='gth-szv', pseudo='gth-pade',
        unit='bohr', mesh=[1, 1, 3], verbose=0)
    target_g = cp.asarray([10.0, 2.0, 2.0], dtype=cp.complex128)
    mf = _PotentialKRKS(cell, target_g)
    solver = GrandCanonicalKRKS(mf, sigma=0.1, nelec=1.0)
    solver.conv_tol = 1e-11
    solver.build()
    return mf, solver, target_g


def test_fixed_n_potential_evaluator_aligns_exact_constant_mode(setup):
    mf, solver, target_g = setup
    input_g = cp.asarray([7.0, -1.0, -1.0], dtype=cp.complex128)
    accepted_guess = mf._numint.pot_guess.copy()
    raw_h = mf._numint.local_potential_to_ao(input_g)[0, 0, 0].item()

    state = potential.evaluate_fixed_n_potential(solver, input_g, 1.0)
    delta = 3.0 / solver.cell.vol

    assert abs(state.delta_v0 - delta) < 1e-15
    assert state.v_in_g[0] == state.v_out_g[0] == target_g[0]
    assert state.residual_g[0] == 0.0
    assert abs(state.electronic.h[0][0, 0].item() - (raw_h + delta)) < 1e-14
    assert abs(
        state.electronic.residual[0][0, 0].item()
        - (state.electronic.fock[0][0, 0].item() - raw_h - delta)
    ) < 1e-14
    cp.testing.assert_array_equal(mf._numint.pot_guess, accepted_guess)
    assert not cp.array_equal(state.lpbe_pot_guess, accepted_guess)


def test_warm_start_commit_is_explicit_and_transactional(setup):
    mf, solver, unused_target = setup
    initial = cp.asarray([8.0, 0.0, 0.0], dtype=cp.complex128)
    accepted_guess = mf._numint.pot_guess.copy()
    first = potential.evaluate_fixed_n_potential(solver, initial, 1.0)
    cp.testing.assert_array_equal(mf._numint.pot_guess, accepted_guess)

    potential.commit_potential_cycle(solver, first)
    cp.testing.assert_array_equal(
        mf._numint.pot_guess, first.lpbe_pot_guess)
    committed = mf._numint.pot_guess.copy()
    potential.evaluate_fixed_n_potential(
        solver, first.v_in_g + first.residual_g, 1.0)
    cp.testing.assert_array_equal(mf._numint.pot_guess, committed)


@pytest.mark.parametrize('preconditioner', ['identity', 'kerker', 'elliptic'])
def test_opt_in_fixed_n_potential_driver_converges(setup, preconditioner):
    mf, solver, target_g = setup
    solver.max_cycle = 8
    initial = cp.asarray([4.0, -2.0, -2.0], dtype=cp.complex128)
    energy = solver.potential_scf(
        v0_g=initial, preconditioner=preconditioner,
        alpha=1.0, anderson_space=0, q0_sq=0.0,
        b_metal=0.0, potential_conv_tol=1e-11,
        max_step_rms=100.0, max_step_abs=100.0)

    assert np.isfinite(energy)
    assert solver.converged, solver.message
    assert solver.cycles == 2
    assert solver.nfev == 2
    assert solver.potential_residual_rms <= 1e-11
    cp.testing.assert_allclose(
        solver._potential_cycle.v_in_g, target_g,
        rtol=0.0, atol=2e-13)
    cp.testing.assert_array_equal(
        mf._numint.pot_guess,
        solver._potential_cycle.lpbe_pot_guess)
    assert 'potential_residual_rms' in solver.scf_summary


def test_initial_density_supplies_first_potential(setup):
    unused_mf, solver, target_g = setup
    solver.max_cycle = 3
    solver.potential_scf(
        preconditioner='identity', alpha=0.5,
        potential_conv_tol=1e-11)
    assert solver.converged
    assert solver.nfev == 2
    cp.testing.assert_allclose(
        solver._potential_cycle.v_in_g, target_g,
        rtol=0.0, atol=2e-13)


def test_fixed_mu_potential_driver_uses_fixed_n_root(setup):
    mf, unused_solver, target_g = setup
    target_h = mf._numint.local_potential_to_ao(target_g)[0, 0, 0].item()
    solver = GrandCanonicalKRKS(mf, sigma=0.1, mu=target_h)
    solver.conv_tol = 1e-11
    solver.conv_tol_coarse = 1e-8
    solver.conv_tol_mu = 1e-10
    solver.max_cycle = 6
    solver.max_outer_cycle = 6
    initial = cp.asarray([4.0, -2.0, -2.0], dtype=cp.complex128)

    energy = solver.potential_scf(
        v0_g=initial, initial_nelec=0.7,
        preconditioner='identity', alpha=1.0, anderson_space=0,
        potential_conv_tol=1e-11,
        max_step_rms=100.0, max_step_abs=100.0)

    assert np.isfinite(energy)
    assert solver.converged, solver.message
    assert solver.outer_cycles == 2
    assert abs(solver.electron_number - 1.0) < 1e-9
    assert abs(solver.mu - target_h) < solver.conv_tol_mu
    assert solver.potential_residual_rms <= solver.conv_tol
