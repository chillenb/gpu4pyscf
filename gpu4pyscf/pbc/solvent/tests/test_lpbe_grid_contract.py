import cupy as cp
import numpy as np
import pytest

from pyscf.pbc import gto

from gpu4pyscf.pbc.solvent.lpbe_as_numint import LPBEMultiGridNumInt
from gpu4pyscf.pbc.tools import pbc as pbc_tools


@pytest.fixture(scope='module')
def cell():
    return gto.M(
        verbose=0,
        a=np.diag([5.0, 5.0, 6.0]),
        atom='C 0.0 0.0 0.0',
        basis=[[0, [1.2, 1.0]], [1, [0.8, 1.0]]],
        pseudo='gth-pade',
        unit='bohr',
        mesh=[10, 10, 12],
        precision=1e-9,
    )


@pytest.mark.parametrize('xc', ['lda,', 'pbe,'])
def test_lpbe_local_potential_round_trip(cell, xc):
    kpts = cell.make_kpts([2, 1, 1])
    nao = cell.nao_nr()
    dm = cp.stack([cp.eye(nao) for _ in kpts])
    ni = LPBEMultiGridNumInt(
        cell, tol=1e-10, ionic_strength=1.0,
        rel_permittivity=20.0)

    nelec, exc, veff = ni.nr_rks(
        cell, None, xc, dm, hermi=1, kpts=kpts, with_j=True)
    grid = veff.lpbe_grid
    round_trip = ni.local_potential_to_ao(grid.vlocal_g, kpts=kpts)

    assert np.isfinite(nelec)
    assert np.isfinite(exc)
    assert np.isfinite(complex(veff.ecoul))
    assert np.isfinite(float(veff.exc))
    assert round_trip.shape == veff.shape == (len(kpts), nao, nao)
    assert round_trip.dtype == veff.dtype == cp.complex128
    cp.testing.assert_allclose(round_trip, veff, rtol=0.0, atol=2e-11)

    ngrids = int(np.prod(cell.mesh))
    assert grid.vlocal_g.shape == grid.rho_g.shape == (ngrids,)
    assert grid.vlocal_g.dtype == grid.rho_g.dtype == cp.complex128
    assert grid.cavity_r.shape == grid.eps_r.shape == tuple(cell.mesh)
    assert grid.lpbe_mass_r.shape == tuple(cell.mesh)
    assert grid.lpbe_pot_guess.shape == (ngrids,)
    assert grid.cavity_r.dtype == grid.eps_r.dtype == cp.float64

    weight = cell.vol / ngrids
    vlocal_r = (
        pbc_tools.ifft(grid.vlocal_g, cell.mesh).real.reshape(-1) / weight)
    recovered_g = pbc_tools.fft(vlocal_r, cell.mesh).reshape(-1) * weight
    cp.testing.assert_allclose(
        recovered_g, grid.vlocal_g, rtol=2e-13, atol=2e-12)


def test_lpbe_grid_result_is_a_persistent_snapshot(cell):
    kpts = cell.make_kpts([1, 1, 1])
    nao = cell.nao_nr()
    dm = cp.eye(nao)[None]
    ni = LPBEMultiGridNumInt(cell, tol=1e-10, rel_permittivity=20.0)

    _, _, first_veff = ni.nr_rks(
        cell, None, 'lda,', dm, hermi=1, kpts=kpts, with_j=True)
    first = first_veff.lpbe_grid
    snapshots = {
        name: getattr(first, name).copy()
        for name in first.__slots__
    }

    _, _, second_veff = ni.nr_rks(
        cell, None, 'lda,', 0.9 * dm, hermi=1, kpts=kpts, with_j=True)
    second = second_veff.lpbe_grid

    for name, expected in snapshots.items():
        actual = getattr(first, name)
        cp.testing.assert_array_equal(actual, expected)
        assert not cp.shares_memory(actual, getattr(second, name))
    assert not cp.shares_memory(first.lpbe_pot_guess, ni.pot_guess)


def test_local_potential_to_ao_validates_input(cell):
    ni = LPBEMultiGridNumInt(cell)
    with pytest.raises(ValueError, match='mesh requires'):
        ni.local_potential_to_ao(cp.zeros(3))
    bad = cp.zeros(int(np.prod(cell.mesh)), dtype=cp.complex128)
    bad[2] = cp.nan
    with pytest.raises(FloatingPointError, match='nonfinite'):
        ni.local_potential_to_ao(bad)
