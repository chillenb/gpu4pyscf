import cupy as cp
import numpy as np
import pytest

from pyscf.pbc import gto

from gpu4pyscf.pbc.tools import elliptic
from gpu4pyscf.pbc.tools import pbc as pbc_tools


def _grid(skew=False):
    lattice = (
        np.asarray([[5.0, 0.2, 0.1], [0.0, 4.0, 0.3], [0.0, 0.0, 6.0]])
        if skew else np.diag([5.0, 4.0, 6.0]))
    cell = gto.M(
        a=lattice, atom='He 0 0 0', basis='gth-szv', pseudo='gth-pade',
        unit='bohr', mesh=[7, 8, 9], verbose=0)
    mesh = tuple(cell.mesh)
    Gv = cp.asarray(pbc_tools.get_Gv(cell, mesh))
    G2 = cp.einsum('gi,gi->g', Gv, Gv)
    return cell, mesh, Gv, G2


def _random_real_field_g(mesh, seed=4, zero_mean=True):
    rng = np.random.default_rng(seed)
    field_r = cp.asarray(rng.standard_normal(mesh))
    field_g = pbc_tools.fft(field_r.reshape(-1), mesh).reshape(-1)
    if zero_mean:
        field_g[0] = 0.0
    return field_g


@pytest.mark.parametrize('skew', [False, True])
def test_constant_operator_and_direct_inverse(skew):
    _, mesh, Gv, G2 = _grid(skew)
    field_g = _random_real_field_g(mesh)
    a = 1.7
    mass = 0.31
    a_r = cp.full(mesh, a)
    m_r = cp.full(mesh, mass)

    applied = elliptic.apply_periodic_elliptic(
        field_g, mesh, Gv, a_r, m_r)
    cp.testing.assert_allclose(
        applied, (a * G2 + mass) * field_g,
        rtol=2e-12, atol=2e-10)

    solved = elliptic.solve_periodic_elliptic(
        applied, mesh, Gv, a_r, m_r, tol=1e-12, maxiter=20)
    assert solved.diagnostics.converged
    assert solved.diagnostics.iterations <= 2
    cp.testing.assert_allclose(
        solved.solution_g, field_g, rtol=2e-11, atol=2e-10)


def test_singular_constant_operator_zero_mean_scaling():
    _, mesh, Gv, G2 = _grid()
    field_g = _random_real_field_g(mesh)
    a = 2.25
    rhs_g = G2 * field_g
    solved = elliptic.solve_periodic_elliptic(
        rhs_g, mesh, Gv, cp.full(mesh, a), cp.zeros(mesh),
        tol=1e-12, maxiter=20)
    assert solved.diagnostics.converged
    assert solved.diagnostics.zero_mean
    assert solved.solution_g[0] == 0.0
    cp.testing.assert_allclose(
        solved.solution_g, field_g / a, rtol=2e-11, atol=2e-10)


def test_variable_operator_is_hermitian_and_positive():
    _, mesh, Gv, _ = _grid(skew=True)
    x_g = _random_real_field_g(mesh, seed=8)
    y_g = _random_real_field_g(mesh, seed=9)
    coordinates = cp.indices(mesh, dtype=cp.float64)
    a_r = 1.1 + 0.2 * cp.sin(2 * np.pi * coordinates[0] / mesh[0])
    m_r = 0.05 * (
        1.0 + cp.cos(2 * np.pi * coordinates[2] / mesh[2]))
    ax_g = elliptic.apply_periodic_elliptic(
        x_g, mesh, Gv, a_r, m_r)
    ay_g = elliptic.apply_periodic_elliptic(
        y_g, mesh, Gv, a_r, m_r)

    lhs = cp.vdot(x_g, ay_g)
    rhs = cp.vdot(ax_g, y_g)
    cp.testing.assert_allclose(lhs, rhs, rtol=2e-12, atol=2e-8)
    assert float(cp.vdot(x_g, ax_g).real.item()) > 0.0

    ax_r = pbc_tools.ifft(ax_g, mesh).reshape(mesh)
    assert float(cp.max(cp.abs(ax_r.imag)).item()) < 2e-11


def test_kerker_is_constant_elliptic_solution():
    _, mesh, Gv, G2 = _grid()
    residual_g = _random_real_field_g(mesh, seed=12)
    q0_sq = 0.74
    rhs_g = G2 * residual_g
    solved = elliptic.solve_periodic_elliptic(
        rhs_g, mesh, Gv, cp.ones(mesh), cp.full(mesh, q0_sq),
        tol=1e-12, maxiter=20)
    kerker = cp.zeros_like(residual_g)
    mask = G2 > 0.0
    kerker[mask] = (
        G2[mask] / (G2[mask] + q0_sq) * residual_g[mask])

    assert solved.diagnostics.converged
    cp.testing.assert_allclose(
        solved.solution_g, kerker, rtol=2e-11, atol=2e-10)


def test_coefficient_and_nonfinite_validation():
    _, mesh, Gv, _ = _grid()
    rhs = cp.zeros(int(np.prod(mesh)), dtype=cp.complex128)
    with pytest.raises(ValueError, match='a must be at least'):
        elliptic.solve_periodic_elliptic(
            rhs, mesh, Gv, cp.zeros(mesh), cp.ones(mesh))
    with pytest.raises(ValueError, match='m must be nonnegative'):
        elliptic.solve_periodic_elliptic(
            rhs, mesh, Gv, cp.ones(mesh), -cp.ones(mesh))
    rhs[1] = cp.nan
    with pytest.raises(FloatingPointError, match='right-hand side'):
        elliptic.solve_periodic_elliptic(
            rhs, mesh, Gv, cp.ones(mesh), cp.ones(mesh))
