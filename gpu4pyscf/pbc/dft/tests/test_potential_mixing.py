from types import SimpleNamespace

import cupy as cp
import numpy as np

from pyscf.pbc import gto

from gpu4pyscf.pbc.dft import potential_mixing as mixing
from gpu4pyscf.pbc.tools import pbc as pbc_tools


def _context(mesh=(7, 8, 9), skew=False):
    lattice = (
        [[5.0, 0.3, 0.1], [0.0, 4.0, 0.2], [0.0, 0.0, 6.0]]
        if skew else np.diag([5.0, 4.0, 6.0]))
    cell = gto.M(
        a=lattice, atom='He 0 0 0', basis='gth-szv', pseudo='gth-pade',
        unit='bohr', mesh=mesh, verbose=0)
    Gv = cp.asarray(pbc_tools.get_Gv(cell, mesh))
    G2 = cp.einsum('gi,gi->g', Gv, Gv)
    return SimpleNamespace(
        cell=cell, mesh=tuple(mesh), Gv=Gv, G2=G2,
        cavity_r=cp.zeros(mesh))


def _field_r(context):
    indices = cp.indices(context.mesh, dtype=cp.float64)
    return (
        cp.cos(2 * np.pi * indices[0] / context.mesh[0])
        + 0.3 * cp.sin(4 * np.pi * indices[2] / context.mesh[2]))


def test_real_reciprocal_conversion_and_grid_metrics():
    context = _context(skew=True)
    field_r = _field_r(context)
    field_g = mixing.real_to_reciprocal(field_r, context)
    recovered_r = mixing.reciprocal_to_real(field_g, context)
    cp.testing.assert_allclose(recovered_r, field_r, rtol=0.0, atol=2e-13)
    assert abs(
        mixing.grid_inner(field_r, field_r, context)
        - context.cell.vol * mixing.grid_rms(field_r) ** 2) < 2e-12


def test_identity_and_kerker_filters_remove_constant_mode():
    context = _context()
    residual_g = mixing.real_to_reciprocal(_field_r(context), context)
    residual_g[0] = 3.2

    identity = mixing.IdentityPreconditioner().apply(
        residual_g, context).value_g
    assert identity[0] == 0.0
    cp.testing.assert_array_equal(identity[1:], residual_g[1:])

    q0_sq = 0.63
    kerker = mixing.KerkerPreconditioner(q0_sq).apply(
        residual_g, context).value_g
    expected = cp.zeros_like(residual_g)
    mask = context.G2 > 0.0
    expected[mask] = (
        context.G2[mask] / (context.G2[mask] + q0_sq)
        * residual_g[mask])
    cp.testing.assert_allclose(kerker, expected, rtol=2e-14, atol=2e-14)


def test_constant_elliptic_matches_kerker():
    context = _context(skew=True)
    residual_g = mixing.real_to_reciprocal(_field_r(context), context)
    q0_sq = 0.81
    elliptic = mixing.EllipticPreconditioner(
        a_out=1.0, b_metal=q0_sq / (4 * np.pi),
        tol=1e-12, maxiter=20)
    actual = elliptic.apply(residual_g, context)
    expected = mixing.KerkerPreconditioner(q0_sq).apply(
        residual_g, context)

    assert actual.diagnostics.success
    assert actual.diagnostics.inner.converged
    cp.testing.assert_allclose(
        actual.value_g, expected.value_g, rtol=2e-11, atol=2e-11)


def test_elliptic_failure_has_deterministic_fallback():
    context = _context()
    indices = cp.indices(context.mesh, dtype=cp.float64)
    context.cavity_r = 0.5 + 0.45 * cp.sin(
        2 * np.pi * indices[0] / context.mesh[0])
    rng = cp.random.RandomState(3)
    residual_r = rng.standard_normal(context.mesh)
    residual_r -= cp.mean(residual_r)
    residual_g = mixing.real_to_reciprocal(residual_r, context)
    fallback = mixing.KerkerPreconditioner(0.4)
    preconditioner = mixing.EllipticPreconditioner(
        a_out=2.0, b_metal=0.2, tol=1e-14, maxiter=1,
        fallback=fallback)

    actual = preconditioner.apply(residual_g, context)
    expected = fallback.apply(residual_g, context)
    assert not actual.diagnostics.success
    assert actual.diagnostics.name == 'elliptic->kerker'
    cp.testing.assert_array_equal(actual.value_g, expected.value_g)


def test_simple_mixing_residual_sign_and_step_trust():
    context = _context()
    potential_g = mixing.real_to_reciprocal(
        cp.zeros(context.mesh), context)
    residual_g = mixing.real_to_reciprocal(_field_r(context), context)
    mixer = mixing.AndersonMixer(
        alpha=0.4, history=0, max_step_rms=0.1)
    result = mixer.propose(
        potential_g, residual_g, mixing.IdentityPreconditioner(), context)
    step_r = mixing.reciprocal_to_real(result.step_g, context)

    assert result.diagnostics.scale < 1.0
    assert abs(mixing.grid_rms(step_r) - 0.1) < 2e-13
    cp.testing.assert_allclose(
        result.potential_g, potential_g + result.step_g,
        rtol=0.0, atol=0.0)


def test_regularized_anderson_secant_update():
    context = _context()
    target_r = _field_r(context)
    zero_g = mixing.real_to_reciprocal(cp.zeros(context.mesh), context)
    target_g = mixing.real_to_reciprocal(target_r, context)
    alpha = 0.25
    v0_g = zero_g
    r0_g = target_g
    v1_g = alpha * target_g
    r1_g = (1.0 - alpha) * target_g
    mixer = mixing.AndersonMixer(
        alpha=alpha, history=4, regularization=1e-14)
    mixer.accept(v0_g, r0_g, context)
    mixer.accept(v1_g, r1_g, context)

    result = mixer.propose(
        v1_g, r1_g, mixing.IdentityPreconditioner(), context)
    assert result.diagnostics.history == 1
    assert not result.diagnostics.fallback
    cp.testing.assert_allclose(
        result.potential_g, target_g, rtol=2e-12, atol=2e-12)


def test_anderson_rejected_trial_does_not_enter_history():
    context = _context()
    potential_g = mixing.real_to_reciprocal(
        cp.zeros(context.mesh), context)
    residual_g = mixing.real_to_reciprocal(_field_r(context), context)
    mixer = mixing.AndersonMixer(alpha=0.3, history=3)
    mixer.accept(potential_g, residual_g, context)
    first = mixer.propose(
        potential_g, residual_g, mixing.IdentityPreconditioner(), context)
    second = mixer.propose(
        potential_g, residual_g, mixing.IdentityPreconditioner(), context)

    assert first.diagnostics.history == second.diagnostics.history == 0
    cp.testing.assert_array_equal(first.potential_g, second.potential_g)
