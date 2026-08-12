"""GPU pseudo-spectral operators for periodic elliptic equations.

This module is intentionally independent of the SCF and solvent drivers.  All
reciprocal-space arrays use the GPU4PySCF multigrid convention

    f_G = (cell.vol / ngrids) * fft(f_R).

The common scale cancels from the linear operator, so intermediate FFTs do not
need to divide and multiply by the real-space quadrature weight.
"""

import time

import cupy as cp
import numpy as np
from cupyx.scipy.sparse.linalg import LinearOperator, cg

from gpu4pyscf.pbc.tools import pbc as pbc_tools


class EllipticSolveDiagnostics:
    __slots__ = (
        'info', 'iterations', 'initial_residual', 'final_residual',
        'wall_time', 'zero_mean',
    )

    def __init__(self, info, iterations, initial_residual, final_residual,
                 wall_time, zero_mean):
        self.info = int(info)
        self.iterations = int(iterations)
        self.initial_residual = float(initial_residual)
        self.final_residual = float(final_residual)
        self.wall_time = float(wall_time)
        self.zero_mean = bool(zero_mean)

    @property
    def converged(self):
        return self.info == 0


class EllipticSolveResult:
    __slots__ = ('solution_g', 'diagnostics')

    def __init__(self, solution_g, diagnostics):
        self.solution_g = solution_g
        self.diagnostics = diagnostics


def _as_flat_complex(values, ngrids, name):
    values = cp.asarray(values, dtype=cp.complex128).reshape(-1)
    if values.size != ngrids:
        raise ValueError(
            '%s has %d values; mesh requires %d' %
            (name, values.size, ngrids))
    if not bool(cp.all(cp.isfinite(values)).item()):
        raise FloatingPointError('%s contains nonfinite values' % name)
    return values


def _as_real_field(values, mesh, name):
    values = cp.asarray(values, dtype=cp.float64)
    if values.size != int(np.prod(mesh)):
        raise ValueError(
            '%s has %d values; mesh requires %d' %
            (name, values.size, int(np.prod(mesh))))
    values = values.reshape(tuple(mesh))
    if not bool(cp.all(cp.isfinite(values)).item()):
        raise FloatingPointError('%s contains nonfinite values' % name)
    return values


def _zero_nyquist_component(values, mesh, axis):
    if mesh is not None and mesh[axis] % 2 == 0:
        view = values.reshape(tuple(mesh))
        index = [slice(None)] * 3
        index[axis] = mesh[axis] // 2
        view[tuple(index)] = 0.0


def reciprocal_gradient(field_g, Gv, out=None, mesh=None):
    """Return ``grad(field)`` in reciprocal space."""
    field_g = cp.asarray(field_g).reshape(-1)
    Gv = cp.asarray(Gv).reshape(-1, 3)
    if field_g.size != Gv.shape[0]:
        raise ValueError('field and reciprocal grid sizes differ')
    if out is None:
        out = cp.empty((3, field_g.size), dtype=cp.complex128)
    elif out.shape != (3, field_g.size):
        raise ValueError('gradient output has the wrong shape')
    for axis in range(3):
        out[axis] = 1j * Gv[:, axis] * field_g
        # A first derivative at the self-conjugate Nyquist frequency has no
        # real-valued spectral representation.  Zero that component so real
        # fields remain real on even meshes.
        _zero_nyquist_component(out[axis], mesh, axis)
    return out


def reciprocal_divergence(vector_g, Gv, out=None, mesh=None):
    """Return ``div(vector)`` in reciprocal space."""
    vector_g = cp.asarray(vector_g)
    Gv = cp.asarray(Gv).reshape(-1, 3)
    if vector_g.shape != (3, Gv.shape[0]):
        raise ValueError('vector field and reciprocal grid sizes differ')
    if out is None:
        out = cp.zeros(Gv.shape[0], dtype=cp.complex128)
    else:
        if out.shape != (Gv.shape[0],):
            raise ValueError('divergence output has the wrong shape')
        out.fill(0.0)
    for axis in range(3):
        component = 1j * Gv[:, axis] * vector_g[axis]
        if mesh is not None:
            component = component.copy()
            _zero_nyquist_component(component, mesh, axis)
        out += component
    return out


def project_zero_mean_g(field_g, copy=True):
    """Remove the constant Fourier component from a scalar field."""
    field_g = cp.asarray(field_g).reshape(-1)
    if field_g.size == 0:
        raise ValueError('cannot project an empty field')
    if copy:
        field_g = field_g.copy()
    field_g[0] = 0.0
    return field_g


def reciprocal_laplacian_symbol(Gv, mesh):
    """Return a real-field-safe ``G2`` symbol for an FFT mesh.

    On a skew cell with an even mesh, the raw ``|G|^2`` values are not
    conjugate-symmetric on Nyquist planes because those self-conjugate integer
    frequencies have an ambiguous sign and reciprocal-vector cross terms do
    not cancel.  Averaging conjugate partners selects the unique Hermitian
    Laplacian symbol.  It is identical to raw ``|G|^2`` away from that case.
    """
    mesh = tuple(int(x) for x in mesh)
    Gv = cp.asarray(Gv, dtype=cp.float64).reshape(-1, 3)
    if Gv.shape[0] != int(np.prod(mesh)):
        raise ValueError('Gv and mesh sizes differ')
    raw = cp.einsum('gi,gi->g', Gv, Gv).reshape(mesh)
    negative = [(-cp.arange(n)) % n for n in mesh]
    conjugate = raw[
        negative[0][:, None, None],
        negative[1][None, :, None],
        negative[2][None, None, :],
    ]
    return (0.5 * (raw + conjugate)).reshape(-1)


def _apply_periodic_elliptic_unchecked(field_g, mesh, Gv, G2, base_a,
                                       delta_a_r, m_r, zero_mean):
    grad_g = reciprocal_gradient(
        field_g, Gv, mesh=mesh).reshape((3,) + mesh)
    # The omitted 1/weight on IFFT and weight on FFT cancel by linearity.
    grad_r_scaled = pbc_tools.ifft(grad_g.reshape(3, -1), mesh).reshape(
        (3,) + mesh)
    flux_g = pbc_tools.fft(
        (delta_a_r * grad_r_scaled).reshape(3, -1), mesh).reshape(3, -1)
    kinetic_g = (
        base_a * G2 * field_g
        - reciprocal_divergence(flux_g, Gv, mesh=mesh))

    field_r_scaled = pbc_tools.ifft(field_g, mesh).reshape(mesh)
    mass_g = pbc_tools.fft((m_r * field_r_scaled).reshape(-1), mesh)
    result = (kinetic_g + mass_g.reshape(-1)).reshape(-1)
    if zero_mean:
        result[0] = 0.0
    return result


def apply_periodic_elliptic(field_g, mesh, Gv, a_r, m_r,
                            zero_mean=False):
    """Apply ``-div(a(r) grad) + m(r)`` without forming a grid matrix."""
    mesh = tuple(int(x) for x in mesh)
    ngrids = int(np.prod(mesh))
    field_g = _as_flat_complex(field_g, ngrids, 'field')
    a_r = _as_real_field(a_r, mesh, 'a')
    m_r = _as_real_field(m_r, mesh, 'm')
    Gv = cp.asarray(Gv, dtype=cp.float64).reshape(-1, 3)
    if Gv.shape[0] != ngrids:
        raise ValueError('Gv and mesh sizes differ')
    G2 = reciprocal_laplacian_symbol(Gv, mesh)
    base_a = float(cp.min(a_r).item())
    delta_a_r = a_r - base_a
    return _apply_periodic_elliptic_unchecked(
        field_g, mesh, Gv, G2, base_a, delta_a_r, m_r,
        bool(zero_mean))


def constant_coefficient_inverse(rhs_g, G2, a=1.0, m=0.0,
                                 zero_mean=None):
    """Apply the exact reciprocal inverse for constant coefficients."""
    a = float(a)
    m = float(m)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError('a must be finite and positive')
    if not np.isfinite(m) or m < 0.0:
        raise ValueError('m must be finite and nonnegative')
    G2 = cp.asarray(G2, dtype=cp.float64).reshape(-1)
    rhs_g = _as_flat_complex(rhs_g, G2.size, 'right-hand side')
    if zero_mean is None:
        zero_mean = m == 0.0
    denominator = a * G2 + m
    out = cp.zeros_like(rhs_g)
    mask = denominator > 0.0
    out[mask] = rhs_g[mask] / denominator[mask]
    if zero_mean:
        out[0] = 0.0
    elif not bool(mask[0].item()):
        raise ValueError('constant mode is singular without zero-mean projection')
    return out


def solve_periodic_elliptic(rhs_g, mesh, Gv, a_r, m_r, tol=1e-8,
                            maxiter=200, x0=None, zero_mean=None,
                            use_preconditioner=True, a_min=1e-12):
    """Solve a periodic variable-coefficient elliptic equation with CG.

    The returned result owns its solution and diagnostics.  No warm-start or
    other global state is mutated.
    """
    mesh = tuple(int(x) for x in mesh)
    ngrids = int(np.prod(mesh))
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError('tol must be finite and positive')
    if int(maxiter) <= 0:
        raise ValueError('maxiter must be positive')
    a_r = _as_real_field(a_r, mesh, 'a')
    m_r = _as_real_field(m_r, mesh, 'm')
    amin = float(cp.min(a_r).item())
    mmin = float(cp.min(m_r).item())
    mmax = float(cp.max(m_r).item())
    if amin < a_min:
        raise ValueError('a must be at least %g (found %g)' % (a_min, amin))
    if mmin < 0.0:
        raise ValueError('m must be nonnegative (found %g)' % mmin)
    if zero_mean is None:
        zero_mean = mmax == 0.0

    Gv = cp.asarray(Gv, dtype=cp.float64).reshape(-1, 3)
    if Gv.shape[0] != ngrids:
        raise ValueError('Gv and mesh sizes differ')
    G2 = reciprocal_laplacian_symbol(Gv, mesh)
    base_a = amin
    delta_a_r = a_r - base_a
    rhs_g = _as_flat_complex(rhs_g, ngrids, 'right-hand side').copy()
    if zero_mean:
        rhs_g[0] = 0.0
    if x0 is None:
        x0 = cp.zeros(ngrids, dtype=cp.complex128)
    else:
        x0 = _as_flat_complex(x0, ngrids, 'initial guess').copy()
        if zero_mean:
            x0[0] = 0.0

    def matvec(vector):
        return _apply_periodic_elliptic_unchecked(
            cp.asarray(vector).reshape(-1), mesh, Gv, G2, base_a,
            delta_a_r, m_r, zero_mean)

    operator = LinearOperator((ngrids, ngrids), matvec=matvec)
    preconditioner = None
    if use_preconditioner:
        mean_a = float(cp.mean(a_r).item())
        mean_m = float(cp.mean(m_r).item())
        denominator = mean_a * G2 + mean_m
        inverse_denominator = cp.zeros_like(denominator)
        mask = denominator > 0.0
        inverse_denominator[mask] = 1.0 / denominator[mask]

        def psolve(vector):
            out = cp.asarray(vector).reshape(-1) * inverse_denominator
            if zero_mean:
                out[0] = 0.0
            return out

        preconditioner = LinearOperator((ngrids, ngrids), matvec=psolve)

    initial_residual = float(cp.linalg.norm(rhs_g - matvec(x0)).item())
    iterations = 0

    def callback(unused_solution):
        nonlocal iterations
        iterations += 1

    start = time.perf_counter()
    solution_g, info = cg(
        operator, rhs_g, x0=x0, M=preconditioner, tol=tol,
        maxiter=int(maxiter), callback=callback)
    wall_time = time.perf_counter() - start
    solution_g = cp.asarray(solution_g).reshape(-1)
    if zero_mean:
        solution_g[0] = 0.0
    final_residual = float(
        cp.linalg.norm(rhs_g - matvec(solution_g)).item())
    diagnostics = EllipticSolveDiagnostics(
        info, iterations, initial_residual, final_residual, wall_time,
        zero_mean)
    return EllipticSolveResult(solution_g.copy(), diagnostics)
