"""Potential-space preconditioners and regularized Anderson mixing.

The fixed-point residual convention in this module is always

    R_G = V_out,G - V_in,G
    V_next = V_in + C R.

All mixers remove the constant Fourier mode.  A fixed-N caller is responsible
for aligning that gauge exactly before invoking these routines.
"""

import cupy as cp
import numpy as np
from numpy.linalg import LinAlgError

from gpu4pyscf.pbc.tools import elliptic
from gpu4pyscf.pbc.tools import pbc as pbc_tools


class PreconditionerDiagnostics:
    __slots__ = ('name', 'success', 'message', 'inner')

    def __init__(self, name, success=True, message='', inner=None):
        self.name = str(name)
        self.success = bool(success)
        self.message = str(message)
        self.inner = inner


class PreconditionerResult:
    __slots__ = ('value_g', 'diagnostics')

    def __init__(self, value_g, diagnostics):
        self.value_g = value_g
        self.diagnostics = diagnostics


class MixingDiagnostics:
    __slots__ = (
        'history', 'condition', 'coefficients', 'step_rms', 'step_max',
        'scale', 'fallback', 'preconditioner',
    )

    def __init__(self, history, condition, coefficients, step_rms,
                 step_max, scale, fallback, preconditioner):
        self.history = int(history)
        self.condition = float(condition)
        self.coefficients = coefficients
        self.step_rms = float(step_rms)
        self.step_max = float(step_max)
        self.scale = float(scale)
        self.fallback = bool(fallback)
        self.preconditioner = preconditioner


class MixingResult:
    __slots__ = ('potential_g', 'step_g', 'diagnostics')

    def __init__(self, potential_g, step_g, diagnostics):
        self.potential_g = potential_g
        self.step_g = step_g
        self.diagnostics = diagnostics


def _context_value(context, name):
    if isinstance(context, dict):
        try:
            return context[name]
        except KeyError:
            raise ValueError('preconditioner context lacks %s' % name) from None
    if not hasattr(context, name):
        raise ValueError('preconditioner context lacks %s' % name)
    return getattr(context, name)


def _mesh(context):
    return tuple(int(x) for x in _context_value(context, 'mesh'))


def _weight(context):
    mesh = _mesh(context)
    cell = _context_value(context, 'cell')
    return float(cell.vol) / int(np.prod(mesh))


def reciprocal_to_real(field_g, context):
    """Convert a normalized reciprocal field to a real ``float64`` grid."""
    mesh = _mesh(context)
    field_g = cp.asarray(field_g, dtype=cp.complex128).reshape(-1)
    if field_g.size != int(np.prod(mesh)):
        raise ValueError('reciprocal field size does not match mesh')
    field_r_complex = (
        pbc_tools.ifft(field_g, mesh).reshape(mesh) / _weight(context))
    imag_max = float(cp.max(cp.abs(field_r_complex.imag)).item())
    real_scale = max(
        1.0, float(cp.max(cp.abs(field_r_complex.real)).item()))
    if imag_max > 1e-10 * real_scale:
        raise ValueError(
            'reciprocal field does not represent a real potential '
            '(max imaginary part %g)' % imag_max)
    return cp.asarray(field_r_complex.real, dtype=cp.float64)


def real_to_reciprocal(field_r, context):
    """Convert a real grid to the normalized reciprocal convention."""
    mesh = _mesh(context)
    field_r = cp.asarray(field_r, dtype=cp.float64)
    if field_r.size != int(np.prod(mesh)):
        raise ValueError('real field size does not match mesh')
    return (
        pbc_tools.fft(field_r.reshape(-1), mesh).reshape(-1)
        * _weight(context))


def grid_inner(left_r, right_r, context):
    """Cell-volume-weighted real-grid inner product."""
    left_r = cp.asarray(left_r, dtype=cp.float64).reshape(-1)
    right_r = cp.asarray(right_r, dtype=cp.float64).reshape(-1)
    if left_r.shape != right_r.shape:
        raise ValueError('grid fields have different sizes')
    return float((cp.dot(left_r, right_r) * _weight(context)).item())


def grid_rms(field_r):
    field_r = cp.asarray(field_r, dtype=cp.float64).reshape(-1)
    return float(cp.sqrt(cp.mean(field_r * field_r)).item())


def zero_mean_real(field_r):
    field_r = cp.asarray(field_r, dtype=cp.float64)
    return field_r - cp.mean(field_r)


class IdentityPreconditioner:
    def apply(self, residual_g, context):
        residual_g = cp.asarray(residual_g, dtype=cp.complex128).reshape(-1)
        if not bool(cp.all(cp.isfinite(residual_g)).item()):
            raise FloatingPointError('residual contains nonfinite values')
        value_g = elliptic.project_zero_mean_g(residual_g)
        return PreconditionerResult(
            value_g,
            PreconditionerDiagnostics('identity'))


class KerkerPreconditioner:
    def __init__(self, q0_sq):
        self.q0_sq = float(q0_sq)
        if not np.isfinite(self.q0_sq) or self.q0_sq < 0.0:
            raise ValueError('q0_sq must be finite and nonnegative')

    def apply(self, residual_g, context):
        residual_g = cp.asarray(residual_g, dtype=cp.complex128).reshape(-1)
        G2 = cp.asarray(_context_value(context, 'G2')).reshape(-1)
        if residual_g.shape != G2.shape:
            raise ValueError('residual and G2 sizes differ')
        if not bool(cp.all(cp.isfinite(residual_g)).item()):
            raise FloatingPointError('residual contains nonfinite values')
        denominator = G2 + self.q0_sq
        value_g = cp.zeros_like(residual_g)
        mask = G2 > 0.0
        value_g[mask] = (
            G2[mask] / denominator[mask] * residual_g[mask])
        return PreconditionerResult(
            value_g, PreconditionerDiagnostics('kerker'))


class EllipticPreconditioner:
    """Lin--Yang paper operator using the accepted LPBE cavity field."""

    def __init__(self, a_out=1.0, b_metal=0.1, tol=1e-8, maxiter=200,
                 fallback=None, a_min=1e-12):
        self.a_out = float(a_out)
        self.b_metal = float(b_metal)
        self.tol = float(tol)
        self.maxiter = int(maxiter)
        self.a_min = float(a_min)
        if not np.isfinite(self.a_out) or self.a_out < 1.0:
            raise ValueError('a_out must be finite and at least one')
        if not np.isfinite(self.b_metal) or self.b_metal < 0.0:
            raise ValueError('b_metal must be finite and nonnegative')
        if not np.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError('tol must be finite and positive')
        if self.maxiter <= 0:
            raise ValueError('maxiter must be positive')
        self.fallback = fallback or IdentityPreconditioner()
        self._warm_rhs_g = None
        self._warm_solution_g = None

    def reset(self):
        self._warm_rhs_g = None
        self._warm_solution_g = None

    def apply(self, residual_g, context):
        mesh = _mesh(context)
        residual_g = cp.asarray(
            residual_g, dtype=cp.complex128).reshape(-1)
        G2 = cp.asarray(_context_value(context, 'G2')).reshape(-1)
        Gv = cp.asarray(_context_value(context, 'Gv')).reshape(-1, 3)
        cavity_r = cp.asarray(
            _context_value(context, 'cavity_r'), dtype=cp.float64)
        if cavity_r.size != int(np.prod(mesh)):
            raise ValueError('cavity and mesh sizes differ')
        cavity_r = cavity_r.reshape(mesh)
        cavity_min = float(cp.min(cavity_r).item())
        cavity_max = float(cp.max(cavity_r).item())
        if cavity_min < -1e-10 or cavity_max > 1.0 + 1e-10:
            raise ValueError('cavity must lie between zero and one')
        cavity_r = cp.clip(cavity_r, 0.0, 1.0)
        a_r = 1.0 + (self.a_out - 1.0) * cavity_r
        b_r = self.b_metal * (1.0 - cavity_r)
        m_r = 4.0 * np.pi * b_r
        rhs_g = G2 * residual_g

        x0 = None
        if self._warm_rhs_g is not None:
            numerator = abs(cp.vdot(self._warm_rhs_g, rhs_g).item())
            denominator = float(
                cp.linalg.norm(self._warm_rhs_g).item()
                * cp.linalg.norm(rhs_g).item())
            if denominator > 0.0 and numerator / denominator >= 0.5:
                x0 = self._warm_solution_g
        try:
            solved = elliptic.solve_periodic_elliptic(
                rhs_g, mesh, Gv, a_r, m_r, tol=self.tol,
                maxiter=self.maxiter, x0=x0, zero_mean=True,
                a_min=self.a_min)
            if not solved.diagnostics.converged:
                raise RuntimeError(
                    'inner CG returned info=%d' % solved.diagnostics.info)
            value_g = solved.solution_g
            # ``apply`` is called only for a current accepted residual; trial
            # evaluations never invoke a preconditioner, so this warm start is
            # transactional at the SCF-driver level.
            self._warm_rhs_g = rhs_g.copy()
            self._warm_solution_g = value_g.copy()
            return PreconditionerResult(
                value_g,
                PreconditionerDiagnostics(
                    'elliptic', inner=solved.diagnostics))
        except (FloatingPointError, RuntimeError) as error:
            fallback = self.fallback.apply(residual_g, context)
            return PreconditionerResult(
                fallback.value_g,
                PreconditionerDiagnostics(
                    'elliptic->%s' % fallback.diagnostics.name,
                    success=False, message=str(error),
                    inner=getattr(error, 'diagnostics', None)))


class AndersonMixer:
    """Regularized Anderson mixer storing accepted real-space history."""

    def __init__(self, alpha=0.5, history=6, regularization=1e-10,
                 coefficient_limit=20.0, max_step_rms=None,
                 max_step_abs=None):
        self.alpha = float(alpha)
        self.history = int(history)
        self.regularization = float(regularization)
        self.coefficient_limit = float(coefficient_limit)
        self.max_step_rms = (
            None if max_step_rms is None else float(max_step_rms))
        self.max_step_abs = (
            None if max_step_abs is None else float(max_step_abs))
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError('alpha must be finite and positive')
        if self.history < 0:
            raise ValueError('history must be nonnegative')
        if not np.isfinite(self.regularization) or self.regularization < 0.0:
            raise ValueError('regularization must be finite and nonnegative')
        if (not np.isfinite(self.coefficient_limit)
                or self.coefficient_limit <= 0.0):
            raise ValueError('coefficient_limit must be finite and positive')
        for name in ('max_step_rms', 'max_step_abs'):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError('%s must be finite and positive' % name)
        self.reset()

    def reset(self):
        self._s = []
        self._y = []
        self._last_v_r = None
        self._last_r_r = None

    def accept(self, potential_g, residual_g, context):
        """Commit one accepted fixed-point evaluation to the history."""
        potential_r = reciprocal_to_real(potential_g, context)
        residual_r = zero_mean_real(reciprocal_to_real(residual_g, context))
        if self._last_v_r is not None and self.history:
            self._s.append(
                zero_mean_real(potential_r - self._last_v_r).reshape(-1))
            self._y.append(
                zero_mean_real(residual_r - self._last_r_r).reshape(-1))
            if len(self._s) > self.history:
                self._s.pop(0)
                self._y.pop(0)
        self._last_v_r = potential_r.copy()
        self._last_r_r = residual_r.copy()

    def _simple_step(self, residual_g, preconditioner, context):
        preconditioned = preconditioner.apply(residual_g, context)
        step_r = self.alpha * reciprocal_to_real(
            preconditioned.value_g, context)
        return zero_mean_real(step_r), preconditioned.diagnostics

    def propose(self, potential_g, residual_g, preconditioner, context):
        """Propose a trial without committing it to accepted history."""
        potential_g = cp.asarray(
            potential_g, dtype=cp.complex128).reshape(-1)
        residual_g = elliptic.project_zero_mean_g(
            cp.asarray(residual_g, dtype=cp.complex128).reshape(-1))
        condition = 1.0
        coefficients = cp.empty(0, dtype=cp.float64)
        fallback = False

        if not self._y:
            step_r, preconditioner_diagnostics = self._simple_step(
                residual_g, preconditioner, context)
        else:
            residual_r = zero_mean_real(
                reciprocal_to_real(residual_g, context)).reshape(-1)
            S = cp.stack(self._s, axis=1)
            Y = cp.stack(self._y, axis=1)
            gram = _weight(context) * Y.T.dot(Y)
            rhs = _weight(context) * Y.T.dot(residual_r)
            scale = max(1.0, float(cp.trace(gram).item()) / len(self._y))
            regularized = gram + self.regularization * scale * cp.eye(
                len(self._y), dtype=cp.float64)
            try:
                gram_eigenvalues = cp.linalg.eigvalsh(regularized)
                smallest = float(gram_eigenvalues[0].item())
                largest = float(gram_eigenvalues[-1].item())
                condition = (
                    largest / smallest if smallest > 0.0 else np.inf)
                coefficients = cp.linalg.solve(regularized, rhs)
                if (not bool(cp.all(cp.isfinite(coefficients)).item())
                        or float(cp.max(cp.abs(coefficients)).item())
                        > self.coefficient_limit):
                    raise FloatingPointError('unsafe Anderson coefficients')
                remaining_r = residual_r - Y.dot(coefficients)
                remaining_g = real_to_reciprocal(
                    remaining_r.reshape(_mesh(context)), context)
                preconditioned = preconditioner.apply(
                    remaining_g, context)
                step_r = (
                    -S.dot(coefficients).reshape(_mesh(context))
                    + self.alpha * reciprocal_to_real(
                        preconditioned.value_g, context))
                step_r = zero_mean_real(step_r)
                preconditioner_diagnostics = preconditioned.diagnostics
            except (FloatingPointError, LinAlgError):
                fallback = True
                self.reset()
                step_r, preconditioner_diagnostics = self._simple_step(
                    residual_g, preconditioner, context)

        step_rms = grid_rms(step_r)
        step_max = float(cp.max(cp.abs(step_r)).item())
        scale = 1.0
        if self.max_step_rms is not None and step_rms > self.max_step_rms:
            scale = min(scale, self.max_step_rms / step_rms)
        if self.max_step_abs is not None and step_max > self.max_step_abs:
            scale = min(scale, self.max_step_abs / step_max)
        if scale < 1.0:
            step_r *= scale
            step_rms *= scale
            step_max *= scale
        step_g = real_to_reciprocal(step_r, context)
        step_g[0] = 0.0
        trial_g = potential_g + step_g
        if not bool(cp.all(cp.isfinite(trial_g)).item()):
            raise FloatingPointError('mixed potential contains nonfinite values')
        diagnostics = MixingDiagnostics(
            len(self._y), condition, coefficients.copy(), step_rms,
            step_max, scale, fallback, preconditioner_diagnostics)
        return MixingResult(trial_g, step_g, diagnostics)
