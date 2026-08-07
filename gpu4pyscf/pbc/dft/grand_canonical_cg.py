import numpy as np
import cupy as cp

from gpu4pyscf.lib import diis, logger


FERMI_DIVDIFF_RTOL = 1e-10
FERMI_RESPONSE_TOL = 1e-30
LINE_SEARCH_SLOPE_RATIO = 1e-1
LINE_SEARCH_ALPHA_RTOL = np.sqrt(np.finfo(float).eps)
LINE_SEARCH_ENERGY_RTOL = 64.0 * np.finfo(float).eps
LINE_SEARCH_OBJECTIVE_FLAT = 1e-8
LINE_SEARCH_RESIDUAL_REDUCTION = 1e-2
LINE_SEARCH_PARETO_RESIDUAL_REDUCTION = 1e-3
LINE_SEARCH_PARETO_ACTIVE_RESIDUAL = 2e-2
LINE_SEARCH_RESIDUAL_GROWTH_RESTART = 1.5
LINE_SEARCH_RESIDUAL_BOUNDARY_SAFETY = 0.9
LINE_SEARCH_PROGRESS_REDUCTION = 1e-1
LINE_SEARCH_STAGNATION_LIMIT = 2
LINE_SEARCH_FLAT_REFINEMENTS = 2
LINE_SEARCH_RESTORATION_EXPANSION_LIMIT = 4
LINE_SEARCH_TANGENT_REFINEMENTS = 4
LINE_SEARCH_NULL_TANGENT_CONSISTENCY_FACTOR = 1e4
LINE_SEARCH_NULL_EVALUATION_LIMIT = 8
LINE_SEARCH_INEXACT_STEP_RATIO = 5e-2
NLCG_POWELL_RESTART = 2e-1
NLCG_DESCENT_COSINE = np.sqrt(np.finfo(float).eps)
NLCG_DISPLACEMENT_HISTORY = 3
NLCG_CANONICAL_RESTORATION_SPACE = 6
NLCG_CANONICAL_RESTORATION_DAMP = 0.125
NLCG_CANONICAL_RESTORATION_FULL_DAMP_RATIO = 0.05
NLCG_CANONICAL_RESTORATION_REDUCTION = 0.1
NLCG_CANONICAL_RESTORATION_PROBE_REDUCTION = 0.5
NLCG_CANONICAL_RESTORATION_NELEC_STEP = 5e-3
NLCG_CANONICAL_RESTORATION_RESPONSE = 10.0
NLCG_CANONICAL_RESTORATION_HISTORY = 12
NLCG_CANONICAL_RESTORATION_WEAK_STEPS = 2
NLCG_PULAY_SPACE = 6
NLCG_PULAY_MIN_VECTORS = 3
# The regularized Pulay direction is an objective-safeguarded quasi-Newton
# probe for ordinarily smeared fixed-N calculations.  It remains disabled
# proactively in the sharp-sigma regime, where even gauge-aligned Pulay
# directions can be objective-descending while amplifying the Fermi-surface
# residual.
NLCG_PULAY_STAGNATION_STEPS = 3
NLCG_PULAY_BEST_REDUCTION = 1e-2
NLCG_PULAY_REGULARIZATION = 1e-3
NLCG_PULAY_MAX_COEFFICIENT = 4.0
NLCG_OCCUPATION_PRECONDITIONER_TARGET = 0.5
NLCG_OCCUPATION_PRECONDITIONER_MIN_ENERGY = 0.1
NLCG_OCCUPATION_PRECONDITIONER_MAX_ENERGY = 10.0
NLCG_OCCUPATION_ACTIVE_GROWTH = LINE_SEARCH_RESIDUAL_GROWTH_RESTART
# Private benchmark switch.  Production uses occupation-preconditioned PR+;
# the demo driver can disable it for equal-checkpoint A/B measurements.
NLCG_OCCUPATION_PR_ENABLED = True
NLCG_NULL_RESPONSE_TOL = 1e-6
NLCG_NULL_ACTIVE_RESIDUAL = 2e-2
NLCG_NULL_MIN_RESIDUAL = 1e-4
# Leave a useful margin above the geometric minimum.  Near the response-mask
# boundary a nominally null component can still have enough finite-temperature
# curvature to consume the entire restoration budget.  Requiring a 20% norm
# share limits proactive probes to components capable of a material cleanup.
NLCG_NULL_MIN_NORM_FRACTION = 0.2
NLCG_NULL_WEAK_REDUCTION = LINE_SEARCH_RESIDUAL_REDUCTION
NLCG_SHARP_SIGMA = 1e-3
NLCG_ORBITAL_TRIGGER_RESIDUAL = 5e-3
NLCG_ORBITAL_PHASE_STEPS = 20
NLCG_ORBITAL_LBFGS_SPACE = 6
NLCG_ORBITAL_LEVEL_SHIFT = .5
NLCG_ORBITAL_INITIAL_ALPHA = 1e-3
NLCG_ORBITAL_MAX_ROTATION = .25
NLCG_ORBITAL_LINE_EVALUATIONS = 10


class _LineSample:
    __slots__ = (
        'alpha', 'state', 'value', 'gradient', 'slope', 'method',
        'line_residual')

    def __init__(self, alpha, state, value, gradient, slope, method,
                 line_residual=None):
        self.alpha = float(alpha)
        self.state = state
        self.value = float(value)
        self.gradient = gradient
        self.slope = float(slope)
        self.method = method
        self.line_residual = (
            None if line_residual is None else float(line_residual))


class _LineSearchResult:
    __slots__ = (
        'sample', 'resolved', 'evaluations', 'reason',
        'restoration', 'consistency', 'slope_interval',
        'alpha_relative_uncertainty', 'normalized_slope')

    def __init__(self, sample, resolved, evaluations, reason,
                 restoration=False, consistency=0.0, slope_interval=None,
                 alpha_relative_uncertainty=np.nan,
                 normalized_slope=np.nan):
        self.sample = sample
        self.resolved = bool(resolved)
        self.evaluations = int(evaluations)
        self.reason = reason
        self.restoration = bool(restoration)
        self.consistency = float(consistency)
        self.slope_interval = (
            None if slope_interval is None else
            tuple(float(alpha) for alpha in slope_interval))
        self.alpha_relative_uncertainty = float(
            alpha_relative_uncertainty)
        self.normalized_slope = float(normalized_slope)


class _OrbitalHistory:
    __slots__ = (
        'pairs', 'previous_gradient', 'previous_step', 'rotation_seed')

    def __init__(self):
        self.clear()

    def clear(self):
        self.pairs = []
        self.previous_gradient = None
        self.previous_step = None
        self.rotation_seed = None


def _fermi_occ(gamma):
    """Evaluate 1 / (1 + exp(gamma)) without exponential overflow."""
    gamma = cp.asarray(gamma)
    positive = gamma >= 0
    out = cp.empty_like(gamma, dtype=cp.result_type(gamma, cp.float64))
    exp_negative = cp.exp(-gamma[positive])
    out[positive] = exp_negative / (1.0 + exp_negative)
    exp_positive = cp.exp(gamma[~positive])
    out[~positive] = 1.0 / (1.0 + exp_positive)
    return out


def fermi_divided_difference(gamma, rho, rtol=FERMI_DIVDIFF_RTOL):
    """Return the Hermitian Frechet divided difference of the Fermi map."""
    gamma = cp.asarray(gamma)
    rho = cp.asarray(rho)
    gamma_i = gamma[:, None]
    gamma_j = gamma[None, :]
    rho_i = rho[:, None]
    rho_j = rho[None, :]
    delta = gamma_i - gamma_j
    tolerance = rtol * cp.maximum(
        1.0, cp.maximum(cp.abs(gamma_i), cp.abs(gamma_j)))
    regular = cp.abs(delta) > tolerance
    safe_delta = cp.where(regular, delta, 1.0)
    midpoint_rho = _fermi_occ(0.5 * (gamma_i + gamma_j))
    divided = cp.where(
        regular,
        (rho_i - rho_j) / safe_delta,
        -midpoint_rho * (1.0 - midpoint_rho),
    )
    indices = cp.arange(gamma.size)
    divided[indices, indices] = -rho * (1.0 - rho)
    divided = 0.5 * (divided + divided.T)
    return cp.minimum(0.0, divided).real


def omega_gradient_wrt_h(h, f, beta, mu, diag_term_multiplier=1.0):
    """Return the real and imaginary gradients of the grand potential."""
    h = 0.5 * (h + h.conj().T)
    gamma_matrix = beta * (
        h - mu * cp.eye(h.shape[0], dtype=h.dtype))
    gamma, u = cp.linalg.eigh(gamma_matrix)
    rho = _fermi_occ(gamma)
    divided = fermi_divided_difference(gamma, rho)

    mismatch_tilde = u.conj().T @ (f - h) @ u
    response_tilde = divided * mismatch_tilde
    if diag_term_multiplier != 1.0:
        diagonal = cp.diag(response_tilde)
        response_tilde = response_tilde + cp.diag(
            (diag_term_multiplier - 1.0) * diagonal)

    gradient = 2.0 * beta * u @ response_tilde @ u.conj().T
    gradient = 0.5 * (gradient + gradient.conj().T)
    return gradient.real, gradient.imag


def mu_gradient_wrt_h(coeff, occ, weight):
    """Return d(mu)/dH for a globally constrained electron number."""
    response = []
    denominator = 0.0
    for u, rho in zip(coeff, occ):
        diagonal = rho * (1.0 - rho)
        value = (u * diagonal[None, :]) @ u.conj().T
        value = 0.5 * (value + value.conj().T)
        response.append(value)
        denominator += weight * float(cp.sum(diagonal).item())
    if not denominator > FERMI_RESPONSE_TOL:
        raise RuntimeError(
            'fixed-electron Fermi response is numerically singular')
    return [value / denominator for value in response]


def objective_gradient(self, state, fixed_n):
    gradient = [
        grad_re + 1j * grad_im
        for grad_re, grad_im in (
            omega_gradient_wrt_h(h, f, self.beta, state.mu)
            for h, f in zip(state.h, state.fock)
        )
    ]
    if fixed_n:
        mu_gradient = mu_gradient_wrt_h(
            state.coeff, state.occ, self.weight)
        gauge_derivative = self._inner(gradient, self.identity)
        gradient = [
            value - gauge_derivative * dmu
            for value, dmu in zip(gradient, mu_gradient)
        ]
    return self._hermi(gradient)


def _line_roundoff(value):
    return LINE_SEARCH_ENERGY_RTOL * max(1.0, abs(value))


def _line_objective_band(origin_value, consistency):
    """Return the absolute objective-equivalence threshold.

    The measured energy/slope inconsistency remains useful for deciding
    whether an interpolation model is trustworthy, but it must not make the
    definition of an objective-flat step tighter or looser.  In particular,
    residual restoration is allowed only inside the fixed 1e-8 Ha band.
    """
    del consistency
    return max(_line_roundoff(origin_value), LINE_SEARCH_OBJECTIVE_FLAT)


def _nonrestoration_objective_change(previous, origin_value, result):
    """Preserve descent history across bounded residual cleanup steps."""
    if result.restoration:
        return previous
    return float(result.sample.value-origin_value)


def _line_alpha_close(left, right):
    scale = max(1.0, abs(left), abs(right))
    return abs(left-right) <= LINE_SEARCH_ALPHA_RTOL * scale


def _line_candidate_is_new(alpha, samples, invalid_alphas=(),
                           consistency=0.0, predicted_slope=None,
                           allow_signed=False):
    if not np.isfinite(alpha) or (not allow_signed and alpha <= 0.0):
        return False
    old_alphas = [sample.alpha for sample in samples] + list(invalid_alphas)
    if any(_line_alpha_close(alpha, old) for old in old_alphas):
        return False
    if consistency > 0.0 and samples:
        nearest = min(samples, key=lambda sample: abs(sample.alpha-alpha))
        slope_scale = abs(nearest.slope)
        if predicted_slope is not None and np.isfinite(predicted_slope):
            slope_scale = max(slope_scale, abs(predicted_slope))
        if abs(alpha-nearest.alpha)*slope_scale <= consistency:
            return False
    return True


def _line_improves(value, origin_value):
    return value < origin_value-_line_roundoff(origin_value)


def _best_line_sample(samples):
    return min(samples, key=lambda sample: (sample.value, abs(sample.alpha)))


def _sample_residual(sample):
    if sample.line_residual is not None:
        return sample.line_residual
    return float(getattr(sample.state, 'residual_rms', np.inf))


def _residual_improves(sample, origin_sample):
    return _sample_residual(sample) <= (
        (1.0-LINE_SEARCH_RESIDUAL_REDUCTION) *
        _sample_residual(origin_sample))


def _residual_secant_step(self, origin_sample, sample):
    """Predict the least-squares residual minimum from two vector samples."""
    origin_residual = getattr(origin_sample.state, 'residual', None)
    sample_residual = getattr(sample.state, 'residual', None)
    if (origin_residual is None or sample_residual is None or
            sample.alpha <= 0.0):
        return None
    difference = [
        new-old for old, new in zip(origin_residual, sample_residual)]
    denominator = self._inner(difference, difference)
    numerator = self._inner(origin_residual, difference)
    if (not np.isfinite(denominator) or not np.isfinite(numerator) or
            denominator <= np.finfo(float).tiny):
        return None
    alpha = -sample.alpha*numerator/denominator
    if not np.isfinite(alpha) or alpha <= 0.0:
        return None
    return float(alpha)


def _lexicographic_line_sample(samples, origin_sample, consistency):
    candidates = [sample for sample in samples
                  if not _line_alpha_close(sample.alpha, 0.0)]
    if not candidates:
        return None
    minimum = min(sample.value for sample in samples)
    candidates = [
        sample for sample in candidates
        if (sample.value <= minimum+consistency and
            sample.value <= origin_sample.value+consistency)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda sample: (
        _sample_residual(sample), abs(sample.slope),
        sample.value, abs(sample.alpha)))


def _inexact_line_sample(samples, origin_sample, consistency):
    return _lexicographic_line_sample(
        samples, origin_sample, consistency)


def _restoration_line_sample(samples, origin_sample, consistency):
    candidates = [
        sample for sample in samples
        if (not _line_alpha_close(sample.alpha, 0.0) and
            sample.value <= origin_sample.value+consistency)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda sample: (
        _sample_residual(sample), abs(sample.slope),
        sample.value, abs(sample.alpha)))


def _pareto_line_sample(samples, origin_sample):
    """Return the best sample that improves both objective and residual."""
    candidates = [
        sample for sample in samples
        if (not _line_alpha_close(sample.alpha, 0.0) and
            _pareto_improves(sample, origin_sample))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda sample: (
        _sample_residual(sample), sample.value,
        abs(sample.slope), abs(sample.alpha)))


def _pareto_improves(sample, origin_sample):
    return (
        _line_improves(sample.value, origin_sample.value) and
        _sample_residual(sample) <=
        (1.0-LINE_SEARCH_PARETO_RESIDUAL_REDUCTION) *
        _sample_residual(origin_sample))


def _bounded_residual_growth_sample(
        samples, origin_sample,
        growth_limit=LINE_SEARCH_RESIDUAL_GROWTH_RESTART):
    candidates = [
        sample for sample in samples
        if (not _line_alpha_close(sample.alpha, 0.0) and
            _line_improves(sample.value, origin_sample.value) and
            _sample_residual(sample) <=
            growth_limit *
            _sample_residual(origin_sample))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda sample: (
        sample.value, _sample_residual(sample), abs(sample.alpha)))


def _residual_growth_boundary_step(
        samples, origin_sample,
        growth_limit=LINE_SEARCH_RESIDUAL_GROWTH_RESTART):
    """Interpolate just inside an adjacent residual-growth bracket."""
    target = growth_limit*_sample_residual(origin_sample)
    if not np.isfinite(target) or target <= 0.0:
        return None
    ordered = sorted(
        (sample for sample in samples if sample.alpha >= 0.0),
        key=lambda sample: sample.alpha)
    brackets = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        left_residual = _sample_residual(left)
        right_residual = _sample_residual(right)
        if (not all(np.isfinite(value) for value in (
                left_residual, right_residual)) or
                left_residual > target or right_residual <= target or
                right_residual <= left_residual or
                _line_alpha_close(left.alpha, right.alpha)):
            continue
        fraction = (target-left_residual)/(right_residual-left_residual)
        if not np.isfinite(fraction) or fraction <= 0.0:
            continue
        fraction = min(1.0, fraction)
        alpha = left.alpha + (
            LINE_SEARCH_RESIDUAL_BOUNDARY_SAFETY*fraction *
            (right.alpha-left.alpha))
        brackets.append((left.value, -left.alpha, alpha))
    if not brackets:
        return None
    unused_value, unused_alpha, alpha = min(brackets)
    if not _line_candidate_is_new(alpha, samples):
        return None
    return float(alpha)


def _harmonic_step(samples):
    """Return a positive-curvature secant root of the line derivative."""
    samples = sorted(samples, key=lambda sample: sample.alpha)
    best = _best_line_sample(samples)
    candidates = []
    for left, right in zip(samples[:-1], samples[1:]):
        width = right.alpha-left.alpha
        curvature = (right.slope-left.slope) / width
        if not np.isfinite(curvature) or curvature <= 0.0:
            continue
        alpha = left.alpha-left.slope/curvature
        if not np.isfinite(alpha):
            continue
        tolerance = LINE_SEARCH_ALPHA_RTOL * max(
            1.0, abs(left.alpha), abs(right.alpha))
        if not left.alpha+tolerance < alpha < right.alpha-tolerance:
            continue
        contains_best = left.alpha <= best.alpha <= right.alpha
        score = (not contains_best, abs(alpha-best.alpha), width)
        candidates.append((score, alpha, curvature,
                           (left.alpha, right.alpha)))
    if not candidates:
        return None, None, None
    unused_score, alpha, curvature, interval = min(candidates)
    return alpha, curvature, interval


def _energy_bracket(samples, invalid_alphas=()):
    best = _best_line_sample(samples)
    slots = [(sample.alpha, sample.value) for sample in samples]
    slots.extend((float(alpha), np.inf) for alpha in invalid_alphas)
    slots.sort()
    index = next(
        i for i, (alpha, unused_value) in enumerate(slots)
        if _line_alpha_close(alpha, best.alpha))
    if index == 0 or index == len(slots)-1:
        return None
    left, right = slots[index-1], slots[index+1]
    tolerance = _line_roundoff(best.value)
    if (left[1] >= best.value-tolerance and
            right[1] >= best.value-tolerance):
        return left[0], right[0]
    return None


def _sample_pair(samples, interval):
    if interval is None:
        return None
    left = next((sample for sample in samples
                 if _line_alpha_close(sample.alpha, interval[0])), None)
    right = next((sample for sample in samples
                  if _line_alpha_close(sample.alpha, interval[1])), None)
    if left is None or right is None or _line_alpha_close(
            left.alpha, right.alpha):
        return None
    return left, right


def _active_line_interval(samples, invalid_alphas=()):
    unused_alpha, unused_curvature, interval = _harmonic_step(samples)
    if interval is not None:
        return interval
    interval = _energy_bracket(samples, invalid_alphas)
    if _sample_pair(samples, interval) is not None:
        return interval
    ordered = sorted(samples, key=lambda sample: sample.alpha)
    best = _best_line_sample(ordered)
    index = ordered.index(best)
    if index == 0 and len(ordered) > 1:
        return ordered[0].alpha, ordered[1].alpha
    if index == len(ordered)-1 and index:
        return ordered[-2].alpha, ordered[-1].alpha
    if 0 < index < len(ordered)-1:
        left = (ordered[index-1].alpha, ordered[index].alpha)
        right = (ordered[index].alpha, ordered[index+1].alpha)
        return min((left, right), key=lambda pair: pair[1]-pair[0])
    return None


def _minimum_line_interval(samples, invalid_alphas=()):
    unused_alpha, unused_curvature, interval = _harmonic_step(samples)
    if interval is not None:
        return interval
    interval = _energy_bracket(samples, invalid_alphas)
    if interval is not None:
        return interval
    ordered = sorted(samples, key=lambda sample: sample.alpha)
    best = _best_line_sample(ordered)
    index = ordered.index(best)
    neighbor = None
    if index == 0 and len(ordered) > 1:
        neighbor = ordered[1]
    elif index == len(ordered)-1 and index:
        neighbor = ordered[-2]
    if (neighbor is not None and
            abs(neighbor.value-best.value) <= _line_roundoff(best.value)):
        return tuple(sorted((best.alpha, neighbor.alpha)))
    return None


def _pair_consistency(left, right, origin_value):
    width = right.alpha-left.alpha
    model_error = abs(
        (right.value-left.value) -
        0.5*(left.slope+right.slope)*width)
    if not np.isfinite(model_error):
        model_error = 0.0
    return max(_line_roundoff(origin_value), model_error)


def _line_consistency(samples, interval, origin_value):
    pair = _sample_pair(samples, interval)
    if pair is None:
        return _line_roundoff(origin_value)
    return _pair_consistency(pair[0], pair[1], origin_value)


def _convex_polynomial_minimum(polynomial, interval):
    lower, upper = interval
    candidates = []
    for root in polynomial.deriv().roots():
        if abs(root.imag) > 1e-10 * max(1.0, abs(root.real)):
            continue
        alpha = float(root.real)
        tolerance = LINE_SEARCH_ALPHA_RTOL * max(
            1.0, abs(lower), abs(upper))
        if not lower+tolerance < alpha < upper-tolerance:
            continue
        curvature = float(polynomial.deriv(2)(alpha))
        value = float(polynomial(alpha))
        if np.isfinite(value) and np.isfinite(curvature) and curvature > 0.0:
            candidates.append((value, alpha, curvature))
    if not candidates:
        return None, None
    unused_value, alpha, curvature = min(candidates)
    return alpha, curvature


def _convex_quadratic_step(alphas, values, interval):
    """Fit three values and return a convex stationary point."""
    alphas = np.asarray(alphas, dtype=float)
    values = np.asarray(values, dtype=float)
    if alphas.size != 3:
        return None, None
    scale = float(np.max(values)-np.min(values))
    if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
        return None, None
    scaled = (values-np.min(values)) / scale
    try:
        polynomial = np.polynomial.Polynomial.fit(
            alphas, scaled, alphas.size-1)
        coefficient_scale = float(np.max(np.abs(polynomial.coef)))
        polynomial = polynomial.trim(
            tol=256.0*np.finfo(float).eps*coefficient_scale)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None, None
    return _convex_polynomial_minimum(polynomial, interval)


def _hermite_step(left, right):
    """Return a convex minimum of the local energy-and-slope cubic."""
    width = right.alpha-left.alpha
    if not np.isfinite(width) or width <= 0.0:
        return None, None, None
    delta = right.value-left.value
    cubic = (left.slope+right.slope)/width**2 - 2.0*delta/width**3
    quadratic = 3.0*delta/width**2 - (
        2.0*left.slope+right.slope)/width
    derivative = np.polynomial.Polynomial(
        [left.slope, 2.0*quadratic, 3.0*cubic])
    tolerance = LINE_SEARCH_ALPHA_RTOL * max(
        1.0, abs(left.alpha), abs(right.alpha))
    candidates = []
    for root in derivative.roots():
        if abs(root.imag) > 1e-10*max(1.0, abs(root.real)):
            continue
        offset = float(root.real)
        alpha = left.alpha+offset
        if not left.alpha+tolerance < alpha < right.alpha-tolerance:
            continue
        curvature = 2.0*quadratic+6.0*cubic*offset
        value = (left.value + left.slope*offset +
                 quadratic*offset**2 + cubic*offset**3)
        if np.isfinite(value) and np.isfinite(curvature) and curvature > 0.0:
            candidates.append((value, alpha, curvature))
    if not candidates:
        return None, None, None
    value, alpha, curvature = min(candidates)
    return alpha, curvature, value


def _quadratic_line_step(samples, interval):
    best = _best_line_sample(samples)
    left = [sample for sample in samples if sample.alpha < best.alpha]
    right = [sample for sample in samples if sample.alpha > best.alpha]
    if not left or not right:
        return None, None, None
    selected = [max(left, key=lambda sample: sample.alpha), best,
                min(right, key=lambda sample: sample.alpha)]
    alphas = np.asarray([sample.alpha for sample in selected])
    values = np.asarray([sample.value for sample in selected])
    scale = float(np.max(values)-np.min(values))
    if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
        return None, None, None
    scaled = (values-np.min(values))/scale
    try:
        polynomial = np.polynomial.Polynomial.fit(alphas, scaled, 2)
        alpha, curvature = _convex_polynomial_minimum(polynomial, interval)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None, None, None
    if alpha is None:
        return None, None, None
    predicted = float(np.min(values)+scale*polynomial(alpha))
    return alpha, curvature, predicted


def _directional_midpoint(best, interval, samples, invalid_alphas):
    lower, upper = interval
    if best.slope < 0.0 and best.alpha < upper:
        return 0.5*(best.alpha+upper)
    if best.slope > 0.0 and best.alpha > lower:
        return 0.5*(lower+best.alpha)

    points = [lower, upper]
    points.extend(
        sample.alpha for sample in samples
        if lower < sample.alpha < upper)
    points.extend(
        alpha for alpha in invalid_alphas
        if lower < alpha < upper)
    points = sorted(points)
    left, right = max(
        zip(points[:-1], points[1:]), key=lambda pair: pair[1]-pair[0])
    return 0.5*(left+right)


def _line_search_proposal(samples, invalid_alphas, initial_step,
                          consistency=0.0, flat=False):
    """Choose a safeguarded absolute line coordinate."""
    samples = sorted(samples, key=lambda sample: sample.alpha)
    all_alphas = [sample.alpha for sample in samples] + list(invalid_alphas)
    if len(all_alphas) == 1:
        return float(initial_step), 'initial'

    best = _best_line_sample(samples)
    positive_slots = sorted(alpha for alpha in all_alphas if alpha > 0.0)
    if _line_alpha_close(best.alpha, 0.0):
        candidate = (float(initial_step) if not positive_slots else
                     0.5*positive_slots[0])
        if _line_candidate_is_new(candidate, samples, invalid_alphas):
            return candidate, 'contract'
        return None, None

    harmonic, curvature, harmonic_interval = _harmonic_step(samples)
    if harmonic_interval is not None:
        pair = _sample_pair(samples, harmonic_interval)
        if pair is not None:
            fitted, unused_curvature, predicted = _hermite_step(*pair)
            predicted_gain = (
                -np.inf if predicted is None else best.value-predicted)
            if (fitted is not None and
                    (not flat or predicted_gain > consistency) and
                    _line_candidate_is_new(
                        fitted, samples, invalid_alphas,
                        consistency if flat else 0.0)):
                return fitted, 'hermite'
        predicted_gain = (
            0.5*best.slope**2/curvature if curvature else 0.0)
        if (harmonic is not None and
                (not flat or predicted_gain > consistency) and
                _line_candidate_is_new(
                    harmonic, samples, invalid_alphas,
                    consistency if flat else 0.0, predicted_slope=0.0)):
            return harmonic, 'harmonic'
        if flat:
            # The consistency estimate belongs to the current bracket.  A
            # root that is functionally duplicated at that scale can still
            # be the one sample needed to shrink an overly broad bracket and
            # measure a useful local consistency band.
            if (harmonic is not None and
                    _line_candidate_is_new(
                        harmonic, samples, invalid_alphas)):
                return harmonic, 'consistency-harmonic'
            midpoint = _directional_midpoint(
                best, harmonic_interval, samples, invalid_alphas)
            if _line_candidate_is_new(
                    midpoint, samples, invalid_alphas):
                return midpoint, 'consistency-bisect'
            return None, None
        midpoint = _directional_midpoint(
            best, harmonic_interval, samples, invalid_alphas)
        if _line_candidate_is_new(midpoint, samples, invalid_alphas):
            return midpoint, 'slot-bisect'
        return None, None

    energy_interval = _energy_bracket(samples, invalid_alphas)
    if energy_interval is not None:
        fitted, unused_curvature, predicted = _quadratic_line_step(
            samples, energy_interval)
        predicted_gain = (
            -np.inf if predicted is None else best.value-predicted)
        if (fitted is not None and
                (not flat or predicted_gain > consistency) and
                _line_candidate_is_new(
                    fitted, samples, invalid_alphas,
                    consistency if flat else 0.0)):
            return fitted, 'quadratic'
        if flat:
            midpoint = _directional_midpoint(
                best, energy_interval, samples, invalid_alphas)
            if _line_candidate_is_new(
                    midpoint, samples, invalid_alphas):
                return midpoint, 'consistency-bisect'
            return None, None
        midpoint = _directional_midpoint(
            best, energy_interval, samples, invalid_alphas)
        if _line_candidate_is_new(midpoint, samples, invalid_alphas):
            return midpoint, 'slot-bisect'
        return None, None

    invalid_above = sorted(
        alpha for alpha in invalid_alphas if alpha > best.alpha)
    if invalid_above:
        candidate = 0.5*(best.alpha+invalid_above[0])
        if _line_candidate_is_new(candidate, samples, invalid_alphas):
            return candidate, 'contract-invalid'

    if best.alpha >= max(sample.alpha for sample in samples):
        if best.slope < 0.0:
            candidate = 2.0*best.alpha
            if _line_candidate_is_new(candidate, samples, invalid_alphas):
                return candidate, 'expand'
        previous = [sample.alpha for sample in samples
                    if sample.alpha < best.alpha]
        if previous:
            candidate = 0.5*(previous[-1]+best.alpha)
            if _line_candidate_is_new(candidate, samples, invalid_alphas):
                return candidate, 'contract'
    return None, None


def _sample_has_positive_curvature(samples, best):
    samples = sorted(samples, key=lambda sample: sample.alpha)
    index = samples.index(best)
    pairs = []
    if index:
        pairs.append((samples[index-1], best))
    if index+1 < len(samples):
        pairs.append((best, samples[index+1]))
    return any(
        (right.slope-left.slope) / (right.alpha-left.alpha) > 0.0
        for left, right in pairs
    )


def _strict_line_tolerances(self):
    alpha_rtol = getattr(self, 'nlcg_line_search_alpha_rtol', None)
    slope_atol = getattr(self, 'nlcg_line_search_slope_atol', None)
    if (alpha_rtol is None) != (slope_atol is None):
        raise ValueError(
            'nlcg_line_search_alpha_rtol and '
            'nlcg_line_search_slope_atol must both be set or both be None')
    if alpha_rtol is None:
        return None
    return float(alpha_rtol), float(slope_atol)


def _strict_slope_interval(samples, reference=None):
    ordered = sorted(samples, key=lambda sample: sample.alpha)
    brackets = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        width = right.alpha-left.alpha
        if width <= 0.0 or left.slope > 0.0 or right.slope < 0.0:
            continue
        curvature = (right.slope-left.slope)/width
        if not np.isfinite(curvature) or curvature <= 0.0:
            continue
        contains_reference = (
            reference is not None and
            left.alpha <= reference.alpha <= right.alpha)
        if reference is not None and not contains_reference:
            continue
        brackets.append((not contains_reference, width, left.alpha,
                         right.alpha))
    if not brackets:
        return None
    unused_contains, unused_width, left, right = min(brackets)
    return float(left), float(right)


def _strict_sample_metrics(samples, sample, direction_norm):
    interval = _strict_slope_interval(samples, sample)
    normalized_slope = (
        abs(sample.slope)/direction_norm
        if direction_norm > np.finfo(float).tiny else np.inf)
    if interval is None or abs(sample.alpha) <= np.finfo(float).tiny:
        return interval, np.inf, float(normalized_slope)
    alpha_uncertainty = (
        (interval[1]-interval[0])/abs(sample.alpha))
    return interval, float(alpha_uncertainty), float(normalized_slope)


def _strict_line_metrics(
        samples, origin_sample, consistency, direction_norm):
    """Return the most stationary energy-equivalent bracket endpoint."""
    minimum = min(sample.value for sample in samples)
    candidates = [
        sample for sample in samples
        if (not _line_alpha_close(sample.alpha, origin_sample.alpha) and
            sample.value <= minimum+consistency)
    ]
    if not candidates:
        return None, None, np.inf, np.inf
    metrics = []
    for index, candidate in enumerate(candidates):
        interval, alpha_uncertainty, normalized_slope = (
            _strict_sample_metrics(samples, candidate, direction_norm))
        if interval is not None:
            metrics.append((
                normalized_slope, candidate.value, alpha_uncertainty,
                abs(candidate.alpha), index, candidate, interval))
    if not metrics:
        return None, None, np.inf, np.inf
    (normalized_slope, unused_value, alpha_uncertainty, unused_alpha,
     unused_index, candidate, interval) = min(metrics)
    return candidate, interval, alpha_uncertainty, normalized_slope


def _line_search(self, origin, origin_gradient, direction, evaluate,
                 objective, initial_step, allow_restoration=False,
                 require_residual_improvement=False,
                 accept_bounded_residual=False,
                 bound_residual_growth=False, residual_metric=None,
                 residual_growth_limit=LINE_SEARCH_RESIDUAL_GROWTH_RESTART,
                 max_evaluations=None):
    evaluation_limit = self.nlcg_max_line_search_evaluations
    if max_evaluations is not None:
        evaluation_limit = min(evaluation_limit, int(max_evaluations))
    strict_tolerances = _strict_line_tolerances(self)
    strict_minimization = (
        strict_tolerances is not None and
        not allow_restoration and
        not require_residual_improvement and
        not accept_bounded_residual)
    direction_norm = _direction_norm(self, direction)
    if residual_metric is None:
        residual_metric = lambda state: state.residual_rms
    slope0 = self._inner(origin_gradient, direction)
    origin_sample = _LineSample(
        0.0, origin, objective(origin), origin_gradient, slope0, 'origin',
        residual_metric(origin))
    samples = [origin_sample]
    invalid_alphas = []
    evaluations = 0
    consistency = _line_roundoff(origin_sample.value)
    objective_band = _line_objective_band(
        origin_sample.value, consistency)
    bounded_restoration_band = objective_band
    active_interval = None
    refinement_interval = None
    refinements = 0
    stagnation = 0
    consistency_probe_done = False
    stopped_reason = 'interpolation stagnated'
    bounded_residual_step = float(initial_step)
    residual_growth_step = None

    def evaluate_alpha(alpha, method, trial_direction=None, cache=True):
        nonlocal evaluations
        if evaluations >= evaluation_limit:
            return None
        if trial_direction is None:
            trial_direction = direction
        evaluations += 1
        try:
            trial = evaluate([
                h+alpha*d for h, d in zip(origin.h, trial_direction)])
            gradient = objective_gradient(
                self, trial, self.nelec is not None)
            value = float(objective(trial))
            slope = self._inner(gradient, trial_direction)
            if not np.isfinite(value) or not np.isfinite(slope):
                raise FloatingPointError('nonfinite line-search sample')
        except FloatingPointError:
            if cache:
                invalid_alphas.append(alpha)
            logger.info(
                self, 'NLCG line alpha = %.8g method = %s is nonfinite',
                alpha, method)
            return None
        except RuntimeError as error:
            if str(error) != (
                    'fixed-electron Fermi response is numerically singular'):
                raise
            if cache:
                invalid_alphas.append(alpha)
            logger.info(
                self, 'NLCG line alpha = %.8g method = %s has a singular '
                'fixed-N Fermi response', alpha, method)
            return None

        sample = _LineSample(
            alpha, trial, value, gradient, slope, method,
            residual_metric(trial))
        if cache:
            samples.append(sample)
            samples.sort(key=lambda item: item.alpha)
        logger.info(
            self, 'NLCG line alpha = %.8g method = %s objective = %.12g '
            'slope = %.6g residual = %.6g line residual = %.6g',
            alpha, method, value, slope, trial.residual_rms,
            _sample_residual(sample))
        return sample

    def refine_residual_growth_boundary():
        bounded = _bounded_residual_growth_sample(
            samples, origin_sample, residual_growth_limit)
        if bounded is None or evaluations >= evaluation_limit:
            return bounded, False
        alpha = _residual_growth_boundary_step(
            samples, origin_sample, residual_growth_limit)
        if alpha is None:
            return bounded, False
        # On a locally descending line, objective gain is approximately
        # proportional to alpha.  Spend another Fock evaluation only when
        # the interpolated extension can break even against the evaluations
        # already invested in this attempt.
        break_even_ratio = 1.0+1.0/max(evaluations, 1)
        if (bounded.alpha <= 0.0 or
                alpha <= break_even_ratio*bounded.alpha):
            return bounded, False
        trial = evaluate_alpha(alpha, 'residual-growth-boundary')
        if trial is None:
            return bounded, False
        refined = _bounded_residual_growth_sample(
            samples, origin_sample, residual_growth_limit)
        if refined is None:
            return bounded, False
        return refined, refined is not bounded

    while evaluations < evaluation_limit:
        if residual_growth_step is not None:
            alpha = residual_growth_step
            residual_growth_step *= 0.5
            method = 'residual-growth-contract'
        elif accept_bounded_residual:
            alpha = bounded_residual_step
            method = ('initial' if evaluations == 0 else
                      'null-contract')
            bounded_residual_step *= 0.5
        else:
            flat = (
                refinements >= LINE_SEARCH_FLAT_REFINEMENTS and
                not strict_minimization)
            alpha, method = _line_search_proposal(
                samples, invalid_alphas, initial_step,
                consistency=consistency, flat=flat)
            if (not strict_minimization and consistency_probe_done and
                    method is not None and
                    method.startswith('consistency-')):
                alpha = None
        if alpha is None:
            if strict_minimization:
                stopped_reason = (
                    'strict line refinement reached floating-point spacing')
            break
        if not _line_candidate_is_new(
                alpha, samples, invalid_alphas,
                allow_signed=alpha < 0.0):
            # Specialized restoration/contract schedules can return to a
            # previously sampled slot after an expansion overshoots.  Keep
            # contracting from the already-updated schedule without ever
            # reevaluating that absolute-origin state.
            if strict_minimization:
                stopped_reason = (
                    'strict line refinement reached floating-point spacing')
                break
            continue

        old_best = _best_line_sample(samples)
        old_residual = min(_sample_residual(sample) for sample in samples)
        old_slope = min(abs(sample.slope) for sample in samples)
        previous_interval = refinement_interval
        sample = evaluate_alpha(alpha, method)
        if sample is None:
            continue
        if method.startswith('consistency-'):
            consistency_probe_done = True

        if (previous_interval is not None and
                previous_interval[0] < alpha < previous_interval[1]):
            refinements += 1
        active_interval = _active_line_interval(samples, invalid_alphas)
        refinement_interval = _minimum_line_interval(
            samples, invalid_alphas)
        consistency = _line_consistency(
            samples, active_interval, origin_sample.value)
        objective_band = _line_objective_band(
            origin_sample.value, consistency)

        best = _best_line_sample(samples)
        energy_progress = best.value < old_best.value-objective_band
        residual_progress = (
            sample.value <= best.value+objective_band and
            _sample_residual(sample) <=
            (1.0-LINE_SEARCH_RESIDUAL_REDUCTION)*old_residual)
        slope_progress = (
            old_slope > np.finfo(float).tiny and
            abs(sample.slope) <=
            (1.0-LINE_SEARCH_PROGRESS_REDUCTION)*old_slope)
        if energy_progress or residual_progress or slope_progress:
            stagnation = 0
        else:
            stagnation += 1
        if active_interval is not None:
            logger.info(
                self, 'NLCG line bracket = [%.8g, %.8g] consistency = %.3g '
                'objective band = %.3g refinements = %d stagnation = %d',
                active_interval[0], active_interval[1], consistency,
                objective_band, refinements, stagnation)
        if strict_minimization:
            (strict_candidate, strict_interval,
             alpha_uncertainty, normalized_slope) = _strict_line_metrics(
                samples, origin_sample, objective_band, direction_norm)
            alpha_met = (
                alpha_uncertainty <= strict_tolerances[0])
            slope_met = normalized_slope <= strict_tolerances[1]
            if strict_interval is None:
                logger.info(
                    self, 'NLCG strict line minimum is not slope-bracketed; '
                    'normalized slope = %.6g (target %.6g)',
                    normalized_slope, strict_tolerances[1])
            else:
                logger.info(
                    self, 'NLCG strict slope bracket = [%.8g, %.8g], '
                    'alpha uncertainty = %.6g (%.6g%%, target %.6g%%), '
                    'normalized slope = %.6g (target %.6g), '
                    'criteria = alpha:%s slope:%s',
                    strict_interval[0], strict_interval[1],
                    alpha_uncertainty, 100.0*alpha_uncertainty,
                    100.0*strict_tolerances[0], normalized_slope,
                    strict_tolerances[1],
                    'met' if alpha_met else 'open',
                    'met' if slope_met else 'open')

        acceptance_consistency = (
            min(objective_band, bounded_restoration_band)
            if accept_bounded_residual else objective_band)
        if (sample.state.residual_rms <= self.conv_tol and
                sample.value <=
                origin_sample.value+acceptance_consistency):
            return _LineSearchResult(
                sample, True, evaluations, 'converged line sample',
                restoration=not _line_improves(
                    sample.value, origin_sample.value),
                consistency=acceptance_consistency)
        bounded_consistency = acceptance_consistency
        bounded_residual_improves = (
            accept_bounded_residual and
            _sample_residual(sample) <=
            (1.0-LINE_SEARCH_RESIDUAL_REDUCTION) *
            _sample_residual(origin_sample))
        if bounded_residual_improves:
            if (sample.value <=
                    origin_sample.value+bounded_consistency):
                return _LineSearchResult(
                    sample, False, evaluations,
                    'Fermi-null residual cleanup', restoration=True,
                    consistency=bounded_consistency)
            remaining_excess = (
                sample.value-origin_sample.value-bounded_consistency)
            tangent_model_trustworthy = (
                consistency <=
                LINE_SEARCH_NULL_TANGENT_CONSISTENCY_FACTOR *
                bounded_consistency)
            exact_direction = (
                _scaled_exact_direction(self, origin_gradient, direction)
                if tangent_model_trustworthy else None)
            if not tangent_model_trustworthy:
                logger.info(
                    self, 'NLCG skipping null tangent correction: '
                    'energy/slope inconsistency %.3g exceeds trust limit '
                    '%.3g', consistency,
                    LINE_SEARCH_NULL_TANGENT_CONSISTENCY_FACTOR *
                    bounded_consistency)
            exact_slope = (
                np.nan if exact_direction is None else
                self._inner(origin_gradient, exact_direction))
            tangent_denominator = -alpha*exact_slope
            correction_points = [(0.0, remaining_excess)]
            correction = (
                1.25*remaining_excess/tangent_denominator
                if (np.isfinite(tangent_denominator) and
                    tangent_denominator > np.finfo(float).tiny)
                else np.nan)
            for correction_index in range(
                    LINE_SEARCH_TANGENT_REFINEMENTS):
                if (exact_direction is None or
                        not np.isfinite(correction) or
                        correction <= correction_points[-1][0] or
                        correction > 0.25):
                    break
                corrected_direction = [
                    residual+correction*exact
                    for residual, exact in zip(direction, exact_direction)]
                method = (
                    'null-tangent' if correction_index == 0 else
                    'null-tangent-refine')
                logger.info(
                    self, 'NLCG %s null tangent correction = %.6g '
                    'for objective excess %.3g',
                    ('applying' if correction_index == 0 else 'refining'),
                    correction, remaining_excess)
                corrected = evaluate_alpha(
                    alpha, method, corrected_direction, cache=False)
                if (corrected is None or
                        _sample_residual(corrected) >
                        (1.0-LINE_SEARCH_RESIDUAL_REDUCTION) *
                        _sample_residual(origin_sample)):
                    break
                if (corrected.value <=
                        origin_sample.value+bounded_consistency):
                    return _LineSearchResult(
                        corrected, False, evaluations,
                        'energy-corrected Fermi-null residual cleanup',
                        restoration=True,
                        consistency=bounded_consistency)
                remaining_excess = (
                    corrected.value-origin_sample.value-
                    bounded_consistency)
                correction_points.append((correction, remaining_excess))
                previous_correction, previous_excess = (
                    correction_points[-2])
                excess_change = remaining_excess-previous_excess
                candidate = np.nan
                if (np.isfinite(excess_change) and
                        abs(excess_change) > np.finfo(float).tiny):
                    candidate = correction - remaining_excess* (
                        correction-previous_correction) / excess_change
                minimum_growth = correction + max(
                    LINE_SEARCH_ALPHA_RTOL,
                    0.1*(correction-previous_correction))
                if not np.isfinite(candidate) or candidate < minimum_growth:
                    candidate = correction + max(
                        correction-previous_correction,
                        LINE_SEARCH_ALPHA_RTOL)
                correction = min(0.25, candidate)
        if (accept_bounded_residual and not bounded_residual_improves and
                _sample_residual(sample) <
                _sample_residual(origin_sample) and
                stagnation < LINE_SEARCH_STAGNATION_LIMIT):
            # An improving probe may guide the trust-radius expansion even
            # when it is not itself energy-admissible.  Once the residual
            # reduction reaches the acceptance threshold, the tangent
            # correction above must still bring the accepted state inside
            # the frozen restoration band.
            expanded_alpha = 2.0*sample.alpha
            if _line_candidate_is_new(
                    expanded_alpha, samples, invalid_alphas):
                bounded_residual_step = expanded_alpha
                continue
        if (accept_bounded_residual and
                method == 'null-contract' and
                sample.value >
                origin_sample.value+_line_roundoff(origin_sample.value)):
            stagnation = 0
        if (accept_bounded_residual and
                stagnation >= LINE_SEARCH_STAGNATION_LIMIT):
            stopped_reason = 'null-restoration stagnation'
            break
        objective_priority = False
        guarded_residual_growth = (
            bound_residual_growth and self.nelec is not None and
            origin.residual_rms <= LINE_SEARCH_PARETO_ACTIVE_RESIDUAL and
            _sample_residual(sample) >
            residual_growth_limit * _sample_residual(origin_sample))
        if guarded_residual_growth:
            bounded = _bounded_residual_growth_sample(
                samples, origin_sample, residual_growth_limit)
            objective_reference = (
                bounded if bounded is not None else origin_sample)
            objective_best = _best_line_sample(samples)
            objective_priority = (
                objective_best.value <
                objective_reference.value-objective_band)
            if objective_priority:
                logger.info(
                    self, 'NLCG retaining residual-growing line sample: '
                    'objective advantage = %.6g exceeds flat band %.3g',
                    objective_reference.value-objective_best.value,
                    objective_band)
                guarded_residual_growth = False
                residual_growth_step = None
            else:
                bounded, refined_boundary = (
                    refine_residual_growth_boundary())
        if guarded_residual_growth:
            if bounded is not None:
                return _LineSearchResult(
                    bounded, False, evaluations,
                    ('occupation residual-growth boundary refinement'
                     if refined_boundary else
                     'occupation residual-growth truncation'),
                    consistency=objective_band)
            residual_growth_step = 0.5*sample.alpha
            continue
        if (method == 'residual-growth-contract' and
                not objective_priority):
            bounded, refined_boundary = (
                refine_residual_growth_boundary())
            if bounded is not None:
                return _LineSearchResult(
                    bounded, False, evaluations,
                    ('occupation residual-growth boundary refinement'
                     if refined_boundary else
                     'occupation residual-growth contraction'),
                    consistency=objective_band)
        pareto = (
            _pareto_line_sample(samples, origin_sample)
            if (allow_restoration or require_residual_improvement)
            else None)
        if (require_residual_improvement and evaluations >= 2 and
                pareto is None and
                origin_sample.value-best.value <= objective_band):
            return _LineSearchResult(
                None, False, evaluations,
                'Pulay direction did not reduce the residual',
                consistency=objective_band)
        if (origin.residual_rms <= LINE_SEARCH_PARETO_ACTIVE_RESIDUAL and
                pareto is not None and sample is not pareto and
                sample.alpha > pareto.alpha and
                best.value >= pareto.value-objective_band and
                _sample_residual(sample) >
                _sample_residual(pareto) /
                (1.0-LINE_SEARCH_PARETO_RESIDUAL_REDUCTION)):
            return _LineSearchResult(
                pareto, False, evaluations,
                'residual-descent truncation',
                consistency=objective_band)
        if strict_minimization:
            (minimum_candidate, slope_interval,
             alpha_uncertainty, normalized_slope) = _strict_line_metrics(
                samples, origin_sample, objective_band, direction_norm)
            resolved_minimum = (
                minimum_candidate is not None and
                origin_sample.value-minimum_candidate.value >
                objective_band and
                alpha_uncertainty <= strict_tolerances[0] and
                normalized_slope <= strict_tolerances[1])
        else:
            minimum_candidate = best
            slope_interval = None
            alpha_uncertainty = np.nan
            normalized_slope = abs(best.slope)/max(
                direction_norm, np.finfo(float).tiny)
            resolved_minimum = (
                origin_sample.value-best.value > objective_band and
                abs(best.slope) <= LINE_SEARCH_SLOPE_RATIO*abs(slope0) and
                _sample_has_positive_curvature(samples, best))
        if resolved_minimum:
            if (pareto is not None and
                    pareto.value <= minimum_candidate.value+objective_band and
                    _sample_residual(pareto) <
                    _sample_residual(minimum_candidate)):
                return _LineSearchResult(
                    pareto, False, evaluations,
                    'residual-descent truncation',
                    consistency=objective_band)
            return _LineSearchResult(
                minimum_candidate, True, evaluations,
                ('resolved strict line minimum' if strict_minimization else
                 'resolved line minimum'),
                consistency=objective_band,
                slope_interval=slope_interval,
                alpha_relative_uncertainty=alpha_uncertainty,
                normalized_slope=normalized_slope)

        if (refinements >= LINE_SEARCH_FLAT_REFINEMENTS and
                not strict_minimization):
            permit_early_residual = (
                allow_restoration or
                origin.residual_rms > LINE_SEARCH_PARETO_ACTIVE_RESIDUAL)
            if permit_early_residual:
                equivalent = _restoration_line_sample(
                    samples, origin_sample, objective_band)
                if (equivalent is not None and
                        equivalent.value <= best.value+objective_band and
                        _residual_improves(equivalent, origin_sample)):
                    if allow_restoration:
                        break
                    return _LineSearchResult(
                        equivalent, False, evaluations,
                        'coarse residual-descent truncation',
                        restoration=True, consistency=objective_band)
            if allow_restoration and not slope_progress:
                # Two flat bracket refinements are enough to establish the
                # fixed 1e-8-Ha restoration regime.  If no positive sample
                # already reduces the residual, yield to the reflected
                # residual probe below instead of spending the entire line
                # budget repeatedly halving toward the origin.
                break
            if (stagnation >= LINE_SEARCH_STAGNATION_LIMIT and
                    not consistency_probe_done):
                continue
            if stagnation >= LINE_SEARCH_STAGNATION_LIMIT:
                break

    if evaluations >= evaluation_limit:
        stopped_reason = 'line-search evaluation limit'

    if (allow_restoration and
            refinements >= LINE_SEARCH_FLAT_REFINEMENTS and
            origin.residual_rms > self.conv_tol and
            evaluations < evaluation_limit):
        restoration_band = (
            min(objective_band, bounded_restoration_band)
            if accept_bounded_residual else objective_band)
        positive_secant = True
        expansions = 0
        while (positive_secant and
               evaluations < evaluation_limit and
               expansions < LINE_SEARCH_RESTORATION_EXPANSION_LIMIT):
            positive = [
                sample for sample in samples
                if (sample.alpha > 0.0 and
                    sample.value <= origin_sample.value+restoration_band)
            ]
            if not positive:
                break
            anchor = min(positive, key=lambda sample: (
                _sample_residual(sample), abs(sample.slope), sample.alpha))
            if _sample_residual(anchor) >= _sample_residual(origin_sample):
                break
            fitted = _residual_secant_step(
                self, origin_sample, anchor)
            if fitted is None:
                break
            upper = 2.0*anchor.alpha
            expanding = fitted > upper
            candidate_alpha = upper if expanding else fitted
            if not _line_candidate_is_new(
                    candidate_alpha, samples, invalid_alphas):
                break

            old_residual = _sample_residual(anchor)
            trial = evaluate_alpha(
                candidate_alpha,
                'residual-expand' if expanding else 'residual-secant')
            if (trial is None or
                    trial.value > origin_sample.value+restoration_band):
                break
            if _sample_residual(trial) >= old_residual:
                alphas = np.asarray([0.0, anchor.alpha, trial.alpha])
                values = np.asarray([
                    origin.residual_rms**2,
                    _sample_residual(anchor)**2,
                    _sample_residual(trial)**2])
                lower, upper = sorted((anchor.alpha, trial.alpha))
                quadratic, unused_curvature = _convex_quadratic_step(
                    alphas, values, (lower, upper))
                if (quadratic is not None and
                        evaluations < evaluation_limit and
                        _line_candidate_is_new(
                            quadratic, samples, invalid_alphas)):
                    fitted_trial = evaluate_alpha(
                        quadratic, 'residual-quadratic')
                    if (fitted_trial is not None and
                            fitted_trial.value >
                            origin_sample.value+restoration_band):
                        fitted_trial = None
                break
            if not expanding:
                if _residual_improves(trial, origin_sample):
                    break
                expansions += 1
                continue
            expansions += 1

        equivalent = _restoration_line_sample(
            samples, origin_sample, restoration_band)
        if (positive_secant and equivalent is not None and
                _residual_improves(equivalent, origin_sample)):
            return _LineSearchResult(
                equivalent, False, evaluations,
                'positive residual restoration',
                restoration=True, consistency=restoration_band)

        # Reflection is specifically the fallback when no admissible
        # positive sample helps.  The matching positive slot is still useful
        # for the residual-squared fit even when its objective lies outside
        # the restoration band; only the accepted sample must satisfy the
        # band.
        positive = [
            sample for sample in samples if sample.alpha > 0.0]
        if positive:
            positive_sample = min(
                positive,
                key=lambda sample: abs(sample.alpha-initial_step))
            reflected_alpha = -positive_sample.alpha
            reflected = None
            if _line_candidate_is_new(
                    reflected_alpha, samples, invalid_alphas,
                    allow_signed=True):
                reflected = evaluate_alpha(
                    reflected_alpha, 'residual-reflect')
            if reflected is not None:
                residual_alphas = np.asarray([
                    reflected_alpha, 0.0, positive_sample.alpha])
                residual_values = np.asarray([
                    reflected.state.residual_rms**2,
                    origin.residual_rms**2,
                    positive_sample.state.residual_rms**2])
                fitted, unused_curvature = _convex_quadratic_step(
                    residual_alphas, residual_values,
                    (reflected_alpha, positive_sample.alpha))
                if (fitted is not None and
                        evaluations < evaluation_limit and
                        _line_candidate_is_new(
                            fitted, samples, invalid_alphas,
                            allow_signed=True)):
                    fitted_sample = evaluate_alpha(
                        fitted, 'residual-quadratic')
                    if fitted_sample is not None:
                        consistency = max(
                            consistency,
                            _line_roundoff(origin_sample.value))

                equivalent = _restoration_line_sample(
                    samples, origin_sample, restoration_band)
                if (equivalent is not None and
                        _residual_improves(equivalent, origin_sample)):
                    return _LineSearchResult(
                        equivalent, False, evaluations,
                        'reflected residual restoration',
                        restoration=True, consistency=restoration_band)

    acceptance_consistency = (
        min(objective_band, bounded_restoration_band)
        if accept_bounded_residual else objective_band)
    candidate = _inexact_line_sample(
        samples, origin_sample, acceptance_consistency)
    restoration_candidate = _restoration_line_sample(
        samples, origin_sample, acceptance_consistency)
    if (candidate is not None and
            candidate.value <
            origin_sample.value-acceptance_consistency):
        if strict_minimization:
            (slope_interval, alpha_uncertainty,
             normalized_slope) = _strict_sample_metrics(
                samples, candidate, direction_norm)
        else:
            slope_interval = None
            alpha_uncertainty = np.nan
            normalized_slope = abs(candidate.slope)/max(
                direction_norm, np.finfo(float).tiny)
        return _LineSearchResult(
            candidate, False, evaluations, stopped_reason,
            consistency=acceptance_consistency,
            slope_interval=slope_interval,
            alpha_relative_uncertainty=alpha_uncertainty,
            normalized_slope=normalized_slope)
    if (restoration_candidate is not None and
            refinements >= LINE_SEARCH_FLAT_REFINEMENTS and
            _residual_improves(restoration_candidate, origin_sample)):
        return _LineSearchResult(
            restoration_candidate, False, evaluations,
            'flat-objective residual restoration',
            restoration=True, consistency=acceptance_consistency)
    return _LineSearchResult(
        None, False, evaluations, 'no acceptable objective or residual sample',
        consistency=acceptance_consistency)


def _direction_norm(self, direction):
    squared = self._inner(direction, direction)
    if not np.isfinite(squared) or squared <= np.finfo(float).tiny:
        return 0.0
    return float(np.sqrt(squared))


def _descent_metrics(self, gradient, direction):
    gradient_norm = _direction_norm(self, gradient)
    direction_norm = _direction_norm(self, direction)
    slope = self._inner(gradient, direction)
    scale = gradient_norm*direction_norm
    cosine = slope/scale if scale > np.finfo(float).tiny else np.nan
    descent = (
        np.isfinite(slope) and np.isfinite(cosine) and
        slope < -NLCG_DESCENT_COSINE*scale)
    return descent, float(slope), float(cosine), direction_norm


def _orient_pulay_direction(self, gradient, direction):
    """Reflect a Pulay root direction when that makes it descending."""
    metrics = _descent_metrics(self, gradient, direction)
    if not metrics[0] and np.isfinite(metrics[1]) and metrics[1] > 0.0:
        reflected = [-value for value in direction]
        reflected_metrics = _descent_metrics(self, gradient, reflected)
        if reflected_metrics[0]:
            return reflected, reflected_metrics, True
    return direction, metrics, False


def _scaled_exact_direction(self, exact_gradient, residual_direction):
    exact_norm = _direction_norm(self, exact_gradient)
    if exact_norm <= np.finfo(float).tiny:
        return None
    target_norm = _direction_norm(self, residual_direction)
    if target_norm <= np.finfo(float).tiny:
        target_norm = 1.0
    scale = target_norm/exact_norm
    return [-scale*gradient for gradient in exact_gradient]


def _tangent_corrected_direction(self, exact_gradient, direction, alpha,
                                 objective_excess):
    """Add a small descent component that keeps a null step on-energy."""
    if (not np.isfinite(alpha) or alpha <= 0.0 or
            not np.isfinite(objective_excess) or objective_excess <= 0.0):
        return None, None
    exact_direction = _scaled_exact_direction(
        self, exact_gradient, direction)
    if exact_direction is None:
        return None, None
    exact_slope = self._inner(exact_gradient, exact_direction)
    denominator = -alpha*exact_slope
    if (not np.isfinite(denominator) or
            denominator <= np.finfo(float).tiny):
        return None, None
    correction = 1.25*objective_excess/denominator
    if (not np.isfinite(correction) or correction <= 0.0 or
            correction > 0.25):
        return None, None
    corrected = [
        residual+correction*exact
        for residual, exact in zip(direction, exact_direction)]
    if _direction_norm(self, corrected) <= np.finfo(float).tiny:
        return None, None
    return corrected, float(correction)


def _orbital_rotation_data(self, state, history):
    """Build a safeguarded L-BFGS direction on the fixed-spectrum manifold."""
    gradients = []
    preconditioned = []
    preconditioners = []
    for eig, coeff, occ, fock in zip(
            state.eig, state.coeff, state.occ, state.fock):
        fock_eigenbasis = coeff.conj().T @ fock @ coeff
        occupation_difference = occ[:, None]-occ[None, :]
        gap = cp.abs(eig[:, None]-eig[None, :])
        denominator = cp.maximum(
            gap+NLCG_ORBITAL_LEVEL_SHIFT*cp.abs(occupation_difference),
            1e-12)
        active = cp.abs(occupation_difference) > 1e-12
        gradient = cp.where(
            active, occupation_difference*fock_eigenbasis, 0.)
        gradient = .5*(gradient-gradient.conj().T)
        inverse_gradient = gradient/denominator
        gradients.append(coeff @ gradient @ coeff.conj().T)
        preconditioned.append(
            coeff @ inverse_gradient @ coeff.conj().T)
        preconditioners.append((active, denominator))

    true_gradient = [-value for value in gradients]
    if (history.previous_gradient is not None and
            history.previous_step is not None):
        difference = [
            new-old for new, old in zip(
                true_gradient, history.previous_gradient)]
        curvature = self._inner(history.previous_step, difference)
        scale = np.sqrt(max(
            0., self._inner(history.previous_step, history.previous_step) *
            self._inner(difference, difference)))
        if (np.isfinite(curvature) and
                curvature > 1e-10*max(scale, np.finfo(float).tiny)):
            history.pairs.append((
                [value.copy() for value in history.previous_step],
                [value.copy() for value in difference],
                1./curvature))
            history.pairs = history.pairs[-NLCG_ORBITAL_LBFGS_SPACE:]
        else:
            history.pairs = []

    method = 'orbital-steepest'
    if history.pairs:
        work = [value.copy() for value in true_gradient]
        first_pass = []
        for step, difference, inverse_curvature in reversed(history.pairs):
            coefficient = inverse_curvature*self._inner(step, work)
            first_pass.append(coefficient)
            work = [
                value-coefficient*delta
                for value, delta in zip(work, difference)]

        inverse_work = []
        for coeff, (active, denominator), value in zip(
                state.coeff, preconditioners, work):
            transformed = coeff.conj().T @ value @ coeff
            transformed = cp.where(active, transformed/denominator, 0.)
            transformed = .5*(transformed-transformed.conj().T)
            inverse_work.append(coeff @ transformed @ coeff.conj().T)
        for ((step, difference, inverse_curvature), coefficient) in zip(
                history.pairs, reversed(first_pass)):
            correction = inverse_curvature*self._inner(
                difference, inverse_work)
            inverse_work = [
                value+(coefficient-correction)*delta
                for value, delta in zip(inverse_work, step)]
        direction = [-value for value in inverse_work]
        method = 'orbital-lbfgs-%d' % len(history.pairs)
    else:
        direction = [value.copy() for value in preconditioned]

    gradient_metric = self._inner(gradients, preconditioned)
    direction_metric = self._inner(gradients, direction)
    direction_norm = _direction_norm(self, direction)
    descent_floor = (
        NLCG_DESCENT_COSINE*np.sqrt(max(0., gradient_metric)) *
        max(direction_norm, np.finfo(float).tiny))
    if (not np.isfinite(direction_metric) or
            direction_metric <= descent_floor):
        history.pairs = []
        direction = [value.copy() for value in preconditioned]
        direction_metric = gradient_metric
        direction_norm = _direction_norm(self, direction)
        method = 'orbital-descent-restart'
    if (not np.isfinite(direction_norm) or
            direction_norm <= np.finfo(float).tiny):
        return None

    generators = []
    for coeff, value in zip(state.coeff, direction):
        generator = coeff.conj().T @ value @ coeff
        generators.append(.5*(generator-generator.conj().T))
    generator_norm = _direction_norm(self, generators)
    if generator_norm <= np.finfo(float).tiny:
        return None
    return {
        'gradients': gradients,
        'true_gradient': true_gradient,
        'direction': direction,
        'generators': generators,
        'norm': generator_norm,
        'gradient_metric': float(gradient_metric),
        'direction_metric': float(direction_metric),
        'method': method,
    }


def _orbital_hamiltonians(state, generators, alpha):
    """Apply an absolute Cayley rotation while preserving H eigenvalues."""
    hamiltonians = []
    for eig, coeff, generator in zip(
            state.eig, state.coeff, generators):
        identity = cp.eye(generator.shape[0], dtype=generator.dtype)
        rotation = cp.linalg.solve(
            identity-.5*alpha*generator,
            identity+.5*alpha*generator)
        rotated_coeff = coeff @ rotation
        value = ((rotated_coeff*eig[None, :]) @
                 rotated_coeff.conj().T)
        hamiltonians.append(.5*(value+value.conj().T))
    return hamiltonians


def _orbital_quadratic_step(origin_value, samples, best):
    lower = [(0., origin_value)] + [
        (sample.alpha, sample.value) for sample in samples
        if sample.alpha < best.alpha]
    upper = [
        (sample.alpha, sample.value) for sample in samples
        if sample.alpha > best.alpha]
    if not lower or not upper:
        return None
    lower_alpha, lower_value = max(lower, key=lambda item: item[0])
    upper_alpha, upper_value = min(upper, key=lambda item: item[0])
    if not (best.value < lower_value and best.value < upper_value):
        return None
    alphas = np.asarray([lower_alpha, best.alpha, upper_alpha])
    energies = np.asarray([lower_value, best.value, upper_value])
    scale = max(float(np.ptp(energies)), np.finfo(float).tiny)
    try:
        polynomial = np.polynomial.Polynomial.fit(
            alphas, (energies-np.min(energies))/scale, 2)
        roots = polynomial.deriv().roots()
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None
    candidates = [
        float(root.real) for root in roots
        if (abs(root.imag) <= 1e-10 and
            lower_alpha < root.real < upper_alpha and
            polynomial.deriv(2)(root.real) > 0.)]
    if not candidates:
        return None
    return min(candidates, key=lambda alpha: abs(alpha-best.alpha))


def _orbital_line_search(self, origin, evaluate, objective, history,
                         equivalence_band=0.0):
    """Minimize the objective on a fixed-spectrum unitary orbital path."""
    data = _orbital_rotation_data(self, origin, history)
    if data is None:
        history.clear()
        return _LineSearchResult(
            None, False, 0, 'orbital rotation direction vanished')
    direction_norm = data['norm']
    if history.rotation_seed is None:
        alpha = NLCG_ORBITAL_INITIAL_ALPHA
    else:
        alpha = history.rotation_seed/direction_norm
    alpha_cap = NLCG_ORBITAL_MAX_ROTATION/direction_norm
    alpha = min(alpha, alpha_cap)
    origin_value = objective(origin)
    maximum = min(
        self.nlcg_max_line_search_evaluations,
        NLCG_ORBITAL_LINE_EVALUATIONS)
    samples = []
    invalid_alphas = []
    evaluations = 0
    bracketed = False

    logger.info(
        self, 'NLCG orbital direction = %s norm = %.6g rotation = %.6g '
        'gradient metric = %.6g direction metric = %.6g',
        data['method'], direction_norm, alpha*direction_norm,
        data['gradient_metric'], data['direction_metric'])

    while evaluations < maximum:
        if not _line_candidate_is_new(
                alpha, samples, invalid_alphas):
            break
        try:
            state = evaluate(_orbital_hamiltonians(
                origin, data['generators'], alpha))
        except FloatingPointError:
            invalid_alphas.append(float(alpha))
            alpha *= .5
            continue
        evaluations += 1
        value = objective(state)
        if not np.isfinite(value):
            invalid_alphas.append(float(alpha))
            alpha *= .5
            continue
        sample = _LineSample(
            alpha, state, value, None, np.nan, 'orbital-sample',
            state.residual_rms)
        samples.append(sample)
        logger.info(
            self, 'NLCG orbital alpha = %.8g rotation = %.6g '
            'objective = %.12g residual = %.6g',
            alpha, alpha*direction_norm, value, state.residual_rms)

        improving = [
            item for item in samples
            if _line_improves(item.value, origin_value)]
        if not improving:
            alpha *= .5
            continue
        best = min(improving, key=lambda item: item.value)
        above = [item for item in samples if item.alpha > best.alpha]
        if above:
            upper = min(above, key=lambda item: item.alpha)
            if upper.value > best.value:
                bracketed = True
                break
            alpha = min(alpha_cap, 2.*upper.alpha)
        else:
            next_alpha = min(alpha_cap, 2.*best.alpha)
            if _line_alpha_close(next_alpha, best.alpha):
                break
            alpha = next_alpha

    improving = [
        sample for sample in samples
        if _line_improves(sample.value, origin_value)]
    if not improving:
        history.clear()
        return _LineSearchResult(
            None, False, evaluations,
            'orbital rotation line found no objective decrease')
    best = min(improving, key=lambda sample: sample.value)
    fitted_alpha = _orbital_quadratic_step(
        origin_value, samples, best)
    if (fitted_alpha is not None and evaluations < maximum and
            _line_candidate_is_new(
                fitted_alpha, samples, invalid_alphas)):
        try:
            fitted_state = evaluate(_orbital_hamiltonians(
                origin, data['generators'], fitted_alpha))
        except FloatingPointError:
            fitted_state = None
        if fitted_state is not None:
            evaluations += 1
            fitted_value = objective(fitted_state)
            if np.isfinite(fitted_value):
                fitted = _LineSample(
                    fitted_alpha, fitted_state, fitted_value, None,
                    np.nan, 'orbital-quadratic',
                    fitted_state.residual_rms)
                samples.append(fitted)
                logger.info(
                    self, 'NLCG orbital fit alpha = %.8g rotation = %.6g '
                    'objective = %.12g residual = %.6g',
                    fitted_alpha, fitted_alpha*direction_norm,
                    fitted_value, fitted_state.residual_rms)
                if fitted.value < best.value:
                    best = fitted

    residual_tiebreak = False
    if np.isfinite(equivalence_band) and equivalence_band > 0.0:
        valid_improving = [
            sample for sample in samples
            if _line_improves(sample.value, origin_value)]
        minimum_value = min(
            sample.value for sample in valid_improving)
        equivalent = [
            sample for sample in valid_improving
            if sample.value <= minimum_value+equivalence_band]
        if equivalent:
            residual_best = min(
                equivalent,
                key=lambda sample: (
                    _sample_residual(sample), sample.value,
                    abs(sample.alpha)))
            if residual_best is not best:
                best = residual_best
                residual_tiebreak = True
                logger.info(
                    self, 'NLCG orbital energy-equivalent residual '
                    'tie-break: objective = %.12g residual = %.6g, '
                    'band = %.6g', best.value,
                    _sample_residual(best), equivalence_band)

    gradient = objective_gradient(self, best.state, True)
    actual_direction = [
        (new-old)/best.alpha
        for old, new in zip(origin.h, best.state.h)]
    best.gradient = gradient
    best.slope = self._inner(gradient, actual_direction)
    history.previous_gradient = [
        value.copy() for value in data['true_gradient']]
    history.previous_step = [
        best.alpha*value for value in data['direction']]
    history.rotation_seed = min(
        NLCG_ORBITAL_MAX_ROTATION,
        2.*best.alpha*direction_norm)
    reason = (
        'resolved orbital rotation minimum' if bracketed else
        'inexact orbital rotation minimum')
    if residual_tiebreak:
        reason += ' with energy-equivalent residual tie-break'
    return _LineSearchResult(
        best, bracketed, evaluations, reason,
        consistency=_line_roundoff(origin_value))


def _occupation_preconditioned_direction(self, state, energy_scale=1.0):
    """Precondition the residual on finite Fermi-response blocks.

    Zero-response blocks do not change the thermodynamic objective to first
    order.  They are handled separately by the bounded null-residual
    restoration, so retaining them here couples a well-scaled occupation
    step to an uncontrolled, objective-invisible Hamiltonian displacement.
    """
    required = ('eig', 'coeff', 'occ', 'residual')
    if not all(hasattr(state, name) for name in required):
        return None
    if not np.isfinite(energy_scale) or energy_scale <= 0.0:
        return None
    direction = []
    for eig, coeff, occ, residual in zip(
            state.eig, state.coeff, state.occ, state.residual):
        gamma = self.beta*(eig-state.mu)
        response = -self.beta*fermi_divided_difference(gamma, occ)
        active = response > NLCG_NULL_RESPONSE_TOL
        scale = active / (1.0+energy_scale*response)
        residual_eigenbasis = coeff.conj().T @ (0.5*residual) @ coeff
        value = coeff @ (scale*residual_eigenbasis) @ coeff.conj().T
        direction.append(0.5*(value+value.conj().T))
    if getattr(self, 'nelec', None) is not None:
        gauge = self._trace_mean(direction)
        direction = [
            value-gauge*identity
            for value, identity in zip(direction, self.identity)]
    if _direction_norm(self, direction) <= np.finfo(float).tiny:
        return None
    return direction


def _occupation_direction_data(
        self, state, exact_gradient, residual_direction, sharp_fixed_n):
    """Build the occupation preconditioner and its local response scale."""
    residual_metric = self._inner(residual_direction, residual_direction)
    residual_slope = self._inner(exact_gradient, residual_direction)
    active_response = (
        -residual_slope/residual_metric
        if (np.isfinite(residual_metric) and
            np.isfinite(residual_slope) and
            residual_metric > np.finfo(float).tiny)
        else np.inf)
    energy_scale = 1.0
    if sharp_fixed_n:
        if (np.isfinite(active_response) and
                active_response > np.finfo(float).tiny):
            energy_scale = np.clip(
                NLCG_OCCUPATION_PRECONDITIONER_TARGET/active_response,
                NLCG_OCCUPATION_PRECONDITIONER_MIN_ENERGY,
                NLCG_OCCUPATION_PRECONDITIONER_MAX_ENERGY)
        else:
            energy_scale = NLCG_OCCUPATION_PRECONDITIONER_MAX_ENERGY
    direction = _occupation_preconditioned_direction(
        self, state, energy_scale)
    return direction, float(active_response), float(energy_scale)


def _occupation_direction_is_proactive(
        fixed_n, sharp_fixed_n, direction, residual_rms,
        orbital_phase_ready):
    """Use response-aware scaling whenever fixed-N coupling is active."""
    return bool(
        fixed_n and direction is not None and
        (residual_rms <= LINE_SEARCH_PARETO_ACTIVE_RESIDUAL or
         (sharp_fixed_n and not orbital_phase_ready)))


def _fermi_null_residual_direction(self, state):
    """Project the fixed-point residual onto zero Fermi-response blocks."""
    required = ('eig', 'coeff', 'occ', 'residual')
    if not all(hasattr(state, name) for name in required):
        return None
    direction = []
    for eig, coeff, occ, residual in zip(
            state.eig, state.coeff, state.occ, state.residual):
        gamma = self.beta*(eig-state.mu)
        response = -self.beta*fermi_divided_difference(gamma, occ)
        mask = response <= NLCG_NULL_RESPONSE_TOL
        residual_eigenbasis = coeff.conj().T @ residual @ coeff
        value = coeff @ (mask*residual_eigenbasis) @ coeff.conj().T
        direction.append(0.5*(value+value.conj().T))
    if getattr(self, 'nelec', None) is not None:
        gauge = self._trace_mean(direction)
        direction = [
            value-gauge*identity
            for value, identity in zip(direction, self.identity)]
    if _direction_norm(self, direction) <= np.finfo(float).tiny:
        return None
    return direction


def _fermi_active_residual_rms(self, state):
    """Return the RMS share carried by finite Fermi-response blocks."""
    null_direction = _fermi_null_residual_direction(self, state)
    if null_direction is None:
        return float(state.residual_rms)
    full_norm = _direction_norm(self, state.residual)
    if full_norm <= np.finfo(float).tiny:
        return 0.0
    active_norm = _direction_norm(self, [
        residual-null for residual, null in zip(
            state.residual, null_direction)])
    return float(state.residual_rms*active_norm/full_norm)


def _fermi_guarded_residual_rms(self, origin, state):
    """Scale active and full residual growth into one line metric.

    The occupation-preconditioned direction deliberately excludes the
    Fermi-null blocks.  A trial can therefore satisfy an active-only trust
    bound while creating a much larger null residual.  Express the full
    residual ratio on the active-residual scale so a single line-search
    bound limits both components.
    """
    origin_active = _fermi_active_residual_rms(self, origin)
    state_active = _fermi_active_residual_rms(self, state)
    tiny = np.finfo(float).tiny
    if not all(np.isfinite(value) for value in (
            origin_active, state_active,
            origin.residual_rms, state.residual_rms)):
        return np.inf
    if origin_active <= tiny or origin.residual_rms <= tiny:
        return max(float(state_active), float(state.residual_rms))
    full_on_active_scale = (
        origin_active*state.residual_rms/origin.residual_rms)
    return float(max(state_active, full_on_active_scale))


def _pulay_direction(self, state, adiis):
    """Build a regularized multidimensional fixed-point direction."""
    required = ('diis_pack', 'diis_unpack', '_sanitize_h')
    if not all(hasattr(self, name) for name in required):
        return None, 'Pulay transport is unavailable'
    try:
        # ``calculate_cycle`` gauge-aligns H and the fixed-N residual, while
        # the stored raw Fock matrix retains its arbitrary scalar gauge.
        # Extrapolate the aligned fixed-point target H+R so chemical-potential
        # shifts cannot pollute the physical Pulay direction.
        fixed_point = [
            h+residual for h, residual in zip(state.h, state.residual)]
        vector = self.diis_pack(fixed_point)
        error = self.diis_pack(state.residual, weight_errors=True)
        adiis.push_err_vec(error)
        adiis.push_vec(vector)
        vectors = adiis.get_num_vec()
        if vectors < NLCG_PULAY_MIN_VECTORS:
            return None, 'Pulay history has %d vectors' % vectors
        slots = list(getattr(adiis, '_bookkeep', range(vectors)))
        if len(slots) != vectors:
            raise ValueError('inconsistent Pulay history')
        gram = np.empty((vectors, vectors))
        for row, left_slot in enumerate(slots):
            left = cp.asarray(adiis.get_err_vec(left_slot))
            for column in range(row+1):
                right = cp.asarray(adiis.get_err_vec(slots[column]))
                value = float(cp.vdot(left, right).real.item())
                gram[row, column] = gram[column, row] = value
        scale = max(float(np.max(np.diag(gram))), np.finfo(float).tiny)
        ridge = NLCG_PULAY_REGULARIZATION*scale
        coefficients = None
        condition = np.inf
        ones = np.ones(vectors)
        for unused_regularization in range(6):
            regularized = gram+ridge*np.eye(vectors)
            condition = float(np.linalg.cond(regularized))
            inverse_ones = np.linalg.solve(regularized, ones)
            denominator = float(ones @ inverse_ones)
            if (np.isfinite(denominator) and
                    abs(denominator) > np.finfo(float).tiny):
                candidate = inverse_ones/denominator
                if (np.all(np.isfinite(candidate)) and
                        np.max(np.abs(candidate)) <=
                        NLCG_PULAY_MAX_COEFFICIENT):
                    coefficients = candidate
                    break
            ridge *= 10.0
        if coefficients is None:
            raise np.linalg.LinAlgError(
                'regularized Pulay coefficients are unstable')
        target = cp.zeros_like(vector)
        for coefficient, slot in zip(coefficients, slots):
            target += coefficient*cp.asarray(adiis.get_vec(slot))
        target = self.diis_unpack(target, state.h)
        target = self._sanitize_h(target)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        adiis.clear()
        return None, 'Pulay history was singular or nonfinite'
    direction = [new-old for old, new in zip(state.h, target)]
    if getattr(self, 'nelec', None) is not None:
        gauge = self._trace_mean(direction)
        direction = [
            value-gauge*identity
            for value, identity in zip(direction, self.identity)]
    if _direction_norm(self, direction) <= np.finfo(float).tiny:
        return None, 'Pulay direction vanished'
    return direction, (
        'regularized Pulay history has %d vectors, condition %.3g, '
        'max coefficient %.3g' % (
            vectors, condition, np.max(np.abs(coefficients))))


def _canonical_charge_proposal(history, current_nelec, current_error,
                               capacity):
    """Propose a bounded electron number for a canonical restoration."""
    distinct = []
    for nelec, error in history:
        if not np.isfinite(nelec) or not np.isfinite(error):
            continue
        if any(abs(nelec-old_nelec) <= LINE_SEARCH_ALPHA_RTOL*max(
                1.0, abs(nelec), abs(old_nelec))
               for old_nelec, unused_error in distinct):
            continue
        distinct.append((float(nelec), float(error)))

    negative = [sample for sample in distinct if sample[1] < 0.0]
    positive = [sample for sample in distinct if sample[1] > 0.0]
    candidate = None
    bracket = None
    if negative and positive:
        left, right = min(
            ((negative_sample, positive_sample)
             for negative_sample in negative
             for positive_sample in positive),
            key=lambda pair: abs(pair[0][0]-pair[1][0]))
        if left[0] > right[0]:
            left, right = right, left
        denominator = right[1]-left[1]
        if abs(denominator) > np.finfo(float).tiny:
            candidate = left[0]-left[1]*(right[0]-left[0])/denominator
            bracket = (left[0], right[0])
    elif len(distinct) >= 2:
        monotonic_pairs = []
        for index, left in enumerate(distinct[:-1]):
            for right in distinct[index+1:]:
                delta_nelec = right[0]-left[0]
                delta_error = right[1]-left[1]
                if (abs(delta_nelec) > np.finfo(float).tiny and
                        delta_error/delta_nelec > 0.0):
                    monotonic_pairs.append((left, right))
        if monotonic_pairs:
            left, right = max(
                monotonic_pairs,
                key=lambda pair: abs(pair[1][0]-pair[0][0]))
            denominator = right[1]-left[1]
            candidate = right[0]-right[1]*(
                right[0]-left[0])/denominator

    if candidate is None or not np.isfinite(candidate):
        candidate = current_nelec - (
            NLCG_CANONICAL_RESTORATION_RESPONSE*current_error)
    if bracket is not None:
        candidate = min(bracket[1], max(bracket[0], candidate))
    else:
        delta = candidate-current_nelec
        delta = min(
            NLCG_CANONICAL_RESTORATION_NELEC_STEP,
            max(-NLCG_CANONICAL_RESTORATION_NELEC_STEP, delta))
        candidate = current_nelec+delta
    margin = 64.0*np.finfo(float).eps*max(1.0, capacity)
    candidate = min(capacity-margin, max(margin, candidate))
    if abs(candidate-current_nelec) <= LINE_SEARCH_ALPHA_RTOL*max(
            1.0, abs(candidate), abs(current_nelec)):
        return None, bracket
    return float(candidate), bracket


def _canonical_restoration_damp(self, origin):
    coarse_tolerance = getattr(self, 'conv_tol_coarse', np.inf)
    if origin.residual_rms <= (
            NLCG_CANONICAL_RESTORATION_FULL_DAMP_RATIO *
            coarse_tolerance):
        return 1.0
    return NLCG_CANONICAL_RESTORATION_DAMP


def _canonical_restoration(self, origin, target_mu, objective,
                           consistency, charge_history):
    """Reduce sharp-sigma charge/tangent coupling inside an energy band."""
    maximum = self.nlcg_max_line_search_evaluations
    evaluations = 0
    samples = []
    canonical_seed = origin.h
    current_nelec = float(origin.nelec)
    origin_value = float(objective(origin))
    tangent_tolerance = max(
        0.1*self.conv_tol,
        NLCG_CANONICAL_RESTORATION_REDUCTION*origin.residual_rms)
    probe_tolerance = tangent_tolerance
    if _canonical_restoration_damp(self, origin) == 1.0:
        probe_tolerance = max(
            tangent_tolerance,
            NLCG_CANONICAL_RESTORATION_PROBE_REDUCTION *
            origin.residual_rms)

    history_bracketed = (
        any(error < 0.0 for unused_nelec, error in charge_history) and
        any(error > 0.0 for unused_nelec, error in charge_history))
    if history_bracketed:
        proposed_nelec, bracket = _canonical_charge_proposal(
            charge_history, current_nelec, 0.0, self.capacity)
        if proposed_nelec is not None and bracket is not None:
            current_nelec = proposed_nelec
            logger.info(
                self, 'NLCG canonical refinement starts at bracketed '
                'N = %.12g in [%.12g, %.12g]',
                current_nelec, bracket[0], bracket[1])

    def solve_tangent(h, nelec, tolerance):
        nonlocal evaluations
        if evaluations >= maximum-1:
            return None
        try:
            state = self.calculate_cycle(h, nelec=nelec)
        except FloatingPointError:
            evaluations += 1
            return None
        evaluations += 1
        adiis = diis.DIIS(self)
        adiis.space = NLCG_CANONICAL_RESTORATION_SPACE
        damp = _canonical_restoration_damp(self, origin)
        while (state.residual_rms > tolerance and
               evaluations < maximum-2):
            fock = self.diis_pack(state.fock)
            residual = self.diis_pack(
                state.residual, weight_errors=True)
            try:
                target = self.diis_unpack(
                    adiis.update(fock, xerr=residual), state.fock)
                target = self._sanitize_h(target)
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                adiis.clear()
                target = self._copy(state.fock)
            direction = [
                new-old for old, new in zip(state.h, target)]
            try:
                trial = self.calculate_cycle([
                    old+damp*step
                    for old, step in zip(state.h, direction)],
                    nelec=nelec)
            except FloatingPointError:
                evaluations += 1
                break
            evaluations += 1
            old_residual = state.residual_rms
            predicted = max(0.0, 1.0-damp)*old_residual
            predicted_reduction = old_residual-predicted
            actual_reduction = old_residual-trial.residual_rms
            if predicted_reduction > np.finfo(float).eps*max(
                    old_residual, np.finfo(float).tiny):
                ratio = actual_reduction/predicted_reduction
            else:
                ratio = np.nan
            if (np.isfinite(ratio) and ratio > 0.75 and
                    actual_reduction >= 0.02*old_residual):
                damp = min(1.0, 2.0*damp)
            state = trial
        return state

    def fixed_mu_candidate(canonical):
        nonlocal evaluations
        if canonical is None or evaluations >= maximum:
            return None
        shift = target_mu-canonical.mu
        try:
            candidate = self.calculate_cycle([
                value+shift*identity
                for value, identity in zip(
                    canonical.h, self.identity)], mu=target_mu)
        except FloatingPointError:
            evaluations += 1
            return None
        evaluations += 1
        value = float(objective(candidate))
        logger.info(
            self, 'NLCG canonical restoration N = %.12g, '
            'mu error = %.6g, tangent residual = %.6g, '
            'fixed-mu residual = %.6g, objective = %.12g, nfev = %d',
            canonical.nelec, canonical.mu-target_mu,
            canonical.residual_rms, candidate.residual_rms, value,
            self.nfev)
        if value <= origin_value+consistency:
            samples.append(candidate)
        elif (candidate.residual_rms < origin.residual_rms and
              evaluations < maximum and value > origin_value):
            fraction = 0.9*consistency/(value-origin_value)
            if np.isfinite(fraction) and 0.0 < fraction < 1.0:
                try:
                    capped = self.calculate_cycle([
                        old+fraction*(new-old)
                        for old, new in zip(origin.h, candidate.h)],
                        mu=target_mu)
                except FloatingPointError:
                    evaluations += 1
                else:
                    evaluations += 1
                    capped_value = float(objective(capped))
                    logger.info(
                        self, 'NLCG capped canonical restoration '
                        'fraction = %.6g, residual = %.6g, '
                        'objective = %.12g, nfev = %d',
                        fraction, capped.residual_rms, capped_value,
                        self.nfev)
                    if capped_value <= origin_value+consistency:
                        samples.append(capped)
        return candidate

    while evaluations < maximum-1:
        bracketed = (
            any(error < 0.0 for unused_nelec, error in charge_history) and
            any(error > 0.0 for unused_nelec, error in charge_history))
        tolerance = tangent_tolerance if bracketed else probe_tolerance
        canonical = solve_tangent(
            canonical_seed, current_nelec, tolerance)
        if canonical is None:
            break
        error = float(canonical.mu-target_mu)
        if canonical.residual_rms <= tolerance:
            charge_history.append((current_nelec, error))
            charge_history[:] = charge_history[
                -NLCG_CANONICAL_RESTORATION_HISTORY:]
        candidate = fixed_mu_candidate(canonical)
        if candidate is not None and candidate.residual_rms <= self.conv_tol:
            break
        if (candidate is not None and
                candidate.residual_rms <= 0.5*origin.residual_rms):
            break
        if evaluations >= maximum-1:
            break
        proposed_nelec, bracket = _canonical_charge_proposal(
            charge_history, current_nelec, error, self.capacity)
        if proposed_nelec is None:
            break
        if bracket is None:
            bracket_text = 'unbracketed'
        else:
            bracket_text = '[%.12g, %.12g]' % bracket
        logger.info(
            self, 'NLCG canonical charge proposal N = %.12g (%s), '
            'remaining evaluations = %d', proposed_nelec, bracket_text,
            maximum-evaluations)
        current_nelec = proposed_nelec
        canonical_seed = canonical.h

    improving = [
        candidate for candidate in samples
        if candidate.residual_rms <= (
            (1.0-LINE_SEARCH_RESIDUAL_REDUCTION)*origin.residual_rms)]
    if not improving:
        return _LineSearchResult(
            None, False, evaluations,
            'bounded canonical restoration did not reduce the residual',
            restoration=True, consistency=consistency)
    accepted_state = min(
        improving,
        key=lambda candidate: (
            candidate.residual_rms, float(objective(candidate))))
    direction = [
        new-old for old, new in zip(origin.h, accepted_state.h)]
    gradient = objective_gradient(self, accepted_state, False)
    slope = self._inner(gradient, direction)
    sample = _LineSample(
        1.0, accepted_state, float(objective(accepted_state)),
        gradient, slope, 'canonical-restoration')
    return _LineSearchResult(
        sample, False, evaluations, 'bounded canonical restoration',
        restoration=True, consistency=consistency)


def _step_from_displacement(direction_norm, displacement):
    if (not np.isfinite(direction_norm) or
            direction_norm <= np.finfo(float).tiny or
            not np.isfinite(displacement) or displacement <= 0.0):
        return None
    alpha = displacement/direction_norm
    if not np.isfinite(alpha) or alpha <= 0.0:
        return None
    return float(alpha)


def _preconditioned_pr_plus(self, old_exact, new_exact,
                            old_descent, new_descent):
    """Return clipped PR+ for descent vectors ``-z_old`` and ``-z_new``."""
    old_z = [-value for value in old_descent]
    new_z = [-value for value in new_descent]
    denominator = self._inner(old_exact, old_z)
    new_metric = self._inner(new_exact, new_z)
    if (not np.isfinite(denominator) or
            not np.isfinite(new_metric) or
            denominator <= np.finfo(float).tiny or
            new_metric <= np.finfo(float).tiny):
        return 0.0, 'nonpositive preconditioned gradient metric'

    numerator = self._inner(
        [new-old for new, old in zip(new_exact, old_exact)], new_z)
    beta_fr = new_metric/denominator
    beta_pr = numerator/denominator
    if not np.isfinite(beta_pr) or not np.isfinite(beta_fr):
        return 0.0, 'nonfinite preconditioned PR coefficient'

    correlation = abs(self._inner(new_exact, old_z)) / np.sqrt(
        new_metric*denominator)
    if not np.isfinite(correlation) or correlation > NLCG_POWELL_RESTART:
        return 0.0, 'Powell restart'

    beta = max(0.0, min(beta_pr, beta_fr))
    if beta == 0.0:
        return 0.0, 'PR+ truncation'
    return float(beta), 'preconditioned PR+'


def nlcg(self, dm0=None, h=None):
    self.build()
    self.converged = False
    self.cycles = 0
    self.outer_cycles = 0
    self.nfev = 0
    self.refinements = 0
    self.message = ''

    if h is None:
        h, unused_nelec = self._initial_h(dm0)
    fixed_n = self.nelec is not None
    objective_name = 'free energy' if fixed_n else 'grand potential'

    def evaluate(values):
        if fixed_n:
            return self.calculate_cycle(values, nelec=self.nelec)
        return self.calculate_cycle(values, mu=self.mu)

    def objective(cycle_data):
        if fixed_n:
            return cycle_data.free_energy
        return cycle_data.grand_potential

    logger.info(
        self, 'NLCG settings: initial step = %.6g, max line evaluations = %d, '
        'sigma = %.6g, objective flat threshold = %.3g Ha; '
        'SCF diis_space and damp do not affect NLCG; bounded canonical '
        'restoration uses private Pulay settings',
        self.nlcg_initial_step, self.nlcg_max_line_search_evaluations,
        getattr(self, 'sigma', np.nan), LINE_SEARCH_OBJECTIVE_FLAT)
    strict_tolerances = _strict_line_tolerances(self)
    if strict_tolerances is None:
        logger.info(
            self, 'NLCG strict line minimization is disabled; resolved '
            'minima use the legacy %.3g relative slope reduction',
            LINE_SEARCH_SLOPE_RATIO)
    else:
        logger.info(
            self, 'NLCG strict line minimization: alpha relative '
            'uncertainty <= %.6g (%.6g%%) and normalized |slope| <= %.6g',
            strict_tolerances[0], 100.0*strict_tolerances[0],
            strict_tolerances[1])

    state = evaluate(h)
    residual_direction = [
        0.5*(fock-hamiltonian)
        for hamiltonian, fock in zip(state.h, state.fock)]
    direction = [value.copy() for value in residual_direction]
    exact_gradient = objective_gradient(self, state, fixed_n)
    direction_has_history = False
    conjugacy_preconditioner = 'residual'
    displacement_history = []
    null_displacement_history = []
    canonical_charge_history = []
    restoration_consistency_floor = 0.0
    weak_restoration_steps = 0
    pulay = diis.DIIS(self)
    pulay.space = NLCG_PULAY_SPACE
    pulay.min_space = 2
    orbital_history = _OrbitalHistory()
    orbital_phase_remaining = 0
    orbital_phase_ready = True
    best_state = state
    best_residual = state.residual_rms
    residual_stagnation = 0
    last_nonrestoration_objective_change = None
    failure_message = None

    for unused_cycle in range(self.max_cycle):
        if state.residual_rms <= self.conv_tol:
            break

        sharp_fixed_n = (
            fixed_n and
            getattr(self, 'sigma', np.inf) <= NLCG_SHARP_SIGMA)
        if (sharp_fixed_n and orbital_phase_ready and
                state.residual_rms <= NLCG_ORBITAL_TRIGGER_RESIDUAL):
            orbital_phase_remaining = NLCG_ORBITAL_PHASE_STEPS
            orbital_phase_ready = False
            orbital_history.clear()
            logger.info(
                self, 'NLCG starting %d-step sharp-sigma orbital phase at '
                'residual %.6g', orbital_phase_remaining,
                state.residual_rms)
        proactive_orbital = orbital_phase_remaining > 0
        pulay_direction, pulay_reason = _pulay_direction(
            self, state, pulay)
        (occupation_direction, active_response,
         occupation_energy_scale) = _occupation_direction_data(
             self, state, exact_gradient, residual_direction,
             sharp_fixed_n)
        primary_kind = (
            'conjugate' if direction_has_history else 'residual')
        null_direction = _fermi_null_residual_direction(self, state)
        active_residual_rms = state.residual_rms
        if sharp_fixed_n and null_direction is not None:
            full_residual_norm = _direction_norm(self, state.residual)
            null_residual_norm = _direction_norm(self, null_direction)
            active_residual_norm = _direction_norm(self, [
                residual-null for residual, null in zip(
                    state.residual, null_direction)])
            if full_residual_norm > np.finfo(float).tiny:
                active_residual_rms = (
                    state.residual_rms*active_residual_norm /
                    full_residual_norm)
            logger.info(
                self, 'NLCG sharp-sigma residual components: active = %.6g, '
                'null = %.6g, full = %.6g, null fraction = %.6g',
                active_residual_norm, null_residual_norm,
                full_residual_norm,
                null_residual_norm/max(
                    full_residual_norm, np.finfo(float).tiny))
        objective_is_flat = (
            last_nonrestoration_objective_change is not None and
            abs(last_nonrestoration_objective_change) <
            LINE_SEARCH_OBJECTIVE_FLAT)
        proactive_null = (
            objective_is_flat and fixed_n and null_direction is not None and
            state.residual_rms >= NLCG_NULL_MIN_RESIDUAL and
            active_residual_rms <= NLCG_NULL_ACTIVE_RESIDUAL and
            _direction_norm(self, null_direction) >=
            NLCG_NULL_MIN_NORM_FRACTION *
            _direction_norm(self, state.residual))
        proactive_pulay = (
            fixed_n and not sharp_fixed_n and
            pulay_direction is not None and
            residual_stagnation >= NLCG_PULAY_STAGNATION_STEPS)
        proactive_occupation = _occupation_direction_is_proactive(
            fixed_n, sharp_fixed_n, occupation_direction,
            state.residual_rms, orbital_phase_ready)
        proactive_canonical = (
            not fixed_n and
            weak_restoration_steps >=
            NLCG_CANONICAL_RESTORATION_WEAK_STEPS and
            restoration_consistency_floor >=
            _line_roundoff(objective(state)))
        canonical_attempts = [
            ('canonical-restoration', None),
            ('canonical-refinement', None)]
        if proactive_canonical:
            attempts = canonical_attempts + [
                ('residual', residual_direction),
                ('exact-gradient', None)]
            logger.info(
                self, 'NLCG escalating after %d weak restoration steps',
                weak_restoration_steps)
        else:
            attempts = []
            if proactive_orbital:
                attempts.append(('orbital-lbfgs', None))
            if proactive_null:
                attempts.append(('null-residual', null_direction))
            if proactive_pulay:
                attempts.append(('pulay', pulay_direction))
                logger.info(
                    self, 'NLCG probing %s Pulay direction after %d cycles '
                    'without a 1%% residual-best improvement (%s)',
                    ('residual-restoration' if objective_is_flat else
                     'objective-safeguarded'),
                    residual_stagnation, pulay_reason)
            # A resolved occupation-preconditioned step can now seed PR+.
            # Give that conjugate proposal one chance before falling back to
            # the plain occupation direction; otherwise proactive response
            # scaling starves NLCG on every low-residual fixed-N cycle.
            if proactive_occupation and primary_kind != 'conjugate':
                logger.info(
                    self, 'NLCG response metric = %.6g; '
                    'occupation energy scale = %.6g',
                    active_response, occupation_energy_scale)
                attempts.append((
                    'occupation-preconditioned', occupation_direction))
            attempts.append((primary_kind, direction))
            if primary_kind == 'conjugate':
                if proactive_occupation:
                    logger.info(
                        self, 'NLCG response metric = %.6g; '
                        'occupation energy scale = %.6g',
                        active_response, occupation_energy_scale)
                    attempts.append((
                        'occupation-preconditioned', occupation_direction))
                attempts.append(('residual', residual_direction))
            if not fixed_n:
                attempts.extend(canonical_attempts)
            if pulay_direction is not None and not proactive_pulay:
                attempts.append(('pulay', pulay_direction))
            if (occupation_direction is not None and
                    not proactive_occupation):
                attempts.append((
                    'occupation-preconditioned', occupation_direction))
            attempts.append(('exact-gradient', None))

        seed_displacement = (
            float(np.median(displacement_history))
            if displacement_history else
            self.nlcg_initial_step*_direction_norm(self, direction))
        line_evaluations = 0
        result = None
        accepted_direction = None
        accepted_direction_norm = 0.0
        accepted_kind = None
        accepted_attempt = None
        accepted_flat_direction = False
        skipped_direction = False
        deferred_growth_steps = []
        last_reason = 'no usable search direction'
        restoration_band = 0.0

        for attempt_index, (direction_kind, attempt_direction) in enumerate(
                attempts):
            if direction_kind == 'orbital-lbfgs':
                attempt_result = _orbital_line_search(
                    self, state, evaluate, objective, orbital_history)
                line_evaluations += attempt_result.evaluations
                last_reason = attempt_result.reason
                if attempt_result.sample is None:
                    orbital_phase_remaining = 0
                    orbital_history.clear()
                    logger.info(
                        self, 'NLCG orbital phase yielded: %s',
                        attempt_result.reason)
                    continue
                result = attempt_result
                accepted = attempt_result.sample
                accepted_direction = [
                    (new-old)/accepted.alpha
                    for old, new in zip(state.h, accepted.state.h)]
                accepted_direction_norm = _direction_norm(
                    self, accepted_direction)
                accepted_kind = direction_kind
                accepted_attempt = attempt_index
                break

            if direction_kind in (
                    'canonical-restoration', 'canonical-refinement'):
                restoration_band = max(
                    restoration_band, restoration_consistency_floor)
                if restoration_band < _line_roundoff(objective(state)):
                    continue
                logger.info(
                    self, 'NLCG %s with consistency = %.3g',
                    ('probing bounded canonical charge/tangent restoration'
                     if direction_kind == 'canonical-restoration' else
                     'refining the canonical charge bracket'),
                    restoration_band)
                attempt_result = _canonical_restoration(
                    self, state, self.mu, objective, restoration_band,
                    canonical_charge_history)
                line_evaluations += attempt_result.evaluations
                last_reason = attempt_result.reason
                if attempt_result.sample is None:
                    logger.info(
                        self, 'NLCG canonical restoration failed: %s',
                        attempt_result.reason)
                    continue
                result = attempt_result
                accepted = attempt_result.sample
                accepted_direction = [
                    new-old for old, new in zip(
                        state.h, accepted.state.h)]
                accepted_direction_norm = _direction_norm(
                    self, accepted_direction)
                accepted_kind = direction_kind
                accepted_attempt = attempt_index
                restoration_consistency_floor = max(
                    restoration_consistency_floor,
                    attempt_result.consistency)
                break

            if direction_kind == 'exact-gradient':
                attempt_direction = _scaled_exact_direction(
                    self, exact_gradient, residual_direction)
                if attempt_direction is None:
                    last_reason = 'exact gradient has zero or nonfinite norm'
                    continue

            if direction_kind == 'conjugate':
                logger.info(
                    self, 'NLCG trying %s-preconditioned PR+ direction',
                    conjugacy_preconditioner)

            if direction_kind == 'pulay':
                (attempt_direction, direction_metrics,
                 reflected_pulay) = _orient_pulay_direction(
                    self, exact_gradient, attempt_direction)
                if reflected_pulay:
                    logger.info(
                        self, 'NLCG reflecting uphill Pulay direction')
            else:
                direction_metrics = _descent_metrics(
                    self, exact_gradient, attempt_direction)
            descent, slope0, descent_cosine, direction_norm = (
                direction_metrics)
            natural_restoration_step = (
                direction_kind in ('residual', 'occupation-preconditioned')
                and np.isfinite(slope0)
                and abs(slope0) < LINE_SEARCH_OBJECTIVE_FLAT)
            if (direction_kind == 'null-residual' or
                    (direction_kind == 'pulay' and objective_is_flat) or
                    natural_restoration_step):
                # These directions already represent target Hamiltonian
                # displacements, so alpha=1 is their scale-aware natural
                # residual-restoration trial.  A locally flat residual or
                # occupation direction must not inherit a much larger
                # displacement from the preceding objective-descent phase;
                # doing so can require dozens of factor-two contractions to
                # return to its useful unit scale.  While the objective is
                # active, a raw Pulay target can be catastrophically large
                # for a metallic Fermi response, so it instead inherits the
                # physical displacement trust scale below.
                attempt_displacement = direction_norm
            else:
                attempt_displacement = seed_displacement
            step_seed = _step_from_displacement(
                direction_norm, attempt_displacement)
            if step_seed is None:
                skipped_direction = True
                last_reason = 'could not scale %s direction' % direction_kind
                continue

            residual_restoration_direction = direction_kind in (
                'residual', 'pulay', 'occupation-preconditioned',
                'null-residual')
            flat_residual_direction = (
                (direction_kind == 'null-residual' and
                 np.isfinite(slope0) and slope0 <= 0.0) or
                (residual_restoration_direction and
                 np.isfinite(slope0) and
                 abs(slope0*step_seed) <
                 LINE_SEARCH_OBJECTIVE_FLAT))
            if not descent and not flat_residual_direction:
                skipped_direction = True
                last_reason = '%s direction is not sufficiently descending' % (
                    direction_kind)
                logger.info(
                    self, 'NLCG rejecting %s direction: slope = %.6g, '
                    'descent cosine = %.6g',
                    direction_kind, slope0, descent_cosine)
                continue
            if flat_residual_direction and not descent:
                logger.info(
                    self, 'NLCG %s direction is objective-flat; '
                    'enabling bounded restoration', direction_kind)

            if attempt_index:
                logger.info(
                    self, 'NLCG retrying line search in %s direction',
                    direction_kind)
            logger.info(
                self, 'NLCG direction = %s norm = %.6g displacement = %.6g '
                'alpha = %.6g slope = %.6g descent cosine = %.6g',
                direction_kind, direction_norm, attempt_displacement,
                step_seed, slope0, descent_cosine)
            line_options = {}
            if direction_kind == 'pulay':
                line_options['require_residual_improvement'] = True
            elif direction_kind == 'null-residual':
                line_options['accept_bounded_residual'] = True
                line_options['max_evaluations'] = (
                    LINE_SEARCH_NULL_EVALUATION_LIMIT)
            elif direction_kind == 'occupation-preconditioned':
                line_options['bound_residual_growth'] = True
                line_options['residual_metric'] = (
                    lambda trial: _fermi_guarded_residual_rms(
                        self, state, trial))
                line_options['residual_growth_limit'] = (
                    NLCG_OCCUPATION_ACTIVE_GROWTH)
            attempt_result = _line_search(
                self, state, exact_gradient, attempt_direction,
                evaluate, objective, step_seed,
                allow_restoration=(
                    residual_restoration_direction and
                    flat_residual_direction),
                **line_options)
            line_evaluations += attempt_result.evaluations
            last_reason = attempt_result.reason
            if residual_restoration_direction:
                restoration_band = max(
                    restoration_band, attempt_result.consistency)
            if attempt_result.sample is None:
                logger.info(
                    self, 'NLCG %s direction failed: %s',
                    direction_kind, attempt_result.reason)
                continue

            if (direction_kind == 'null-residual' and
                    attempt_result.sample.method.startswith(
                        'null-tangent')):
                candidate_direction = [
                    (new-old)/attempt_result.sample.alpha
                    for old, new in zip(
                        state.h, attempt_result.sample.state.h)]
                candidate_direction_norm = _direction_norm(
                    self, candidate_direction)
            else:
                candidate_direction = attempt_direction
                candidate_direction_norm = direction_norm
            candidate_flat_direction = (
                flat_residual_direction and not descent)

            objective_band = _line_objective_band(
                objective(state), attempt_result.consistency)
            objective_decrease = (
                objective(state)-attempt_result.sample.value)
            if (not attempt_result.restoration and
                    objective_decrease > objective_band):
                # Any material thermodynamic descent is authoritative until
                # the objective is genuinely flat.  This includes an
                # inexact line-limit fallback: conjugacy is restarted for
                # that case below, while the accepted state preserves the
                # objective progress.  Residual growth must not steer the
                # iteration into a nonstationary residual minimum.
                logger.info(
                    self, 'NLCG accepting %s %s objective step '
                    'before residual retries: decrease = %.6g, flat band '
                    '= %.3g, residual ratio = %.6g',
                    ('resolved' if attempt_result.resolved else 'inexact'),
                    direction_kind, objective_decrease, objective_band,
                    (_sample_residual(attempt_result.sample) /
                     max(state.residual_rms, np.finfo(float).tiny)))
                result = attempt_result
                accepted_direction = candidate_direction
                accepted_direction_norm = candidate_direction_norm
                accepted_kind = direction_kind
                accepted_attempt = attempt_index
                accepted_flat_direction = candidate_flat_direction
                break

            if direction_kind == 'null-residual':
                null_ratio = (
                    _sample_residual(attempt_result.sample) /
                    max(state.residual_rms, np.finfo(float).tiny))
                improves_residual_best = (
                    _sample_residual(attempt_result.sample) <=
                    (1.0-LINE_SEARCH_RESIDUAL_REDUCTION) *
                    best_residual)
                if (null_ratio > 1.0-NLCG_NULL_WEAK_REDUCTION and
                        not improves_residual_best):
                    last_reason = (
                        'Fermi-null cleanup was weak and did not improve '
                        'the residual best')
                    logger.info(
                        self, 'NLCG null-residual direction yielded after '
                        'a weak cleanup: residual ratio %.6g, best ratio '
                        '%.6g', null_ratio,
                        (_sample_residual(attempt_result.sample) /
                         max(best_residual, np.finfo(float).tiny)))
                    continue

            if (direction_kind == 'pulay' and
                    _sample_residual(attempt_result.sample) >
                    (1.0-LINE_SEARCH_PARETO_RESIDUAL_REDUCTION) *
                    state.residual_rms):
                last_reason = (
                    'Pulay direction did not reduce the residual')
                logger.info(
                    self, 'NLCG Pulay direction rejected: residual %.6g '
                    '-> %.6g', state.residual_rms,
                    _sample_residual(attempt_result.sample))
                continue

            if (_sample_residual(attempt_result.sample) >
                    LINE_SEARCH_RESIDUAL_GROWTH_RESTART *
                    state.residual_rms):
                logger.info(
                    self, 'NLCG deferring %s objective step with '
                    'residual ratio %.6g',
                    direction_kind,
                    (_sample_residual(attempt_result.sample) /
                     state.residual_rms))
                deferred_growth_steps.append((
                    attempt_result, candidate_direction,
                    candidate_direction_norm, direction_kind,
                    attempt_index, candidate_flat_direction))
                continue

            result = attempt_result
            accepted_direction = candidate_direction
            accepted_direction_norm = candidate_direction_norm
            accepted_kind = direction_kind
            accepted_attempt = attempt_index
            accepted_flat_direction = candidate_flat_direction
            break

        if deferred_growth_steps:
            candidate_steps = list(deferred_growth_steps)
            if result is not None:
                candidate_steps.append((
                    result, accepted_direction, accepted_direction_norm,
                    accepted_kind, accepted_attempt,
                    accepted_flat_direction))
            minimum_deferred_value = min(
                entry[0].sample.value for entry in candidate_steps)
            deferred_band = _line_objective_band(
                objective(state), LINE_SEARCH_OBJECTIVE_FLAT)
            objective_equivalent_steps = [
                entry for entry in candidate_steps
                if entry[0].sample.value <=
                minimum_deferred_value+deferred_band
            ]
            (result, accepted_direction, accepted_direction_norm,
             accepted_kind, accepted_attempt,
             accepted_flat_direction) = min(
                objective_equivalent_steps,
                key=lambda entry: (
                    _sample_residual(entry[0].sample),
                    entry[0].sample.value))
            if (_sample_residual(result.sample) >
                    LINE_SEARCH_RESIDUAL_GROWTH_RESTART *
                    state.residual_rms):
                logger.info(
                    self, 'NLCG accepting deferred %s objective step after '
                    'retries; residual ratio = %.6g', accepted_kind,
                    (_sample_residual(result.sample) /
                     state.residual_rms))

        self.cycles += 1
        if result is None:
            failure_message = (
                'NLCG line search failed after exact-gradient direction: %s' %
                last_reason)
            logger.info(
                self, 'NLCG cycle %d: %s; line evaluations = %d; nfev = %d',
                self.cycles, failure_message, line_evaluations, self.nfev)
            break

        accepted = result.sample
        # A bounded residual cleanup does not establish that the
        # thermodynamic descent has flattened.  Keep the latest genuine
        # objective-search change so Pulay restoration is not re-enabled
        # between alternating descent and null-cleanup steps.
        last_nonrestoration_objective_change = (
            _nonrestoration_objective_change(
                last_nonrestoration_objective_change,
                objective(state), result))
        new_residual_direction = [
            0.5*(fock-hamiltonian)
            for hamiltonian, fock in zip(
                accepted.state.h, accepted.state.fock)]
        new_exact_gradient = accepted.gradient
        residual_ratio = accepted.state.residual_rms / max(
            state.residual_rms, np.finfo(float).tiny)

        accepted_preconditioner = 'residual'
        old_preconditioned_direction = residual_direction
        new_preconditioned_direction = new_residual_direction
        if (accepted_kind == 'occupation-preconditioned' or
                (accepted_kind == 'conjugate' and
                 conjugacy_preconditioner == 'occupation')):
            accepted_preconditioner = 'occupation'
            old_preconditioned_direction = occupation_direction
            (new_preconditioned_direction, unused_response,
             unused_energy_scale) = _occupation_direction_data(
                 self, accepted.state, new_exact_gradient,
                 new_residual_direction, sharp_fixed_n)

        if (not fixed_n and
                state.residual_rms <= getattr(
                    self, 'conv_tol_coarse', self.conv_tol) and
                abs(accepted.value-objective(state)) <= result.consistency):
            restoration_consistency_floor = max(
                restoration_consistency_floor, result.consistency)

        beta = 0.0
        beta_reason = 'line-search restart'
        may_continue_conjugacy = (
            accepted.state.residual_rms > self.conv_tol and
            result.resolved and not result.restoration and
            accepted_kind in (
                'residual', 'conjugate', 'occupation-preconditioned') and
            accepted_attempt == 0 and not skipped_direction and
            not accepted_flat_direction and
            residual_ratio <= LINE_SEARCH_RESIDUAL_GROWTH_RESTART and
            old_preconditioned_direction is not None and
            new_preconditioned_direction is not None and
            (accepted_kind != 'occupation-preconditioned' or
             NLCG_OCCUPATION_PR_ENABLED))
        if may_continue_conjugacy:
            beta, beta_reason = _preconditioned_pr_plus(
                self, exact_gradient, new_exact_gradient,
                old_preconditioned_direction,
                new_preconditioned_direction)
        elif result.restoration:
            beta_reason = 'residual restoration'
        elif residual_ratio > LINE_SEARCH_RESIDUAL_GROWTH_RESTART:
            beta_reason = 'residual growth restart'
        elif accepted_attempt:
            beta_reason = 'direction retry'
        elif accepted_kind == 'pulay':
            beta_reason = 'Pulay direction'
        elif accepted_kind == 'orbital-lbfgs':
            beta_reason = 'fixed-spectrum orbital phase'
        elif not result.resolved:
            beta_reason = 'inexact line minimum'

        if beta > 0.0:
            next_direction = [
                preconditioned+beta*old_direction
                for preconditioned, old_direction in zip(
                    new_preconditioned_direction, accepted_direction)]
            descent, unused_slope, unused_cosine, unused_norm = (
                _descent_metrics(
                    self, new_exact_gradient, next_direction))
            if not descent:
                beta = 0.0
                beta_reason = 'post-PR descent restart'
        if beta == 0.0:
            # ``primary_kind == residual`` must always carry the actual
            # fixed-point residual.  A plain occupation direction remains a
            # separate proactive fallback on the next cycle.
            next_direction = [
                value.copy() for value in new_residual_direction]

        displacement = abs(accepted.alpha)*accepted_direction_norm
        if (accepted_kind == 'null-residual' and
                np.isfinite(displacement) and
                displacement > np.finfo(float).tiny):
            next_null_displacement = float(displacement)
            if (result.restoration and
                    residual_ratio <=
                    1.0-LINE_SEARCH_RESIDUAL_REDUCTION):
                next_null_displacement *= 2.0
            null_displacement_history.append(next_null_displacement)
            null_displacement_history = null_displacement_history[
                -NLCG_DISPLACEMENT_HISTORY:]
        # The exact-gradient fallback sees the sharp Fermi-response
        # curvature that the residual preconditions away.  Its accepted
        # displacement can consequently be many orders of magnitude smaller
        # and is not representative of a useful seed for the next
        # preconditioned direction.
        if (accepted_kind not in (
                'exact-gradient', 'pulay', 'canonical-restoration',
                'canonical-refinement', 'null-residual',
                'orbital-lbfgs') and
                np.isfinite(displacement) and
                displacement > np.finfo(float).tiny):
            displacement_history.append(float(displacement))
            displacement_history = displacement_history[
                -NLCG_DISPLACEMENT_HISTORY:]

        state = accepted.state
        exact_gradient = new_exact_gradient
        residual_direction = new_residual_direction
        direction = next_direction
        direction_has_history = beta > 0.0
        conjugacy_preconditioner = (
            accepted_preconditioner if direction_has_history else 'residual')
        if accepted_kind == 'orbital-lbfgs':
            orbital_phase_remaining = max(
                0, orbital_phase_remaining-1)
            if orbital_phase_remaining == 0:
                logger.info(
                    self, 'NLCG sharp-sigma orbital phase reached its '
                    '%d-step limit', NLCG_ORBITAL_PHASE_STEPS)
        else:
            orbital_phase_remaining = 0
            # A yielded or completed fixed-spectrum phase is a bounded
            # attempt, not a probe to repeat on every subsequent low-residual
            # cycle.  Rearm it only after the iteration leaves the trigger
            # region and later re-enters.
            if (not sharp_fixed_n or
                    state.residual_rms > NLCG_ORBITAL_TRIGGER_RESIDUAL):
                orbital_phase_ready = True
            orbital_history.clear()
        if ((result.restoration or not result.resolved) and
                residual_ratio >
                1.0-LINE_SEARCH_PROGRESS_REDUCTION):
            weak_restoration_steps += 1
        else:
            weak_restoration_steps = 0
        if state.residual_rms < best_state.residual_rms:
            best_state = state
        if (state.residual_rms <=
                (1.0-NLCG_PULAY_BEST_REDUCTION)*best_residual):
            best_residual = state.residual_rms
            residual_stagnation = 0
        else:
            residual_stagnation += 1
        if accepted_kind == 'pulay':
            residual_stagnation = 0
        acceptance = (
            'restoration' if result.restoration else
            ('resolved' if result.resolved else 'inexact'))

        logger.info(
            self,
            'NLCG cycle %d: %s = %.12g, residual = %e, mu = %.12g, '
            'nelec = %.12g, step = %.8g, displacement = %.6g, '
            'direction = %s, beta = %.6g (%s), residual ratio = %.6g, '
            'consistency = %.3g, line evaluations = %d, nfev = %d, '
            'fit = %s, acceptance = %s (%s), alpha uncertainty = %.6g '
            '(%.6g%%), normalized slope = %.6g, slope bracket = %s',
            self.cycles, objective_name, objective(state),
            state.residual_rms, state.mu, state.nelec, accepted.alpha,
            displacement, accepted_kind, beta, beta_reason,
            residual_ratio, result.consistency, line_evaluations, self.nfev,
            accepted.method, acceptance, result.reason,
            result.alpha_relative_uncertainty,
            100.0*result.alpha_relative_uncertainty,
            result.normalized_slope,
            ('none' if result.slope_interval is None else
             '[%.8g, %.8g]' % result.slope_interval))
        callback = getattr(self, 'callback', None)
        if callable(callback):
            callback({
                'solver': self,
                'cycle_data': state,
                'best_cycle_data': best_state,
                'cycle': self.cycles,
                'line_search_result': result,
                'direction_kind': accepted_kind,
            })

    self._nlcg_best_cycle_data = best_state
    converged = state.residual_rms <= self.conv_tol
    if converged:
        self.message = 'converged NLCG residual'
    elif failure_message is not None:
        self.message = failure_message
    else:
        self.message = 'maximum NLCG cycles reached'
    return self._finalize(state, converged)


def fixed_mu_diis(self, dm0=None, h=None):
    self.build()
    self.converged = False
    self.cycles = 0
    self.outer_cycles = 0
    self.nfev = 0
    self.refinements = 0
    self.message = ''

    if h is None:
        h, unused_nelec = self._initial_h(dm0)
    state = self.calculate_cycle(h, mu=self.mu)
    adiis = diis.DIIS(self)
    adiis.space = self.diis_space

    for unused_cycle in range(self.max_cycle):
        if state.residual_rms <= self.conv_tol:
            break
        fock = self.diis_pack(state.fock)
        error = self.diis_pack(state.residual, weight_errors=False)
        h = self.diis_unpack(adiis.update(fock, xerr=error), state.fock)
        state = self.calculate_cycle(h, mu=self.mu)
        self.cycles += 1
        logger.info(
            self,
            'Fixed-mu DIIS cycle %d: grand potential = %f, residual = %e, '
            'nelec = %f',
            self.cycles, state.grand_potential, state.residual_rms,
            state.nelec)

    converged = state.residual_rms <= self.conv_tol
    self.message = ('converged fixed-mu gradient DIIS' if converged else
                    'maximum fixed-mu DIIS cycles reached')
    return self._finalize(state, converged)
