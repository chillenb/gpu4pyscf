import numpy as np
import cupy as cp
from scipy.interpolate import PchipInterpolator

from gpu4pyscf.lib import diis, logger


FERMI_DIVDIFF_RTOL = 1e-10
FERMI_RESPONSE_TOL = 1e-30
LINE_SEARCH_SLOPE_RATIO = 1e-1
LINE_SEARCH_ALPHA_RTOL = np.sqrt(np.finfo(float).eps)
LINE_SEARCH_ENERGY_RTOL = 64.0 * np.finfo(float).eps


class _LineSample:
    __slots__ = ('alpha', 'state', 'value', 'gradient', 'slope', 'method')

    def __init__(self, alpha, state, value, gradient, slope, method):
        self.alpha = float(alpha)
        self.state = state
        self.value = float(value)
        self.gradient = gradient
        self.slope = float(slope)
        self.method = method


class _LineSearchResult:
    __slots__ = ('sample', 'resolved', 'evaluations', 'reason')

    def __init__(self, sample, resolved, evaluations, reason):
        self.sample = sample
        self.resolved = bool(resolved)
        self.evaluations = int(evaluations)
        self.reason = reason


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


def _line_alpha_close(left, right):
    scale = max(1.0, abs(left), abs(right))
    return abs(left-right) <= LINE_SEARCH_ALPHA_RTOL * scale


def _line_candidate_is_new(alpha, samples, invalid_alphas=()):
    if not np.isfinite(alpha) or alpha <= 0.0:
        return False
    return not any(
        _line_alpha_close(alpha, old)
        for old in [x.alpha for x in samples] + list(invalid_alphas)
    )


def _line_improves(value, origin_value):
    tolerance = LINE_SEARCH_ENERGY_RTOL * max(1.0, abs(origin_value))
    return value < origin_value-tolerance


def _best_line_sample(samples):
    return min(samples, key=lambda sample: (sample.value, sample.alpha))


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
    tolerance = LINE_SEARCH_ENERGY_RTOL * max(1.0, abs(best.value))
    if (left[1] >= best.value-tolerance and
            right[1] >= best.value-tolerance):
        return left[0], right[0]
    return None


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


def _polynomial_step(alphas, values, interval):
    """Fit three to five energies and return a convex stationary point."""
    alphas = np.asarray(alphas, dtype=float)
    values = np.asarray(values, dtype=float)
    if not 3 <= alphas.size <= 5:
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


def _spline_step(alphas, values, interval):
    """Fit a shape-preserving cubic spline and return a convex minimum."""
    alphas = np.asarray(alphas, dtype=float)
    values = np.asarray(values, dtype=float)
    if alphas.size <= 5:
        return None, None
    order = np.argsort(alphas)
    alphas = alphas[order]
    values = values[order]
    scale = float(np.max(values)-np.min(values))
    if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
        return None, None
    scaled = (values-np.min(values)) / scale
    try:
        spline = PchipInterpolator(alphas, scaled, extrapolate=False)
        roots = spline.derivative().roots(extrapolate=False)
    except (ValueError, FloatingPointError):
        return None, None

    lower, upper = interval
    candidates = []
    tolerance = LINE_SEARCH_ALPHA_RTOL * max(
        1.0, abs(lower), abs(upper))
    for root in roots:
        alpha = float(root)
        if not lower+tolerance < alpha < upper-tolerance:
            continue
        value = float(spline(alpha))
        curvature = float(spline.derivative(2)(alpha))
        if np.isfinite(value) and np.isfinite(curvature) and curvature > 0.0:
            candidates.append((value, alpha, curvature))
    if not candidates:
        return None, None
    unused_value, alpha, curvature = min(candidates)
    return alpha, curvature


def _fitted_line_step(samples, interval, harmonic):
    samples = sorted(samples, key=lambda sample: sample.alpha)
    alphas = [sample.alpha for sample in samples]
    values = [sample.value for sample in samples]
    if len(samples) <= 5:
        alpha, curvature = _polynomial_step(alphas, values, interval)
        names = {3: 'quadratic', 4: 'cubic', 5: 'quartic'}
        method = names.get(len(samples))
    else:
        alpha, curvature = _spline_step(alphas, values, interval)
        method = 'spline'
    if alpha is None:
        return None, None, None

    if harmonic is not None:
        best = _best_line_sample(samples)
        fitted_delta = alpha-best.alpha
        harmonic_delta = harmonic-best.alpha
        tolerance = LINE_SEARCH_ALPHA_RTOL * max(
            1.0, abs(alpha), abs(harmonic), abs(best.alpha))
        if (abs(fitted_delta) > tolerance and
                abs(harmonic_delta) > tolerance and
                fitted_delta*harmonic_delta < 0.0):
            return None, None, None
    return alpha, curvature, method


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


def _line_search_proposal(samples, invalid_alphas, initial_step):
    """Choose a new absolute line coordinate and describe the proposal."""
    samples = sorted(samples, key=lambda sample: sample.alpha)
    all_alphas = [sample.alpha for sample in samples] + list(invalid_alphas)
    if len(all_alphas) == 1:
        return float(initial_step), 'initial'

    best = _best_line_sample(samples)
    harmonic, unused_curvature, harmonic_interval = _harmonic_step(samples)
    interval = harmonic_interval
    if interval is None:
        interval = _energy_bracket(samples, invalid_alphas)

    if interval is not None:
        fitted, unused_curvature, method = _fitted_line_step(
            samples, interval, harmonic)
        if (fitted is not None and
                _line_candidate_is_new(fitted, samples, invalid_alphas)):
            return fitted, method
        if (harmonic is not None and
                _line_candidate_is_new(harmonic, samples, invalid_alphas)):
            return harmonic, 'harmonic'
        midpoint = _directional_midpoint(
            best, interval, samples, invalid_alphas)
        if _line_candidate_is_new(midpoint, samples, invalid_alphas):
            return midpoint, 'slot-bisect'
        return None, None

    positive_slots = sorted(alpha for alpha in all_alphas if alpha > 0.0)
    if _line_alpha_close(best.alpha, 0.0):
        if not positive_slots:
            candidate = float(initial_step)
        else:
            candidate = 0.5*positive_slots[0]
        if _line_candidate_is_new(candidate, samples, invalid_alphas):
            return candidate, 'contract'
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


def _line_search(self, origin, origin_gradient, direction, evaluate,
                 objective, initial_step):
    slope0 = self._inner(origin_gradient, direction)
    origin_sample = _LineSample(
        0.0, origin, objective(origin), origin_gradient, slope0, 'origin')
    samples = [origin_sample]
    invalid_alphas = []
    evaluations = 0
    stopped_reason = 'interpolation stagnated'

    while evaluations < self.nlcg_max_line_search_evaluations:
        alpha, method = _line_search_proposal(
            samples, invalid_alphas, initial_step)
        if alpha is None:
            break
        evaluations += 1
        try:
            trial = evaluate([
                h+alpha*d for h, d in zip(origin.h, direction)])
            gradient = objective_gradient(
                self, trial, self.nelec is not None)
            value = float(objective(trial))
            slope = self._inner(gradient, direction)
            if not np.isfinite(value) or not np.isfinite(slope):
                raise FloatingPointError('nonfinite line-search sample')
        except FloatingPointError:
            invalid_alphas.append(alpha)
            logger.info(
                self, 'NLCG line alpha = %.8g method = %s is nonfinite',
                alpha, method)
            continue

        sample = _LineSample(
            alpha, trial, value, gradient, slope, method)
        samples.append(sample)
        samples.sort(key=lambda item: item.alpha)
        logger.info(
            self, 'NLCG line alpha = %.8g method = %s objective = %.12g '
            'slope = %.6g residual = %.6g',
            alpha, method, value, slope, trial.residual_rms)

        best = _best_line_sample(samples)
        energy_tolerance = LINE_SEARCH_ENERGY_RTOL * max(
            1.0, abs(origin_sample.value))
        if (trial.residual_rms <= self.conv_tol and
                value <= origin_sample.value+energy_tolerance):
            return _LineSearchResult(
                sample, True, evaluations, 'converged line sample')
        if (_line_improves(best.value, origin_sample.value) and
                abs(best.slope) <= LINE_SEARCH_SLOPE_RATIO*abs(slope0) and
                _sample_has_positive_curvature(samples, best)):
            return _LineSearchResult(
                best, True, evaluations, 'resolved line minimum')

    best = _best_line_sample(samples)
    if _line_improves(best.value, origin_sample.value):
        if evaluations >= self.nlcg_max_line_search_evaluations:
            stopped_reason = 'line-search evaluation limit'
        return _LineSearchResult(
            best, False, evaluations, stopped_reason)
    return _LineSearchResult(
        None, False, evaluations, 'no lower objective sample')


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

    state = evaluate(h)
    gradient = [
        0.5 * (f-h) for h, f in zip(state.h, state.fock)]
    direction = [g.copy() for g in gradient]
    exact_gradient = objective_gradient(self, state, fixed_n)
    step_seed = self.nlcg_initial_step
    direction_has_history = False
    failure_message = None

    for unused_cycle in range(self.max_cycle):
        if state.residual_rms <= self.conv_tol:
            break

        slope0 = self._inner(exact_gradient, direction)
        restarted = False
        direction_kind = (
            'conjugate' if direction_has_history else 'residual')
        if ((not np.isfinite(slope0) or slope0 >= 0.0) and
                direction_kind == 'conjugate'):
            direction = [g for g in gradient]
            direction_has_history = False
            restarted = True
            direction_kind = 'residual'
            slope0 = self._inner(exact_gradient, direction)
            logger.info(
                self, 'NLCG restarting a non-descent conjugate direction')

        if not np.isfinite(slope0) or slope0 >= 0.0:
            direction = [-g for g in exact_gradient]
            restarted = True
            direction_kind = 'exact-gradient'
            slope0 = self._inner(exact_gradient, direction)
            logger.info(
                self, 'NLCG restarting in the exact-gradient direction')

        if not np.isfinite(slope0) or slope0 >= 0.0:
            failure_message = 'NLCG exact gradient is not a descent direction'
            self.cycles += 1
            break

        result = _line_search(
            self, state, exact_gradient, direction, evaluate, objective,
            step_seed)
        line_evaluations = result.evaluations
        retried = False
        if result.sample is None and direction_kind == 'conjugate':
            direction = [g for g in gradient]
            direction_has_history = False
            restarted = True
            retried = True
            direction_kind = 'residual'
            slope0 = self._inner(exact_gradient, direction)
            if np.isfinite(slope0) and slope0 < 0.0:
                logger.info(
                    self, 'NLCG retrying line search in residual direction')
                retry = _line_search(
                    self, state, exact_gradient, direction, evaluate,
                    objective, self.nlcg_initial_step)
                line_evaluations += retry.evaluations
                result = retry

        if result.sample is None and direction_kind == 'residual':
            direction = [-g for g in exact_gradient]
            direction_has_history = False
            restarted = True
            retried = True
            direction_kind = 'exact-gradient'
            slope0 = self._inner(exact_gradient, direction)
            if np.isfinite(slope0) and slope0 < 0.0:
                logger.info(
                    self,
                    'NLCG retrying line search in exact-gradient direction')
                retry = _line_search(
                    self, state, exact_gradient, direction, evaluate,
                    objective, self.nlcg_initial_step)
                line_evaluations += retry.evaluations
                result = retry

        self.cycles += 1
        if result.sample is None:
            failure_message = (
                'NLCG line search failed to lower %s' % objective_name)
            logger.info(
                self, 'NLCG cycle %d: %s; line evaluations = %d',
                self.cycles, result.reason, line_evaluations)
            break

        accepted = result.sample
        new_gradient = [
            0.5 * (f-h) for h, f in zip(
                accepted.state.h, accepted.state.fock)]
        beta = 0.0
        denominator = self._inner(gradient, gradient)
        if (result.resolved and not restarted and not retried and
                np.isfinite(denominator) and
                denominator > np.finfo(float).tiny):
            numerator = self._inner(
                new_gradient,
                [new-old for new, old in zip(new_gradient, gradient)])
            if np.isfinite(numerator):
                beta = max(0.0, numerator/denominator)

        state = accepted.state
        exact_gradient = accepted.gradient
        gradient = new_gradient
        direction = [
            g + beta*d for g, d in zip(new_gradient, direction)]
        direction_has_history = beta > 0.0
        step_seed = float(np.clip(
            accepted.alpha,
            0.25*self.nlcg_initial_step,
            4.0*self.nlcg_initial_step))

        logger.info(
            self,
            'NLCG cycle %d: %s = %.12g, residual = %e, mu = %.12g, '
            'nelec = %.12g, step = %.8g, line evaluations = %d, '
            'fit = %s (%s)',
            self.cycles, objective_name, objective(state),
            state.residual_rms, state.mu, state.nelec, accepted.alpha,
            line_evaluations, accepted.method, result.reason)

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
