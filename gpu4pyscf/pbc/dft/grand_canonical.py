"""Finite-temperature direct minimisation for periodic restricted Kohn--Sham DFT.

This module deliberately does not hook into the ordinary SCF kernel.  The
optimisation variables are Hermitian matrices in a fixed, orthonormal AO
coordinate system, the electron ensemble may be fixed-mu or fixed-N, and the
supplied :class:`KRKS` object remains the authoritative evaluator of the DFT
functional.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import inspect
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import cupy as cp

from gpu4pyscf.lib import logger

try:  # cupyx supplies the most efficient, branch-stable implementation.
    from cupyx.scipy.special import expit as _cupy_expit
except ImportError:  # pragma: no cover - all supported GPU installations have cupyx
    _cupy_expit = None


__all__ = [
    'GrandCanonicalConfig', 'GrandCanonicalResult', 'GrandCanonicalKRKS',
    'IterationRecord', 'fermi_occupations', 'fermi_entropy',
    'fermi_divided_difference',
]


@dataclass
class GrandCanonicalConfig:
    max_cycle: int = 100
    conv_tol_omega: float = 1.0e-8
    conv_tol_grad_rms: float = 1.0e-6
    conv_tol_residual_rms: float = 1.0e-5
    conv_tol_density_rms: float = 1.0e-7
    conv_tol_nelec: float = 1.0e-7
    required_consecutive_conv: int = 2

    cg_update: str = 'fletcher-reeves'
    cg_restart_interval: int = 20
    cg_beta_max: float = 5.0
    descent_tolerance: float = 1.0e-12
    preconditioned_descent_cosine_min: float = 0.05
    exact_gradient_polish_residual_rms: float = 1.0e-4
    mu_electron_number_tol: float = 1.0e-12
    mu_max_cycle: int = 100

    # Optional residual-DIIS final polishing.  The direct optimizer hands off
    # once the residual reaches the switch threshold.  DIIS then prioritizes
    # the fixed-point residual over objective changes below the configured
    # noise allowance.
    diis_switch_residual_rms: Optional[float] = None
    diis_space: int = 6
    diis_regularization: float = 1.0e-10
    diis_max_condition: float = 1.0e12
    diis_max_coefficient_l1: float = 10.0
    diis_backtrack_factor: float = 0.5
    diis_initial_damping: float = 1.0
    diis_max_backtracks: int = 8
    diis_min_residual_reduction: float = 1.0e-3
    diis_max_objective_increase: float = 1.0e-5
    diis_max_delta_nelec: float = 5.0e-2
    diis_trust_shrink_ratio: float = 0.25
    diis_trust_expand_ratio: float = 0.75
    diis_trust_expansion: float = 2.0
    diis_trust_expand_min_relative_reduction: float = 2.0e-2

    # Optional fixed-mu globalization through canonical continuation.  Each
    # inner solve fixes N; outer scalar secant steps zero the optimized
    # chemical-potential error before a one-Fock fixed-mu verification.
    canonical_continuation: bool = False
    canonical_continuation_max_outer: int = 16
    canonical_continuation_coarse_residual_tol: float = 4.0e-6
    canonical_continuation_bracketed_residual_tol: float = 1.0e-8
    canonical_continuation_handoff_delta_nelec: float = 2.0e-5
    canonical_continuation_unbracketed_handoff_delta_nelec: float = 2.0e-5
    canonical_continuation_initial_delta_nelec: float = 3.0e-2
    # Deprecated absolute cap retained for constructor compatibility.  None
    # selects the neutral-electron fraction appended below.
    canonical_continuation_max_delta_nelec: Optional[float] = None
    canonical_continuation_min_delta_nelec: float = 1.0e-8
    canonical_continuation_initial_damping: float = 0.125
    canonical_continuation_diis_max_coefficient_l1: float = 50.0

    # Optional branch selector for low-temperature calculations.  This shifts
    # only the initial auxiliary Hamiltonian by a scalar; mu and every
    # subsequently evaluated Fock matrix retain their physical energy zero.
    initial_electron_number: Optional[float] = None

    line_search_c1: float = 1.0e-4
    line_search_c2: float = 0.1
    line_search_max_evals: int = 12
    line_search_zoom_evals: int = 12
    line_search_alpha_init: float = 1.0
    line_search_alpha_cap: float = 4.0
    line_search_alpha_min: float = 1.0e-8
    line_search_growth: float = 2.0
    line_search_max_h_rms_step: float = 0.5
    armijo_backtrack_factor: float = 0.5
    line_search_nelec_guard_residual_rms: Optional[float] = 1.0e-2
    line_search_max_delta_nelec: float = 1.0
    line_search_nelec_guard_max_delta_nelec: float = 5.0e-2

    fermi_divdiff_rtol: float = 1.0e-10
    hermiticity_tol: float = 1.0e-10
    orthogonality_tol: float = 1.0e-9

    check_time_reversal: bool = True
    enforce_time_reversal: bool = True
    checkpoint_interval: int = 1
    checkpoint_path: Optional[str] = None
    verbose: Optional[int] = None

    # Experimental low-temperature NLCG controls.
    line_search_method: str = 'strong-wolfe'
    line_search_max_trials: int = 64
    line_search_nelec_feasible_alpha: bool = True
    line_search_nelec_alpha_bisections: int = 24
    hager_zhang_delta: float = 1.0e-1
    hager_zhang_sigma: float = 9.0e-1
    hager_zhang_expansion: float = 5.0
    hager_zhang_shrinkage: float = 6.6e-1
    hager_zhang_max_evals: int = 20
    hager_zhang_objective_noise: float = 1.0e-10
    hager_zhang_theta: float = 2.0
    nlcg_exact_gradient_blend: bool = True
    nlcg_exact_gradient_polish: bool = True
    nlcg_residual_filter_rms: Optional[float] = None
    nlcg_residual_filter_max_relative_increase: float = 0.0
    nlcg_residual_filter_min_relative_reduction: float = 2.0e-2
    nlcg_residual_filter_objective_noise: float = 1.0e-10
    # Once the residual filter is active, a unit trial is usually much larger
    # than the useful local step.  This opt-in warm start tries a modest first
    # step, then reuses the last accepted step within conservative bounds.
    # The separate evaluation cap applies only in this residual-polish regime.
    nlcg_residual_filter_warm_start: bool = False
    nlcg_residual_filter_initial_alpha: float = 1.0e-1
    nlcg_residual_filter_alpha_min: float = 2.0e-2
    nlcg_residual_filter_alpha_max: float = 2.0e-1
    nlcg_residual_filter_max_evals: Optional[int] = None
    canonical_continuation_handoff_delta_mu: float = 1.0e-6
    canonical_continuation_verification_residual_tol: float = 1.0e-6
    canonical_continuation_verification_density_tol: float = 1.0e-9
    canonical_continuation_root_nelec_tol: float = 1.0e-8
    canonical_continuation_max_delta_nelec_fraction: float = 0.1


@dataclass(frozen=True)
class IterationRecord:
    cycle: int
    grand_potential: float
    dft_total_energy: float
    mu_n: float
    entropy_energy: float
    electron_number: float
    delta_omega: float
    delta_nelec: float
    grad_rms: float
    residual_rms: float
    density_change_rms: float
    alpha: float
    phi_prime_0: float
    cg_beta: float
    restart_reason: str
    line_search_evals: int
    free_energy: float = np.nan
    chemical_potential: float = np.nan
    objective: float = np.nan
    delta_objective: float = np.nan
    optimizer: str = 'nlcg'
    search_direction_source: str = 'nlcg'
    strong_wolfe: bool = False
    line_search_message: str = ''
    diis_history_size: int = 0
    diis_condition: float = np.nan
    diis_coefficient_l1: float = np.nan
    diis_damping: float = np.nan
    diis_history_action: str = ''
    diis_predicted_residual_rms: float = np.nan
    diis_trust_ratio: float = np.nan
    diis_next_damping: float = np.nan
    fock_evaluations: int = 0
    line_search_method: str = 'strong-wolfe'
    weak_wolfe: bool = False
    approximate_wolfe: bool = False
    curvature_qualified: bool = False
    line_search_objective_allowance: float = 0.0
    cheap_nelec_evaluations: int = 0
    cheap_nelec_alpha_reductions: int = 0
    residual_filter_active: bool = False
    residual_filter_qualified: bool = False
    residual_filter_ratio: float = np.nan
    residual_filter_rejections: int = 0


@dataclass(frozen=True)
class _GCState:
    h_orth: list
    gamma: list
    eigenvalues: list
    u: list
    occupations: list
    p_orth: list
    dm_ao: cp.ndarray
    veff: Any
    fock_ao: cp.ndarray
    fock_orth: list
    auxiliary_mu: float
    chemical_potential: float
    gauge_shift: float
    electronic_energy: float
    nuclear_energy: float
    dft_total_energy: float
    electron_number: float
    entropy: float
    entropy_energy: float
    free_energy: float
    grand_potential: float
    objective: float
    gradient: list
    z: list
    residual: list
    grad_rms: float
    residual_rms: float


@dataclass(frozen=True)
class _LineSearchResult:
    success: bool
    state: Optional[_GCState]
    alpha: float = 0.0
    nfev: int = 0
    strong_wolfe: bool = False
    force_restart: bool = False
    message: str = ''
    trust_boundary: bool = False
    weak_wolfe: bool = False
    approximate_wolfe: bool = False
    curvature_qualified: bool = False
    objective_allowance: float = 0.0
    line_search_method: str = 'strong-wolfe'
    cheap_nelec_evaluations: int = 0
    cheap_nelec_alpha_reductions: int = 0
    residual_filter_active: bool = False
    residual_filter_qualified: bool = False
    residual_filter_ratio: float = np.nan
    residual_filter_rejections: int = 0


@dataclass(frozen=True)
class _HZPoint:
    alpha: float
    state: Optional[_GCState]
    phi: float
    dphi: float
    failed: bool = False


@dataclass
class _DIISItem:
    fock: list
    residual: list


@dataclass
class _DIISRunContext:
    """Mutable residual-DIIS state that can be advanced more than once."""

    state: _GCState
    previous: Optional[_GCState]
    history: list[_DIISItem]
    damping_hint: float
    niter: int
    next_cycle: int
    converged: bool = False
    message: str = 'residual-DIIS iteration has not started'


@dataclass(frozen=True)
class _DIISStepResult:
    step: _LineSearchResult
    condition: float
    coefficient_l1: float
    history_action: str
    predicted_residual_rms: float


@dataclass(frozen=True)
class _GCWorkspace:
    """Immutable mean-field data shared by every solve in one calculation."""

    mf: Any
    kpts: np.ndarray
    nkpts: int
    weights: cp.ndarray
    s_ao: tuple
    hcore_ao: tuple
    hcore_for_energy: Any
    x_ao2orth: tuple
    nao: int
    north: tuple[int, ...]
    identity: tuple
    ndof: float
    tr_pairs: tuple[tuple[int, int], ...]
    time_reversal_enabled: bool
    checkpoint_fingerprint: str
    nuclear_energy: float
    energy_vhf_keyword: str
    check_time_reversal: bool
    enforce_time_reversal: bool


@dataclass(frozen=True)
class _KernelOutcome:
    """Internal optimizer outcome that has not been published to ``mf``."""

    state: _GCState
    previous: Optional[_GCState]
    converged: bool
    message: str
    niter: int
    density_change: float


@dataclass(frozen=True)
class _FixedNPoint:
    """Private fixed-N result used by canonical continuation."""

    state: _GCState
    converged: bool
    message: str
    niter: int
    nfev: int
    history: tuple[IterationRecord, ...]
    density_change: float

    @property
    def h_orth(self) -> list:
        return self.state.h_orth

    @property
    def fock_orth(self) -> list:
        return self.state.fock_orth

    @property
    def p_orth(self) -> list:
        return self.state.p_orth

    @property
    def dm_ao(self) -> cp.ndarray:
        return self.state.dm_ao

    @property
    def occupations(self) -> list:
        return self.state.occupations

    @property
    def mu(self) -> float:
        return self.state.chemical_potential

    @property
    def electron_number(self) -> float:
        return self.state.electron_number

    @property
    def residual_rms(self) -> float:
        return self.state.residual_rms


@dataclass
class _FixedNSession:
    """One fixed-N solver and its resumable DIIS context."""

    electron_number: float
    solver: Any
    context: _DIISRunContext
    published_history: int = 0
    published_nfev: int = 0
    published_niter: int = 0


@dataclass(frozen=True)
class _CanonicalSample:
    """One converged scalar-root observation and optional live DIIS session."""

    electron_number: float
    mu_error: float
    residual_rms: float
    fock_orth: list
    session: Optional[_FixedNSession] = None


@dataclass
class _CanonicalWork:
    """Canonical continuation work whose Fock accounting has one definition."""

    initialization_nfev: int
    history: list[IterationRecord]
    fixed_n_nfev: int = 0
    fixed_n_niter: int = 0
    verification_nfev: int = 0
    refinements: int = 0

    @property
    def total_nfev(self) -> int:
        return (
            self.initialization_nfev + self.fixed_n_nfev +
            self.verification_nfev)

    @property
    def total_niter(self) -> int:
        return self.fixed_n_niter


@dataclass
class _CanonicalVerification:
    """One-Fock physical verification state and diagnostics."""

    attempts: int = 0
    failures: int = 0
    residual_rms: float = np.nan
    grad_rms: float = np.nan
    delta_nelec: float = np.nan
    density_rms: float = np.nan
    verified_state: Optional[_GCState] = None
    verified_source: Optional[_FixedNPoint] = None
    last_state: Optional[_GCState] = None
    last_source: Optional[_FixedNPoint] = None


@dataclass
class GrandCanonicalResult:
    converged: bool
    message: str
    niter: int
    nfev: int

    mu: float
    sigma: float
    beta: float

    grand_potential: float
    dft_total_energy: float
    electronic_energy: float
    nuclear_energy: float
    entropy: float
    entropy_energy: float
    electron_number: float

    h_orth: list
    fock_orth: list
    dm_ao: cp.ndarray
    p_orth: list
    occupations: list
    mo_coeff: Any
    mo_occ: Any
    mo_energy: Any

    true_gradient_rms: float
    residual_rms: float
    density_change_rms: float
    history: list[IterationRecord]
    veff: Any = None
    checkpoint_path: Optional[str] = None
    free_energy: float = np.nan
    fixed_electron_number: bool = False
    target_electron_number: Optional[float] = None
    cheap_nelec_rejections: int = 0
    canonical_continuation_steps: int = 0
    canonical_continuation_evaluations: int = 0
    canonical_continuation_mu_error: float = np.nan
    canonical_continuation_delta_nelec: float = np.nan
    cheap_nelec_evaluations: int = 0
    cheap_nelec_alpha_reductions: int = 0
    residual_filter_acceptances: int = 0
    residual_filter_rejections: int = 0
    canonical_verification_attempts: int = 0
    canonical_verification_evaluations: int = 0
    canonical_verification_failures: int = 0
    canonical_verification_residual_rms: float = np.nan
    canonical_verification_grad_rms: float = np.nan
    canonical_verification_delta_nelec: float = np.nan
    canonical_verification_density_rms: float = np.nan
    canonical_terminal_mode: str = ''
    canonical_continuation_refinements: int = 0


def _as_float(value: Any, name: str = 'value') -> float:
    """Return a finite real scalar, allowing a CuPy scalar only at the edge."""
    if isinstance(value, cp.ndarray):
        if value.size != 1:
            raise ValueError(f'{name} must be scalar')
        value = value.item()
    value = complex(value)
    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        raise FloatingPointError(f'{name} is nonfinite')
    if abs(value.imag) > 1.0e-9 * max(1.0, abs(value.real)):
        raise ValueError(f'{name} has a significant imaginary part: {value.imag}')
    return float(value.real)


def _blocks(values: Any, name: str = 'matrix blocks') -> list:
    """Normalise a stacked array or a sequence to a list of CuPy matrices."""
    if isinstance(values, cp.ndarray) and values.ndim == 3:
        values = [values[k] for k in range(values.shape[0])]
    elif isinstance(values, (list, tuple)):
        values = list(values)
    else:
        array = cp.asarray(values)
        if array.ndim != 3:
            raise ValueError(f'{name} must be a stack or list of rank-two matrices')
        values = [array[k] for k in range(array.shape[0])]
    result = [cp.asarray(value) for value in values]
    if not result or any(value.ndim != 2 for value in result):
        raise ValueError(f'{name} must contain at least one rank-two matrix')
    return result


def _stack_or_list(values: Sequence) -> Any:
    if not values:
        return []
    if all(value.shape == values[0].shape for value in values):
        return cp.stack(values)
    return list(values)


def fermi_occupations(gamma: cp.ndarray) -> cp.ndarray:
    """Evaluate ``1 / (1 + exp(gamma))`` without exponential overflow."""
    gamma = cp.asarray(gamma)
    if _cupy_expit is not None:
        return _cupy_expit(-gamma)
    positive = gamma >= 0
    out = cp.empty_like(gamma, dtype=cp.result_type(gamma, cp.float64))
    exp_neg = cp.exp(-gamma[positive])
    out[positive] = exp_neg / (1.0 + exp_neg)
    exp_pos = cp.exp(gamma[~positive])
    out[~positive] = 1.0 / (1.0 + exp_pos)
    return out


def fermi_entropy(gamma: cp.ndarray, occupations: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Return ``q log(q) + (1-q) log(1-q)`` stably, element by element."""
    gamma = cp.asarray(gamma)
    q = fermi_occupations(gamma) if occupations is None else cp.asarray(occupations)
    positive = gamma >= 0
    entropy = cp.empty_like(q, dtype=cp.result_type(q, cp.float64))
    entropy[positive] = (-q[positive] * gamma[positive]
                         - cp.logaddexp(0.0, -gamma[positive]))
    entropy[~positive] = ((1.0 - q[~positive]) * gamma[~positive]
                          - cp.logaddexp(0.0, gamma[~positive]))
    return entropy


def fermi_divided_difference(gamma: cp.ndarray, occupations: cp.ndarray,
                             rtol: float = 1.0e-10) -> cp.ndarray:
    """The Hermitian Fr\u00e9chet divided-difference matrix of the Fermi function."""
    gamma = cp.asarray(gamma)
    q = cp.asarray(occupations)
    gi = gamma[:, None]
    gj = gamma[None, :]
    qi = q[:, None]
    qj = q[None, :]
    delta = gi - gj
    tol = rtol * cp.maximum(1.0, cp.maximum(cp.abs(gi), cp.abs(gj)))
    regular = cp.abs(delta) > tol
    safe_delta = cp.where(regular, delta, 1.0)
    midpoint_q = fermi_occupations(0.5 * (gi + gj))
    value = cp.where(regular, (qi - qj) / safe_delta,
                     -midpoint_q * (1.0 - midpoint_q))
    diag = -q * (1.0 - q)
    idx = cp.arange(gamma.size)
    value[idx, idx] = diag
    # Numerical roundoff can make a theoretically non-positive value positive.
    return cp.minimum(0.0, 0.5 * (value + value.T)).real


class GrandCanonicalKRKS:
    """Minimise finite-temperature KRKS at fixed ``mu`` or electron number.

    The object composes a regular GPU4PySCF ``KRKS`` instance.  It intentionally
    never invokes orbital-rotation or CIAH machinery.  Pass ``electron_number``
    instead of ``mu`` to minimize the Helmholtz free energy while solving for
    the chemical potential at every evaluation.
    """

    def __init__(self, mf: Any, mu: Optional[float] = None,
                 sigma: Optional[float] = None,
                 config: Optional[GrandCanonicalConfig] = None,
                 electron_number: Optional[float] = None,
                 _workspace: Optional[_GCWorkspace] = None):
        self.mf = mf
        if sigma is None:
            raise TypeError('sigma is required')
        self.sigma = _as_float(sigma, 'sigma')
        if self.sigma <= 0.0:
            raise ValueError('sigma must be positive')
        self.fixed_electron_number = electron_number is not None
        self.target_electron_number = (
            None if electron_number is None
            else _as_float(electron_number, 'electron_number'))
        if self.fixed_electron_number:
            self.mu = None if mu is None else _as_float(mu, 'mu')
        else:
            if mu is None:
                raise TypeError('mu is required unless electron_number is specified')
            self.mu = _as_float(mu, 'mu')
        self.beta = 1.0 / self.sigma
        self.config = config or GrandCanonicalConfig()
        self.config.cg_update = self._canonical_cg_update(self.config.cg_update)
        self.config.line_search_method = self._canonical_line_search_method(
            self.config.line_search_method)
        self._validate_line_search_config()
        self._validate_diis_config()
        self._validate_nelec_guard_config()
        self._validate_canonical_continuation_config()
        if (self.config.nlcg_residual_filter_rms is not None and
                self.config.line_search_method != 'hager-zhang'):
            raise ValueError(
                'the NLCG residual filter requires Hager-Zhang')
        self.verbose = (getattr(mf, 'verbose', logger.NOTE)
                        if self.config.verbose is None else self.config.verbose)
        self.log = logger.new_logger(mf, self.verbose)
        self.history: list[IterationRecord] = []
        self.nfev = 0
        self.ncheap_nelec_reject = 0
        self._last_trial_rejected_by_nelec = False
        self.ncheap_nelec_evaluations = 0
        self.ncheap_nelec_alpha_reductions = 0
        self.nresidual_filter_acceptances = 0
        self.nresidual_filter_rejections = 0
        self.canonical_verification_attempts = 0
        self.canonical_verification_evaluations = 0
        self.canonical_verification_failures = 0
        self.canonical_verification_residual_rms = np.nan
        self.canonical_verification_grad_rms = np.nan
        self.canonical_verification_delta_nelec = np.nan
        self.canonical_verification_density_rms = np.nan
        self.canonical_terminal_mode = ''
        self._nlcg_residual_previous_alpha: Optional[float] = None
        if _workspace is None:
            self._prepare_fixed_basis_data()
            self._workspace = self._capture_workspace()
        else:
            self._install_workspace(_workspace)
        capacity = 2.0 * sum(float(self.weights[k].item()) * n
                             for k, n in enumerate(self.north))
        if (self.fixed_electron_number and
                not 0.0 < self.target_electron_number < capacity):
            raise ValueError(
                'electron_number must lie strictly between 0 and the '
                f'retained-basis capacity ({capacity:g})')
        if self.fixed_electron_number and self.config.initial_electron_number is not None:
            raise ValueError(
                'initial_electron_number is only meaningful at fixed mu; '
                'electron_number already fixes every canonical density')
        if self.config.initial_electron_number is not None:
            target = _as_float(
                self.config.initial_electron_number,
                'initial_electron_number',
            )
            if not 0.0 < target < capacity:
                raise ValueError(
                    'initial_electron_number must lie strictly between 0 and '
                    f'the retained-basis capacity ({capacity:g})')
            self.config.initial_electron_number = target

    # ---- fixed basis data and validation ---------------------------------

    @staticmethod
    def _canonical_cg_update(value: str) -> str:
        if not isinstance(value, str):
            raise TypeError('cg_update must be a string')
        key = value.strip().lower().replace('_', '-').replace(' ', '-')
        aliases = {
            'fr': 'fletcher-reeves',
            'fletcher-reeves': 'fletcher-reeves',
            'pr': 'polak-ribiere',
            'polak-ribiere': 'polak-ribiere',
            'hs': 'hestenes-stiefel',
            'hestenes-stiefel': 'hestenes-stiefel',
            'hz': 'hager-zhang',
            'hager-zhang': 'hager-zhang',
            'cg-descent': 'hager-zhang',
        }
        try:
            return aliases[key]
        except KeyError as error:
            choices = ', '.join(('fletcher-reeves', 'polak-ribiere',
                                 'hestenes-stiefel', 'hager-zhang'))
            raise ValueError(
                f'unsupported cg_update {value!r}; choose one of {choices}') from error

    @staticmethod
    def _canonical_line_search_method(value: str) -> str:
        if not isinstance(value, str):
            raise TypeError('line_search_method must be a string')
        key = value.strip().lower().replace('_', '-').replace(' ', '-')
        aliases = {
            'strong': 'strong-wolfe',
            'wolfe': 'strong-wolfe',
            'strong-wolfe': 'strong-wolfe',
            'hz': 'hager-zhang',
            'hager-zhang': 'hager-zhang',
            'cg-descent': 'hager-zhang',
        }
        try:
            return aliases[key]
        except KeyError as error:
            raise ValueError(
                'line_search_method must be strong-wolfe or hager-zhang') from error

    def _validate_line_search_config(self) -> None:
        c1 = self.config.line_search_c1
        c2 = self.config.line_search_c2
        if (not np.isfinite(c1) or not np.isfinite(c2) or
                not 0.0 < c1 < c2 < 1.0):
            raise ValueError(
                'line-search constants must satisfy 0 < line_search_c1 < '
                'line_search_c2 < 1')
        for name in ('line_search_max_evals', 'line_search_zoom_evals',
                     'line_search_max_trials',
                     'line_search_nelec_alpha_bisections',
                     'hager_zhang_max_evals'):
            value = getattr(self.config, name)
            if (not isinstance(value, int) or isinstance(value, bool) or
                    value < 1):
                raise ValueError(f'{name} must be a positive integer')
        for name in ('line_search_alpha_init', 'line_search_alpha_cap',
                     'line_search_alpha_min', 'line_search_growth',
                     'line_search_max_h_rms_step'):
            value = getattr(self.config, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if self.config.line_search_growth <= 1.0:
            raise ValueError('line_search_growth must exceed 1')
        backtrack = self.config.armijo_backtrack_factor
        if not np.isfinite(backtrack) or not 0.0 < backtrack < 1.0:
            raise ValueError(
                'armijo_backtrack_factor must lie strictly between 0 and 1')
        delta = self.config.hager_zhang_delta
        sigma = self.config.hager_zhang_sigma
        if (not np.isfinite(delta) or not np.isfinite(sigma) or
                not 0.0 < delta < 0.5 or not delta < sigma < 1.0):
            raise ValueError(
                'Hager-Zhang constants must satisfy 0 < delta < 0.5 and '
                'delta < sigma < 1')
        expansion = self.config.hager_zhang_expansion
        shrinkage = self.config.hager_zhang_shrinkage
        if not np.isfinite(expansion) or expansion <= 1.0:
            raise ValueError('hager_zhang_expansion must exceed 1')
        if (not np.isfinite(shrinkage) or
                not 0.0 < shrinkage < 1.0):
            raise ValueError(
                'hager_zhang_shrinkage must lie strictly between 0 and 1')
        noise = self.config.hager_zhang_objective_noise
        if not np.isfinite(noise) or noise < 0.0:
            raise ValueError(
                'hager_zhang_objective_noise must be finite and nonnegative')
        theta = self.config.hager_zhang_theta
        if not np.isfinite(theta) or theta <= 0.25:
            raise ValueError('hager_zhang_theta must exceed 0.25')
        for name in ('line_search_nelec_feasible_alpha',
                     'nlcg_exact_gradient_blend',
                     'nlcg_exact_gradient_polish',
                     'nlcg_residual_filter_warm_start'):
            if not isinstance(getattr(self.config, name), bool):
                raise TypeError(f'{name} must be boolean')
        filter_rms = self.config.nlcg_residual_filter_rms
        if filter_rms is not None:
            filter_rms = _as_float(
                filter_rms, 'nlcg_residual_filter_rms')
            if filter_rms <= 0.0:
                raise ValueError(
                    'nlcg_residual_filter_rms must be positive when enabled')
            self.config.nlcg_residual_filter_rms = filter_rms
        increase = self.config.nlcg_residual_filter_max_relative_increase
        if (not np.isfinite(increase) or
                not 0.0 <= increase < 1.0):
            raise ValueError(
                'nlcg_residual_filter_max_relative_increase must lie in '
                '[0, 1)')
        reduction = self.config.nlcg_residual_filter_min_relative_reduction
        if (not np.isfinite(reduction) or
                not 0.0 < reduction < 1.0):
            raise ValueError(
                'nlcg_residual_filter_min_relative_reduction must lie '
                'strictly between 0 and 1')
        filter_noise = self.config.nlcg_residual_filter_objective_noise
        if not np.isfinite(filter_noise) or filter_noise < 0.0:
            raise ValueError(
                'nlcg_residual_filter_objective_noise must be finite and '
                'nonnegative')
        warm_alpha_names = (
            'nlcg_residual_filter_initial_alpha',
            'nlcg_residual_filter_alpha_min',
            'nlcg_residual_filter_alpha_max',
        )
        for name in warm_alpha_names:
            value = getattr(self.config, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        warm_min = self.config.nlcg_residual_filter_alpha_min
        warm_initial = self.config.nlcg_residual_filter_initial_alpha
        warm_max = self.config.nlcg_residual_filter_alpha_max
        if not warm_min <= warm_initial <= warm_max:
            raise ValueError(
                'residual-filter alpha bounds must satisfy alpha_min <= '
                'initial_alpha <= alpha_max')
        filter_max_evals = self.config.nlcg_residual_filter_max_evals
        if (filter_max_evals is not None and
                (not isinstance(filter_max_evals, int) or
                 isinstance(filter_max_evals, bool) or
                 filter_max_evals < 1)):
            raise ValueError(
                'nlcg_residual_filter_max_evals must be a positive integer '
                'or None')
        if (filter_rms is None and
                (self.config.nlcg_residual_filter_warm_start or
                 filter_max_evals is not None)):
            raise ValueError(
                'residual-filter warm starts and evaluation caps require '
                'nlcg_residual_filter_rms')

    def _validate_diis_config(self) -> None:
        switch = self.config.diis_switch_residual_rms
        if switch is not None:
            switch = _as_float(switch, 'diis_switch_residual_rms')
            if switch <= 0.0:
                raise ValueError(
                    'diis_switch_residual_rms must be positive when enabled')
            if switch < self.config.conv_tol_residual_rms:
                raise ValueError(
                    'diis_switch_residual_rms may not be smaller than '
                    'conv_tol_residual_rms')
            self.config.diis_switch_residual_rms = switch
        for name, minimum in (
                ('diis_space', 2), ('diis_max_backtracks', 0)):
            value = getattr(self.config, name)
            if (not isinstance(value, int) or isinstance(value, bool) or
                    value < minimum):
                relation = 'at least 2' if minimum == 2 else 'nonnegative'
                raise ValueError(f'{name} must be an integer that is {relation}')
        positive = (
            'diis_regularization', 'diis_max_condition',
            'diis_max_coefficient_l1', 'diis_max_objective_increase',
            'diis_max_delta_nelec',
        )
        for name in positive:
            value = getattr(self.config, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        reduction = self.config.diis_min_residual_reduction
        if not np.isfinite(reduction) or not 0.0 <= reduction < 1.0:
            raise ValueError(
                'diis_min_residual_reduction must lie in [0, 1)')
        factor = self.config.diis_backtrack_factor
        if not np.isfinite(factor) or not 0.0 < factor < 1.0:
            raise ValueError('diis_backtrack_factor must lie strictly between 0 and 1')
        initial_damping = self.config.diis_initial_damping
        if (not np.isfinite(initial_damping) or
                not 0.0 < initial_damping <= 1.0):
            raise ValueError('diis_initial_damping must lie in (0, 1]')
        shrink = self.config.diis_trust_shrink_ratio
        expand = self.config.diis_trust_expand_ratio
        if (not np.isfinite(shrink) or not np.isfinite(expand) or
                not 0.0 <= shrink < expand <= 1.0):
            raise ValueError(
                'DIIS trust ratios must satisfy 0 <= shrink < expand <= 1')
        expansion = self.config.diis_trust_expansion
        if not np.isfinite(expansion) or expansion <= 1.0:
            raise ValueError('diis_trust_expansion must be finite and exceed 1')
        min_expand_reduction = (
            self.config.diis_trust_expand_min_relative_reduction)
        if (not np.isfinite(min_expand_reduction) or
                not 0.0 <= min_expand_reduction < 1.0):
            raise ValueError(
                'diis_trust_expand_min_relative_reduction must lie in [0, 1)')

    def _validate_nelec_guard_config(self) -> None:
        threshold = self.config.line_search_nelec_guard_residual_rms
        if threshold is not None:
            threshold = _as_float(
                threshold, 'line_search_nelec_guard_residual_rms')
            if threshold <= 0.0:
                raise ValueError(
                    'line_search_nelec_guard_residual_rms must be positive '
                    'when enabled')
            self.config.line_search_nelec_guard_residual_rms = threshold
        for name in ('line_search_max_delta_nelec',
                     'line_search_nelec_guard_max_delta_nelec'):
            maximum = getattr(self.config, name)
            if not np.isfinite(maximum) or maximum <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if (self.config.line_search_nelec_guard_max_delta_nelec >
                self.config.line_search_max_delta_nelec):
            raise ValueError(
                'line_search_nelec_guard_max_delta_nelec may not exceed '
                'line_search_max_delta_nelec')
    def _validate_canonical_continuation_config(self) -> None:
        if not isinstance(self.config.canonical_continuation, bool):
            raise TypeError('canonical_continuation must be boolean')
        max_outer = self.config.canonical_continuation_max_outer
        if (not isinstance(max_outer, int) or isinstance(max_outer, bool) or
                max_outer < 1):
            raise ValueError(
                'canonical_continuation_max_outer must be a positive integer')
        positive = (
            'canonical_continuation_coarse_residual_tol',
            'canonical_continuation_bracketed_residual_tol',
            'canonical_continuation_handoff_delta_nelec',
            'canonical_continuation_unbracketed_handoff_delta_nelec',
            'canonical_continuation_handoff_delta_mu',
            'canonical_continuation_initial_delta_nelec',
            'canonical_continuation_max_delta_nelec_fraction',
            'canonical_continuation_min_delta_nelec',
            'canonical_continuation_initial_damping',
            'canonical_continuation_diis_max_coefficient_l1',
            'canonical_continuation_verification_residual_tol',
            'canonical_continuation_verification_density_tol',
            'canonical_continuation_root_nelec_tol',
        )
        for name in positive:
            value = getattr(self.config, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        absolute_cap = self.config.canonical_continuation_max_delta_nelec
        if (absolute_cap is not None and
                (not np.isfinite(absolute_cap) or absolute_cap <= 0.0)):
            raise ValueError(
                'canonical_continuation_max_delta_nelec must be finite and '
                'positive when supplied')
        if (self.config.canonical_continuation_bracketed_residual_tol >
                self.config.canonical_continuation_coarse_residual_tol):
            raise ValueError(
                'canonical_continuation_bracketed_residual_tol may not '
                'exceed canonical_continuation_coarse_residual_tol')
        if (self.config.canonical_continuation_min_delta_nelec >
                self.config.canonical_continuation_initial_delta_nelec):
            raise ValueError(
                'canonical_continuation_min_delta_nelec may not exceed '
                'canonical_continuation_initial_delta_nelec')
        if (self.config.canonical_continuation_unbracketed_handoff_delta_nelec >
                self.config.canonical_continuation_handoff_delta_nelec):
            raise ValueError(
                'canonical_continuation_unbracketed_handoff_delta_nelec may '
                'not exceed canonical_continuation_handoff_delta_nelec')
        if self.config.canonical_continuation_max_delta_nelec_fraction > 1.0:
            raise ValueError(
                'canonical_continuation_max_delta_nelec_fraction may not '
                'exceed 1')
        if self.config.canonical_continuation_initial_damping > 1.0:
            raise ValueError(
                'canonical_continuation_initial_damping may not exceed 1')
    def _prepare_fixed_basis_data(self) -> None:
        required = ('cell', 'kpts', 'get_ovlp', 'get_hcore', 'get_veff',
                    'energy_elec', 'energy_nuc', 'get_init_guess',
                    'check_linear_dependency')
        missing = [name for name in required if not hasattr(self.mf, name)]
        if missing:
            raise TypeError('mf is not KRKS-compatible; missing ' + ', '.join(missing))
        if hasattr(self.mf, 'istype') and not self.mf.istype('KRKS'):
            raise TypeError('GrandCanonicalKRKS requires a restricted periodic KRKS object')
        if getattr(self.mf, 'smearing_method', None):
            raise ValueError('remove the standard smearing decorator before grand-canonical optimisation')
        existing_sigma = getattr(self.mf, 'sigma', 0.0)
        if existing_sigma not in (None, 0, 0.0):
            raise ValueError('mf already has nonzero smearing; this would double count entropy')

        self._validate_functional()
        kpts = self.mf.kpts
        if not isinstance(kpts, np.ndarray):
            raise NotImplementedError('symmetry-reduced KPoints objects are not supported')
        self.kpts = np.asarray(kpts, dtype=float)
        if self.kpts.ndim != 2 or self.kpts.shape[1] != 3 or self.kpts.shape[0] == 0:
            raise ValueError('kpts must be a full array with shape (nkpts, 3)')
        self.nkpts = len(self.kpts)
        self.weights = cp.full(self.nkpts, 1.0 / self.nkpts, dtype=cp.float64)

        # Build is conventional setup, not an SCF iteration.  Mock evaluators
        # in the unit tests need not provide it.
        if hasattr(self.mf, 'build'):
            self.mf.build()
        self.s_ao = _blocks(self.mf.get_ovlp(self.mf.cell, self.mf.kpts), 'overlap')
        self.hcore_ao = _blocks(self.mf.get_hcore(self.mf.cell, self.mf.kpts), 'hcore')
        if len(self.s_ao) != self.nkpts or len(self.hcore_ao) != self.nkpts:
            raise ValueError('overlap, hcore, and k-point counts differ')
        try:
            x_raw = self.mf.check_linear_dependency(
                _stack_or_list(self.s_ao),
                time_reversal_symmetry=getattr(self.mf, 'time_reversal_symmetry', False))
        except TypeError:  # GPU4PySCF versions before the keyword was added
            x_raw = self.mf.check_linear_dependency(_stack_or_list(self.s_ao))
        self.x_ao2orth = _blocks(x_raw, 'orthogonalizers')
        if len(self.x_ao2orth) != self.nkpts:
            raise ValueError('orthogonalizer and k-point counts differ')
        self.nao = self.s_ao[0].shape[0]
        self.north = []
        self.identity = []
        for k, (s, h, x) in enumerate(zip(self.s_ao, self.hcore_ao, self.x_ao2orth)):
            if s.shape != (self.nao, self.nao) or h.shape != s.shape:
                raise ValueError(f'inconsistent AO dimensions at k-point {k}')
            if x.shape[0] != self.nao or x.shape[1] == 0:
                raise ValueError(f'invalid orthogonalizer at k-point {k}')
            overlap_error = cp.max(cp.abs(x.conj().T @ s @ x - cp.eye(x.shape[1]))).item()
            if overlap_error > self.config.orthogonality_tol:
                raise ValueError(f'X^H S X is not identity at k-point {k}: {overlap_error:g}')
            self.north.append(x.shape[1])
            self.identity.append(cp.eye(x.shape[1], dtype=x.dtype))
        self.ndof = sum(float(self.weights[k].item()) * n * n
                         for k, n in enumerate(self.north))
        self._tr_pairs, self._time_reversal_enabled = self._initialise_time_reversal()
        self._checkpoint_fingerprint = self._mean_field_fingerprint()

    def _capture_workspace(self) -> _GCWorkspace:
        """Freeze setup data that is invariant throughout one solver run."""
        energy_parameters = inspect.signature(self.mf.energy_elec).parameters
        energy_vhf_keyword = (
            'vhf' if ('vhf' in energy_parameters or
                      any(parameter.kind == inspect.Parameter.VAR_KEYWORD
                          for parameter in energy_parameters.values()))
            else 'vhf_kpts')
        nuclear_energy = _as_float(
            self.mf.energy_nuc(), 'nuclear energy')
        return _GCWorkspace(
            mf=self.mf,
            kpts=self.kpts,
            nkpts=self.nkpts,
            weights=self.weights,
            s_ao=tuple(self.s_ao),
            hcore_ao=tuple(self.hcore_ao),
            hcore_for_energy=_stack_or_list(self.hcore_ao),
            x_ao2orth=tuple(self.x_ao2orth),
            nao=self.nao,
            north=tuple(self.north),
            identity=tuple(self.identity),
            ndof=self.ndof,
            tr_pairs=tuple(self._tr_pairs),
            time_reversal_enabled=self._time_reversal_enabled,
            checkpoint_fingerprint=self._checkpoint_fingerprint,
            nuclear_energy=nuclear_energy,
            energy_vhf_keyword=energy_vhf_keyword,
            check_time_reversal=self.config.check_time_reversal,
            enforce_time_reversal=self.config.enforce_time_reversal,
        )

    def _install_workspace(self, workspace: _GCWorkspace) -> None:
        """Install a validated workspace without rebuilding immutable data."""
        if not isinstance(workspace, _GCWorkspace):
            raise TypeError('_workspace must be a _GCWorkspace')
        if workspace.mf is not self.mf:
            raise ValueError('a shared workspace belongs to a different mf')
        if (workspace.check_time_reversal !=
                self.config.check_time_reversal or
                workspace.enforce_time_reversal !=
                self.config.enforce_time_reversal):
            raise ValueError(
                'shared workspace time-reversal settings do not match config')
        self._workspace = workspace
        self.kpts = workspace.kpts
        self.nkpts = workspace.nkpts
        self.weights = workspace.weights
        self.s_ao = list(workspace.s_ao)
        self.hcore_ao = list(workspace.hcore_ao)
        self.x_ao2orth = list(workspace.x_ao2orth)
        self.nao = workspace.nao
        self.north = list(workspace.north)
        self.identity = list(workspace.identity)
        self.ndof = workspace.ndof
        self._tr_pairs = list(workspace.tr_pairs)
        self._time_reversal_enabled = workspace.time_reversal_enabled
        self._checkpoint_fingerprint = workspace.checkpoint_fingerprint

    def _spawn_fixed_n(
            self, electron_number: float,
            config: GrandCanonicalConfig) -> 'GrandCanonicalKRKS':
        """Create a fixed-N run context sharing all immutable mf setup."""
        if self.fixed_electron_number:
            raise AssertionError('fixed-N child requires a fixed-mu parent')
        return GrandCanonicalKRKS(
            self.mf, mu=self.mu, sigma=self.sigma, config=config,
            electron_number=electron_number, _workspace=self._workspace)

    def _mean_field_fingerprint(self) -> str:
        """A conservative restart guard for cell, basis, and DFT configuration."""
        cell = self.mf.cell
        try:
            cell_data = cell.dumps()
        except (AttributeError, TypeError):
            cell_data = repr((type(cell).__module__, type(cell).__qualname__,
                              getattr(cell, 'atom', None), getattr(cell, 'basis', None),
                              getattr(cell, 'pseudo', None), getattr(cell, 'a', None),
                              getattr(cell, 'unit', None)))
        configuration = repr((type(self.mf).__module__, type(self.mf).__qualname__,
                              cell_data, getattr(self.mf, 'xc', None),
                              getattr(self.mf, 'exxdiv', None), self.kpts.tolist(),
                              tuple(self.north)))
        return hashlib.sha256(configuration.encode()).hexdigest()

    def _validate_functional(self) -> None:
        if getattr(self.mf, 'do_nlc', lambda: False)():
            raise NotImplementedError('nonlocal correlation is not supported by GrandCanonicalKRKS')
        ni = getattr(self.mf, '_numint', None)
        libxc = getattr(ni, 'libxc', None)
        xc = getattr(self.mf, 'xc', None)
        if libxc is None or xc is None:
            return  # lightweight fixed-Fock test evaluators
        if libxc.is_hybrid_xc(xc):
            raise NotImplementedError('hybrid and range-separated functionals are not supported')
        # gpu4pyscf.dft.libxc intentionally exposes only the GPU entry points;
        # the functional-family classifier remains in the PySCF libxc module.
        from pyscf.dft import libxc as pyscf_libxc
        xctype = pyscf_libxc.xc_type(xc).upper()
        if xctype not in ('LDA', 'GGA'):
            raise NotImplementedError(f'{xctype} functionals are not supported')

    def _find_time_reversal_pairs(self) -> list[tuple[int, int]]:
        if hasattr(self.mf, 'iter_kpt_pairs'):
            pairs = []
            for pair in self.mf.iter_kpt_pairs():
                if len(pair) >= 2 and np.isscalar(pair[0]) and np.isscalar(pair[1]):
                    i, j = int(pair[0]), int(pair[1])
                    if i != j and i < j:
                        pairs.append((i, j))
            if pairs:
                return pairs
        try:
            scaled = np.asarray(self.mf.cell.get_scaled_kpts(self.kpts), dtype=float)
        except (AttributeError, TypeError):
            scaled = self.kpts
        pairs = []
        for i in range(self.nkpts):
            candidates = []
            for j in range(self.nkpts):
                delta = scaled[i] + scaled[j]
                delta -= np.rint(delta)
                candidates.append(np.linalg.norm(delta))
            j = int(np.argmin(candidates))
            if candidates[j] < 1.0e-8 and i < j:
                pairs.append((i, j))
        return pairs

    def _initialise_time_reversal(self) -> tuple[list[tuple[int, int]], bool]:
        pairs = self._find_time_reversal_pairs()
        if not self.config.check_time_reversal or not self.config.enforce_time_reversal:
            return pairs, False
        valid = True
        for i, j in pairs:
            if self.north[i] != self.north[j]:
                valid = False
                break
            errors = (
                cp.max(cp.abs(self.s_ao[j] - self.s_ao[i].conj())).item(),
                cp.max(cp.abs(self.hcore_ao[j] - self.hcore_ao[i].conj())).item(),
                cp.max(cp.abs(self.x_ao2orth[j] - self.x_ao2orth[i].conj())).item(),
            )
            if max(errors) > self.config.orthogonality_tol:
                valid = False
                break
        if not valid:
            self.log.warn('GrandCanonicalKRKS: time-reversal gauge check failed; projection disabled')
        return pairs, valid

    # ---- block algebra -----------------------------------------------------

    def copy_blocks(self, a: Sequence) -> list:
        return [x.copy() for x in a]

    def zeros_like_blocks(self, a: Sequence) -> list:
        return [cp.zeros_like(x) for x in a]

    def axpy(self, alpha: float, x: Sequence, y: Sequence) -> list:
        return [alpha * a + b for a, b in zip(x, y)]

    def scale_blocks(self, alpha: float, x: Sequence) -> list:
        return [alpha * a for a in x]

    def hermitize_blocks(self, blocks: Sequence) -> list:
        return [0.5 * (x + x.conj().T) for x in blocks]

    def project_time_reversal(self, blocks: Sequence) -> tuple[list, float]:
        result = self.copy_blocks(blocks)
        if not self._time_reversal_enabled:
            return result, 0.0
        before = self.copy_blocks(result)
        for i, j in self._tr_pairs:
            pair_average = 0.5 * (result[i] + result[j].conj())
            result[i] = pair_average
            result[j] = pair_average.conj()
        return result, self.norm(self.axpy(-1.0, before, result))

    def all_finite(self, blocks: Sequence) -> bool:
        return all(bool(cp.all(cp.isfinite(x)).item()) for x in blocks)

    def inner(self, a: Sequence, b: Sequence) -> float:
        return sum(float((self.weights[k] * cp.vdot(x, y).real).item())
                   for k, (x, y) in enumerate(zip(a, b)))

    def norm(self, a: Sequence) -> float:
        return max(0.0, self.inner(a, a)) ** 0.5

    def rms(self, a: Sequence) -> float:
        return (self.inner(a, a) / self.ndof) ** 0.5

    def max_block_rms(self, a: Sequence) -> float:
        return max((float(cp.linalg.norm(x).item()) / x.shape[0] for x in a), default=0.0)

    def trace_mean(self, blocks: Sequence) -> float:
        numerator = sum(float((self.weights[k] * cp.trace(value).real).item())
                        for k, value in enumerate(blocks))
        denominator = sum(float(self.weights[k].item()) * value.shape[0]
                          for k, value in enumerate(blocks))
        return numerator / denominator

    # ---- state construction ------------------------------------------------

    def _sanitize_h(self, h_orth: Sequence) -> list:
        h = _blocks(h_orth, 'h_orth')
        if len(h) != self.nkpts:
            raise ValueError('h_orth and k-point counts differ')
        for k, matrix in enumerate(h):
            if matrix.shape != (self.north[k], self.north[k]):
                raise ValueError(f'h_orth[{k}] has shape {matrix.shape}; expected '
                                 f'({self.north[k]}, {self.north[k]})')
        if not self.all_finite(h):
            raise FloatingPointError('auxiliary Hamiltonian contains nonfinite elements')
        h = self.hermitize_blocks(h)
        h, _ = self.project_time_reversal(h)
        h = self.hermitize_blocks(h)
        return h

    def _solve_chemical_potential(self, orbital_energies: Sequence) -> float:
        """Find mu such that the Fermi occupations have the target electron count."""
        target = self.target_electron_number
        margin = max(1.0, 50.0 * self.sigma)
        lower = min(float(cp.min(value).item()) for value in orbital_energies) - margin
        upper = max(float(cp.max(value).item()) for value in orbital_energies) + margin

        def electron_number(mu: float) -> float:
            return 2.0 * sum(
                float((self.weights[k] * cp.sum(fermi_occupations(
                    self.beta * (value - mu)))).item())
                for k, value in enumerate(orbital_energies))

        midpoint = 0.5 * (lower + upper)
        for _ in range(self.config.mu_max_cycle):
            midpoint = 0.5 * (lower + upper)
            nelec = electron_number(midpoint)
            if abs(nelec - target) <= self.config.mu_electron_number_tol:
                return midpoint
            if nelec < target:
                lower = midpoint
            else:
                upper = midpoint
        nelec = electron_number(midpoint)
        if abs(nelec - target) > self.config.mu_electron_number_tol:
            raise RuntimeError(
                'chemical-potential solve did not reach the target electron '
                f'number: {nelec:.15g} vs {target:.15g}')
        return midpoint

    def _thermal_density(self, h: Sequence) -> tuple[list, list, list, list,
                                                            list, cp.ndarray, float]:
        gamma_blocks, eigvals, eigenvectors, occupations, p_blocks = [], [], [], [], []
        h_eigenpairs = [cp.linalg.eigh(hk) for hk in h]
        orbital_energies = [pair[0] for pair in h_eigenpairs]
        auxiliary_mu = (self._solve_chemical_potential(orbital_energies)
                        if self.fixed_electron_number else self.mu)
        dm = cp.empty((self.nkpts, self.nao, self.nao),
                      dtype=cp.result_type(*[x.dtype for x in h], cp.complex128))
        for k, (hk, x, identity, eigenpair) in enumerate(zip(
                h, self.x_ao2orth, self.identity, h_eigenpairs)):
            gamma = self.beta * (hk - auxiliary_mu * identity)
            gamma = 0.5 * (gamma + gamma.conj().T)
            energy, vector = eigenpair
            value = self.beta * (energy - auxiliary_mu)
            q = fermi_occupations(value)
            p = (vector * q[None, :]) @ vector.conj().T
            p = 0.5 * (p + p.conj().T)
            dmk = 2.0 * x @ p @ x.conj().T
            dm[k] = 0.5 * (dmk + dmk.conj().T)
            gamma_blocks.append(gamma)
            eigvals.append(value)
            eigenvectors.append(vector)
            occupations.append(q)
            p_blocks.append(p)
        p_blocks, _ = self.project_time_reversal(p_blocks)
        dm_blocks, _ = self.project_time_reversal([dm[k] for k in range(self.nkpts)])
        dm = cp.stack(self.hermitize_blocks(dm_blocks))
        if not self.all_finite(p_blocks) or not bool(cp.all(cp.isfinite(dm)).item()):
            raise FloatingPointError('thermal density contains nonfinite elements')
        return (gamma_blocks, eigvals, eigenvectors, occupations, p_blocks, dm,
                auxiliary_mu)

    def _electron_number(self, p_blocks: Sequence, dm: cp.ndarray) -> float:
        north = 2.0 * sum(float((self.weights[k] * cp.trace(p).real).item())
                           for k, p in enumerate(p_blocks))
        nao_value = 0.0
        for k, (d, s) in enumerate(zip(dm, self.s_ao)):
            trace = cp.einsum('ij,ji->', d, s)
            trace = _as_float(trace, 'AO electron-count trace')
            nao_value += float(self.weights[k].item()) * trace
        if abs(north - nao_value) > 1.0e-8 * max(1.0, abs(north)):
            raise ValueError(f'AO and orthogonal electron counts disagree: {nao_value} vs {north}')
        return north

    def _entropy(self, gamma: Sequence, occupations: Sequence) -> tuple[float, float]:
        summed = sum(float((self.weights[k] * cp.sum(fermi_entropy(g, q))).item())
                     for k, (g, q) in enumerate(zip(gamma, occupations)))
        entropy = -2.0 * summed
        entropy_energy = -self.sigma * entropy
        return entropy, entropy_energy

    def _to_orth(self, matrices: Sequence) -> list:
        mats = _blocks(matrices, 'AO matrices')
        output = []
        for k, (a, x) in enumerate(zip(mats, self.x_ao2orth)):
            transformed = x.conj().T @ a @ x
            anti = self.max_block_rms([transformed - transformed.conj().T])
            if anti > self.config.hermiticity_tol:
                raise FloatingPointError(f'Fock matrix at k-point {k} is not Hermitian ({anti:g})')
            output.append(0.5 * (transformed + transformed.conj().T))
        return output

    def _fock_from_veff(self, dm: cp.ndarray, veff: Any) -> cp.ndarray:
        """Build the full AO Fock matrix represented by a tagged potential.

        Ordinary KRKS potentials are complete arrays, so ``hcore + veff`` is
        authoritative.  Decorators such as PeriodicLPBE attach an additional
        response potential and expose the matched assembly through get_fock.
        Calling it at cycle=-1 bypasses DIIS, damping, and level shifting.
        """
        hcore = self._workspace.hcore_for_energy
        if getattr(veff, 'v_solvent', None) is not None:
            if not hasattr(self.mf, 'get_fock'):
                raise TypeError('tagged solvent potential requires mf.get_fock')
            fock = self.mf.get_fock(
                h1e=hcore, vhf=veff, dm=dm, cycle=-1, diis=None,
                level_shift_factor=0.0, damp_factor=0.0)
            return cp.stack(_blocks(fock, 'decorated Fock matrices'))
        return cp.stack([hcore_k + cp.asarray(veff)[k]
                         for k, hcore_k in enumerate(self.hcore_ao)])

    def _exact_gradient(self, h: Sequence, fock: Sequence, eigenvalues: Sequence,
                        eigenvectors: Sequence, occupations: Sequence) -> list:
        gradient, occupation_response = [], []
        for hk, fk, gamma, vector, q in zip(h, fock, eigenvalues, eigenvectors, occupations):
            a_tilde = vector.conj().T @ (fk - hk) @ vector
            divided = fermi_divided_difference(gamma, q, self.config.fermi_divdiff_rtol)
            value = 2.0 * self.beta * vector @ (divided * a_tilde) @ vector.conj().T
            gradient.append(0.5 * (value + value.conj().T))
        gradient, _ = self.project_time_reversal(gradient)
        gradient = self.hermitize_blocks(gradient)
        if self.fixed_electron_number:
            for vector, q in zip(eigenvectors, occupations):
                diagonal = -q * (1.0 - q)
                response = (vector * diagonal[None, :]) @ vector.conj().T
                occupation_response.append(0.5 * (response + response.conj().T))
            occupation_response, _ = self.project_time_reversal(occupation_response)
            occupation_response = self.hermitize_blocks(occupation_response)
            response_trace = sum(
                float((self.weights[k] * cp.trace(value).real).item())
                for k, value in enumerate(occupation_response))
            if abs(response_trace) < 1.0e-30:
                raise RuntimeError('fixed-electron Fermi response is numerically singular')
            gradient_trace = sum(
                float((self.weights[k] * cp.trace(value).real).item())
                for k, value in enumerate(gradient))
            correction = gradient_trace / response_trace
            gradient = [g - correction * response
                        for g, response in zip(gradient, occupation_response)]
            gradient, _ = self.project_time_reversal(gradient)
            gradient = self.hermitize_blocks(gradient)
        return gradient

    def evaluate(self, h_orth: Sequence) -> _GCState:
        """Fully evaluate a density, DFT energy, exact gradient, and residual."""
        self.nfev += 1
        h = self._sanitize_h(h_orth)
        gamma, eigenvalues, vector, q, p, dm, auxiliary_mu = self._thermal_density(h)
        nelec = self._electron_number(p, dm)
        # Keep the tagged potential returned by get_veff alive and pass that
        # exact object to energy_elec.  Converting it for Fock construction does
        # not mutate or replace the tagged object.
        veff = self.mf.get_veff(self.mf.cell, dm, dm_last=None, vhf_last=None,
                                hermi=1, kpts=self.mf.kpts, kpts_band=None)
        fock_ao = self._fock_from_veff(dm, veff)
        # KRKS calls this argument ``vhf`` while KSCF-style decorators, such
        # as PeriodicLPBE, retain the older ``vhf_kpts`` spelling.  In either
        # case the exact tagged object returned above must be passed through.
        electronic_energy, _ = self.mf.energy_elec(
            dm_kpts=dm, h1e_kpts=self._workspace.hcore_for_energy,
            **{self._workspace.energy_vhf_keyword: veff})
        electronic_energy = _as_float(electronic_energy, 'electronic energy')
        nuclear_energy = self._workspace.nuclear_energy
        dft_total_energy = electronic_energy + nuclear_energy
        fock = self._to_orth(fock_ao)
        entropy, entropy_energy = self._entropy(eigenvalues, q)
        free_energy = dft_total_energy + entropy_energy
        gradient = self._exact_gradient(h, fock, eigenvalues, vector, q)
        mismatch = self.hermitize_blocks([hk - fk for hk, fk in zip(h, fock)])
        gauge_shift = self.trace_mean(mismatch) if self.fixed_electron_number else 0.0
        if self.fixed_electron_number:
            mismatch = [value - gauge_shift * identity
                        for value, identity in zip(mismatch, self.identity)]
        chemical_potential = auxiliary_mu - gauge_shift
        omega = free_energy - chemical_potential * nelec
        objective = free_energy if self.fixed_electron_number else omega
        if not all(np.isfinite(value) for value in
                   (free_energy, omega, objective, chemical_potential)):
            raise FloatingPointError('finite-temperature objective is nonfinite')
        z = self.hermitize_blocks([0.5 * value for value in mismatch])
        residual = self.scale_blocks(-1.0, z)
        descent = self.inner(gradient, residual)
        if descent > 1.0e-10 * max(1.0, self.norm(gradient) * self.norm(residual)):
            raise RuntimeError('exact gradient and residual have inconsistent descent signs')
        return _GCState(
            h, gamma, eigenvalues, vector, q, p, dm, veff, fock_ao, fock,
            auxiliary_mu, chemical_potential, gauge_shift, electronic_energy,
            nuclear_energy, dft_total_energy, nelec, entropy, entropy_energy,
            free_energy, omega, objective, gradient, z, residual,
            self.rms(gradient), self.rms(mismatch))

    # ---- initialisation and checkpointing ---------------------------------

    def _initial_h_from_dm(self, dm: Any) -> list:
        # This is a genuine Fock build even though it precedes the first
        # objective evaluation.  Count it so result.nfev is the total number
        # of expensive Fock constructions from a fresh density guess.
        self.nfev += 1
        dm_blocks = self.hermitize_blocks(_blocks(dm, 'initial density'))
        if len(dm_blocks) != self.nkpts:
            raise ValueError('initial density and k-point counts differ')
        dm_blocks, _ = self.project_time_reversal(dm_blocks)
        dm_stack = cp.stack(self.hermitize_blocks(dm_blocks))
        veff = self.mf.get_veff(self.mf.cell, dm_stack, dm_last=None, vhf_last=None,
                                hermi=1, kpts=self.mf.kpts, kpts_band=None)
        fock = self._to_orth(self._fock_from_veff(dm_stack, veff))
        return self._sanitize_h(fock)

    def _shift_initial_h_to_nelec(self, h: Sequence, target: float) -> list:
        """Select an initial occupation basin without changing physical ``mu``.

        A scalar shift of the auxiliary Hamiltonian changes its initial Fermi
        occupations but is still an ordinary point in the unconstrained
        optimisation space.  This is useful when a very small ``sigma`` makes
        the Fermi map effectively discontinuous and the unshifted initial Fock
        matrix lands in a different integer-occupation basin.
        """
        h = self._sanitize_h(h)
        eigenvalues = [cp.linalg.eigvalsh(hk) for hk in h]

        def electron_number(shift: float) -> float:
            return 2.0 * sum(
                float((self.weights[k] * cp.sum(fermi_occupations(
                    self.beta * (value + shift - self.mu)))).item())
                for k, value in enumerate(eigenvalues))

        spectral_radius = max(
            [1.0, self.sigma]
            + [float(cp.max(cp.abs(value - self.mu)).item())
               for value in eigenvalues])
        lower, upper = -spectral_radius, spectral_radius
        while electron_number(lower) < target:
            lower *= 2.0
        while electron_number(upper) > target:
            upper *= 2.0
        for _ in range(100):
            midpoint = 0.5 * (lower + upper)
            if electron_number(midpoint) > target:
                lower = midpoint
            else:
                upper = midpoint
        shift = 0.5 * (lower + upper)
        shifted = [hk + shift * identity
                   for hk, identity in zip(h, self.identity)]
        achieved = electron_number(shift)
        self.log.info(
            'GC initial auxiliary shift = %.12g Ha; N = %.12g (target %.12g)',
            shift, achieved, target)
        return self._sanitize_h(shifted)

    def _load_checkpoint_h(self) -> Optional[list]:
        filename = self.config.checkpoint_path
        if not filename:
            return None
        path = Path(filename)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as checkpoint:
            kpts = checkpoint['kpts']
            ranks = checkpoint['ranks']
            if 'fingerprint' not in checkpoint:
                raise ValueError('checkpoint lacks the required mean-field fingerprint')
            if kpts.shape != self.kpts.shape or not np.allclose(kpts, self.kpts):
                raise ValueError('checkpoint k-point mesh does not match this calculation')
            if tuple(ranks.tolist()) != tuple(self.north):
                raise ValueError('checkpoint overlap ranks do not match this calculation')
            if str(checkpoint['fingerprint'].item()) != self._checkpoint_fingerprint:
                raise ValueError('checkpoint cell, basis, or mean-field configuration does not match')
            return [cp.asarray(checkpoint[f'h_{k}']) for k in range(self.nkpts)]

    def _initial_h(self, dm0: Any = None, h0: Any = None) -> list:
        if h0 is not None:
            return self._sanitize_h(h0)
        checkpoint_h = self._load_checkpoint_h()
        if checkpoint_h is not None:
            return self._sanitize_h(checkpoint_h)
        if dm0 is None:
            dm0 = self.mf.get_init_guess(self.mf.cell)
        h = self._initial_h_from_dm(dm0)
        if self.config.initial_electron_number is not None:
            h = self._shift_initial_h_to_nelec(
                h, self.config.initial_electron_number)
        return h

    def _checkpoint(self, state: _GCState, cycle: int, *,
                    force: bool = False) -> None:
        filename = self.config.checkpoint_path
        if not filename or self.config.checkpoint_interval <= 0:
            return
        if not force and cycle % self.config.checkpoint_interval:
            return
        values = {f'h_{k}': h.get() for k, h in enumerate(state.h_orth)}
        values.update({
            'mu': state.chemical_potential, 'auxiliary_mu': state.auxiliary_mu,
            'sigma': self.sigma, 'kpts': self.kpts,
            'weights': cp.asnumpy(self.weights), 'ranks': np.asarray(self.north),
            'x_dims': np.asarray([x.shape for x in self.x_ao2orth]),
            'fingerprint': self._checkpoint_fingerprint,
            'grand_potential': state.grand_potential,
            'free_energy': state.free_energy,
            'fixed_electron_number': self.fixed_electron_number,
            'target_electron_number': (np.nan if self.target_electron_number is None
                                       else self.target_electron_number),
            'electron_number': state.electron_number, 'cycle': cycle,
        })
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        np.savez(filename, **values)

    # ---- nonlinear CG and line search -------------------------------------

    def _is_descent(self, state: _GCState, direction: Sequence) -> bool:
        gnorm, dnorm = self.norm(state.gradient), self.norm(direction)
        if gnorm == 0.0 or dnorm == 0.0:
            return False
        return self.inner(state.gradient, direction) < -self.config.descent_tolerance * gnorm * dnorm

    def _descent_cosine(self, state: _GCState, direction: Sequence) -> float:
        gnorm, dnorm = self.norm(state.gradient), self.norm(direction)
        if gnorm == 0.0 or dnorm == 0.0:
            return 0.0
        return -self.inner(state.gradient, direction) / (gnorm * dnorm)

    def _blend_poorly_aligned_direction(self, state: _GCState,
                                        direction: Sequence) -> tuple[list, bool]:
        """Add frontier sensitivity without discarding the Fock residual."""
        if (self._descent_cosine(state, direction) >=
                self.config.preconditioned_descent_cosine_min):
            return self.copy_blocks(direction), False
        gradient = self.scale_blocks(-1.0, state.gradient)
        gnorm, dnorm = self.norm(gradient), self.norm(direction)
        if gnorm == 0.0 or dnorm == 0.0:
            return self.copy_blocks(direction), False
        blended = self.axpy(dnorm / gnorm, gradient, direction)
        blended = self.hermitize_blocks(blended)
        blended, _ = self.project_time_reversal(blended)
        if self._is_descent(state, blended):
            return blended, True
        return self.copy_blocks(direction), False

    def _restart_direction(self, state: _GCState) -> tuple[list, str]:
        residual = self.copy_blocks(state.residual)
        if self._is_descent(state, residual):
            if self.config.nlcg_exact_gradient_blend:
                residual, blended = self._blend_poorly_aligned_direction(
                    state, residual)
                if blended:
                    return residual, (
                        'restarted with blended residual/exact gradient')
            return residual, 'restarted with preconditioned residual'
        gradient = self.scale_blocks(-1.0, state.gradient)
        if self._is_descent(state, gradient):
            return gradient, 'restarted with exact steepest descent'
        return residual, 'no usable restart direction'

    def _ensure_descent(self, state: _GCState, direction: Sequence) -> tuple[list, bool, str]:
        direction = self.hermitize_blocks(direction)
        direction, projected_change = self.project_time_reversal(direction)
        if projected_change > self.config.hermiticity_tol * max(1.0, self.norm(direction)):
            direction, reason = self._restart_direction(state)
            return direction, True, 'time-reversal projection changed CG direction; ' + reason
        if (self.config.nlcg_exact_gradient_polish and
                state.residual_rms <=
                self.config.exact_gradient_polish_residual_rms):
            gradient = self.scale_blocks(-1.0, state.gradient)
            if self._is_descent(state, gradient):
                return gradient, True, 'exact-gradient final polishing'
        if self._is_descent(state, direction):
            if self.config.nlcg_exact_gradient_blend:
                direction, blended = self._blend_poorly_aligned_direction(
                    state, direction)
                if blended:
                    return direction, True, (
                        'blended poorly aligned direction with exact gradient')
            return direction, False, ''
        direction, reason = self._restart_direction(state)
        if self._is_descent(state, direction):
            return direction, True, reason
        return direction, True, 'restart direction is not downhill'

    def _cg_beta(self, old: _GCState, new: _GCState,
                 old_direction: Sequence) -> tuple[float, str]:
        """Return the selected preconditioned nonlinear-CG update.

        ``z`` is the positive-preconditioned-gradient representation and the
        search direction is formed as ``-z_new + beta * old_direction``.
        """
        update = self.config.cg_update
        old_gz = self.inner(old.gradient, old.z)
        if update == 'fletcher-reeves':
            numerator = self.inner(new.gradient, new.z)
            denominator = old_gz
        elif update == 'hager-zhang':
            delta_gradient = self.axpy(
                -1.0, old.gradient, new.gradient)
            delta_z = self.axpy(-1.0, old.z, new.z)
            denominator = self.inner(old_direction, delta_gradient)
            if (not np.isfinite(denominator) or
                    abs(denominator) <= 1.0e-30):
                return 0.0, 'ill-conditioned Hager Zhang denominator'
            preconditioned_curvature = self.inner(
                delta_gradient, delta_z)
            if (not np.isfinite(preconditioned_curvature) or
                    preconditioned_curvature <= 1.0e-30):
                return 0.0, (
                    'nonpositive flexible Hager Zhang preconditioned curvature')
            first = self.inner(delta_gradient, new.z) / denominator
            second = (
                self.config.hager_zhang_theta *
                preconditioned_curvature *
                self.inner(old_direction, new.gradient) /
                (denominator * denominator))
            numerator = first - second
        else:
            delta_z = self.axpy(-1.0, old.z, new.z)
            numerator = self.inner(new.gradient, delta_z)
            if update == 'polak-ribiere':
                denominator = old_gz
            else:  # hestenes-stiefel
                delta_gradient = self.axpy(-1.0, old.gradient, new.gradient)
                denominator = self.inner(old_direction, delta_gradient)

        label = update.replace('-', ' ').title()
        finite = np.isfinite(numerator) and np.isfinite(denominator)
        valid_denominator = (abs(denominator) > 1.0e-30
                             if update in ('hestenes-stiefel', 'hager-zhang')
                             else denominator > 1.0e-30)
        if not finite or not valid_denominator:
            return 0.0, f'ill-conditioned {label} denominator'
        candidate = (numerator if update == 'hager-zhang'
                     else numerator / denominator)
        if not np.isfinite(candidate) or abs(candidate) > self.config.cg_beta_max:
            return 0.0, f'invalid {label} beta'
        return candidate, ''

    # ---- residual DIIS ---------------------------------------------------

    def _should_start_diis(self, state: _GCState) -> bool:
        threshold = self.config.diis_switch_residual_rms
        return threshold is not None and state.residual_rms <= threshold

    def _append_diis_item(self, history: list[_DIISItem],
                          state: _GCState) -> None:
        history.append(_DIISItem(
            self.copy_blocks(state.fock_orth),
            self.copy_blocks(state.residual)))
        if len(history) > self.config.diis_space:
            del history[0]

    def _diis_coefficients(
            self, history: list[_DIISItem]) -> tuple[np.ndarray, float,
                                                      float, str]:
        """Return regularized Pulay coefficients, pruning unsafe history."""
        action = ''
        while len(history) >= 2:
            size = len(history)
            gram = np.empty((size, size), dtype=float)
            for i, item_i in enumerate(history):
                for j in range(i + 1):
                    value = self.inner(item_i.residual, history[j].residual)
                    gram[i, j] = gram[j, i] = value
            scale = max(float(np.max(np.diag(gram))), 1.0e-30)
            regularized = gram / scale
            regularized += self.config.diis_regularization * np.eye(size)
            try:
                condition = float(np.linalg.cond(regularized))
                augmented = np.zeros((size + 1, size + 1), dtype=float)
                augmented[:size, :size] = regularized
                augmented[:size, size] = 1.0
                augmented[size, :size] = 1.0
                rhs = np.zeros(size + 1, dtype=float)
                rhs[size] = 1.0
                coefficients = np.linalg.solve(augmented, rhs)[:size]
                coefficient_l1 = float(np.sum(np.abs(coefficients)))
                valid = (
                    np.isfinite(condition)
                    and condition <= self.config.diis_max_condition
                    and np.all(np.isfinite(coefficients))
                    and coefficient_l1 <= self.config.diis_max_coefficient_l1)
            except (FloatingPointError, np.linalg.LinAlgError):
                condition = np.inf
                coefficient_l1 = np.inf
                valid = False
            if valid:
                return coefficients, condition, coefficient_l1, action
            if len(history) > 2:
                del history[0]
                action = 'dropped oldest ill-conditioned DIIS vector'
                continue
            latest = history[-1]
            history[:] = [latest]
            return (np.ones(1), condition, 1.0,
                    'reset ill-conditioned DIIS history')
        return np.ones(1), np.nan, 1.0, action or 'fixed-point seed'

    def _diis_target(self, history: Sequence[_DIISItem],
                     coefficients: Sequence[float]) -> list:
        target = self.scale_blocks(float(coefficients[0]), history[0].fock)
        for coefficient, item in zip(coefficients[1:], history[1:]):
            target = self.axpy(float(coefficient), item.fock, target)
        target = self.hermitize_blocks(target)
        target, _ = self.project_time_reversal(target)
        return self.hermitize_blocks(target)

    def _diis_predicted_residual_rms(
            self, state: _GCState, damping: float) -> float:
        """Ideal linear residual after taking a fraction of a DIIS solve."""
        return max(0.0, 1.0 - damping) * state.residual_rms

    def _next_diis_damping(
            self, state: _GCState, trial: _GCState,
            predicted_residual_rms: float, accepted_damping: float,
            starting_damping: float) -> tuple[float, float]:
        """Update the DIIS trust radius from predicted versus actual progress."""
        predicted_reduction = state.residual_rms - predicted_residual_rms
        actual_reduction = state.residual_rms - trial.residual_rms
        scale = max(state.residual_rms, np.finfo(float).tiny)
        relative_reduction = actual_reduction / scale
        if predicted_reduction > np.finfo(float).eps * scale:
            ratio = actual_reduction / predicted_reduction
        else:
            ratio = np.nan

        next_damping = accepted_damping
        if (np.isfinite(ratio) and
                ratio < self.config.diis_trust_shrink_ratio):
            next_damping *= self.config.diis_backtrack_factor
        elif (np.isfinite(ratio) and
              ratio > self.config.diis_trust_expand_ratio and
              relative_reduction >=
              self.config.diis_trust_expand_min_relative_reduction and
              accepted_damping >= starting_damping * (1.0 - 1.0e-12)):
            next_damping *= self.config.diis_trust_expansion
        next_damping = min(1.0, max(
            self.config.line_search_alpha_min, next_damping))
        return next_damping, float(ratio)

    def _diis_trial_acceptable(
            self, state: _GCState, trial: _GCState) -> tuple[bool, str]:
        residual_limit = state.residual_rms * (
            1.0 - self.config.diis_min_residual_reduction)
        if not trial.residual_rms < residual_limit:
            return False, 'residual did not satisfy the trust envelope'
        objective_increase = trial.objective - state.objective
        if objective_increase > self.config.diis_max_objective_increase:
            return False, 'objective increase exceeded DIIS noise allowance'
        if (not self.fixed_electron_number and
                abs(trial.electron_number - state.electron_number) >
                self.config.diis_max_delta_nelec):
            return False, 'electron-number change exceeded DIIS safeguard'
        return True, ''

    def _try_diis_target(self, state: _GCState, target: Sequence,
                         starting_damping: float = 1.0,
                         max_backtracks: Optional[int] = None,
                         ) -> tuple[Optional[_GCState], float, str]:
        direction = self.axpy(-1.0, state.h_orth, target)
        if not self.all_finite(direction) or self.norm(direction) == 0.0:
            return None, 0.0, 'zero or nonfinite DIIS direction'
        damping = min(1.0, max(
            self.config.line_search_alpha_min, starting_damping))
        if max_backtracks is None:
            max_backtracks = self.config.diis_max_backtracks
        last_reason = 'no DIIS trial evaluated'
        for _ in range(max_backtracks + 1):
            trial = self._trial(state, direction, damping)
            if trial is not None:
                acceptable, last_reason = self._diis_trial_acceptable(
                    state, trial)
                self.log.debug(
                    'DIIS trust trial: damping = %.6g, residual %.6g -> '
                    '%.6g, delta objective = %.3g, delta N = %.3g: %s',
                    damping, state.residual_rms, trial.residual_rms,
                    trial.objective - state.objective,
                    trial.electron_number - state.electron_number,
                    'accepted' if acceptable else last_reason)
                if acceptable:
                    return trial, damping, ''
            else:
                last_reason = 'DIIS trial evaluation failed'
            damping *= self.config.diis_backtrack_factor
        return None, 0.0, last_reason

    def _diis_step(
            self, state: _GCState,
            history: list[_DIISItem],
            starting_damping: float = 1.0) -> _DIISStepResult:
        start_nfev = self.nfev
        latest = history[-1]
        coefficients, condition, coefficient_l1, action = (
            self._diis_coefficients(history))
        target = self._diis_target(history, coefficients)
        model_backtracks = (
            min(2, self.config.diis_max_backtracks)
            if len(history) > 1 else self.config.diis_max_backtracks)
        trial, damping, rejection = self._try_diis_target(
            state, target, starting_damping, model_backtracks)
        if trial is None and len(history) > 1:
            trial, damping, rejection = self._try_diis_target(
                state, latest.fock, starting_damping,
                self.config.diis_max_backtracks)
            fallback_action = (
                'latest-Fock fallback after rejected Pulay model')
            action = ((action + '; ') if action else '') + fallback_action
        nfev = self.nfev - start_nfev
        if trial is None:
            message = 'residual-DIIS failed: ' + rejection
            return _DIISStepResult(
                _LineSearchResult(False, None, nfev=nfev, message=message),
                condition, coefficient_l1, action, np.nan)
        predicted_residual_rms = self._diis_predicted_residual_rms(
            state, damping)
        message = 'residual-DIIS accepted'
        if damping < 1.0:
            message += f' with damping {damping:.6g}'
        return _DIISStepResult(
            _LineSearchResult(True, trial, damping, nfev,
                              False, False, message),
            condition, coefficient_l1, action,
            predicted_residual_rms)

    def _alpha_cap(self, direction: Sequence) -> float:
        block_rms = self.max_block_rms(direction)
        if block_rms == 0.0:
            return 0.0
        return min(self.config.line_search_alpha_cap,
                   self.config.line_search_max_h_rms_step / block_rms)

    def _electron_number_at_mu(self, h_orth: Sequence, mu: float) -> float:
        """Evaluate N(H, mu) without a density or Fock construction."""
        eigenvalues = [cp.linalg.eigvalsh(hk) for hk in h_orth]
        return self._electron_number_from_eigenvalues(eigenvalues, mu)

    def _electron_number_from_eigenvalues(
            self, eigenvalues: Sequence, mu: float) -> float:
        return 2.0 * sum(
            float((self.weights[k] * cp.sum(fermi_occupations(
                self.beta * (value - mu)))).item())
            for k, value in enumerate(eigenvalues))

    def _cheap_fixed_mu_electron_number(self, h_orth: Sequence) -> float:
        """Evaluate N(H) without constructing a density or building a Fock matrix."""
        if self.fixed_electron_number:
            return self.target_electron_number
        return self._electron_number_at_mu(h_orth, self.mu)

    def _active_nelec_limit(self, state: _GCState) -> float:
        threshold = self.config.line_search_nelec_guard_residual_rms
        maximum = self.config.line_search_max_delta_nelec
        if threshold is not None and state.residual_rms <= threshold:
            maximum = min(
                maximum,
                self.config.line_search_nelec_guard_max_delta_nelec)
        return maximum

    def _charge_feasible_alpha_cap(
            self, state: _GCState, direction: Sequence, requested: float,
            maximum: Optional[float] = None) -> tuple[float, bool]:
        """Return a cheaply charge-feasible upper step on a fixed line.

        Fermi occupations require only Hermitian diagonalizations, so rejected
        charge proposals must not consume the expensive Fock-evaluation
        budget.  The feasible set need not be globally monotone in ``alpha``;
        this routine only locates the first local boundary between a feasible
        smaller step and an infeasible requested step.  Every later trial is
        still checked independently by :meth:`_trial`.
        """
        requested = float(requested)
        if (self.fixed_electron_number or
                not self.config.line_search_nelec_feasible_alpha or
                requested <= 0.0):
            return requested, False
        if maximum is None:
            maximum = self._active_nelec_limit(state)
        maximum = float(maximum)
        cache: dict[float, float] = {0.0: 0.0}

        def delta_nelec(alpha: float) -> float:
            key = float(alpha)
            if key not in cache:
                candidate = self._sanitize_h(
                    self.axpy(key, direction, state.h_orth))
                value = (self._cheap_fixed_mu_electron_number(candidate) -
                         state.electron_number)
                cache[key] = value
                self.ncheap_nelec_evaluations += 1
            return cache[key]

        if abs(delta_nelec(requested)) <= maximum + 1.0e-10:
            return requested, False

        high = requested
        low = 0.0
        alpha = requested
        for _ in range(self.config.line_search_max_trials):
            alpha *= self.config.armijo_backtrack_factor
            self.ncheap_nelec_alpha_reductions += 1
            if alpha < self.config.line_search_alpha_min:
                return 0.0, True
            if abs(delta_nelec(alpha)) <= maximum + 1.0e-10:
                low = alpha
                break
            high = alpha
        else:
            return 0.0, True

        for _ in range(self.config.line_search_nelec_alpha_bisections):
            if high - low <= self.config.line_search_alpha_min:
                break
            midpoint = 0.5 * (low + high)
            self.ncheap_nelec_alpha_reductions += 1
            error = abs(delta_nelec(midpoint)) - maximum
            if error <= 0.0:
                low = midpoint
                if abs(error) <= 1.0e-10:
                    break
            else:
                high = midpoint
        return low, True

    def _trial(self, state: _GCState, direction: Sequence,
               alpha: float) -> Optional[_GCState]:
        self._last_trial_rejected_by_nelec = False
        try:
            candidate = self._sanitize_h(
                self.axpy(alpha, direction, state.h_orth))
            if self.fixed_electron_number:
                return self.evaluate(candidate)

            trial_nelec = self._cheap_fixed_mu_electron_number(candidate)
            self.ncheap_nelec_evaluations += 1
            maximum = self._active_nelec_limit(state)
            if (abs(trial_nelec - state.electron_number) >
                    maximum + 1.0e-10):
                self._last_trial_rejected_by_nelec = True
                self.ncheap_nelec_reject += 1
                self.log.debug(
                    'Rejected trial before Fock build: alpha = %.6g, '
                    'residual RMS = %.6g, N = %.12g -> %.12g',
                    alpha, state.residual_rms, state.electron_number,
                    trial_nelec)
                return None
            return self.evaluate(candidate)
        except (ArithmeticError, FloatingPointError, ValueError, RuntimeError,
                cp.linalg.LinAlgError):
            return None

    @staticmethod
    def _cubic_minimizer(a: float, fa: float, dfa: float,
                         b: float, fb: float, dfb: float) -> Optional[float]:
        if not all(np.isfinite(x) for x in (a, fa, dfa, b, fb, dfb)) or a == b:
            return None
        try:
            d1 = dfa + dfb - 3.0 * (fa - fb) / (a - b)
            discriminant = d1 * d1 - dfa * dfb
            if discriminant < 0.0:
                return None
            d2 = np.sqrt(discriminant)
            if b < a:
                d2 = -d2
            denominator = dfb - dfa + 2.0 * d2
            if denominator == 0.0:
                return None
            return b - (b - a) * (dfb + d2 - d1) / denominator
        except (FloatingPointError, ZeroDivisionError):
            return None

    def _residual_filter_metrics(
            self, old: _GCState,
            trial: Optional[_GCState]) -> tuple[bool, bool, bool, float]:
        threshold = self.config.nlcg_residual_filter_rms
        active = (
            threshold is not None and old.residual_rms <= threshold)
        if not active or trial is None:
            return active, not active, False, np.nan
        ratio = (trial.residual_rms / old.residual_rms
                 if old.residual_rms > 0.0 else np.inf)
        tolerance = 32.0 * np.finfo(float).eps * max(
            1.0, old.residual_rms, trial.residual_rms)
        bounded_limit = min(
            threshold,
            (1.0 +
             self.config.nlcg_residual_filter_max_relative_increase) *
            old.residual_rms)
        strong_limit = (
            (1.0 -
             self.config.nlcg_residual_filter_min_relative_reduction) *
            old.residual_rms)
        return (True,
                trial.residual_rms <= bounded_limit + tolerance,
                trial.residual_rms <= strong_limit + tolerance,
                ratio)

    def _nlcg_residual_alpha_init(
            self, state: _GCState) -> Optional[float]:
        """Return the opt-in HZ warm start in the residual-polish regime."""
        threshold = self.config.nlcg_residual_filter_rms
        if (not self.config.nlcg_residual_filter_warm_start or
                threshold is None or state.residual_rms > threshold):
            return None
        alpha = self._nlcg_residual_previous_alpha
        if alpha is None or not np.isfinite(alpha) or alpha <= 0.0:
            alpha = self.config.nlcg_residual_filter_initial_alpha
        return float(np.clip(
            alpha,
            self.config.nlcg_residual_filter_alpha_min,
            self.config.nlcg_residual_filter_alpha_max))

    def _finish_line_search(
            self, result: _LineSearchResult, start_nfev: int,
            cheap_start: int, reduction_start: int,
            method: str) -> _LineSearchResult:
        return replace(
            result, nfev=self.nfev - start_nfev,
            line_search_method=method,
            cheap_nelec_evaluations=(
                self.ncheap_nelec_evaluations - cheap_start),
            cheap_nelec_alpha_reductions=(
                self.ncheap_nelec_alpha_reductions - reduction_start))

    @staticmethod
    def _combine_line_search_work(
            primary: _LineSearchResult,
            fallback: _LineSearchResult) -> _LineSearchResult:
        """Keep accepted fallback metadata while reporting all search work."""
        detail = fallback.message
        if primary.message:
            detail += (f'; after {primary.line_search_method} failure: '
                       f'{primary.message}')
        return replace(
            fallback,
            nfev=primary.nfev + fallback.nfev,
            cheap_nelec_evaluations=(
                primary.cheap_nelec_evaluations +
                fallback.cheap_nelec_evaluations),
            cheap_nelec_alpha_reductions=(
                primary.cheap_nelec_alpha_reductions +
                fallback.cheap_nelec_alpha_reductions),
            residual_filter_rejections=(
                primary.residual_filter_rejections +
                fallback.residual_filter_rejections),
            message=detail)

    def _zoom(
            self, state0: _GCState, direction: Sequence, phi0: float,
            dphi0: float, lo_a: float, lo_state: _GCState, hi_a: float,
            hi_state: Optional[_GCState],
            best: Optional[tuple[float, _GCState]], nfev: int,
            c2: Optional[float] = None, *,
            start_nfev: Optional[int] = None, trial_count: int = 0,
            cheap_start: Optional[int] = None,
            reduction_start: Optional[int] = None
            ) -> _LineSearchResult:
        c1 = self.config.line_search_c1
        c2 = self.config.line_search_c2 if c2 is None else c2
        start_nfev = (self.nfev - nfev if start_nfev is None
                      else start_nfev)
        cheap_start = (self.ncheap_nelec_evaluations if cheap_start is None
                       else cheap_start)
        reduction_start = (
            self.ncheap_nelec_alpha_reductions
            if reduction_start is None else reduction_start)

        def finish(value: _LineSearchResult) -> _LineSearchResult:
            return self._finish_line_search(
                value, start_nfev, cheap_start, reduction_start,
                'strong-wolfe')

        lo_phi = lo_state.objective
        lo_dphi = self.inner(lo_state.gradient, direction)
        hi_phi = np.inf if hi_state is None else hi_state.objective
        hi_dphi = (np.nan if hi_state is None else
                   self.inner(hi_state.gradient, direction))
        zoom_start_nfev = self.nfev
        while (self.nfev - zoom_start_nfev <
               self.config.line_search_zoom_evals and
               trial_count < self.config.line_search_max_trials and
               self.nfev - start_nfev <
               self.config.line_search_max_evals):
            lower, upper = min(lo_a, hi_a), max(lo_a, hi_a)
            alpha = self._cubic_minimizer(
                lo_a, lo_phi, lo_dphi, hi_a, hi_phi, hi_dphi)
            margin = 0.1 * (upper - lower)
            if alpha is None or not (lower + margin < alpha < upper - margin):
                alpha = 0.5 * (lo_a + hi_a)
            if abs(hi_a - lo_a) < self.config.line_search_alpha_min:
                break
            trial = self._trial(state0, direction, alpha)
            trial_count += 1
            nelec_boundary = (
                trial is None and self._last_trial_rejected_by_nelec)
            phi = np.inf if trial is None else trial.objective
            if (trial is None or phi > phi0 + c1 * alpha * dphi0 or
                    phi >= lo_phi):
                hi_a, hi_state, hi_phi = alpha, trial, phi
                hi_dphi = (np.nan if trial is None else
                            self.inner(trial.gradient, direction))
                if nelec_boundary and best is not None:
                    return finish(_LineSearchResult(
                        True, best[1], best[0], force_restart=True,
                        message=('accepted Armijo point at electron-number '
                                 'trust boundary'), trust_boundary=True))
                continue
            dphi = self.inner(trial.gradient, direction)
            if phi <= phi0 + c1 * alpha * dphi0:
                if best is None or phi < best[1].objective:
                    best = (alpha, trial)
            if abs(dphi) <= c2 * abs(dphi0):
                return finish(_LineSearchResult(
                    True, trial, alpha, strong_wolfe=True,
                    curvature_qualified=True, message='strong Wolfe'))
            if dphi * (hi_a - lo_a) >= 0.0:
                hi_a, hi_state, hi_phi, hi_dphi = (
                    lo_a, lo_state, lo_phi, lo_dphi)
            lo_a, lo_state, lo_phi, lo_dphi = alpha, trial, phi, dphi
        if best is not None:
            return finish(_LineSearchResult(
                True, best[1], best[0], force_restart=True,
                message='accepted best Armijo point after zoom'))
        return finish(_LineSearchResult(
            False, None, message='zoom found no Armijo point'))

    def _hager_zhang_line_search(
            self, state: _GCState, direction: Sequence,
            alpha_init: Optional[float] = None,
            max_evals_override: Optional[int] = None, *,
            residual_filter_enabled: bool = True
            ) -> _LineSearchResult:
        """Cached Hager--Zhang weak/approximate-Wolfe line search."""
        start_nfev = self.nfev
        cheap_start = self.ncheap_nelec_evaluations
        reduction_start = self.ncheap_nelec_alpha_reductions
        filter_rejection_start = self.nresidual_filter_rejections
        filter_active = (
            residual_filter_enabled and
            self.config.nlcg_residual_filter_rms is not None and
            state.residual_rms <= self.config.nlcg_residual_filter_rms)
        maximum_evals = self.config.hager_zhang_max_evals
        if max_evals_override is not None:
            if (not isinstance(max_evals_override, int) or
                    isinstance(max_evals_override, bool) or
                    max_evals_override < 0):
                raise ValueError(
                    'Hager-Zhang max_evals_override must be a nonnegative '
                    'integer or None')
            maximum_evals = min(maximum_evals, max_evals_override)
        if (filter_active and
                self.config.nlcg_residual_filter_max_evals is not None):
            maximum_evals = min(
                maximum_evals,
                self.config.nlcg_residual_filter_max_evals)

        def finish(value: _LineSearchResult) -> _LineSearchResult:
            ratio = value.residual_filter_ratio
            if (value.state is not None and filter_active and
                    not np.isfinite(ratio)):
                ratio = (value.state.residual_rms / state.residual_rms
                         if state.residual_rms > 0.0 else np.inf)
            value = replace(
                value, residual_filter_active=filter_active,
                residual_filter_ratio=ratio,
                residual_filter_rejections=(
                    self.nresidual_filter_rejections -
                    filter_rejection_start))
            return self._finish_line_search(
                value, start_nfev, cheap_start, reduction_start,
                'hager-zhang')

        phi0 = state.objective
        dphi0 = self.inner(state.gradient, direction)
        if dphi0 >= 0.0:
            return finish(_LineSearchResult(
                False, None,
                message='Hager-Zhang search called with non-descent direction'))
        alpha_max = self._alpha_cap(direction)
        alpha_max, _ = self._charge_feasible_alpha_cap(
            state, direction, alpha_max)
        if alpha_max < self.config.line_search_alpha_min:
            return finish(_LineSearchResult(
                False, None, message='Hager-Zhang step cap below minimum'))
        alpha_start = (self.config.line_search_alpha_init
                       if alpha_init is None else alpha_init)
        if not np.isfinite(alpha_start) or alpha_start <= 0.0:
            raise ValueError(
                'line-search initial alpha must be finite and positive')
        delta = self.config.hager_zhang_delta
        sigma = self.config.hager_zhang_sigma
        noise = self.config.hager_zhang_objective_noise
        threshold = phi0 + noise
        cache: dict[float, _HZPoint] = {
            0.0: _HZPoint(0.0, state, phi0, dphi0, False)}
        trial_count = 0
        best_armijo: Optional[_HZPoint] = None
        residual_vetoed: set[float] = set()

        def evaluate(alpha: float) -> Optional[_HZPoint]:
            nonlocal trial_count, best_armijo
            alpha = min(alpha_max, max(0.0, float(alpha)))
            alpha = 0.0 if alpha == 0.0 else alpha
            if alpha in cache:
                return cache[alpha]
            if (trial_count >= self.config.line_search_max_trials or
                    self.nfev - start_nfev >= maximum_evals):
                return None
            trial = self._trial(state, direction, alpha)
            trial_count += 1
            phi = np.inf if trial is None else trial.objective
            dphi = (np.nan if trial is None else
                     self.inner(trial.gradient, direction))
            failed = (trial is None or not np.isfinite(phi) or
                      not np.isfinite(dphi))
            point = _HZPoint(alpha, trial, phi, dphi, failed)
            cache[alpha] = point
            if filter_active:
                _, bounded, _, _ = self._residual_filter_metrics(
                    state, trial)
            else:
                bounded = True
            if (not failed and
                    phi <= phi0 + delta * alpha * dphi0 and
                    bounded and
                    (best_armijo is None or phi < best_armijo.phi)):
                best_armijo = point
            return point

        def accepted(point: Optional[_HZPoint]
                     ) -> Optional[_LineSearchResult]:
            if point is None or point.state is None:
                return None
            if point.failed:
                return None
            ordinary = (
                point.phi <= phi0 + delta * point.alpha * dphi0 and
                point.dphi >= sigma * dphi0)
            approximate = (
                noise > 0.0 and point.phi <= threshold and
                point.dphi >= sigma * dphi0 and
                point.dphi <= (2.0 * delta - 1.0) * dphi0)
            if filter_active:
                active, bounded, strong, residual_ratio = (
                    self._residual_filter_metrics(state, point.state))
            else:
                active, bounded, strong, residual_ratio = (
                    False, True, False, np.nan)
            if active and (ordinary or approximate) and not bounded:
                if point.alpha not in residual_vetoed:
                    residual_vetoed.add(point.alpha)
                    self.nresidual_filter_rejections += 1
                ordinary = approximate = False
            if ordinary:
                return finish(_LineSearchResult(
                    True, point.state, point.alpha,
                    weak_wolfe=True, curvature_qualified=True,
                    message='Hager-Zhang ordinary weak Wolfe',
                    residual_filter_ratio=residual_ratio))
            if approximate:
                return finish(_LineSearchResult(
                    True, point.state, point.alpha,
                    weak_wolfe=True, approximate_wolfe=True,
                    curvature_qualified=True,
                    objective_allowance=noise,
                    message='Hager-Zhang approximate Wolfe',
                    residual_filter_ratio=residual_ratio))
            if (active and strong and
                    point.phi <= phi0 +
                    self.config.nlcg_residual_filter_objective_noise):
                self.nresidual_filter_acceptances += 1
                return finish(_LineSearchResult(
                    True, point.state, point.alpha,
                    force_restart=True,
                    objective_allowance=(
                        self.config.nlcg_residual_filter_objective_noise),
                    message='accepted residual-qualified Hager-Zhang point',
                    residual_filter_active=True,
                    residual_filter_qualified=True,
                    residual_filter_ratio=residual_ratio))
            return None

        def refine_interval(
                low: _HZPoint, right: float
                ) -> tuple[_HZPoint, Optional[_HZPoint],
                           Optional[_LineSearchResult]]:
            """Implement the HZ update U3 until the bracket is valid."""
            high: Optional[_HZPoint] = None
            while (right - low.alpha >
                   self.config.line_search_alpha_min):
                middle = 0.5 * (low.alpha + right)
                point = evaluate(middle)
                result = accepted(point)
                if result is not None:
                    return low, high, result
                if point is None:
                    break
                if point.failed:
                    right = middle
                elif point.dphi >= 0.0:
                    high = point
                    break
                elif point.phi <= threshold:
                    low = point
                else:
                    right = middle
            return low, high, None

        def update_bracket(
                low: _HZPoint, high: _HZPoint, alpha: float
                ) -> tuple[_HZPoint, Optional[_HZPoint],
                           Optional[_LineSearchResult], Optional[_HZPoint]]:
            point = evaluate(alpha)
            result = accepted(point)
            if result is not None:
                return low, high, result, point
            if point is None:
                return low, high, None, None
            if point.failed:
                low, refined, result = refine_interval(low, point.alpha)
                return low, refined, result, point
            if point.dphi >= 0.0:
                return low, point, None, point
            if point.phi <= threshold:
                return point, high, None, point
            low, refined, result = refine_interval(low, point.alpha)
            return low, refined, result, point

        def secant(left: _HZPoint, right: _HZPoint) -> float:
            denominator = right.dphi - left.dphi
            if (not np.isfinite(denominator) or
                    abs(denominator) <= 1.0e-30):
                return 0.5 * (left.alpha + right.alpha)
            alpha = ((left.alpha * right.dphi -
                      right.alpha * left.dphi) / denominator)
            if (not np.isfinite(alpha) or
                    not left.alpha < alpha < right.alpha):
                return 0.5 * (left.alpha + right.alpha)
            return alpha

        low = cache[0.0]
        high: Optional[_HZPoint] = None
        alpha = min(alpha_start, alpha_max)
        while True:
            point = evaluate(alpha)
            result = accepted(point)
            if result is not None:
                return result
            if point is None:
                break
            if point.failed:
                low, high, result = refine_interval(low, point.alpha)
                if result is not None:
                    return result
                break
            if point.dphi >= 0.0:
                high = point
                break
            if point.phi > threshold:
                low, high, result = refine_interval(low, point.alpha)
                if result is not None:
                    return result
                break
            low = point
            grown = min(
                self.config.hager_zhang_expansion * alpha, alpha_max)
            if grown <= alpha:
                break
            alpha = grown

        if high is not None:
            while (high.alpha - low.alpha >
                   self.config.line_search_alpha_min):
                old_low, old_high = low, high
                old_width = high.alpha - low.alpha
                alpha = secant(low, high)
                low, high, result, point = update_bracket(
                    low, high, alpha)
                if result is not None:
                    return result
                if high is None or point is None:
                    break
                second = None
                if point.alpha == low.alpha and point.alpha != old_low.alpha:
                    second = secant(old_low, point)
                elif (point.alpha == high.alpha and
                      point.alpha != old_high.alpha):
                    second = secant(point, old_high)
                if (second is not None and
                        low.alpha < second < high.alpha):
                    low, high, result, _ = update_bracket(
                        low, high, second)
                    if result is not None:
                        return result
                    if high is None:
                        break
                if (high.alpha - low.alpha >
                        self.config.hager_zhang_shrinkage * old_width):
                    midpoint = 0.5 * (low.alpha + high.alpha)
                    low, high, result, _ = update_bracket(
                        low, high, midpoint)
                    if result is not None:
                        return result
                    if high is None:
                        break
                if (low.alpha == old_low.alpha and
                        high.alpha == old_high.alpha):
                    break
                if (trial_count >= self.config.line_search_max_trials or
                        self.nfev - start_nfev >= maximum_evals):
                    break

        if best_armijo is not None:
            return finish(_LineSearchResult(
                True, best_armijo.state, best_armijo.alpha,
                force_restart=True,
                message=('accepted best Armijo point after Hager-Zhang '
                         'search')))
        return finish(_LineSearchResult(
            False, None,
            message='Hager-Zhang search found no acceptable point'))

    def _line_search(
            self, state: _GCState, direction: Sequence,
            c2: Optional[float] = None,
            alpha_init: Optional[float] = None, *,
            method_override: Optional[str] = None,
            max_evals_override: Optional[int] = None,
            residual_filter_enabled: bool = True
            ) -> _LineSearchResult:
        method = (self.config.line_search_method
                  if method_override is None else
                  self._canonical_line_search_method(method_override))
        if method == 'hager-zhang':
            return self._hager_zhang_line_search(
                state, direction, alpha_init=alpha_init,
                max_evals_override=max_evals_override,
                residual_filter_enabled=residual_filter_enabled)
        if max_evals_override is not None:
            raise ValueError(
                'max_evals_override is supported only by Hager-Zhang')

        start_nfev = self.nfev
        cheap_start = self.ncheap_nelec_evaluations
        reduction_start = self.ncheap_nelec_alpha_reductions

        def finish(value: _LineSearchResult) -> _LineSearchResult:
            return self._finish_line_search(
                value, start_nfev, cheap_start, reduction_start,
                'strong-wolfe')

        dphi0 = self.inner(state.gradient, direction)
        if dphi0 >= 0.0:
            return finish(_LineSearchResult(
                False, None,
                message='line search called with non-descent direction'))
        alpha_max = self._alpha_cap(direction)
        alpha_max, _ = self._charge_feasible_alpha_cap(
            state, direction, alpha_max)
        if alpha_max < self.config.line_search_alpha_min:
            return finish(_LineSearchResult(
                False, None, message='step cap below minimum'))
        alpha_start = (self.config.line_search_alpha_init
                       if alpha_init is None else alpha_init)
        if not np.isfinite(alpha_start) or alpha_start <= 0.0:
            raise ValueError(
                'line-search initial alpha must be finite and positive')
        alpha = min(alpha_start, alpha_max)
        phi0 = state.objective
        c1 = self.config.line_search_c1
        c2 = self.config.line_search_c2 if c2 is None else c2
        previous_alpha, previous_state = 0.0, state
        best: Optional[tuple[float, _GCState]] = None
        trial_count = 0
        while (trial_count < self.config.line_search_max_trials and
               self.nfev - start_nfev <
               self.config.line_search_max_evals):
            trial = self._trial(state, direction, alpha)
            trial_count += 1
            phi = np.inf if trial is None else trial.objective
            armijo = (
                trial is not None and
                phi <= phi0 + c1 * alpha * dphi0)
            if armijo and (best is None or phi < best[1].objective):
                best = (alpha, trial)
            if (trial is None or not armijo or
                    (previous_alpha > 0.0 and
                     phi >= previous_state.objective)):
                return self._zoom(
                    state, direction, phi0, dphi0,
                    previous_alpha, previous_state, alpha, trial,
                    best, self.nfev - start_nfev, c2,
                    start_nfev=start_nfev, trial_count=trial_count,
                    cheap_start=cheap_start,
                    reduction_start=reduction_start)
            dphi = self.inner(trial.gradient, direction)
            if abs(dphi) <= c2 * abs(dphi0):
                return finish(_LineSearchResult(
                    True, trial, alpha, strong_wolfe=True,
                    curvature_qualified=True, message='strong Wolfe'))
            if dphi >= 0.0:
                return self._zoom(
                    state, direction, phi0, dphi0,
                    alpha, trial, previous_alpha, previous_state,
                    best, self.nfev - start_nfev, c2,
                    start_nfev=start_nfev, trial_count=trial_count,
                    cheap_start=cheap_start,
                    reduction_start=reduction_start)
            previous_alpha, previous_state = alpha, trial
            grown = min(self.config.line_search_growth * alpha, alpha_max)
            if grown <= alpha:
                break
            alpha = grown
        if best is not None:
            return finish(_LineSearchResult(
                True, best[1], best[0], force_restart=True,
                message='accepted best Armijo point'))
        return finish(_LineSearchResult(
            False, None, message='no Armijo point'))

    def _armijo_fallback(
            self, state: _GCState, direction: Sequence, *,
            alpha_init: Optional[float] = None,
            max_evals_override: Optional[int] = None,
            residual_filter_enabled: bool = True
            ) -> _LineSearchResult:
        start_nfev = self.nfev
        cheap_start = self.ncheap_nelec_evaluations
        reduction_start = self.ncheap_nelec_alpha_reductions
        filter_rejection_start = self.nresidual_filter_rejections
        filter_active = (
            residual_filter_enabled and
            self.config.nlcg_residual_filter_rms is not None and
            state.residual_rms <= self.config.nlcg_residual_filter_rms)
        maximum_evals = self.config.line_search_max_evals
        if max_evals_override is not None:
            if (not isinstance(max_evals_override, int) or
                    isinstance(max_evals_override, bool) or
                    max_evals_override < 0):
                raise ValueError(
                    'fallback max_evals_override must be a nonnegative '
                    'integer or None')
            maximum_evals = min(maximum_evals, max_evals_override)

        def finish(value: _LineSearchResult) -> _LineSearchResult:
            ratio = value.residual_filter_ratio
            if (value.state is not None and filter_active and
                    not np.isfinite(ratio)):
                ratio = (value.state.residual_rms / state.residual_rms
                         if state.residual_rms > 0.0 else np.inf)
            value = replace(
                value, residual_filter_active=filter_active,
                residual_filter_ratio=ratio,
                residual_filter_rejections=(
                    self.nresidual_filter_rejections -
                    filter_rejection_start))
            return self._finish_line_search(
                value, start_nfev, cheap_start, reduction_start, 'armijo')

        alpha_max = self._alpha_cap(direction)
        alpha_max, _ = self._charge_feasible_alpha_cap(
            state, direction, alpha_max)
        if alpha_max < self.config.line_search_alpha_min:
            return finish(_LineSearchResult(
                False, None, message='fallback step cap below minimum'))
        dphi0 = self.inner(state.gradient, direction)
        if dphi0 >= 0.0:
            return finish(_LineSearchResult(
                False, None, message='fallback residual is not downhill'))
        if alpha_init is None:
            alpha = min(1.0, alpha_max)
        else:
            if not np.isfinite(alpha_init) or alpha_init <= 0.0:
                raise ValueError(
                    'fallback initial alpha must be finite and positive')
            alpha = min(float(alpha_init), alpha_max)
        trial_count = 0
        while (trial_count < self.config.line_search_max_trials and
               self.nfev - start_nfev < maximum_evals):
            trial = self._trial(state, direction, alpha)
            trial_count += 1
            armijo = (
                trial is not None and
                trial.objective <= state.objective +
                self.config.line_search_c1 * alpha * dphi0)
            if residual_filter_enabled:
                active, bounded, strong, residual_ratio = (
                    self._residual_filter_metrics(state, trial))
            else:
                active, bounded, strong, residual_ratio = (
                    False, True, False, np.nan)
            if active and armijo and not bounded:
                self.nresidual_filter_rejections += 1
            elif armijo:
                return finish(_LineSearchResult(
                    True, trial, alpha, force_restart=True,
                    message='monotone Armijo fallback',
                    residual_filter_ratio=residual_ratio))
            elif (active and strong and trial is not None and
                  trial.objective <= state.objective +
                  self.config.nlcg_residual_filter_objective_noise):
                self.nresidual_filter_acceptances += 1
                return finish(_LineSearchResult(
                    True, trial, alpha, force_restart=True,
                    objective_allowance=(
                        self.config.nlcg_residual_filter_objective_noise),
                    message='residual-qualified Armijo fallback',
                    residual_filter_active=True,
                    residual_filter_qualified=True,
                    residual_filter_ratio=residual_ratio))
            alpha *= self.config.armijo_backtrack_factor
            if alpha < self.config.line_search_alpha_min:
                break
        return finish(_LineSearchResult(
            False, None, message='fallback found no Armijo point'))

    def _metrics(self, state: _GCState, previous: Optional[_GCState]) -> tuple[float, float, float, float]:
        if previous is None:
            return np.inf, np.inf, np.inf, np.inf
        delta_objective = state.objective - previous.objective
        delta_nelec = abs(state.electron_number - previous.electron_number)
        density_change = self.rms(self.axpy(-1.0, previous.p_orth, state.p_orth))
        return delta_objective, delta_nelec, density_change, abs(delta_objective)

    def _meets_convergence(self, state: _GCState, previous: Optional[_GCState]) -> bool:
        if previous is None:
            return state.grad_rms < self.config.conv_tol_grad_rms and state.residual_rms < self.config.conv_tol_residual_rms
        delta_objective, delta_nelec, density_change, abs_delta_objective = self._metrics(state, previous)
        return (abs_delta_objective < self.config.conv_tol_omega and
                state.grad_rms < self.config.conv_tol_grad_rms and
                state.residual_rms < self.config.conv_tol_residual_rms and
                density_change < self.config.conv_tol_density_rms and
                delta_nelec < self.config.conv_tol_nelec)

    def _verify_accepted_step(self, state: _GCState, accepted: _GCState,
                              direction: Sequence,
                              line_search: _LineSearchResult,
                              dphi0: float) -> None:
        expected = self._sanitize_h(
            self.axpy(line_search.alpha, direction, state.h_orth))
        armijo_slope = line_search.alpha * dphi0
        mismatch = self.max_block_rms(self.axpy(-1.0, expected, accepted.h_orth))
        if mismatch > 1.0e-8:
            raise RuntimeError(f'accepted state is not the evaluated step (mismatch {mismatch:g})')
        if line_search.residual_filter_qualified:
            active, _, strong, _ = self._residual_filter_metrics(
                state, accepted)
            if not active or not strong:
                raise RuntimeError(
                    'accepted residual-qualified point does not meet its '
                    'residual reduction')
            if (accepted.objective > state.objective +
                    line_search.objective_allowance + 1.0e-12):
                raise RuntimeError(
                    'accepted residual-qualified point exceeds its absolute '
                    'objective allowance')
        elif line_search.approximate_wolfe:
            if (accepted.objective > state.objective +
                    line_search.objective_allowance + 1.0e-12):
                raise RuntimeError(
                    'accepted approximate-Wolfe point exceeds its absolute '
                    'objective allowance')
            dphi = self.inner(accepted.gradient, direction)
            lower = self.config.hager_zhang_sigma * dphi0
            upper = ((2.0 * self.config.hager_zhang_delta - 1.0) *
                     dphi0)
            tolerance = 1.0e-12 * max(
                1.0, abs(dphi0), abs(dphi))
            if dphi < lower - tolerance or dphi > upper + tolerance:
                raise RuntimeError(
                    'accepted approximate-Wolfe point violates derivative '
                    'bounds')
        else:
            armijo_constant = (
                self.config.hager_zhang_delta
                if line_search.line_search_method == 'hager-zhang' else
                self.config.line_search_c1)
            if (accepted.objective > state.objective +
                    armijo_constant * armijo_slope + 1.0e-12):
                raise RuntimeError(
                    'accepted line-search point does not satisfy Armijo '
                    'decrease')

    def _record(self, cycle: int, old: _GCState, new: _GCState, line_search: _LineSearchResult,
                dphi0: float, beta: float, restart_reason: str,
                optimizer: str = 'nlcg', search_direction_source: str = 'nlcg',
                diis_history_size: int = 0,
                diis_condition: float = np.nan,
                diis_coefficient_l1: float = np.nan,
                diis_damping: float = np.nan,
                diis_history_action: str = '',
                diis_predicted_residual_rms: float = np.nan,
                diis_trust_ratio: float = np.nan,
                diis_next_damping: float = np.nan) -> None:
        delta_objective, delta_nelec, density_change, _ = self._metrics(new, old)
        delta_omega = new.grand_potential - old.grand_potential
        self.history.append(IterationRecord(
            cycle, new.grand_potential, new.dft_total_energy,
            -new.chemical_potential * new.electron_number,
            new.entropy_energy, new.electron_number, delta_omega, delta_nelec,
            new.grad_rms, new.residual_rms, density_change, line_search.alpha, dphi0,
            beta, restart_reason, line_search.nfev, new.free_energy,
            new.chemical_potential, new.objective, delta_objective,
            optimizer, search_direction_source,
            line_search.strong_wolfe, line_search.message,
            diis_history_size, diis_condition, diis_coefficient_l1,
            diis_damping, diis_history_action,
            diis_predicted_residual_rms, diis_trust_ratio,
            diis_next_damping,
            self.nfev,
            line_search_method=line_search.line_search_method,
            weak_wolfe=line_search.weak_wolfe,
            approximate_wolfe=line_search.approximate_wolfe,
            curvature_qualified=line_search.curvature_qualified,
            line_search_objective_allowance=(
                line_search.objective_allowance),
            cheap_nelec_evaluations=(
                line_search.cheap_nelec_evaluations),
            cheap_nelec_alpha_reductions=(
                line_search.cheap_nelec_alpha_reductions),
            residual_filter_active=line_search.residual_filter_active,
            residual_filter_qualified=(
                line_search.residual_filter_qualified),
            residual_filter_ratio=line_search.residual_filter_ratio,
            residual_filter_rejections=(
                line_search.residual_filter_rejections)))
        if self.fixed_electron_number:
            self.log.info('Canonical cycle %d  A = %.12g  E_DFT = %.12g  '
                          'N = %.10g  mu = %.12g  |g|_rms = %.3g  '
                          '|F-H|_rms = %.3g  alpha = %.3g',
                          cycle, new.free_energy, new.dft_total_energy,
                          new.electron_number, new.chemical_potential,
                          new.grad_rms, new.residual_rms, line_search.alpha)
        else:
            self.log.info('GC cycle %d  Omega = %.12g  E_DFT = %.12g  N = %.10g  '
                          '|g|_rms = %.3g  |F-H|_rms = %.3g  alpha = %.3g',
                          cycle, new.grand_potential, new.dft_total_energy,
                          new.electron_number, new.grad_rms, new.residual_rms,
                          line_search.alpha)
    def _record_diis(
            self, cycle: int, old: _GCState, new: _GCState,
            step: _LineSearchResult, history_size: int, condition: float,
            coefficient_l1: float, history_action: str,
            predicted_residual_rms: float, trust_ratio: float,
            next_damping: float) -> None:
        self._record(
            cycle, old, new, step, np.nan, np.nan, history_action,
            optimizer='diis', search_direction_source='residual-diis',
            diis_history_size=history_size, diis_condition=condition,
            diis_coefficient_l1=coefficient_l1,
            diis_damping=step.alpha,
            diis_history_action=history_action,
            diis_predicted_residual_rms=predicted_residual_rms,
            diis_trust_ratio=trust_ratio,
            diis_next_damping=next_damping)
        self.log.info(
            'DIIS cycle %d  residual %.6g -> %.6g  delta objective = %.3g  '
            'delta N = %.3g  damping = %.3g -> %.3g  trust ratio = %.3g  '
            'history = %d  cond = %.3g  %s',
            cycle, old.residual_rms, new.residual_rms,
            new.objective - old.objective,
            new.electron_number - old.electron_number, step.alpha,
            next_damping, trust_ratio, history_size, condition,
            history_action)

    # ---- fixed-mu canonical continuation --------------------------------

    def _canonical_fixed_mu_candidate(
            self, result: GrandCanonicalResult | _FixedNPoint
            ) -> tuple[list, float, float, float]:
        """Return the density-preserving fixed-mu H and physical defects.

        A fixed-N auxiliary Hamiltonian has an arbitrary scalar gauge.  Its
        optimized chemical potential is physical, while its Fock matrix is
        already vacuum aligned.  Removing the H-F gauge and shifting from the
        optimized to the requested chemical potential preserves H-mu, hence
        every occupation and the density, exactly.
        """
        if self.fixed_electron_number:
            raise AssertionError(
                'a fixed-mu candidate requires a fixed-mu parent solver')
        mismatch = self.hermitize_blocks([
            h - f for h, f in zip(result.h_orth, result.fock_orth)])
        gauge_shift = self.trace_mean(mismatch)
        scalar_shift = self.mu - result.mu - gauge_shift
        h_fixed_mu = self._sanitize_h([
            h + scalar_shift * identity
            for h, identity in zip(result.h_orth, self.identity)])
        predicted_mismatch = self.hermitize_blocks([
            h - f for h, f in zip(h_fixed_mu, result.fock_orth)])
        delta_nelec = (
            self._electron_number_at_mu(result.fock_orth, self.mu) -
            result.electron_number)
        return (h_fixed_mu, gauge_shift, delta_nelec,
                self.rms(predicted_mismatch))

    def _solve_fixed_n_point(self, h0: Any) -> _FixedNPoint:
        """Converge one canonical point without publishing it to ``mf``."""
        if not self.fixed_electron_number:
            raise AssertionError('fixed-N point solve requires fixed N')
        self.history = []
        self.nfev = 0
        self._reset_run_diagnostics()
        state = self.evaluate(self._initial_h(h0=h0))
        outcome = self._run_diis(
            state, None, niter=0, cycle_start=0)
        return _FixedNPoint(
            state=outcome.state,
            converged=outcome.converged,
            message=outcome.message,
            niter=outcome.niter,
            nfev=self.nfev,
            history=tuple(self.history),
            density_change=outcome.density_change,
        )

    def _reset_run_diagnostics(self) -> None:
        self.ncheap_nelec_reject = 0
        self._last_trial_rejected_by_nelec = False
        self.ncheap_nelec_evaluations = 0
        self.ncheap_nelec_alpha_reductions = 0
        self.nresidual_filter_acceptances = 0
        self.nresidual_filter_rejections = 0
        self.canonical_verification_attempts = 0
        self.canonical_verification_evaluations = 0
        self.canonical_verification_failures = 0
        self.canonical_verification_residual_rms = np.nan
        self.canonical_verification_grad_rms = np.nan
        self.canonical_verification_delta_nelec = np.nan
        self.canonical_verification_density_rms = np.nan
        self.canonical_terminal_mode = ''

    def _canonical_continuation_config(
            self, residual_tolerance: float,
            initial_damping: float) -> GrandCanonicalConfig:
        """Return an immediate-DIIS configuration for one fixed-N solve."""
        return replace(
            self.config,
            canonical_continuation=False,
            checkpoint_path=None,
            checkpoint_interval=0,
            initial_electron_number=None,
            conv_tol_residual_rms=residual_tolerance,
            # Canonical continuation is a fixed-point globalization.  Enter
            # residual DIIS immediately instead of spending low-temperature
            # objective line searches to discover the same local model.
            diis_switch_residual_rms=max(1.0, residual_tolerance),
            diis_initial_damping=initial_damping,
            diis_max_coefficient_l1=max(
                self.config.diis_max_coefficient_l1,
                self.config.canonical_continuation_diis_max_coefficient_l1),
            required_consecutive_conv=1,
        )

    def _canonical_secant_step_cap(self) -> float:
        """Return the configured cap for post-initial secant proposals."""
        absolute = self.config.canonical_continuation_max_delta_nelec
        if absolute is not None:
            return float(absolute)
        return (
            self.config.canonical_continuation_max_delta_nelec_fraction *
            float(self.mf.cell.nelectron))

    def _canonical_continuation_proposal(
            self, samples: Sequence[tuple[float, float] | _CanonicalSample],
            physical_fock: Sequence, current_nelec: float) -> float:
        """Propose N from the latest secant, or the Fock for the first move."""
        samples = self._canonical_sample_coordinates(samples)
        minimum_step = self.config.canonical_continuation_min_delta_nelec
        proposal = None
        current_n, current_error = samples[-1]
        previous = next(
            ((n, error) for n, error in reversed(samples[:-1])
             if abs(n - current_n) >= minimum_step), None)
        if previous is not None:
            denominator = current_error - previous[1]
            if denominator != 0.0:
                local = (current_n - current_error *
                         (current_n - previous[0]) / denominator)
                if np.isfinite(local):
                    proposal = local
        if proposal is None:
            proposal = self._electron_number_at_mu(physical_fock, self.mu)
        maximum_step = (
            self.config.canonical_continuation_initial_delta_nelec
            if previous is None else self._canonical_secant_step_cap())
        if abs(proposal - current_nelec) < minimum_step:
            proposal = (
                current_nelec - np.sign(current_error or 1.0) * minimum_step)

        delta = min(maximum_step,
                    max(-maximum_step, proposal - current_nelec))
        capacity = 2.0 * sum(
            float(self.weights[k].item()) * n
            for k, n in enumerate(self.north))
        margin = min(minimum_step, 0.25 * capacity)
        return min(capacity - margin,
                   max(margin, current_nelec + delta))

    @staticmethod
    def _canonical_bracket(
            samples: Sequence[_CanonicalSample]
            ) -> Optional[tuple[_CanonicalSample, _CanonicalSample]]:
        """Return the narrowest evaluated sign bracket, ordered by N."""
        negative = [sample for sample in samples if sample.mu_error < 0.0]
        positive = [sample for sample in samples if sample.mu_error > 0.0]
        if not negative or not positive:
            return None
        left, right = min(
            ((negative_item, positive_item)
             for negative_item in negative for positive_item in positive),
            key=lambda pair: abs(
                pair[0].electron_number - pair[1].electron_number))
        return tuple(sorted(
            (left, right), key=lambda sample: sample.electron_number))

    @staticmethod
    def _canonical_sample_coordinates(
            samples: Sequence[tuple[float, float] | _CanonicalSample]
            ) -> list[tuple[float, float]]:
        return [
            (sample.electron_number, sample.mu_error)
            if isinstance(sample, _CanonicalSample) else sample
            for sample in samples]

    def _canonical_sample_index(
            self,
            samples: Sequence[_CanonicalSample],
            electron_number: float) -> Optional[int]:
        tolerance = self.config.canonical_continuation_root_nelec_tol
        return next((
            index for index, sample in enumerate(samples)
            if abs(sample.electron_number - electron_number) <= max(
                tolerance,
                32.0 * np.finfo(float).eps * max(
                    1.0, abs(sample.electron_number),
                    abs(electron_number)))), None)

    @staticmethod
    def _continuation_history(
            records: Sequence[IterationRecord], cycle_offset: int,
            electron_number: float,
            evaluation_offset: int = 0) -> list[IterationRecord]:
        prefix = f'canonical continuation N={electron_number:.12g}'
        return [replace(
            record,
            cycle=cycle_offset + record.cycle,
            fock_evaluations=(
                evaluation_offset + record.fock_evaluations),
            restart_reason=(prefix +
                            (('; ' + record.restart_reason)
                             if record.restart_reason else '')))
            for record in records]

    def _start_canonical_session(
            self, h: Sequence, electron_number: float,
            residual_tolerance: float,
            work: _CanonicalWork) -> tuple[_FixedNPoint, _FixedNSession]:
        """Start, advance, and account for one fixed-N DIIS session."""
        canonical_solver = self._spawn_fixed_n(
            electron_number,
            self._canonical_continuation_config(
                residual_tolerance,
                self.config.canonical_continuation_initial_damping))
        canonical_solver.history = []
        canonical_solver.nfev = 0
        canonical_solver._reset_run_diagnostics()
        state = canonical_solver.evaluate(
            canonical_solver._initial_h(h0=h))
        context = canonical_solver._new_diis_context(
            state, None, niter=0, cycle_start=0)
        session = _FixedNSession(
            float(electron_number), canonical_solver, context)
        point = self._advance_canonical_session(
            session, residual_tolerance, work)
        return point, session

    def _advance_canonical_session(
            self, session: _FixedNSession,
            residual_tolerance: float,
            work: _CanonicalWork) -> _FixedNPoint:
        """Resume a fixed-N session and publish only its incremental work."""
        solver = session.solver
        context = session.context
        evaluation_offset = work.total_nfev - session.published_nfev
        cycle_offset = work.fixed_n_niter - session.published_niter
        outcome = solver._advance_diis(context, residual_tolerance)

        nfev = solver.nfev - session.published_nfev
        niter = context.niter - session.published_niter
        records = solver.history[session.published_history:]
        work.history.extend(self._continuation_history(
            records, cycle_offset, session.electron_number,
            evaluation_offset))
        work.fixed_n_nfev += nfev
        work.fixed_n_niter += niter

        session.published_history = len(solver.history)
        session.published_nfev = solver.nfev
        session.published_niter = context.niter
        return _FixedNPoint(
            state=outcome.state,
            converged=outcome.converged,
            message=outcome.message,
            niter=niter,
            nfev=nfev,
            history=tuple(records),
            density_change=outcome.density_change,
        )

    def _observe_canonical_point(
            self, samples: list[_CanonicalSample], electron_number: float,
            mu_error: float, point: _FixedNPoint,
            session: Optional[_FixedNSession]) -> None:
        """Record one converged point in chronological secant order."""
        sample = _CanonicalSample(
            electron_number, mu_error, point.residual_rms,
            self.copy_blocks(point.fock_orth), session)
        duplicate = self._canonical_sample_index(samples, electron_number)
        if duplicate is None:
            samples.append(sample)
        else:
            # A tighter same-N solve supplies the latest secant observation.
            previous = samples.pop(duplicate)
            samples.append(
                sample if point.residual_rms <= previous.residual_rms
                else previous)

    def _prune_canonical_sessions(
            self, samples: list[_CanonicalSample],
            bracket: Optional[tuple[_CanonicalSample, _CanonicalSample]],
            ) -> None:
        """Retain only the latest sessions and active bracket endpoints."""
        keep = {id(sample) for sample in samples[-2:]}
        if bracket is not None:
            keep.update(id(sample) for sample in bracket)
        for index, sample in enumerate(samples):
            if sample.session is not None and id(sample) not in keep:
                samples[index] = replace(sample, session=None)

    def _verify_canonical_point(
            self, point: _FixedNPoint, h_fixed_mu: Sequence,
            work: _CanonicalWork,
            verification: _CanonicalVerification
            ) -> tuple[_GCState, bool]:
        """Spend exactly one parent Fock build on physical fixed-mu checks."""
        self.nfev = work.total_nfev
        before_verification = self.nfev
        state = self.evaluate(h_fixed_mu)
        verification_work = self.nfev - before_verification
        verification.attempts += 1
        work.verification_nfev += verification_work
        verification.residual_rms = state.residual_rms
        verification.grad_rms = state.grad_rms
        verification.delta_nelec = (
            self._electron_number_at_mu(state.fock_orth, self.mu) -
            state.electron_number)
        verification.density_rms = self.rms(
            self.axpy(-1.0, point.p_orth, state.p_orth))
        verification.last_state = state
        verification.last_source = point
        accepted = (
            verification_work == 1 and
            state.residual_rms <=
            self.config.canonical_continuation_verification_residual_tol and
            abs(verification.delta_nelec) <=
            self.config.canonical_continuation_handoff_delta_nelec and
            verification.density_rms <=
            self.config.canonical_continuation_verification_density_tol)
        return state, accepted

    def _finalize_canonical_search(
            self, terminal_state: _GCState,
            terminal_source: _FixedNPoint, terminal_success: bool,
            distinct_n_proposals: int, work: _CanonicalWork,
            verification: _CanonicalVerification) -> GrandCanonicalResult:
        """Publish the sole terminal canonical state and its global counts."""
        # Compute these before _finalize publishes the optimized state and
        # replaces self.mu with its chemical potential.
        mu_error = terminal_source.mu - self.mu
        (_, _, delta_nelec,
         _) = self._canonical_fixed_mu_candidate(terminal_source)
        self.nfev = work.total_nfev
        self.history = work.history
        density_change = self.rms(
            self.axpy(-1.0, terminal_source.p_orth,
                      terminal_state.p_orth))
        terminal_mode = (
            'canonical-verification' if terminal_success else
            'canonical-verification-failed')
        message = (
            f'canonical continuation ({distinct_n_proposals} fixed-N points, '
            f'{work.refinements} refinements, '
            f'{work.fixed_n_nfev} fixed-N Fock evaluations, '
            f'{work.verification_nfev} verification evaluations); ' +
            ('converged by one-Fock fixed-mu verification'
             if terminal_success else
             'failed to satisfy canonical root verification'))
        self.canonical_verification_attempts = verification.attempts
        self.canonical_verification_evaluations = work.verification_nfev
        self.canonical_verification_failures = verification.failures
        self.canonical_verification_residual_rms = verification.residual_rms
        self.canonical_verification_grad_rms = verification.grad_rms
        self.canonical_verification_delta_nelec = verification.delta_nelec
        self.canonical_verification_density_rms = verification.density_rms
        self.canonical_terminal_mode = terminal_mode
        self._checkpoint(terminal_state, work.total_niter, force=True)
        result = self._finalize(
            terminal_state, terminal_success, message,
            work.total_niter, density_change)
        self.mf.canonical_continuation_refinements_gc = work.refinements
        self.mf.scf_summary.update({
            'canonical_continuation_steps': distinct_n_proposals,
            'canonical_continuation_evaluations': work.fixed_n_nfev,
            'canonical_continuation_mu_error': mu_error,
            'canonical_continuation_delta_nelec': delta_nelec,
            'canonical_continuation_refinements': work.refinements,
            'fock_evaluations_total': work.total_nfev,
        })
        return replace(
            result,
            canonical_continuation_steps=distinct_n_proposals,
            canonical_continuation_evaluations=work.fixed_n_nfev,
            canonical_continuation_mu_error=mu_error,
            canonical_continuation_delta_nelec=delta_nelec,
            canonical_continuation_refinements=work.refinements,
        )

    def _kernel_canonical_continuation(
            self, dm0: Any = None, h0: Any = None) -> GrandCanonicalResult:
        """Globalize a fixed-mu solve through automatic fixed-N continuation."""
        if self.fixed_electron_number:
            raise AssertionError('canonical continuation requires fixed mu')
        self.history = []
        self.nfev = 0
        self._reset_run_diagnostics()
        h = self._initial_h(dm0, h0)
        work = _CanonicalWork(self.nfev, [])
        current_nelec = self._electron_number_at_mu(h, self.mu)
        samples: list[_CanonicalSample] = []
        bracket: Optional[tuple[_CanonicalSample, _CanonicalSample]] = None
        best_handoff_score = np.inf
        best_canonical_result: Optional[_FixedNPoint] = None
        last_canonical_result: Optional[_FixedNPoint] = None
        distinct_n_proposals = 0
        proposed_nelec: list[float] = []
        force_tight_refinement = False
        pending_session: Optional[_FixedNSession] = None
        allow_resume = True
        failed_inner_nelec: Optional[float] = None
        verification_repair_nelec: Optional[float] = None
        verification = _CanonicalVerification()
        reached_outer_limit = False
        max_passes = 4 * self.config.canonical_continuation_max_outer + 4
        pass_count = 0

        while True:
            if pass_count >= max_passes:
                self.log.warn(
                    'Canonical continuation reached its safety limit of %d '
                    'fixed-N passes; proceeding to fixed-mu verification',
                    max_passes)
                break
            pass_count += 1
            distinct_index = next((
                index for index, value in enumerate(proposed_nelec)
                if abs(value - current_nelec) <= max(
                    self.config.canonical_continuation_root_nelec_tol,
                    32.0 * np.finfo(float).eps *
                    max(1.0, abs(value), abs(current_nelec)))), None)
            is_distinct = distinct_index is None
            if (is_distinct and distinct_n_proposals >=
                    self.config.canonical_continuation_max_outer):
                reached_outer_limit = True
                break
            if is_distinct:
                distinct_index = distinct_n_proposals
                proposed_nelec.append(float(current_nelec))
                distinct_n_proposals += 1
            else:
                work.refinements += 1
                if pending_session is None and allow_resume:
                    sample_index = self._canonical_sample_index(
                        samples, current_nelec)
                    if sample_index is not None:
                        candidate_session = samples[sample_index].session
                        if (candidate_session is not None and
                                candidate_session.context.converged):
                            pending_session = candidate_session
            had_bracket = bracket is not None
            tight_canonical_solve = had_bracket or force_tight_refinement
            force_tight_refinement = False
            residual_tolerance = (
                self.config.canonical_continuation_bracketed_residual_tol
                if (tight_canonical_solve or had_bracket) else
                self.config.canonical_continuation_coarse_residual_tol)
            if pending_session is not None:
                session = pending_session
                pending_session = None
                canonical_result = self._advance_canonical_session(
                    session, residual_tolerance, work)
            else:
                canonical_result, session = self._start_canonical_session(
                    h, current_nelec, residual_tolerance, work)
            allow_resume = True
            last_canonical_result = canonical_result
            h = canonical_result.h_orth
            error = canonical_result.mu - self.mu
            target_nelec = self._electron_number_at_mu(
                canonical_result.fock_orth, self.mu)
            handoff_delta_nelec = (
                target_nelec - canonical_result.electron_number)
            if canonical_result.converged:
                self._observe_canonical_point(
                    samples, current_nelec, error, canonical_result, session)
                bracket = self._canonical_bracket(samples)
                self._prune_canonical_sessions(samples, bracket)
                bracket = self._canonical_bracket(samples)
                failed_inner_nelec = None
            bracketed = bracket is not None
            label = ('continuation' if is_distinct else 'refinement')
            self.log.info(
                'Canonical %s %d: N = %.12g, optimized mu = '
                '%.12g, target mu = %.12g, delta mu = %.3g, residual = %.3g, '
                'physical delta N = %.3g, Fock evaluations = %d',
                label, distinct_index, current_nelec,
                canonical_result.mu, self.mu, error,
                canonical_result.residual_rms, handoff_delta_nelec,
                canonical_result.nfev)

            handoff_score = max(
                abs(handoff_delta_nelec) /
                (self.config.canonical_continuation_handoff_delta_nelec
                 if bracketed else
                 self.config.
                 canonical_continuation_unbracketed_handoff_delta_nelec),
                abs(error) /
                self.config.canonical_continuation_handoff_delta_mu)
            if (canonical_result.converged and
                    handoff_score < best_handoff_score):
                best_handoff_score = handoff_score
                best_canonical_result = canonical_result

            # Canonical continuation always terminates by fixed-mu verification.
            if not canonical_result.converged:
                self.log.warn(
                    'Canonical continuation inner solve did not '
                    'converge: %s', canonical_result.message)
                if best_canonical_result is None:
                    break
                same_failure = (
                    failed_inner_nelec is not None and
                    abs(failed_inner_nelec - current_nelec) <=
                    32.0 * np.finfo(float).eps *
                    max(1.0, abs(current_nelec)))
                if not same_failure:
                    # A failed solve is not a scalar observation.  Retry
                    # its N once from the best physical Fock while leaving
                    # the evaluated sign bracket untouched.
                    failed_inner_nelec = current_nelec
                    bracket = self._canonical_bracket(samples)
                    h = self.copy_blocks(
                        best_canonical_result.fock_orth)
                    force_tight_refinement = True
                    pending_session = None
                    allow_resume = False
                    self.log.warn(
                        'Retrying failed fixed-N solve once at N = %.12g',
                        current_nelec)
                    continue

                retry_nelec = None
                bracket = self._canonical_bracket(samples)
                if bracket is not None:
                    lo, hi = (sample.electron_number for sample in bracket)
                    retry_nelec = lo + 0.5 * (hi - lo)
                    if abs(retry_nelec - current_nelec) <= (
                            32.0 * np.finfo(float).eps *
                            max(1.0, abs(current_nelec))):
                        # The failed point was already the midpoint.  Split
                        # its interval to the best endpoint instead of
                        # immediately abandoning a valid outer bracket.
                        best_endpoint = min(
                            bracket, key=lambda sample: abs(sample.mu_error))
                        retry_nelec = (
                            current_nelec +
                            0.5 * (best_endpoint.electron_number -
                                   current_nelec))
                elif samples:
                    retry_nelec = 0.5 * (
                        current_nelec + samples[-1].electron_number)
                if (retry_nelec is not None and
                        abs(retry_nelec - current_nelec) >
                        1.0e-14 * max(1.0, abs(current_nelec))):
                    h = self.copy_blocks(
                        best_canonical_result.fock_orth)
                    current_nelec = retry_nelec
                    failed_inner_nelec = None
                    force_tight_refinement = True
                    pending_session = None
                    allow_resume = False
                    self.log.warn(
                        'Retrying fixed-N continuation at safeguarded '
                        'N = %.12g', current_nelec)
                    continue
                break
            (h_fixed_mu, _, physical_delta_nelec, _) = (
                self._canonical_fixed_mu_candidate(canonical_result))
            canonical_ready = (
                canonical_result.residual_rms <=
                self.config.
                canonical_continuation_bracketed_residual_tol and
                abs(error) <=
                self.config.canonical_continuation_handoff_delta_mu and
                abs(physical_delta_nelec) <=
                self.config.
                canonical_continuation_handoff_delta_nelec)
            if canonical_ready:
                candidate_state, verification_ok = (
                    self._verify_canonical_point(
                        canonical_result, h_fixed_mu, work, verification))
                if verification_ok:
                    verification.verified_state = candidate_state
                    verification.verified_source = canonical_result
                    self.log.info(
                        'Canonical root verified at fixed mu with one '
                        'Fock build: delta mu = %.3g, delta N = %.3g, '
                        'residual = %.3g, gradient = %.3g',
                        error, physical_delta_nelec,
                        candidate_state.residual_rms,
                        candidate_state.grad_rms)
                    break
                verification.failures += 1
                self.log.warn(
                    'One-Fock fixed-mu verification failed: residual '
                    '%.3g, gradient %.3g, delta N %.3g, density change '
                    '%.3g; restarting fixed-N repair from the verification '
                    'Fock',
                    candidate_state.residual_rms,
                    candidate_state.grad_rms,
                    verification.delta_nelec,
                    verification.density_rms)
                same_repair = (
                    verification_repair_nelec is not None and
                    abs(verification_repair_nelec -
                        canonical_result.electron_number) <=
                    32.0 * np.finfo(float).eps * max(
                        1.0, abs(canonical_result.electron_number)))
                if not same_repair:
                    # Verification exposed residual shape error rather than
                    # a new scalar observation.  Re-solve the same N once
                    # from the newly evaluated physical Fock.
                    verification_repair_nelec = (
                        canonical_result.electron_number)
                    h = self.copy_blocks(candidate_state.fock_orth)
                    current_nelec = canonical_result.electron_number
                    bracket = self._canonical_bracket(samples)
                    force_tight_refinement = True
                    pending_session = None
                    allow_resume = False
                    continue

                # A persistent failure must not cycle forever at the same
                # N.  Use the verification Fock response only to choose a
                # safeguarded interior recovery point; its optimized mu is
                # evaluated by the next fixed-N solve before secant sees it.
                bracket = self._canonical_bracket(samples)
                if bracket is not None:
                    lo, hi = (sample.electron_number for sample in bracket)
                    preferred = (
                        canonical_result.electron_number +
                        verification.delta_nelec)
                    margin = min(
                        self.config.canonical_continuation_root_nelec_tol,
                        0.25 * (hi - lo))
                    if lo + margin < preferred < hi - margin:
                        retry_nelec = preferred
                    else:
                        retry_nelec = lo + 0.5 * (hi - lo)
                    no_progress_tolerance = max(
                        self.config.canonical_continuation_root_nelec_tol,
                        32.0 * np.finfo(float).eps *
                        max(1.0, abs(current_nelec)))
                    if (abs(retry_nelec - current_nelec) <=
                            no_progress_tolerance):
                        best_endpoint = min(
                            bracket, key=lambda sample: abs(sample.mu_error))
                        retry_nelec = (
                            current_nelec +
                            0.5 * (best_endpoint.electron_number -
                                   current_nelec))
                    if (abs(retry_nelec - current_nelec) <=
                            no_progress_tolerance):
                        self.log.warn(
                            'Persistent fixed-mu verification recovery made '
                            'no resolvable electron-number progress')
                        break
                    h = self.copy_blocks(candidate_state.fock_orth)
                    current_nelec = retry_nelec
                    verification_repair_nelec = None
                    force_tight_refinement = True
                    pending_session = None
                    allow_resume = False
                    continue
                self.log.warn(
                    'Persistent fixed-mu verification failure left no '
                    'resolvable sign bracket')
                break

            root_coordinates_ready = (
                abs(error) <=
                self.config.canonical_continuation_handoff_delta_mu and
                abs(physical_delta_nelec) <=
                self.config.
                canonical_continuation_handoff_delta_nelec)
            if root_coordinates_ready:
                # The scalar root is ready but the fixed-N shape is not;
                # converge the same physical state to the tight inner
                # tolerance before spending the verification Fock.
                h = self.copy_blocks(canonical_result.h_orth)
                current_nelec = canonical_result.electron_number
                force_tight_refinement = True
                pending_session = session
                allow_resume = False
                continue

            if bracket is not None:
                # A coarse value can supply the first sign change but may not
                # be accurate enough to steer an expensive secant iteration.
                # Re-solve the best endpoint before trusting the samples, and
                # tighten again near the root so the requested physical charge
                # tolerance is not dominated by fixed-N residual noise.
                best_endpoint = min(
                    bracket, key=lambda sample: abs(sample.mu_error))
                endpoint_index = self._canonical_sample_index(
                    samples, best_endpoint.electron_number)
                endpoint_tolerance = (
                    self.config.canonical_continuation_bracketed_residual_tol)
                if (endpoint_index is not None and
                        samples[endpoint_index].residual_rms >
                        endpoint_tolerance):
                    h = self.copy_blocks(samples[endpoint_index].fock_orth)
                    current_nelec = best_endpoint.electron_number
                    force_tight_refinement = True
                    pending_session = best_endpoint.session
                    allow_resume = False
                    self.log.info(
                        'Refining secant endpoint N = %.12g from residual %.3g '
                        'to %.3g before the next root proposal',
                        current_nelec, samples[endpoint_index].residual_rms,
                        endpoint_tolerance)
                    continue
                bracket_width = abs(
                    bracket[1].electron_number -
                    bracket[0].electron_number)
                if bracket_width <= (
                        self.config.canonical_continuation_root_nelec_tol):
                    self.log.warn(
                        'Canonical continuation sign bracket resolved at '
                        'width %.3g without satisfying physical verification',
                        bracket_width)
                    break

            proposal = self._canonical_continuation_proposal(
                samples, canonical_result.fock_orth, current_nelec)
            proposal_h = None
            if bracket is not None:
                left, right = bracket
                if (left.electron_number <= proposal <=
                        right.electron_number):
                    fraction = (
                        (proposal - left.electron_number) /
                        (right.electron_number - left.electron_number))
                    proposal_h = self._sanitize_h([
                        (1.0 - fraction) * h_left + fraction * h_right
                        for h_left, h_right in zip(
                            left.fock_orth, right.fock_orth)])
            method = 'Fock-response' if len(samples) == 1 else 'secant'
            self.log.info(
                'Canonical continuation %s proposal N = %.12g '
                '(maximum |delta N| = %.12g)',
                method, proposal,
                (self.config.canonical_continuation_initial_delta_nelec
                 if len(samples) == 1 else
                 self._canonical_secant_step_cap()))
            if abs(proposal - current_nelec) <= (
                    1.0e-14 * max(1.0, abs(current_nelec))):
                self.log.warn(
                    'Canonical continuation scalar root stagnated at '
                    'N = %.12g', current_nelec)
                break
            h = (proposal_h if proposal_h is not None else
                 self.copy_blocks(canonical_result.fock_orth))
            current_nelec = proposal
        if reached_outer_limit:
            self.log.warn(
                'Canonical continuation reached its maximum of %d distinct '
                'N proposals; '
                'proceeding to fixed-mu verification',
                self.config.canonical_continuation_max_outer)

        # Canonical continuation always returns its fixed-mu verification.
        terminal_success = verification.verified_state is not None
        terminal_state = verification.verified_state
        terminal_source = verification.verified_source
        if terminal_state is None:
            reuse_last_verification = (
                verification.last_state is not None and
                (best_canonical_result is None or
                 verification.last_source is best_canonical_result))
            if reuse_last_verification:
                terminal_state = verification.last_state
                terminal_source = verification.last_source
            else:
                terminal_source = (
                    best_canonical_result or last_canonical_result)
                if terminal_source is None:  # pragma: no cover
                    raise RuntimeError(
                        'canonical continuation produced no state')
                (fallback_h, _, _,
                 _) = self._canonical_fixed_mu_candidate(terminal_source)
                self.nfev = work.total_nfev
                before_verification = self.nfev
                terminal_state = self.evaluate(fallback_h)
                verification_work = self.nfev - before_verification
                verification.attempts += 1
                work.verification_nfev += verification_work
                verification.failures += 1
                verification.residual_rms = terminal_state.residual_rms
                verification.grad_rms = terminal_state.grad_rms
                verification.delta_nelec = (
                    self._electron_number_at_mu(
                        terminal_state.fock_orth, self.mu) -
                    terminal_state.electron_number)
                verification.density_rms = self.rms(
                    self.axpy(-1.0, terminal_source.p_orth,
                              terminal_state.p_orth))

        if terminal_source is None:  # pragma: no cover
            raise RuntimeError(
                'fixed-mu verification lacks its canonical source')
        return self._finalize_canonical_search(
            terminal_state, terminal_source, terminal_success,
            distinct_n_proposals, work, verification)

    def kernel(self, dm0: Any = None, h0: Any = None) -> GrandCanonicalResult:
        """Run the configured safeguarded direct minimizer."""
        if (self.config.canonical_continuation and
                not self.fixed_electron_number):
            return self._kernel_canonical_continuation(dm0=dm0, h0=h0)
        return self._kernel_nlcg(dm0=dm0, h0=h0)

    def _kernel_nlcg(self, dm0: Any = None,
                     h0: Any = None) -> GrandCanonicalResult:
        """Run safeguarded fixed-mu or fixed-electron nonlinear CG."""
        self.history = []
        self.nfev = 0
        self._reset_run_diagnostics()
        self._nlcg_residual_previous_alpha = None
        state = self.evaluate(self._initial_h(dm0, h0))
        previous: Optional[_GCState] = None
        direction = self.copy_blocks(state.residual)
        consecutive = 0
        message = 'maximum cycles reached'
        converged = False
        niter = 0
        for cycle in range(self.config.max_cycle):
            if self._meets_convergence(state, previous):
                consecutive += 1
                if previous is None or consecutive >= self.config.required_consecutive_conv:
                    converged, message = True, 'converged'
                    break
            else:
                consecutive = 0
            if self._should_start_diis(state):
                self.log.info(
                    'Switching from NLCG to residual DIIS at |F-H|_rms = %.6g; '
                    'CG memory reset', state.residual_rms)
                return self._kernel_diis(
                    state, previous, niter=niter, cycle_start=cycle)
            direction, restarted, restart_reason = self._ensure_descent(state, direction)
            if not self._is_descent(state, direction):
                if state.grad_rms < self.config.conv_tol_grad_rms and state.residual_rms < self.config.conv_tol_residual_rms:
                    converged, message = True, 'stationary initial state'
                else:
                    message = 'persistent loss of descent'
                break
            if self._alpha_cap(direction) < self.config.line_search_alpha_min:
                direction, cap_reason = self._restart_direction(state)
                if self._alpha_cap(direction) < self.config.line_search_alpha_min:
                    message = 'stagnation: step cap below minimum'
                    break
                restarted, restart_reason = True, 'step cap restart; ' + cap_reason
            dphi0 = self.inner(state.gradient, direction)
            alpha_init = self._nlcg_residual_alpha_init(state)
            line_search = self._line_search(
                state, direction, alpha_init=alpha_init)
            if not line_search.success:
                primary_line_search = line_search
                direction, fallback_reason = self._restart_direction(state)
                fallback_max_evals = None
                filter_max_evals = (
                    self.config.nlcg_residual_filter_max_evals)
                filter_threshold = self.config.nlcg_residual_filter_rms
                if (filter_max_evals is not None and
                        filter_threshold is not None and
                        state.residual_rms <= filter_threshold):
                    fallback_max_evals = max(
                        0, filter_max_evals - primary_line_search.nfev)
                if fallback_max_evals == 0:
                    fallback_line_search = _LineSearchResult(
                        False, None, line_search_method='armijo',
                        message=('residual-filter line-search evaluation '
                                 'cap exhausted'))
                else:
                    fallback_line_search = self._armijo_fallback(
                        state, direction,
                        alpha_init=(
                            None if alpha_init is None else
                            alpha_init *
                            self.config.armijo_backtrack_factor),
                        max_evals_override=fallback_max_evals)
                line_search = self._combine_line_search_work(
                    primary_line_search, fallback_line_search)
                restarted = True
                restart_reason = fallback_reason + '; ' + line_search.message
                dphi0 = self.inner(state.gradient, direction)
            if not line_search.success or line_search.state is None:
                message = 'line-search failure: ' + line_search.message
                break
            if alpha_init is not None:
                self._nlcg_residual_previous_alpha = line_search.alpha
            new_state = line_search.state
            self._verify_accepted_step(
                state, new_state, direction, line_search, dphi0)
            beta = 0.0
            if restarted or line_search.force_restart:
                restart_reason = restart_reason or line_search.message
            elif cycle > 0 and cycle % self.config.cg_restart_interval != 0:
                beta, beta_reason = self._cg_beta(state, new_state, direction)
                restart_reason = restart_reason or beta_reason
            else:
                restart_reason = restart_reason or 'scheduled CG restart'
            self._record(cycle, state, new_state, line_search, dphi0, beta, restart_reason)
            niter += 1
            self._checkpoint(new_state, niter)
            old_direction = direction
            state, previous = new_state, state
            proposed = self.axpy(beta, old_direction, state.residual)
            direction, lost_descent, descent_reason = self._ensure_descent(state, proposed)
            if lost_descent:
                restart_reason = descent_reason
        else:
            # The loop did not break; evaluate the just-accepted final state.
            if self._meets_convergence(state, previous):
                consecutive += 1
                if consecutive >= self.config.required_consecutive_conv:
                    converged, message = True, 'converged at maximum cycle'
        density_change = (0.0 if previous is None else self._metrics(state, previous)[2])
        return self._finalize(state, converged, message, niter, density_change)

    def _new_diis_context(
            self, state: _GCState, previous: Optional[_GCState],
            niter: int, cycle_start: int) -> _DIISRunContext:
        return _DIISRunContext(
            state=state,
            previous=previous,
            history=[],
            damping_hint=self.config.diis_initial_damping,
            niter=niter,
            next_cycle=cycle_start,
        )

    def _advance_diis(
            self, context: _DIISRunContext,
            residual_tolerance: Optional[float] = None) -> _KernelOutcome:
        """Advance a fresh or previously stopped residual-DIIS context."""
        tolerance = (self.config.conv_tol_residual_rms
                     if residual_tolerance is None else
                     float(residual_tolerance))
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError('DIIS residual tolerance must be positive')
        context.converged = False
        context.message = (
            'maximum cycles reached during residual-DIIS polishing')

        for cycle in range(context.next_cycle, self.config.max_cycle):
            state = context.state
            if state.residual_rms < tolerance:
                context.converged = True
                context.message = 'converged residual-DIIS fixed point'
                break
            self._append_diis_item(context.history, state)
            starting_damping = context.damping_hint
            result = self._diis_step(
                state, context.history, starting_damping)
            history_size = len(context.history)
            step = result.step
            if not step.success or step.state is None:
                context.message = step.message
                break
            new_state = step.state
            context.damping_hint, trust_ratio = self._next_diis_damping(
                state, new_state, result.predicted_residual_rms,
                step.alpha, starting_damping)
            history_action = result.history_action
            self._record_diis(
                cycle, state, new_state, step, history_size,
                result.condition, result.coefficient_l1,
                history_action, result.predicted_residual_rms,
                trust_ratio, context.damping_hint)
            context.niter += 1
            context.next_cycle = cycle + 1
            self._checkpoint(new_state, context.niter)
            context.previous, context.state = state, new_state
        else:
            if context.state.residual_rms < tolerance:
                context.converged = True
                context.message = (
                    'converged residual-DIIS fixed point at maximum cycle')
        density_change = (
            0.0 if context.previous is None else
            self._metrics(context.state, context.previous)[2])
        return _KernelOutcome(
            context.state, context.previous, context.converged,
            context.message, context.niter, density_change)

    def _run_diis(self, state: _GCState, previous: Optional[_GCState],
                  niter: int, cycle_start: int) -> _KernelOutcome:
        """Run a fresh DIIS context and return its unpublished outcome."""
        return self._advance_diis(self._new_diis_context(
            state, previous, niter, cycle_start))

    def _kernel_diis(self, state: _GCState, previous: Optional[_GCState],
                     niter: int, cycle_start: int) -> GrandCanonicalResult:
        """Run DIIS and publish its terminal state for a public solve."""
        outcome = self._run_diis(state, previous, niter, cycle_start)
        return self._finalize(
            outcome.state, outcome.converged, outcome.message,
            outcome.niter, outcome.density_change)

    # ---- public state finalisation ----------------------------------------

    def _finalize(self, state: _GCState, converged: bool, message: str,
                  niter: int, density_change: float) -> GrandCanonicalResult:
        coeff = [x @ u for x, u in zip(self.x_ao2orth, state.u)]
        energy = [state.auxiliary_mu + value / self.beta - state.gauge_shift
                  for value in state.eigenvalues]
        occ = [2.0 * q for q in state.occupations]
        mo_coeff = _stack_or_list(coeff)
        mo_energy = _stack_or_list(energy)
        mo_occ = _stack_or_list(occ)
        self.mf.converged = converged
        self.mf.mo_coeff = mo_coeff
        self.mf.mo_energy = mo_energy
        self.mf.mo_occ = mo_occ
        self.mf.e_tot = state.dft_total_energy
        self.mf.free_energy = state.free_energy
        self.mf.grand_potential = state.grand_potential
        self.mf.electron_number_gc = state.electron_number
        self.mf.entropy_gc = state.entropy
        self.mf.entropy_energy_gc = state.entropy_energy
        self.mu = state.chemical_potential
        self.mf.mu_gc = state.chemical_potential
        self.mf.sigma_gc = self.sigma
        self.mf.cheap_nelec_rejections_gc = self.ncheap_nelec_reject
        self.mf.cheap_nelec_evaluations_gc = (
            self.ncheap_nelec_evaluations)
        self.mf.cheap_nelec_alpha_reductions_gc = (
            self.ncheap_nelec_alpha_reductions)
        self.mf.residual_filter_acceptances_gc = (
            self.nresidual_filter_acceptances)
        self.mf.residual_filter_rejections_gc = (
            self.nresidual_filter_rejections)
        self.mf.canonical_verification_attempts_gc = (
            self.canonical_verification_attempts)
        self.mf.canonical_verification_evaluations_gc = (
            self.canonical_verification_evaluations)
        self.mf.canonical_verification_failures_gc = (
            self.canonical_verification_failures)
        self.mf.canonical_verification_residual_rms_gc = (
            self.canonical_verification_residual_rms)
        self.mf.canonical_verification_grad_rms_gc = (
            self.canonical_verification_grad_rms)
        self.mf.canonical_verification_delta_nelec_gc = (
            self.canonical_verification_delta_nelec)
        self.mf.canonical_verification_density_rms_gc = (
            self.canonical_verification_density_rms)
        self.mf.canonical_terminal_mode_gc = self.canonical_terminal_mode
        self.mf.fixed_electron_number = self.fixed_electron_number
        self.mf.target_electron_number = self.target_electron_number
        self.mf.h_aux_gc = self.copy_blocks(state.h_orth)
        self.mf.dm_gc = state.dm_ao
        if not hasattr(self.mf, 'scf_summary') or self.mf.scf_summary is None:
            self.mf.scf_summary = {}
        self.mf.scf_summary.update({
            'grand_potential': state.grand_potential,
            'free_energy': state.free_energy,
            'electron_number_gc': state.electron_number,
            'entropy_gc': state.entropy,
            'entropy_energy_gc': state.entropy_energy,
            'mu_gc': state.chemical_potential,
            'sigma_gc': self.sigma,
            'cheap_nelec_rejections_gc': self.ncheap_nelec_reject,
            'cheap_nelec_evaluations_gc': (
                self.ncheap_nelec_evaluations),
            'cheap_nelec_alpha_reductions_gc': (
                self.ncheap_nelec_alpha_reductions),
            'residual_filter_acceptances_gc': (
                self.nresidual_filter_acceptances),
            'residual_filter_rejections_gc': (
                self.nresidual_filter_rejections),
            'canonical_verification_attempts': (
                self.canonical_verification_attempts),
            'canonical_verification_evaluations': (
                self.canonical_verification_evaluations),
            'canonical_verification_failures': (
                self.canonical_verification_failures),
            'canonical_verification_residual_rms': (
                self.canonical_verification_residual_rms),
            'canonical_verification_grad_rms': (
                self.canonical_verification_grad_rms),
            'canonical_verification_delta_nelec': (
                self.canonical_verification_delta_nelec),
            'canonical_verification_density_rms': (
                self.canonical_verification_density_rms),
            'canonical_terminal_mode': self.canonical_terminal_mode,
            'fixed_electron_number': self.fixed_electron_number,
        })
        return GrandCanonicalResult(
            converged, message, niter, self.nfev, state.chemical_potential,
            self.sigma, self.beta,
            state.grand_potential, state.dft_total_energy, state.electronic_energy,
            state.nuclear_energy, state.entropy, state.entropy_energy,
            state.electron_number, self.copy_blocks(state.h_orth),
            self.copy_blocks(state.fock_orth), state.dm_ao, self.copy_blocks(state.p_orth),
            self.copy_blocks(state.occupations), mo_coeff, mo_occ, mo_energy,
            state.grad_rms, state.residual_rms, density_change, list(self.history),
            state.veff, self.config.checkpoint_path, state.free_energy,
            self.fixed_electron_number, self.target_electron_number,
            self.ncheap_nelec_reject,
            cheap_nelec_evaluations=self.ncheap_nelec_evaluations,
            cheap_nelec_alpha_reductions=(
                self.ncheap_nelec_alpha_reductions),
            residual_filter_acceptances=(
                self.nresidual_filter_acceptances),
            residual_filter_rejections=(
                self.nresidual_filter_rejections),
            canonical_verification_attempts=(
                self.canonical_verification_attempts),
            canonical_verification_evaluations=(
                self.canonical_verification_evaluations),
            canonical_verification_failures=(
                self.canonical_verification_failures),
            canonical_verification_residual_rms=(
                self.canonical_verification_residual_rms),
            canonical_verification_grad_rms=(
                self.canonical_verification_grad_rms),
            canonical_verification_delta_nelec=(
                self.canonical_verification_delta_nelec),
            canonical_verification_density_rms=(
                self.canonical_verification_density_rms),
            canonical_terminal_mode=self.canonical_terminal_mode)
