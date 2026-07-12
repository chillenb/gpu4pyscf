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
class _BrentRoot:
    """Stateful safeguarded Brent--Dekker scalar root iteration.

    ``a`` and ``b`` always bracket a root and ``b`` is the endpoint with the
    smaller residual magnitude.  A proposal must be evaluated and supplied to
    :meth:`update` before another proposal is requested.  Keeping this state
    outside the fixed-N solver is important: each scalar function evaluation
    is itself a fully converged electronic-structure calculation.
    """

    a: float
    fa: float
    b: float
    fb: float
    xtol: float
    c: float
    fc: float
    d: float
    mflag: bool = True
    pending: Optional[float] = None
    last_method: str = 'initial-bracket'
    interpolation_steps: int = 0
    bisection_steps: int = 0

    @classmethod
    def from_bracket(
            cls, x0: float, f0: float, x1: float, f1: float,
            xtol: float) -> '_BrentRoot':
        values = (x0, f0, x1, f1, xtol)
        if not all(np.isfinite(value) for value in values):
            raise ValueError('Brent bracket values and tolerance must be finite')
        if xtol <= 0.0:
            raise ValueError('Brent x tolerance must be positive')
        if x0 == x1:
            raise ValueError('Brent bracket endpoints must be distinct')
        if f0 == 0.0 or f1 == 0.0 or np.signbit(f0) == np.signbit(f1):
            raise ValueError('Brent bracket must have strictly opposite signs')
        a, fa, b, fb = float(x0), float(f0), float(x1), float(f1)
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        return cls(
            a=a, fa=fa, b=b, fb=fb, xtol=float(xtol),
            c=a, fc=fa, d=a)

    @property
    def bracket(self) -> tuple[float, float]:
        return tuple(sorted((self.a, self.b)))

    @property
    def width(self) -> float:
        return abs(self.b - self.a)

    @property
    def converged(self) -> bool:
        """Whether the root or its representable x interval is resolved."""
        lo, hi = self.bracket
        return (
            self.fb == 0.0 or self.width <= self.xtol or
            not lo < np.nextafter(lo, hi) < hi)

    def proposal(self) -> float:
        """Return the next inverse-quadratic/secant or bisection proposal."""
        if self.pending is not None:
            raise RuntimeError('the pending Brent proposal has not been updated')
        if self.converged:
            raise RuntimeError('the Brent root interval has converged')

        use_iqi = (
            self.fa != self.fc and self.fb != self.fc and
            self.fa != self.fb)
        if use_iqi:
            s = (
                self.a * self.fb * self.fc /
                ((self.fa - self.fb) * (self.fa - self.fc)) +
                self.b * self.fa * self.fc /
                ((self.fb - self.fa) * (self.fb - self.fc)) +
                self.c * self.fa * self.fb /
                ((self.fc - self.fa) * (self.fc - self.fb)))
            method = 'inverse-quadratic'
        else:
            denominator = self.fb - self.fa
            s = (self.b - self.fb * (self.b - self.a) / denominator
                 if denominator != 0.0 else np.nan)
            method = 'secant'

        # Brent's acceptance tests reject interpolation that is outside the
        # protected part of the bracket or is not contracting quickly enough.
        protected = 0.75 * self.a + 0.25 * self.b
        protected_lo, protected_hi = sorted((protected, self.b))
        reject = (
            not np.isfinite(s) or
            not (protected_lo < s < protected_hi) or
            (self.mflag and
             abs(s - self.b) >= 0.5 * abs(self.b - self.c)) or
            (not self.mflag and
             abs(s - self.b) >= 0.5 * abs(self.c - self.d)) or
            (self.mflag and abs(self.b - self.c) < self.xtol) or
            (not self.mflag and abs(self.c - self.d) < self.xtol))
        if reject:
            s = self.a + 0.5 * (self.b - self.a)
            self.mflag = True
            method = 'bisection'
        else:
            self.mflag = False

        # Roundoff must not turn a valid interpolation into an endpoint
        # reevaluation.  Enforce the usual minimum move toward the opposite
        # endpoint; ``converged`` handles intervals too narrow to resolve.
        lo, hi = self.bracket
        minimum_step = min(self.xtol, 0.5 * (hi - lo))
        if abs(s - self.b) < minimum_step:
            s = self.b + np.copysign(minimum_step, self.a - self.b)
            self.mflag = True
            method = 'bisection'
        if not (lo < s < hi):
            s = lo + 0.5 * (hi - lo)
            self.mflag = True
            method = 'bisection'
        if not (lo < s < hi):  # pragma: no cover - guarded by converged
            raise RuntimeError('no representable point inside Brent bracket')
        if method == 'bisection':
            self.bisection_steps += 1
        else:
            self.interpolation_steps += 1
        self.pending = float(s)
        self.last_method = method
        return self.pending

    def update(self, x: float, fx: float) -> None:
        """Update the sign-changing bracket with an evaluated proposal."""
        if self.pending is None:
            raise RuntimeError('Brent update requires a pending proposal')
        scale = max(1.0, abs(self.pending), abs(x))
        if abs(x - self.pending) > 32.0 * np.finfo(float).eps * scale:
            raise ValueError('Brent update does not match its pending proposal')
        if not np.isfinite(fx):
            raise ValueError('Brent function value must be finite')
        lo, hi = self.bracket
        x_value = self.pending
        if not lo < x_value < hi:
            raise ValueError('Brent update must lie strictly inside its bracket')

        old_b, old_fb = self.b, self.fb
        self.d = self.c
        self.c = old_b
        self.fc = old_fb
        if fx == 0.0:
            self.a, self.fa = old_b, old_fb
            self.b, self.fb = x_value, 0.0
        elif np.signbit(self.fa) != np.signbit(fx):
            self.b, self.fb = x_value, float(fx)
        else:
            self.a, self.fa = x_value, float(fx)
        if abs(self.fa) < abs(self.fb):
            self.a, self.b = self.b, self.a
            self.fa, self.fb = self.fb, self.fa
        self.pending = None


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

    optimizer: str = 'nlcg'
    lbfgs_history_size: int = 5
    lbfgs_curvature_tol: float = 1.0e-8
    lbfgs_min_pair_step_rms: float = 1.0e-12
    lbfgs_descent_cosine_min: float = 1.0e-4
    lbfgs_initial_metric: str = 'fermi'
    lbfgs_inverse_metric_cap: float = 1.0
    lbfgs_metric_scale_min: float = 0.1
    lbfgs_metric_scale_max: float = 10.0
    lbfgs_scalar_h0_min: float = 1.0e-8
    lbfgs_scalar_h0_max: float = 1.0
    lbfgs_line_search_c2: float = 0.9
    lbfgs_cap_unit_step_with_history: bool = True
    lbfgs_clear_on_non_wolfe: bool = True

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
    diis_model_max_backtracks: int = 2
    diis_max_trust_model_repairs: int = 2
    diis_trust_interpolation_residual_increase: float = 0.10
    diis_min_residual_reduction: float = 1.0e-3
    diis_max_objective_increase: float = 1.0e-5
    diis_max_delta_nelec: float = 5.0e-2
    diis_trust_shrink_ratio: float = 0.25
    diis_trust_expand_ratio: float = 0.75
    diis_trust_expansion: float = 2.0
    diis_trust_expand_min_relative_reduction: float = 2.0e-2
    diis_max_restoration_residual_increase: float = 0.0

    # Optional fixed-mu globalization through canonical continuation.  Each
    # inner solve fixes N; the outer scalar Brent iteration zeros the optimized
    # chemical-potential error before a one-Fock fixed-mu verification.
    # Iterative fixed-mu polishing remains available as an opt-in compatibility
    # mode.
    canonical_continuation: bool = False
    canonical_continuation_max_outer: int = 16
    canonical_continuation_coarse_residual_tol: float = 4.0e-6
    canonical_continuation_bracketed_residual_tol: float = 1.0e-8
    canonical_continuation_handoff_delta_nelec: float = 2.0e-5
    canonical_continuation_unbracketed_handoff_delta_nelec: float = 2.0e-5
    canonical_continuation_initial_delta_nelec: float = 3.0e-2
    canonical_continuation_max_delta_nelec: float = 1.0
    canonical_continuation_min_delta_nelec: float = 1.0e-5
    canonical_continuation_initial_damping: float = 0.125
    canonical_continuation_final_damping: float = 1.0 / 256.0
    canonical_continuation_diis_max_coefficient_l1: float = 50.0
    canonical_continuation_interpolation_refine_width: float = 5.0e-2

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

    # Appended to preserve positional compatibility of the pre-existing
    # public configuration dataclass.  Keyword construction is recommended.
    line_search_nelec_guard_mode: str = 'reject'
    line_search_nelec_trust_initial: float = 2.5e-1
    line_search_nelec_trust_min: float = 1.0e-3
    line_search_nelec_trust_shrink: float = 5.0e-1
    line_search_nelec_trust_expand: float = 2.0
    line_search_nelec_trust_bad_ratio: float = 2.5e-1
    line_search_nelec_trust_good_ratio: float = 7.5e-1
    diis_preserve_accepted_history: bool = False
    lbfgs_use_projected_pairs: bool = False

    # Experimental low-temperature NLCG controls.  These fields are appended
    # to retain positional compatibility with earlier GrandCanonicalConfig
    # construction.  ``trial`` preserves the original post-trial occupation
    # projection; ``direction`` projects one endpoint before line search and
    # then follows the resulting fixed one-dimensional path.
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
    nlcg_nelec_projection_strategy: str = 'trial'
    nlcg_exact_gradient_blend: bool = True
    nlcg_exact_gradient_polish: bool = True
    nlcg_reset_on_preprojection: bool = True
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
    # Optional in-kernel NLCG -> L-BFGS handoff.  The already evaluated state
    # is reused, L-BFGS memory starts empty, and the iteration/Fock counters
    # remain continuous.  A separate line-search choice lets Hager--Zhang
    # NLCG hand off to the occupation-projected strong-Wolfe L-BFGS path.
    lbfgs_switch_residual_rms: Optional[float] = None
    lbfgs_switch_line_search_method: str = 'strong-wolfe'
    # Optional fixed-mu NLCG/Hager--Zhang prefix for automatic canonical
    # continuation.  The prefix stops once the scalar-gauge-free canonical
    # shape and the cheap one-Fock electron-number estimate are both ready.
    canonical_continuation_precondition_residual_rms: Optional[float] = None
    canonical_continuation_precondition_max_delta_nelec: float = 5.0e-2
    canonical_continuation_precondition_min_fock_evaluations: int = 8
    canonical_continuation_precondition_min_iterations: int = 3
    canonical_continuation_precondition_confirmations: int = 1
    canonical_continuation_precondition_max_fock_evaluations: int = 24
    canonical_continuation_precondition_initial_delta_nelec: float = 2.0e-2
    canonical_continuation_handoff_delta_mu: float = 1.0e-6
    canonical_continuation_final_polish: bool = False
    canonical_continuation_verification_residual_tol: float = 1.0e-6
    canonical_continuation_verification_density_tol: float = 1.0e-9
    canonical_continuation_root_nelec_tol: float = 1.0e-8


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
    lbfgs_history_size: int = 0
    lbfgs_pair_added: bool = False
    lbfgs_sy: float = np.nan
    lbfgs_curvature_cosine: float = np.nan
    lbfgs_metric_scale: float = np.nan
    lbfgs_history_action: str = ''
    strong_wolfe: bool = False
    line_search_message: str = ''
    descent_cosine: float = np.nan
    diis_history_size: int = 0
    diis_condition: float = np.nan
    diis_coefficient_l1: float = np.nan
    diis_damping: float = np.nan
    diis_history_action: str = ''
    diis_predicted_residual_rms: float = np.nan
    diis_trust_ratio: float = np.nan
    diis_next_damping: float = np.nan
    nelec_projection_applied: bool = False
    nelec_projection_mode: str = 'reject'
    raw_delta_nelec: float = np.nan
    projected_delta_nelec: float = np.nan
    nelec_projection_parameter: float = 0.0
    nelec_trust_radius: float = np.nan
    nelec_trust_ratio: float = np.nan
    nelec_projection_response_fallback: bool = False
    nelec_projection_correction_rms: float = np.nan
    fock_evaluations: int = 0
    line_search_method: str = 'strong-wolfe'
    weak_wolfe: bool = False
    approximate_wolfe: bool = False
    curvature_qualified: bool = False
    line_search_objective_allowance: float = 0.0
    cheap_nelec_evaluations: int = 0
    cheap_nelec_alpha_reductions: int = 0
    direction_preprojected: bool = False
    direction_projection_mode: str = 'reject'
    direction_raw_endpoint_scale: float = np.nan
    direction_raw_delta_nelec: float = np.nan
    direction_projected_delta_nelec: float = np.nan
    direction_accepted_delta_nelec: float = np.nan
    direction_projection_correction_rms: float = np.nan
    direction_projection_response_fallback: bool = False
    direction_projection_trust_radius: float = np.nan
    direction_projection_trust_ratio: float = np.nan
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
    actual_step: Optional[list] = None
    nelec_projection_applied: bool = False
    nelec_projection_mode: str = 'reject'
    raw_delta_nelec: float = np.nan
    projected_delta_nelec: float = np.nan
    nelec_projection_parameter: float = 0.0
    nelec_trust_radius: float = np.nan
    nelec_trust_ratio: float = np.nan
    nelec_projection_response_fallback: bool = False
    nelec_projection_correction_rms: float = np.nan
    weak_wolfe: bool = False
    approximate_wolfe: bool = False
    curvature_qualified: bool = False
    objective_allowance: float = 0.0
    line_search_method: str = 'strong-wolfe'
    cheap_nelec_evaluations: int = 0
    cheap_nelec_alpha_reductions: int = 0
    direction_preprojected: bool = False
    direction_projection_mode: str = 'reject'
    direction_raw_endpoint_scale: float = np.nan
    direction_raw_delta_nelec: float = np.nan
    direction_projected_delta_nelec: float = np.nan
    direction_accepted_delta_nelec: float = np.nan
    direction_projection_correction_rms: float = np.nan
    direction_projection_response_fallback: bool = False
    direction_projection_trust_radius: float = np.nan
    direction_projection_trust_ratio: float = np.nan
    residual_filter_active: bool = False
    residual_filter_qualified: bool = False
    residual_filter_ratio: float = np.nan
    residual_filter_rejections: int = 0


@dataclass(frozen=True)
class _TrialInfo:
    actual_step: Optional[list] = None
    projected: bool = False
    mode: str = 'reject'
    raw_delta_nelec: float = np.nan
    projected_delta_nelec: float = np.nan
    parameter: float = 0.0
    trust_radius: float = np.nan
    actual_slope: float = np.nan
    response_fallback: bool = False
    rejection_reason: str = ''
    correction_rms: float = np.nan


@dataclass(frozen=True)
class _PreparedDirection:
    direction: list
    success: bool = True
    message: str = ''
    alpha_cap: Optional[float] = None
    preprojected: bool = False
    mode: str = 'reject'
    raw_endpoint_scale: float = np.nan
    raw_delta_nelec: float = np.nan
    projected_delta_nelec: float = np.nan
    correction_rms: float = np.nan
    response_fallback: bool = False
    trust_radius: float = np.nan
    reset_memory: bool = False


@dataclass(frozen=True)
class _HZPoint:
    alpha: float
    state: Optional[_GCState]
    phi: float
    dphi: float
    trial_info: _TrialInfo
    charge_boundary: bool = False
    failed: bool = False


@dataclass
class _LBFGSPair:
    s: list
    y: list
    rho: float
    sy: float
    s_norm: float
    y_norm: float
    curvature_cosine: float


@dataclass
class _DIISItem:
    h: list
    fock: list
    residual: list


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
    nelec_projection_attempts: int = 0
    nelec_projection_acceptances: int = 0
    nelec_projection_fallbacks: int = 0
    max_raw_delta_nelec: float = 0.0
    max_projected_delta_nelec: float = 0.0
    max_nelec_projection_correction: float = 0.0
    final_nelec_trust_radius: float = np.nan
    last_nelec_trust_ratio: float = np.nan
    cheap_nelec_evaluations: int = 0
    cheap_nelec_alpha_reductions: int = 0
    direction_projection_attempts: int = 0
    direction_projection_acceptances: int = 0
    direction_projection_fallbacks: int = 0
    max_direction_projection_correction: float = 0.0
    residual_filter_acceptances: int = 0
    residual_filter_rejections: int = 0
    lbfgs_switches: int = 0
    lbfgs_switch_cycle: int = -1
    lbfgs_switch_nfev: int = -1
    lbfgs_switch_actual_residual_rms: float = np.nan
    canonical_precondition_iterations: int = 0
    canonical_precondition_evaluations: int = 0
    canonical_precondition_residual_rms: float = np.nan
    canonical_precondition_canonical_residual_rms: float = np.nan
    canonical_precondition_delta_nelec: float = np.nan
    canonical_precondition_electron_number: float = np.nan
    canonical_precondition_mu_proxy: float = np.nan
    canonical_precondition_trigger: str = ''
    canonical_continuation_mu_error_source: str = ''
    canonical_verification_attempts: int = 0
    canonical_verification_evaluations: int = 0
    canonical_verification_failures: int = 0
    canonical_verification_residual_rms: float = np.nan
    canonical_verification_grad_rms: float = np.nan
    canonical_verification_delta_nelec: float = np.nan
    canonical_verification_density_rms: float = np.nan
    canonical_terminal_mode: str = ''


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
        self.config.optimizer = self._canonical_optimizer(self.config.optimizer)
        self.config.lbfgs_initial_metric = self._canonical_lbfgs_metric(
            self.config.lbfgs_initial_metric)
        self.config.line_search_nelec_guard_mode = (
            self._canonical_nelec_guard_mode(
                self.config.line_search_nelec_guard_mode))
        self.config.line_search_method = self._canonical_line_search_method(
            self.config.line_search_method)
        self.config.lbfgs_switch_line_search_method = (
            self._canonical_line_search_method(
                self.config.lbfgs_switch_line_search_method))
        self.config.nlcg_nelec_projection_strategy = (
            self._canonical_nlcg_projection_strategy(
                self.config.nlcg_nelec_projection_strategy))
        self._validate_line_search_config()
        self._validate_lbfgs_config()
        self._validate_diis_config()
        self._validate_nelec_guard_config()
        self._validate_canonical_continuation_config()
        if (not self.fixed_electron_number and
                self.config.line_search_method == 'hager-zhang' and
                self.config.line_search_nelec_guard_mode != 'reject'):
            if (self.config.optimizer != 'nlcg' or
                    self.config.nlcg_nelec_projection_strategy !=
                    'direction'):
                raise ValueError(
                    'Hager-Zhang with occupation projection requires the '
                    'NLCG fixed-direction projection strategy; post-trial '
                    'projection is not a one-dimensional line search')
        if self.config.nlcg_residual_filter_rms is not None:
            if (self.config.optimizer != 'nlcg' or
                    self.config.line_search_method != 'hager-zhang' or
                    self.config.nlcg_nelec_projection_strategy !=
                    'direction'):
                raise ValueError(
                    'the NLCG residual filter requires Hager-Zhang and the '
                    'fixed-direction projection strategy')
        if (self.config.lbfgs_switch_residual_rms is not None and
                self.config.optimizer != 'nlcg'):
            raise ValueError(
                'lbfgs_switch_residual_rms requires optimizer="nlcg"')
        if (not self.fixed_electron_number and
                self.config.lbfgs_switch_residual_rms is not None and
                self.config.lbfgs_switch_line_search_method ==
                'hager-zhang' and
                self.config.line_search_nelec_guard_mode != 'reject'):
            raise ValueError(
                'an occupation-projected NLCG-to-L-BFGS handoff requires '
                'the strong-Wolfe L-BFGS line search')
        self.verbose = (getattr(mf, 'verbose', logger.NOTE)
                        if self.config.verbose is None else self.config.verbose)
        self.log = logger.new_logger(mf, self.verbose)
        if ((self.config.optimizer == 'lbfgs' or
             self.config.lbfgs_switch_residual_rms is not None) and
                self.fixed_electron_number and
                self.config.lbfgs_initial_metric == 'fermi'):
            self.log.info(
                'Fixed-electron L-BFGS uses the scalar initial metric; the '
                'Fermi inverse metric is defined only at fixed mu')
            self.config.lbfgs_initial_metric = 'scalar'
        self.history: list[IterationRecord] = []
        self._lbfgs_history: list[_LBFGSPair] = []
        self._diis_history: list[_DIISItem] = []
        self._last_lbfgs_metric_scale = np.nan
        self.nfev = 0
        self.ncheap_nelec_reject = 0
        self._last_trial_rejected_by_nelec = False
        self._last_trial_info = _TrialInfo()
        self._nelec_trust_radius = (
            self.config.line_search_nelec_trust_initial)
        self.nnelec_projection_attempts = 0
        self.nnelec_projection_acceptances = 0
        self.nnelec_projection_fallbacks = 0
        self.max_raw_delta_nelec = 0.0
        self.max_projected_delta_nelec = 0.0
        self.max_nelec_projection_correction = 0.0
        self.last_nelec_trust_ratio = np.nan
        self.ncheap_nelec_evaluations = 0
        self.ncheap_nelec_alpha_reductions = 0
        self.ndirection_projection_attempts = 0
        self.ndirection_projection_acceptances = 0
        self.ndirection_projection_fallbacks = 0
        self.max_direction_projection_correction = 0.0
        self.nresidual_filter_acceptances = 0
        self.nresidual_filter_rejections = 0
        self.nlbfgs_switches = 0
        self.lbfgs_switch_cycle = -1
        self.lbfgs_switch_nfev = -1
        self.lbfgs_switch_actual_residual_rms = np.nan
        self.canonical_precondition_iterations = 0
        self.canonical_precondition_evaluations = 0
        self.canonical_precondition_residual_rms = np.nan
        self.canonical_precondition_canonical_residual_rms = np.nan
        self.canonical_precondition_delta_nelec = np.nan
        self.canonical_precondition_electron_number = np.nan
        self.canonical_precondition_mu_proxy = np.nan
        self.canonical_precondition_trigger = ''
        self.canonical_verification_attempts = 0
        self.canonical_verification_evaluations = 0
        self.canonical_verification_failures = 0
        self.canonical_verification_residual_rms = np.nan
        self.canonical_verification_grad_rms = np.nan
        self.canonical_verification_delta_nelec = np.nan
        self.canonical_verification_density_rms = np.nan
        self.canonical_terminal_mode = ''
        self._canonical_precondition_streak = 0
        self._canonical_precondition_last_nfev = -1
        self._nlcg_residual_previous_alpha: Optional[float] = None
        if _workspace is None:
            self._prepare_fixed_basis_data()
            self._workspace = self._capture_workspace()
        else:
            self._install_workspace(_workspace)
        bytes_per_pair = 2 * sum(
            n * n * int(x.dtype.itemsize)
            for n, x in zip(self.north, self.x_ao2orth))
        self._lbfgs_history_allocation_bytes = (
            self.config.lbfgs_history_size * bytes_per_pair)
        if self.config.optimizer == 'lbfgs':
            self.log.info(
                'L-BFGS history capacity %d pairs; estimated GPU allocation %.3f MiB',
                self.config.lbfgs_history_size,
                self._lbfgs_history_allocation_bytes / 2.0**20)
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
    def _canonical_optimizer(value: str) -> str:
        if not isinstance(value, str):
            raise TypeError('optimizer must be a string')
        key = value.strip().lower().replace('_', '-').replace(' ', '-')
        aliases = {
            'cg': 'nlcg',
            'nlcg': 'nlcg',
            'nonlinear-cg': 'nlcg',
            'nonlinear-conjugate-gradient': 'nlcg',
            'lbfgs': 'lbfgs',
            'l-bfgs': 'lbfgs',
            'limited-memory-bfgs': 'lbfgs',
        }
        try:
            return aliases[key]
        except KeyError as error:
            raise ValueError(
                f'unsupported optimizer {value!r}; choose nlcg or lbfgs') from error

    @staticmethod
    def _canonical_lbfgs_metric(value: str) -> str:
        if not isinstance(value, str):
            raise TypeError('lbfgs_initial_metric must be a string')
        key = value.strip().lower().replace('_', '-').replace(' ', '-')
        aliases = {
            'fermi': 'fermi',
            'fermi-response': 'fermi',
            'scalar': 'scalar',
            'identity': 'scalar',
        }
        try:
            return aliases[key]
        except KeyError as error:
            raise ValueError(
                'lbfgs_initial_metric must be fermi or scalar') from error

    @staticmethod
    def _canonical_nelec_guard_mode(value: str) -> str:
        if not isinstance(value, str):
            raise TypeError('line_search_nelec_guard_mode must be a string')
        key = value.strip().lower().replace('_', '-').replace(' ', '-')
        aliases = {
            'reject': 'reject',
            'scalar': 'scalar-shift',
            'scalar-shift': 'scalar-shift',
            'fermi': 'fermi-response',
            'response': 'fermi-response',
            'fermi-response': 'fermi-response',
        }
        try:
            return aliases[key]
        except KeyError as error:
            raise ValueError(
                'line_search_nelec_guard_mode must be reject, scalar-shift, '
                'or fermi-response') from error

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

    @staticmethod
    def _canonical_nlcg_projection_strategy(value: str) -> str:
        if not isinstance(value, str):
            raise TypeError('nlcg_nelec_projection_strategy must be a string')
        key = value.strip().lower().replace('_', '-').replace(' ', '-')
        aliases = {
            'trial': 'trial',
            'post-trial': 'trial',
            'direction': 'direction',
            'preproject': 'direction',
            'preprojected': 'direction',
            'fixed-direction': 'direction',
        }
        try:
            return aliases[key]
        except KeyError as error:
            raise ValueError(
                'nlcg_nelec_projection_strategy must be trial or direction') from error

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
                     'nlcg_reset_on_preprojection',
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

    def _validate_lbfgs_config(self) -> None:
        switch = self.config.lbfgs_switch_residual_rms
        if switch is not None:
            switch = _as_float(switch, 'lbfgs_switch_residual_rms')
            if switch <= 0.0:
                raise ValueError(
                    'lbfgs_switch_residual_rms must be positive when enabled')
            if switch < self.config.conv_tol_residual_rms:
                raise ValueError(
                    'lbfgs_switch_residual_rms may not be smaller than '
                    'conv_tol_residual_rms')
            self.config.lbfgs_switch_residual_rms = switch
        if (not isinstance(self.config.lbfgs_history_size, int) or
                isinstance(self.config.lbfgs_history_size, bool) or
                self.config.lbfgs_history_size < 0):
            raise ValueError('lbfgs_history_size must be a nonnegative integer')
        if switch is not None and self.config.lbfgs_history_size == 0:
            raise ValueError(
                'lbfgs_history_size must be positive when the NLCG-to-L-BFGS '
                'handoff is enabled')
        positive = (
            'lbfgs_curvature_tol', 'lbfgs_min_pair_step_rms',
            'lbfgs_descent_cosine_min', 'lbfgs_inverse_metric_cap',
            'lbfgs_metric_scale_min', 'lbfgs_metric_scale_max',
            'lbfgs_scalar_h0_min', 'lbfgs_scalar_h0_max',
        )
        for name in positive:
            value = getattr(self.config, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if self.config.lbfgs_metric_scale_min > self.config.lbfgs_metric_scale_max:
            raise ValueError(
                'lbfgs_metric_scale_min may not exceed lbfgs_metric_scale_max')
        if self.config.lbfgs_scalar_h0_min > self.config.lbfgs_scalar_h0_max:
            raise ValueError(
                'lbfgs_scalar_h0_min may not exceed lbfgs_scalar_h0_max')
        c2 = self.config.lbfgs_line_search_c2
        if (not np.isfinite(c2) or not self.config.line_search_c1 < c2 < 1.0):
            raise ValueError(
                'lbfgs_line_search_c2 must be finite and lie between '
                'line_search_c1 and 1')
        for name in ('lbfgs_cap_unit_step_with_history',
                     'lbfgs_clear_on_non_wolfe',
                     'lbfgs_use_projected_pairs'):
            if not isinstance(getattr(self.config, name), bool):
                raise TypeError(f'{name} must be boolean')

    def _validate_diis_config(self) -> None:
        if not isinstance(self.config.diis_preserve_accepted_history, bool):
            raise TypeError('diis_preserve_accepted_history must be boolean')
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
        lbfgs_switch = self.config.lbfgs_switch_residual_rms
        if (switch is not None and lbfgs_switch is not None and
                switch >= lbfgs_switch):
            raise ValueError(
                'diis_switch_residual_rms must be smaller than '
                'lbfgs_switch_residual_rms so the L-BFGS phase is reachable')
        for name, minimum in (
                ('diis_space', 2), ('diis_max_backtracks', 0),
                ('diis_model_max_backtracks', 0),
                ('diis_max_trust_model_repairs', 0)):
            value = getattr(self.config, name)
            if (not isinstance(value, int) or isinstance(value, bool) or
                    value < minimum):
                relation = 'at least 2' if minimum == 2 else 'nonnegative'
                raise ValueError(f'{name} must be an integer that is {relation}')
        if self.config.diis_model_max_backtracks > self.config.diis_max_backtracks:
            raise ValueError(
                'diis_model_max_backtracks may not exceed '
                'diis_max_backtracks')
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
        restoration = self.config.diis_max_restoration_residual_increase
        if not np.isfinite(restoration) or not 0.0 <= restoration < 1.0:
            raise ValueError(
                'diis_max_restoration_residual_increase must lie in [0, 1)')
        interpolation = (
            self.config.diis_trust_interpolation_residual_increase)
        if not np.isfinite(interpolation) or not 0.0 <= interpolation < 1.0:
            raise ValueError(
                'diis_trust_interpolation_residual_increase must lie in '
                '[0, 1)')
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
        for name in ('line_search_nelec_trust_initial',
                     'line_search_nelec_trust_min'):
            value = getattr(self.config, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if (self.config.line_search_nelec_trust_min >
                self.config.line_search_nelec_trust_initial):
            raise ValueError(
                'line_search_nelec_trust_min may not exceed '
                'line_search_nelec_trust_initial')
        shrink = self.config.line_search_nelec_trust_shrink
        expansion = self.config.line_search_nelec_trust_expand
        if not np.isfinite(shrink) or not 0.0 < shrink <= 1.0:
            raise ValueError(
                'line_search_nelec_trust_shrink must lie in (0, 1]')
        if not np.isfinite(expansion) or expansion < 1.0:
            raise ValueError(
                'line_search_nelec_trust_expand must be at least 1')
        bad = self.config.line_search_nelec_trust_bad_ratio
        good = self.config.line_search_nelec_trust_good_ratio
        if (not np.isfinite(bad) or not np.isfinite(good) or
                not 0.0 <= bad < good <= 1.0):
            raise ValueError(
                'electron-number trust ratios must satisfy '
                '0 <= bad < good <= 1')

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
            'canonical_continuation_max_delta_nelec',
            'canonical_continuation_min_delta_nelec',
            'canonical_continuation_initial_damping',
            'canonical_continuation_final_damping',
            'canonical_continuation_diis_max_coefficient_l1',
            'canonical_continuation_interpolation_refine_width',
            'canonical_continuation_verification_residual_tol',
            'canonical_continuation_verification_density_tol',
            'canonical_continuation_root_nelec_tol',
        )
        for name in positive:
            value = getattr(self.config, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
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
        if (self.config.canonical_continuation_initial_delta_nelec >
                self.config.canonical_continuation_max_delta_nelec):
            raise ValueError(
                'canonical_continuation_initial_delta_nelec may not exceed '
                'canonical_continuation_max_delta_nelec')
        if self.config.canonical_continuation_initial_damping > 1.0:
            raise ValueError(
                'canonical_continuation_initial_damping may not exceed 1')
        if self.config.canonical_continuation_final_damping > 1.0:
            raise ValueError(
                'canonical_continuation_final_damping may not exceed 1')
        if not isinstance(
                self.config.canonical_continuation_final_polish, bool):
            raise TypeError(
                'canonical_continuation_final_polish must be boolean')
        switch = (
            self.config.canonical_continuation_precondition_residual_rms)
        if switch is not None:
            switch = _as_float(
                switch,
                'canonical_continuation_precondition_residual_rms')
            if switch <= 0.0:
                raise ValueError(
                    'canonical_continuation_precondition_residual_rms must '
                    'be positive when enabled')
            self.config.canonical_continuation_precondition_residual_rms = (
                switch)
            if not self.config.canonical_continuation:
                raise ValueError(
                    'canonical-continuation preconditioning requires '
                    'canonical_continuation=True')
            if self.fixed_electron_number:
                raise ValueError(
                    'canonical-continuation preconditioning is available '
                    'only at fixed chemical potential')
            if (self.config.optimizer != 'nlcg' or
                    self.config.line_search_method != 'hager-zhang' or
                    self.config.nlcg_nelec_projection_strategy !=
                    'direction'):
                raise ValueError(
                    'canonical-continuation preconditioning requires NLCG, '
                    'Hager-Zhang, and fixed-direction occupation projection')
        for name in (
                'canonical_continuation_precondition_max_delta_nelec',
                'canonical_continuation_precondition_initial_delta_nelec'):
            value = getattr(self.config, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        for name in (
                'canonical_continuation_precondition_min_fock_evaluations',
                'canonical_continuation_precondition_confirmations',
                'canonical_continuation_precondition_max_fock_evaluations'):
            value = getattr(self.config, name)
            if (not isinstance(value, int) or isinstance(value, bool) or
                    value < 1):
                raise ValueError(f'{name} must be a positive integer')
        minimum_iterations = (
            self.config.canonical_continuation_precondition_min_iterations)
        if (not isinstance(minimum_iterations, int) or
                isinstance(minimum_iterations, bool) or
                minimum_iterations < 0):
            raise ValueError(
                'canonical_continuation_precondition_min_iterations must '
                'be a nonnegative integer')
        if (self.config.
                canonical_continuation_precondition_max_fock_evaluations <
                self.config.
                canonical_continuation_precondition_min_fock_evaluations):
            raise ValueError(
                'canonical_continuation_precondition_max_fock_evaluations '
                'may not be smaller than the minimum')
        if (self.config.
                canonical_continuation_precondition_initial_delta_nelec >
                self.config.canonical_continuation_max_delta_nelec):
            raise ValueError(
                'canonical_continuation_precondition_initial_delta_nelec '
                'may not exceed canonical_continuation_max_delta_nelec')
        if (self.config.
                canonical_continuation_precondition_initial_delta_nelec <
                self.config.canonical_continuation_min_delta_nelec):
            raise ValueError(
                'canonical_continuation_precondition_initial_delta_nelec '
                'may not be smaller than '
                'canonical_continuation_min_delta_nelec')

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

    # ---- limited-memory BFGS ---------------------------------------------

    def _apply_fermi_inverse_metric(self, state: _GCState,
                                    blocks: Sequence) -> list:
        """Apply the capped fixed-mu Fermi-response inverse Hessian."""
        if self.fixed_electron_number:
            raise NotImplementedError(
                'the Fermi inverse metric is not defined for fixed electron number')
        output = []
        inverse_floor = 1.0 / self.config.lbfgs_inverse_metric_cap
        for value, vector, gamma, occupation in zip(
                blocks, state.u, state.eigenvalues, state.occupations):
            divided = fermi_divided_difference(
                gamma, occupation, self.config.fermi_divdiff_rtol)
            response = -4.0 * self.beta * divided
            inverse_response = 1.0 / cp.maximum(response, inverse_floor)
            transformed = vector.conj().T @ value @ vector
            result = vector @ (inverse_response * transformed) @ vector.conj().T
            output.append(0.5 * (result + result.conj().T))
        output, _ = self.project_time_reversal(output)
        return self.hermitize_blocks(output)

    def _lbfgs_metric_scale(self, state: _GCState, pair: _LBFGSPair) -> float:
        metric_y = self._apply_fermi_inverse_metric(state, pair.y)
        denominator = self.inner(pair.y, metric_y)
        if not np.isfinite(denominator) or denominator <= 0.0:
            return 1.0
        scale = pair.sy / denominator
        if not np.isfinite(scale) or scale <= 0.0:
            return 1.0
        return float(np.clip(
            scale, self.config.lbfgs_metric_scale_min,
            self.config.lbfgs_metric_scale_max))

    def _lbfgs_scalar_scale(self, pair: _LBFGSPair) -> float:
        denominator = self.inner(pair.y, pair.y)
        if not np.isfinite(denominator) or denominator <= 0.0:
            return self.config.lbfgs_scalar_h0_min
        scale = pair.sy / denominator
        if not np.isfinite(scale) or scale <= 0.0:
            return self.config.lbfgs_scalar_h0_min
        return float(np.clip(
            scale, self.config.lbfgs_scalar_h0_min,
            self.config.lbfgs_scalar_h0_max))

    def _nonfinite_direction(self, state: _GCState) -> list:
        return [cp.full_like(value, cp.nan) for value in state.gradient]

    def _lbfgs_direction(self, state: _GCState,
                         history: Sequence[_LBFGSPair]) -> tuple[list, bool, str]:
        """Construct ``-B g`` using exact-gradient secant pairs."""
        self._last_lbfgs_metric_scale = np.nan
        if not history:
            return (self.copy_blocks(state.residual), False,
                    'empty L-BFGS history')

        q = self.copy_blocks(state.gradient)
        alpha_rev = []
        for pair in reversed(history):
            alpha = pair.rho * self.inner(pair.s, q)
            if not np.isfinite(alpha):
                return (self._nonfinite_direction(state), True,
                        'nonfinite first-loop coefficient')
            alpha = float(alpha)
            q = self.axpy(-alpha, pair.y, q)
            alpha_rev.append(alpha)
        if not self.all_finite(q):
            return (self._nonfinite_direction(state), True,
                    'nonfinite first-loop vector')

        if self.config.lbfgs_initial_metric == 'fermi':
            result = self._apply_fermi_inverse_metric(state, q)
            scale = self._lbfgs_metric_scale(state, history[-1])
        else:
            scale = self._lbfgs_scalar_scale(history[-1])
            result = self.copy_blocks(q)
        if not np.isfinite(scale):
            return (self._nonfinite_direction(state), True,
                    'nonfinite initial-metric scale')
        self._last_lbfgs_metric_scale = float(scale)
        result = self.scale_blocks(float(scale), result)

        for pair, alpha in zip(history, reversed(alpha_rev)):
            beta = pair.rho * self.inner(pair.y, result)
            if not np.isfinite(beta):
                return (self._nonfinite_direction(state), True,
                        'nonfinite second-loop coefficient')
            result = self.axpy(alpha - float(beta), pair.s, result)
        direction = self.scale_blocks(-1.0, result)
        direction = self.hermitize_blocks(direction)
        direction, _ = self.project_time_reversal(direction)
        direction = self.hermitize_blocks(direction)
        return direction, True, ''

    def _ensure_lbfgs_descent(
            self, state: _GCState, direction: Sequence,
            used_history: bool) -> tuple[list, bool, str]:
        """Validate a quasi-Newton direction without NLCG polishing/blending."""
        direction = self.hermitize_blocks(direction)
        direction, _ = self.project_time_reversal(direction)
        direction = self.hermitize_blocks(direction)
        source = 'L-BFGS' if used_history else 'restart'
        if not self.all_finite(direction):
            restart, reason = self._restart_direction(state)
            return restart, True, f'nonfinite {source} direction; {reason}'
        gnorm, dnorm = self.norm(state.gradient), self.norm(direction)
        if gnorm == 0.0 or dnorm == 0.0:
            restart, reason = self._restart_direction(state)
            return restart, True, f'zero {source} direction; {reason}'
        directional = self.inner(state.gradient, direction)
        cosine = -directional / (gnorm * dnorm)
        valid = (
            directional < -self.config.descent_tolerance * gnorm * dnorm
            and cosine >= self.config.lbfgs_descent_cosine_min)
        if valid:
            return direction, False, ''
        restart, reason = self._restart_direction(state)
        return restart, True, f'rejected {source} direction; {reason}'

    def _update_lbfgs_history(
            self, history: list[_LBFGSPair], old_state: _GCState,
            new_state: _GCState, line_search: _LineSearchResult,
            fallback_used: bool = False) -> dict[str, Any]:
        """Validate and commit an exact-gradient secant pair."""
        info = {
            'pair_added': False,
            'sy': np.nan,
            'curvature_cosine': np.nan,
            'action': 'no pair',
        }
        projected_pair = line_search.nelec_projection_applied
        if (projected_pair and
                not self.config.lbfgs_use_projected_pairs):
            history.clear()
            info['action'] = 'history cleared after occupation projection'
            return info
        if projected_pair:
            if line_search.nelec_projection_response_fallback:
                history.clear()
                info['action'] = (
                    'history cleared after response-fallback projection')
                return info
            ratio = line_search.nelec_trust_ratio
            if (not np.isfinite(ratio) or
                    ratio < self.config.line_search_nelec_trust_good_ratio):
                history.clear()
                info['action'] = (
                    'history cleared after unreliable projected model')
                return info
        if not projected_pair:
            non_wolfe = (
                fallback_used or not (
                    line_search.strong_wolfe or
                    line_search.curvature_qualified))
            if ((line_search.force_restart or non_wolfe) and
                    not line_search.trust_boundary):
                if (line_search.force_restart or
                        self.config.lbfgs_clear_on_non_wolfe):
                    history.clear()
                    info['action'] = (
                        'history cleared after non-Wolfe acceptance')
                else:
                    info['action'] = (
                        'pair skipped after non-Wolfe acceptance')
                return info

        if projected_pair:
            if line_search.actual_step is None:
                history.clear()
                info['action'] = (
                    'history cleared: projected pair lacks actual step')
                return info
            s = self.copy_blocks(line_search.actual_step)
            endpoint_step = self.axpy(
                -1.0, old_state.h_orth, new_state.h_orth)
            if (not self.all_finite(s) or
                    not self.all_finite(endpoint_step)):
                history.clear()
                info['action'] = (
                    'history cleared after nonfinite projected step')
                return info
            mismatch = self.max_block_rms(
                self.axpy(-1.0, endpoint_step, s))
            if not np.isfinite(mismatch) or mismatch > 1.0e-8:
                history.clear()
                info['action'] = (
                    'history cleared after inconsistent projected step')
                return info
            # The accepted endpoints are authoritative.  The stored actual
            # step is an invariant check, not a second secant displacement.
            s = endpoint_step
        else:
            s = self.axpy(-1.0, old_state.h_orth, new_state.h_orth)
        y = self.axpy(-1.0, old_state.gradient, new_state.gradient)
        s = self.hermitize_blocks(s)
        y = self.hermitize_blocks(y)
        s, _ = self.project_time_reversal(s)
        y, _ = self.project_time_reversal(y)
        s = self.hermitize_blocks(s)
        y = self.hermitize_blocks(y)
        if not self.all_finite(s) or not self.all_finite(y):
            history.clear()
            info['action'] = 'history cleared after nonfinite curvature'
            return info

        sy = self.inner(s, y)
        s_norm = self.norm(s)
        y_norm = self.norm(y)
        scalars_finite = all(np.isfinite(value) for value in
                             (sy, s_norm, y_norm))
        if not scalars_finite:
            history.clear()
            info['action'] = 'history cleared after nonfinite curvature'
            return info
        info['sy'] = sy
        if s_norm > 0.0 and y_norm > 0.0:
            info['curvature_cosine'] = sy / (s_norm * y_norm)
        if sy <= 0.0:
            if projected_pair:
                info['action'] = 'projected pair skipped: bad curvature'
            else:
                history.clear()
                info['action'] = 'history cleared after bad curvature'
            return info
        if (self.rms(s) < self.config.lbfgs_min_pair_step_rms or
                s_norm == 0.0 or y_norm == 0.0 or
                sy < self.config.lbfgs_curvature_tol * s_norm * y_norm):
            prefix = 'projected ' if projected_pair else ''
            info['action'] = f'{prefix}pair skipped: weak curvature'
            return info
        if self.config.lbfgs_history_size == 0:
            prefix = 'projected ' if projected_pair else ''
            info['action'] = (
                f'{prefix}pair skipped: history capacity is zero')
            return info

        rho = 1.0 / sy
        if not np.isfinite(rho):
            history.clear()
            info['action'] = 'history cleared after nonfinite curvature'
            return info
        pair = _LBFGSPair(
            self.copy_blocks(s), self.copy_blocks(y), float(rho),
            float(sy), float(s_norm), float(y_norm),
            float(info['curvature_cosine']))
        history.append(pair)
        prefix = 'projected ' if projected_pair else ''
        if len(history) > self.config.lbfgs_history_size:
            del history[0]
            info['action'] = f'{prefix}pair added; oldest pair evicted'
        else:
            info['action'] = f'{prefix}pair added'
        info['pair_added'] = True
        return info

    # ---- residual DIIS ---------------------------------------------------

    def _should_start_diis(self, state: _GCState) -> bool:
        threshold = self.config.diis_switch_residual_rms
        return threshold is not None and state.residual_rms <= threshold

    def _should_start_lbfgs(self, state: _GCState) -> bool:
        threshold = self.config.lbfgs_switch_residual_rms
        return threshold is not None and state.residual_rms <= threshold

    def _canonical_precondition_enabled(self) -> bool:
        return (
            self.config.canonical_continuation and
            not self.fixed_electron_number and
            self.config.
            canonical_continuation_precondition_residual_rms is not None)

    def _canonical_precondition_metrics(
            self, state: _GCState) -> tuple[float, float, float, float]:
        """Return canonical-shape RMS, charge defect, gauge, and mu proxy."""
        mismatch = self.hermitize_blocks([
            h - f for h, f in zip(state.h_orth, state.fock_orth)])
        gauge_shift = self.trace_mean(mismatch)
        canonical_mismatch = [
            value - gauge_shift * identity
            for value, identity in zip(mismatch, self.identity)]
        canonical_residual_rms = self.rms(canonical_mismatch)
        self.ncheap_nelec_evaluations += 1
        delta_nelec = (
            self._electron_number_at_mu(state.fock_orth, self.mu) -
            state.electron_number)
        return (canonical_residual_rms, delta_nelec, gauge_shift,
                self.mu - gauge_shift)

    def _should_start_canonical_continuation(
            self, state: _GCState, niter: int) -> bool:
        if not self._canonical_precondition_enabled():
            return False
        if self._canonical_precondition_last_nfev == self.nfev:
            return bool(self.canonical_precondition_trigger)
        (canonical_residual_rms, delta_nelec, gauge_shift,
         mu_proxy) = self._canonical_precondition_metrics(state)
        self.canonical_precondition_residual_rms = state.residual_rms
        self.canonical_precondition_canonical_residual_rms = (
            canonical_residual_rms)
        self.canonical_precondition_delta_nelec = delta_nelec
        self.canonical_precondition_electron_number = state.electron_number
        self.canonical_precondition_mu_proxy = mu_proxy
        self._canonical_precondition_last_nfev = self.nfev

        warmed_up = (
            self.nfev >= self.config.
            canonical_continuation_precondition_min_fock_evaluations and
            niter >= self.config.
            canonical_continuation_precondition_min_iterations)
        eligible = (
            warmed_up and
            canonical_residual_rms <= self.config.
            canonical_continuation_precondition_residual_rms and
            abs(delta_nelec) <= self.config.
            canonical_continuation_precondition_max_delta_nelec)
        if eligible:
            self._canonical_precondition_streak += 1
        else:
            self._canonical_precondition_streak = 0
        if (self._canonical_precondition_streak >= self.config.
                canonical_continuation_precondition_confirmations):
            self.canonical_precondition_trigger = 'criteria-confirmed'
        elif (self.nfev >= self.config.
                canonical_continuation_precondition_max_fock_evaluations):
            self.canonical_precondition_trigger = 'max-fock-budget'

        self.log.info(
            'Canonical precondition probe: Fock %d, accepted %d, full '
            'residual %.6g, canonical residual %.6g, gauge %.6g, mu proxy '
            '%.12g, delta N_FP %.6g, streak %d, trigger %s',
            self.nfev, niter, state.residual_rms,
            canonical_residual_rms, gauge_shift, mu_proxy, delta_nelec,
            self._canonical_precondition_streak,
            self.canonical_precondition_trigger or 'none')
        return bool(self.canonical_precondition_trigger)

    def _start_canonical_continuation_from_prefix(
            self, state: _GCState, niter: int, *,
            trigger: Optional[str] = None) -> GrandCanonicalResult:
        """Hand an evaluated fixed-mu state to fixed-N continuation."""
        if self._canonical_precondition_last_nfev != self.nfev:
            (canonical_residual_rms, delta_nelec, _,
             mu_proxy) = self._canonical_precondition_metrics(state)
            self.canonical_precondition_residual_rms = state.residual_rms
            self.canonical_precondition_canonical_residual_rms = (
                canonical_residual_rms)
            self.canonical_precondition_delta_nelec = delta_nelec
            self.canonical_precondition_electron_number = (
                state.electron_number)
            self.canonical_precondition_mu_proxy = mu_proxy
            self._canonical_precondition_last_nfev = self.nfev
        if trigger is not None:
            self.canonical_precondition_trigger = trigger
        self.canonical_precondition_iterations = niter
        self.canonical_precondition_evaluations = self.nfev
        self.log.info(
            'Fixed-mu electron-number estimate ready after %d accepted '
            'steps and %d Fock evaluations (N = %.12g, trigger = %s); '
            'starting canonical continuation',
            niter, self.nfev, state.electron_number,
            self.canonical_precondition_trigger)
        return self._kernel_canonical_continuation(
            seed_state=state, prefix_history=list(self.history),
            prefix_niter=niter, prefix_nfev=self.nfev)

    def _append_diis_item(self, history: list[_DIISItem],
                          state: _GCState) -> None:
        history.append(_DIISItem(
            self.copy_blocks(state.h_orth),
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
            self, state: _GCState, trial: _GCState,
            residual_target_rms: Optional[float] = None,
            allow_restoration: bool = False,
            best_residual_rms: Optional[float] = None) -> tuple[bool, str]:
        residual_limit = state.residual_rms * (
            1.0 - self.config.diis_min_residual_reduction)
        if residual_target_rms is not None:
            residual_limit = min(residual_limit, residual_target_rms)
        monotone = trial.residual_rms < residual_limit
        restoration = False
        if (not monotone and allow_restoration and
                self.config.diis_max_restoration_residual_increase > 0.0 and
                best_residual_rms is not None):
            restoration_limit = best_residual_rms * (
                1.0 +
                self.config.diis_max_restoration_residual_increase)
            restoration = (
                trial.residual_rms <= restoration_limit and
                trial.objective < state.objective)
        if not monotone and not restoration:
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
                         residual_target_rms: Optional[float] = None,
                         allow_restoration: bool = False,
                         best_residual_rms: Optional[float] = None,
                         ) -> tuple[Optional[_GCState], float, str,
                                    Optional[_GCState]]:
        direction = self.axpy(-1.0, state.h_orth, target)
        if not self.all_finite(direction) or self.norm(direction) == 0.0:
            return None, 0.0, 'zero or nonfinite DIIS direction', None
        damping = min(1.0, max(
            self.config.line_search_alpha_min, starting_damping))
        if max_backtracks is None:
            max_backtracks = self.config.diis_max_backtracks
        last_reason = 'no DIIS trial evaluated'
        best_rejected = None
        rejected_samples: list[tuple[float, float]] = []
        for _ in range(max_backtracks + 1):
            trial = self._trial(
                state, direction, damping, allow_nelec_projection=False)
            if trial is not None:
                acceptable, last_reason = self._diis_trial_acceptable(
                    state, trial, residual_target_rms=residual_target_rms,
                    allow_restoration=allow_restoration,
                    best_residual_rms=best_residual_rms)
                self.log.debug(
                    'DIIS trust trial: damping = %.6g, residual %.6g -> '
                    '%.6g, delta objective = %.3g, delta N = %.3g: %s',
                    damping, state.residual_rms, trial.residual_rms,
                    trial.objective - state.objective,
                    trial.electron_number - state.electron_number,
                    'accepted' if acceptable else last_reason)
                if acceptable:
                    return trial, damping, '', best_rejected
                interpolation_limit = state.residual_rms * (
                    1.0 +
                    self.config.diis_trust_interpolation_residual_increase)
                # Damping decreases monotonically, so retain the first
                # rejected point inside the interpolation envelope.  It is
                # the most widely separated local secant and avoids filling
                # the model with nearly duplicate tiny-step trials.
                if (best_rejected is None and
                        trial.residual_rms <= interpolation_limit):
                    best_rejected = trial
                rejected_samples.append((damping, trial.residual_rms))
                if len(rejected_samples) >= 2:
                    (alpha0, residual0), (alpha1, residual1) = (
                        rejected_samples[-2:])
                    slope = ((residual1 - residual0) /
                             (alpha1 - alpha0))
                    intercept = residual1 - slope * alpha1
                    residual_limit = state.residual_rms * (
                        1.0 - self.config.diis_min_residual_reduction)
                    if residual_target_rms is not None:
                        residual_limit = min(
                            residual_limit, residual_target_rms)
                    # If the two smallest sampled trust radii extrapolate to
                    # an unacceptable residual even at zero radius, another
                    # expensive Fock trial cannot repair this DIIS model.
                    # Return the nearby rejected state so the outer model can
                    # augment or prune its history immediately.
                    local_samples = (
                        max(residual0, residual1) <= interpolation_limit)
                    if (local_samples and np.isfinite(slope) and slope > 0.0 and
                            np.isfinite(intercept) and
                            intercept >= residual_limit):
                        last_reason = (
                            'backtracking secant predicts no acceptable '
                            'residual for this DIIS model')
                        break
            else:
                last_reason = 'DIIS trial evaluation failed'
            damping *= self.config.diis_backtrack_factor
        return None, 0.0, last_reason, best_rejected

    def _diis_step(
            self, state: _GCState,
            history: list[_DIISItem], starting_damping: float = 1.0,
            residual_target_rms: Optional[float] = None,
            allow_restoration: bool = False,
            best_residual_rms: Optional[float] = None,
            ) -> tuple[_LineSearchResult, float, float, str, float]:
        start_nfev = self.nfev
        action_parts = []
        model_repairs = 0
        while True:
            accepted_history = (
                list(history)
                if self.config.diis_preserve_accepted_history else None)
            coefficients, condition, coefficient_l1, coefficient_action = (
                self._diis_coefficients(history))
            if coefficient_action:
                action_parts.append(coefficient_action)
            target = self._diis_target(history, coefficients)
            max_backtracks = (
                self.config.diis_model_max_backtracks
                if len(history) > 1 else self.config.diis_max_backtracks)
            trial, damping, rejection, rejected_state = self._try_diis_target(
                state, target, starting_damping, max_backtracks,
                residual_target_rms=residual_target_rms,
                allow_restoration=allow_restoration,
                best_residual_rms=best_residual_rms)
            if accepted_history is not None:
                model_history_size = len(history)
                history[:] = accepted_history
                if trial is not None:
                    if len(accepted_history) != model_history_size:
                        action_parts.append(
                            'restored accepted DIIS history after temporary '
                            'coefficient pruning')
                    break
                if model_history_size > 1:
                    fallback_target = self.copy_blocks(history[-1].fock)
                    (trial, damping, fallback_rejection,
                     _) = self._try_diis_target(
                         state, fallback_target, starting_damping,
                         self.config.diis_max_backtracks,
                         residual_target_rms=residual_target_rms,
                         allow_restoration=allow_restoration,
                         best_residual_rms=best_residual_rms)
                    action_parts.append(
                        'preserved accepted DIIS history; tried latest-Fock '
                        'fallback after rejected model')
                    if trial is not None:
                        break
                    rejection = fallback_rejection
                break
            if trial is not None:
                break
            if (rejected_state is not None and
                    model_repairs <
                    self.config.diis_max_trust_model_repairs):
                self._append_diis_item(history, rejected_state)
                model_repairs += 1
                action_parts.append(
                    'augmented DIIS model with rejected trust trial')
                continue
            if len(history) == 1:
                break
            del history[0]
            action_parts.append(
                'dropped oldest DIIS vector after rejected model')
        action = '; '.join(dict.fromkeys(action_parts))
        nfev = self.nfev - start_nfev
        if trial is None:
            message = 'residual-DIIS failed: ' + rejection
            return (_LineSearchResult(False, None, nfev=nfev,
                                      message=message),
                    condition, coefficient_l1, action, np.nan)
        predicted_residual_rms = self._diis_predicted_residual_rms(
            state, damping)
        message = 'residual-DIIS accepted'
        if damping < 1.0:
            message += f' with damping {damping:.6g}'
        return (_LineSearchResult(True, trial, damping, nfev,
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

    def _active_nelec_limit(self, state: _GCState,
                            use_adaptive_radius: bool) -> float:
        threshold = self.config.line_search_nelec_guard_residual_rms
        maximum = self.config.line_search_max_delta_nelec
        if threshold is not None and state.residual_rms <= threshold:
            maximum = min(
                maximum,
                self.config.line_search_nelec_guard_max_delta_nelec)
        if use_adaptive_radius:
            maximum = min(maximum, self._nelec_trust_radius)
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
            maximum = self._active_nelec_limit(
                state, use_adaptive_radius=False)
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

    def _prepare_nlcg_direction(
            self, state: _GCState,
            direction: Sequence) -> _PreparedDirection:
        """Optionally occupation-limit one endpoint before line search.

        Projecting the endpoint once and searching along ``P(H+d)-H`` keeps
        every accepted trial on a single line.  This makes Wolfe derivative
        conditions meaningful, unlike applying a different nonlinear
        occupation correction after each trial.
        """
        direction = self.hermitize_blocks(direction)
        if (self.fixed_electron_number or
                self.config.nlcg_nelec_projection_strategy != 'direction' or
                self.config.line_search_nelec_guard_mode == 'reject'):
            return _PreparedDirection(direction=self.copy_blocks(direction))

        endpoint_scale = min(1.0, self._alpha_cap(direction))
        if endpoint_scale < self.config.line_search_alpha_min:
            return _PreparedDirection(
                direction=self.copy_blocks(direction), success=False,
                message='direction endpoint lies below the minimum step')

        maximum = self._active_nelec_limit(
            state, use_adaptive_radius=True)
        for _ in range(8):
            raw_candidate = self._sanitize_h(
                self.axpy(endpoint_scale, direction, state.h_orth))
            raw_nelec = self._cheap_fixed_mu_electron_number(raw_candidate)
            self.ncheap_nelec_evaluations += 1
            raw_delta = raw_nelec - state.electron_number
            self.max_raw_delta_nelec = max(
                self.max_raw_delta_nelec, abs(raw_delta))
            if abs(raw_delta) <= maximum + 1.0e-10:
                return _PreparedDirection(
                    direction=self.copy_blocks(direction),
                    alpha_cap=endpoint_scale,
                    raw_endpoint_scale=endpoint_scale,
                    raw_delta_nelec=raw_delta,
                    projected_delta_nelec=raw_delta,
                    trust_radius=maximum)

            target = (state.electron_number +
                      np.copysign(maximum, raw_delta))
            self.ndirection_projection_attempts += 1
            try:
                projected, _, response_fallback = (
                    self._project_trial_electron_number(
                        raw_candidate, target))
            except (ArithmeticError, FloatingPointError, ValueError,
                    RuntimeError, cp.linalg.LinAlgError):
                endpoint_scale *= self.config.armijo_backtrack_factor
                continue
            if response_fallback:
                self.ndirection_projection_fallbacks += 1
            projected_nelec = self._cheap_fixed_mu_electron_number(projected)
            self.ncheap_nelec_evaluations += 1
            projected_delta = projected_nelec - state.electron_number
            projected_direction = self.hermitize_blocks(
                self.axpy(-1.0, state.h_orth, projected))
            correction = self.max_block_rms(
                self.axpy(-1.0, raw_candidate, projected))
            self.max_projected_delta_nelec = max(
                self.max_projected_delta_nelec, abs(projected_delta))
            self.max_direction_projection_correction = max(
                self.max_direction_projection_correction, correction)
            if (abs(projected_nelec - target) <= 1.0e-10 and
                    self.max_block_rms(projected_direction) <=
                    self.config.line_search_max_h_rms_step and
                    self._is_descent(state, projected_direction)):
                return _PreparedDirection(
                    direction=projected_direction, alpha_cap=1.0,
                    preprojected=True,
                    mode=self.config.line_search_nelec_guard_mode,
                    raw_endpoint_scale=endpoint_scale,
                    raw_delta_nelec=raw_delta,
                    projected_delta_nelec=projected_delta,
                    correction_rms=correction,
                    response_fallback=response_fallback,
                    trust_radius=maximum,
                    reset_memory=(
                        self.config.nlcg_reset_on_preprojection))
            endpoint_scale *= self.config.armijo_backtrack_factor
            if endpoint_scale < self.config.line_search_alpha_min:
                break
        return _PreparedDirection(
            direction=self.copy_blocks(direction), success=False,
            message='failed to construct a downhill charge-limited direction',
            raw_endpoint_scale=endpoint_scale, trust_radius=maximum)

    def _charge_capped_direction_fallback(
            self, state: _GCState, direction: Sequence,
            message: str) -> _PreparedDirection:
        """Use an unmodified line with the same frozen cheap charge radius."""
        direction = self.hermitize_blocks(direction)
        valid = (self._is_descent(state, direction) and
                 self._alpha_cap(direction) >=
                 self.config.line_search_alpha_min)
        return _PreparedDirection(
            direction=self.copy_blocks(direction), success=valid,
            message=message, mode=self.config.line_search_nelec_guard_mode,
            trust_radius=self._active_nelec_limit(
                state, use_adaptive_radius=True),
            reset_memory=True)

    def _decorate_direction_projection_result(
            self, old: _GCState, new: _GCState,
            result: _LineSearchResult,
            prepared: _PreparedDirection) -> _LineSearchResult:
        """Attach fixed-direction diagnostics and update its charge radius."""
        accepted_delta = new.electron_number - old.electron_number
        ratio = np.nan
        if prepared.preprojected:
            predicted = -result.alpha * self.inner(
                old.gradient, prepared.direction)
            actual = old.objective - new.objective
            ratio = actual / predicted if predicted > 0.0 else -np.inf
            if ratio < self.config.line_search_nelec_trust_bad_ratio:
                self._shrink_nelec_trust_radius(prepared.trust_radius)
            elif (ratio >= self.config.line_search_nelec_trust_good_ratio and
                  not result.residual_filter_qualified and
                  abs(accepted_delta) >= 0.8 * prepared.trust_radius and
                  self.config.line_search_nelec_trust_expand > 1.0):
                self._nelec_trust_radius = min(
                    self.config.line_search_max_delta_nelec,
                    self.config.line_search_nelec_trust_expand *
                    min(self._nelec_trust_radius,
                        prepared.trust_radius))
            self.ndirection_projection_acceptances += 1
            self.last_nelec_trust_ratio = ratio
        return replace(
            result,
            direction_preprojected=prepared.preprojected,
            direction_projection_mode=prepared.mode,
            direction_raw_endpoint_scale=prepared.raw_endpoint_scale,
            direction_raw_delta_nelec=prepared.raw_delta_nelec,
            direction_projected_delta_nelec=(
                prepared.projected_delta_nelec),
            direction_accepted_delta_nelec=accepted_delta,
            direction_projection_correction_rms=prepared.correction_rms,
            direction_projection_response_fallback=(
                prepared.response_fallback),
            direction_projection_trust_radius=prepared.trust_radius,
            direction_projection_trust_ratio=ratio)

    def _spectral_charge_root(
            self, eigenvalues: Sequence, response: Sequence,
            target_nelec: float) -> float:
        """Return signed lambda such that N(e + lambda*r) hits target."""
        current = self._electron_number_from_eigenvalues(
            eigenvalues, self.mu)
        error = current - target_nelec
        if abs(error) <= 1.0e-10:
            return 0.0
        sign = 1.0 if error > 0.0 else -1.0

        def shifted_nelec(magnitude: float) -> float:
            return self._electron_number_from_eigenvalues(
                [value + sign * magnitude * scale
                 for value, scale in zip(eigenvalues, response)],
                self.mu)

        lower = 0.0
        upper = max(self.sigma, 1.0e-8)
        for _ in range(100):
            shifted_error = shifted_nelec(upper) - target_nelec
            if sign * shifted_error <= 0.0:
                break
            upper *= 2.0
        else:
            raise RuntimeError(
                'failed to bracket electron-number projection root')
        for _ in range(100):
            midpoint = 0.5 * (lower + upper)
            shifted_error = shifted_nelec(midpoint) - target_nelec
            if abs(shifted_error) <= 1.0e-10:
                lower = upper = midpoint
                break
            if sign * shifted_error > 0.0:
                lower = midpoint
            else:
                upper = midpoint
        return sign * 0.5 * (lower + upper)

    def _scalar_shift_to_nelec(
            self, candidate: Sequence, target_nelec: float,
            eigenvalues: Optional[Sequence] = None) -> tuple[list, float]:
        if eigenvalues is None:
            eigenvalues = [cp.linalg.eigvalsh(hk) for hk in candidate]
        parameter = self._spectral_charge_root(
            eigenvalues, [cp.ones_like(value) for value in eigenvalues],
            target_nelec)
        projected = [hk + parameter * identity
                     for hk, identity in zip(candidate, self.identity)]
        return self._sanitize_h(projected), parameter

    def _project_trial_electron_number(
            self, candidate: Sequence,
            target_nelec: float) -> tuple[list, float, bool]:
        mode = self.config.line_search_nelec_guard_mode
        if mode == 'scalar-shift':
            projected, parameter = self._scalar_shift_to_nelec(
                candidate, target_nelec)
            return projected, parameter, False
        if mode != 'fermi-response':  # pragma: no cover - validated
            raise AssertionError('electron-number projection mode unreachable')

        eigenpairs = [cp.linalg.eigh(hk) for hk in candidate]
        eigenvalues = [pair[0] for pair in eigenpairs]
        occupations = [fermi_occupations(
            self.beta * (value - self.mu)) for value in eigenvalues]
        response = [q * (1.0 - q) for q in occupations]
        response_max = max(
            float(cp.max(value).item()) for value in response)
        if not np.isfinite(response_max) or response_max < 1.0e-14:
            projected, parameter = self._scalar_shift_to_nelec(
                candidate, target_nelec, eigenvalues=eigenvalues)
            return projected, parameter, True
        response = [value / response_max for value in response]
        # Avoid a long futile bracketing loop when only a small frontier
        # subspace responds.  Levels with exactly zero f(1-f) remain fixed for
        # every response parameter, which gives exact asymptotic charge bounds.
        response_min_nelec = 0.0
        response_max_nelec = 0.0
        for k, (q, scale) in enumerate(zip(occupations, response)):
            responsive = scale > 0.0
            fixed_occupation = float(
                cp.sum(cp.where(responsive, 0.0, q)).item())
            weight = float(self.weights[k].item())
            response_min_nelec += 2.0 * weight * fixed_occupation
            response_max_nelec += 2.0 * weight * (
                fixed_occupation +
                float(cp.count_nonzero(responsive).item()))
        if (target_nelec < response_min_nelec - 1.0e-10 or
                target_nelec > response_max_nelec + 1.0e-10):
            projected, parameter = self._scalar_shift_to_nelec(
                candidate, target_nelec, eigenvalues=eigenvalues)
            return projected, parameter, True
        try:
            parameter = self._spectral_charge_root(
                eigenvalues, response, target_nelec)
        except RuntimeError:
            # At very low temperature, only a small frontier subspace may
            # have nonzero f(1-f).  It can then lack enough capacity to reach
            # the requested charge boundary even though its maximum response
            # is numerically resolvable.  A common spectral shift is always
            # monotone over the complete retained spectrum.
            projected, parameter = self._scalar_shift_to_nelec(
                candidate, target_nelec, eigenvalues=eigenvalues)
            return projected, parameter, True
        projected = []
        for hk, (_, vector), scale in zip(candidate, eigenpairs, response):
            correction = ((vector * (parameter * scale)[None, :]) @
                          vector.conj().T)
            projected.append(hk + correction)
        projected = self._sanitize_h(projected)
        # Symmetry projection can change the root at roundoff level.  Finish
        # with a common spectral shift so that the trust boundary is exact.
        if abs(self._electron_number_at_mu(projected, self.mu) -
               target_nelec) > 1.0e-10:
            projected, _ = self._scalar_shift_to_nelec(
                projected, target_nelec)
        # A nearly singular frozen Fermi response can formally reach the
        # charge target only through an enormous localized spectral change.
        # Such a correction cannot produce an admissible Hamiltonian step and
        # previously reached ~1e21 RMS in rejected slab trials.  Use the
        # globally monotone scalar projection before the pathological response
        # reaches the expensive evaluator.
        correction_rms = self.max_block_rms(
            self.axpy(-1.0, candidate, projected))
        if (not np.isfinite(correction_rms) or
                correction_rms > self.config.line_search_max_h_rms_step):
            projected, parameter = self._scalar_shift_to_nelec(
                candidate, target_nelec, eigenvalues=eigenvalues)
            return projected, parameter, True
        return projected, parameter, False

    def _shrink_nelec_trust_radius(
            self, active_radius: Optional[float] = None) -> None:
        if self.config.line_search_nelec_trust_shrink == 1.0:
            return
        base = self._nelec_trust_radius
        if active_radius is not None and np.isfinite(active_radius):
            base = min(base, active_radius)
        self._nelec_trust_radius = max(
            self.config.line_search_nelec_trust_min,
            self.config.line_search_nelec_trust_shrink *
            base)

    def _accept_projected_trial(
            self, old_state: _GCState, new_state: _GCState,
            info: _TrialInfo) -> float:
        predicted = -info.actual_slope
        actual = old_state.objective - new_state.objective
        ratio = actual / predicted if predicted > 0.0 else -np.inf
        if ratio < self.config.line_search_nelec_trust_bad_ratio:
            self._shrink_nelec_trust_radius(info.trust_radius)
        elif (ratio >= self.config.line_search_nelec_trust_good_ratio and
              self.config.line_search_nelec_trust_expand > 1.0):
            base = self._nelec_trust_radius
            if np.isfinite(info.trust_radius):
                base = min(base, info.trust_radius)
            self._nelec_trust_radius = min(
                self.config.line_search_max_delta_nelec,
                self.config.line_search_nelec_trust_expand *
                base)
        self.nnelec_projection_acceptances += 1
        self.last_nelec_trust_ratio = ratio
        return ratio

    def _projected_line_search_result(
            self, old_state: _GCState, new_state: _GCState,
            alpha: float, nfev: int, message: str,
            info: Optional[_TrialInfo] = None) -> _LineSearchResult:
        info = self._last_trial_info if info is None else info
        ratio = self._accept_projected_trial(old_state, new_state, info)
        return _LineSearchResult(
            True, new_state, alpha, nfev, False, True, message, True,
            self.copy_blocks(info.actual_step), True, info.mode,
            info.raw_delta_nelec, info.projected_delta_nelec,
            info.parameter, info.trust_radius, ratio,
            info.response_fallback, info.correction_rms)

    def _trial(self, state: _GCState, direction: Sequence, alpha: float,
               allow_nelec_projection: bool = True,
               nelec_limit_override: Optional[float] = None
               ) -> Optional[_GCState]:
        self._last_trial_rejected_by_nelec = False
        self._last_trial_info = _TrialInfo()
        try:
            raw_candidate = self._sanitize_h(
                self.axpy(alpha, direction, state.h_orth))
            actual_step = self.hermitize_blocks(
                self.axpy(-1.0, state.h_orth, raw_candidate))
            if self.fixed_electron_number:
                self._last_trial_info = _TrialInfo(
                    actual_step=self.copy_blocks(actual_step),
                    actual_slope=self.inner(state.gradient, actual_step))
                return self.evaluate(raw_candidate)

            raw_nelec = self._cheap_fixed_mu_electron_number(raw_candidate)
            self.ncheap_nelec_evaluations += 1
            raw_delta = raw_nelec - state.electron_number
            self.max_raw_delta_nelec = max(
                self.max_raw_delta_nelec, abs(raw_delta))
            projection_enabled = (
                allow_nelec_projection and
                self.config.line_search_nelec_guard_mode != 'reject')
            maximum = (self._active_nelec_limit(
                state, use_adaptive_radius=projection_enabled)
                if nelec_limit_override is None else
                float(nelec_limit_override))
            if (abs(raw_delta) > maximum + 1.0e-10 and
                    projection_enabled):
                target = (state.electron_number +
                          np.copysign(maximum, raw_delta))
                candidate, parameter, response_fallback = (
                    self._project_trial_electron_number(
                        raw_candidate, target))
                projected_nelec = self._cheap_fixed_mu_electron_number(
                    candidate)
                self.ncheap_nelec_evaluations += 1
                actual_step = self.hermitize_blocks(
                    self.axpy(-1.0, state.h_orth, candidate))
                actual_slope = self.inner(state.gradient, actual_step)
                correction = self.max_block_rms(self.axpy(
                    -1.0, raw_candidate, candidate))
                self.nnelec_projection_attempts += 1
                if response_fallback:
                    self.nnelec_projection_fallbacks += 1
                self.max_nelec_projection_correction = max(
                    self.max_nelec_projection_correction, correction)
                projected_delta = (
                    projected_nelec - state.electron_number)
                if abs(projected_nelec - target) > 1.0e-10:
                    raise RuntimeError(
                        'electron-number projection missed its trust boundary')
                self.max_projected_delta_nelec = max(
                    self.max_projected_delta_nelec,
                    abs(projected_delta))
                self._last_trial_info = _TrialInfo(
                    actual_step=self.copy_blocks(actual_step),
                    projected=True,
                    mode=self.config.line_search_nelec_guard_mode,
                    raw_delta_nelec=raw_delta,
                    projected_delta_nelec=projected_delta,
                    parameter=parameter,
                    trust_radius=maximum,
                    actual_slope=actual_slope,
                    response_fallback=response_fallback,
                    correction_rms=correction)
                if not self._is_descent(state, actual_step):
                    self._last_trial_info = replace(
                        self._last_trial_info,
                        rejection_reason='projected step is not downhill')
                    self._shrink_nelec_trust_radius(maximum)
                    return None
                if (self.max_block_rms(actual_step) >
                        self.config.line_search_max_h_rms_step):
                    self._last_trial_info = replace(
                        self._last_trial_info,
                        rejection_reason='projected step exceeds H trust cap')
                    return None
                return self.evaluate(candidate)

            if abs(raw_delta) > maximum + 1.0e-10:
                self._last_trial_rejected_by_nelec = True
                self.ncheap_nelec_reject += 1
                self._last_trial_info = _TrialInfo(
                    actual_step=self.copy_blocks(actual_step),
                    mode='reject', raw_delta_nelec=raw_delta,
                    projected_delta_nelec=raw_delta,
                    trust_radius=maximum,
                    actual_slope=self.inner(state.gradient, actual_step),
                    rejection_reason='electron-number limit exceeded')
                self.log.debug(
                    'Rejected trial before Fock build: alpha = %.6g, '
                    'residual RMS = %.6g, N = %.12g -> %.12g',
                    alpha, state.residual_rms, state.electron_number,
                    raw_nelec)
                return None
            self._last_trial_info = _TrialInfo(
                self.copy_blocks(actual_step), False,
                self.config.line_search_nelec_guard_mode,
                raw_delta, raw_delta, 0.0, maximum,
                self.inner(state.gradient, actual_step))
            return self.evaluate(raw_candidate)
        except (ArithmeticError, FloatingPointError, ValueError, RuntimeError, cp.linalg.LinAlgError):
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
            reduction_start: Optional[int] = None,
            allow_nelec_projection: bool = True,
            nelec_limit_override: Optional[float] = None
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
            trial = self._trial(
                state0, direction, alpha,
                allow_nelec_projection=allow_nelec_projection,
                nelec_limit_override=nelec_limit_override)
            trial_count += 1
            trial_info = self._last_trial_info
            if trial_info.projected:
                if (trial is not None and
                        trial.objective <= phi0 +
                        c1 * trial_info.actual_slope):
                    return finish(self._projected_line_search_result(
                        state0, trial, alpha, 0,
                        'accepted occupation-projected Armijo zoom point',
                        info=trial_info))
                if trial is not None:
                    self._shrink_nelec_trust_radius(
                        trial_info.trust_radius)
                hi_a, hi_state, hi_phi, hi_dphi = (
                    alpha, None, np.inf, np.nan)
                continue
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
            alpha_cap_override: Optional[float] = None,
            nelec_limit_override: Optional[float] = None,
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
        if alpha_cap_override is not None:
            if (not np.isfinite(alpha_cap_override) or
                    alpha_cap_override <= 0.0):
                raise ValueError(
                    'alpha_cap_override must be finite and positive')
            alpha_max = min(alpha_max, alpha_cap_override)
        maximum = (self._active_nelec_limit(
            state, use_adaptive_radius=False)
            if nelec_limit_override is None else
            float(nelec_limit_override))
        alpha_max, _ = self._charge_feasible_alpha_cap(
            state, direction, alpha_max, maximum=maximum)
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
            0.0: _HZPoint(
                0.0, state, phi0, dphi0, _TrialInfo(), False, False)}
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
            trial = self._trial(
                state, direction, alpha, allow_nelec_projection=False,
                nelec_limit_override=maximum)
            trial_count += 1
            info = self._last_trial_info
            charge_boundary = (
                trial is None and self._last_trial_rejected_by_nelec)
            phi = np.inf if trial is None else trial.objective
            dphi = (np.nan if trial is None else
                     self.inner(trial.gradient, direction))
            failed = (trial is None or info.projected or
                      not np.isfinite(phi) or not np.isfinite(dphi))
            point = _HZPoint(
                alpha, trial, phi, dphi, info, charge_boundary, failed)
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
            if point.trial_info.projected:
                info = point.trial_info
                if point.phi <= phi0 + self.config.line_search_c1 * info.actual_slope:
                    return finish(self._projected_line_search_result(
                        state, point.state, point.alpha, 0,
                        'accepted off-line projected Armijo Hager-Zhang point',
                        info=info))
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
            alpha_init: Optional[float] = None,
            alpha_cap_override: Optional[float] = None, *,
            allow_nelec_projection: bool = True,
            nelec_limit_override: Optional[float] = None,
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
                alpha_cap_override=alpha_cap_override,
                nelec_limit_override=nelec_limit_override,
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
        if alpha_cap_override is not None:
            if (not np.isfinite(alpha_cap_override) or
                    alpha_cap_override <= 0.0):
                raise ValueError(
                    'alpha_cap_override must be finite and positive')
            alpha_max = min(alpha_max, alpha_cap_override)
        projection_enabled = (
            allow_nelec_projection and
            self.config.line_search_nelec_guard_mode != 'reject')
        if not projection_enabled:
            maximum = (self._active_nelec_limit(
                state, use_adaptive_radius=False)
                if nelec_limit_override is None else
                float(nelec_limit_override))
            alpha_max, _ = self._charge_feasible_alpha_cap(
                state, direction, alpha_max, maximum=maximum)
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
            trial = self._trial(
                state, direction, alpha,
                allow_nelec_projection=allow_nelec_projection,
                nelec_limit_override=nelec_limit_override)
            trial_count += 1
            trial_info = self._last_trial_info
            if trial_info.projected:
                projected_armijo = (
                    trial is not None and
                    trial.objective <=
                    phi0 + c1 * trial_info.actual_slope)
                if projected_armijo:
                    return finish(self._projected_line_search_result(
                        state, trial, alpha, 0,
                        'accepted occupation-projected Armijo point',
                        info=trial_info))
                if trial is not None:
                    self._shrink_nelec_trust_radius(
                        trial_info.trust_radius)
                alpha *= self.config.armijo_backtrack_factor
                if alpha < self.config.line_search_alpha_min:
                    break
                continue
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
                    reduction_start=reduction_start,
                    allow_nelec_projection=allow_nelec_projection,
                    nelec_limit_override=nelec_limit_override)
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
                    reduction_start=reduction_start,
                    allow_nelec_projection=allow_nelec_projection,
                    nelec_limit_override=nelec_limit_override)
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
            alpha_cap_override: Optional[float] = None,
            allow_nelec_projection: bool = True,
            nelec_limit_override: Optional[float] = None,
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
        if alpha_cap_override is not None:
            alpha_max = min(alpha_max, alpha_cap_override)
        projection_enabled = (
            allow_nelec_projection and
            self.config.line_search_nelec_guard_mode != 'reject')
        if not projection_enabled:
            maximum = (self._active_nelec_limit(
                state, use_adaptive_radius=False)
                if nelec_limit_override is None else
                float(nelec_limit_override))
            alpha_max, _ = self._charge_feasible_alpha_cap(
                state, direction, alpha_max, maximum=maximum)
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
            trial = self._trial(
                state, direction, alpha,
                allow_nelec_projection=allow_nelec_projection,
                nelec_limit_override=nelec_limit_override)
            trial_count += 1
            trial_info = self._last_trial_info
            if trial_info.projected:
                if (trial is not None and
                        trial.objective <= state.objective +
                        self.config.line_search_c1 *
                        trial_info.actual_slope):
                    return finish(self._projected_line_search_result(
                        state, trial, alpha, 0,
                        'accepted occupation-projected Armijo fallback',
                        info=trial_info))
                if trial is not None:
                    self._shrink_nelec_trust_radius(
                        trial_info.trust_radius)
                alpha *= self.config.armijo_backtrack_factor
                if alpha < self.config.line_search_alpha_min:
                    break
                continue
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
        if line_search.nelec_projection_applied:
            actual_step = line_search.actual_step
            if actual_step is None:  # pragma: no cover - internal invariant
                raise RuntimeError(
                    'projected line-search result is missing its actual step')
            expected = self._sanitize_h(
                self.axpy(1.0, actual_step, state.h_orth))
            armijo_slope = self.inner(state.gradient, actual_step)
        else:
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
                if (line_search.line_search_method == 'hager-zhang' and
                    not line_search.nelec_projection_applied) else
                self.config.line_search_c1)
            if (accepted.objective > state.objective +
                    armijo_constant * armijo_slope + 1.0e-12):
                raise RuntimeError(
                    'accepted line-search point does not satisfy Armijo '
                    'decrease')

    def _record(self, cycle: int, old: _GCState, new: _GCState, line_search: _LineSearchResult,
                dphi0: float, beta: float, restart_reason: str,
                optimizer: str = 'nlcg', search_direction_source: str = 'nlcg',
                lbfgs_history_size: int = 0, lbfgs_pair_added: bool = False,
                lbfgs_sy: float = np.nan,
                lbfgs_curvature_cosine: float = np.nan,
                lbfgs_metric_scale: float = np.nan,
                lbfgs_history_action: str = '',
                descent_cosine: float = np.nan,
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
            optimizer, search_direction_source, lbfgs_history_size,
            lbfgs_pair_added, lbfgs_sy, lbfgs_curvature_cosine,
            lbfgs_metric_scale, lbfgs_history_action,
            line_search.strong_wolfe, line_search.message, descent_cosine,
            diis_history_size, diis_condition, diis_coefficient_l1,
            diis_damping, diis_history_action,
            diis_predicted_residual_rms, diis_trust_ratio,
            diis_next_damping,
            line_search.nelec_projection_applied,
            line_search.nelec_projection_mode,
            line_search.raw_delta_nelec,
            line_search.projected_delta_nelec,
            line_search.nelec_projection_parameter,
            line_search.nelec_trust_radius,
            line_search.nelec_trust_ratio,
            line_search.nelec_projection_response_fallback,
            line_search.nelec_projection_correction_rms,
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
            direction_preprojected=line_search.direction_preprojected,
            direction_projection_mode=(
                line_search.direction_projection_mode),
            direction_raw_endpoint_scale=(
                line_search.direction_raw_endpoint_scale),
            direction_raw_delta_nelec=(
                line_search.direction_raw_delta_nelec),
            direction_projected_delta_nelec=(
                line_search.direction_projected_delta_nelec),
            direction_accepted_delta_nelec=(
                line_search.direction_accepted_delta_nelec),
            direction_projection_correction_rms=(
                line_search.direction_projection_correction_rms),
            direction_projection_response_fallback=(
                line_search.direction_projection_response_fallback),
            direction_projection_trust_radius=(
                line_search.direction_projection_trust_radius),
            direction_projection_trust_ratio=(
                line_search.direction_projection_trust_ratio),
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
        if line_search.nelec_projection_applied:
            self.log.info(
                'Occupation projection (%s): raw delta N = %.6g, accepted '
                'delta N = %.6g, radius = %.6g, parameter = %.6g, '
                'correction RMS = %.6g, trust ratio = %.6g',
                line_search.nelec_projection_mode,
                line_search.raw_delta_nelec,
                line_search.projected_delta_nelec,
                line_search.nelec_trust_radius,
                line_search.nelec_projection_parameter,
                line_search.nelec_projection_correction_rms,
                line_search.nelec_trust_ratio)

    def _record_lbfgs(
            self, cycle: int, old: _GCState, new: _GCState,
            line_search: _LineSearchResult, dphi0: float,
            history_size: int, direction_source: str,
            pair_info: dict[str, Any], restart_reason: str,
            descent_cosine: float) -> None:
        self._record(
            cycle, old, new, line_search, dphi0, np.nan, restart_reason,
            optimizer='lbfgs', search_direction_source=direction_source,
            lbfgs_history_size=history_size,
            lbfgs_pair_added=pair_info['pair_added'],
            lbfgs_sy=pair_info['sy'],
            lbfgs_curvature_cosine=pair_info['curvature_cosine'],
            lbfgs_metric_scale=self._last_lbfgs_metric_scale,
            lbfgs_history_action=pair_info['action'],
            descent_cosine=descent_cosine)
        self.log.info(
            'L-BFGS history = %d  direction = %s  Wolfe = %s  '
            'metric scale = %.4g  %s',
            history_size, direction_source,
            line_search.curvature_qualified,
            self._last_lbfgs_metric_scale, pair_info['action'])

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
            self, result: GrandCanonicalResult
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

    def _fixed_n_view(self, state: _GCState) -> _GCState:
        """Reinterpret an evaluated fixed-mu state at its current fixed N."""
        if not self.fixed_electron_number:
            raise AssertionError('fixed-N state view requires fixed N')
        if abs(state.electron_number - self.target_electron_number) > max(
                self.config.mu_electron_number_tol, 1.0e-9):
            raise ValueError(
                'fixed-N state view target does not match the source state')
        h = self.copy_blocks(state.h_orth)
        fock = self.copy_blocks(state.fock_orth)
        mismatch = self.hermitize_blocks([
            hk - fk for hk, fk in zip(h, fock)])
        gauge_shift = self.trace_mean(mismatch)
        mismatch = [
            value - gauge_shift * identity
            for value, identity in zip(mismatch, self.identity)]
        mismatch = self.hermitize_blocks(mismatch)
        chemical_potential = state.auxiliary_mu - gauge_shift
        gradient = self._exact_gradient(
            h, fock, state.eigenvalues, state.u, state.occupations)
        z = self.hermitize_blocks([0.5 * value for value in mismatch])
        residual = self.scale_blocks(-1.0, z)
        grand_potential = (
            state.free_energy -
            chemical_potential * state.electron_number)
        return _GCState(
            h, state.gamma, state.eigenvalues, state.u, state.occupations,
            state.p_orth, state.dm_ao, state.veff, state.fock_ao, fock,
            state.auxiliary_mu, chemical_potential, gauge_shift,
            state.electronic_energy, state.nuclear_energy,
            state.dft_total_energy, state.electron_number, state.entropy,
            state.entropy_energy, state.free_energy, grand_potential,
            state.free_energy, gradient, z, residual,
            self.rms(gradient), self.rms(mismatch))

    def _reset_run_diagnostics(self) -> None:
        self.ncheap_nelec_reject = 0
        self._last_trial_rejected_by_nelec = False
        self._last_trial_info = _TrialInfo()
        self._nelec_trust_radius = (
            self.config.line_search_nelec_trust_initial)
        self.nnelec_projection_attempts = 0
        self.nnelec_projection_acceptances = 0
        self.nnelec_projection_fallbacks = 0
        self.max_raw_delta_nelec = 0.0
        self.max_projected_delta_nelec = 0.0
        self.max_nelec_projection_correction = 0.0
        self.last_nelec_trust_ratio = np.nan
        self.ncheap_nelec_evaluations = 0
        self.ncheap_nelec_alpha_reductions = 0
        self.ndirection_projection_attempts = 0
        self.ndirection_projection_acceptances = 0
        self.ndirection_projection_fallbacks = 0
        self.max_direction_projection_correction = 0.0
        self.nresidual_filter_acceptances = 0
        self.nresidual_filter_rejections = 0
        self.nlbfgs_switches = 0
        self.lbfgs_switch_cycle = -1
        self.lbfgs_switch_nfev = -1
        self.lbfgs_switch_actual_residual_rms = np.nan
        self.canonical_precondition_iterations = 0
        self.canonical_precondition_evaluations = 0
        self.canonical_precondition_residual_rms = np.nan
        self.canonical_precondition_canonical_residual_rms = np.nan
        self.canonical_precondition_delta_nelec = np.nan
        self.canonical_precondition_electron_number = np.nan
        self.canonical_precondition_mu_proxy = np.nan
        self.canonical_precondition_trigger = ''
        self.canonical_verification_attempts = 0
        self.canonical_verification_evaluations = 0
        self.canonical_verification_failures = 0
        self.canonical_verification_residual_rms = np.nan
        self.canonical_verification_grad_rms = np.nan
        self.canonical_verification_delta_nelec = np.nan
        self.canonical_verification_density_rms = np.nan
        self.canonical_terminal_mode = ''
        self._canonical_precondition_streak = 0
        self._canonical_precondition_last_nfev = -1

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
            lbfgs_switch_residual_rms=None,
            canonical_continuation_precondition_residual_rms=None,
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

    def _canonical_continuation_proposal(
            self, samples: Sequence[tuple[float, float]],
            physical_fock: Sequence, current_nelec: float,
            maximum_step: float) -> float:
        """Propose N before a chemical-potential root has been bracketed."""
        minimum_step = self.config.canonical_continuation_min_delta_nelec
        proposal = None
        negative = any(error < 0.0 for _, error in samples)
        positive = any(error > 0.0 for _, error in samples)
        if negative and positive:
            raise ValueError(
                'a bracketed canonical root must use Brent--Dekker')

        # Use the measured screened dmu/dN once two canonical states are
        # available.  The vacuum-aligned Fock projection is useful for the
        # first move but can be extremely aggressive at low temperature; a
        # positive local secant avoids geometrically growing past the root.
        current_n, current_error = samples[-1]
        previous = next(
            ((n, error) for n, error in reversed(samples[:-1])
             if abs(n - current_n) >= minimum_step), None)
        if previous is not None:
            slope = ((current_error - previous[1]) /
                     (current_n - previous[0]))
            if np.isfinite(slope) and slope > 0.0:
                local = current_n - current_error / slope
                if np.isfinite(local):
                    proposal = local
        if proposal is None:
            proposal = self._electron_number_at_mu(physical_fock, self.mu)
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

    def _canonical_brent_from_samples(
            self, samples: Sequence[tuple[float, float]]) -> Optional[_BrentRoot]:
        """Build Brent state from the narrowest evaluated sign bracket."""
        negative = [(n, error) for n, error in samples if error < 0.0]
        positive = [(n, error) for n, error in samples if error > 0.0]
        if not negative or not positive:
            return None
        left, right = min(
            ((negative_item, positive_item)
             for negative_item in negative for positive_item in positive),
            key=lambda pair: abs(pair[0][0] - pair[1][0]))
        return _BrentRoot.from_bracket(
            left[0], left[1], right[0], right[1],
            self.config.canonical_continuation_root_nelec_tol)

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

    def _kernel_canonical_continuation(
            self, dm0: Any = None, h0: Any = None, *,
            seed_state: Optional[_GCState] = None,
            prefix_history: Optional[Sequence[IterationRecord]] = None,
            prefix_niter: int = 0,
            prefix_nfev: int = 0) -> GrandCanonicalResult:
        """Globalize a fixed-mu solve through automatic fixed-N continuation."""
        if self.fixed_electron_number:
            raise AssertionError('canonical continuation requires fixed mu')
        if seed_state is None:
            self.history = []
            self.nfev = 0
            self._reset_run_diagnostics()
            h = self._initial_h(dm0, h0)
            initialization_evaluations = self.nfev
            current_nelec = self._electron_number_at_mu(h, self.mu)
            prefix_records: list[IterationRecord] = []
            prefix_niter = 0
            prefix_nfev = 0
        else:
            if dm0 is not None or h0 is not None:
                raise ValueError(
                    'an evaluated canonical seed may not be combined with '
                    'dm0 or h0')
            if prefix_nfev != self.nfev:
                raise ValueError(
                    'canonical prefix Fock count does not match the solver')
            h = self.copy_blocks(seed_state.h_orth)
            current_nelec = seed_state.electron_number
            initialization_evaluations = 0
            prefix_records = [replace(
                record,
                restart_reason=(
                    'fixed-mu canonical precondition' +
                    (('; ' + record.restart_reason)
                     if record.restart_reason else '')))
                for record in (prefix_history or ())]
        samples: list[tuple[float, float]] = []
        sample_residuals: list[float] = []
        sample_focks: list[list] = []
        brent_root: Optional[_BrentRoot] = None
        continuation_history: list[IterationRecord] = []
        continuation_evaluations = 0
        continuation_iterations = 0
        best_error = np.inf
        best_error_source = 'none'
        best_handoff_delta_nelec = np.inf
        best_handoff_score = np.inf
        best_h = self.copy_blocks(h)
        best_canonical_result: Optional[GrandCanonicalResult] = None
        last_canonical_result: Optional[GrandCanonicalResult] = None
        outer_steps = 0
        force_tight_refinement = False
        failed_inner_nelec: Optional[float] = None
        verification_repair_nelec: Optional[float] = None
        verified_state: Optional[_GCState] = None
        verified_source_result: Optional[GrandCanonicalResult] = None
        last_verification_state: Optional[_GCState] = None
        last_verification_source: Optional[GrandCanonicalResult] = None
        canonical_verification_attempts = 0
        canonical_verification_evaluations = 0
        canonical_verification_failures = 0
        canonical_verification_residual_rms = np.nan
        canonical_verification_grad_rms = np.nan
        canonical_verification_delta_nelec = np.nan
        canonical_verification_density_rms = np.nan
        outer_trust_radius = (
            self.config.
            canonical_continuation_precondition_initial_delta_nelec
            if seed_state is not None else
            self.config.canonical_continuation_initial_delta_nelec)

        for outer in range(self.config.canonical_continuation_max_outer):
            had_bracket = (
                brent_root is not None or
                (any(value < 0.0 for _, value in samples) and
                 any(value > 0.0 for _, value in samples)))
            tight_canonical_solve = (
                not self.config.canonical_continuation_final_polish and
                (had_bracket or force_tight_refinement))
            force_tight_refinement = False
            residual_tolerance = (
                self.config.canonical_continuation_bracketed_residual_tol
                if (tight_canonical_solve or had_bracket) else
                self.config.canonical_continuation_coarse_residual_tol)
            # A conservative fixed-point seed is cheaper than trying and
            # rejecting several over-aggressive full DIIS extrapolations.
            # The trust update expands this damping rapidly when warranted.
            initial_damping = (
                self.config.canonical_continuation_initial_damping)
            canonical_solver = self._spawn_fixed_n(
                current_nelec,
                self._canonical_continuation_config(
                    residual_tolerance, initial_damping))
            if outer == 0 and seed_state is not None:
                canonical_solver.history = []
                canonical_solver.nfev = 0
                canonical_solver._reset_run_diagnostics()
                canonical_state = canonical_solver._fixed_n_view(seed_state)
                canonical_result = canonical_solver._kernel_diis(
                    canonical_state, None, niter=0, cycle_start=0)
            else:
                canonical_result = canonical_solver.kernel(h0=h)
            last_canonical_result = canonical_result
            outer_steps += 1
            stage_evaluation_offset = (
                prefix_nfev + initialization_evaluations +
                continuation_evaluations +
                canonical_verification_evaluations)
            continuation_evaluations += canonical_result.nfev
            stage_history = self._continuation_history(
                canonical_result.history,
                prefix_niter + continuation_iterations,
                current_nelec, stage_evaluation_offset)
            continuation_history.extend(stage_history)
            continuation_iterations += canonical_result.niter
            h = canonical_result.h_orth
            error = canonical_result.mu - self.mu
            target_nelec = self._electron_number_at_mu(
                canonical_result.fock_orth, self.mu)
            handoff_delta_nelec = (
                target_nelec - canonical_result.electron_number)
            if canonical_result.converged:
                duplicate = next(
                    (index for index, (sample_nelec, _) in enumerate(samples)
                     if abs(sample_nelec - current_nelec) <=
                     32.0 * np.finfo(float).eps *
                     max(1.0, abs(sample_nelec), abs(current_nelec))), None)
                if duplicate is None:
                    samples.append((current_nelec, error))
                    sample_residuals.append(canonical_result.residual_rms)
                    sample_focks.append(self.copy_blocks(
                        canonical_result.fock_orth))
                    if brent_root is not None:
                        if (brent_root.pending is not None and
                                abs(brent_root.pending - current_nelec) <=
                                32.0 * np.finfo(float).eps *
                                max(1.0, abs(current_nelec))):
                            brent_root.update(current_nelec, error)
                        else:
                            # A safeguarded recovery can deliberately leave a
                            # pending interpolation.  Reconstruct the bracket
                            # from evaluated data rather than pretending that
                            # unevaluated point updated Brent's history.
                            brent_root = None
                else:
                    # A coarse-to-tight solve at exactly the same N refines a
                    # function value; it is not another scalar-root step.
                    if (canonical_result.residual_rms <=
                            sample_residuals[duplicate]):
                        samples[duplicate] = (current_nelec, error)
                        sample_residuals[duplicate] = (
                            canonical_result.residual_rms)
                        sample_focks[duplicate] = self.copy_blocks(
                            canonical_result.fock_orth)
                    brent_root = None
                if brent_root is None:
                    brent_root = self._canonical_brent_from_samples(samples)
                failed_inner_nelec = None
            bracketed = brent_root is not None
            self.log.info(
                'Canonical continuation %d: N = %.12g, optimized mu = '
                '%.12g, target mu = %.12g, delta mu = %.3g, residual = %.3g, '
                'physical delta N = %.3g, Fock evaluations = %d',
                outer, current_nelec, canonical_result.mu, self.mu, error,
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
                best_handoff_delta_nelec = handoff_delta_nelec
                best_error = error
                best_error_source = 'evaluated'
                best_h = self._canonical_fixed_mu_candidate(
                    canonical_result)[0]
                best_canonical_result = canonical_result

            if not self.config.canonical_continuation_final_polish:
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
                        brent_root = self._canonical_brent_from_samples(samples)
                        h = self.copy_blocks(
                            best_canonical_result.fock_orth)
                        force_tight_refinement = True
                        self.log.warn(
                            'Retrying failed fixed-N solve once at N = %.12g',
                            current_nelec)
                        continue

                    retry_nelec = None
                    brent_root = self._canonical_brent_from_samples(samples)
                    if brent_root is not None and not brent_root.converged:
                        lo, hi = brent_root.bracket
                        retry_nelec = lo + 0.5 * (hi - lo)
                        if abs(retry_nelec - current_nelec) <= (
                                32.0 * np.finfo(float).eps *
                                max(1.0, abs(current_nelec))):
                            # The failed point was already the midpoint.  Split
                            # its interval to Brent's best endpoint instead of
                            # immediately abandoning a valid outer bracket.
                            retry_nelec = (
                                current_nelec +
                                0.5 * (brent_root.b - current_nelec))
                    elif samples:
                        retry_nelec = 0.5 * (
                            current_nelec + samples[-1][0])
                    if (retry_nelec is not None and
                            abs(retry_nelec - current_nelec) >
                            1.0e-14 * max(1.0, abs(current_nelec))):
                        outer_trust_radius = max(
                            self.config.canonical_continuation_min_delta_nelec,
                            0.5 * outer_trust_radius)
                        h = self.copy_blocks(
                            best_canonical_result.fock_orth)
                        current_nelec = retry_nelec
                        brent_root = None
                        failed_inner_nelec = None
                        force_tight_refinement = True
                        self.log.warn(
                            'Retrying fixed-N continuation at safeguarded '
                            'N = %.12g', current_nelec)
                        continue
                    break
                (h_fixed_mu, _, physical_delta_nelec,
                 predicted_residual_rms) = (
                    self._canonical_fixed_mu_candidate(canonical_result))
                canonical_ready = (
                    canonical_result.residual_rms <=
                    self.config.
                    canonical_continuation_bracketed_residual_tol and
                    abs(error) <=
                    self.config.canonical_continuation_handoff_delta_mu and
                    abs(physical_delta_nelec) <=
                    self.config.
                    canonical_continuation_handoff_delta_nelec and
                    predicted_residual_rms <=
                    self.config.
                    canonical_continuation_verification_residual_tol)
                if canonical_ready:
                    self.nfev = (
                        prefix_nfev + initialization_evaluations +
                        continuation_evaluations +
                        canonical_verification_evaluations)
                    before_verification = self.nfev
                    candidate_state = self.evaluate(h_fixed_mu)
                    verification_work = self.nfev - before_verification
                    canonical_verification_attempts += 1
                    canonical_verification_evaluations += verification_work
                    canonical_verification_residual_rms = (
                        candidate_state.residual_rms)
                    canonical_verification_grad_rms = candidate_state.grad_rms
                    canonical_verification_delta_nelec = (
                        self._electron_number_at_mu(
                            candidate_state.fock_orth, self.mu) -
                        candidate_state.electron_number)
                    canonical_verification_density_rms = self.rms(
                        self.axpy(-1.0, canonical_result.p_orth,
                                  candidate_state.p_orth))
                    verification_ok = (
                        verification_work == 1 and
                        candidate_state.residual_rms <=
                        self.config.
                        canonical_continuation_verification_residual_tol and
                        abs(canonical_verification_delta_nelec) <=
                        self.config.
                        canonical_continuation_handoff_delta_nelec and
                        canonical_verification_density_rms <=
                        self.config.
                        canonical_continuation_verification_density_tol)
                    last_verification_state = candidate_state
                    last_verification_source = canonical_result
                    if verification_ok:
                        verified_state = candidate_state
                        verified_source_result = canonical_result
                        best_error = error
                        best_error_source = 'evaluated'
                        best_handoff_delta_nelec = physical_delta_nelec
                        self.log.info(
                            'Canonical root verified at fixed mu with one '
                            'Fock build: delta mu = %.3g, delta N = %.3g, '
                            'residual = %.3g, gradient = %.3g',
                            error, physical_delta_nelec,
                            candidate_state.residual_rms,
                            candidate_state.grad_rms)
                        break
                    canonical_verification_failures += 1
                    self.log.warn(
                        'One-Fock fixed-mu verification failed: residual '
                        '%.3g, gradient %.3g, delta N %.3g, density change '
                        '%.3g; resuming fixed-N refinement',
                        candidate_state.residual_rms,
                        candidate_state.grad_rms,
                        canonical_verification_delta_nelec,
                        canonical_verification_density_rms)
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
                        brent_root = self._canonical_brent_from_samples(samples)
                        force_tight_refinement = True
                        continue

                    # A persistent failure must not cycle forever at the same
                    # N.  Use the verification Fock response only to choose a
                    # safeguarded interior recovery point; its optimized mu is
                    # evaluated by the next fixed-N solve before Brent sees it.
                    brent_root = self._canonical_brent_from_samples(samples)
                    if brent_root is not None and not brent_root.converged:
                        lo, hi = brent_root.bracket
                        preferred = (
                            canonical_result.electron_number +
                            canonical_verification_delta_nelec)
                        margin = min(
                            self.config.canonical_continuation_root_nelec_tol,
                            0.25 * (hi - lo))
                        if lo + margin < preferred < hi - margin:
                            retry_nelec = preferred
                        else:
                            retry_nelec = lo + 0.5 * (hi - lo)
                        h = self.copy_blocks(candidate_state.fock_orth)
                        current_nelec = retry_nelec
                        brent_root = None
                        verification_repair_nelec = None
                        force_tight_refinement = True
                        continue
                    self.log.warn(
                        'Persistent fixed-mu verification failure left no '
                        'resolvable Brent interval')
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
                    continue

            if brent_root is not None:
                # A coarse value can supply the first sign change but may not
                # be accurate enough to steer an expensive Brent iteration.
                # Re-solve the best endpoint before trusting the bracket, and
                # tighten again near the root so the requested physical charge
                # tolerance is not dominated by fixed-N residual noise.
                endpoint_index = next(
                    (index for index, (sample_nelec, _) in enumerate(samples)
                     if abs(sample_nelec - brent_root.b) <=
                     32.0 * np.finfo(float).eps * max(
                         1.0, abs(sample_nelec), abs(brent_root.b))), None)
                endpoint_tolerance = (
                    self.config.canonical_continuation_bracketed_residual_tol)
                if (endpoint_index is not None and
                        sample_residuals[endpoint_index] > endpoint_tolerance):
                    h = self.copy_blocks(sample_focks[endpoint_index])
                    current_nelec = brent_root.b
                    force_tight_refinement = True
                    self.log.info(
                        'Refining Brent endpoint N = %.12g from residual %.3g '
                        'to %.3g before the next root proposal',
                        current_nelec, sample_residuals[endpoint_index],
                        endpoint_tolerance)
                    continue
            # Continuation is a globalization device, not the final solver.
            # Require both a safe physical-Fock Fermi response and a physically
            # meaningful optimized chemical potential.  The charge test is
            # essential at low temperature because the same delta mu is
            # benign in a gap but can move many electrons near a crossing.
            unbracketed_handoff_limit = (
                self.config.canonical_continuation_unbracketed_handoff_delta_nelec)
            handoff_limit = (
                self.config.canonical_continuation_handoff_delta_nelec
                if bracketed else unbracketed_handoff_limit)
            mu_handoff_ready = (
                abs(error) <=
                self.config.canonical_continuation_handoff_delta_mu)
            if (self.config.canonical_continuation_final_polish and
                    canonical_result.converged and mu_handoff_ready and
                    abs(handoff_delta_nelec) <= handoff_limit):
                best_h = self._canonical_fixed_mu_candidate(
                    canonical_result)[0]
                best_handoff_delta_nelec = handoff_delta_nelec
                best_error = error
                best_error_source = 'evaluated'
                self.log.info(
                    'Canonical continuation reached the fixed-mu handoff '
                    'window (delta N = %.3g, delta mu = %.3g, bracketed = '
                    '%s); starting fixed-mu polish', handoff_delta_nelec,
                    error, bracketed)
                break
            if not canonical_result.converged:
                self.log.warn(
                    'Canonical continuation inner solve did not converge: %s; '
                    'proceeding to fixed-mu verification from the best state',
                    canonical_result.message)
                break

            if (not bracketed and len(samples) >= 2 and
                    np.sign(samples[-1][1]) == np.sign(samples[-2][1])):
                outer_trust_radius = min(
                    self.config.canonical_continuation_max_delta_nelec,
                    2.0 * outer_trust_radius)
            if brent_root is not None:
                if brent_root.converged:
                    self.log.warn(
                        'Canonical continuation Brent interval resolved at '
                        'width %.3g without satisfying physical verification',
                        brent_root.width)
                    break
                proposal = brent_root.proposal()
                endpoint_focks = []
                for endpoint in (brent_root.a, brent_root.b):
                    endpoint_index = next(
                        (index for index, (sample_nelec, _) in
                         enumerate(samples)
                         if abs(sample_nelec - endpoint) <=
                         32.0 * np.finfo(float).eps *
                         max(1.0, abs(sample_nelec), abs(endpoint))), None)
                    endpoint_focks.append(
                        None if endpoint_index is None else
                        sample_focks[endpoint_index])
                proposal_h = None
                if all(value is not None for value in endpoint_focks):
                    fraction = ((proposal - brent_root.a) /
                                (brent_root.b - brent_root.a))
                    proposal_h = self._sanitize_h([
                        (1.0 - fraction) * h_a + fraction * h_b
                        for h_a, h_b in zip(*endpoint_focks)])
                self.log.info(
                    'Canonical continuation Brent %s proposal N = %.12g '
                    'inside [%.12g, %.12g]',
                    brent_root.last_method, proposal,
                    brent_root.bracket[0], brent_root.bracket[1])
            else:
                proposal = self._canonical_continuation_proposal(
                    samples, canonical_result.fock_orth,
                    current_nelec, outer_trust_radius)
                proposal_h = None
            if abs(proposal - current_nelec) <= (
                    1.0e-14 * max(1.0, abs(current_nelec))):
                self.log.warn(
                    'Canonical continuation scalar root stagnated at '
                    'N = %.12g', current_nelec)
                break
            h = (proposal_h if proposal_h is not None else
                 self.copy_blocks(canonical_result.fock_orth))
            current_nelec = proposal
        else:
            self.log.warn(
                'Canonical continuation reached its maximum of %d outer steps; '
                'proceeding to fixed-mu verification',
                self.config.canonical_continuation_max_outer)

        if not self.config.canonical_continuation_final_polish:
            terminal_success = verified_state is not None
            terminal_state = verified_state
            terminal_source = verified_source_result
            if terminal_state is None:
                reuse_last_verification = (
                    last_verification_state is not None and
                    (best_canonical_result is None or
                     last_verification_source is best_canonical_result))
                if reuse_last_verification:
                    terminal_state = last_verification_state
                    terminal_source = last_verification_source
                else:
                    terminal_source = (
                        best_canonical_result or last_canonical_result)
                    if terminal_source is None:  # pragma: no cover
                        raise RuntimeError(
                            'canonical continuation produced no state')
                    (fallback_h, _, fallback_delta_nelec,
                     _) = self._canonical_fixed_mu_candidate(terminal_source)
                    self.nfev = (
                        prefix_nfev + initialization_evaluations +
                        continuation_evaluations +
                        canonical_verification_evaluations)
                    before_verification = self.nfev
                    terminal_state = self.evaluate(fallback_h)
                    verification_work = self.nfev - before_verification
                    canonical_verification_attempts += 1
                    canonical_verification_evaluations += verification_work
                    canonical_verification_failures += 1
                    canonical_verification_residual_rms = (
                        terminal_state.residual_rms)
                    canonical_verification_grad_rms = terminal_state.grad_rms
                    canonical_verification_delta_nelec = (
                        self._electron_number_at_mu(
                            terminal_state.fock_orth, self.mu) -
                        terminal_state.electron_number)
                    canonical_verification_density_rms = self.rms(
                        self.axpy(-1.0, terminal_source.p_orth,
                                  terminal_state.p_orth))
                    best_handoff_delta_nelec = fallback_delta_nelec

            if terminal_source is None:  # pragma: no cover
                raise RuntimeError(
                    'fixed-mu verification lacks its canonical source')
            source_error = terminal_source.mu - self.mu
            (_, _, source_delta_nelec,
             _) = self._canonical_fixed_mu_candidate(terminal_source)
            best_error = source_error
            best_error_source = 'evaluated'
            best_handoff_delta_nelec = source_delta_nelec
            total_evaluations = (
                prefix_nfev + initialization_evaluations +
                continuation_evaluations +
                canonical_verification_evaluations)
            total_iterations = prefix_niter + continuation_iterations
            combined_history = prefix_records + continuation_history
            self.nfev = total_evaluations
            self.history = combined_history
            density_change = self.rms(
                self.axpy(-1.0, terminal_source.p_orth,
                          terminal_state.p_orth))
            terminal_mode = (
                'canonical-verification' if terminal_success else
                'canonical-verification-failed')
            message = (
                f'canonical continuation ({outer_steps} outer steps, '
                f'{continuation_evaluations} fixed-N Fock evaluations, '
                f'{canonical_verification_evaluations} verification '
                f'evaluations); ' +
                ('converged by one-Fock fixed-mu verification'
                 if terminal_success else
                 'failed to satisfy canonical root verification'))
            self.canonical_verification_attempts = (
                canonical_verification_attempts)
            self.canonical_verification_evaluations = (
                canonical_verification_evaluations)
            self.canonical_verification_failures = (
                canonical_verification_failures)
            self.canonical_verification_residual_rms = (
                canonical_verification_residual_rms)
            self.canonical_verification_grad_rms = (
                canonical_verification_grad_rms)
            self.canonical_verification_delta_nelec = (
                canonical_verification_delta_nelec)
            self.canonical_verification_density_rms = (
                canonical_verification_density_rms)
            self.canonical_terminal_mode = terminal_mode
            self._checkpoint(terminal_state, total_iterations, force=True)
            final_result = self._finalize(
                terminal_state, terminal_success, message,
                total_iterations, density_change)
            self.mf.canonical_continuation_mu_error_source_gc = (
                best_error_source)
            self.mf.canonical_verification_attempts_gc = (
                canonical_verification_attempts)
            self.mf.canonical_verification_evaluations_gc = (
                canonical_verification_evaluations)
            self.mf.canonical_verification_failures_gc = (
                canonical_verification_failures)
            self.mf.canonical_terminal_mode_gc = terminal_mode
            self.mf.scf_summary.update({
                'canonical_continuation_steps': outer_steps,
                'canonical_continuation_evaluations': (
                    continuation_evaluations),
                'canonical_continuation_mu_error': best_error,
                'canonical_continuation_mu_error_source': best_error_source,
                'canonical_continuation_delta_nelec': (
                    best_handoff_delta_nelec),
                'canonical_precondition_iterations': (
                    self.canonical_precondition_iterations),
                'canonical_precondition_evaluations': (
                    self.canonical_precondition_evaluations),
                'canonical_precondition_residual_rms': (
                    self.canonical_precondition_residual_rms),
                'canonical_precondition_canonical_residual_rms': (
                    self.canonical_precondition_canonical_residual_rms),
                'canonical_precondition_delta_nelec': (
                    self.canonical_precondition_delta_nelec),
                'canonical_precondition_electron_number': (
                    self.canonical_precondition_electron_number),
                'canonical_precondition_mu_proxy': (
                    self.canonical_precondition_mu_proxy),
                'canonical_precondition_trigger': (
                    self.canonical_precondition_trigger),
                'canonical_verification_attempts': (
                    canonical_verification_attempts),
                'canonical_verification_evaluations': (
                    canonical_verification_evaluations),
                'canonical_verification_failures': (
                    canonical_verification_failures),
                'canonical_verification_residual_rms': (
                    canonical_verification_residual_rms),
                'canonical_verification_grad_rms': (
                    canonical_verification_grad_rms),
                'canonical_verification_delta_nelec': (
                    canonical_verification_delta_nelec),
                'canonical_verification_density_rms': (
                    canonical_verification_density_rms),
                'canonical_terminal_mode': terminal_mode,
                'fock_evaluations_total': total_evaluations,
            })
            return replace(
                final_result,
                canonical_continuation_steps=outer_steps,
                canonical_continuation_evaluations=(
                    continuation_evaluations),
                canonical_continuation_mu_error=best_error,
                canonical_continuation_mu_error_source=best_error_source,
                canonical_continuation_delta_nelec=(
                    best_handoff_delta_nelec),
                canonical_precondition_iterations=(
                    self.canonical_precondition_iterations),
                canonical_precondition_evaluations=(
                    self.canonical_precondition_evaluations),
                canonical_precondition_residual_rms=(
                    self.canonical_precondition_residual_rms),
                canonical_precondition_canonical_residual_rms=(
                    self.canonical_precondition_canonical_residual_rms),
                canonical_precondition_delta_nelec=(
                    self.canonical_precondition_delta_nelec),
                canonical_precondition_electron_number=(
                    self.canonical_precondition_electron_number),
                canonical_precondition_mu_proxy=(
                    self.canonical_precondition_mu_proxy),
                canonical_precondition_trigger=(
                    self.canonical_precondition_trigger),
                canonical_verification_attempts=(
                    canonical_verification_attempts),
                canonical_verification_evaluations=(
                    canonical_verification_evaluations),
                canonical_verification_failures=(
                    canonical_verification_failures),
                canonical_verification_residual_rms=(
                    canonical_verification_residual_rms),
                canonical_verification_grad_rms=(
                    canonical_verification_grad_rms),
                canonical_verification_delta_nelec=(
                    canonical_verification_delta_nelec),
                canonical_verification_density_rms=(
                    canonical_verification_density_rms),
                canonical_terminal_mode=terminal_mode,
            )

        h = best_h

        pre_evaluations = (
            prefix_nfev + initialization_evaluations +
            continuation_evaluations)
        pre_iterations = prefix_niter + continuation_iterations
        precondition_diagnostics = {
            'iterations': self.canonical_precondition_iterations,
            'evaluations': self.canonical_precondition_evaluations,
            'residual_rms': self.canonical_precondition_residual_rms,
            'canonical_residual_rms': (
                self.canonical_precondition_canonical_residual_rms),
            'delta_nelec': self.canonical_precondition_delta_nelec,
            'electron_number': self.canonical_precondition_electron_number,
            'mu_proxy': self.canonical_precondition_mu_proxy,
            'trigger': self.canonical_precondition_trigger,
        }
        saved_initial_damping = self.config.diis_initial_damping
        saved_max_coefficient_l1 = self.config.diis_max_coefficient_l1
        self.config.diis_initial_damping = min(
            saved_initial_damping,
            self.config.canonical_continuation_final_damping)
        self.config.diis_max_coefficient_l1 = max(
            saved_max_coefficient_l1,
            self.config.canonical_continuation_diis_max_coefficient_l1)
        try:
            if self.config.optimizer == 'nlcg':
                final_result = self._kernel_nlcg(
                    h0=h, allow_canonical_handoff=False)
            elif self.config.optimizer == 'lbfgs':
                final_result = self._kernel_lbfgs(h0=h)
            else:  # pragma: no cover - validated during construction
                raise AssertionError('validated optimizer is unreachable')
        finally:
            self.config.diis_initial_damping = saved_initial_damping
            self.config.diis_max_coefficient_l1 = saved_max_coefficient_l1

        final_history = self._continuation_history(
            final_result.history, pre_iterations, final_result.electron_number,
            pre_evaluations)
        # The final records are fixed-mu, not canonical; remove the stage
        # prefix while retaining globally increasing cycle numbers.
        final_history = [replace(
            record,
            restart_reason=(record.restart_reason.split('; ', 1)[1]
                            if '; ' in record.restart_reason else ''))
            for record in final_history]
        combined_history = (
            prefix_records + continuation_history + final_history)
        total_evaluations = pre_evaluations + final_result.nfev
        total_iterations = pre_iterations + final_result.niter
        switch_cycle = final_result.lbfgs_switch_cycle
        switch_nfev = final_result.lbfgs_switch_nfev
        if switch_cycle >= 0:
            switch_cycle += pre_iterations
        if switch_nfev >= 0:
            switch_nfev += pre_evaluations
        self.nfev = total_evaluations
        self.ncheap_nelec_reject = final_result.cheap_nelec_rejections
        self.history = combined_history
        self.canonical_precondition_iterations = (
            precondition_diagnostics['iterations'])
        self.canonical_precondition_evaluations = (
            precondition_diagnostics['evaluations'])
        self.canonical_precondition_residual_rms = (
            precondition_diagnostics['residual_rms'])
        self.canonical_precondition_canonical_residual_rms = (
            precondition_diagnostics['canonical_residual_rms'])
        self.canonical_precondition_delta_nelec = (
            precondition_diagnostics['delta_nelec'])
        self.canonical_precondition_electron_number = (
            precondition_diagnostics['electron_number'])
        self.canonical_precondition_mu_proxy = (
            precondition_diagnostics['mu_proxy'])
        self.canonical_precondition_trigger = (
            precondition_diagnostics['trigger'])
        self.mf.canonical_precondition_iterations_gc = (
            self.canonical_precondition_iterations)
        self.mf.canonical_precondition_evaluations_gc = (
            self.canonical_precondition_evaluations)
        self.mf.canonical_precondition_residual_rms_gc = (
            self.canonical_precondition_residual_rms)
        self.mf.canonical_precondition_canonical_residual_rms_gc = (
            self.canonical_precondition_canonical_residual_rms)
        self.mf.canonical_precondition_delta_nelec_gc = (
            self.canonical_precondition_delta_nelec)
        self.mf.canonical_precondition_electron_number_gc = (
            self.canonical_precondition_electron_number)
        self.mf.canonical_precondition_mu_proxy_gc = (
            self.canonical_precondition_mu_proxy)
        self.mf.canonical_precondition_trigger_gc = (
            self.canonical_precondition_trigger)
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
        self.mf.canonical_continuation_mu_error_source_gc = (
            best_error_source)
        self.canonical_terminal_mode = 'fixed-mu-polish'
        self.mf.canonical_terminal_mode_gc = 'fixed-mu-polish'
        self.mf.scf_summary.update({
            'canonical_continuation_steps': outer_steps,
            'canonical_continuation_evaluations': continuation_evaluations,
            'canonical_continuation_mu_error': best_error,
            'canonical_continuation_mu_error_source': best_error_source,
            'canonical_continuation_delta_nelec': (
                best_handoff_delta_nelec),
            'canonical_precondition_iterations': (
                self.canonical_precondition_iterations),
            'canonical_precondition_evaluations': (
                self.canonical_precondition_evaluations),
            'canonical_precondition_residual_rms': (
                self.canonical_precondition_residual_rms),
            'canonical_precondition_canonical_residual_rms': (
                self.canonical_precondition_canonical_residual_rms),
            'canonical_precondition_delta_nelec': (
                self.canonical_precondition_delta_nelec),
            'canonical_precondition_electron_number': (
                self.canonical_precondition_electron_number),
            'canonical_precondition_mu_proxy': (
                self.canonical_precondition_mu_proxy),
            'canonical_precondition_trigger': (
                self.canonical_precondition_trigger),
            'canonical_terminal_mode': 'fixed-mu-polish',
            'fock_evaluations_total': total_evaluations,
            'lbfgs_switch_cycle_gc': switch_cycle,
            'lbfgs_switch_nfev_gc': switch_nfev,
        })
        self.lbfgs_switch_cycle = switch_cycle
        self.lbfgs_switch_nfev = switch_nfev
        self.mf.lbfgs_switch_cycle_gc = switch_cycle
        self.mf.lbfgs_switch_nfev_gc = switch_nfev
        message = (
            f'canonical continuation ({outer_steps} outer steps, '
            f'{continuation_evaluations} Fock evaluations); '
            f'{final_result.message}')
        return replace(
            final_result,
            message=message,
            niter=total_iterations,
            nfev=total_evaluations,
            history=combined_history,
            canonical_continuation_steps=outer_steps,
            canonical_continuation_evaluations=continuation_evaluations,
            canonical_continuation_mu_error=best_error,
            canonical_continuation_mu_error_source=best_error_source,
            canonical_continuation_delta_nelec=best_handoff_delta_nelec,
            lbfgs_switch_cycle=switch_cycle,
            lbfgs_switch_nfev=switch_nfev,
            canonical_precondition_iterations=(
                self.canonical_precondition_iterations),
            canonical_precondition_evaluations=(
                self.canonical_precondition_evaluations),
            canonical_precondition_residual_rms=(
                self.canonical_precondition_residual_rms),
            canonical_precondition_canonical_residual_rms=(
                self.canonical_precondition_canonical_residual_rms),
            canonical_precondition_delta_nelec=(
                self.canonical_precondition_delta_nelec),
            canonical_precondition_electron_number=(
                self.canonical_precondition_electron_number),
            canonical_precondition_mu_proxy=(
                self.canonical_precondition_mu_proxy),
            canonical_precondition_trigger=(
                self.canonical_precondition_trigger),
            canonical_terminal_mode='fixed-mu-polish',
        )

    def kernel(self, dm0: Any = None, h0: Any = None) -> GrandCanonicalResult:
        """Run the configured safeguarded direct minimizer."""
        if (self.config.canonical_continuation and
                not self.fixed_electron_number and
                not self._canonical_precondition_enabled()):
            return self._kernel_canonical_continuation(dm0=dm0, h0=h0)
        if self.config.optimizer == 'nlcg':
            return self._kernel_nlcg(dm0=dm0, h0=h0)
        if self.config.optimizer == 'lbfgs':
            return self._kernel_lbfgs(dm0=dm0, h0=h0)
        raise AssertionError('validated optimizer is unreachable')

    def _kernel_nlcg(self, dm0: Any = None,
                     h0: Any = None, *,
                     allow_canonical_handoff: bool = True
                     ) -> GrandCanonicalResult:
        """Run safeguarded fixed-mu or fixed-electron nonlinear CG."""
        self.history = []
        self.nfev = 0
        self._reset_run_diagnostics()
        self._nlcg_residual_previous_alpha = None
        self._lbfgs_history = []
        self._diis_history = []
        state = self.evaluate(self._initial_h(dm0, h0))
        previous: Optional[_GCState] = None
        direction = self.copy_blocks(state.residual)
        consecutive = 0
        message = 'maximum cycles reached'
        converged = False
        niter = 0
        canonical_prefix = (
            allow_canonical_handoff and
            self._canonical_precondition_enabled())

        for cycle in range(self.config.max_cycle):
            if (canonical_prefix and
                    self._should_start_canonical_continuation(state, niter)):
                return self._start_canonical_continuation_from_prefix(
                    state, niter)
            if self._meets_convergence(state, previous):
                if canonical_prefix:
                    return self._start_canonical_continuation_from_prefix(
                        state, niter, trigger='fixed-mu-converged')
                consecutive += 1
                if previous is None or consecutive >= self.config.required_consecutive_conv:
                    converged, message = True, 'converged'
                    break
            else:
                consecutive = 0
            if not canonical_prefix and self._should_start_diis(state):
                self.log.info(
                    'Switching from NLCG to residual DIIS at |F-H|_rms = %.6g; '
                    'CG memory reset', state.residual_rms)
                return self._kernel_diis(
                    state, previous, niter=niter, cycle_start=cycle)
            if not canonical_prefix and self._should_start_lbfgs(state):
                self.nlbfgs_switches += 1
                self.lbfgs_switch_cycle = cycle
                self.lbfgs_switch_nfev = self.nfev
                self.lbfgs_switch_actual_residual_rms = state.residual_rms
                self._nlcg_residual_previous_alpha = None
                self.log.info(
                    'Switching from NLCG to L-BFGS at |F-H|_rms = %.6g; '
                    'CG and L-BFGS memory reset', state.residual_rms)
                return self._kernel_lbfgs_from_state(
                    state, previous, niter=niter, cycle_start=cycle,
                    line_search_method=(
                        self.config.lbfgs_switch_line_search_method),
                    residual_filter_enabled=False)
            direction, restarted, restart_reason = self._ensure_descent(state, direction)
            if not self._is_descent(state, direction):
                if canonical_prefix:
                    return self._start_canonical_continuation_from_prefix(
                        state, niter, trigger='loss-of-descent')
                if state.grad_rms < self.config.conv_tol_grad_rms and state.residual_rms < self.config.conv_tol_residual_rms:
                    converged, message = True, 'stationary initial state'
                else:
                    message = 'persistent loss of descent'
                break
            if self._alpha_cap(direction) < self.config.line_search_alpha_min:
                direction, cap_reason = self._restart_direction(state)
                if self._alpha_cap(direction) < self.config.line_search_alpha_min:
                    if canonical_prefix:
                        return self._start_canonical_continuation_from_prefix(
                            state, niter, trigger='step-cap-failure')
                    message = 'stagnation: step cap below minimum'
                    break
                restarted, restart_reason = True, 'step cap restart; ' + cap_reason
            fixed_direction_projection = (
                not self.fixed_electron_number and
                self.config.nlcg_nelec_projection_strategy == 'direction')
            prepared = self._prepare_nlcg_direction(state, direction)
            if not prepared.success:
                failed_preparation = prepared.message
                direction, prepare_reason = self._restart_direction(state)
                prepared = self._charge_capped_direction_fallback(
                    state, direction,
                    'fixed-direction preparation failed; using a cheap '
                    'charge-capped residual line')
                restarted = True
                restart_reason = (
                    prepare_reason + '; ' + failed_preparation + '; ' +
                    prepared.message)
            if not prepared.success:
                if canonical_prefix:
                    return self._start_canonical_continuation_from_prefix(
                        state, niter,
                        trigger='direction-preparation-failure')
                message = 'direction preparation failure: ' + prepared.message
                break
            direction = prepared.direction
            nelec_limit = (
                prepared.trust_radius
                if fixed_direction_projection and
                np.isfinite(prepared.trust_radius) else None)
            dphi0 = self.inner(state.gradient, direction)
            alpha_init = self._nlcg_residual_alpha_init(state)
            prefix_remaining_evaluations = None
            if canonical_prefix:
                prefix_remaining_evaluations = max(
                    0,
                    self.config.
                    canonical_continuation_precondition_max_fock_evaluations -
                    self.nfev)
            line_search = self._line_search(
                state, direction,
                alpha_init=alpha_init,
                alpha_cap_override=prepared.alpha_cap,
                allow_nelec_projection=not fixed_direction_projection,
                nelec_limit_override=nelec_limit,
                max_evals_override=prefix_remaining_evaluations)
            if not line_search.success and canonical_prefix:
                trigger = (
                    'max-fock-budget'
                    if self.nfev >= self.config.
                    canonical_continuation_precondition_max_fock_evaluations
                    else 'line-search-failure')
                return self._start_canonical_continuation_from_prefix(
                    state, niter, trigger=trigger)
            if not line_search.success:
                primary_line_search = line_search
                direction, fallback_reason = self._restart_direction(state)
                prepared = self._prepare_nlcg_direction(state, direction)
                if not prepared.success:
                    failed_preparation = prepared.message
                    prepared = self._charge_capped_direction_fallback(
                        state, direction,
                        'fallback fixed-direction preparation failed; using '
                        'a cheap charge-capped residual line')
                    fallback_reason += (
                        '; ' + failed_preparation + '; ' + prepared.message)
                if not prepared.success:
                    message = 'fallback direction preparation failure'
                    break
                direction = prepared.direction
                nelec_limit = (
                    prepared.trust_radius
                    if fixed_direction_projection and
                    np.isfinite(prepared.trust_radius) else None)
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
                        max_evals_override=fallback_max_evals,
                        alpha_cap_override=prepared.alpha_cap,
                        allow_nelec_projection=(
                            not fixed_direction_projection),
                        nelec_limit_override=nelec_limit)
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
            line_search = self._decorate_direction_projection_result(
                state, new_state, line_search, prepared)
            self._verify_accepted_step(
                state, new_state, direction, line_search, dphi0)
            beta = 0.0
            if (restarted or line_search.force_restart or
                    prepared.reset_memory):
                if prepared.reset_memory:
                    restart_reason = (
                        restart_reason or
                        'memory reset after fixed-direction occupation '
                        'projection')
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
            if canonical_prefix:
                return self._start_canonical_continuation_from_prefix(
                    state, niter, trigger='maximum-cycles')
            if self._meets_convergence(state, previous):
                consecutive += 1
                if consecutive >= self.config.required_consecutive_conv:
                    converged, message = True, 'converged at maximum cycle'
        density_change = (0.0 if previous is None else self._metrics(state, previous)[2])
        return self._finalize(state, converged, message, niter, density_change)

    def _kernel_lbfgs(self, dm0: Any = None,
                      h0: Any = None) -> GrandCanonicalResult:
        """Run safeguarded exact-gradient limited-memory BFGS."""
        self.history = []
        self.nfev = 0
        self._reset_run_diagnostics()
        state = self.evaluate(self._initial_h(dm0, h0))
        return self._kernel_lbfgs_from_state(
            state, None, niter=0, cycle_start=0,
            line_search_method=self.config.line_search_method,
            residual_filter_enabled=True)

    def _kernel_lbfgs_from_state(
            self, state: _GCState, previous: Optional[_GCState], *,
            niter: int, cycle_start: int, line_search_method: str,
            residual_filter_enabled: bool) -> GrandCanonicalResult:
        """Continue from an evaluated state with fresh L-BFGS memory."""
        line_search_method = self._canonical_line_search_method(
            line_search_method)
        if not isinstance(residual_filter_enabled, bool):
            raise TypeError('residual_filter_enabled must be boolean')
        lbfgs_history: list[_LBFGSPair] = []
        self._lbfgs_history = lbfgs_history
        self._last_lbfgs_metric_scale = np.nan
        consecutive = 0
        message = 'maximum cycles reached'
        converged = False

        for cycle in range(cycle_start, self.config.max_cycle):
            if self._meets_convergence(state, previous):
                consecutive += 1
                if (previous is None or
                        consecutive >= self.config.required_consecutive_conv):
                    converged, message = True, 'converged'
                    break
            else:
                consecutive = 0

            if self._should_start_diis(state):
                lbfgs_history.clear()
                self._last_lbfgs_metric_scale = np.nan
                self.log.info(
                    'Switching from L-BFGS to residual DIIS at |F-H|_rms = '
                    '%.6g; L-BFGS history reset', state.residual_rms)
                return self._kernel_diis(
                    state, previous, niter=niter, cycle_start=cycle)

            # At an exact stationary point no further accepted state exists
            # with which to satisfy density-change or consecutive-state
            # criteria.  This is the same terminal condition used by NLCG
            # after it discovers that no downhill direction remains, but it
            # must be checked before constructing a fresh L-BFGS direction so
            # valid history is not cleared merely because the gradient is zero.
            if (state.grad_rms < self.config.conv_tol_grad_rms and
                    state.residual_rms < self.config.conv_tol_residual_rms):
                converged, message = True, 'stationary state'
                break

            direction, used_history, direction_reason = self._lbfgs_direction(
                state, lbfgs_history)
            direction, reset, reset_reason = self._ensure_lbfgs_descent(
                state, direction, used_history)
            if reset:
                lbfgs_history.clear()
                used_history = False
                self._last_lbfgs_metric_scale = np.nan
                direction_reason = reset_reason
            if not self._is_descent(state, direction):
                if (state.grad_rms < self.config.conv_tol_grad_rms and
                        state.residual_rms < self.config.conv_tol_residual_rms):
                    converged, message = True, 'stationary initial state'
                else:
                    message = 'persistent loss of descent'
                break

            if self._alpha_cap(direction) < self.config.line_search_alpha_min:
                lbfgs_history.clear()
                direction, restart_reason = self._restart_direction(state)
                used_history = False
                self._last_lbfgs_metric_scale = np.nan
                direction_reason = 'step cap restart; ' + restart_reason
                if (not self._is_descent(state, direction) or
                        self._alpha_cap(direction) <
                        self.config.line_search_alpha_min):
                    message = 'stagnation: step cap below minimum'
                    break

            dphi0 = self.inner(state.gradient, direction)
            descent_cosine = self._descent_cosine(state, direction)
            line_search = self._line_search(
                state, direction, c2=self.config.lbfgs_line_search_c2,
                alpha_init=1.0,
                alpha_cap_override=(
                    1.0 if (used_history and
                            self.config.lbfgs_cap_unit_step_with_history)
                    else None),
                method_override=line_search_method,
                residual_filter_enabled=residual_filter_enabled)
            fallback_used = False
            if not line_search.success:
                primary_line_search = line_search
                lbfgs_history.clear()
                direction, restart_reason = self._restart_direction(state)
                used_history = False
                self._last_lbfgs_metric_scale = np.nan
                dphi0 = self.inner(state.gradient, direction)
                descent_cosine = self._descent_cosine(state, direction)
                fallback_line_search = self._armijo_fallback(
                    state, direction,
                    residual_filter_enabled=residual_filter_enabled)
                line_search = self._combine_line_search_work(
                    primary_line_search, fallback_line_search)
                fallback_used = True
                direction_reason = restart_reason + '; ' + line_search.message
            if not line_search.success or line_search.state is None:
                message = 'line-search failure: ' + line_search.message
                break

            new_state = line_search.state
            self._verify_accepted_step(
                state, new_state, direction, line_search, dphi0)
            pair_info = self._update_lbfgs_history(
                lbfgs_history, state, new_state, line_search, fallback_used)
            if used_history:
                direction_source = 'lbfgs'
            elif 'blended' in direction_reason:
                direction_source = 'residual/exact-gradient fallback'
            elif 'exact' in direction_reason:
                direction_source = 'exact-gradient fallback'
            else:
                direction_source = 'residual'
            self._record_lbfgs(
                cycle, state, new_state, line_search, dphi0,
                len(lbfgs_history), direction_source, pair_info,
                direction_reason, descent_cosine)
            niter += 1
            self._checkpoint(new_state, niter)
            previous, state = state, new_state
        else:
            if self._meets_convergence(state, previous):
                consecutive += 1
                if consecutive >= self.config.required_consecutive_conv:
                    converged, message = True, 'converged at maximum cycle'
        density_change = (0.0 if previous is None
                          else self._metrics(state, previous)[2])
        return self._finalize(state, converged, message, niter, density_change)

    def _kernel_diis(self, state: _GCState, previous: Optional[_GCState],
                     niter: int, cycle_start: int) -> GrandCanonicalResult:
        """Polish a locally converged direct-minimization state with DIIS."""
        diis_history: list[_DIISItem] = []
        self._diis_history = diis_history
        converged = False
        message = 'maximum cycles reached during residual-DIIS polishing'
        damping_hint = self.config.diis_initial_damping
        best_residual_rms = state.residual_rms
        restoration_pending = False

        for cycle in range(cycle_start, self.config.max_cycle):
            if state.residual_rms < self.config.conv_tol_residual_rms:
                converged = True
                message = 'converged residual-DIIS fixed point'
                break
            self._append_diis_item(diis_history, state)
            starting_damping = damping_hint
            residual_target_rms = (
                best_residual_rms *
                (1.0 - self.config.diis_min_residual_reduction)
                if restoration_pending else None)
            (step, condition, coefficient_l1, history_action,
             predicted_residual_rms) = self._diis_step(
                 state, diis_history, starting_damping,
                 residual_target_rms=residual_target_rms,
                 allow_restoration=not restoration_pending,
                 best_residual_rms=best_residual_rms)
            if not step.success or step.state is None:
                message = step.message
                break
            new_state = step.state
            used_restoration = new_state.residual_rms >= best_residual_rms
            if used_restoration:
                history_action = ((history_action + '; ')
                                  if history_action else '') + (
                    'accepted bounded nonmonotone restoration')
                restoration_pending = True
            else:
                best_residual_rms = new_state.residual_rms
                restoration_pending = False
            damping_hint, trust_ratio = self._next_diis_damping(
                state, new_state, predicted_residual_rms, step.alpha,
                starting_damping)
            self._record_diis(
                cycle, state, new_state, step, len(diis_history),
                condition, coefficient_l1, history_action,
                predicted_residual_rms, trust_ratio, damping_hint)
            niter += 1
            self._checkpoint(new_state, niter)
            previous, state = state, new_state
        else:
            if state.residual_rms < self.config.conv_tol_residual_rms:
                converged = True
                message = 'converged residual-DIIS fixed point at maximum cycle'
        density_change = (0.0 if previous is None
                          else self._metrics(state, previous)[2])
        return self._finalize(state, converged, message, niter, density_change)

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
        self.mf.nelec_projection_attempts_gc = (
            self.nnelec_projection_attempts)
        self.mf.nelec_projection_acceptances_gc = (
            self.nnelec_projection_acceptances)
        self.mf.nelec_projection_fallbacks_gc = (
            self.nnelec_projection_fallbacks)
        self.mf.max_raw_delta_nelec_gc = self.max_raw_delta_nelec
        self.mf.max_projected_delta_nelec_gc = (
            self.max_projected_delta_nelec)
        self.mf.max_nelec_projection_correction_gc = (
            self.max_nelec_projection_correction)
        self.mf.final_nelec_trust_radius_gc = self._nelec_trust_radius
        self.mf.last_nelec_trust_ratio_gc = self.last_nelec_trust_ratio
        self.mf.cheap_nelec_evaluations_gc = (
            self.ncheap_nelec_evaluations)
        self.mf.cheap_nelec_alpha_reductions_gc = (
            self.ncheap_nelec_alpha_reductions)
        self.mf.direction_projection_attempts_gc = (
            self.ndirection_projection_attempts)
        self.mf.direction_projection_acceptances_gc = (
            self.ndirection_projection_acceptances)
        self.mf.direction_projection_fallbacks_gc = (
            self.ndirection_projection_fallbacks)
        self.mf.max_direction_projection_correction_gc = (
            self.max_direction_projection_correction)
        self.mf.residual_filter_acceptances_gc = (
            self.nresidual_filter_acceptances)
        self.mf.residual_filter_rejections_gc = (
            self.nresidual_filter_rejections)
        self.mf.lbfgs_switches_gc = self.nlbfgs_switches
        self.mf.lbfgs_switch_cycle_gc = self.lbfgs_switch_cycle
        self.mf.lbfgs_switch_nfev_gc = self.lbfgs_switch_nfev
        self.mf.lbfgs_switch_residual_rms_gc = (
            self.lbfgs_switch_actual_residual_rms)
        self.mf.canonical_precondition_iterations_gc = (
            self.canonical_precondition_iterations)
        self.mf.canonical_precondition_evaluations_gc = (
            self.canonical_precondition_evaluations)
        self.mf.canonical_precondition_residual_rms_gc = (
            self.canonical_precondition_residual_rms)
        self.mf.canonical_precondition_canonical_residual_rms_gc = (
            self.canonical_precondition_canonical_residual_rms)
        self.mf.canonical_precondition_delta_nelec_gc = (
            self.canonical_precondition_delta_nelec)
        self.mf.canonical_precondition_electron_number_gc = (
            self.canonical_precondition_electron_number)
        self.mf.canonical_precondition_mu_proxy_gc = (
            self.canonical_precondition_mu_proxy)
        self.mf.canonical_precondition_trigger_gc = (
            self.canonical_precondition_trigger)
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
            'nelec_projection_attempts_gc': (
                self.nnelec_projection_attempts),
            'nelec_projection_acceptances_gc': (
                self.nnelec_projection_acceptances),
            'nelec_projection_fallbacks_gc': (
                self.nnelec_projection_fallbacks),
            'max_raw_delta_nelec_gc': self.max_raw_delta_nelec,
            'max_projected_delta_nelec_gc': (
                self.max_projected_delta_nelec),
            'max_nelec_projection_correction_gc': (
                self.max_nelec_projection_correction),
            'final_nelec_trust_radius_gc': self._nelec_trust_radius,
            'last_nelec_trust_ratio_gc': self.last_nelec_trust_ratio,
            'cheap_nelec_evaluations_gc': (
                self.ncheap_nelec_evaluations),
            'cheap_nelec_alpha_reductions_gc': (
                self.ncheap_nelec_alpha_reductions),
            'direction_projection_attempts_gc': (
                self.ndirection_projection_attempts),
            'direction_projection_acceptances_gc': (
                self.ndirection_projection_acceptances),
            'direction_projection_fallbacks_gc': (
                self.ndirection_projection_fallbacks),
            'max_direction_projection_correction_gc': (
                self.max_direction_projection_correction),
            'residual_filter_acceptances_gc': (
                self.nresidual_filter_acceptances),
            'residual_filter_rejections_gc': (
                self.nresidual_filter_rejections),
            'lbfgs_switches_gc': self.nlbfgs_switches,
            'lbfgs_switch_cycle_gc': self.lbfgs_switch_cycle,
            'lbfgs_switch_nfev_gc': self.lbfgs_switch_nfev,
            'lbfgs_switch_residual_rms_gc': (
                self.lbfgs_switch_actual_residual_rms),
            'canonical_precondition_iterations': (
                self.canonical_precondition_iterations),
            'canonical_precondition_evaluations': (
                self.canonical_precondition_evaluations),
            'canonical_precondition_residual_rms': (
                self.canonical_precondition_residual_rms),
            'canonical_precondition_canonical_residual_rms': (
                self.canonical_precondition_canonical_residual_rms),
            'canonical_precondition_delta_nelec': (
                self.canonical_precondition_delta_nelec),
            'canonical_precondition_electron_number': (
                self.canonical_precondition_electron_number),
            'canonical_precondition_mu_proxy': (
                self.canonical_precondition_mu_proxy),
            'canonical_precondition_trigger': (
                self.canonical_precondition_trigger),
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
            nelec_projection_attempts=self.nnelec_projection_attempts,
            nelec_projection_acceptances=(
                self.nnelec_projection_acceptances),
            nelec_projection_fallbacks=self.nnelec_projection_fallbacks,
            max_raw_delta_nelec=self.max_raw_delta_nelec,
            max_projected_delta_nelec=self.max_projected_delta_nelec,
            max_nelec_projection_correction=(
                self.max_nelec_projection_correction),
            final_nelec_trust_radius=self._nelec_trust_radius,
            last_nelec_trust_ratio=self.last_nelec_trust_ratio,
            cheap_nelec_evaluations=self.ncheap_nelec_evaluations,
            cheap_nelec_alpha_reductions=(
                self.ncheap_nelec_alpha_reductions),
            direction_projection_attempts=(
                self.ndirection_projection_attempts),
            direction_projection_acceptances=(
                self.ndirection_projection_acceptances),
            direction_projection_fallbacks=(
                self.ndirection_projection_fallbacks),
            max_direction_projection_correction=(
                self.max_direction_projection_correction),
            residual_filter_acceptances=(
                self.nresidual_filter_acceptances),
            residual_filter_rejections=(
                self.nresidual_filter_rejections),
            lbfgs_switches=self.nlbfgs_switches,
            lbfgs_switch_cycle=self.lbfgs_switch_cycle,
            lbfgs_switch_nfev=self.lbfgs_switch_nfev,
            lbfgs_switch_actual_residual_rms=(
                self.lbfgs_switch_actual_residual_rms),
            canonical_precondition_iterations=(
                self.canonical_precondition_iterations),
            canonical_precondition_evaluations=(
                self.canonical_precondition_evaluations),
            canonical_precondition_residual_rms=(
                self.canonical_precondition_residual_rms),
            canonical_precondition_canonical_residual_rms=(
                self.canonical_precondition_canonical_residual_rms),
            canonical_precondition_delta_nelec=(
                self.canonical_precondition_delta_nelec),
            canonical_precondition_electron_number=(
                self.canonical_precondition_electron_number),
            canonical_precondition_mu_proxy=(
                self.canonical_precondition_mu_proxy),
            canonical_precondition_trigger=(
                self.canonical_precondition_trigger),
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
