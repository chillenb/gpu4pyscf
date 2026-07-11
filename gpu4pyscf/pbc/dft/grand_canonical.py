"""Finite-temperature direct minimisation for periodic restricted Kohn--Sham DFT.

This module deliberately does not hook into the ordinary SCF kernel.  The
optimisation variables are Hermitian matrices in a fixed, orthonormal AO
coordinate system, the electron ensemble may be fixed-mu or fixed-N, and the
supplied :class:`KRKS` object remains the authoritative evaluator of the DFT
functional.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    diis_max_backtracks: int = 8
    diis_min_residual_reduction: float = 1.0e-3
    diis_max_objective_increase: float = 1.0e-5
    diis_max_delta_nelec: float = 5.0e-2

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
                 electron_number: Optional[float] = None):
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
        self._validate_lbfgs_config()
        self._validate_diis_config()
        self._validate_nelec_guard_config()
        self.verbose = (getattr(mf, 'verbose', logger.NOTE)
                        if self.config.verbose is None else self.config.verbose)
        self.log = logger.new_logger(mf, self.verbose)
        if (self.config.optimizer == 'lbfgs' and self.fixed_electron_number and
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
        self._prepare_fixed_basis_data()
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
        }
        try:
            return aliases[key]
        except KeyError as error:
            choices = ', '.join(('fletcher-reeves', 'polak-ribiere',
                                 'hestenes-stiefel'))
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

    def _validate_lbfgs_config(self) -> None:
        if (not isinstance(self.config.lbfgs_history_size, int) or
                isinstance(self.config.lbfgs_history_size, bool) or
                self.config.lbfgs_history_size < 0):
            raise ValueError('lbfgs_history_size must be a nonnegative integer')
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
                     'lbfgs_clear_on_non_wolfe'):
            if not isinstance(getattr(self.config, name), bool):
                raise TypeError(f'{name} must be boolean')

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
        for name, minimum in (('diis_space', 2), ('diis_max_backtracks', 0)):
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
        hcore = _stack_or_list(self.hcore_ao)
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
        energy_parameters = inspect.signature(self.mf.energy_elec).parameters
        vhf_keyword = 'vhf' if ('vhf' in energy_parameters or
                                any(p.kind == inspect.Parameter.VAR_KEYWORD
                                    for p in energy_parameters.values())) else 'vhf_kpts'
        electronic_energy, _ = self.mf.energy_elec(
            dm_kpts=dm, h1e_kpts=_stack_or_list(self.hcore_ao),
            **{vhf_keyword: veff})
        electronic_energy = _as_float(electronic_energy, 'electronic energy')
        nuclear_energy = _as_float(self.mf.energy_nuc(), 'nuclear energy')
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

    def _checkpoint(self, state: _GCState, cycle: int) -> None:
        filename = self.config.checkpoint_path
        if not filename or self.config.checkpoint_interval <= 0:
            return
        if cycle % self.config.checkpoint_interval:
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
            residual, blended = self._blend_poorly_aligned_direction(
                state, residual)
            if blended:
                return residual, 'restarted with blended residual/exact gradient'
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
        if state.residual_rms <= self.config.exact_gradient_polish_residual_rms:
            gradient = self.scale_blocks(-1.0, state.gradient)
            if self._is_descent(state, gradient):
                return gradient, True, 'exact-gradient final polishing'
        if self._is_descent(state, direction):
            direction, blended = self._blend_poorly_aligned_direction(
                state, direction)
            if blended:
                return direction, True, 'blended poorly aligned direction with exact gradient'
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
        valid_denominator = (abs(denominator) > 1.0e-30 if update == 'hestenes-stiefel'
                             else denominator > 1.0e-30)
        if not finite or not valid_denominator:
            return 0.0, f'ill-conditioned {label} denominator'
        candidate = numerator / denominator
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
        non_wolfe = fallback_used or not line_search.strong_wolfe
        if line_search.force_restart or non_wolfe:
            if line_search.force_restart or self.config.lbfgs_clear_on_non_wolfe:
                history.clear()
                info['action'] = 'history cleared after non-Wolfe acceptance'
            else:
                info['action'] = 'pair skipped after non-Wolfe acceptance'
            return info

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
            history.clear()
            info['action'] = 'history cleared after bad curvature'
            return info
        if (self.rms(s) < self.config.lbfgs_min_pair_step_rms or
                s_norm == 0.0 or y_norm == 0.0 or
                sy < self.config.lbfgs_curvature_tol * s_norm * y_norm):
            info['action'] = 'pair skipped: weak curvature'
            return info
        if self.config.lbfgs_history_size == 0:
            info['action'] = 'pair skipped: history capacity is zero'
            return info

        pair = _LBFGSPair(
            self.copy_blocks(s), self.copy_blocks(y), float(1.0 / sy),
            float(sy), float(s_norm), float(y_norm),
            float(info['curvature_cosine']))
        history.append(pair)
        if len(history) > self.config.lbfgs_history_size:
            del history[0]
            info['action'] = 'pair added; oldest pair evicted'
        else:
            info['action'] = 'pair added'
        info['pair_added'] = True
        return info

    # ---- residual DIIS ---------------------------------------------------

    def _should_start_diis(self, state: _GCState) -> bool:
        threshold = self.config.diis_switch_residual_rms
        return threshold is not None and state.residual_rms <= threshold

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

    def _diis_trial_acceptable(self, state: _GCState,
                               trial: _GCState) -> tuple[bool, str]:
        residual_limit = state.residual_rms * (
            1.0 - self.config.diis_min_residual_reduction)
        if not trial.residual_rms < residual_limit:
            return False, 'residual did not decrease sufficiently'
        objective_increase = trial.objective - state.objective
        if objective_increase > self.config.diis_max_objective_increase:
            return False, 'objective increase exceeded DIIS noise allowance'
        if (not self.fixed_electron_number and
                abs(trial.electron_number - state.electron_number) >
                self.config.diis_max_delta_nelec):
            return False, 'electron-number change exceeded DIIS safeguard'
        return True, ''

    def _try_diis_target(self, state: _GCState,
                         target: Sequence) -> tuple[Optional[_GCState],
                                                    float, str]:
        direction = self.axpy(-1.0, state.h_orth, target)
        if not self.all_finite(direction) or self.norm(direction) == 0.0:
            return None, 0.0, 'zero or nonfinite DIIS direction'
        damping = 1.0
        last_reason = 'no DIIS trial evaluated'
        for _ in range(self.config.diis_max_backtracks + 1):
            trial = self._trial(state, direction, damping)
            if trial is not None:
                acceptable, last_reason = self._diis_trial_acceptable(
                    state, trial)
                if acceptable:
                    return trial, damping, ''
            else:
                last_reason = 'DIIS trial evaluation failed'
            damping *= self.config.diis_backtrack_factor
        return None, 0.0, last_reason

    def _diis_step(
            self, state: _GCState,
            history: list[_DIISItem]) -> tuple[_LineSearchResult, float,
                                                float, str]:
        start_nfev = self.nfev
        coefficients, condition, coefficient_l1, action = (
            self._diis_coefficients(history))
        target = self._diis_target(history, coefficients)
        trial, damping, rejection = self._try_diis_target(state, target)
        if trial is None and len(history) > 1:
            latest = history[-1]
            history[:] = [latest]
            action = ((action + '; ') if action else '') + (
                'cleared DIIS history after rejected extrapolation')
            target = self.copy_blocks(latest.fock)
            coefficients = np.ones(1)
            coefficient_l1 = 1.0
            trial, damping, rejection = self._try_diis_target(state, target)
        nfev = self.nfev - start_nfev
        if trial is None:
            message = 'residual-DIIS failed: ' + rejection
            return (_LineSearchResult(False, None, nfev=nfev,
                                      message=message),
                    condition, coefficient_l1, action)
        message = 'residual-DIIS accepted'
        if damping < 1.0:
            message += f' with damping {damping:.6g}'
        return (_LineSearchResult(True, trial, damping, nfev,
                                  False, False, message),
                condition, coefficient_l1, action)

    def _alpha_cap(self, direction: Sequence) -> float:
        block_rms = self.max_block_rms(direction)
        if block_rms == 0.0:
            return 0.0
        return min(self.config.line_search_alpha_cap,
                   self.config.line_search_max_h_rms_step / block_rms)

    def _cheap_fixed_mu_electron_number(self, h_orth: Sequence) -> float:
        """Evaluate N(H) without constructing a density or building a Fock matrix."""
        if self.fixed_electron_number:
            return self.target_electron_number
        return 2.0 * sum(
            float((self.weights[k] * cp.sum(fermi_occupations(
                self.beta * (cp.linalg.eigvalsh(hk) - self.mu)))).item())
            for k, hk in enumerate(h_orth))

    def _reject_trial_by_electron_number(
            self, state: _GCState, candidate: Sequence) -> tuple[bool, float]:
        threshold = self.config.line_search_nelec_guard_residual_rms
        if self.fixed_electron_number:
            return False, state.electron_number
        electron_number = self._cheap_fixed_mu_electron_number(candidate)
        maximum = self.config.line_search_max_delta_nelec
        if threshold is not None and state.residual_rms <= threshold:
            maximum = min(
                maximum,
                self.config.line_search_nelec_guard_max_delta_nelec)
        rejected = (abs(electron_number - state.electron_number) >
                    maximum)
        return rejected, electron_number

    def _trial(self, state: _GCState, direction: Sequence, alpha: float) -> Optional[_GCState]:
        try:
            candidate = self._sanitize_h(
                self.axpy(alpha, direction, state.h_orth))
            rejected, electron_number = self._reject_trial_by_electron_number(
                state, candidate)
            if rejected:
                self.ncheap_nelec_reject += 1
                self.log.debug(
                    'Rejected trial before Fock build: alpha = %.6g, '
                    'residual RMS = %.6g, N = %.12g -> %.12g',
                    alpha, state.residual_rms, state.electron_number,
                    electron_number)
                return None
            return self.evaluate(candidate)
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

    def _zoom(self, state0: _GCState, direction: Sequence, phi0: float, dphi0: float,
              lo_a: float, lo_state: _GCState, hi_a: float,
              hi_state: Optional[_GCState], best: Optional[tuple[float, _GCState]],
              nfev: int, c2: Optional[float] = None) -> _LineSearchResult:
        c1 = self.config.line_search_c1
        c2 = self.config.line_search_c2 if c2 is None else c2
        lo_phi = lo_state.objective
        lo_dphi = self.inner(lo_state.gradient, direction)
        hi_phi = np.inf if hi_state is None else hi_state.objective
        hi_dphi = np.nan if hi_state is None else self.inner(hi_state.gradient, direction)
        for _ in range(self.config.line_search_zoom_evals):
            lower, upper = min(lo_a, hi_a), max(lo_a, hi_a)
            alpha = self._cubic_minimizer(lo_a, lo_phi, lo_dphi, hi_a, hi_phi, hi_dphi)
            margin = 0.1 * (upper - lower)
            if alpha is None or not (lower + margin < alpha < upper - margin):
                alpha = 0.5 * (lo_a + hi_a)
            if abs(hi_a - lo_a) < self.config.line_search_alpha_min:
                break
            trial = self._trial(state0, direction, alpha)
            nfev += 1
            phi = np.inf if trial is None else trial.objective
            if trial is None or phi > phi0 + c1 * alpha * dphi0 or phi >= lo_phi:
                hi_a, hi_state, hi_phi, hi_dphi = alpha, trial, phi, np.nan if trial is None else self.inner(trial.gradient, direction)
                continue
            dphi = self.inner(trial.gradient, direction)
            if phi <= phi0 + c1 * alpha * dphi0:
                if best is None or phi < best[1].objective:
                    best = (alpha, trial)
            if abs(dphi) <= c2 * abs(dphi0):
                return _LineSearchResult(True, trial, alpha, nfev, True, False, 'strong Wolfe')
            if dphi * (hi_a - lo_a) >= 0.0:
                hi_a, hi_state, hi_phi, hi_dphi = lo_a, lo_state, lo_phi, lo_dphi
            lo_a, lo_state, lo_phi, lo_dphi = alpha, trial, phi, dphi
        if best is not None:
            return _LineSearchResult(True, best[1], best[0], nfev, False, True,
                                     'accepted best Armijo point after zoom')
        return _LineSearchResult(False, None, nfev=nfev, message='zoom found no Armijo point')

    def _line_search(self, state: _GCState, direction: Sequence,
                     c2: Optional[float] = None,
                     alpha_init: Optional[float] = None,
                     alpha_cap_override: Optional[float] = None) -> _LineSearchResult:
        dphi0 = self.inner(state.gradient, direction)
        if dphi0 >= 0.0:
            return _LineSearchResult(False, None, message='line search called with non-descent direction')
        alpha_max = self._alpha_cap(direction)
        if alpha_cap_override is not None:
            if not np.isfinite(alpha_cap_override) or alpha_cap_override <= 0.0:
                raise ValueError('alpha_cap_override must be finite and positive')
            alpha_max = min(alpha_max, alpha_cap_override)
        if alpha_max < self.config.line_search_alpha_min:
            return _LineSearchResult(False, None, message='step cap below minimum')
        alpha_start = (self.config.line_search_alpha_init
                       if alpha_init is None else alpha_init)
        if not np.isfinite(alpha_start) or alpha_start <= 0.0:
            raise ValueError('line-search initial alpha must be finite and positive')
        alpha = min(alpha_start, alpha_max)
        phi0 = state.objective
        c1 = self.config.line_search_c1
        c2 = self.config.line_search_c2 if c2 is None else c2
        previous_alpha, previous_state = 0.0, state
        best: Optional[tuple[float, _GCState]] = None
        nfev = 0
        for it in range(self.config.line_search_max_evals):
            trial = self._trial(state, direction, alpha)
            nfev += 1
            phi = np.inf if trial is None else trial.objective
            armijo = trial is not None and phi <= phi0 + c1 * alpha * dphi0
            if armijo and (best is None or phi < best[1].objective):
                best = (alpha, trial)
            if trial is None or not armijo or (it > 0 and phi >= previous_state.objective):
                return self._zoom(state, direction, phi0, dphi0,
                                  previous_alpha, previous_state, alpha, trial,
                                  best, nfev, c2)
            dphi = self.inner(trial.gradient, direction)
            if abs(dphi) <= c2 * abs(dphi0):
                return _LineSearchResult(True, trial, alpha, nfev, True, False, 'strong Wolfe')
            if dphi >= 0.0:
                return self._zoom(state, direction, phi0, dphi0,
                                  alpha, trial, previous_alpha, previous_state,
                                  best, nfev, c2)
            previous_alpha, previous_state = alpha, trial
            grown = min(self.config.line_search_growth * alpha, alpha_max)
            if grown <= alpha:
                break
            alpha = grown
        if best is not None:
            return _LineSearchResult(True, best[1], best[0], nfev, False, True,
                                     'accepted best Armijo point')
        return _LineSearchResult(False, None, nfev=nfev, message='no Armijo point')

    def _armijo_fallback(self, state: _GCState, direction: Sequence) -> _LineSearchResult:
        alpha_max = self._alpha_cap(direction)
        if alpha_max < self.config.line_search_alpha_min:
            return _LineSearchResult(False, None, message='fallback step cap below minimum')
        dphi0 = self.inner(state.gradient, direction)
        if dphi0 >= 0.0:
            return _LineSearchResult(False, None, message='fallback residual is not downhill')
        alpha = min(1.0, alpha_max)
        nfev = 0
        for _ in range(self.config.line_search_max_evals):
            trial = self._trial(state, direction, alpha)
            nfev += 1
            if trial is not None and trial.objective <= state.objective + self.config.line_search_c1 * alpha * dphi0:
                return _LineSearchResult(True, trial, alpha, nfev, False, True,
                                         'monotone Armijo fallback')
            alpha *= self.config.armijo_backtrack_factor
            if alpha < self.config.line_search_alpha_min:
                break
        return _LineSearchResult(False, None, nfev=nfev, message='fallback found no Armijo point')

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
                              direction: Sequence, alpha: float, dphi0: float) -> None:
        expected = self._sanitize_h(self.axpy(alpha, direction, state.h_orth))
        mismatch = self.max_block_rms(self.axpy(-1.0, expected, accepted.h_orth))
        if mismatch > 1.0e-8:
            raise RuntimeError(f'accepted state is not the evaluated step (mismatch {mismatch:g})')
        if accepted.objective > state.objective + self.config.line_search_c1 * alpha * dphi0 + 1.0e-12:
            raise RuntimeError('accepted line-search point does not satisfy Armijo decrease')

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
                diis_history_action: str = '') -> None:
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
            diis_damping, diis_history_action))
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
            history_size, direction_source, line_search.strong_wolfe,
            self._last_lbfgs_metric_scale, pair_info['action'])

    def _record_diis(
            self, cycle: int, old: _GCState, new: _GCState,
            step: _LineSearchResult, history_size: int, condition: float,
            coefficient_l1: float, history_action: str) -> None:
        self._record(
            cycle, old, new, step, np.nan, np.nan, history_action,
            optimizer='diis', search_direction_source='residual-diis',
            diis_history_size=history_size, diis_condition=condition,
            diis_coefficient_l1=coefficient_l1,
            diis_damping=step.alpha,
            diis_history_action=history_action)
        self.log.info(
            'DIIS cycle %d  residual %.6g -> %.6g  delta objective = %.3g  '
            'delta N = %.3g  damping = %.3g  history = %d  cond = %.3g  %s',
            cycle, old.residual_rms, new.residual_rms,
            new.objective - old.objective,
            new.electron_number - old.electron_number, step.alpha,
            history_size, condition, history_action)

    def kernel(self, dm0: Any = None, h0: Any = None) -> GrandCanonicalResult:
        """Run the configured safeguarded direct minimizer."""
        if self.config.optimizer == 'nlcg':
            return self._kernel_nlcg(dm0=dm0, h0=h0)
        if self.config.optimizer == 'lbfgs':
            return self._kernel_lbfgs(dm0=dm0, h0=h0)
        raise AssertionError('validated optimizer is unreachable')

    def _kernel_nlcg(self, dm0: Any = None,
                     h0: Any = None) -> GrandCanonicalResult:
        """Run safeguarded fixed-mu or fixed-electron nonlinear CG."""
        self.history = []
        self.nfev = 0
        self.ncheap_nelec_reject = 0
        state = self.evaluate(self._initial_h(dm0, h0))
        previous: Optional[_GCState] = None
        direction = self.copy_blocks(state.residual)
        consecutive = 0
        message = 'maximum cycles reached'
        converged = False
        niter = 0
        force_restart = False

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
            line_search = self._line_search(state, direction)
            if not line_search.success:
                direction, fallback_reason = self._restart_direction(state)
                line_search = self._armijo_fallback(state, direction)
                restarted = True
                restart_reason = fallback_reason + '; ' + line_search.message
                dphi0 = self.inner(state.gradient, direction)
            if not line_search.success or line_search.state is None:
                message = 'line-search failure: ' + line_search.message
                break
            new_state = line_search.state
            self._verify_accepted_step(state, new_state, direction, line_search.alpha, dphi0)
            beta = 0.0
            if force_restart or restarted or line_search.force_restart:
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
            force_restart = line_search.force_restart or lost_descent
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

    def _kernel_lbfgs(self, dm0: Any = None,
                      h0: Any = None) -> GrandCanonicalResult:
        """Run safeguarded exact-gradient limited-memory BFGS."""
        self.history = []
        self.nfev = 0
        self.ncheap_nelec_reject = 0
        state = self.evaluate(self._initial_h(dm0, h0))
        previous: Optional[_GCState] = None
        lbfgs_history: list[_LBFGSPair] = []
        self._lbfgs_history = lbfgs_history
        consecutive = 0
        message = 'maximum cycles reached'
        converged = False
        niter = 0

        for cycle in range(self.config.max_cycle):
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
                    else None))
            fallback_used = False
            if not line_search.success:
                lbfgs_history.clear()
                direction, restart_reason = self._restart_direction(state)
                used_history = False
                self._last_lbfgs_metric_scale = np.nan
                dphi0 = self.inner(state.gradient, direction)
                descent_cosine = self._descent_cosine(state, direction)
                line_search = self._armijo_fallback(state, direction)
                fallback_used = True
                direction_reason = restart_reason + '; ' + line_search.message
            if not line_search.success or line_search.state is None:
                message = 'line-search failure: ' + line_search.message
                break

            new_state = line_search.state
            self._verify_accepted_step(
                state, new_state, direction, line_search.alpha, dphi0)
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

        for cycle in range(cycle_start, self.config.max_cycle):
            if state.residual_rms < self.config.conv_tol_residual_rms:
                converged = True
                message = 'converged residual-DIIS fixed point'
                break
            self._append_diis_item(diis_history, state)
            step, condition, coefficient_l1, history_action = (
                self._diis_step(state, diis_history))
            if not step.success or step.state is None:
                message = step.message
                break
            new_state = step.state
            self._record_diis(
                cycle, state, new_state, step, len(diis_history),
                condition, coefficient_l1, history_action)
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
            self.ncheap_nelec_reject)
