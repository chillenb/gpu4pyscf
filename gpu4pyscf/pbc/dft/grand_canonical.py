"""Grand-canonical direct minimisation for periodic restricted Kohn--Sham DFT.

This module deliberately does not hook into the ordinary SCF kernel.  The
optimisation variables are Hermitian matrices in a fixed, orthonormal AO
coordinate system and the supplied :class:`KRKS` object remains the
authoritative evaluator of the DFT functional.
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

    cg_restart_interval: int = 20
    cg_beta_max: float = 5.0
    descent_tolerance: float = 1.0e-12

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
    electronic_energy: float
    nuclear_energy: float
    dft_total_energy: float
    electron_number: float
    entropy: float
    entropy_energy: float
    grand_potential: float
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
    """Minimise the finite-temperature KRKS grand potential at fixed ``mu``.

    The object composes a regular GPU4PySCF ``KRKS`` instance.  It intentionally
    never invokes orbital-rotation or CIAH machinery.
    """

    def __init__(self, mf: Any, mu: float, sigma: float,
                 config: Optional[GrandCanonicalConfig] = None):
        self.mf = mf
        self.mu = _as_float(mu, 'mu')
        self.sigma = _as_float(sigma, 'sigma')
        if self.sigma <= 0.0:
            raise ValueError('sigma must be positive')
        self.beta = 1.0 / self.sigma
        self.config = config or GrandCanonicalConfig()
        self.verbose = (getattr(mf, 'verbose', logger.NOTE)
                        if self.config.verbose is None else self.config.verbose)
        self.log = logger.new_logger(mf, self.verbose)
        self.history: list[IterationRecord] = []
        self.nfev = 0
        self._prepare_fixed_basis_data()

    # ---- fixed basis data and validation ---------------------------------

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

    def _thermal_density(self, h: Sequence) -> tuple[list, list, list, list, cp.ndarray]:
        gamma_blocks, eigvals, eigenvectors, occupations, p_blocks = [], [], [], [], []
        dm = cp.empty((self.nkpts, self.nao, self.nao),
                      dtype=cp.result_type(*[x.dtype for x in h], cp.complex128))
        for k, (hk, x, identity) in enumerate(zip(h, self.x_ao2orth, self.identity)):
            gamma = self.beta * (hk - self.mu * identity)
            gamma = 0.5 * (gamma + gamma.conj().T)
            value, vector = cp.linalg.eigh(gamma)
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
        return gamma_blocks, eigvals, eigenvectors, occupations, p_blocks, dm

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

    def _exact_gradient(self, h: Sequence, fock: Sequence, eigenvalues: Sequence,
                        eigenvectors: Sequence, occupations: Sequence) -> list:
        gradient = []
        for hk, fk, gamma, vector, q in zip(h, fock, eigenvalues, eigenvectors, occupations):
            a_tilde = vector.conj().T @ (fk - hk) @ vector
            divided = fermi_divided_difference(gamma, q, self.config.fermi_divdiff_rtol)
            value = 2.0 * self.beta * vector @ (divided * a_tilde) @ vector.conj().T
            gradient.append(0.5 * (value + value.conj().T))
        gradient, _ = self.project_time_reversal(gradient)
        return self.hermitize_blocks(gradient)

    def evaluate(self, h_orth: Sequence) -> _GCState:
        """Fully evaluate a density, DFT energy, exact gradient, and residual."""
        self.nfev += 1
        h = self._sanitize_h(h_orth)
        gamma, eigenvalues, vector, q, p, dm = self._thermal_density(h)
        nelec = self._electron_number(p, dm)
        # Keep the tagged potential returned by get_veff alive and pass that
        # exact object to energy_elec.  Converting it for Fock construction does
        # not mutate or replace the tagged object.
        veff = self.mf.get_veff(self.mf.cell, dm, dm_last=None, vhf_last=None,
                                hermi=1, kpts=self.mf.kpts, kpts_band=None)
        fock_ao = cp.stack([hcore + cp.asarray(veff)[k]
                            for k, hcore in enumerate(self.hcore_ao)])
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
        omega = dft_total_energy - self.mu * nelec + entropy_energy
        if not np.isfinite(omega):
            raise FloatingPointError('grand potential is nonfinite')
        gradient = self._exact_gradient(h, fock, eigenvalues, vector, q)
        z = self.hermitize_blocks([0.5 * (hk - fk) for hk, fk in zip(h, fock)])
        residual = self.scale_blocks(-1.0, z)
        descent = self.inner(gradient, residual)
        if descent > 1.0e-10 * max(1.0, self.norm(gradient) * self.norm(residual)):
            raise RuntimeError('exact gradient and residual have inconsistent descent signs')
        return _GCState(
            h, gamma, eigenvalues, vector, q, p, dm, veff, fock_ao, fock,
            electronic_energy, nuclear_energy, dft_total_energy, nelec, entropy,
            entropy_energy, omega, gradient, z, residual, self.rms(gradient),
            self.rms([fk - hk for hk, fk in zip(h, fock)]))

    # ---- initialisation and checkpointing ---------------------------------

    def _initial_h_from_dm(self, dm: Any) -> list:
        dm_blocks = self.hermitize_blocks(_blocks(dm, 'initial density'))
        if len(dm_blocks) != self.nkpts:
            raise ValueError('initial density and k-point counts differ')
        dm_blocks, _ = self.project_time_reversal(dm_blocks)
        dm_stack = cp.stack(self.hermitize_blocks(dm_blocks))
        veff = self.mf.get_veff(self.mf.cell, dm_stack, dm_last=None, vhf_last=None,
                                hermi=1, kpts=self.mf.kpts, kpts_band=None)
        fock = self._to_orth(cp.stack([hcore + cp.asarray(veff)[k]
                                       for k, hcore in enumerate(self.hcore_ao)]))
        return self._sanitize_h(fock)

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
        return self._initial_h_from_dm(dm0)

    def _checkpoint(self, state: _GCState, cycle: int) -> None:
        filename = self.config.checkpoint_path
        if not filename or self.config.checkpoint_interval <= 0:
            return
        if cycle % self.config.checkpoint_interval:
            return
        values = {f'h_{k}': h.get() for k, h in enumerate(state.h_orth)}
        values.update({
            'mu': self.mu, 'sigma': self.sigma, 'kpts': self.kpts,
            'weights': cp.asnumpy(self.weights), 'ranks': np.asarray(self.north),
            'x_dims': np.asarray([x.shape for x in self.x_ao2orth]),
            'fingerprint': self._checkpoint_fingerprint,
            'grand_potential': state.grand_potential,
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

    def _ensure_descent(self, state: _GCState, direction: Sequence) -> tuple[list, bool, str]:
        direction = self.hermitize_blocks(direction)
        direction, projected_change = self.project_time_reversal(direction)
        if projected_change > self.config.hermiticity_tol * max(1.0, self.norm(direction)):
            direction = self.copy_blocks(state.residual)
            return direction, True, 'time-reversal projection changed CG direction'
        if self._is_descent(state, direction):
            return direction, False, ''
        direction = self.copy_blocks(state.residual)
        if self._is_descent(state, direction):
            return direction, True, 'lost descent; restarted CG'
        return direction, True, 'residual is not a descent direction'

    def _alpha_cap(self, direction: Sequence) -> float:
        block_rms = self.max_block_rms(direction)
        if block_rms == 0.0:
            return 0.0
        return min(self.config.line_search_alpha_cap,
                   self.config.line_search_max_h_rms_step / block_rms)

    def _trial(self, state: _GCState, direction: Sequence, alpha: float) -> Optional[_GCState]:
        try:
            candidate = self.axpy(alpha, direction, state.h_orth)
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
              nfev: int) -> _LineSearchResult:
        c1, c2 = self.config.line_search_c1, self.config.line_search_c2
        lo_phi = lo_state.grand_potential
        lo_dphi = self.inner(lo_state.gradient, direction)
        hi_phi = np.inf if hi_state is None else hi_state.grand_potential
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
            phi = np.inf if trial is None else trial.grand_potential
            if trial is None or phi > phi0 + c1 * alpha * dphi0 or phi >= lo_phi:
                hi_a, hi_state, hi_phi, hi_dphi = alpha, trial, phi, np.nan if trial is None else self.inner(trial.gradient, direction)
                continue
            dphi = self.inner(trial.gradient, direction)
            if phi <= phi0 + c1 * alpha * dphi0:
                if best is None or phi < best[1].grand_potential:
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

    def _line_search(self, state: _GCState, direction: Sequence) -> _LineSearchResult:
        dphi0 = self.inner(state.gradient, direction)
        if dphi0 >= 0.0:
            return _LineSearchResult(False, None, message='line search called with non-descent direction')
        alpha_max = self._alpha_cap(direction)
        if alpha_max < self.config.line_search_alpha_min:
            return _LineSearchResult(False, None, message='step cap below minimum')
        alpha = min(self.config.line_search_alpha_init, alpha_max)
        phi0 = state.grand_potential
        c1, c2 = self.config.line_search_c1, self.config.line_search_c2
        previous_alpha, previous_state = 0.0, state
        best: Optional[tuple[float, _GCState]] = None
        nfev = 0
        for it in range(self.config.line_search_max_evals):
            trial = self._trial(state, direction, alpha)
            nfev += 1
            phi = np.inf if trial is None else trial.grand_potential
            armijo = trial is not None and phi <= phi0 + c1 * alpha * dphi0
            if armijo and (best is None or phi < best[1].grand_potential):
                best = (alpha, trial)
            if trial is None or not armijo or (it > 0 and phi >= previous_state.grand_potential):
                return self._zoom(state, direction, phi0, dphi0,
                                  previous_alpha, previous_state, alpha, trial, best, nfev)
            dphi = self.inner(trial.gradient, direction)
            if abs(dphi) <= c2 * abs(dphi0):
                return _LineSearchResult(True, trial, alpha, nfev, True, False, 'strong Wolfe')
            if dphi >= 0.0:
                return self._zoom(state, direction, phi0, dphi0,
                                  alpha, trial, previous_alpha, previous_state, best, nfev)
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
            if trial is not None and trial.grand_potential <= state.grand_potential + self.config.line_search_c1 * alpha * dphi0:
                return _LineSearchResult(True, trial, alpha, nfev, False, True,
                                         'monotone Armijo fallback')
            alpha *= self.config.armijo_backtrack_factor
            if alpha < self.config.line_search_alpha_min:
                break
        return _LineSearchResult(False, None, nfev=nfev, message='fallback found no Armijo point')

    def _metrics(self, state: _GCState, previous: Optional[_GCState]) -> tuple[float, float, float, float]:
        if previous is None:
            return np.inf, np.inf, np.inf, np.inf
        delta_omega = state.grand_potential - previous.grand_potential
        delta_nelec = abs(state.electron_number - previous.electron_number)
        density_change = self.rms(self.axpy(-1.0, previous.p_orth, state.p_orth))
        return delta_omega, delta_nelec, density_change, abs(delta_omega)

    def _meets_convergence(self, state: _GCState, previous: Optional[_GCState]) -> bool:
        if previous is None:
            return state.grad_rms < self.config.conv_tol_grad_rms and state.residual_rms < self.config.conv_tol_residual_rms
        delta_omega, delta_nelec, density_change, abs_delta_omega = self._metrics(state, previous)
        return (abs_delta_omega < self.config.conv_tol_omega and
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
        if accepted.grand_potential > state.grand_potential + self.config.line_search_c1 * alpha * dphi0 + 1.0e-12:
            raise RuntimeError('accepted line-search point does not satisfy Armijo decrease')

    def _record(self, cycle: int, old: _GCState, new: _GCState, line_search: _LineSearchResult,
                dphi0: float, beta: float, restart_reason: str) -> None:
        delta_omega, delta_nelec, density_change, _ = self._metrics(new, old)
        self.history.append(IterationRecord(
            cycle, new.grand_potential, new.dft_total_energy, -self.mu * new.electron_number,
            new.entropy_energy, new.electron_number, delta_omega, delta_nelec,
            new.grad_rms, new.residual_rms, density_change, line_search.alpha, dphi0,
            beta, restart_reason, line_search.nfev))
        self.log.info('GC cycle %d  Omega = %.12g  E_DFT = %.12g  N = %.10g  '
                      '|g|_rms = %.3g  |F-H|_rms = %.3g  alpha = %.3g',
                      cycle, new.grand_potential, new.dft_total_energy,
                      new.electron_number, new.grad_rms, new.residual_rms,
                      line_search.alpha)

    def kernel(self, dm0: Any = None, h0: Any = None) -> GrandCanonicalResult:
        """Run safeguarded nonlinear-CG direct minimisation."""
        self.history = []
        self.nfev = 0
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
            direction, restarted, restart_reason = self._ensure_descent(state, direction)
            if not self._is_descent(state, direction):
                if state.grad_rms < self.config.conv_tol_grad_rms and state.residual_rms < self.config.conv_tol_residual_rms:
                    converged, message = True, 'stationary initial state'
                else:
                    message = 'persistent loss of descent'
                break
            if self._alpha_cap(direction) < self.config.line_search_alpha_min:
                direction = self.copy_blocks(state.residual)
                if self._alpha_cap(direction) < self.config.line_search_alpha_min:
                    message = 'stagnation: step cap below minimum'
                    break
                restarted, restart_reason = True, 'step cap restart'
            dphi0 = self.inner(state.gradient, direction)
            line_search = self._line_search(state, direction)
            if not line_search.success:
                direction = self.copy_blocks(state.residual)
                line_search = self._armijo_fallback(state, direction)
                restarted, restart_reason = True, line_search.message
                dphi0 = self.inner(state.gradient, direction)
            if not line_search.success or line_search.state is None:
                message = 'line-search failure: ' + line_search.message
                break
            new_state = line_search.state
            self._verify_accepted_step(state, new_state, direction, line_search.alpha, dphi0)
            denominator = self.inner(state.gradient, state.z)
            numerator = self.inner(new_state.gradient, new_state.z)
            beta = 0.0
            if force_restart or restarted or line_search.force_restart:
                restart_reason = restart_reason or line_search.message
            elif cycle > 0 and cycle % self.config.cg_restart_interval != 0:
                if denominator > 1.0e-30 and np.isfinite(denominator) and np.isfinite(numerator):
                    candidate = numerator / denominator
                    if 0.0 <= candidate <= self.config.cg_beta_max:
                        beta = candidate
                    else:
                        restart_reason = 'invalid Fletcher-Reeves beta'
                else:
                    restart_reason = 'ill-conditioned Fletcher-Reeves denominator'
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

    # ---- public state finalisation ----------------------------------------

    def _finalize(self, state: _GCState, converged: bool, message: str,
                  niter: int, density_change: float) -> GrandCanonicalResult:
        coeff = [x @ u for x, u in zip(self.x_ao2orth, state.u)]
        energy = [self.mu + value / self.beta for value in state.eigenvalues]
        occ = [2.0 * q for q in state.occupations]
        mo_coeff = _stack_or_list(coeff)
        mo_energy = _stack_or_list(energy)
        mo_occ = _stack_or_list(occ)
        self.mf.converged = converged
        self.mf.mo_coeff = mo_coeff
        self.mf.mo_energy = mo_energy
        self.mf.mo_occ = mo_occ
        self.mf.e_tot = state.dft_total_energy
        self.mf.grand_potential = state.grand_potential
        self.mf.electron_number_gc = state.electron_number
        self.mf.entropy_gc = state.entropy
        self.mf.entropy_energy_gc = state.entropy_energy
        self.mf.mu_gc = self.mu
        self.mf.sigma_gc = self.sigma
        self.mf.h_aux_gc = self.copy_blocks(state.h_orth)
        self.mf.dm_gc = state.dm_ao
        if not hasattr(self.mf, 'scf_summary') or self.mf.scf_summary is None:
            self.mf.scf_summary = {}
        self.mf.scf_summary.update({
            'grand_potential': state.grand_potential,
            'electron_number_gc': state.electron_number,
            'entropy_gc': state.entropy,
            'entropy_energy_gc': state.entropy_energy,
            'mu_gc': self.mu,
            'sigma_gc': self.sigma,
        })
        return GrandCanonicalResult(
            converged, message, niter, self.nfev, self.mu, self.sigma, self.beta,
            state.grand_potential, state.dft_total_energy, state.electronic_energy,
            state.nuclear_energy, state.entropy, state.entropy_energy,
            state.electron_number, self.copy_blocks(state.h_orth),
            self.copy_blocks(state.fock_orth), state.dm_ao, self.copy_blocks(state.p_orth),
            self.copy_blocks(state.occupations), mo_coeff, mo_occ, mo_energy,
            state.grad_rms, state.residual_rms, density_change, list(self.history),
            state.veff, self.config.checkpoint_path)
