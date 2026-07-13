# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Finite-temperature grand-canonical KRKS for periodic systems.

Fixed-electron-number problems are solved with residual DIIS.  At fixed
chemical potential, a safeguarded secant iteration selects the electron
number of a sequence of fixed-N problems.  This avoids applying DIIS directly
to the discontinuously changing low-temperature occupations.
'''

import numpy as np
import cupy as cp

from pyscf import __config__, lib
from pyscf.scf.addons import _fermi_smearing_occ, _smearing_optimize

from gpu4pyscf.lib import diis, logger

try:
    from cupyx.scipy.special import expit as _expit
except ImportError:  # pragma: no cover
    _expit = None


__all__ = ['GrandCanonicalKRKS']


_DIIS_BACKTRACK = .5
_DIIS_MAX_BACKTRACK = 8
_DIIS_MIN_REDUCTION = 1e-3
_DIIS_TRUST_SHRINK = .25
_DIIS_TRUST_EXPAND = .75
_DIIS_EXPANSION = 2.
_DIIS_EXPAND_REDUCTION = 2e-2
_MIN_DAMP = 1e-8
_INITIAL_NELEC_STEP = 3e-2
_MAX_NELEC_STEP_FRACTION = .1
_ROOT_NELEC_TOL = 1e-8
_VERIFY_RESIDUAL_TOL = 1e-6
_VERIFY_DENSITY_TOL = 1e-9
_HERMITICITY_TOL = 1e-10
_ORTHOGONALITY_TOL = 1e-9


def _as_float(value, name='value'):
    if isinstance(value, cp.ndarray):
        if value.size != 1:
            raise ValueError('%s must be scalar' % name)
        value = value.item()
    value = complex(value)
    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        raise FloatingPointError('%s is nonfinite' % name)
    if abs(value.imag) > 1e-9 * max(1., abs(value.real)):
        raise ValueError('%s has a significant imaginary part: %g' %
                         (name, value.imag))
    return float(value.real)


def _blocks(values, name='matrix blocks'):
    if isinstance(values, cp.ndarray) and values.ndim == 3:
        values = [values[k] for k in range(values.shape[0])]
    elif isinstance(values, (list, tuple)):
        values = list(values)
    else:
        values = cp.asarray(values)
        if values.ndim != 3:
            raise ValueError('%s must be a stack or list of matrices' % name)
        values = [values[k] for k in range(values.shape[0])]
    out = [cp.asarray(x) for x in values]
    if not out or any(x.ndim != 2 for x in out):
        raise ValueError('%s must contain rank-two matrices' % name)
    return out


def _stack_or_list(values):
    if all(x.shape == values[0].shape for x in values):
        return cp.stack(values)
    return list(values)


def _fermi_occ(gamma):
    gamma = cp.asarray(gamma)
    if _expit is not None:
        return _expit(-gamma)
    positive = gamma >= 0
    out = cp.empty_like(gamma, dtype=cp.result_type(gamma, cp.float64))
    exp_neg = cp.exp(-gamma[positive])
    out[positive] = exp_neg / (1. + exp_neg)
    exp_pos = cp.exp(gamma[~positive])
    out[~positive] = 1. / (1. + exp_pos)
    return out


def _fermi_entropy(gamma, occ):
    gamma = cp.asarray(gamma)
    occ = cp.asarray(occ)
    positive = gamma >= 0
    out = cp.empty_like(occ, dtype=cp.result_type(occ, cp.float64))
    out[positive] = (-occ[positive] * gamma[positive]
                     - cp.logaddexp(0., -gamma[positive]))
    out[~positive] = ((1. - occ[~positive]) * gamma[~positive]
                      - cp.logaddexp(0., gamma[~positive]))
    return out


class _State:
    __slots__ = (
        'h', 'eig', 'coeff', 'occ', 'p', 'dm', 'fock',
        'aux_mu', 'mu', 'gauge', 'e_tot', 'nelec',
        'entropy', 'entropy_energy', 'free_energy', 'grand_potential',
        'residual', 'residual_rms')

    def __init__(self, h, eig, coeff, occ, p, dm, fock, aux_mu, mu, gauge,
                 e_tot, nelec, entropy, entropy_energy, free_energy,
                 grand_potential, residual, residual_rms):
        self.h = h
        self.eig = eig
        self.coeff = coeff
        self.occ = occ
        self.p = p
        self.dm = dm
        self.fock = fock
        self.aux_mu = aux_mu
        self.mu = mu
        self.gauge = gauge
        self.e_tot = e_tot
        self.nelec = nelec
        self.entropy = entropy
        self.entropy_energy = entropy_energy
        self.free_energy = free_energy
        self.grand_potential = grand_potential
        self.residual = residual
        self.residual_rms = residual_rms


class _Session:
    __slots__ = ('nelec', 'state', 'previous', 'diis', 'damp', 'cycles',
                 'converged', 'message')

    def __init__(self, nelec, state, adiis, damp):
        self.nelec = nelec
        self.state = state
        self.previous = None
        self.diis = adiis
        self.damp = damp
        self.cycles = 0
        self.converged = False
        self.message = 'fixed-N iteration has not started'


class GrandCanonicalKRKS(lib.StreamObject):
    '''Finite-temperature KRKS at fixed chemical potential or electron count.

    Args:
        mf : KRKS
            A configured GPU4PySCF periodic restricted Kohn-Sham object.
        mu : float or None
            Chemical potential in Hartree.  Required unless ``nelec`` is set.
        sigma : float
            Fermi smearing width in Hartree.
        nelec : float or None
            Electron count for a canonical calculation.
    '''

    max_cycle = getattr(
        __config__, 'pbc_dft_grand_canonical_max_cycle', 100)
    max_outer_cycle = getattr(
        __config__, 'pbc_dft_grand_canonical_max_outer_cycle', 16)
    conv_tol = getattr(
        __config__, 'pbc_dft_grand_canonical_conv_tol', 1e-8)
    conv_tol_coarse = getattr(
        __config__, 'pbc_dft_grand_canonical_conv_tol_coarse', 4e-6)
    conv_tol_mu = getattr(
        __config__, 'pbc_dft_grand_canonical_conv_tol_mu', 1e-6)
    conv_tol_nelec = getattr(
        __config__, 'pbc_dft_grand_canonical_conv_tol_nelec', 2e-5)
    diis_space = getattr(
        __config__, 'pbc_dft_grand_canonical_diis_space', 6)
    damp = getattr(
        __config__, 'pbc_dft_grand_canonical_damp', .125)
    callback = None
    enforce_time_reversal = True

    _keys = {
        'mf', 'mu', 'sigma', 'nelec', 'max_cycle', 'max_outer_cycle',
        'conv_tol', 'conv_tol_coarse', 'conv_tol_mu', 'conv_tol_nelec',
        'diis_space', 'damp', 'callback', 'enforce_time_reversal',
        'converged', 'cycles', 'outer_cycles', 'nfev', 'e_tot',
        'free_energy', 'grand_potential', 'electron_number', 'entropy',
        'entropy_energy', 'residual_rms', 'mo_energy', 'mo_coeff', 'mo_occ',
        'refinements', 'verification_attempts', 'message', 'scf_summary',
    }

    def __init__(self, mf, mu=None, sigma=None, nelec=None):
        if sigma is None:
            raise TypeError('sigma is required')
        sigma = _as_float(sigma, 'sigma')
        if sigma <= 0:
            raise ValueError('sigma must be positive')
        if nelec is None and mu is None:
            raise TypeError('mu is required unless nelec is specified')
        if nelec is not None and mu is not None:
            raise TypeError('mu and nelec select different ensembles')
        if nelec is not None:
            nelec = _as_float(nelec, 'nelec')
        if mu is not None:
            mu = _as_float(mu, 'mu')

        self.mf = mf
        self.cell = mf.cell
        self.mu = mu
        self.sigma = sigma
        self.beta = 1. / sigma
        self.nelec = nelec
        self.verbose = getattr(mf, 'verbose', logger.NOTE)
        self.stdout = getattr(mf, 'stdout', None)
        self.max_memory = getattr(mf, 'max_memory', 0)

        self.converged = False
        self.cycles = 0
        self.outer_cycles = 0
        self.nfev = 0
        self.refinements = 0
        self.verification_attempts = 0
        self.message = ''
        self.e_tot = None
        self.free_energy = None
        self.grand_potential = None
        self.electron_number = None
        self.entropy = None
        self.entropy_energy = None
        self.residual_rms = None
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.scf_summary = {}

        self._built = False
        self._state = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('ensemble = %s', 'fixed N' if self.nelec is not None else
                 'fixed mu')
        log.info('sigma = %.12g Ha', self.sigma)
        if self.nelec is None:
            log.info('mu = %.12g Ha', self.mu)
            log.info('max outer cycles = %d', self.max_outer_cycle)
            log.info('mu tolerance = %g', self.conv_tol_mu)
            log.info('electron-number tolerance = %g', self.conv_tol_nelec)
        else:
            log.info('nelec = %.12g', self.nelec)
        log.info('residual tolerance = %g', self.conv_tol)
        log.info('coarse residual tolerance = %g', self.conv_tol_coarse)
        log.info('max fixed-N cycles = %d', self.max_cycle)
        log.info('DIIS space = %d', self.diis_space)
        log.info('initial damping = %g', self.damp)
        return self

    def check_sanity(self):
        if not isinstance(self.max_cycle, int) or self.max_cycle < 1:
            raise ValueError('max_cycle must be a positive integer')
        if not isinstance(self.max_outer_cycle, int) or self.max_outer_cycle < 1:
            raise ValueError('max_outer_cycle must be a positive integer')
        if not isinstance(self.diis_space, int) or self.diis_space < 2:
            raise ValueError('diis_space must be an integer of at least 2')
        for name in ('conv_tol', 'conv_tol_coarse', 'conv_tol_mu',
                     'conv_tol_nelec', 'damp'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError('%s must be finite and positive' % name)
        if self.conv_tol_coarse < self.conv_tol:
            raise ValueError('conv_tol_coarse may not be tighter than conv_tol')
        if self.damp > 1:
            raise ValueError('damp may not exceed 1')
        return self

    def build(self):
        if self._built:
            return self
        self.check_sanity()
        required = ('cell', 'kpts', 'get_ovlp', 'get_hcore', 'get_veff',
                    'energy_elec', 'energy_nuc', 'get_init_guess',
                    'check_linear_dependency')
        missing = [name for name in required if not hasattr(self.mf, name)]
        if missing:
            raise TypeError('mf is not KRKS-compatible; missing ' +
                            ', '.join(missing))
        if hasattr(self.mf, 'istype') and not self.mf.istype('KRKS'):
            raise TypeError('GrandCanonicalKRKS requires a periodic KRKS object')
        if getattr(self.mf, 'smearing_method', None):
            raise ValueError('remove the standard smearing decorator first')
        if getattr(self.mf, 'sigma', 0.) not in (None, 0, 0.):
            raise ValueError('mf already has nonzero smearing')
        self._check_functional()

        kpts = self.mf.kpts
        if not isinstance(kpts, np.ndarray):
            raise NotImplementedError('symmetry-reduced KPoints are not supported')
        self.kpts = np.asarray(kpts, dtype=float)
        if self.kpts.ndim != 2 or self.kpts.shape[1] != 3 or len(self.kpts) == 0:
            raise ValueError('kpts must have shape (nkpts,3)')
        self.nkpts = len(self.kpts)
        self.weight = 1. / self.nkpts

        if hasattr(self.mf, 'build'):
            self.mf.build()
        self.s_ao = _blocks(
            self.mf.get_ovlp(self.cell, self.mf.kpts), 'overlap')
        self.hcore_ao = _blocks(
            self.mf.get_hcore(self.cell, self.mf.kpts), 'hcore')
        if len(self.s_ao) != self.nkpts or len(self.hcore_ao) != self.nkpts:
            raise ValueError('overlap, hcore, and k-point counts differ')
        try:
            x = self.mf.check_linear_dependency(
                _stack_or_list(self.s_ao),
                time_reversal_symmetry=getattr(
                    self.mf, 'time_reversal_symmetry', False))
        except TypeError:
            x = self.mf.check_linear_dependency(_stack_or_list(self.s_ao))
        self.x_ao2orth = _blocks(x, 'orthogonalizers')
        if len(self.x_ao2orth) != self.nkpts:
            raise ValueError('orthogonalizer and k-point counts differ')

        self.nao = self.s_ao[0].shape[0]
        self.north = []
        self.identity = []
        for k, (s, h, x) in enumerate(zip(
                self.s_ao, self.hcore_ao, self.x_ao2orth)):
            if s.shape != (self.nao, self.nao) or h.shape != s.shape:
                raise ValueError('inconsistent AO dimensions at k-point %d' % k)
            if x.shape[0] != self.nao or x.shape[1] == 0:
                raise ValueError('invalid orthogonalizer at k-point %d' % k)
            error = cp.max(cp.abs(
                x.conj().T.dot(s).dot(x) - cp.eye(x.shape[1]))).item()
            if error > _ORTHOGONALITY_TOL:
                raise ValueError('X^H S X is not identity at k-point %d' % k)
            self.north.append(x.shape[1])
            self.identity.append(cp.eye(x.shape[1], dtype=x.dtype))
        self.ndof = self.weight * sum(n*n for n in self.north)
        self.capacity = 2. * self.weight * sum(self.north)
        if self.nelec is not None and not 0 < self.nelec < self.capacity:
            raise ValueError('nelec must lie between 0 and %g' % self.capacity)

        self.hcore = _stack_or_list(self.hcore_ao)
        self.e_nuc = _as_float(self.mf.energy_nuc(), 'nuclear energy')
        self._tr_pairs, self._time_reversal = self._init_time_reversal()
        self._built = True
        return self

    def _check_functional(self):
        if getattr(self.mf, 'do_nlc', lambda: False)():
            raise NotImplementedError('nonlocal correlation is not supported')
        ni = getattr(self.mf, '_numint', None)
        libxc = getattr(ni, 'libxc', None)
        xc = getattr(self.mf, 'xc', None)
        if libxc is None or xc is None:
            return
        if libxc.is_hybrid_xc(xc):
            raise NotImplementedError('hybrid functionals are not supported')
        from pyscf.dft import libxc as pyscf_libxc
        xctype = pyscf_libxc.xc_type(xc).upper()
        if xctype not in ('LDA', 'GGA'):
            raise NotImplementedError('%s functionals are not supported' % xctype)

    def _find_time_reversal_pairs(self):
        if hasattr(self.mf, 'iter_kpt_pairs'):
            pairs = []
            for pair in self.mf.iter_kpt_pairs():
                if len(pair) >= 2 and np.isscalar(pair[0]):
                    i, j = int(pair[0]), int(pair[1])
                    if i < j:
                        pairs.append((i, j))
            if pairs:
                return pairs
        try:
            scaled = np.asarray(
                self.cell.get_scaled_kpts(self.kpts), dtype=float)
        except (AttributeError, TypeError):
            scaled = self.kpts
        pairs = []
        for i in range(self.nkpts):
            distances = []
            for j in range(self.nkpts):
                delta = scaled[i] + scaled[j]
                delta -= np.rint(delta)
                distances.append(np.linalg.norm(delta))
            j = int(np.argmin(distances))
            if distances[j] < 1e-8 and i < j:
                pairs.append((i, j))
        return pairs

    def _init_time_reversal(self):
        pairs = self._find_time_reversal_pairs()
        if not self.enforce_time_reversal:
            return pairs, False
        valid = True
        for i, j in pairs:
            if self.north[i] != self.north[j]:
                valid = False
                break
            error = max(
                cp.max(cp.abs(self.s_ao[j] - self.s_ao[i].conj())).item(),
                cp.max(cp.abs(
                    self.hcore_ao[j] - self.hcore_ao[i].conj())).item(),
                cp.max(cp.abs(
                    self.x_ao2orth[j] -
                    self.x_ao2orth[i].conj())).item())
            if error > _ORTHOGONALITY_TOL:
                valid = False
                break
        if not valid:
            logger.warn(self, 'Time-reversal gauge check failed; projection disabled')
        return pairs, valid

    def _copy(self, blocks):
        return [x.copy() for x in blocks]

    def _hermi(self, blocks):
        return [.5 * (x + x.conj().T) for x in blocks]

    def _project_time_reversal(self, blocks):
        out = self._copy(blocks)
        if not self._time_reversal:
            return out
        for i, j in self._tr_pairs:
            value = .5 * (out[i] + out[j].conj())
            out[i] = value
            out[j] = value.conj()
        return out

    def _inner(self, a, b):
        return self.weight * sum(
            float(cp.vdot(x, y).real.item()) for x, y in zip(a, b))

    def _rms(self, blocks):
        return max(0., self._inner(blocks, blocks) / self.ndof) ** .5

    def _density_rms(self, a, b):
        return self._rms([x-y for x, y in zip(a, b)])

    def _trace_mean(self, blocks):
        numerator = sum(float(cp.trace(x).real.item()) for x in blocks)
        return numerator / sum(x.shape[0] for x in blocks)

    def _sanitize_h(self, h):
        h = _blocks(h, 'auxiliary Hamiltonian')
        if len(h) != self.nkpts:
            raise ValueError('Hamiltonian and k-point counts differ')
        for k, x in enumerate(h):
            if x.shape != (self.north[k], self.north[k]):
                raise ValueError('invalid Hamiltonian shape at k-point %d' % k)
            if not bool(cp.all(cp.isfinite(x)).item()):
                raise FloatingPointError('Hamiltonian contains nonfinite values')
        return self._hermi(self._project_time_reversal(self._hermi(h)))

    def _to_orth(self, matrices):
        matrices = _blocks(matrices, 'AO matrices')
        out = []
        for k, (a, x) in enumerate(zip(matrices, self.x_ao2orth)):
            value = x.conj().T.dot(a).dot(x)
            error = cp.linalg.norm(value - value.conj().T).item() / value.shape[0]
            if error > _HERMITICITY_TOL:
                raise FloatingPointError(
                    'Fock matrix at k-point %d is not Hermitian' % k)
            out.append(.5 * (value + value.conj().T))
        return out

    def _fock_from_veff(self, dm, veff):
        if getattr(veff, 'v_solvent', None) is not None:
            if not hasattr(self.mf, 'get_fock'):
                raise TypeError('tagged solvent potential requires mf.get_fock')
            fock = self.mf.get_fock(
                h1e=self.hcore, vhf=veff, dm=dm, cycle=-1, diis=None,
                level_shift_factor=0., damp_factor=0.)
            return cp.stack(_blocks(fock, 'decorated Fock matrices'))
        veff_array = cp.asarray(veff)
        return cp.stack([
            hcore + veff_array[k]
            for k, hcore in enumerate(self.hcore_ao)])

    def _initial_h(self, dm0=None):
        if dm0 is None:
            try:
                dm0 = self.mf.get_init_guess(self.cell, kpts=self.mf.kpts)
            except TypeError:
                dm0 = self.mf.get_init_guess(self.cell)
        dm = cp.stack(self._hermi(self._project_time_reversal(
            _blocks(dm0, 'initial density'))))
        self.nfev += 1
        veff = self.mf.get_veff(
            self.cell, dm, dm_last=None, vhf_last=None, hermi=1,
            kpts=self.mf.kpts, kpts_band=None)
        return self._sanitize_h(self._to_orth(
            self._fock_from_veff(dm, veff)))

    def _solve_mu(self, orbital_energies, nelec):
        mo_energy = np.concatenate([
            cp.asnumpy(cp.asarray(x)).ravel() for x in orbital_energies])
        nocc = nelec * self.nkpts / 2.
        mu, _ = _smearing_optimize(
            _fermi_smearing_occ, mo_energy, nocc, self.sigma)
        mu = np.asarray(mu).reshape(-1)
        if mu.size != 1:
            raise ValueError('PySCF smearing returned a nonscalar mu')
        mu = _as_float(mu[0], 'chemical potential')
        error = abs(self._nelec_from_eig(orbital_energies, mu) - nelec)
        if error > 1e-10:
            raise RuntimeError('chemical-potential solve has charge error %g' % error)
        return mu

    def _nelec_from_eig(self, orbital_energies, mu):
        return 2. * self.weight * sum(float(cp.sum(
            _fermi_occ(self.beta * (x-mu))).item())
            for x in orbital_energies)

    def _nelec_at_mu(self, h, mu):
        return self._nelec_from_eig([cp.linalg.eigvalsh(x) for x in h], mu)

    def _evaluate(self, h, nelec=None, mu=None):
        if (nelec is None) == (mu is None):
            raise TypeError('specify exactly one of nelec and mu')
        self.nfev += 1
        h = self._sanitize_h(h)
        eigenpairs = [cp.linalg.eigh(x) for x in h]
        eig = [x[0] for x in eigenpairs]
        coeff = [x[1] for x in eigenpairs]
        aux_mu = self._solve_mu(eig, nelec) if nelec is not None else mu

        occ = []
        p = []
        dm = cp.empty((self.nkpts, self.nao, self.nao),
                      dtype=cp.result_type(*h, cp.complex128))
        entropy_sum = 0.
        for k, (energy, c, x) in enumerate(zip(
                eig, coeff, self.x_ao2orth)):
            gamma = self.beta * (energy-aux_mu)
            q = _fermi_occ(gamma)
            density = (c * q[None, :]).dot(c.conj().T)
            density = .5 * (density + density.conj().T)
            dmk = 2. * x.dot(density).dot(x.conj().T)
            dm[k] = .5 * (dmk + dmk.conj().T)
            occ.append(q)
            p.append(density)
            entropy_sum += float(cp.sum(_fermi_entropy(gamma, q)).item())
        p = self._hermi(self._project_time_reversal(p))
        dm = cp.stack(self._hermi(self._project_time_reversal(
            [dm[k] for k in range(self.nkpts)])))

        electron_number = 2. * self.weight * sum(
            float(cp.trace(x).real.item()) for x in p)
        ao_nelec = self.weight * sum(_as_float(
            cp.einsum('ij,ji->', d, s), 'AO electron count')
            for d, s in zip(dm, self.s_ao))
        if abs(electron_number-ao_nelec) > 1e-8 * max(1., electron_number):
            raise ValueError('AO and orthogonal electron counts disagree')

        veff = self.mf.get_veff(
            self.cell, dm, dm_last=None, vhf_last=None, hermi=1,
            kpts=self.mf.kpts, kpts_band=None)
        fock_ao = self._fock_from_veff(dm, veff)
        e_elec = _as_float(
            self.mf.energy_elec(dm, self.hcore, veff)[0],
            'electronic energy')
        fock = self._to_orth(fock_ao)
        e_tot = e_elec + self.e_nuc
        entropy = -2. * self.weight * entropy_sum
        entropy_energy = -self.sigma * entropy
        free_energy = e_tot + entropy_energy

        mismatch = self._hermi([x-y for x, y in zip(h, fock)])
        gauge = self._trace_mean(mismatch) if nelec is not None else 0.
        if nelec is not None:
            mismatch = [x-gauge*eye for x, eye in zip(mismatch, self.identity)]
        residual = [-x for x in mismatch]
        physical_mu = aux_mu-gauge
        grand_potential = free_energy-physical_mu*electron_number
        values = (e_tot, free_energy, grand_potential, physical_mu,
                  electron_number)
        if not all(np.isfinite(x) for x in values):
            raise FloatingPointError('finite-temperature state is nonfinite')
        return _State(
            h, eig, coeff, occ, p, dm, fock, aux_mu, physical_mu, gauge,
            e_tot, electron_number, entropy, entropy_energy, free_energy,
            grand_potential, residual, self._rms(mismatch))

    def _pack(self, blocks, weight_errors=False):
        scale = self.weight**.5 if weight_errors else 1.
        return cp.concatenate([(scale*x).ravel() for x in blocks])

    def _unpack(self, vector, template):
        out = []
        offset = 0
        for x in template:
            size = x.size
            out.append(vector[offset:offset+size].reshape(x.shape))
            offset += size
        if offset != vector.size:
            raise ValueError('packed DIIS vector has the wrong size')
        return out

    def _new_session(self, h, nelec):
        state = self._evaluate(h, nelec=nelec)
        adiis = diis.DIIS(self)
        adiis.space = self.diis_space
        return _Session(float(nelec), state, adiis, self.damp)

    def _try_target(self, session, target, max_backtrack):
        state = session.state
        direction = [x-y for x, y in zip(target, state.h)]
        if not all(bool(cp.all(cp.isfinite(x)).item()) for x in direction):
            return None, 0.
        damp = min(1., max(_MIN_DAMP, session.damp))
        limit = state.residual_rms * (1.-_DIIS_MIN_REDUCTION)
        for unused in range(max_backtrack+1):
            h = self._sanitize_h([
                x+damp*d for x, d in zip(state.h, direction)])
            trial = self._evaluate(h, nelec=session.nelec)
            logger.debug(self,
                         'DIIS damping %.6g residual %.6g -> %.6g',
                         damp, state.residual_rms, trial.residual_rms)
            if trial.residual_rms < limit:
                return trial, damp
            damp *= _DIIS_BACKTRACK
        return None, 0.

    def _update_damp(self, old, new, accepted, starting):
        predicted = max(0., 1.-accepted) * old.residual_rms
        predicted_reduction = old.residual_rms-predicted
        actual_reduction = old.residual_rms-new.residual_rms
        scale = max(old.residual_rms, np.finfo(float).tiny)
        ratio = (actual_reduction/predicted_reduction
                 if predicted_reduction > np.finfo(float).eps*scale
                 else np.nan)
        damp = accepted
        if np.isfinite(ratio) and ratio < _DIIS_TRUST_SHRINK:
            damp *= _DIIS_BACKTRACK
        elif (np.isfinite(ratio) and ratio > _DIIS_TRUST_EXPAND and
              actual_reduction/scale >= _DIIS_EXPAND_REDUCTION and
              accepted >= starting*(1.-1e-12)):
            damp *= _DIIS_EXPANSION
        return min(1., max(_MIN_DAMP, damp)), ratio

    def _advance_session(self, session, tolerance):
        session.converged = False
        session.message = 'maximum fixed-N cycles reached'
        while session.cycles < self.max_cycle:
            state = session.state
            if state.residual_rms <= tolerance:
                session.converged = True
                session.message = 'converged fixed-N residual'
                break
            fock = self._pack(state.fock)
            residual = self._pack(state.residual, weight_errors=True)
            try:
                target = self._unpack(
                    session.diis.update(fock, xerr=residual), state.fock)
                target = self._sanitize_h(target)
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                session.diis.clear()
                target = self._copy(state.fock)
            starting = session.damp
            trial, accepted = self._try_target(
                session, target, min(2, _DIIS_MAX_BACKTRACK))
            if trial is None:
                trial, accepted = self._try_target(
                    session, state.fock, _DIIS_MAX_BACKTRACK)
            if trial is None:
                session.diis.clear()
                session.message = 'residual DIIS could not reduce the residual'
                break
            session.damp, ratio = self._update_damp(
                state, trial, accepted, starting)
            session.previous = state
            session.state = trial
            session.cycles += 1
            self.cycles += 1
            logger.info(
                self, 'Fixed-N cycle %d  N = %.12g  mu = %.12g  '
                'A = %.12g  residual = %.6g  damping = %.3g  ratio = %.3g',
                self.cycles, session.nelec, trial.mu, trial.free_energy,
                trial.residual_rms, accepted, ratio)
            if callable(self.callback):
                self.callback({
                    'solver': self, 'state': trial, 'cycle': self.cycles,
                    'electron_number': session.nelec})
        if session.state.residual_rms <= tolerance:
            session.converged = True
            session.message = 'converged fixed-N residual'
        return session

    @staticmethod
    def _bracket(samples):
        negative = [x for x in samples if x[1] < 0]
        positive = [x for x in samples if x[1] > 0]
        if not negative or not positive:
            return None
        pair = min(((a, b) for a in negative for b in positive),
                   key=lambda x: abs(x[0][0]-x[1][0]))
        return tuple(sorted(pair, key=lambda x: x[0]))

    @staticmethod
    def _sample_index(samples, nelec):
        tolerance = max(
            _ROOT_NELEC_TOL,
            32*np.finfo(float).eps*max(1., abs(nelec)))
        return next((i for i, x in enumerate(samples)
                     if abs(x[0]-nelec) <= tolerance), None)

    def _observe(self, samples, session):
        sample = (session.nelec, session.state.mu-self.mu,
                  session.state, session)
        index = self._sample_index(samples, session.nelec)
        if index is not None:
            samples.pop(index)
        samples.append(sample)
        return sample

    def _prune_sessions(self, samples, bracket):
        keep = {id(x) for x in samples[-2:]}
        if bracket is not None:
            keep.update(id(x) for x in bracket)
        for i, sample in enumerate(samples):
            if sample[3] is not None and id(sample) not in keep:
                samples[i] = sample[:3] + (None,)

    def _secant_proposal(self, samples, state):
        current_n, current_error = samples[-1][:2]
        previous = next((x for x in reversed(samples[:-1])
                         if abs(x[0]-current_n) > _ROOT_NELEC_TOL), None)
        if previous is None:
            proposal = self._nelec_at_mu(state.fock, self.mu)
            maximum = _INITIAL_NELEC_STEP
        else:
            denominator = current_error-previous[1]
            proposal = np.nan
            if denominator != 0:
                proposal = current_n-current_error*(
                    current_n-previous[0])/denominator
            if not np.isfinite(proposal):
                proposal = self._nelec_at_mu(state.fock, self.mu)
            maximum = _MAX_NELEC_STEP_FRACTION * float(self.cell.nelectron)
        proposal = current_n + np.clip(proposal-current_n, -maximum, maximum)

        bracket = self._bracket(samples)
        if bracket is not None:
            left, right = bracket[0][0], bracket[1][0]
            margin = min(_ROOT_NELEC_TOL, .25*(right-left))
            if not left+margin < proposal < right-margin:
                proposal = left+.5*(right-left)
        margin = min(_ROOT_NELEC_TOL, .25*self.capacity)
        return min(self.capacity-margin, max(margin, proposal))

    def _proposal_h(self, proposal, state, bracket):
        if bracket is None:
            return self._copy(state.fock)
        left, right = bracket
        if not left[0] <= proposal <= right[0]:
            return self._copy(state.fock)
        fraction = (proposal-left[0])/(right[0]-left[0])
        return self._sanitize_h([
            (1.-fraction)*a+fraction*b
            for a, b in zip(left[2].fock, right[2].fock)])

    def _fixed_mu_candidate(self, state):
        shift = self.mu-state.aux_mu
        h = self._sanitize_h([
            x+shift*eye for x, eye in zip(state.h, self.identity)])
        delta_nelec = self._nelec_at_mu(state.fock, self.mu)-state.nelec
        return h, delta_nelec

    def _verify(self, source):
        h, unused = self._fixed_mu_candidate(source)
        state = self._evaluate(h, mu=self.mu)
        self.verification_attempts += 1
        delta_nelec = self._nelec_at_mu(state.fock, self.mu)-state.nelec
        density_rms = self._density_rms(state.p, source.p)
        accepted = (
            state.residual_rms <= max(self.conv_tol, _VERIFY_RESIDUAL_TOL) and
            abs(delta_nelec) <= self.conv_tol_nelec and
            density_rms <= _VERIFY_DENSITY_TOL)
        logger.info(
            self, 'Fixed-mu verification residual = %.6g  delta N = %.3g  '
            'density change = %.3g', state.residual_rms, delta_nelec,
            density_rms)
        return state, accepted

    def _kernel_fixed_n(self, h):
        session = self._new_session(h, self.nelec)
        self._advance_session(session, self.conv_tol)
        self.message = session.message
        return session.state, session.converged

    def _kernel_fixed_mu(self, h):
        samples = []
        current_n = self._nelec_at_mu(h, self.mu)
        best = None
        best_score = np.inf
        pending = None
        force_tight = False
        repaired_verification = False
        distinct = []

        for unused_pass in range(4*self.max_outer_cycle+4):
            distinct_tol = max(
                _ROOT_NELEC_TOL,
                32*np.finfo(float).eps*max(1., abs(current_n)))
            is_distinct = not any(
                abs(x-current_n) <= distinct_tol for x in distinct)
            if is_distinct:
                if len(distinct) >= self.max_outer_cycle:
                    self.message = 'maximum fixed-mu outer cycles reached'
                    break
                distinct.append(float(current_n))
                self.outer_cycles += 1
            else:
                self.refinements += 1

            bracket = self._bracket(samples)
            tolerance = (self.conv_tol if bracket is not None or force_tight
                         else self.conv_tol_coarse)
            force_tight = False
            if pending is None:
                session = self._new_session(h, current_n)
            else:
                session = pending
                pending = None
            self._advance_session(session, tolerance)
            state = session.state
            if not session.converged:
                if best is None:
                    best = state
                self.message = 'fixed-N inner solve failed: ' + session.message
                break

            sample = self._observe(samples, session)
            bracket = self._bracket(samples)
            self._prune_sessions(samples, bracket)
            error = sample[1]
            physical_delta = self._nelec_at_mu(state.fock, self.mu)-state.nelec
            score = max(abs(error)/self.conv_tol_mu,
                        abs(physical_delta)/self.conv_tol_nelec)
            if score < best_score:
                best_score = score
                best = state
            logger.info(
                self, 'Fixed-mu outer cycle %d  N = %.12g  optimized mu = '
                '%.12g  delta mu = %.3g  delta N = %.3g  residual = %.6g',
                self.outer_cycles, current_n, state.mu, error,
                physical_delta, state.residual_rms)

            root_ready = (abs(error) <= self.conv_tol_mu and
                          abs(physical_delta) <= self.conv_tol_nelec)
            if root_ready and state.residual_rms > self.conv_tol:
                h = self._copy(state.h)
                pending = session
                force_tight = True
                continue
            if root_ready:
                verified, accepted = self._verify(state)
                if accepted:
                    self.message = 'converged fixed-mu secant search'
                    return verified, True
                if repaired_verification:
                    best = state
                    self.message = 'fixed-mu verification failed after repair'
                    break
                repaired_verification = True
                h = self._copy(verified.fock)
                current_n = state.nelec
                pending = None
                force_tight = True
                continue

            if bracket is not None:
                endpoint = min(bracket, key=lambda x: abs(x[1]))
                if endpoint[2].residual_rms > self.conv_tol:
                    h = self._copy(endpoint[2].fock)
                    current_n = endpoint[0]
                    pending = endpoint[3]
                    force_tight = True
                    continue
                if abs(bracket[1][0]-bracket[0][0]) <= _ROOT_NELEC_TOL:
                    self.message = 'fixed-mu electron-number bracket stagnated'
                    break

            proposal = self._secant_proposal(samples, state)
            if abs(proposal-current_n) <= 1e-14*max(1., abs(current_n)):
                self.message = 'fixed-mu secant search stagnated'
                break
            h = self._proposal_h(proposal, state, bracket)
            current_n = proposal
        else:  # pragma: no cover
            self.message = 'fixed-mu safety iteration limit reached'

        if best is None:
            raise RuntimeError(self.message or 'fixed-mu search produced no state')
        verified, unused = self._verify(best)
        return verified, False

    def _finalize(self, state, converged):
        mo_coeff = [x.dot(c) for x, c in zip(self.x_ao2orth, state.coeff)]
        mo_energy = [e-state.gauge for e in state.eig]
        mo_occ = [2.*x for x in state.occ]
        self.mo_coeff = _stack_or_list(mo_coeff)
        self.mo_energy = _stack_or_list(mo_energy)
        self.mo_occ = _stack_or_list(mo_occ)
        self.converged = bool(converged)
        self.e_tot = state.e_tot
        self.free_energy = state.free_energy
        self.grand_potential = state.grand_potential
        self.electron_number = state.nelec
        self.mu = state.mu
        self.entropy = state.entropy
        self.entropy_energy = state.entropy_energy
        self.residual_rms = state.residual_rms
        self._state = state
        self.scf_summary = {
            'e_tot': self.e_tot,
            'free_energy': self.free_energy,
            'grand_potential': self.grand_potential,
            'electron_number': self.electron_number,
            'mu': self.mu,
            'sigma': self.sigma,
            'entropy': self.entropy,
            'residual_rms': self.residual_rms,
            'fock_evaluations': self.nfev,
            'fixed_n_cycles': self.cycles,
            'outer_cycles': self.outer_cycles,
            'refinements': self.refinements,
            'verification_attempts': self.verification_attempts,
        }

        self.mf.converged = self.converged
        self.mf.mo_coeff = self.mo_coeff
        self.mf.mo_energy = self.mo_energy
        self.mf.mo_occ = self.mo_occ
        self.mf.e_tot = self.e_tot
        self.mf.free_energy = self.free_energy
        self.mf.grand_potential = self.grand_potential
        self.mf.electron_number_gc = self.electron_number
        self.mf.mu_gc = self.mu
        self.mf.sigma_gc = self.sigma
        if not hasattr(self.mf, 'scf_summary') or self.mf.scf_summary is None:
            self.mf.scf_summary = {}
        self.mf.scf_summary.update(self.scf_summary)
        return self.e_tot

    def kernel(self, dm0=None):
        self.build()
        self.dump_flags()
        self.converged = False
        self.cycles = 0
        self.outer_cycles = 0
        self.nfev = 0
        self.refinements = 0
        self.verification_attempts = 0
        self.message = ''
        h = self._initial_h(dm0)
        if self.nelec is None:
            state, converged = self._kernel_fixed_mu(h)
        else:
            state, converged = self._kernel_fixed_n(h)
        logger.info(self, '%s; total Fock evaluations = %d',
                    self.message, self.nfev)
        return self._finalize(state, converged)

    scf = kernel
