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

import csv
from pathlib import Path

import numpy as np
import cupy as cp
import h5py
from scipy.interpolate import PchipInterpolator
from scipy.special import expit

from pyscf import __config__, lib
from pyscf.scf import chkfile
from pyscf.scf.addons import _fermi_smearing_occ, _smearing_optimize

from gpu4pyscf.lib import diis, logger
from gpu4pyscf.scf import hf as mol_hf
from gpu4pyscf.pbc.dft.grand_canonical_cg import fixed_mu_diis, nlcg


__all__ = ['GrandCanonicalKRKS']


_HERMITICITY_TOL = getattr(
    __config__, 'pbc_dft_grand_canonical_hermiticity_tol', 1e-10)
_ORTHOGONALITY_TOL = getattr(
    __config__, 'pbc_dft_grand_canonical_orthogonality_tol', 1e-9)


def _as_float(value, name='value'):
    if isinstance(value, cp.ndarray):
        if value.size != 1:
            raise ValueError('%s must be scalar' % name)
        value = value.item()
    value = complex(value)
    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        raise FloatingPointError('%s is nonfinite' % name)
    if abs(value.imag) > 1e-9 * max(1.0, abs(value.real)):
        raise ValueError('%s has a significant imaginary part: %g' % (name, value.imag))
    return float(value.real)


def coerce_to_list_of_matrices(values, name='matrix blocks'):
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
    return expit(-cp.asarray(gamma))


def _fermi_entropy(occ):
    occ = cp.asarray(occ)
    f = occ[(occ>0) & (occ<1)]
    entropy = -(f*cp.log(f) + (1-f)*cp.log(1-f)).sum()
    return entropy


class GCSolverCycle:
    __slots__ = (
        'h', 'eig', 'coeff', 'occ', 'p', 'dm', 'fock',
        'mu', 'e_tot', 'nelec',
        'entropy', 'entropy_energy', 'free_energy', 'grand_potential',
        'residual', 'residual_rms', 'grid_data')

    def __init__(self, h, eig, coeff, occ, p, dm, fock, mu,
                 e_tot, nelec, entropy, entropy_energy, free_energy,
                 grand_potential, residual, residual_rms, grid_data=None):
        self.h = h
        self.eig = eig
        self.coeff = coeff
        self.occ = occ
        self.p = p
        self.dm = dm
        self.fock = fock
        self.mu = mu
        self.e_tot = e_tot
        self.nelec = nelec
        self.entropy = entropy
        self.entropy_energy = entropy_energy
        self.free_energy = free_energy
        self.grand_potential = grand_potential
        self.residual = residual
        self.residual_rms = residual_rms
        self.grid_data = grid_data


class FixedNCalc:
    __slots__ = ('nelec', 'cycle_data', 'diis', 'damp', 'cycles',
                 'converged', 'message')

    def __init__(self, nelec, cycle_data, adiis, damp):
        self.nelec = nelec
        self.cycle_data = cycle_data
        self.diis = adiis
        self.damp = damp
        self.cycles = 0
        self.converged = False
        self.message = 'fixed-N iteration has not started'


class MuSample:
    __slots__ = ('cycle_data', 'delta_mu', 'fixed_n_calc')

    def __init__(self, cycle_data, delta_mu, fixed_n_calc):
        self.cycle_data = cycle_data
        self.delta_mu = delta_mu
        self.fixed_n_calc = fixed_n_calc

    @property
    def nelec(self):
        return self.cycle_data.nelec


class MuSeed:
    __slots__ = ('nelec', 'delta_mu', 'fixed_n_calc')

    def __init__(self, nelec, delta_mu):
        self.nelec = nelec
        self.delta_mu = delta_mu
        self.fixed_n_calc = None


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

    nlcg = nlcg
    fixed_mu_diis = fixed_mu_diis

    max_cycle = getattr(__config__, 'pbc_dft_grand_canonical_max_cycle', 100)
    max_outer_cycle = getattr(__config__, 'pbc_dft_grand_canonical_max_outer_cycle', 16)
    conv_tol = getattr(__config__, 'pbc_dft_grand_canonical_conv_tol', 1e-8)
    conv_tol_coarse = getattr(__config__, 'pbc_dft_grand_canonical_conv_tol_coarse', 1e-6)
    conv_tol_mu = getattr(__config__, 'pbc_dft_grand_canonical_conv_tol_mu', 1e-6)
    nlcg_initial_step = getattr(
        __config__, 'pbc_dft_grand_canonical_nlcg_initial_step', 1.0)
    nlcg_max_line_search_evaluations = getattr(
        __config__,
        'pbc_dft_grand_canonical_nlcg_max_line_search_evaluations', 6)
    nlcg_line_search_alpha_rtol = getattr(
        __config__,
        'pbc_dft_grand_canonical_nlcg_line_search_alpha_rtol', None)
    nlcg_line_search_slope_atol = getattr(
        __config__,
        'pbc_dft_grand_canonical_nlcg_line_search_slope_atol', None)
    tighten_mu_threshold = getattr(
        __config__, 'pbc_dft_grand_canonical_tighten_mu_threshold', 1e-3)
    diis_space = getattr(__config__, 'pbc_dft_grand_canonical_diis_space', 6)
    damp = getattr(__config__, 'pbc_dft_grand_canonical_damp', 0.125)
    diis_trust_expand = getattr(__config__, 'pbc_dft_grand_canonical_diis_trust_expand', 0.75)
    diis_expansion = getattr(__config__, 'pbc_dft_grand_canonical_diis_expansion', 2.0)
    diis_expand_reduction = getattr(__config__, 'pbc_dft_grand_canonical_diis_expand_reduction', 2e-2)
    min_damp = getattr(__config__, 'pbc_dft_grand_canonical_min_damp', 1e-8)
    initial_nelec_step = getattr(__config__, 'pbc_dft_grand_canonical_initial_nelec_step', 3e-2)
    max_nelec_step_fraction = getattr(__config__, 'pbc_dft_grand_canonical_max_nelec_step_fraction', 0.1)
    root_nelec_tol = getattr(__config__, 'pbc_dft_grand_canonical_root_nelec_tol', 1e-8)
    callback = None

    _keys = {
        'mf', 'mu', 'sigma', 'nelec', 'max_cycle', 'max_outer_cycle',
        'conv_tol', 'conv_tol_coarse', 'conv_tol_mu',
        'nlcg_initial_step', 'nlcg_max_line_search_evaluations',
        'nlcg_line_search_alpha_rtol', 'nlcg_line_search_slope_atol',
        'tighten_mu_threshold',
        'diis_space', 'damp', 'callback',
        'diis_trust_expand', 'diis_expansion',
        'diis_expand_reduction', 'min_damp', 'initial_nelec_step',
        'max_nelec_step_fraction', 'root_nelec_tol',
        'converged', 'cycles', 'outer_cycles', 'nfev', 'e_tot',
        'free_energy', 'grand_potential', 'electron_number', 'entropy',
        'entropy_energy', 'residual_rms', 'mo_energy', 'mo_coeff', 'mo_occ',
        'refinements', 'message', 'scf_summary',
        'potential_residual_rms',
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
        self.chkfile = getattr(mf, 'chkfile', None)

        self.converged = False
        self.cycles = 0
        self.outer_cycles = 0
        self.nfev = 0
        self.refinements = 0
        self.message = ''
        self.e_tot = None
        self.free_energy = None
        self.grand_potential = None
        self.electron_number = None
        self.entropy = None
        self.entropy_energy = None
        self.residual_rms = None
        self.potential_residual_rms = None
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.scf_summary = {}

        self._built = False
        self._cycle_data = None
        self._potential_cycle = None

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
            log.info('tight-residual mu threshold = %g',
                     self.tighten_mu_threshold)
        else:
            log.info('nelec = %.12g', self.nelec)
        log.info('residual tolerance = %g', self.conv_tol)
        log.info('coarse residual tolerance = %g', self.conv_tol_coarse)
        log.info('max fixed-N cycles = %d', self.max_cycle)
        log.info('NLCG initial line-search step = %g',
                 self.nlcg_initial_step)
        log.info('NLCG max line-search evaluations = %d',
                 self.nlcg_max_line_search_evaluations)
        if (self.nlcg_line_search_alpha_rtol is None and
                self.nlcg_line_search_slope_atol is None):
            log.info('NLCG strict line minimization = disabled')
        elif (self.nlcg_line_search_alpha_rtol is None or
                self.nlcg_line_search_slope_atol is None):
            log.info('NLCG strict line minimization configuration is '
                     'incomplete')
        else:
            log.info('NLCG line-search alpha relative tolerance = %g '
                     '(%.6g%%)', self.nlcg_line_search_alpha_rtol,
                     100.0*self.nlcg_line_search_alpha_rtol)
            log.info('NLCG normalized line-search slope tolerance = %g',
                     self.nlcg_line_search_slope_atol)
        log.info('DIIS space = %d', self.diis_space)
        log.info('initial damping = %g', self.damp)
        return self

    def check_sanity(self):
        if not isinstance(self.max_cycle, int) or self.max_cycle < 1:
            raise ValueError('max_cycle must be a positive integer')
        if not isinstance(self.max_outer_cycle, int) or self.max_outer_cycle < 1:
            raise ValueError('max_outer_cycle must be a positive integer')
        if (not isinstance(self.nlcg_max_line_search_evaluations, int) or
                self.nlcg_max_line_search_evaluations < 2):
            raise ValueError(
                'nlcg_max_line_search_evaluations must be an integer of '
                'at least 2')
        strict_alpha = self.nlcg_line_search_alpha_rtol
        strict_slope = self.nlcg_line_search_slope_atol
        if (strict_alpha is None) != (strict_slope is None):
            raise ValueError(
                'nlcg_line_search_alpha_rtol and '
                'nlcg_line_search_slope_atol must both be set or both be '
                'None')
        if strict_alpha is not None:
            for name, value in (
                    ('nlcg_line_search_alpha_rtol', strict_alpha),
                    ('nlcg_line_search_slope_atol', strict_slope)):
                if not np.isfinite(value) or value <= 0:
                    raise ValueError('%s must be finite and positive' % name)
        if not isinstance(self.diis_space, int) or self.diis_space < 2:
            raise ValueError('diis_space must be an integer of at least 2')
        for name in ('conv_tol', 'conv_tol_coarse', 'conv_tol_mu',
                     'tighten_mu_threshold', 'damp',
                     'diis_trust_expand', 'diis_expansion',
                     'diis_expand_reduction', 'min_damp',
                     'nlcg_initial_step',
                     'initial_nelec_step', 'max_nelec_step_fraction',
                     'root_nelec_tol'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError('%s must be finite and positive' % name)
        if self.conv_tol_coarse < self.conv_tol:
            raise ValueError('conv_tol_coarse may not be tighter than conv_tol')
        if self.damp > 1:
            raise ValueError('damp may not exceed 1')
        if self.diis_trust_expand >= 1:
            raise ValueError('diis_trust_expand must be less than 1')
        if self.diis_expansion < 1:
            raise ValueError('diis_expansion may not be less than 1')
        if self.diis_expand_reduction >= 1:
            raise ValueError('diis_expand_reduction must be less than 1')
        if self.min_damp > 1:
            raise ValueError('min_damp may not exceed 1')
        return self

    def dump_chk(self, envs):
        if self.chkfile:
            self.mf.chkfile = self.chkfile
            self.mf.dump_chk(envs)
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

        self.mf.chkfile = self.chkfile

        if hasattr(self.mf, 'build'):
            self.mf.build()
        self.s_ao = coerce_to_list_of_matrices(
            self.mf.get_ovlp(self.cell, self.mf.kpts), 'overlap')
        self.hcore_ao = coerce_to_list_of_matrices(
            self.mf.get_hcore(self.cell, self.mf.kpts), 'hcore')
        if len(self.s_ao) != self.nkpts or len(self.hcore_ao) != self.nkpts:
            raise ValueError('overlap, hcore, and k-point counts differ')
        x = self.mf.check_linear_dependency(_stack_or_list(self.s_ao))
        self.x_ao2orth = coerce_to_list_of_matrices(x, 'orthogonalizers')
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

    def _copy(self, blocks):
        return [x.copy() for x in blocks]

    def _hermi(self, blocks):
        return [0.5 * (x + x.conj().T) for x in blocks]

    def _inner(self, a, b):
        return self.weight * sum(float(cp.vdot(x, y).real.item()) for x, y in zip(a, b))

    def _rms(self, blocks):
        return max(0.0, self._inner(blocks, blocks) / self.ndof) ** 0.5

    def _trace_mean(self, blocks):
        numerator = sum(float(cp.trace(x).real.item()) for x in blocks)
        return numerator / sum(x.shape[0] for x in blocks)

    def _sanitize_h(self, h):
        h = coerce_to_list_of_matrices(h, 'auxiliary Hamiltonian')
        if len(h) != self.nkpts:
            raise ValueError('Hamiltonian and k-point counts differ')
        for k, x in enumerate(h):
            if x.shape != (self.north[k], self.north[k]):
                raise ValueError('invalid Hamiltonian shape at k-point %d' % k)
            if not bool(cp.all(cp.isfinite(x)).item()):
                raise FloatingPointError('Hamiltonian contains nonfinite values')
        return self._hermi(h)

    def _to_orth(self, matrices):
        matrices = coerce_to_list_of_matrices(matrices, 'AO matrices')
        out = []
        for k, (a, x) in enumerate(zip(matrices, self.x_ao2orth)):
            value = x.conj().T.dot(a).dot(x)
            error = cp.linalg.norm(value - value.conj().T).item() / value.shape[0]
            if error > _HERMITICITY_TOL:
                raise FloatingPointError('Fock matrix at k-point %d is not Hermitian' % k)
            out.append(0.5 * (value + value.conj().T))
        return out

    def _fock_from_veff(self, dm, veff):
        if getattr(veff, 'v_solvent', None) is not None:
            if not hasattr(self.mf, 'get_fock'):
                raise TypeError('tagged solvent potential requires mf.get_fock')
            fock = self.mf.get_fock(
                h1e=self.hcore, vhf=veff, dm=dm, cycle=-1, diis=None, level_shift_factor=0.0, damp_factor=0.0
            )
            return cp.stack(coerce_to_list_of_matrices(fock, 'decorated Fock matrices'))
        veff_array = cp.asarray(veff)
        return cp.stack([hcore + veff_array[k] for k, hcore in enumerate(self.hcore_ao)])

    def _initial_h(self, dm0=None):
        if dm0 is None:
            try:
                dm0 = self.mf.get_init_guess(self.cell, kpts=self.mf.kpts, key=self.mf.init_guess)
            except TypeError:
                dm0 = self.mf.get_init_guess(self.cell, key=self.mf.init_guess)
        dm = cp.stack(self._hermi(coerce_to_list_of_matrices(dm0, 'initial density')))
        nelec = self.weight * sum(_as_float(
            cp.einsum('ij,ji->', d, s), 'initial electron number')
            for d, s in zip(dm, self.s_ao))
        self.nfev += 1
        veff = self.mf.get_veff(
            self.cell, dm, dm_last=None, vhf_last=None, hermi=1,
            kpts=self.mf.kpts, kpts_band=None)
        h = self._sanitize_h(self._to_orth(self._fock_from_veff(dm, veff)))
        return h, nelec

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
        error = abs(self.nelec_from_eig(orbital_energies, mu) - nelec)
        if error > 1e-10:
            raise RuntimeError('chemical-potential solve has charge error %g' % error)
        return mu

    def nelec_from_eig(self, orbital_energies, mu):
        return 2.0 * self.weight * sum(float(cp.sum(_fermi_occ(self.beta * (x - mu))).item()) for x in orbital_energies)

    def nelec_at_mu(self, h, mu):
        return self.nelec_from_eig([cp.linalg.eigvalsh(x) for x in h], mu)

    def calculate_cycle(self, h, nelec=None, mu=None,
                        align_fixed_n_gauge=True):
        if (nelec is None) == (mu is None):
            raise TypeError('specify exactly one of nelec and mu')
        self.nfev += 1
        h = self._sanitize_h(h)
        eigenpairs = [cp.linalg.eigh(x) for x in h]
        eig = [x[0] for x in eigenpairs]
        coeff = [x[1] for x in eigenpairs]
        mu = self._solve_mu(eig, nelec) if nelec is not None else mu

        occ = []
        p = []
        dm = cp.empty((self.nkpts, self.nao, self.nao),
                      dtype=cp.result_type(*h, cp.complex128))
        entropy_sum = 0.
        for k, (energy, c, x) in enumerate(zip(eig, coeff, self.x_ao2orth)):
            gamma = self.beta * (energy - mu)
            q = _fermi_occ(gamma)
            density = (c * q[None, :]).dot(c.conj().T)
            density = 0.5 * (density + density.conj().T)
            dmk = 2.0 * x.dot(density).dot(x.conj().T)
            dm[k] = 0.5 * (dmk + dmk.conj().T)
            occ.append(q)
            p.append(density)
            entropy_sum += float(_fermi_entropy(q))
        p = self._hermi(p)
        dm = cp.stack(self._hermi([dm[k] for k in range(self.nkpts)]))

        electron_number = 2.0 * self.weight * sum(float(cp.trace(x).real.item()) for x in p)
        ao_nelec = self.weight * sum(
            _as_float(cp.einsum('ij,ji->', d, s), 'AO electron count') for d, s in zip(dm, self.s_ao)
        )
        if abs(electron_number - ao_nelec) > 1e-8 * max(1.0, electron_number):
            raise ValueError('AO and orthogonal electron counts disagree')

        veff = self.mf.get_veff(self.cell, dm, dm_last=None, vhf_last=None, hermi=1, kpts=self.mf.kpts, kpts_band=None)
        fock_ao = self._fock_from_veff(dm, veff)
        e_elec = _as_float(self.mf.energy_elec(dm, self.hcore, veff)[0], 'electronic energy')
        fock = self._to_orth(fock_ao)
        e_tot = e_elec + self.e_nuc
        entropy = 2.0 * self.weight * entropy_sum
        entropy_energy = -self.sigma * entropy
        free_energy = e_tot + entropy_energy

        mismatch = self._hermi([x - y for x, y in zip(h, fock)])
        if nelec is not None and align_fixed_n_gauge:
            gauge = self._trace_mean(mismatch)
            h = [x - gauge*eye for x, eye in zip(h, self.identity)]
            eig = [x - gauge for x in eig]
            mu -= gauge
            mismatch = [x - gauge*eye
                        for x, eye in zip(mismatch, self.identity)]
        residual = [-x for x in mismatch]
        grand_potential = free_energy - mu * electron_number
        if not all(np.isfinite(x) for x in (
                e_tot, free_energy, grand_potential, mu, electron_number)):
            raise FloatingPointError('finite-temperature cycle_data is nonfinite')
        return GCSolverCycle(
            h, eig, coeff, occ, p, dm, fock, mu,
            e_tot, electron_number, entropy, entropy_energy, free_energy,
            grand_potential, residual, self._rms(mismatch),
            getattr(veff, 'lpbe_grid', None))

    def diis_pack(self, blocks, weight_errors=False):
        scale = self.weight**.5 if weight_errors else 1.
        return cp.concatenate([(scale*x).ravel() for x in blocks])

    def diis_unpack(self, vector, template):
        out = []
        offset = 0
        for x in template:
            size = x.size
            out.append(vector[offset:offset+size].reshape(x.shape))
            offset += size
        if offset != vector.size:
            raise ValueError('packed DIIS vector has the wrong size')
        return out

    def start_fixed_n_calc(self, h, nelec):
        cycle_data = self.calculate_cycle(h, nelec=nelec)
        adiis = diis.DIIS(self)
        adiis.space = self.diis_space
        self.cycles += 1

        logger.info(
            self,
            'Fixed-N cycle %d  N = %.12g  mu = %.12g  A = %.12g  '
            'residual = %.6g',
            self.cycles,
            nelec,
            cycle_data.mu,
            cycle_data.free_energy,
            cycle_data.residual_rms,
        )
        return FixedNCalc(float(nelec), cycle_data, adiis, self.damp)

    def diis_trial(self, fixed_n_calc, target):
        cycle_data = fixed_n_calc.cycle_data
        direction = [x - y for x, y in zip(target, cycle_data.h)]
        if not all(bool(cp.all(cp.isfinite(x)).item()) for x in direction):
            return None, 0.0
        damp = min(1.0, max(self.min_damp, fixed_n_calc.damp))
        h = self._sanitize_h(
            [x + damp*d for x, d in zip(cycle_data.h, direction)])
        trial = self.calculate_cycle(h, nelec=fixed_n_calc.nelec)
        logger.debug(
            self, 'DIIS damping %.6g residual %.6g -> %.6g',
            damp, cycle_data.residual_rms, trial.residual_rms)
        return trial, damp

    def diis_update_damp(self, old, new, step, starting):
        predicted = max(0.0, 1.0-step) * old.residual_rms
        predicted_reduction = old.residual_rms - predicted
        actual_reduction = old.residual_rms - new.residual_rms
        scale = max(old.residual_rms, np.finfo(float).tiny)
        if predicted_reduction > np.finfo(float).eps*scale:
            ratio = actual_reduction / predicted_reduction
        else:
            ratio = np.nan
        damp = step
        if (
            np.isfinite(ratio)
            and ratio > self.diis_trust_expand
            and actual_reduction / scale >= self.diis_expand_reduction
            and step >= starting * (1.0 - 1e-12)
            and False
        ):
            damp *= self.diis_expansion
        return min(1.0, max(self.min_damp, damp)), ratio

    def fixed_n_subproblem(self, fixed_n_calc, tolerance, target_mu=None, dump_chk=False):
        fixed_n_calc.converged = False
        fixed_n_calc.message = 'maximum fixed-N cycles reached'
        original_tolerance = tolerance
        while fixed_n_calc.cycles < self.max_cycle:
            cycle_data = fixed_n_calc.cycle_data
            if (target_mu is not None and
                    abs(cycle_data.mu-target_mu) < self.tighten_mu_threshold):
                tolerance = self.conv_tol
            else:
                tolerance = original_tolerance
            if cycle_data.residual_rms <= tolerance:
                fixed_n_calc.converged = True
                fixed_n_calc.message = 'converged fixed-N residual'
                break
            fock = self.diis_pack(cycle_data.fock)
            residual = self.diis_pack(cycle_data.residual, weight_errors=True)
            try:
                target = self.diis_unpack(fixed_n_calc.diis.update(fock, xerr=residual), cycle_data.fock)
                target = self._sanitize_h(target)
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                fixed_n_calc.diis.clear()
                target = self._copy(cycle_data.fock)
            starting = fixed_n_calc.damp
            trial, step = self.diis_trial(fixed_n_calc, target)
            if trial is None:
                trial, step = self.diis_trial(
                    fixed_n_calc, cycle_data.fock)
            if trial is None:
                fixed_n_calc.diis.clear()
                fixed_n_calc.message = 'residual DIIS could not reduce the residual'
                break
            fixed_n_calc.damp, ratio = self.diis_update_damp(
                cycle_data, trial, step, starting)
            fixed_n_calc.cycle_data = trial
            fixed_n_calc.cycles += 1
            self.cycles += 1
            logger.info(
                self,
                'Fixed-N cycle %d  N = %.12g  mu = %.12g  A = %.12g  '
                'residual = %.6g  damping = %.3g  ratio = %.3g',
                self.cycles,
                fixed_n_calc.nelec,
                trial.mu,
                trial.free_energy,
                trial.residual_rms,
                step,
                ratio,
            )

            if dump_chk and self.chkfile:
                mo_coeff = [x.dot(c) for x, c in zip(self.x_ao2orth, trial.coeff)]
                self.dump_chk({
                    'e_tot': trial.e_tot,
                    'mo_coeff': _stack_or_list(mo_coeff),
                    'mo_occ': _stack_or_list([2.*x for x in trial.occ]),
                    'mo_energy': _stack_or_list(trial.eig),
                })

            if callable(self.callback):
                self.callback({
                    'solver': self,
                    'cycle_data': trial,
                    'cycle': self.cycles,
                    'electron_number': fixed_n_calc.nelec,
                })
        cycle_data = fixed_n_calc.cycle_data
        if (target_mu is not None and
                abs(cycle_data.mu-target_mu) < self.tighten_mu_threshold):
            tolerance = self.conv_tol
        else:
            tolerance = original_tolerance
        if cycle_data.residual_rms <= tolerance:
            fixed_n_calc.converged = True
            fixed_n_calc.message = 'converged fixed-N residual'
        return fixed_n_calc

    @staticmethod
    def search_mu_root_bracket(nelecs, delta_mus):
        # Monotonic mu(N) makes the nearest samples on either side the bracket.
        nelecs = np.asarray(nelecs)
        delta_mus = np.asarray(delta_mus)
        negative = np.flatnonzero(delta_mus < 0)
        positive = np.flatnonzero(delta_mus > 0)
        if negative.size == 0 or positive.size == 0:
            return None
        left = negative[np.argmax(nelecs[negative])]
        right = positive[np.argmin(nelecs[positive])]
        return int(left), int(right)

    def secant_proposal(self, samples, h):
        last_sample = samples[-1]
        current_n = last_sample.nelec
        current_error = last_sample.delta_mu
        previous = next((x for x in reversed(samples[:-1]) if abs(x.nelec - current_n) > self.root_nelec_tol), None)
        if previous is None:
            proposal = self.nelec_at_mu(h, self.mu)
            maximum = self.initial_nelec_step
        else:
            denominator = current_error - previous.delta_mu
            proposal = np.nan
            if denominator != 0:
                proposal = current_n - current_error * (current_n - previous.nelec) / denominator
            if not np.isfinite(proposal):
                proposal = self.nelec_at_mu(h, self.mu)
            maximum = self.max_nelec_step_fraction * float(self.cell.nelectron)
        proposal = current_n + np.clip(proposal - current_n, -maximum, maximum)

        bracket_indices = self.search_mu_root_bracket([x.nelec for x in samples], [x.delta_mu for x in samples])
        if bracket_indices is not None:
            bracket = tuple(samples[i] for i in bracket_indices)
            left, right = bracket[0].nelec, bracket[1].nelec
            margin = min(self.root_nelec_tol, 0.25 * (right - left))
            if not left + margin < proposal < right - margin:
                proposal = left + 0.5 * (right - left)
        margin = min(self.root_nelec_tol, 0.25 * self.capacity)
        return min(self.capacity - margin, max(margin, proposal))

    def _kernel_fixed_mu(self, h, current_n, seed_samples=None):
        samples = list(seed_samples or ())
        margin = min(self.root_nelec_tol, 0.25 * self.capacity)
        current_n = min(self.capacity - margin, max(margin, current_n))
        best = None
        best_score = np.inf
        pending = None
        distinct = []

        for unused_pass in range(4 * self.max_outer_cycle + 4):
            distinct_tol = max(self.root_nelec_tol, 32 * np.finfo(float).eps * max(1.0, abs(current_n)))
            is_distinct = not any(abs(x - current_n) <= distinct_tol for x in distinct)
            if is_distinct:
                if len(distinct) >= self.max_outer_cycle:
                    self.message = 'maximum fixed-mu outer cycles reached'
                    break
                distinct.append(float(current_n))
                self.outer_cycles += 1
            else:
                self.refinements += 1

            tolerance = self.conv_tol_coarse

            if samples:
                min_delta_mu = min(abs(x.delta_mu) for x in samples)
                tolerance = min(tolerance, min_delta_mu**1.8)
                tolerance = max(tolerance, self.conv_tol)
            if pending is not None:
                tolerance = self.conv_tol
            logger.info(self, 'Tolerance = %.6g', tolerance)

            if pending is None:
                fixed_n_calc = self.start_fixed_n_calc(h, current_n)
            else:
                fixed_n_calc = pending
                pending = None
            self.fixed_n_subproblem(fixed_n_calc, tolerance, target_mu=self.mu)
            cycle_data = fixed_n_calc.cycle_data
            if not fixed_n_calc.converged:
                if best is None:
                    best = cycle_data
                self.message = 'fixed-N inner solve failed: ' + fixed_n_calc.message
                break

            sample = MuSample(
                cycle_data, cycle_data.mu-self.mu, fixed_n_calc)
            for i, old_sample in enumerate(samples):
                if abs(old_sample.nelec - fixed_n_calc.nelec) <= self.root_nelec_tol:
                    samples.pop(i)
                    break
            samples.append(sample)

            bracket_indices = self.search_mu_root_bracket([x.nelec for x in samples], [x.delta_mu for x in samples])
            bracket = None if bracket_indices is None else tuple(samples[i] for i in bracket_indices)
            retained_samples = list(samples[-2:])
            if bracket is not None:
                retained_samples.extend(bracket)
            for old_sample in samples:
                retained = any(old_sample is x for x in retained_samples)
                if old_sample.fixed_n_calc is not None and not retained:
                    old_sample.fixed_n_calc = None

            error = sample.delta_mu

            score = abs(error) / self.conv_tol_mu
            if score < best_score:
                best_score = score
                best = cycle_data
            logger.info(
                self,
                'Fixed-mu outer cycle %d  N = %.12g  optimized mu = '
                '%.12g  delta mu = %.3g  residual = %.6g',
                self.outer_cycles,
                current_n,
                cycle_data.mu,
                error,
                cycle_data.residual_rms,
            )

            root_ready = abs(error) <= self.conv_tol_mu
            if root_ready and cycle_data.residual_rms > self.conv_tol:
                h = self._copy(cycle_data.h)
                pending = fixed_n_calc
                continue
            if root_ready:
                self.message = 'converged fixed-mu secant search'
                return cycle_data, True

            if bracket is not None:
                if abs(bracket[1].nelec - bracket[0].nelec) <= self.root_nelec_tol:
                    self.message = 'fixed-mu electron-number bracket stagnated'
                    break

            proposal = self.secant_proposal(samples, cycle_data.h)
            if abs(proposal - current_n) <= 1e-14 * max(1.0, abs(current_n)):
                self.message = 'fixed-mu secant search stagnated'
                break

            h = self._copy(cycle_data.h)
            current_n = proposal
        else:  # pragma: no cover
            self.message = 'fixed-mu safety iteration limit reached'

        if best is None:
            raise RuntimeError(self.message or 'fixed-mu search produced no cycle_data')
        return best, False

    def _initialize_scan_output(self, output_prefix, columns):
        prefix = Path(output_prefix)
        csv_path = Path(str(prefix) + '.csv')
        h5_path = Path(str(prefix) + '.h5')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
        with h5py.File(h5_path, 'w') as h5f:
            h5f.create_dataset('sigma', data=self.sigma)
            for name, dtype in columns.items():
                if dtype is str:
                    dtype = h5py.string_dtype(encoding='utf-8')
                h5f.create_dataset(
                    name, shape=(0,), maxshape=(None,), dtype=dtype)
            h5f.create_group('points')
            h5f.attrs['completed_points'] = 0
        return csv_path, h5_path

    def _append_scan_output(
            self, csv_path, h5_path, columns, record, cycle_data):
        index = record['index']
        with h5py.File(h5_path, 'a') as h5f:
            point = h5f['points'].create_group(str(index))
            data = {
                'mo_coeff': [cp.asnumpy(x.dot(c)) for x, c in zip(
                    self.x_ao2orth, cycle_data.coeff)],
                'mo_occ': [cp.asnumpy(2.*x) for x in cycle_data.occ],
                'mo_energy': [cp.asnumpy(x) for x in cycle_data.eig],
            }
            for name, values in data.items():
                group = point.create_group(name)
                for k, value in enumerate(values):
                    group.create_dataset(str(k), data=value)
            for name in columns:
                dataset = h5f[name]
                dataset.resize(index+1, axis=0)
                dataset[index] = record[name]
            point.attrs['complete'] = True
            h5f.attrs['completed_points'] = index+1
            h5f.flush()
        with csv_path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writerow(record)
            stream.flush()

    def nelec_scan(self, nelecs, output_prefix, dm0=None,
                   nelec_scan_h5=None):
        self.build()
        nelecs = [_as_float(x, 'nelec') for x in nelecs]
        if not nelecs:
            raise ValueError('nelecs must not be empty')
        for nelec in nelecs:
            if not 0 < nelec < self.capacity:
                raise ValueError(
                    'nelec must lie between 0 and %g' % self.capacity)
        restart_points = (None if nelec_scan_h5 is None else
                          self._nelec_scan_restarts(nelec_scan_h5))

        self.converged = False
        self.cycles = 0
        self.outer_cycles = 0
        self.nfev = 0
        self.refinements = 0
        self.message = ''
        h = None if restart_points is not None else self._initial_h(dm0)[0]
        records = []
        last_cycle_data = None
        last_converged = False
        columns = {
            'index': np.int64,
            'requested_nelec': np.float64,
            'electron_number': np.float64,
            'converged': np.bool_,
            'message': str,
            'mu': np.float64,
            'e_tot': np.float64,
            'free_energy': np.float64,
            'grand_potential': np.float64,
            'entropy': np.float64,
            'entropy_energy': np.float64,
            'residual_rms': np.float64,
            'cycles': np.int64,
            'fock_evaluations': np.int64,
        }
        csv_path, h5_path = self._initialize_scan_output(
            output_prefix, columns)

        for index, nelec in enumerate(nelecs):
            cycle_start = self.cycles
            nfev_start = self.nfev
            if restart_points is not None:
                nearest = min(
                    restart_points,
                    key=lambda x: (
                        abs(x['nelec']-nelec), x['residual_rms']))
                if (last_cycle_data is None or
                        abs(nearest['nelec']-nelec) <
                        abs(last_cycle_data.nelec-nelec)):
                    h = self._initial_h(nearest['dm'])[0]
            fixed_n_calc = self.start_fixed_n_calc(h, nelec)
            self.fixed_n_subproblem(fixed_n_calc, self.conv_tol)
            cycle_data = fixed_n_calc.cycle_data
            record = {
                'index': index,
                'requested_nelec': nelec,
                'electron_number': cycle_data.nelec,
                'converged': bool(fixed_n_calc.converged),
                'message': fixed_n_calc.message,
                'mu': cycle_data.mu,
                'e_tot': cycle_data.e_tot,
                'free_energy': cycle_data.free_energy,
                'grand_potential': cycle_data.grand_potential,
                'entropy': cycle_data.entropy,
                'entropy_energy': cycle_data.entropy_energy,
                'residual_rms': cycle_data.residual_rms,
                'cycles': self.cycles-cycle_start,
                'fock_evaluations': self.nfev-nfev_start,
            }
            records.append(record)
            self._append_scan_output(
                csv_path, h5_path, columns, record, cycle_data)

            h = self._copy(cycle_data.h)
            last_cycle_data = cycle_data
            last_converged = fixed_n_calc.converged

        self.nelec = nelecs[-1]
        self.message = records[-1]['message']
        self._finalize(last_cycle_data, last_converged)

        return records

    @staticmethod
    def _read_nelec_scan_csv(filename):
        required = {
            'index', 'electron_number', 'converged', 'mu', 'residual_rms'}
        records = []
        with Path(filename).open(newline='') as stream:
            reader = csv.DictReader(stream)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    'N-scan CSV is missing columns: ' +
                    ', '.join(sorted(missing)))
            for row in reader:
                converged = row['converged'].strip().lower()
                if converged not in ('true', 'false'):
                    raise ValueError('invalid converged value in N-scan CSV')
                records.append({
                    'index': int(row['index']),
                    'nelec': float(row['electron_number']),
                    'mu': float(row['mu']),
                    'converged': converged == 'true',
                    'residual_rms': float(row['residual_rms']),
                })
        return records

    def _read_nelec_scan_h5(self, filename):
        required = {
            'index', 'electron_number', 'converged', 'mu', 'residual_rms'}
        records = []
        with h5py.File(filename, 'r') as h5f:
            missing = required.difference(h5f.keys())
            if missing:
                raise ValueError(
                    'N-scan HDF5 is missing datasets: ' +
                    ', '.join(sorted(missing)))
            if 'sigma' not in h5f:
                raise ValueError('N-scan HDF5 is missing sigma')
            sigma = float(h5f['sigma'][()])
            if not np.isclose(sigma, self.sigma, rtol=1e-12, atol=1e-14):
                raise ValueError('N-scan sigma does not match solver sigma')
            completed = int(h5f.attrs.get(
                'completed_points', len(h5f['index'])))
            if any(len(h5f[name]) != completed for name in required):
                raise ValueError('N-scan HDF5 scalar datasets are incomplete')
            if 'points' not in h5f or len(h5f['points']) != completed:
                raise ValueError('N-scan HDF5 point groups are incomplete')
            for i in range(completed):
                index = int(h5f['index'][i])
                key = str(index)
                if (key not in h5f['points'] or
                        not bool(h5f['points'][key].attrs.get(
                            'complete', False))):
                    raise ValueError(
                        'N-scan HDF5 contains an incomplete point')
                records.append({
                    'index': index,
                    'nelec': float(h5f['electron_number'][i]),
                    'mu': float(h5f['mu'][i]),
                    'converged': bool(h5f['converged'][i]),
                    'residual_rms': float(h5f['residual_rms'][i]),
                })
        return records

    @staticmethod
    def _crosscheck_nelec_scan_records(csv_records, h5_records):
        csv_records = sorted(csv_records, key=lambda x: x['index'])
        h5_records = sorted(h5_records, key=lambda x: x['index'])
        if len(csv_records) != len(h5_records):
            raise ValueError('N-scan CSV and HDF5 point counts differ')
        for csv_record, h5_record in zip(csv_records, h5_records):
            if (csv_record['index'] != h5_record['index'] or
                    csv_record['converged'] != h5_record['converged']):
                raise ValueError('N-scan CSV and HDF5 records differ')
            for name in ('nelec', 'mu', 'residual_rms'):
                if not np.isclose(
                        csv_record[name], h5_record[name],
                        rtol=1e-10, atol=1e-12, equal_nan=True):
                    raise ValueError('N-scan CSV and HDF5 records differ')

    def _prepare_mu_calibration(self, records):
        records = [dict(record) for record in records
                   if record['converged'] and
                   all(np.isfinite(record[name]) for name in (
                       'nelec', 'mu', 'residual_rms'))]
        records.sort(key=lambda x: x['nelec'])
        unique = []
        for record in records:
            if not 0 < record['nelec'] < self.capacity:
                raise ValueError(
                    'calibration nelec must lie between 0 and %g' %
                    self.capacity)
            if (unique and abs(record['nelec']-unique[-1]['nelec']) <=
                    self.root_nelec_tol):
                if record['residual_rms'] < unique[-1]['residual_rms']:
                    unique[-1] = record
            else:
                unique.append(record)
        if len(unique) < 2:
            raise ValueError(
                'at least two converged N-scan points are required')
        if any(right['mu'] <= left['mu']
               for left, right in zip(unique, unique[1:])):
            raise ValueError('N-scan mu must increase strictly with nelec')
        return unique

    def _mu_scan_seed(self, calibration, target_mu):
        nelecs = np.asarray([x['nelec'] for x in calibration])
        mus = np.asarray([x['mu'] for x in calibration])
        if target_mu < mus[0]:
            left, right = 0, 1
            nelec = (nelecs[left] +
                     (target_mu-mus[left]) *
                     (nelecs[right]-nelecs[left]) /
                     (mus[right]-mus[left]))
        elif target_mu > mus[-1]:
            left, right = len(mus)-2, len(mus)-1
            nelec = (nelecs[left] +
                     (target_mu-mus[left]) *
                     (nelecs[right]-nelecs[left]) /
                     (mus[right]-mus[left]))
        elif target_mu == mus[0]:
            left, right = 0, 1
            nelec = nelecs[0]
        elif target_mu == mus[-1]:
            left, right = len(mus)-2, len(mus)-1
            nelec = nelecs[-1]
        else:
            model = PchipInterpolator(nelecs, mus, extrapolate=False)
            roots = np.asarray(
                model.solve(y=target_mu, extrapolate=False))
            roots = roots[
                np.isfinite(roots) &
                (roots >= nelecs[0]-self.root_nelec_tol) &
                (roots <= nelecs[-1]+self.root_nelec_tol)]
            roots.sort()
            distinct = []
            for root in roots:
                if (not distinct or
                        abs(root-distinct[-1]) > self.root_nelec_tol):
                    distinct.append(float(root))
            if len(distinct) != 1:
                raise RuntimeError(
                    'PCHIP mu(N) interpolation did not have a unique root')
            nelec = distinct[0]
            right = min(len(mus)-1, int(np.searchsorted(
                mus, target_mu, side='right')))
            left = right-1

        margin = min(self.root_nelec_tol, 0.25*self.capacity)
        nelec = min(self.capacity-margin, max(margin, float(nelec)))
        pair = [calibration[left], calibration[right]]
        pair.sort(key=lambda x: abs(x['nelec']-nelec), reverse=True)
        seeds = [MuSeed(x['nelec'], x['mu']-target_mu) for x in pair]
        return nelec, seeds

    def _density_from_nelec_scan_h5(self, filename, index):
        with h5py.File(filename, 'r') as h5f:
            point = h5f['points'][str(index)]
            coeff_group = point['mo_coeff']
            occ_group = point['mo_occ']
            coeff_keys = sorted(coeff_group, key=int)
            occ_keys = sorted(occ_group, key=int)
            expected = [str(k) for k in range(self.nkpts)]
            if coeff_keys != expected or occ_keys != expected:
                raise ValueError(
                    'N-scan orbital k-points do not match the solver')
            dm = []
            for key in expected:
                coeff = cp.asarray(coeff_group[key][()])
                occ = cp.asarray(occ_group[key][()])
                if (coeff.ndim != 2 or occ.ndim != 1 or
                        coeff.shape != (self.nao, occ.size)):
                    raise ValueError('invalid N-scan orbital dimensions')
                value = (coeff*occ[None, :]).dot(coeff.conj().T)
                dm.append(0.5*(value+value.conj().T))
        return cp.stack(dm)

    def _nelec_scan_restarts(self, filename):
        records = [dict(record) for record in
                   self._read_nelec_scan_h5(filename)
                   if record['converged'] and
                   0 < record['nelec'] < self.capacity and
                   all(np.isfinite(record[name]) for name in (
                       'nelec', 'mu', 'residual_rms'))]
        records.sort(key=lambda x: x['nelec'])
        unique = []
        for record in records:
            if (unique and abs(record['nelec']-unique[-1]['nelec']) <=
                    self.root_nelec_tol):
                if record['residual_rms'] < unique[-1]['residual_rms']:
                    unique[-1] = record
            else:
                unique.append(record)
        if not unique:
            raise ValueError(
                'N-scan HDF5 contains no usable converged points')
        for record in unique:
            record['dm'] = self._density_from_nelec_scan_h5(
                filename, record['index'])
        return unique

    def mu_scan(self, mus, output_prefix, nelec_scan_csv=None,
                nelec_scan_h5=None, dm0=None):
        self.build()
        mus = [_as_float(x, 'mu') for x in mus]
        if not mus:
            raise ValueError('mus must not be empty')
        if nelec_scan_csv is None and nelec_scan_h5 is None:
            raise TypeError(
                'nelec_scan_csv or nelec_scan_h5 is required')

        csv_records = (None if nelec_scan_csv is None else
                       self._read_nelec_scan_csv(nelec_scan_csv))
        h5_records = (None if nelec_scan_h5 is None else
                      self._read_nelec_scan_h5(nelec_scan_h5))
        if csv_records is not None and h5_records is not None:
            self._crosscheck_nelec_scan_records(csv_records, h5_records)
        calibration = self._prepare_mu_calibration(
            h5_records if h5_records is not None else csv_records)
        initial_nelec, unused_seeds = self._mu_scan_seed(
            calibration, mus[0])

        self.converged = False
        self.cycles = 0
        self.outer_cycles = 0
        self.nfev = 0
        self.refinements = 0
        self.message = ''
        self.nelec = None
        if nelec_scan_h5 is not None:
            nearest = min(
                calibration,
                key=lambda x: abs(x['nelec']-initial_nelec))
            dm0 = self._density_from_nelec_scan_h5(
                nelec_scan_h5, nearest['index'])
        h, unused_nelec = self._initial_h(dm0)

        columns = {
            'index': np.int64,
            'requested_mu': np.float64,
            'optimized_mu': np.float64,
            'interpolated_nelec': np.float64,
            'electron_number': np.float64,
            'converged': np.bool_,
            'message': str,
            'e_tot': np.float64,
            'free_energy': np.float64,
            'grand_potential': np.float64,
            'entropy': np.float64,
            'entropy_energy': np.float64,
            'residual_rms': np.float64,
            'cycles': np.int64,
            'outer_cycles': np.int64,
            'refinements': np.int64,
            'fock_evaluations': np.int64,
        }
        csv_path, h5_path = self._initialize_scan_output(
            output_prefix, columns)
        records = []
        last_cycle_data = None
        last_converged = False

        for index, target_mu in enumerate(mus):
            interpolated_nelec, seed_samples = self._mu_scan_seed(
                calibration, target_mu)
            cycle_start = self.cycles
            outer_start = self.outer_cycles
            refinement_start = self.refinements
            nfev_start = self.nfev
            self.mu = target_mu
            self.message = ''
            cycle_data, converged = self._kernel_fixed_mu(
                h, interpolated_nelec, seed_samples=seed_samples)
            record = {
                'index': index,
                'requested_mu': target_mu,
                'optimized_mu': cycle_data.mu,
                'interpolated_nelec': interpolated_nelec,
                'electron_number': cycle_data.nelec,
                'converged': bool(converged),
                'message': self.message,
                'e_tot': cycle_data.e_tot,
                'free_energy': cycle_data.free_energy,
                'grand_potential': cycle_data.grand_potential,
                'entropy': cycle_data.entropy,
                'entropy_energy': cycle_data.entropy_energy,
                'residual_rms': cycle_data.residual_rms,
                'cycles': self.cycles-cycle_start,
                'outer_cycles': self.outer_cycles-outer_start,
                'refinements': self.refinements-refinement_start,
                'fock_evaluations': self.nfev-nfev_start,
            }
            records.append(record)
            self._append_scan_output(
                csv_path, h5_path, columns, record, cycle_data)
            h = self._copy(cycle_data.h)
            last_cycle_data = cycle_data
            last_converged = converged
            if converged:
                calibration = self._prepare_mu_calibration(
                    calibration + [{
                        'index': None,
                        'nelec': cycle_data.nelec,
                        'mu': cycle_data.mu,
                        'converged': True,
                        'residual_rms': cycle_data.residual_rms,
                    }])

        self.nelec = None
        self.message = records[-1]['message']
        self._finalize(last_cycle_data, last_converged)
        return records

    def _finalize(self, cycle_data, converged):
        mo_coeff = [x.dot(c) for x, c in zip(self.x_ao2orth, cycle_data.coeff)]
        mo_energy = [e for e in cycle_data.eig]
        mo_occ = [2.*x for x in cycle_data.occ]
        self.mo_coeff = _stack_or_list(mo_coeff)
        self.mo_energy = _stack_or_list(mo_energy)
        self.mo_occ = _stack_or_list(mo_occ)
        self.converged = bool(converged)
        self.e_tot = cycle_data.e_tot
        self.free_energy = cycle_data.free_energy
        self.grand_potential = cycle_data.grand_potential
        self.electron_number = cycle_data.nelec
        self.mu = cycle_data.mu
        self.entropy = cycle_data.entropy
        self.entropy_energy = cycle_data.entropy_energy
        self.residual_rms = cycle_data.residual_rms
        self._cycle_data = cycle_data
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

    def kernel(self, dm0=None, h=None, initial_nelec=None, dump_chk=True):
        self.build()
        self.dump_flags()
        self.converged = False
        self.cycles = 0
        self.outer_cycles = 0
        self.nfev = 0
        self.refinements = 0
        self.message = ''

        if dump_chk and self.chkfile:
            chkfile.save_mol(self.cell, self.chkfile)

        if h is None:
            h, initial_nelec = self._initial_h(dm0)
        if initial_nelec is None:
            initial_nelec = self.nelec
        if self.nelec is None:
            logger.info(self, 'Initial density electron number = %.12g',
                        initial_nelec)
            cycle_data, converged = self._kernel_fixed_mu(h, initial_nelec)
        else:
            fixed_n_calc = self.start_fixed_n_calc(h, self.nelec)
            self.fixed_n_subproblem(fixed_n_calc, self.conv_tol, dump_chk=dump_chk)
            self.message = fixed_n_calc.message
            cycle_data, converged = fixed_n_calc.cycle_data, fixed_n_calc.converged
        logger.info(self, '%s; total Fock evaluations = %d',
                    self.message, self.nfev)
        return self._finalize(cycle_data, converged)

    scf = kernel
