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

'''Finite-temperature grand-canonical KRKS and KUKS for periodic systems.

Fixed-electron-number problems are solved with residual DIIS.  At fixed
chemical potential, a safeguarded secant iteration selects the electron
number of a sequence of fixed-N problems.  This avoids applying DIIS directly
to the discontinuously changing low-temperature occupations.

Internally every matrix quantity (auxiliary Hamiltonian, Fock matrix,
orbitals, occupations) is a flat list of blocks.  For KRKS there is one block
per k-point; for KUKS there are 2*nkpts blocks ordered spin-major,
``[alpha k0, alpha k1, ..., beta k0, beta k1, ...]``.  Both spins share a
single chemical potential, so in KUKS only the total electron number is
fixed (or controlled by mu) and the magnetization relaxes self-consistently.
'''

import csv
import threading
from pathlib import Path

import numpy as np
import cupy as cp
import h5py
from scipy.interpolate import PchipInterpolator
from scipy.special import expit

from pyscf import __config__, lib
from pyscf.scf import chkfile

from gpu4pyscf.lib import diis, logger
from gpu4pyscf.lib.cusolver import eigh_batched
from gpu4pyscf.scf import hf as mol_hf
from gpu4pyscf.pbc.dft.grand_canonical_cg import fixed_mu_diis, fixed_mu_diis_inner, nlcg


__all__ = ['GrandCanonicalKRKS', 'GrandCanonicalKS', 'GrandCanonicalKUKS']


# Relative to max|F| of the AO Fock matrix.
_HERMITICITY_TOL = getattr(
    __config__, 'pbc_dft_grand_canonical_hermiticity_tol', 1e-10)
# Absolute floor on max|X^H S X - 1|; raised to the rounding error of forming
# X^H S X when the overlap matrix is badly conditioned (see build()).
_ORTHOGONALITY_TOL = getattr(
    __config__, 'pbc_dft_grand_canonical_orthogonality_tol', 1e-9)


class _BackgroundWriter:
    '''Run host-side file writes (h5py, csv) in a background thread.

    h5py releases the GIL inside HDF5 reads and writes, so the thread that drives the
    GPU keeps launching work while a scan point or chkfile goes to disk.  Callers move
    every array to the host before submit().  At most one write is in flight: submit()
    and wait() join the previous one and re-raise its exception.
    '''
    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self._thread = None
        self._error = None
        self._label = None

    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    def wait(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError('background write %s failed' % self._label) from error

    def submit(self, fn, *args, label=''):
        self.wait()
        self._label = label
        if not self.enabled:
            fn(*args)
            return

        def run():
            try:
                fn(*args)
            except BaseException as e:   # re-raised by the next wait()
                self._error = e
        self._thread = threading.Thread(target=run, name='gc-writer', daemon=False)
        self._thread.start()


def _eigh_blocks(blocks, vectors=True):
    '''cupy.linalg.eigh (eigvalsh) of every Hermitian block, batched over blocks of
    equal size.  Returns lists (w, v) or w, in block order.'''
    groups = {}
    for i, x in enumerate(blocks):
        groups.setdefault((x.shape, x.dtype), []).append(i)
    w = [None] * len(blocks)
    v = [None] * len(blocks)
    for idx in groups.values():
        if len(idx) == 1:
            x = blocks[idx[0]]
            if vectors:
                w[idx[0]], v[idx[0]] = cp.linalg.eigh(x)
            else:
                w[idx[0]] = cp.linalg.eigvalsh(x)
            continue
        out = eigh_batched(cp.stack([blocks[i] for i in idx]), vectors=vectors)
        if vectors:
            for j, i in enumerate(idx):
                w[i], v[i] = out[0][j], out[1][j]
        else:
            for j, i in enumerate(idx):
                w[i] = out[j]
    return (w, v) if vectors else w


def _write_scan_point(csv_path, h5_path, columns, record, data, occ_threshold):
    '''Append one N/mu-scan point (host arrays only) to the scan h5, then its CSV row.'''
    index = record['index']
    with h5py.File(h5_path, 'a') as h5f:
        point = h5f['points'].create_group(str(index))
        for name, values in data.items():
            group = point.create_group(name)
            for k, value in enumerate(values):
                group.create_dataset(str(k), data=value)
        for name in columns:
            dataset = h5f[name]
            dataset.resize(index+1, axis=0)
            dataset[index] = record[name]
        if occ_threshold is not None:
            h5f.attrs['mo_occ_threshold'] = occ_threshold
        point.attrs['complete'] = True
        h5f.attrs['completed_points'] = index+1
        h5f.flush()
    with open(csv_path, 'a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writerow(record)
        stream.flush()


def _write_chk(chk_path, cell, kpts, envs):
    '''mf.dump_chk of a periodic SCF, from host arrays.'''
    chkfile.dump_scf(cell, chk_path, envs['e_tot'], envs['mo_energy'],
                     envs['mo_coeff'], envs['mo_occ'], overwrite_mol=False)
    with lib.H5FileWrap(chk_path, 'a') as fh5:
        fh5['scf/kpts'] = kpts


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


def _uniform_stack(blocks):
    '''cp.stack(blocks) when every block has the same shape and dtype, else None.'''
    if len(blocks) == 0:
        return None
    first = blocks[0]
    if any(x.shape != first.shape or x.dtype != first.dtype for x in blocks):
        return None
    return cp.stack(blocks)


def _ht(a):
    '''Conjugate transpose of every matrix in a (..., n, m) stack.'''
    return a.conj().swapaxes(-1, -2)


def _concat_ravel(arrays):
    return cp.concatenate([cp.asarray(x).ravel() for x in arrays])


def _fermi_occ(gamma):
    return expit(-cp.asarray(gamma))


def _fermi_entropy(occ):
    # cp.where instead of boolean indexing: no host sync for the mask size
    f = cp.asarray(occ)
    inside = (f > 0) & (f < 1)
    g = cp.where(inside, f, 0.5)
    return cp.where(inside, -(g*cp.log(g) + (1-g)*cp.log(1-g)), 0.).sum()


class GCSolverCycle:
    '''One evaluation of the finite-temperature KS map.

    ``h``, ``eig``, ``coeff``, ``occ``, ``p``, ``fock`` and ``residual`` are
    lists of blocks (see the module docstring); ``dm`` has the shape of the
    underlying mf (nkpts,nao,nao) or (2,nkpts,nao,nao).  ``spin_nelec`` holds
    the electron number of each spin channel (one entry for KRKS).
    '''
    __slots__ = (
        'h', 'eig', 'coeff', 'occ', 'p', 'dm', 'fock',
        'mu', 'e_tot', 'nelec',
        'entropy', 'entropy_energy', 'free_energy', 'grand_potential',
        'residual', 'residual_rms', 'grid_data', 'spin_nelec')

    def __init__(self, h, eig, coeff, occ, p, dm, fock, mu,
                 e_tot, nelec, entropy, entropy_energy, free_energy,
                 grand_potential, residual, residual_rms, grid_data=None,
                 spin_nelec=None):
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
        self.spin_nelec = spin_nelec

    @property
    def magnetization(self):
        '''N_alpha - N_beta per cell (0 for KRKS).'''
        if self.spin_nelec is None or len(self.spin_nelec) < 2:
            return 0.
        return self.spin_nelec[0] - self.spin_nelec[1]


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


class _GrandCanonicalKSBase(lib.StreamObject):
    '''Spin-generic finite-temperature KS solver; use GrandCanonicalKRKS or
    GrandCanonicalKUKS.

    Subclasses set ``nspin`` (number of spin channels, 1 or 2) and
    ``_mf_type`` (the class name the mf must have).  Each orbital holds
    ``occ_scale = 2/nspin`` electrons when fully occupied.
    '''

    nspin = None
    _mf_type = None

    nlcg = nlcg
    fixed_mu_diis = fixed_mu_diis
    fixed_mu_diis_inner = fixed_mu_diis_inner

    max_cycle = getattr(__config__, 'pbc_dft_grand_canonical_max_cycle', 100)
    max_outer_cycle = getattr(__config__, 'pbc_dft_grand_canonical_max_outer_cycle', 16)
    conv_tol = getattr(__config__, 'pbc_dft_grand_canonical_conv_tol', 1e-8)
    conv_tol_coarse = getattr(__config__, 'pbc_dft_grand_canonical_conv_tol_coarse', 4e-6)
    conv_tol_mu = getattr(__config__, 'pbc_dft_grand_canonical_conv_tol_mu', 1e-6)
    diis_max_dmu = getattr(__config__, 'pbc_dft_grand_canonical_diis_max_dmu', 5e-3)
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
        __config__, 'pbc_dft_grand_canonical_tighten_mu_threshold', 5e-4)
    diis_space = getattr(__config__, 'pbc_dft_grand_canonical_diis_space', 12)
    damp = getattr(__config__, 'pbc_dft_grand_canonical_damp', 0.8)
    diis_trust_expand = getattr(__config__, 'pbc_dft_grand_canonical_diis_trust_expand', 0.75)
    diis_expansion = getattr(__config__, 'pbc_dft_grand_canonical_diis_expansion', 2.0)
    diis_expand_reduction = getattr(__config__, 'pbc_dft_grand_canonical_diis_expand_reduction', 2e-2)
    min_damp = getattr(__config__, 'pbc_dft_grand_canonical_min_damp', 1e-8)
    initial_nelec_step = getattr(__config__, 'pbc_dft_grand_canonical_initial_nelec_step', 3e-2)
    max_nelec_step_fraction = getattr(__config__, 'pbc_dft_grand_canonical_max_nelec_step_fraction', 0.1)
    root_nelec_tol = getattr(__config__, 'pbc_dft_grand_canonical_root_nelec_tol', 1e-8)
    callback = None
    # Write scan points and chkfiles from a background thread while the GPU works on.
    async_io = getattr(__config__, 'pbc_dft_grand_canonical_async_io', True)
    # Diagonalize one k of each (k, -k) pair (k-point sets closed under inversion).
    # N-scans: start each point from H linearly extrapolated in N through the two
    # previous converged points (0 starts from the previous point's H).
    scan_extrapolate = getattr(__config__, 'pbc_dft_grand_canonical_scan_extrapolate', 0)
    time_reversal_eigh = getattr(
        __config__, 'pbc_dft_grand_canonical_time_reversal_eigh', True)
    # Scan h5: keep only MOs with occupation >= this (None keeps every MO).  The
    # density, hence every restart/seed, is unchanged by dropping MOs with
    # occupation below ~1e-18.
    scan_mo_occ_threshold = getattr(
        __config__, 'pbc_dft_grand_canonical_scan_mo_occ_threshold', None)

    _keys = {
        'async_io', 'scan_mo_occ_threshold', 'time_reversal_eigh', 'scan_extrapolate',
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
        'potential_residual_rms', 'magnetization',
    }

    def __init__(self, mf, mu=None, sigma=None, nelec=None):
        if sigma is None:
            raise TypeError('sigma is required')
        sigma = _as_float(sigma, 'sigma')
        if sigma <= 0:
            raise ValueError('sigma must be positive')
        if nelec is None and mu is None:
            nelec = mf.cell.tot_electrons(1)
            logger.info(self, "Setting nelec to %f by default." % nelec)
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
        self.magnetization = None
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.scf_summary = {}

        self._built = False
        self._cycle_data = None
        self._potential_cycle = None
        self._writer = None
        self._chk_pending = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('ensemble = %s', 'fixed N' if self.nelec is not None else
                 'fixed mu')
        if self.nspin == 2:
            log.info('spin channels = 2 (common mu, magnetization relaxes)')
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

    def _dump_chk_async(self, cycle_data, force=False):
        '''Dump cycle_data to the chkfile from the background writer.  A dump requested
        while the previous one is still being written is deferred (only the newest
        deferred cycle is kept); _flush_chk(), at the end of kernel(), writes it.'''
        if not self.chkfile:
            return
        writer = self._get_writer()
        if not force and writer.busy():
            self._chk_pending = cycle_data
            return
        self._chk_pending = None
        self.mf.chkfile = self.chkfile
        kpts = np.asarray(cp.asnumpy(cp.asarray(self.mf.kpts)), dtype=float)
        writer.submit(_write_chk, self.chkfile, self.cell, kpts,
                      self._chk_envs(cycle_data), label='chkfile')

    def _flush_chk(self):
        if self._chk_pending is not None:
            self._dump_chk_async(self._chk_pending, force=True)
        self.wait_io()

    def _chk_envs(self, cycle_data):
        '''Orbitals of cycle_data for dump_chk, padded to nao at every k-point
        the way KSCF.eig pads k-points that dropped linearly dependent AOs:
        zero coefficients, zero occupation, and a large orbital energy.
        Arrays have shape (nkpts,...) for KRKS and (2,nkpts,...) for KUKS.'''
        mo_coeff = [cp.asnumpy(x.dot(c))
                    for x, c in zip(self.x_ao2orth, cycle_data.coeff)]
        nblocks = len(mo_coeff)
        coeff = np.zeros((nblocks, self.nao, self.nao),
                         dtype=np.result_type(*mo_coeff))
        energy = np.empty((nblocks, self.nao))
        occ = np.zeros((nblocks, self.nao))
        for k, (c, e, q) in enumerate(zip(mo_coeff, cycle_data.eig,
                                          cycle_data.occ)):
            e = cp.asnumpy(e)
            nmo = e.size
            coeff[k, :, :nmo] = c
            energy[k, :nmo] = e
            energy[k, nmo:] = abs(e.max()) * 2 + 1e5
            occ[k, :nmo] = cp.asnumpy(self.occ_scale*q)
        shape = self._spin_kpts_shape
        return {'e_tot': cycle_data.e_tot,
                'mo_coeff': coeff.reshape(shape + coeff.shape[1:]),
                'mo_occ': occ.reshape(shape + occ.shape[1:]),
                'mo_energy': energy.reshape(shape + energy.shape[1:])}

    def build(self):
        if self._built:
            return self
        self.check_sanity()
        required = ('cell', 'kpts', 'get_ovlp', 'get_hcore', 'get_veff',
                    'energy_elec', 'energy_nuc', 'get_init_guess',
                    'check_linear_dependency')
        missing = [name for name in required if not hasattr(self.mf, name)]
        if missing:
            raise TypeError('mf is not %s-compatible; missing %s' % (
                self._mf_type, ', '.join(missing)))
        if hasattr(self.mf, 'istype') and not self.mf.istype(self._mf_type):
            raise TypeError('%s requires a periodic %s object' % (
                self.__class__.__name__, self._mf_type))
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
        self._kinv = self._inversion_partners()
        if self.nspin not in (1, 2):
            raise TypeError('nspin must be 1 or 2')
        self.occ_scale = 2. / self.nspin
        self._spin_kpts_shape = ((self.nkpts,) if self.nspin == 1 else
                                 (self.nspin, self.nkpts))

        self.mf.chkfile = self.chkfile

        if hasattr(self.mf, 'build'):
            self.mf.build()
        s_ao = coerce_to_list_of_matrices(
            self.mf.get_ovlp(self.cell, self.mf.kpts), 'overlap')
        hcore_ao = coerce_to_list_of_matrices(
            self.mf.get_hcore(self.cell, self.mf.kpts), 'hcore')
        if len(s_ao) != self.nkpts or len(hcore_ao) != self.nkpts:
            raise ValueError('overlap, hcore, and k-point counts differ')
        x = self.mf.check_linear_dependency(_stack_or_list(s_ao))
        x_ao2orth = coerce_to_list_of_matrices(x, 'orthogonalizers')
        # With X(-k) = X(k)* the orthogonal-basis Hamiltonian obeys H(-k) = H(k)*
        # whenever the density does (see _time_reversal), so only one k of each
        # (k, -k) pair needs an eigendecomposition.
        self._x_time_reversal = bool(self.time_reversal_eigh and
                                     self._kinv is not None)
        if self._x_time_reversal:
            x_ao2orth = [x_ao2orth[j].conj() if j < k else x_ao2orth[k]
                         for k, j in enumerate(self._kinv)]
        if len(x_ao2orth) != self.nkpts:
            raise ValueError('orthogonalizer and k-point counts differ')

        self.nao = s_ao[0].shape[0]
        north = []
        identity = []
        worst = 0.
        for k, (s, h, x) in enumerate(zip(s_ao, hcore_ao, x_ao2orth)):
            if s.shape != (self.nao, self.nao) or h.shape != s.shape:
                raise ValueError('inconsistent AO dimensions at k-point %d' % k)
            if x.shape[0] != self.nao or x.shape[1] == 0:
                raise ValueError('invalid orthogonalizer at k-point %d' % k)
            error = cp.max(cp.abs(
                x.conj().T.dot(s).dot(x) - cp.eye(x.shape[1]))).item()
            # Rounding in X^H S X is up to about nao*eps*||S||*||X||^2, where
            # ||S||*||X||^2 is cond(S) on the retained space (~1e6 for Cu and
            # Au slabs).  A wrong X gives errors of order one.
            cond = (cp.linalg.eigvalsh(s)[-1] *
                    cp.linalg.eigvalsh(x.conj().T.dot(x))[-1]).item()
            tol = max(_ORTHOGONALITY_TOL,
                      self.nao * np.finfo(float).eps * cond)
            if error > tol:
                raise ValueError(
                    'X^H S X is not identity at k-point %d: max error %.3g, '
                    'tolerance %.3g, cond(S) %.3g' % (k, error, tol, cond))
            worst = max(worst, error / tol)
            north.append(x.shape[1])
            identity.append(cp.eye(x.shape[1], dtype=x.dtype))
        logger.debug(self, 'max |X^H S X - 1| / tolerance = %.3g', worst)

        # Per-block AO data: the k-point lists repeated for each spin, so
        # they zip directly with the spin-major block lists of a cycle.
        self.s_ao = s_ao * self.nspin
        self.hcore_ao = hcore_ao * self.nspin
        self.x_ao2orth = x_ao2orth * self.nspin
        self.north = north * self.nspin
        self.identity = identity * self.nspin
        self.nblocks = self.nspin * self.nkpts
        self.ndof = self.weight * sum(n*n for n in self.north)
        self.capacity = self.occ_scale * self.weight * sum(self.north)
        if self.nelec is not None and not 0 < self.nelec < self.capacity:
            raise ValueError('nelec must lie between 0 and %g' % self.capacity)

        # Per-k-point core Hamiltonian, as mf.energy_elec expects it.
        self.hcore = _stack_or_list(hcore_ao)
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
        stack = _uniform_stack(blocks)
        if stack is not None:
            return list(0.5 * (stack + _ht(stack)))
        return [0.5 * (x + x.conj().T) for x in blocks]

    def _inner(self, a, b):
        sa, sb = _uniform_stack(a), _uniform_stack(b)
        if sa is not None and sb is not None and sa.shape == sb.shape:
            return self.weight * float(cp.vdot(sa, sb).real.item())
        return self.weight * sum(float(cp.vdot(x, y).real.item()) for x, y in zip(a, b))

    def _rms(self, blocks):
        return max(0.0, self._inner(blocks, blocks) / self.ndof) ** 0.5

    def _trace_mean(self, blocks):
        stack = _uniform_stack(blocks)
        if stack is not None:
            numerator = float(cp.trace(stack, axis1=1, axis2=2).real.sum().item())
        else:
            numerator = sum(float(cp.trace(x).real.item()) for x in blocks)
        return numerator / sum(x.shape[0] for x in blocks)

    def _blocks(self, values, name='matrices'):
        '''Split an mf-shaped (nkpts,...) or (2,nkpts,...) stack into the
        spin-major list of blocks.'''
        if self.nspin == 1:
            blocks = coerce_to_list_of_matrices(values, name)
        else:
            if not (isinstance(values, (list, tuple)) and len(values) == 2):
                values = cp.asarray(values)
                if values.ndim != 4 or values.shape[0] != 2:
                    raise ValueError('%s must have shape (2,nkpts,n,n)' % name)
            blocks = (coerce_to_list_of_matrices(values[0], name) +
                      coerce_to_list_of_matrices(values[1], name))
        if len(blocks) != self.nblocks:
            raise ValueError('%s: expected %d blocks, got %d' % (
                name, self.nblocks, len(blocks)))
        return blocks

    def _from_blocks(self, blocks):
        '''Stack equally shaped blocks into the mf layout.'''
        out = cp.stack(blocks)
        return out.reshape(self._spin_kpts_shape + out.shape[1:])

    def _spin_slices(self):
        return [slice(s*self.nkpts, (s+1)*self.nkpts)
                for s in range(self.nspin)]

    def _initial_dm(self, dm0=None):
        '''Initial AO density in the mf layout.'''
        raise NotImplementedError

    def _inversion_partners(self, tol=1e-6):
        '''Index of -k in self.kpts for every k, or None if the k-point set is not
        closed under inversion.'''
        scaled = self.cell.get_scaled_kpts(self.kpts)
        diff = scaled[:, None, :] + scaled[None, :, :]
        match = np.all(abs(diff - np.rint(diff)) < tol, axis=2)
        if not np.all(match.sum(axis=1) == 1):
            logger.info(self, 'k-points not closed under inversion; '
                        'D(-k) = D(k)* is not enforced')
            return None
        return match.argmax(axis=1)

    def _time_reversal(self, blocks):
        '''Enforce D(-k) = D(k)* in each spin block. A real-AO Hamiltonian has
        H(-k) = H(k)*, so the exact density does too, but rounding in an ill-conditioned
        basis (and far-from-converged steps) breaks it; the multigrid transform of D(k)
        to the supercell then finds a large imaginary part.'''
        if self._kinv is None:
            return blocks
        stack = _uniform_stack(blocks)
        if stack is not None:
            stack = stack.reshape((self.nspin, self.nkpts) + stack.shape[1:])
            partner = stack[:, self._kinv].conj()
            if self.verbose >= logger.DEBUG:
                logger.debug(self, 'max |D(k) - D(-k)*| = %.3g',
                             float(cp.abs(stack - partner).max().item()))
            out = 0.5 * (stack + partner)
            return list(out.reshape((-1,) + out.shape[2:]))
        out = []
        worst = 0.
        for sl in self._spin_slices():
            spin = blocks[sl]
            for d, j in zip(spin, self._kinv):
                partner = spin[j].conj()
                worst = max(worst, cp.max(cp.abs(d - partner)).item())
                out.append(0.5 * (d + partner))
        logger.debug(self, 'max |D(k) - D(-k)*| = %.3g', worst)
        return out

    def _spin_info(self, cycle_data):
        '''Suffix for log lines: the magnetization in KUKS, nothing in KRKS.'''
        if self.nspin == 1:
            return ''
        return '  M = %.8g' % cycle_data.magnetization

    def _sanitize_h(self, h):
        h = coerce_to_list_of_matrices(h, 'auxiliary Hamiltonian')
        if len(h) != self.nblocks:
            raise ValueError('Hamiltonian and block counts differ')
        stack = _uniform_stack(h)
        if stack is not None and stack.shape[1:] == (self.north[0],)*2 and \
                all(n == self.north[0] for n in self.north):
            if not bool(cp.isfinite(stack).all().item()):
                raise FloatingPointError('Hamiltonian contains nonfinite values')
            h = list(0.5 * (stack + _ht(stack)))
        else:
            for k, x in enumerate(h):
                if x.shape != (self.north[k], self.north[k]):
                    raise ValueError('invalid Hamiltonian shape at block %d' % k)
                if not bool(cp.all(cp.isfinite(x)).item()):
                    raise FloatingPointError('Hamiltonian contains nonfinite values')
            h = self._hermi(h)
        if getattr(self, '_x_time_reversal', False):
            h = self._time_reversal(h)
        return h

    def _eigh_h(self, h):
        '''Eigenpairs of the sanitized orthogonal-basis Hamiltonian blocks.  With
        time-reversal symmetry only one k of each (k, -k) pair is diagonalized; the
        partner gets the same eigenvalues and conjugated eigenvectors.'''
        if not getattr(self, '_x_time_reversal', False):
            return _eigh_blocks(h)
        nk = self.nkpts
        partner = [s*nk + int(self._kinv[k]) for s in range(self.nspin) for k in range(nk)]
        reps = [b for b in range(self.nblocks) if partner[b] >= b]
        w, v = _eigh_blocks([h[b] for b in reps])
        eig = [None] * self.nblocks
        coeff = [None] * self.nblocks
        for b, wb, vb in zip(reps, w, v):
            eig[b], coeff[b] = wb, vb
        for b in range(self.nblocks):
            if eig[b] is None:
                eig[b] = eig[partner[b]]
                coeff[b] = coeff[partner[b]].conj()
        return eig, coeff

    def _x_stack(self):
        '''x_ao2orth as one (nblocks, nao, north) stack, or None if north varies.'''
        if getattr(self, '_x_stack_cache', None) is None or \
                self._x_stack_cache[0] is not self.x_ao2orth:
            self._x_stack_cache = (self.x_ao2orth, _uniform_stack(self.x_ao2orth))
        return self._x_stack_cache[1]

    def _to_orth(self, matrices):
        matrices = coerce_to_list_of_matrices(matrices, 'AO matrices')
        a = _uniform_stack(matrices)
        x = self._x_stack()
        if a is not None and x is not None and a.shape[0] == x.shape[0]:
            error = cp.abs(a - _ht(a)).max(axis=(1, 2))
            scale = cp.maximum(1.0, cp.abs(a).max(axis=(1, 2)))
            ratio = (error / scale).get()
            if (ratio > _HERMITICITY_TOL).any():
                k = int(ratio.argmax())
                raise FloatingPointError(
                    'Fock matrix at block %d is not Hermitian: '
                    'max |F - F^H| / max |F| = %.3g' % (k, ratio[k]))
            logger.debug1(self, 'max |F - F^H| / max |F| = %.3g', ratio.max())
            value = cp.matmul(cp.matmul(_ht(x), a), x)
            return list(0.5 * (value + _ht(value)))
        out = []
        worst = 0.
        for k, (a, x) in enumerate(zip(matrices, self.x_ao2orth)):
            # X^H A X is Hermitian whenever A is, so test A itself.  Testing
            # X^H A X would amplify rounding by ||X||^2 = 1/min eig(S).
            error = cp.max(cp.abs(a - a.conj().T)).item()
            scale = max(1.0, cp.max(cp.abs(a)).item())
            if error > _HERMITICITY_TOL * scale:
                raise FloatingPointError(
                    'Fock matrix at block %d is not Hermitian: '
                    'max |F - F^H| = %.3g, max |F| = %.3g' % (k, error, scale))
            worst = max(worst, error / scale)
            value = x.conj().T.dot(a).dot(x)
            out.append(0.5 * (value + value.conj().T))
        logger.debug1(self, 'max |F - F^H| / max |F| = %.3g', worst)
        return out

    def _fock_from_veff(self, dm, veff):
        if getattr(veff, 'v_solvent', None) is not None:
            if not hasattr(self.mf, 'get_fock'):
                raise TypeError('tagged solvent potential requires mf.get_fock')
            fock = self.mf.get_fock(
                h1e=self.hcore, vhf=veff, dm=dm, cycle=-1, diis=None, level_shift_factor=0.0, damp_factor=0.0
            )
            return self._blocks(fock, 'decorated Fock matrices')
        veff_blocks = self._blocks(cp.asarray(veff), 'veff')
        return [hcore + v for hcore, v in zip(self.hcore_ao, veff_blocks)]

    def _initial_h(self, dm0=None):
        dm_blocks = self._time_reversal(self._hermi(self._blocks(
            self._initial_dm(dm0), 'initial density')))
        dm = self._from_blocks(dm_blocks)
        nelec = self.weight * sum(_as_float(
            cp.einsum('ij,ji->', d, s), 'initial electron number')
            for d, s in zip(dm_blocks, self.s_ao))
        self.nfev += 1
        veff = self.mf.get_veff(
            self.cell, dm, dm_last=None, vhf_last=None, hermi=1,
            kpts=self.mf.kpts, kpts_band=None)
        h = self._sanitize_h(self._to_orth(self._fock_from_veff(dm, veff)))
        return h, nelec

    def _solve_mu(self, orbital_energies, nelec):
        '''Fermi level mu with nelec_from_eig(orbital_energies, mu) == nelec, on the GPU:
        safeguarded Newton inside a sign-changing bracket, one host sync per step.'''
        e = _concat_ravel(orbital_energies)
        nocc = nelec / (self.occ_scale * self.weight)       # sum of occupations
        if not 0 < nocc < e.size:
            raise ValueError('nelec outside (0, capacity)')
        beta = self.beta

        def occ_sum(mu):
            q = _fermi_occ(beta * (e - mu))
            out = cp.stack([q.sum(), (q * (1 - q)).sum()]).get()
            return out[0] - nocc, out[1] * beta

        # Bracket from the sorted spectrum: the nocc-th level +- 40 sigma (widened if
        # the smearing tails need it), else the whole spectrum +- 10 Ha.
        es = cp.sort(e)
        k = int(nocc)
        lo = float(es[max(k - 1, 0)].item()) - 40. / beta
        hi = float(es[min(k, e.size - 1)].item()) + 40. / beta
        f_lo, f_hi = occ_sum(lo)[0], occ_sum(hi)[0]
        if not (f_lo <= 0 <= f_hi):
            lo, hi = float(es[0].item()) - 10., float(es[-1].item()) + 10.
            f_lo, f_hi = occ_sum(lo)[0], occ_sum(hi)[0]
        if f_lo == 0:
            return self._check_mu(orbital_energies, nelec, lo)
        if f_hi == 0:
            return self._check_mu(orbital_energies, nelec, hi)
        mu = 0.5 * (lo + hi)
        tol = 1e-13 * max(1.0, nocc)
        best = (np.inf, mu)
        for _ in range(200):
            f, df = occ_sum(mu)
            if abs(f) < best[0]:
                best = (abs(f), mu)
            if abs(f) <= tol:
                break
            if f < 0:
                lo = mu
            else:
                hi = mu
            step = mu - f / df if df > 0 else np.nan
            if lo < step < hi:
                mu = step
            else:
                mu = 0.5 * (lo + hi)
            if hi - lo <= 4 * np.spacing(max(abs(lo), abs(hi))):
                break
        return self._check_mu(orbital_energies, nelec, best[1])

    def _check_mu(self, orbital_energies, nelec, mu):
        mu = _as_float(mu, 'chemical potential')
        error = abs(self.nelec_from_eig(orbital_energies, mu) - nelec)
        if error > 1e-10:
            raise RuntimeError('chemical-potential solve has charge error %g' % error)
        return mu

    def nelec_from_eig(self, orbital_energies, mu):
        e = _concat_ravel(orbital_energies)
        return self.occ_scale * self.weight * float(
            cp.sum(_fermi_occ(self.beta * (e - mu))).item())

    def nelec_at_mu(self, h, mu):
        return self.nelec_from_eig(_eigh_blocks(h, vectors=False), mu)

    def _density_blocks(self, eig, coeff, mu, h):
        '''Occupations, orthogonal and AO densities block by block (blocks of
        different size).'''
        occ = []
        p = []
        dm = cp.empty((self.nblocks, self.nao, self.nao),
                      dtype=cp.result_type(*h, cp.complex128))
        entropy_sum = 0.
        for k, (energy, c, x) in enumerate(zip(eig, coeff, self.x_ao2orth)):
            gamma = self.beta * (energy - mu)
            q = _fermi_occ(gamma)
            density = (c * q[None, :]).dot(c.conj().T)
            density = 0.5 * (density + density.conj().T)
            dmk = self.occ_scale * x.dot(density).dot(x.conj().T)
            dm[k] = 0.5 * (dmk + dmk.conj().T)
            occ.append(q)
            p.append(density)
            entropy_sum += float(_fermi_entropy(q))
        p = self._hermi(p)
        dm_blocks = self._time_reversal(
            self._hermi([dm[k] for k in range(self.nblocks)]))

        spin_nelec = tuple(
            self.occ_scale * self.weight *
            sum(float(cp.trace(x).real.item()) for x in p[sl])
            for sl in self._spin_slices())
        ao_nelec = self.weight * sum(
            _as_float(cp.einsum('ij,ji->', d, s), 'AO electron count') for d, s in zip(dm_blocks, self.s_ao)
        )
        return occ, p, dm_blocks, entropy_sum, spin_nelec, ao_nelec

    def _density_stacked(self, eig, coeff, mu, x_stack):
        '''_density_blocks for equally sized blocks, as batched GPU operations with
        a single host sync.'''
        e = cp.stack(eig)
        c = cp.stack(coeff)
        q = _fermi_occ(self.beta * (e - mu))
        p = cp.matmul(c * q[:, None, :], _ht(c))
        p = 0.5 * (p + _ht(p))
        dm = self.occ_scale * cp.matmul(cp.matmul(x_stack, p), _ht(x_stack))
        dm = 0.5 * (dm + _ht(dm))
        dm_blocks = self._time_reversal(list(dm))
        s_stack = self._s_stack()
        traces = cp.trace(p, axis1=1, axis2=2).real.reshape(self.nspin, self.nkpts).sum(axis=1)
        ao = cp.einsum('kij,kji->', cp.stack(dm_blocks), s_stack)
        scalars = cp.concatenate([traces.astype(cp.complex128), ao.reshape(1).astype(cp.complex128),
                                  _fermi_entropy(q).reshape(1).astype(cp.complex128)]).get()
        spin_nelec = tuple(self.occ_scale * self.weight * float(t.real)
                           for t in scalars[:self.nspin])
        ao_nelec = self.weight * _as_float(scalars[self.nspin], 'AO electron count')
        entropy_sum = float(scalars[self.nspin + 1].real)
        return list(q), list(p), dm_blocks, entropy_sum, spin_nelec, ao_nelec

    def _s_stack(self):
        if getattr(self, '_s_stack_cache', None) is None or \
                self._s_stack_cache[0] is not self.s_ao:
            self._s_stack_cache = (self.s_ao, cp.stack(self.s_ao))
        return self._s_stack_cache[1]

    def calculate_cycle(self, h, nelec=None, mu=None,
                        align_fixed_n_gauge=True):
        if (nelec is None) == (mu is None):
            raise TypeError('specify exactly one of nelec and mu')
        self.nfev += 1
        h = self._sanitize_h(h)
        eig, coeff = self._eigh_h(h)
        mu = self._solve_mu(eig, nelec) if nelec is not None else mu

        x_stack = self._x_stack()
        if x_stack is not None and len(set(c.shape for c in coeff)) == 1:
            occ, p, dm_blocks, entropy_sum, spin_nelec, ao_nelec = \
                self._density_stacked(eig, coeff, mu, x_stack)
        else:
            occ, p, dm_blocks, entropy_sum, spin_nelec, ao_nelec = \
                self._density_blocks(eig, coeff, mu, h)
        dm = self._from_blocks(dm_blocks)
        electron_number = sum(spin_nelec)
        if abs(electron_number - ao_nelec) > 1e-8 * max(1.0, electron_number):
            raise ValueError('AO and orthogonal electron counts disagree')

        veff = self.mf.get_veff(self.cell, dm, dm_last=None, vhf_last=None, hermi=1, kpts=self.mf.kpts, kpts_band=None)
        fock_ao = self._fock_from_veff(dm, veff)
        e_elec = _as_float(self.mf.energy_elec(dm, self.hcore, veff)[0], 'electronic energy')
        fock = self._to_orth(fock_ao)
        e_tot = e_elec + self.e_nuc
        entropy = self.occ_scale * self.weight * entropy_sum
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
            getattr(veff, 'lpbe_grid', None), spin_nelec)

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
            'residual = %.6g%s',
            self.cycles,
            nelec,
            cycle_data.mu,
            cycle_data.free_energy,
            cycle_data.residual_rms,
            self._spin_info(cycle_data),
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
            residual = self.diis_pack(cycle_data.residual)

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
                'residual = %.6g  damping = %.3g  ratio = %.3g%s',
                self.cycles,
                fixed_n_calc.nelec,
                trial.mu,
                trial.free_energy,
                trial.residual_rms,
                step,
                ratio,
                self._spin_info(trial),
            )

            if dump_chk and self.chkfile:
                self._dump_chk_async(trial)

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

    def _kernel_fixed_mu(self, h, current_n, seed_samples=None, dump_chk=False):
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
            self.fixed_n_subproblem(fixed_n_calc, tolerance, target_mu=self.mu, dump_chk=False)
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
                '%.12g  delta mu = %.3g  residual = %.6g%s',
                self.outer_cycles,
                current_n,
                cycle_data.mu,
                error,
                cycle_data.residual_rms,
                self._spin_info(cycle_data),
            )

            if abs(error) <= self.diis_max_dmu:
                return self.fixed_mu_diis_inner(h=cycle_data.h, dump_chk=dump_chk)


            # root_ready = abs(error) <= self.conv_tol_mu
            # if root_ready and cycle_data.residual_rms > self.conv_tol:
            #     h = self._copy(cycle_data.h)
            #     pending = fixed_n_calc
            #     continue
            # if root_ready:
            #     self.message = 'converged fixed-mu secant search'
            #     return cycle_data, True

            if bracket is not None and abs(bracket[1].nelec - bracket[0].nelec) <= self.root_nelec_tol:
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
        self.wait_io()
        prefix = Path(output_prefix)
        csv_path = Path(str(prefix) + '.csv')
        h5_path = Path(str(prefix) + '.h5')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
        with h5py.File(h5_path, 'w') as h5f:
            h5f.create_dataset('sigma', data=self.sigma)
            # points/<i>/mo_*/<b> are indexed by spin-major block b.
            h5f.attrs['nspin'] = self.nspin
            h5f.attrs['nkpts'] = self.nkpts
            for name, dtype in columns.items():
                if dtype is str:
                    dtype = h5py.string_dtype(encoding='utf-8')
                h5f.create_dataset(
                    name, shape=(0,), maxshape=(None,), dtype=dtype)
            h5f.create_group('points')
            h5f.attrs['completed_points'] = 0
        return csv_path, h5_path

    def _add_spin_columns(self, columns):
        if self.nspin == 2:
            columns['nelec_alpha'] = np.float64
            columns['nelec_beta'] = np.float64
            columns['magnetization'] = np.float64
        return columns

    def _add_spin_record(self, record, cycle_data):
        if self.nspin == 2:
            record['nelec_alpha'] = cycle_data.spin_nelec[0]
            record['nelec_beta'] = cycle_data.spin_nelec[1]
            record['magnetization'] = cycle_data.magnetization
        return record

    def _get_writer(self):
        if self._writer is None or self._writer.enabled != bool(self.async_io):
            if self._writer is not None:
                self._writer.wait()
            self._writer = _BackgroundWriter(self.async_io)
        return self._writer

    def wait_io(self):
        '''Block until every background write (scan point, chkfile) is on disk.'''
        if self._writer is not None:
            self._writer.wait()
        return self

    def _scan_point_data(self, cycle_data):
        '''Host copies of the orbitals of one scan point, optionally without the MOs
        whose occupation is below scan_mo_occ_threshold.'''
        threshold = self.scan_mo_occ_threshold
        data = {'mo_coeff': [], 'mo_occ': [], 'mo_energy': []}
        for x, c, q, e in zip(self.x_ao2orth, cycle_data.coeff, cycle_data.occ,
                              cycle_data.eig):
            occ = self.occ_scale*q
            coeff = x.dot(c)
            if threshold is not None:
                # occupations are Fermi-ordered (ascending energy): keep a prefix
                nkeep = max(1, int(cp.count_nonzero(occ >= threshold).item()))
                coeff, occ, e = coeff[:, :nkeep], occ[:nkeep], e[:nkeep]
            data['mo_coeff'].append(cp.asnumpy(coeff))
            data['mo_occ'].append(cp.asnumpy(occ))
            data['mo_energy'].append(cp.asnumpy(e))
        return data

    def _append_scan_output(
            self, csv_path, h5_path, columns, record, cycle_data):
        data = self._scan_point_data(cycle_data)
        self._get_writer().submit(
            _write_scan_point, csv_path, h5_path, columns, dict(record), data,
            self.scan_mo_occ_threshold, label='scan point %d' % record['index'])

    def _extrapolate_h(self, history, nelec):
        '''Starting Hamiltonian for the next N-scan point: linear extrapolation in N
        through the converged Hamiltonians of the two previous points.'''
        (n0, h0), (n1, h1) = history
        if abs(n1 - n0) < 1e-12:
            return self._copy(h1)
        t = (nelec - n1) / (n1 - n0)
        logger.debug(self, 'N-scan start: extrapolating H from N = %g, %g (t = %g)',
                     n0, n1, t)
        return [x1 + t * (x1 - x0) for x0, x1 in zip(h0, h1)]

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
        history = []      # (nelec, h) of the last converged, chained points
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
        self._add_spin_columns(columns)
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
                    history = []
            if self.scan_extrapolate and len(history) >= 2:
                h = self._extrapolate_h(history[-2:], nelec)
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
            self._add_spin_record(record, cycle_data)
            records.append(record)
            self._append_scan_output(
                csv_path, h5_path, columns, record, cycle_data)

            h = self._copy(cycle_data.h)
            if fixed_n_calc.converged:
                history = (history + [(float(nelec), h)])[-2:]
            else:
                history = []
            last_cycle_data = cycle_data
            last_converged = fixed_n_calc.converged

        self.nelec = nelecs[-1]
        self.message = records[-1]['message']
        self.wait_io()
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

    def _merge_mu_calibration(self, calibration, record):
        '''Add a converged fixed-mu point to the mu(N) calibration.

        A fixed-mu solution only determines N to roughly the residual
        tolerance, so a point whose mu matches an existing calibration point
        within conv_tol_mu is the same state; keep the better-converged one
        instead of creating a near-duplicate that breaks strict monotonicity.
        '''
        merged = []
        for old in calibration:
            if abs(old['mu']-record['mu']) <= self.conv_tol_mu:
                if old['residual_rms'] < record['residual_rms']:
                    return calibration
            else:
                merged.append(old)
        merged.append(record)
        try:
            return self._prepare_mu_calibration(merged)
        except ValueError as err:
            logger.warn(self, 'mu-scan point N = %.12g, mu = %.12g not added '
                        'to the mu(N) calibration: %s',
                        record['nelec'], record['mu'], err)
            return calibration

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
        '''AO density of an N-scan point in the layout of the scan's solver:
        (nkpts,nao,nao) for a KRKS scan, (2,nkpts,nao,nao) for a KUKS scan.
        _initial_dm converts it to this solver's spin treatment.'''
        with h5py.File(filename, 'r') as h5f:
            point = h5f['points'][str(index)]
            coeff_group = point['mo_coeff']
            occ_group = point['mo_occ']
            coeff_keys = sorted(coeff_group, key=int)
            occ_keys = sorted(occ_group, key=int)
            # Files written before KUKS support carry no nspin attribute.
            nspin = int(h5f.attrs.get('nspin', 1))
            expected = [str(k) for k in range(nspin*self.nkpts)]
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
        dm = cp.stack(dm)
        if nspin == 2:
            dm = dm.reshape((2, self.nkpts) + dm.shape[1:])
        return dm

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
        self._add_spin_columns(columns)
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
            self._add_spin_record(record, cycle_data)
            records.append(record)
            self._append_scan_output(
                csv_path, h5_path, columns, record, cycle_data)
            h = self._copy(cycle_data.h)
            last_cycle_data = cycle_data
            last_converged = converged
            if converged:
                calibration = self._merge_mu_calibration(calibration, {
                    'index': None,
                    'nelec': cycle_data.nelec,
                    'mu': cycle_data.mu,
                    'converged': True,
                    'residual_rms': cycle_data.residual_rms,
                })

        self.nelec = None
        self.message = records[-1]['message']
        self.wait_io()
        self._finalize(last_cycle_data, last_converged)
        return records

    def _stack_spin(self, blocks):
        '''Per-k stacks (or lists, for ragged k-points) of each spin, stacked
        once more over spin for KUKS.'''
        per_spin = [_stack_or_list(blocks[sl]) for sl in self._spin_slices()]
        if self.nspin == 1:
            return per_spin[0]
        if (all(isinstance(x, cp.ndarray) for x in per_spin) and
                per_spin[0].shape == per_spin[1].shape):
            return cp.stack(per_spin)
        return per_spin

    def _finalize(self, cycle_data, converged):
        mo_coeff = [x.dot(c) for x, c in zip(self.x_ao2orth, cycle_data.coeff)]
        mo_energy = [e for e in cycle_data.eig]
        mo_occ = [self.occ_scale*x for x in cycle_data.occ]
        self.mo_coeff = self._stack_spin(mo_coeff)
        self.mo_energy = self._stack_spin(mo_energy)
        self.mo_occ = self._stack_spin(mo_occ)
        self.magnetization = cycle_data.magnetization
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
        if self.nspin == 2:
            self.scf_summary['magnetization'] = self.magnetization

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
        if self.nspin == 2:
            self.mf.magnetization_gc = self.magnetization
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
            cycle_data, converged = self._kernel_fixed_mu(h, initial_nelec, dump_chk=dump_chk)
        else:
            fixed_n_calc = self.start_fixed_n_calc(h, self.nelec)
            self.fixed_n_subproblem(fixed_n_calc, self.conv_tol, dump_chk=dump_chk)
            self.message = fixed_n_calc.message
            cycle_data, converged = fixed_n_calc.cycle_data, fixed_n_calc.converged
        logger.info(self, '%s; total Fock evaluations = %d',
                    self.message, self.nfev)
        if dump_chk:
            self._flush_chk()
        return self._finalize(cycle_data, converged)

    scf = kernel


class GrandCanonicalKRKS(_GrandCanonicalKSBase):
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

    nspin = 1
    _mf_type = 'KRKS'

    def _initial_dm(self, dm0=None):
        '''Initial density; a spin-resolved (2,nkpts,nao,nao) density, e.g.
        from a KUKS scan, is summed over spin.'''
        if dm0 is None:
            dm0 = self.mf.get_init_guess(self.cell, key=self.mf.init_guess)
        if not isinstance(dm0, (list, tuple)):
            dm0 = cp.asarray(dm0)
            if dm0.ndim == 4:
                if dm0.shape[0] != 2:
                    raise ValueError('initial density has an invalid shape')
                dm0 = dm0[0] + dm0[1]
        return dm0


class GrandCanonicalKUKS(_GrandCanonicalKSBase):
    '''Finite-temperature KUKS at fixed chemical potential or electron count.

    Both spin channels share one chemical potential: at fixed mu, and at fixed
    total ``nelec``, the magnetization N_alpha - N_beta is variational.

    Args:
        mf : KUKS
            A configured GPU4PySCF periodic unrestricted Kohn-Sham object.
        mu : float or None
            Chemical potential in Hartree.  Required unless ``nelec`` is set.
        sigma : float
            Fermi smearing width in Hartree.
        nelec : float or None
            Total electron count for a canonical calculation.

    Attributes:
        init_magnetization : float or None
            If set, the initial guess from ``mf.get_init_guess`` and any
            spin-summed (nkpts,nao,nao) initial density (``dm0`` or a KRKS
            N-scan seed) are rescaled to this N_alpha - N_beta per cell.  A
            spin-symmetric start stays spin-symmetric, so set this (or pass a
            spin-polarized ``dm0``) to look for magnetic solutions.
            Spin-resolved ``dm0`` and KUKS N-scan seeds are used unchanged.
    '''

    nspin = 2
    _mf_type = 'KUKS'
    init_magnetization = getattr(
        __config__, 'pbc_dft_grand_canonical_init_magnetization', None)

    _keys = {'init_magnetization'}

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        if self.init_magnetization is not None:
            logger.info(self, 'initial magnetization = %.12g',
                        self.init_magnetization)
        return self

    def _initial_dm(self, dm0=None):
        from_guess = dm0 is None
        if from_guess:
            key = self.mf.init_guess
            if isinstance(key, str) and key.lower()[:3] == 'chk':
                # KUHF.get_init_guess rescales each spin density to mf.nelec
                # (from cell.spin) when it is off by more than 0.01 e/cell,
                # which would erase the excess charge and the magnetization
                # stored in the chkfile.  KUHF.init_guess_by_chkfile defaults
                # to project=True, which renormalizes every MO and turns the
                # zero columns padding dropped AOs (see _chk_envs) into NaN;
                # project=None projects only if the basis set changed.
                try:
                    dm0 = self.mf.init_guess_by_chkfile(project=None)
                except (OSError, KeyError):
                    logger.warn(self, 'Fail in reading %s. Use MINAO initial '
                                'guess', self.mf.chkfile)
                    dm0 = self.mf.get_init_guess(self.cell, key='minao')
            else:
                dm0 = self.mf.get_init_guess(self.cell, key=key)
        if isinstance(dm0, (list, tuple)) and len(dm0) == 2:
            dm = cp.stack([cp.asarray(dm0[0]), cp.asarray(dm0[1])])
        else:
            dm = cp.asarray(dm0)
        spin_summed = dm.ndim == 3
        if spin_summed:
            dm = cp.stack([.5*dm, .5*dm])
        elif dm.ndim != 4 or dm.shape[0] != 2:
            raise ValueError('initial density must have shape (nkpts,nao,nao) '
                             'or (2,nkpts,nao,nao)')
        if self.init_magnetization is not None and (from_guess or spin_summed):
            dm = self._magnetize(dm, self.init_magnetization)
        return dm

    def _magnetize(self, dm, magnetization):
        '''Rescale the spin densities of dm to N_alpha - N_beta =
        magnetization, keeping the total density.'''
        magnetization = _as_float(magnetization, 'init_magnetization')
        total = dm[0] + dm[1]
        nelec = self.weight * sum(_as_float(
            cp.einsum('ij,ji->', d, s), 'initial electron number')
            for d, s in zip(total, self.s_ao))
        if not abs(magnetization) < nelec:
            raise ValueError('|init_magnetization| must be smaller than the '
                             'initial electron number %g' % nelec)
        logger.info(self, 'Initial guess magnetization set to %.12g',
                    magnetization)
        return cp.stack([total * (.5 + .5*magnetization/nelec),
                         total * (.5 - .5*magnetization/nelec)])


def GrandCanonicalKS(mf, *args, **kwargs):
    '''GrandCanonicalKRKS or GrandCanonicalKUKS, whichever matches mf.'''
    if hasattr(mf, 'istype') and mf.istype('KUKS'):
        return GrandCanonicalKUKS(mf, *args, **kwargs)
    return GrandCanonicalKRKS(mf, *args, **kwargs)
