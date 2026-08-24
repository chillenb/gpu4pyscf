#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
#
# Author: Tianyu Zhu <zhutianyu1991@gmail.com>
# Author: Christopher Hillenbrand <chillenbrand15@gmail.com>
# Author: Chaoqun Zhang <cq_zhang@outlook.com>
# Author: Jincheng Yu <pimetamon@gmail.com>
# Author: Jiachen Li <lijiachen.duke@gmail.com>
#

"""
Periodic spin-restricted random phase approximation (direct RPA) with N^4 scaling.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14, 053020 (2012)
"""

import time

import cupy as cp
import numpy as np

from pyscf import lib
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots
from pyscf.lib import temporary_env
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.tools import k2gamma

from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem
from gpu4pyscf.pbc.df.df import GDF
from gpu4pyscf.pbc.dft import gen_grid, numint
from gpu4pyscf.pbc.lib.kpts_helper import kk_adapted_iter


def _to_float(value):
    if isinstance(value, cp.ndarray):
        return float(value.item())
    return float(value)


def get_frozen_mask(rpa):
    """Return CPU boolean masks for the active orbitals at each k-point."""
    masks = [np.ones(x.shape[-1], dtype=bool) for x in rpa._scf.mo_occ]
    frozen = rpa.frozen
    if frozen is None:
        return masks
    if isinstance(frozen, (int, np.integer)):
        if frozen < 0:
            raise ValueError('The number of frozen orbitals cannot be negative')
        frozen_by_k = [np.arange(frozen)] * rpa.nkpts
    else:
        frozen = list(frozen)
        if not frozen:
            return masks
        if isinstance(frozen[0], (int, np.integer)):
            frozen_by_k = [np.asarray(frozen, dtype=int)] * rpa.nkpts
        else:
            if len(frozen) != rpa.nkpts:
                raise ValueError('Frozen orbital lists must match the number of k-points')
            frozen_by_k = [np.asarray(x, dtype=int) for x in frozen]

    for k, idx in enumerate(frozen_by_k):
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f'Duplicate frozen orbital index at k-point {k}')
        if np.any(idx < 0) or np.any(idx >= masks[k].size):
            raise ValueError(f'Frozen orbital index out of range at k-point {k}')
        masks[k][idx] = False
    return masks


def _mo_energy_frozen(rpa, mo_energy):
    masks = get_frozen_mask(rpa)
    return cp.stack([cp.asarray(mo_energy[k])[masks[k]] for k in range(rpa.nkpts)])


def _mo_frozen(rpa, mo_coeff):
    masks = get_frozen_mask(rpa)
    return cp.stack([cp.asarray(mo_coeff[k])[:, masks[k]] for k in range(rpa.nkpts)])


def _mo_occ_frozen(rpa, mo_occ):
    masks = get_frozen_mask(rpa)
    return cp.stack([cp.asarray(mo_occ[k])[masks[k]] for k in range(rpa.nkpts)])


def _validate_df(rpa):
    mydf = rpa.with_df
    if not isinstance(mydf, GDF):
        raise NotImplementedError('GPU KRPA requires gpu4pyscf.pbc.df.GDF')
    if isinstance(rpa.kpts, KPoints):
        raise NotImplementedError('GPU KRPA does not support symmetry-reduced KPoints')

    kpts = np.asarray(rpa.kpts).reshape(-1, 3)
    scaled = rpa.mol.get_scaled_kpts(kpts)
    is_gamma = np.linalg.norm(scaled - np.rint(scaled), axis=1) < 1e-10
    if not np.any(is_gamma):
        raise NotImplementedError('GPU KRPA requires a gamma-containing Monkhorst-Pack mesh')
    if not mydf.has_kpts(kpts):
        raise ValueError('The GDF object does not contain the KRPA k-point mesh')

    if getattr(mydf, '_j_only', False):
        logger.warn(rpa, 'Rebuilding j-only GDF integrals for KRPA')
        mydf._j_only = False
        mydf.reset()
    if mydf._cderi is None:
        mydf.build(j_only=False)
    if np.prod(mydf.kmesh) != rpa.nkpts:
        raise ValueError('GDF k-mesh is inconsistent with the KRPA k-points')
    return mydf


def _transfer_layout(mydf, desired_kidx):
    """Locate an RPA transfer in the independent GDF transfer set."""
    kk_conserv = k2gamma.double_translation_indices(mydf.kmesh)
    desired_kidx = np.asarray(desired_kidx, dtype=int)
    for kp, kp_conj, ki_idx, kj_idx in kk_adapted_iter(mydf.kmesh):
        if not np.array_equal(ki_idx, np.arange(len(ki_idx))):
            raise RuntimeError('Unexpected GDF k-point ordering')
        if np.array_equal(kj_idx, desired_kidx):
            return int(kp), False
        ki_conj, kj_conj = np.where(kk_conserv == kp_conj)
        if (np.array_equal(ki_conj, np.arange(len(ki_conj))) and
                np.array_equal(kj_conj, desired_kidx)):
            return int(kp), True
    raise RuntimeError('RPA momentum transfer was not found in the GDF tensor')


def _transform_cderi(rpa, desired_kidx, mo_left, mo_right):
    """Transform one momentum-transfer sector from AO to selected MO blocks.

    Returns a list of ``(metric_sign, Lij_by_kpoint)`` entries.  Positive and
    negative metric sectors are kept separate until they are assembled into the
    signed auxiliary response.
    """
    mydf = _validate_df(rpa)
    kp, use_conjugate = _transfer_layout(mydf, desired_kidx)
    nkpts = rpa.nkpts
    nao = rpa.mol.nao
    desired_kidx = np.asarray(desired_kidx, dtype=int)
    mo_left = [cp.asarray(x) for x in mo_left]
    mo_right = [cp.asarray(x) for x in mo_right]

    naux_pos = mydf._cderi[kp].shape[0]
    sector_sizes = [(1, naux_pos)]
    if (rpa.mol.dimension == 2 and mydf._cderip is not None and
            kp in mydf._cderip):
        sector_sizes.append((-1, mydf._cderip[kp].shape[0]))

    transformed = {}
    for sign, naux in sector_sizes:
        transformed[sign] = [
            cp.empty((naux, mo_left[i].shape[1],
                      mo_right[desired_kidx[i]].shape[1]), dtype=cp.complex128)
            for i in range(nkpts)
        ]

    bytes_per_aux = max(
        nkpts * nao * nao * np.dtype(np.complex128).itemsize * 2, 1)
    blksize = int(get_avail_mem() * 0.15 // bytes_per_aux)
    blksize = max(1, min(blksize, mydf.blockdim, naux_pos))
    logger.debug1(rpa, 'KRPA GDF transfer %d block size %d', kp, blksize)

    for p0, p1 in lib.prange(0, naux_pos, blksize):
        aux_iter = iter(((kp, p0, p1),))
        for _, Lpq, sign in mydf.loop(
                blksize, kpts=rpa.kpts, aux_iter=aux_iter):
            if sign > 0:
                q0, q1 = p0, p1
            else:
                q0, q1 = 0, Lpq.shape[1]
            for i in range(nkpts):
                j = int(desired_kidx[i])
                if use_conjugate:
                    Lpq_i = Lpq[j].conj().transpose(0, 2, 1)
                else:
                    Lpq_i = Lpq[i]
                if mo_left[i].shape[1] == 0 or mo_right[j].shape[1] == 0:
                    continue
                buf = contract('Lpq,qj->Lpj', Lpq_i, mo_right[j])
                contract('pi,Lpj->Lij', mo_left[i].conj(), buf,
                         out=transformed[sign][i][q0:q1])
            Lpq = None

    return [(sign, transformed[sign]) for sign, _ in sector_sizes]


def _stack_metric_sectors(sectors):
    if not sectors:
        raise RuntimeError('No density-fitting auxiliary functions were generated')
    nkpts = len(sectors[0][1])
    signs = cp.concatenate([
        cp.full(blocks[0].shape[0], sign, dtype=cp.float64)
        for sign, blocks in sectors
    ])
    Lij = []
    for k in range(nkpts):
        blocks = [sector[1][k] for sector in sectors]
        Lij.append(blocks[0] if len(blocks) == 1 else cp.concatenate(blocks, axis=0))
    return Lij, signs


def _apply_metric(Pi, signs):
    """Apply the signed DF metric on the left auxiliary index."""
    return Pi * signs[:, None]


def kernel(rpa, mo_energy, mo_coeff, nw=None, with_e_hf=None):
    """RPA correlation and total energy

    Parameters
    ----------
    rpa : KRPA
        rpa object
    mo_energy : double array
        molecular orbital energies
    mo_coeff : double ndarray
        molecular orbital coefficients
    nw : int, optional
        number of frequency point on imaginary axis, by default None
    with_e_hf : float, optional
        extra input HF energy, by default None

    Returns
    -------
    e_tot : float
        RPA total energy
    e_hf : float
        HF energy (exact exchange for given mo_coeff)
    e_corr : float
        RPA correlation energy
    """
    mo_energy = cp.asarray(mo_energy)
    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = _mo_occ_frozen(rpa, rpa._scf.mo_occ)
    _validate_df(rpa)

    # Compute HF exchange energy (EXX)
    if with_e_hf is None:
        with temporary_env(rpa.with_df, verbose=0), temporary_env(rpa.mol, verbose=0):
            dm = rpa._scf.make_rdm1()
            e_1e = cp.einsum('kij,kji->', dm, rpa._scf.get_hcore()).real / rpa.nkpts
            e_j = (cp.einsum('kij,kji->', dm,
                              rpa._scf.get_j(rpa.mol, dm)).real *
                   (0.5 / rpa.nkpts))
            e_x = get_rpa_exx(rpa, acfd=rpa.acfd_exx, correction_only=False)
            e_nuc = _to_float(rpa._scf.energy_nuc())
            e_hf = _to_float(e_1e + e_j) + e_x + e_nuc
    else:
        e_hf = _to_float(with_e_hf)
        logger.debug(rpa, f'  Setting EXX energy explicitly to {e_hf}')

    is_metal = hasattr(rpa._scf, 'sigma')

    # Turn off FC for metals
    if is_metal and rpa.fc:
        logger.warn(rpa, 'FC not available for metals - setting rpa.fc to False')
        rpa.fc = False

    # Grids for integration on imaginary axis
    freqs, wts = rpa.get_grids(nw=nw, mo_energy=mo_energy)

    # Compute RPA correlation energy
    if rpa.outcore:
        if is_metal:
            e_corr = get_rpa_ecorr_outcore_metal(
                rpa, freqs, wts, mo_energy, mo_coeff, mo_occ)
        else:
            e_corr = get_rpa_ecorr_outcore(
                rpa, freqs, wts, mo_energy, mo_coeff)
    else:
        e_corr = get_rpa_ecorr(
            rpa, freqs, wts, mo_energy, mo_coeff, mo_occ)

    # Compute total energy
    e_tot = float(e_hf + e_corr)

    logger.debug(rpa, f'  RPA total energy = {e_tot}')
    logger.debug(rpa, f'  EXX energy = {e_hf}, RPA corr energy = {e_corr}')

    return e_tot, e_hf, e_corr


def get_idx_metal(mo_occ, threshold=1.0e-6):
    """Get index of occupied/virtual/fractional orbitals of metals.

    Parameters
    ----------
    mo_occ : double 1d array
        occupation number
    threshold : double, optional
        threshold to determine fractionally occupied orbitals, by default 1.0e-6

    Returns
    -------
    idx_occ : list
        list of occupied orbital indexes
    idx_frac : list
        list of fractionally occupied orbital indexes
    idx_vir : list
        list of virtual orbital indexes
    """
    if isinstance(mo_occ, cp.ndarray):
        mo_occ = cp.asnumpy(mo_occ)
    else:
        mo_occ = np.asarray(mo_occ)

    mask_occ = mo_occ > 2.0 - threshold
    mask_vir = mo_occ < threshold
    mask_frac = ~(mask_occ | mask_vir)

    idx_occ = np.where(mask_occ)[0]
    idx_frac = np.where(mask_frac)[0]
    idx_vir = np.where(mask_vir)[0]

    return idx_occ, idx_frac, idx_vir


def get_rho_response(omega, mo_energy, Lia, kidx):
    """Build the unsigned insulating density response in the DF basis."""
    Lia = cp.asarray(Lia)
    mo_energy = cp.asarray(mo_energy)
    nkpts, naux = Lia.shape[:2]
    nocc = Lia.shape[2]
    Pi = cp.zeros((naux, naux), dtype=cp.complex128)
    for i in range(nkpts):
        a = int(kidx[i])
        eia = mo_energy[i, :nocc, None] - mo_energy[a, None, nocc:]
        rho_accum_inner(Pi, eia, omega, Lia[i], alpha=4.0 / nkpts)
    return Pi


def get_rho_response_head(omega, mo_energy, qij):
    """Compute the finite-size head response at one imaginary frequency."""
    mo_energy = cp.asarray(mo_energy)
    qij = cp.asarray(qij)
    nkpts, nocc = qij.shape[:2]
    Pi_00 = cp.zeros((), dtype=cp.complex128)
    for k in range(nkpts):
        eia = mo_energy[k, :nocc, None] - mo_energy[k, None, nocc:]
        weight = eia / (omega**2 + eia**2)
        Pi_00 += 4.0 / nkpts * cp.einsum(
            'ia,ia->', weight, qij[k].conj() * qij[k])
    return Pi_00


def get_rho_response_wing(omega, mo_energy, Lia, qij):
    """Compute the unsigned finite-size wing response."""
    Lia = cp.asarray(Lia)
    mo_energy = cp.asarray(mo_energy)
    qij = cp.asarray(qij)
    nkpts, naux, nocc, nvir = Lia.shape
    Pi = cp.zeros(naux, dtype=cp.complex128)
    for k in range(nkpts):
        eia = mo_energy[k, :nocc, None] - mo_energy[k, None, nocc:]
        eia_q = eia * qij[k].conj() / (omega**2 + eia**2)
        Pi += (4.0 / nkpts *
               Lia[k].reshape(naux, nocc * nvir).dot(eia_q.ravel()))
    return Pi


def get_qij(rpa, q, mo_energy, mo_coeff, uniform_grids=False):
    """Compute the long-wavelength pair-density matrix on GPU grid blocks."""
    if rpa.mol.dimension != 3:
        raise NotImplementedError(
            'KRPA finite-size correction is implemented for 3D cells only')

    cell = rpa.mol
    nocc = rpa.nocc
    nmo = rpa.nmo
    nvir = nmo - nocc
    mo_energy = cp.asarray(mo_energy)
    mo_coeff = cp.asarray(mo_coeff)

    if uniform_grids:
        grids = gen_grid.UniformGrids(cell).build()
    else:
        grids = gen_grid.BeckeGrids(cell)
        grids.level = 4
        grids.build()

    ni = numint.KNumInt()
    ao_ao_grad = cp.zeros((rpa.nkpts, 3, cell.nao, cell.nao),
                          dtype=cp.complex128)
    for ao_ks, weights, _ in ni.block_loop(
            cell, grids, deriv=1, kpts=rpa.kpts):
        for k in range(rpa.nkpts):
            ao = ao_ks[k, 0]
            ao_grad = ao_ks[k, 1:4]
            aow = ao.conj() * weights[:, None]
            ao_ao_grad[k] += contract('gm,xgn->xmn', aow, ao_grad)

    q = cp.asarray(q)
    qij = cp.empty((rpa.nkpts, nocc, nvir), dtype=cp.complex128)
    for k in range(rpa.nkpts):
        q_ao = -1j * cp.einsum('x,xmn->mn', q, ao_ao_grad[k])
        q_mo = mo_coeff[k, :, :nocc].conj().T.dot(q_ao)
        q_mo = q_mo.dot(mo_coeff[k, :, nocc:])
        enm = 1.0 / (mo_energy[k, nocc:, None] -
                     mo_energy[k, None, :nocc])
        qij[k] = enm.T * q_mo / np.sqrt(cell.vol)
    return qij


def get_rho_response_metal(omega, mo_energy, mo_occ, Lpq, kidx):
    """Get Pi=PV for metallic systems.
    P is density-density response function.
    V is two-electron integral.
    See equation 24 in doi.org/10.1021/acs.jctc.0c00704.

    NOTE: this function is different from the one in krgw_ac.py.
    They should be merged in the future. The metal version here
    is more efficient both in memory and computational time.

    Parameters
    ----------
    omega : double
        real position of imaginary frequency
    mo_energy : double ndarray
        orbital energy
    mo_occ : double ndarray
        occupation number
    Lpq : list of complex ndarray
        three-center density-fitting matrix in MO.
        Lpq[ki] contains the naux x (nocc_i + nfrac_i) x (nfrac_i + nvir_i) sub-block.
    kidx : list
        momentum-conserved k-point list kj=kidx[ki]

    Returns
    -------
    Pi : complex ndarray
        Pi in auxiliary basis at freq iw
    """
    mo_energy = cp.asarray(mo_energy)
    mo_occ = cp.asarray(mo_occ)
    Lpq = [cp.asarray(x) for x in Lpq]
    nkpts = len(Lpq)
    naux = Lpq[0].shape[0]

    # Compute Pi for kL
    Pi = cp.zeros(shape=[naux, naux], dtype=cp.complex128)
    for i in range(nkpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]

        idx_occ_i, idx_frac_i, _ = get_idx_metal(mo_occ[i])
        idx_occ_a, idx_frac_a, _ = get_idx_metal(mo_occ[a])

        nocc_i = len(idx_occ_i)
        nfrac_i = len(idx_frac_i)
        nocc_a = len(idx_occ_a)
        nfrac_a = len(idx_frac_a)

        # occupied + fractional
        idx_i = slice(0, nocc_i + nfrac_i)

        # fractional + virtual
        idx_a = slice(nocc_a, len(mo_occ[a]))

        eia = mo_energy[i, idx_i, None] - mo_energy[a, None, idx_a]
        fia = (mo_occ[i][idx_i, None] - mo_occ[a][None, idx_a]) / 2.0

        # factor of 0.5 is for double counting
        if nfrac_a:
            fia[nocc_i:, :nfrac_a] *= 0.5
        # Response from both spin-up and spin-down density
        rho_accum_inner(Pi, eia, omega, Lpq[i], alpha=4.0 / nkpts, fia=fia)

    return Pi


def rho_accum_inner(Pi, eia, omega, Lov, alpha=0.0, fia=None):
    """Get contribution to response function from current occupied-virtual block.

    Parameters
    ----------
    Pi : complex 2d array
        density-density response function, will be overwritten
    eia : double 2d array
        occupied-virtual orbital energy difference
    omega : double
        real position of imaginary frequency
    Lov : complex 3d array
        occupied-virtual block of three-center density-fitting matrix in MO
    alpha : float, optional
        prefactor, by default 0.0
    fia : double 2d array, optional
        occupied-virtual occupation number difference, by default None
    """
    Lov = cp.asarray(Lov)
    eia = cp.asarray(eia)
    naux, nocc, nvir = Lov.shape

    if fia is None:
        eia = eia / (omega**2 + eia**2)
    else:
        eia = eia * fia / (omega**2 + eia**2)
    Lov_2d = Lov.reshape(naux, nocc * nvir)
    Pia = (Lov * eia).reshape(naux, nocc * nvir)
    Pi += alpha * Pia.dot(Lov_2d.conj().T)

    return


def rho_wing_accum_inner(Pi_P0, eia, omega, Lov, qov, alpha=0.0):
    """Accumulate the finite-size-correction wing response for one OV slice.

    Parameters
    ----------
    Pi_P0 : complex 1d array
        finite-size correction to density-density response function, will be overwritten
    eia : double 2d array
        occupied-virtual orbital energy difference
    omega : double
        frequency
    Lov : complex 3d array
        occupied-virtual block of three-center density-fitting matrix in MO
    qov : complex 2d array
        virtual-occupied correction
    alpha : float, optional
        prefactor, by default 0.0
    """
    Lov = cp.asarray(Lov)
    eia = cp.asarray(eia)
    qov = cp.asarray(qov)
    naux, nocc, nvir = Lov.shape
    eia_q = eia * qov.conj() / (omega**2 + eia**2)
    Pi_P0 += alpha * Lov.reshape(naux, nocc * nvir).dot(eia_q.ravel())

    return


def get_rpa_ecorr(rpa, freqs, wts, mo_energy=None, mo_coeff=None,
                  mo_occ=None):
    """Compute RPA correlation energy.

    Parameters
    ----------
    rpa : KRPA
        rpa object
    freqs : double 1d array
            frequency grid
        wts : double 1d array
            weight of grids

    Returns
    -------
    e_corr : double
        correlation energy
    """
    if mo_coeff is None:
        mo_coeff = _mo_frozen(rpa, rpa._scf.mo_coeff)
    if mo_energy is None:
        mo_energy = _mo_energy_frozen(rpa, rpa._scf.mo_energy)
    if mo_occ is None:
        mo_occ = _mo_occ_frozen(rpa, rpa._scf.mo_occ)
    mo_coeff = cp.asarray(mo_coeff)
    mo_energy = cp.asarray(mo_energy)
    mo_occ = cp.asarray(mo_occ)

    nocc = rpa.nocc
    nkpts = rpa.nkpts
    is_metal = hasattr(rpa._scf, 'sigma')
    kconserv_table = get_kconserv_ria_efficient(rpa.mol, rpa.kpts)

    if rpa.fc:
        qij, q_abs, nq_pts = rpa.get_q_mesh(mo_energy, mo_coeff)

    e_corr = cp.zeros((), dtype=cp.complex128)
    for kL in range(nkpts):
        kidx = kconserv_table[kL]
        if is_metal:
            mo_left = []
            mo_right = []
            for k in range(nkpts):
                idx_occ, idx_frac, _ = get_idx_metal(mo_occ[k])
                nocc_k = len(idx_occ)
                nfrac_k = len(idx_frac)
                mo_left.append(mo_coeff[k, :, :nocc_k + nfrac_k])
                mo_right.append(mo_coeff[k, :, nocc_k:])
        else:
            mo_left = [mo_coeff[k, :, :nocc] for k in range(nkpts)]
            mo_right = [mo_coeff[k, :, nocc:] for k in range(nkpts)]

        sectors = _transform_cderi(rpa, kidx, mo_left, mo_right)
        Lij_by_k, signs = _stack_metric_sectors(sectors)
        if not is_metal:
            Lij = cp.stack(Lij_by_k)

        for w, omega in enumerate(freqs):
            if is_metal:
                Pi_unsigned = get_rho_response_metal(
                    omega, mo_energy, mo_occ, Lij_by_k, kidx)
            else:
                Pi_unsigned = get_rho_response(
                    omega, mo_energy, Lij, kidx)

            if kL == 0 and rpa.fc:
                for iq in range(nq_pts):
                    qnorm = np.linalg.norm(q_abs[iq])
                    Pi_00 = (4.0 * np.pi / qnorm**2 *
                             get_rho_response_head(omega, mo_energy, qij[iq]))
                    Pi_P0 = (np.sqrt(4.0 * np.pi) / qnorm *
                             get_rho_response_wing(
                                 omega, mo_energy, Lij, qij[iq]))
                    Pi_fc = cp.zeros((len(signs) + 1, len(signs) + 1),
                                     dtype=cp.complex128)
                    Pi_fc[0, 0] = Pi_00
                    Pi_fc[0, 1:] = Pi_P0.conj()
                    Pi_fc[1:, 0] = Pi_P0
                    Pi_fc[1:, 1:] = Pi_unsigned
                    signs_fc = cp.concatenate((cp.ones(1), signs))
                    e_corr += get_rpa_ecorr_w(
                        _apply_metric(Pi_fc, signs_fc), wts[w])
            else:
                e_corr += get_rpa_ecorr_w(
                    _apply_metric(Pi_unsigned, signs), wts[w])

    e_corr = e_corr.real / (2.0 * np.pi * nkpts)
    return _to_float(e_corr)


def get_rpa_ecorr_outcore(rpa, freqs, wts, mo_energy=None,
                          mo_coeff=None):
    """Low-memory routine to compute RPA correlation energy.

    Parameters
    ----------
    rpa : KRPA
        rpa object
    freqs : double 1d array
        frequency grid
    wts : double 1d array
        weight of grids

    Returns
    -------
    e_corr : double
        correlation energy
    """
    if rpa.segsize <= 0:
        raise ValueError('KRPA segsize must be positive')
    if mo_coeff is None:
        mo_coeff = _mo_frozen(rpa, rpa._scf.mo_coeff)
    if mo_energy is None:
        mo_energy = _mo_energy_frozen(rpa, rpa._scf.mo_energy)
    mo_coeff = cp.asarray(mo_coeff)
    mo_energy = cp.asarray(mo_energy)

    nocc = rpa.nocc
    nkpts = rpa.nkpts
    nw = len(freqs)
    kconserv_table = get_kconserv_ria_efficient(rpa.mol, rpa.kpts)
    if rpa.fc:
        qij, q_abs, nq_pts = rpa.get_q_mesh(mo_energy, mo_coeff)

    e_corr = cp.zeros((), dtype=cp.complex128)
    for kL in range(nkpts):
        kidx = kconserv_table[kL]
        Pi = Pi_P0 = signs = None
        for orb_start in range(0, nocc, rpa.segsize):
            orb_end = min(orb_start + rpa.segsize, nocc)
            mo_left = [mo_coeff[k, :, orb_start:orb_end]
                       for k in range(nkpts)]
            mo_right = [mo_coeff[k, :, nocc:] for k in range(nkpts)]
            sectors = _transform_cderi(rpa, kidx, mo_left, mo_right)
            Lij_by_k, signs_iter = _stack_metric_sectors(sectors)
            if signs is None:
                signs = signs_iter
                naux = len(signs)
                Pi = cp.zeros((nw, naux, naux), dtype=cp.complex128)
                if kL == 0 and rpa.fc:
                    Pi_P0 = cp.zeros((nq_pts, nw, naux),
                                     dtype=cp.complex128)
            elif not bool(cp.array_equal(signs, signs_iter).item()):
                raise RuntimeError('Inconsistent GDF metric between orbital segments')

            for i in range(nkpts):
                j = int(kidx[i])
                Lij_slice = Lij_by_k[i]
                eia = (mo_energy[i, orb_start:orb_end, None] -
                       mo_energy[j, None, nocc:])
                for w, omega in enumerate(freqs):
                    rho_accum_inner(Pi[w], eia, omega, Lij_slice,
                                    alpha=4.0 / nkpts)
                    if kL == 0 and rpa.fc:
                        for iq in range(nq_pts):
                            rho_wing_accum_inner(
                                Pi_P0[iq, w], eia, omega, Lij_slice,
                                qij[iq, i, orb_start:orb_end],
                                alpha=4.0 / nkpts)

        for w, omega in enumerate(freqs):
            if kL == 0 and rpa.fc:
                for iq in range(nq_pts):
                    qnorm = np.linalg.norm(q_abs[iq])
                    Pi_00 = (4.0 * np.pi / qnorm**2 *
                             get_rho_response_head(omega, mo_energy, qij[iq]))
                    Pi_P0_iq = (np.sqrt(4.0 * np.pi) / qnorm *
                                Pi_P0[iq, w])
                    Pi_fc = cp.zeros((len(signs) + 1, len(signs) + 1),
                                     dtype=cp.complex128)
                    Pi_fc[0, 0] = Pi_00
                    Pi_fc[0, 1:] = Pi_P0_iq.conj()
                    Pi_fc[1:, 0] = Pi_P0_iq
                    Pi_fc[1:, 1:] = Pi[w]
                    signs_fc = cp.concatenate((cp.ones(1), signs))
                    e_corr += get_rpa_ecorr_w(
                        _apply_metric(Pi_fc, signs_fc), wts[w])
            else:
                e_corr += get_rpa_ecorr_w(
                    _apply_metric(Pi[w], signs), wts[w])

    e_corr = e_corr.real / (2.0 * np.pi * nkpts)
    return _to_float(e_corr)


def get_rpa_ecorr_outcore_metal(rpa, freqs, wts, mo_energy=None,
                                mo_coeff=None, mo_occ=None):
    """Low-memory routine to compute RPA correlation energy for metals.

    Parameters
    ----------
    rpa : KRPA
        rpa object
    freqs : double 1d array
        frequency grid
    wts : double 1d array
        weight of grids

    Returns
    -------
    e_corr : double
        correlation energy
    """
    if rpa.segsize <= 0:
        raise ValueError('KRPA segsize must be positive')
    if mo_coeff is None:
        mo_coeff = _mo_frozen(rpa, rpa._scf.mo_coeff)
    if mo_energy is None:
        mo_energy = _mo_energy_frozen(rpa, rpa._scf.mo_energy)
    if mo_occ is None:
        mo_occ = _mo_occ_frozen(rpa, rpa._scf.mo_occ)
    mo_coeff = cp.asarray(mo_coeff)
    mo_energy = cp.asarray(mo_energy)
    mo_occ = cp.asarray(mo_occ)

    nkpts = rpa.nkpts
    nw = len(freqs)
    orbital_info = []
    for k in range(nkpts):
        idx_occ, idx_frac, _ = get_idx_metal(mo_occ[k])
        orbital_info.append((len(idx_occ), len(idx_frac)))
    max_left = max(nocc + nfrac for nocc, nfrac in orbital_info)
    kconserv_table = get_kconserv_ria_efficient(rpa.mol, rpa.kpts)

    e_corr = cp.zeros((), dtype=cp.complex128)
    for kL in range(nkpts):
        kidx = kconserv_table[kL]
        Pi = signs = None
        for orb_start in range(0, max_left, rpa.segsize):
            mo_left = []
            mo_right = []
            for k in range(nkpts):
                nocc_k, nfrac_k = orbital_info[k]
                orb_end_k = min(orb_start + rpa.segsize,
                                nocc_k + nfrac_k)
                mo_left.append(mo_coeff[k, :, orb_start:orb_end_k])
                mo_right.append(mo_coeff[k, :, nocc_k:])

            sectors = _transform_cderi(rpa, kidx, mo_left, mo_right)
            Lij_by_k, signs_iter = _stack_metric_sectors(sectors)
            if signs is None:
                signs = signs_iter
                Pi = cp.zeros((nw, len(signs), len(signs)),
                              dtype=cp.complex128)
            elif not bool(cp.array_equal(signs, signs_iter).item()):
                raise RuntimeError('Inconsistent GDF metric between orbital segments')

            for i in range(nkpts):
                j = int(kidx[i])
                nocc_i, nfrac_i = orbital_info[i]
                nocc_j, nfrac_j = orbital_info[j]
                orb_end = min(orb_start + rpa.segsize, nocc_i + nfrac_i)
                if orb_end <= orb_start:
                    continue
                eia = (mo_energy[i, orb_start:orb_end, None] -
                       mo_energy[j, None, nocc_j:])
                fia = (mo_occ[i, orb_start:orb_end, None] -
                       mo_occ[j, None, nocc_j:]) / 2.0
                if nfrac_j:
                    if orb_start >= nocc_i:
                        fia[:, :nfrac_j] *= 0.5
                    elif orb_end > nocc_i:
                        fia[nocc_i - orb_start:, :nfrac_j] *= 0.5
                for w, omega in enumerate(freqs):
                    rho_accum_inner(
                        Pi[w], eia, omega, Lij_by_k[i],
                        alpha=4.0 / nkpts, fia=fia)

        for w in range(nw):
            e_corr += get_rpa_ecorr_w(
                _apply_metric(Pi[w], signs), wts[w])

    e_corr = e_corr.real / (2.0 * np.pi * nkpts)
    return _to_float(e_corr)


def get_rpa_ecorr_w(Pi_w, wts_w):
    """Get contribution to RPA correlation energy from a single frequency.

    Parameters
    ----------
    Pi_w : complex 2d array
        density-density response function at a single frequency
    wts_w : double
        weights of the frequency

    Returns
    -------
    e_corr : double
        correlation energy
    """
    Pi_w = cp.asarray(Pi_w)
    dielectric = cp.eye(Pi_w.shape[0], dtype=Pi_w.dtype) - Pi_w
    ec_w = cp.trace(Pi_w) + cp.linalg.slogdet(dielectric)[1]
    return ec_w * wts_w


def get_rpa_exx(rpa, acfd=False, correction_only=False):
    """Calculate RPA exchange energy.
    For gapped systems, Hartree-Fock and adiabatic connection fluctuation dissipation exchange energies are the same.
    For metallic systems, they are different.
    The ACFD exchange energy is given by equation 12 in doi.org/10.1103/PhysRevB.81.115126

    Parameters
    ----------
    rpa : KRPA
        rpa object
    acfd : bool, optional
        calculate ACFD exchange energy, by default False
    correction_only : bool, optional
        only calculate the correction term, by default False

    Returns
    -------
    ex : double
        exchange energy
    """
    mo_coeff = cp.asarray(rpa._scf.mo_coeff)
    mo_occ = cp.asarray(rpa._scf.mo_occ)
    nkpts = rpa.nkpts
    is_metal = hasattr(rpa._scf, 'sigma')
    kconserv_table = get_kconserv_ria_efficient(rpa.mol, rpa.kpts)

    nocc_by_k = []
    for k in range(nkpts):
        if is_metal:
            idx_occ, idx_frac, _ = get_idx_metal(mo_occ[k])
            nocc_by_k.append(len(idx_occ) + len(idx_frac))
        else:
            nocc_by_k.append(int(cp.count_nonzero(mo_occ[k]).item()))
    mo_occ_coeff = [mo_coeff[k, :, :nocc_by_k[k]] for k in range(nkpts)]

    ex = cp.zeros((), dtype=cp.complex128)
    for kL in range(nkpts):
        kidx = kconserv_table[kL]
        sectors = _transform_cderi(
            rpa, kidx, mo_occ_coeff, mo_occ_coeff)
        Lij_by_k, signs = _stack_metric_sectors(sectors)
        for km in range(nkpts):
            kn = int(kidx[km])
            Lij = Lij_by_k[km]
            if is_metal:
                nocc_i = nocc_by_k[km]
                nocc_j = nocc_by_k[kn]
                occ_i = mo_occ[km, :nocc_i, None]
                occ_j = mo_occ[kn, None, :nocc_j]
                if acfd:
                    occ_weight = cp.minimum(occ_i, occ_j) / 2.0
                    if correction_only:
                        occ_weight -= occ_i * occ_j / 4.0
                else:
                    occ_weight = occ_i * occ_j / 4.0
                Lij_weighted = Lij * occ_weight[None]
                ex -= cp.einsum(
                    'Lij,Lij,L->', Lij_weighted.conj(), Lij, signs)
            else:
                ex -= cp.einsum('Lij,Lij,L->', Lij.conj(), Lij, signs)

    ex = ex.real / nkpts**2

    if rpa._scf.exxdiv == 'ewald' and rpa._scf.cell.dimension != 0:
        madelung = tools.pbc.madelung(rpa._scf.cell, rpa.kpts)
        ex -= madelung * cp.sum(mo_occ**2) / (4.0 * nkpts)
        if acfd:
            for k in range(nkpts):
                idx_occ, idx_frac, _ = get_idx_metal(mo_occ[k])
                nactive_occ = len(idx_occ) + len(idx_frac)
                f_i = mo_occ[k, :nactive_occ] / 2.0
                ex -= madelung * cp.sum(f_i - f_i * f_i) / nkpts

    return _to_float(ex)


def get_kconserv_ria_efficient(cell, kpts, tol=1e-12):
    r"""Get the momentum conservation array for single excitation amplitudes
    for a set of k-points with appropriate k-shift.


    Given k-point indices (kshift, m) the array kconserv[kshift,m] returns
    the index n that satisfies momentum conservation,

        (k(m) - k(n) - k(kshift)) \dot a = 2n\pi
    """
    # The conservation table is built for momentum transfers, which are invariant
    # under a uniform shift of all k-points.
    kpts = kpts - kpts[0]

    nkpts = kpts.shape[0]
    a = cell.lattice_vectors() / (2 * np.pi)

    kconserv = np.zeros((nkpts, nkpts), dtype=int)
    kvKM = -kpts[:, None, :] + kpts[:, :]
    for N, kvN in enumerate(kpts):
        kvKMN = np.einsum('wx,kmx->wkm', a, kvKM - kvN, optimize=True)
        # check whether (1/(2pi) k_{KLN} dot a) is an integer
        kvKMN_int = np.rint(kvKMN)
        mask = np.einsum('wkm->km', abs(kvKMN - kvKMN_int), optimize=True) < tol
        kconserv[mask] = N
    return kconserv


class KRPA(lib.StreamObject):
    _keys = {
        'mol', '_scf', 'max_memory', 'frozen', 'grids_alg', 'outcore',
        'segsize', 'fc', 'fc_grid', 'acfd_exx', '_nocc', '_nmo', 'kpts',
        'nkpts', 'mo_energy', 'mo_coeff', 'mo_occ', 'e_corr', 'e_hf',
        'e_tot', 'with_df',
    }

    def __init__(self, mf, frozen=None):
        if not isinstance(getattr(mf, 'mo_coeff', None), cp.ndarray):
            raise TypeError(
                'GPU KRPA requires a GPU mean-field object; '
                'call mf.to_gpu() first')
        self.mol = mf.mol  # mol object
        self._scf = mf  # mean-field object
        self.verbose = self.mol.verbose  # verbose level
        self.stdout = self.mol.stdout  # standard output
        self.max_memory = mf.max_memory  # max memory in MB

        # options
        self.frozen = frozen  # frozen orbital options
        self.grids_alg = 'legendre'  # algorithm to generate grids
        self.outcore = False  # low-memory routine
        self.segsize = 50  # number of orbitals in one segment for outcore
        self.fc = False  # finite-size correction
        self.fc_grid = False  # grids for finite-size correction
        self.acfd_exx = False  # calculate ACFD exchange energy

        # don't modify the following attributes, they are not input options
        self._nocc = None  # number of occupied orbitals
        self._nmo = None  # number of orbitals (exclude frozen orbitals)
        self.kpts = mf.kpts  # k-points
        self.nkpts = len(self.kpts)  # number of k-points
        self.mo_energy = cp.array(mf.mo_energy, copy=True)  # orbital energy
        self.mo_coeff = cp.array(mf.mo_coeff, copy=True)  # orbital coefficient
        self.mo_occ = cp.array(mf.mo_occ, copy=True)  # occupation number
        self.e_corr = None  # correlation energy
        self.e_hf = None  # Hartree-Fock energy
        self.e_tot = None  # total energy

        # KRPA must use GDF integrals
        if isinstance(getattr(mf, 'with_df', None), GDF):
            self.with_df = mf.with_df
        else:
            raise NotImplementedError('GPU KRPA requires gpu4pyscf.pbc.df.GDF')

        return

    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        log.info(f'RPA nocc = {nocc}, nvir = {nvir}, nkpts = {nkpts}')
        if self.frozen is not None:
            log.info(f'frozen orbitals = {str(self.frozen)}')
        log.info('grid type = %s', self.grids_alg)
        log.info('outcore mode = %s', self.outcore)
        if self.outcore is True:
            log.info('outcore segment size = %d', self.segsize)
        log.info('RPA finite size corrections = %s', self.fc)
        log.info('ACFD exchange energy = %s', self.acfd_exx)
        log.info('')
        return

    @property
    def nocc(self):
        if self._nocc is not None:
            return self._nocc
        frozen_mask = get_frozen_mask(self)
        nkpts = len(self._scf.mo_energy)
        nelec = 0.0
        for k in range(nkpts):
            nelec += _to_float(cp.sum(self._scf.mo_occ[k][frozen_mask[k]]))
        nelec = int(nelec / nkpts)
        return nelec // 2

    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        if self._nmo is not None:
            return self._nmo
        frozen_mask = get_frozen_mask(self)
        return len(self._scf.mo_energy[0][frozen_mask[0]])

    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    def get_nocc(self, per_kpoint=False):
        if not per_kpoint:
            return self.nocc
        masks = get_frozen_mask(self)
        return [int(round(_to_float(cp.sum(self._scf.mo_occ[k][masks[k]])) / 2.0))
                for k in range(self.nkpts)]

    def get_nmo(self, per_kpoint=False):
        if not per_kpoint:
            return self.nmo
        return [int(mask.sum()) for mask in get_frozen_mask(self)]

    get_frozen_mask = get_frozen_mask

    def kernel(self, mo_energy=None, mo_coeff=None, nw=None, with_e_hf=None):
        """RPA correlation and total energy

        Calculated total energy, HF energy and RPA correlation energy
        are stored in self.e_tot, self.e_hf, self.e_corr

        Parameters
        ----------
        mo_energy : double array
            molecular orbital energies
        mo_coeff : double ndarray
            molecular orbital coefficients
        nw : int, optional
            number of frequency point on imaginary axis, by default None
        with_e_hf : float, optional
            If given, overrides the HF energy computation.

        Returns
        -------
        e_tot : float
            RPA total energy
        e_hf : float
            HF energy (exact exchange for given mo_coeff)
        e_corr : float
            RPA correlation energy
        """
        if mo_coeff is None:
            mo_coeff = _mo_frozen(self, self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = _mo_energy_frozen(self, self._scf.mo_energy)

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        self.e_tot, self.e_hf, self.e_corr = kernel(
            self, mo_energy, mo_coeff, nw=nw, with_e_hf=with_e_hf)
        logger.timer(self, 'RPA', *cput0)
        return self.e_tot, self.e_hf, self.e_corr

    def get_grids(self, alg=None, nw=None, mo_energy=None):
        """Generate grids for integration.

        Parameters
        ----------
        alg : str, optional
            algorithm for generating grids, by default None
        nw : int, optional
            number of grids, by default None
        mo_energy : double 2d array, optional
            orbital energy, used for minimax grids, by default None

        Returns
        -------
        freqs : double 1d array
            frequency grid
        wts : double 1d array
            weight of grids
        """
        if alg is None:
            alg = self.grids_alg
        if mo_energy is None:
            mo_energy = _mo_energy_frozen(self, self._scf.mo_energy)
        if alg == 'legendre':
            nw = 40 if nw is None else nw
            freqs, wts = _get_scaled_legendre_roots(nw)
        else:
            raise NotImplementedError('Grids algorithm not implemented!')

        return freqs, wts

    def get_q_mesh(self, mo_energy, mo_coeff):
        """Get q-mesh for finite size correction.
        Equation 39-42 in doi.org/10.1021/acs.jctc.0c00704

        Parameters
        ----------
        mo_energy : double 2d array
            orbital energy
        mo_coeff : double 3d array
            coefficient from AO to MO

        Returns
        -------
        qij : double 1d array
            q-mesh grids
        q_abs : double 1d array
            absolute positions of q-mesh grids
        nq_pts : init
            number of q-mesh grids
        """
        nocc = self.nocc
        nmo = self.nmo
        nkpts = self.nkpts
        # Set up q mesh for q->0 finite size correction
        if not self.fc_grid:
            q_pts = np.array([1e-3, 0, 0], dtype=np.double).reshape(1, 3)
        else:
            Nq = 3
            q_pts = np.zeros(shape=[Nq**3 - 1, 3], dtype=np.double)
            for i in range(Nq):
                for j in range(Nq):
                    for k in range(Nq):
                        if i == 0 and j == 0 and k == 0:
                            continue
                        else:
                            q_pts[i * Nq**2 + j * Nq + k - 1, 0] = k * 5e-4
                            q_pts[i * Nq**2 + j * Nq + k - 1, 1] = j * 5e-4
                            q_pts[i * Nq**2 + j * Nq + k - 1, 2] = i * 5e-4
        nq_pts = len(q_pts)
        q_abs = self.mol.get_abs_kpts(q_pts)

        # qij = <psi_ik | exp(i*q*r) | psi_a,k-q> / sqrt(Omega)
        qij = cp.empty((nq_pts, nkpts, nocc, nmo - nocc),
                       dtype=cp.complex128)
        for k in range(nq_pts):
            qij[k] = get_qij(self, q_abs[k], mo_energy, mo_coeff)

        return qij, q_abs, nq_pts

    def get_acfd_exx(self, correction_only=False):
        """Calculate ACFD exchange energy.

        Parameters
        ----------
        correction_only : bool
            only return the correction term

        Returns
        -------
        ex_acfd : double
            ACFD exchange energy
        """
        ex_acfd = get_rpa_exx(self, acfd=True, correction_only=correction_only)
        return ex_acfd

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        from pyscf.pbc.gw.krpa import KRPA as CPUKRPA
        out = CPUKRPA(self._scf.to_cpu(), frozen=self.frozen)
        return utils.to_cpu(self, out=out)
