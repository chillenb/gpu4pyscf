# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# modified by Xiaojie Wu <wxj6000@gmail.com>; Zhichen Pu <hoshishin@163.com>

"""
DIIS
"""

import numpy as np
import cupy as cp
import scipy.linalg
import scipy.optimize
import pyscf.scf.diis as cpu_diis
import gpu4pyscf.lib as lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, eigh, sandwich_dot, pack_tril, unpack_tril, get_avail_mem,
    asarray)

# J. Mol. Struct. 114, 31-34 (1984); DOI:10.1016/S0022-2860(84)87198-7
# PCCP, 4, 11 (2002); DOI:10.1039/B108658H
# GEDIIS, JCTC, 2, 835 (2006); DOI:10.1021/ct050275a
# C2DIIS, IJQC, 45, 31 (1993); DOI:10.1002/qua.560450106
# SCF-EDIIS, JCP 116, 8255 (2002); DOI:10.1063/1.1470195

# error vector = SDF-FDS
# error vector = F_ai ~ (S-SDS)*S^{-1}FDS = FDS - SDFDS ~ FDS-SDF in converge
class CDIIS(lib.diis.DIIS):
    incore = None

    def __init__(self, mf=None, filename=None):
        lib.diis.DIIS.__init__(self, mf, filename)
        self.rollback = False
        self.Corth = None
        self.space = 8
        self.damp = 0
        self.ndamp_cycles = -1

    def update(self, s, d, f, *args, **kwargs):
        if d.dtype == cp.complex128:
            s = s.astype(cp.complex128)
        errvec = self._sdf_err_vec(s, d, f)
        if self.incore is None:
            mem_avail = get_avail_mem()
            self.incore = errvec.nbytes*2 * (20+self.space) < mem_avail
            if not self.incore:
                logger.debug(self, 'Large system detected. DIIS intermediates '
                             'are saved in the host memory')
        f_prev = kwargs.get('f_prev', None)
        cycle = kwargs.get('cycle', 0)
        if self.Corth.ndim == 3:
            nao, nmo = self.Corth.shape[-2:]
        else:
            assert self.Corth.ndim == 2
            nao, nmo = self.Corth.shape
        errvec = pack_tril(errvec.reshape(-1,nmo,nmo))
        f_tril = pack_tril(f.reshape(-1,nao,nao))

        if abs(self.damp) > 1e-6 and cycle == self.ndamp_cycles:
            logger.debug(self, 'DIIS is being cleared after %d cycles', self.ndamp_cycles)
            self.clear()

        if abs(self.damp) < 1e-6 or f_prev is None or cycle >= self.ndamp_cycles:
            logger.debug(self, 'DIIS damping factor %s, damping is inactive', self.damp)
            xnew = lib.diis.DIIS.update(self, f_tril, xerr=errvec)
        else:
            logger.debug(self, 'DIIS damping factor %s, damping active', self.damp)
            f_prev_tril = pack_tril(f_prev.reshape(-1,nao,nao))
            xnew = lib.diis.DIIS.update(self, f_tril*(1-self.damp) + f_prev_tril*self.damp, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return unpack_tril(xnew).reshape(f.shape)

    def get_num_vec(self):
        if self.rollback:
            return self._head
        else:
            return len(self._bookkeep)

    def _sdf_err_vec(self, s, d, f):
        '''error vector = SDF - FDS'''
        if f.ndim == s.ndim+1: # UHF
            assert len(f) == 2
            if s.ndim == 2: # molecular SCF or single k-point
                if self.Corth is None:
                    self.Corth = eigh(f[0], s)[1]
                sdf = cp.empty_like(f)
                s.dot(d[0]).dot(f[0], out=sdf[0])
                s.dot(d[1]).dot(f[1], out=sdf[1])
                sdf = sandwich_dot(sdf, self.Corth)
                errvec = sdf - sdf.conj().transpose(0,2,1)
            else: # k-points
                if self.Corth is None:
                    self.Corth = cp.empty_like(s)
                    for k, (fk, sk) in enumerate(zip(f[0], s)):
                        self.Corth[k] = eigh(fk, sk)[1]
                Corth = asarray(self.Corth)
                sdf = cp.empty_like(f)
                tmp = None
                tmp = contract('Kij,Kjk->Kik', d[0], f[0], out=tmp)
                contract('Kij,Kjk->Kik', s, tmp, out=sdf[0])
                tmp = contract('Kpq,Kqj->Kpj', sdf[0], Corth, out=tmp)
                contract('Kpj,Kpi->Kij', tmp, Corth.conj(), out=sdf[0])

                tmp = contract('Kij,Kjk->Kik', d[1], f[1], out=tmp)
                contract('Kij,Kjk->Kik', s, tmp, out=sdf[1])
                tmp = contract('Kpq,Kqj->Kpj', sdf[1], Corth, out=tmp)
                contract('Kpj,Kpi->Kij', tmp, Corth.conj(), out=sdf[1])
                errvec = sdf - sdf.conj().transpose(0,1,3,2)
        else: # RHF
            assert f.ndim == s.ndim
            if f.ndim == 2: # molecular SCF or single k-point
                if self.Corth is None:
                    self.Corth = eigh(f, s)[1]
                sdf = s.dot(d).dot(f)
                sdf = sandwich_dot(sdf, self.Corth)
                errvec = sdf - sdf.conj().T
            else: # k-points
                if self.Corth is None:
                    self.Corth = cp.empty_like(s)
                    for k, (fk, sk) in enumerate(zip(f, s)):
                        self.Corth[k] = eigh(fk, sk)[1]
                sd = contract('Kij,Kjk->Kik', s, d)
                sdf = contract('Kij,Kjk->Kik', sd, f)
                Corth = asarray(self.Corth)
                sdf = contract('Kpq,Kqj->Kpj', sdf, Corth)
                sdf = contract('Kpj,Kpi->Kij', sdf, Corth.conj())
                errvec = sdf - sdf.conj().transpose(0,2,1)
        return errvec.ravel()

SCFDIIS = SCF_DIIS = DIIS = CDIIS

class EDIIS(lib.diis.DIIS):
    '''SCF-EDIIS
    Ref: JCP 116, 8255 (2002); DOI:10.1063/1.1470195
    '''
    def update(self, s, d, f, mf, h1e, vhf, *args, **kwargs):
        if self._head >= self.space:
            self._head = 0
        if not self._buffer:
            shape = (self.space,) + f.shape
            self._buffer['dm'  ] = cp.zeros(shape, dtype=f.dtype)
            self._buffer['fock'] = cp.zeros(shape, dtype=f.dtype)
            self._buffer['etot'] = cp.zeros(self.space)
        self._buffer['dm'  ][self._head] = d
        self._buffer['fock'][self._head] = f
        self._buffer['etot'][self._head] = mf.energy_elec(d, h1e, vhf)[0]
        self._head += 1

        ds = self._buffer['dm'  ]
        fs = self._buffer['fock']
        es = self._buffer['etot']
        etot, c = ediis_minimize(es.get(), ds.get(), fs.get())
        c = cp.asarray(c)
        logger.debug1(self, 'E %s  diis-c %s', etot, c)
        fock = cp.einsum('i,i...pq->...pq', c, fs)
        return fock

def ediis_minimize(es, ds, fs):
    nx = es.size
    nao = ds.shape[-1]
    ds = ds.reshape(nx,-1,nao,nao)
    fs = fs.reshape(nx,-1,nao,nao)
    df = np.einsum('inpq,jnqp->ij', ds, fs).real
    diag = df.diagonal()
    df = diag[:,None] + diag - df - df.T

    def costf(x):
        c = x**2 / (x**2).sum()
        return np.einsum('i,i', c, es) - np.einsum('i,ij,j', c, df, c)

    def grad(x):
        x2sum = (x**2).sum()
        c = x**2 / x2sum
        fc = es - 2*np.einsum('i,ik->k', c, df)
        cx = np.diag(x*x2sum) - np.einsum('k,n->kn', x**2, x)
        cx *= 2/x2sum**2
        return np.einsum('k,kn->n', fc, cx)

    if False:
        x0 = np.random.random(nx)
        dfx0 = np.zeros_like(x0)
        for i in range(nx):
            x1 = x0.copy()
            x1[i] += 1e-4
            dfx0[i] = (costf(x1) - costf(x0))*1e4
        print((dfx0 - grad(x0)) / dfx0)

    res = scipy.optimize.minimize(costf, np.ones(nx), method='BFGS',
                                  jac=grad, tol=1e-9)
    return res.fun, (res.x**2)/(res.x**2).sum()
