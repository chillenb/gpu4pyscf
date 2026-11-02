# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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

'''Truncated and smooth truncated Coulomb (sTC) exchange kernels'''

import unittest
import numpy as np
import cupy as cp
import scipy.special
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as scf_cpu
from pyscf.pbc.tools import pbc as pbc_cpu
from pyscf.pbc.df import fft as fft_cpu, aft as aft_cpu
from gpu4pyscf.pbc.tools import pbc as pbc_gpu, stc
from gpu4pyscf.pbc.df import fft, aft, df
from gpu4pyscf.pbc import scf

try:
    from pyscf.pbc.df import rsdf as rsdf_cpu
    _HAS_CPU_STC = hasattr(pbc_cpu, 'precompute_lr_exx')
except ImportError:
    _HAS_CPU_STC = False

def setUpModule():
    global cell, kpts, dm
    cell = pgto.M(atom='C 0 0 0; C 0.89 0.89 0.89', unit='Angstrom',
                  a=np.array([[0, 1.78, 1.78], [1.78, 0, 1.78], [1.78, 1.78, 0]]),
                  basis='gth-szv', pseudo='gth-pade', verbose=0,
                  precision=1e-10, mesh=[25]*3)
    kpts = cell.make_kpts([2,2,2])
    dm = np.asarray(scf_cpu.KRHF(cell, kpts).get_init_guess(key='atom'))

def tearDownModule():
    global cell, kpts, dm
    del cell, kpts, dm

class _Holder:
    omega_dot_Rc = 4.
    def __init__(self, kpts):
        self.kpts = kpts

class KnownValues(unittest.TestCase):
    def test_erfcx_complex(self):
        rng = np.random.default_rng(1)
        a = np.append(10**rng.uniform(-2, 1.8, 2000), rng.uniform(-20, 20, 2000))
        b = np.append(10**rng.uniform(-6, 2.4, 2000), rng.uniform(-20, 20, 2000))
        z = a + 1j*b
        ref = scipy.special.erfcx(z)
        out = stc.erfcx(z).get()
        finite = np.isfinite(ref)
        assert abs(out[finite] - ref[finite]).max() / abs(ref[finite]).max() < 1e-14
        assert (abs(out[finite] - ref[finite]) / abs(ref[finite])).max() < 1e-13

    def test_sph_truncated_lr(self):
        # 4pi/q \int_0^Rc erf(omega r) sin(q r) dr by quadrature
        x, w = np.polynomial.legendre.leggauss(400)
        for Rc, omega in [(3., .1), (10., .3)]:
            r = .5*Rc*(x + 1)
            wr = .5*Rc*w
            q = np.append(10.**np.arange(-6, 0, .5), np.linspace(.1, 3, 20))
            ref = 4*np.pi * (wr*scipy.special.erf(omega*r)*r*np.sinc(q[:,None]*r/np.pi)).sum(axis=1)
            out = stc.sph_truncated_lr(q**2, Rc, omega).get()
            assert abs(out - ref).max() / abs(ref).max() < 1e-12
            ref0 = 4*np.pi * (wr*scipy.special.erf(omega*r)*r).sum()
            out0 = stc.sph_truncated_lr(np.zeros(1), Rc, omega).get()[0]
            assert abs(out0 - ref0) < 1e-12 * ref0

    @unittest.skipIf(not _HAS_CPU_STC, 'requires pyscf with sTC')
    def test_get_coulG(self):
        # Without wrap-around, k+G are identical in pyscf and gpu4pyscf
        for q in (kpts[7] - kpts[0], kpts[3] - kpts[5], np.zeros(3)):
            for exxdiv in stc.TRUNC_EXXDIV + stc.STC_EXXDIV:
                for omega in (0., .3, -.3):
                    for withSR in (True, False):
                        ref = pbc_cpu.get_coulG(cell, q, exxdiv, _Holder(kpts),
                                                omega=omega, withSR=withSR,
                                                wrap_around=False)
                        out = pbc_gpu.get_coulG(cell, q, exxdiv, _Holder(kpts),
                                                omega=omega, withSR=withSR,
                                                wrap_around=False, kpts=kpts).get()
                        assert abs(out - ref).max() < 1e-12 * abs(ref).max()

    @unittest.skipIf(not _HAS_CPU_STC, 'requires pyscf with sTC')
    def test_get_coulG_off_lattice(self):
        q = np.array([.11, -.07, .05])
        for exxdiv in ('vcut_ws', 'smooth_vcut_ws'):
            ref = pbc_cpu.get_coulG(cell, q, exxdiv, _Holder(kpts), wrap_around=False)
            out = pbc_gpu.get_coulG(cell, q, exxdiv, _Holder(kpts), wrap_around=False,
                                    kpts=kpts).get()
            assert abs(out - ref).max() < 1e-12 * abs(ref).max()

    @unittest.skipIf(not _HAS_CPU_STC, 'requires pyscf with sTC')
    def test_fftdf_aftdf_k(self):
        for exxdiv in stc.STC_EXXDIV:
            for omega in (None, .3, -.3):
                ref = fft_cpu.FFTDF(cell, kpts).get_jk(
                    dm, kpts=kpts, with_j=False, omega=omega, exxdiv=exxdiv)[1]
                vk = fft.FFTDF(cell, kpts).get_jk(
                    cp.asarray(dm), kpts=kpts, with_j=False, omega=omega, exxdiv=exxdiv)[1]
                assert abs(vk.get() - ref).max() < 1e-9
                vk = aft.AFTDF(cell, kpts).set(mesh=cell.mesh).get_jk(
                    cp.asarray(dm), kpts=kpts, with_j=False, omega=omega, exxdiv=exxdiv)[1]
                assert abs(vk.get() - ref).max() < 1e-9

    def test_fftdf_k_gamma_point(self):
        dm0 = dm[0].real
        for exxdiv in stc.STC_EXXDIV:
            mydf = fft.FFTDF(cell)
            vk0 = mydf.get_jk(cp.asarray(dm0), with_j=False, exxdiv=exxdiv)[1]
            vk1 = mydf.get_jk(cp.asarray(dm0[None]), kpts=np.zeros((1,3)),
                              with_j=False, exxdiv=exxdiv)[1]
            assert abs(vk0 - vk1[0]).max() < 1e-12
            if _HAS_CPU_STC:
                ref = fft_cpu.FFTDF(cell).get_jk(dm0, with_j=False, exxdiv=exxdiv)[1]
                assert abs(vk0.get() - ref).max() < 1e-9

    @unittest.skipIf(not _HAS_CPU_STC, 'requires pyscf with sTC')
    def test_gdf(self):
        vj_ref = df.GDF(cell, kpts).get_jk(cp.asarray(dm), kpts=kpts, with_k=False)[0]
        for exxdiv in stc.STC_EXXDIV:
            mydf = df.GDF(cell, kpts)
            mydf.exxdiv = exxdiv
            vj, vk = mydf.get_jk(cp.asarray(dm), kpts=kpts, exxdiv=exxdiv)
            # J matrix is computed with the full-range Coulomb tensor
            assert abs(vj - vj_ref).max() < 1e-10
            ref = rsdf_cpu.RSGDF(cell, kpts, exxdiv=exxdiv).get_jk(
                dm, kpts=kpts, with_j=False, exxdiv=exxdiv)[1]
            assert abs(vk.get() - ref).max() < 1e-6
            # Within the DF error of the FFTDF result
            vk_fft = fft.FFTDF(cell, kpts).get_jk(
                cp.asarray(dm), kpts=kpts, with_j=False, exxdiv=exxdiv)[1]
            assert abs(vk - vk_fft).max() < 1e-3

    def test_gdf_j_before_k(self):
        # The full-range tensor built by get_j is kept for the Coulomb matrix
        exxdiv = 'smooth_vcut_ws'
        mydf = df.GDF(cell, kpts)
        vj0 = mydf.get_jk(cp.asarray(dm), kpts=kpts, with_k=False)[0]
        cderi = mydf._cderi
        vj, vk = mydf.get_jk(cp.asarray(dm), kpts=kpts, exxdiv=exxdiv)
        assert mydf._stc.exxdiv == exxdiv
        assert mydf._with_df_j._cderi is cderi
        assert abs(vj - vj0).max() < 1e-12
        self.assertRaises(RuntimeError, mydf.get_jk, cp.asarray(dm), kpts=kpts,
                          exxdiv='ewald')

    def test_gdf_omega(self):
        exxdiv = 'smooth_vcut_ws'
        mydf = df.GDF(cell, kpts)
        mydf.exxdiv = exxdiv
        vk = mydf.get_jk(cp.asarray(dm), kpts=kpts, with_j=False, omega=.3, exxdiv=exxdiv)[1]
        ws_lr = mydf._rsh_df['aft%.6f%s' % (.3, exxdiv)]._ws_lr_exx
        mydf.get_jk(cp.asarray(dm), kpts=kpts, with_j=False, omega=.3, exxdiv=exxdiv)
        assert mydf._rsh_df['aft%.6f%s' % (.3, exxdiv)]._ws_lr_exx is ws_lr
        vk_fft = fft.FFTDF(cell, kpts).get_jk(
            cp.asarray(dm), kpts=kpts, with_j=False, omega=.3, exxdiv=exxdiv)[1]
        assert abs(vk - vk_fft).max() < 1e-7
        self.assertRaises(NotImplementedError, mydf.get_jk, cp.asarray(dm),
                          kpts=kpts, omega=-.3, exxdiv=exxdiv)

    @unittest.skipIf(not _HAS_CPU_STC, 'requires pyscf with sTC')
    def test_krhf(self):
        for exxdiv in stc.STC_EXXDIV:
            mf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
            mf.conv_tol = 1e-10
            e_gpu = mf.kernel()
            mf = scf_cpu.KRHF(cell, kpts, exxdiv=exxdiv)
            mf.conv_tol = 1e-10
            e_cpu = mf.kernel()
            assert abs(e_gpu - e_cpu) < 1e-8

if __name__ == '__main__':
    print("Full Tests for truncated Coulomb exchange")
    unittest.main()
