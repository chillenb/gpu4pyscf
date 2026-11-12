#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

'''LPBE solve on the half spectrum (fft='r2c'; the 'auto' default on odd meshes) vs the
complex solve (fft='c2c'; 'auto' on even meshes).'''

import unittest
import numpy as np
import cupy as cp
import cupyx.scipy.fft as fft
from pyscf.pbc import gto

import gpu4pyscf.pbc.solvent.lpbe_as_numint as L
from gpu4pyscf.pbc.solvent.lpbe_as_numint import (
    lpbe_inner, LPBEMultiGridNumInt, multigrid_lpbe, _get_Gv_bases, _hermitian_dot)


def setUpModule():
    global cell
    cell = gto.M(
        verbose=0,
        a=np.diag([3.6, 3.2, 10.5]),
        atom='''C 0. 0. 0.; C 1.8 1.8 1.8''',
        basis='gth-dzvp', pseudo='gth-pbe', precision=1e-10, unit='Bohr')


def tearDownModule():
    global cell
    del cell


def _solve_both(mesh):
    c = cell.copy()
    c.mesh = list(mesh)
    c.build()
    mf = c.KRKS(xc='pbe', kpts=c.make_kpts([1, 1, 1])).to_gpu()
    dm = mf.get_init_guess(key='atom')
    ni = LPBEMultiGridNumInt(c, tol=1e-12, ionic_strength=1.0)
    ni.build()
    weight = c.vol / np.prod(mesh)
    rhoR = ni.get_rho(dm, kpts=mf.kpts)
    rhoG = (fft.fftn(rhoR.reshape(*mesh)) * weight).reshape(-1)
    Gv_bases = _get_Gv_bases(mesh, c.reciprocal_vectors())
    out = {}
    for mode in ('c2c', 'r2c'):
        opts = dict(ni.options, fft=mode)
        out[mode] = lpbe_inner(ni, rhoG.copy(), Gv_bases, options=opts)
    return out


def _real_part_G(vg, mesh):
    '''G-space field of the real part of the real-space field (its Hermitian part).'''
    return fft.fftn(fft.ifftn(vg.reshape(mesh)).real)


class KnownValues(unittest.TestCase):
    def test_hermitian_dot(self):
        # half-spectrum weighted product == full complex product, odd and even nz
        for mesh in ((5, 6, 7), (5, 6, 8)):
            a = cp.random.random(mesh)
            b = cp.random.random(mesh)
            ref = float(cp.vdot(fft.fftn(a), fft.fftn(b)).real)
            ah = fft.rfftn(a)
            bh = fft.rfftn(b)
            self.assertAlmostEqual(float(_hermitian_dot(ah, bh, mesh)) / ref, 1.0, 12)

    def test_odd_mesh_same_as_c2c(self):
        mesh = (25, 25, 75)
        out = _solve_both(mesh)
        e = {m: float(cp.real(o['E_coul_corr'] + o['Ecav']).get()) for m, o in out.items()}
        self.assertAlmostEqual(e['r2c'] / e['c2c'], 1.0, 13)
        vc = _real_part_G(out['c2c']['vcorr_g'], mesh)
        dv = float(abs(_real_part_G(out['r2c']['vcorr_g'], mesh) - vc).max() / abs(vc).max())
        self.assertLess(dv, 1e-11)
        # the warm-start guess is the half spectrum
        self.assertEqual(out['r2c']['pot_guess'].size, 25 * 25 * (75 // 2 + 1))

    def test_auto_mode(self):
        # 'auto' (default): R2C on odd meshes, the unchanged C2C solve on even meshes
        for mesh, nguess in (((25, 25, 75), 25 * 25 * 38), ((24, 24, 76), 24 * 24 * 76)):
            c = cell.copy()
            c.mesh = list(mesh)
            c.build()
            mf = c.KRKS(xc='pbe', kpts=c.make_kpts([1, 1, 1])).to_gpu()
            ni = LPBEMultiGridNumInt(c, tol=1e-10, ionic_strength=1.0)
            ni.build()
            rhoR = ni.get_rho(mf.get_init_guess(key='atom'), kpts=mf.kpts)
            rhoG = (fft.fftn(rhoR.reshape(*mesh)) * (c.vol / np.prod(mesh))).reshape(-1)
            out = lpbe_inner(ni, rhoG, _get_Gv_bases(mesh, c.reciprocal_vectors()),
                             options=dict(ni.options))
            self.assertEqual(out['pot_guess'].size, nguess)

    def test_scf_same_as_c2c(self):
        c = cell.copy()
        c.mesh = [25, 25, 75]
        c.build()
        kpts = c.make_kpts([2, 1, 1])
        e = {}
        for mode in ('c2c', 'r2c'):
            mf = multigrid_lpbe(c.KRKS(xc='pbe', kpts=kpts).to_gpu(), tol=1e-12,
                                ionic_strength=1.0, fft=mode)
            mf.conv_tol = 1e-10
            e[mode] = mf.kernel()
            self.assertTrue(mf.converged)
        self.assertAlmostEqual(e['r2c'], e['c2c'], 9)

    def test_warm_start_from_c2c_guess(self):
        mesh = (25, 25, 75)
        c = cell.copy()
        c.mesh = list(mesh)
        c.build()
        mf = c.KRKS(xc='pbe', kpts=c.make_kpts([1, 1, 1])).to_gpu()
        ni = LPBEMultiGridNumInt(c, tol=1e-12, ionic_strength=1.0)
        ni.build()
        rhoR = ni.get_rho(mf.get_init_guess(key='atom'), kpts=mf.kpts)
        rhoG = (fft.fftn(rhoR.reshape(*mesh)) * (c.vol / np.prod(mesh))).reshape(-1)
        Gv_bases = _get_Gv_bases(mesh, c.reciprocal_vectors())
        full = lpbe_inner(ni, rhoG.copy(), Gv_bases, options=dict(ni.options, fft='c2c'))
        # a converged full-spectrum guess must be accepted and need (almost) no iterations
        n = [0]
        orig = L.cg_opt

        def counted(*a, callback=None, **kw):
            def cb(x):
                n[0] += 1
            return orig(*a, callback=cb, **kw)
        L.cg_opt = counted
        try:
            lpbe_inner(ni, rhoG.copy(), Gv_bases, options=dict(ni.options, fft='r2c'),
                       pot_guess=full['pot_guess'].copy())
        finally:
            L.cg_opt = orig
        self.assertLessEqual(n[0], 3)


if __name__ == '__main__':
    print('Full Tests for LPBE R2C solve')
    unittest.main()
