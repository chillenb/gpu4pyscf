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

'''LPBE solve on the half spectrum (fft='r2c', the 'auto' default) vs the complex solve
(fft='c2c'), and the Nyquist convention of the first derivatives.'''

import unittest
import numpy as np
import cupy as cp
import cupyx.scipy.fft as fft
from pyscf.pbc import gto

import gpu4pyscf.pbc.solvent.lpbe_as_numint as L
from gpu4pyscf.pbc.solvent.lpbe_as_numint import (
    lpbe_inner, LPBEMultiGridNumInt, multigrid_lpbe, _get_Gv_bases, _hermitian_dot,
    _first_derivative_bases)
from gpu4pyscf.pbc.dft.multigrid_v3 import _apply_Gv_1j


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


def _solve_both(mesh, **opts):
    c = cell.copy()
    c.mesh = list(mesh)
    c.build()
    mf = c.KRKS(xc='pbe', kpts=c.make_kpts([1, 1, 1])).to_gpu()
    dm = mf.get_init_guess(key='atom')
    ni = LPBEMultiGridNumInt(c, **dict(dict(tol=1e-12, ionic_strength=1.0), **opts))
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

    def test_first_derivative_is_real(self):
        # i G' phi of a real field is the spectrum of a real field on an even mesh too;
        # with the Nyquist factor kept it is not.  Odd meshes are untouched.
        mesh = (6, 8, 10)
        b = cell.reciprocal_vectors()
        full = _get_Gv_bases(mesh, b)
        deriv = _first_derivative_bases(full)
        phiG = fft.fftn(cp.random.random(mesh))
        for n in range(3):
            g = fft.ifftn(_apply_Gv_1j(phiG, deriv[0][n], deriv[1][n], deriv[2][n]))
            self.assertLess(float(abs(g.imag).max() / abs(g.real).max()), 1e-13)
        g = fft.ifftn(_apply_Gv_1j(phiG, full[0][0], full[1][0], full[2][0]))
        self.assertGreater(float(abs(g.imag).max() / abs(g.real).max()), 1e-3)
        odd = _get_Gv_bases((5, 7, 9), b)
        for a, d in zip(odd, _first_derivative_bases(odd)):
            self.assertTrue(bool(cp.array_equal(a, d)))

    def _check_same(self, mesh, places=12, **opts):
        out = _solve_both(mesh, **opts)
        e = {m: float(cp.real(o['E_coul_corr'] + o['Ecav']).get()) for m, o in out.items()}
        self.assertAlmostEqual(e['r2c'] / e['c2c'], 1.0, places)
        vc = _real_part_G(out['c2c']['vcorr_g'], mesh)
        dv = float(abs(_real_part_G(out['r2c']['vcorr_g'], mesh) - vc).max() / abs(vc).max())
        self.assertLess(dv, 1e-10)
        self.assertEqual(out['r2c']['pot_guess'].size, mesh[0] * mesh[1] * (mesh[2] // 2 + 1))

    def test_odd_mesh_same_as_c2c(self):
        self._check_same((25, 25, 75))

    def test_even_mesh_same_as_c2c(self):
        # with the Nyquist-zeroed first derivatives both solves are the same real operator
        self._check_same((24, 24, 76))
        self._check_same((24, 25, 75))          # one even axis

    def test_even_mesh_no_electrolyte(self):
        # the Nyquist corner modes are null modes of -div(eps grad) without electrolyte
        # both CG solves stop at tol 1e-12 (the C2C one also iterates on the ~1e-10 imaginary
        # round-off of the density), so they agree to ~1e-12 here
        self._check_same((24, 24, 76), places=11, has_electrolyte=False)

    def test_vacuum_limit(self):
        # eps = 1, no electrolyte, no cavitation: the solvation correction must vanish at
        # every G != 0 (phi and the vacuum reference use the same discrete operator)
        for mesh in ((25, 25, 75), (24, 24, 76)):
            solvated = _solve_both(mesh, cav_tension=0.0)['c2c']['vcorr_g']
            scale = float(abs(_real_part_G(solvated, mesh)).max())
            out = _solve_both(mesh, rel_permittivity=1.0, has_electrolyte=False,
                              cav_tension=0.0)
            for mode, o in out.items():
                v = _real_part_G(o['vcorr_g'], mesh).reshape(-1)
                self.assertLess(float(abs(v[1:]).max()), 1e-9 * scale,
                                msg='%s %s: %.2e vs %.2e' % (mode, mesh, float(abs(v[1:]).max()), scale))

    def test_auto_mode(self):
        # 'auto' (default) is the half-spectrum solve on odd and even meshes
        for mesh in ((25, 25, 75), (24, 24, 76)):
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
            self.assertEqual(out['pot_guess'].size, mesh[0] * mesh[1] * (mesh[2] // 2 + 1))

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
