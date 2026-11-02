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

'''Orbital basis with h functions (l=5) in the PBC integrals, numint and
multigrid. Synthetic small cells, compared against pyscf on CPU.'''

import unittest
import numpy as np
import cupy as cp
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.df import df as df_cpu, ft_ao as ft_ao_cpu
from pyscf.pbc.dft import numint as numint_cpu
from pyscf.pbc.dft import gen_grid as gen_grid_cpu
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.df.df import GDF
from gpu4pyscf.pbc.dft import numint, gen_grid
from gpu4pyscf.pbc.dft import multigrid_v3 as multigrid

# (h,h) and (h,g) pairs need the wide ft_aopair kernel instance
BASIS = [[0, (1.2, .6), (.5, .5)],
         [1, (.8, 1)],
         [2, (.6, 1)],
         [4, (.9, 1)],
         [5, (1.1, 1)],
         [5, (.45, 1)]]

def make_cell(cart=False, a=4.):
    return pbcgto.M(atom='He .2 .3 .4; He 1.9 1.6 2.3', basis=BASIS,
                    a=np.eye(3)*a, unit='B', cart=cart, precision=1e-9,
                    mesh=[27]*3, verbose=0)

def setUpModule():
    global cell, cell_cart, kpts
    cell = make_cell()
    cell_cart = make_cell(cart=True)
    kpts = cell.make_kpts([2,1,1])

def tearDownModule():
    global cell, cell_cart, kpts
    del cell, cell_cart, kpts

def random_dm(cell, nkpts=None, seed=1):
    rng = np.random.default_rng(seed)
    nao = cell.nao
    c = rng.random((nao, nao)) - .5
    dm = c.dot(c.T) / nao
    if nkpts is not None:
        dm = np.array([dm] * nkpts)
    return dm

class KnownValues(unittest.TestCase):
    def test_lmax(self):
        self.assertEqual(cell._bas[:,1].max(), 5)

    def test_int1e(self):
        for c in (cell, cell_cart):
            ref = c.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
            dat = int1e.int1e_ovlp(c, kpts).get()
            self.assertLess(abs(dat - ref).max(), 1e-9)
            ref = c.pbc_intor('int1e_kin', hermi=1, kpts=kpts)
            dat = int1e.int1e_kin(c, kpts).get()
            self.assertLess(abs(dat - ref).max(), 1e-9)

    def test_eval_ao(self):
        for c in (cell, cell_cart):
            # Tight precision for the reference to avoid differences in the
            # lattice-sum screening
            c_ref = c.copy()
            c_ref.precision = 1e-16
            coords = gen_grid.UniformGrids(c).build().coords
            ni = numint.KNumInt()
            for deriv in (0, 1, 2):
                ao = ni.eval_ao(c, coords, kpts, deriv=deriv)
                ref = ni.to_cpu().eval_ao(c_ref, coords.get(), kpts, deriv=deriv)
                for k in range(len(kpts)):
                    self.assertLess(abs(ao[k].get() - ref[k]).max() * c.vol, 1e-8,
                                    f'cart={c.cart} deriv={deriv}')

    def test_ft_aopair(self):
        Gv = cell.get_Gv([7,5,5])
        dat = ft_ao.ft_aopair(cell, Gv).get()
        ref = ft_ao_cpu.ft_aopair(cell, Gv)
        self.assertLess(abs(dat - ref).max(), 1e-9)

        kpti = kptj = cell.get_abs_kpts([.3, .1, .2])
        dat = ft_ao.ft_aopair(cell, Gv, kpti_kptj=(kpti, kptj)).get()
        ref = ft_ao_cpu.ft_aopair(cell, Gv, kpti_kptj=(kpti, kptj))
        self.assertLess(abs(dat - ref).max(), 1e-9)

        dat = ft_ao.ft_aopair(cell_cart, Gv).get()
        ref = ft_ao_cpu.ft_aopair(cell_cart, Gv)
        self.assertLess(abs(dat - ref).max(), 1e-9)

    def test_gdf_jk_gamma_point(self):
        dm = random_dm(cell)
        jref, kref = df_cpu.GDF(cell).get_jk(dm, exxdiv='ewald')
        mydf = GDF(cell)
        mydf.is_gamma_point = True
        vj, vk = mydf.get_jk(cp.asarray(dm), exxdiv='ewald')
        self.assertLess(abs(vj.get() - jref).max(), 1e-7)
        self.assertLess(abs(vk.get() - kref).max(), 1e-7)

    def test_gdf_jk_kpts(self):
        dm = random_dm(cell, len(kpts))
        jref, kref = df_cpu.GDF(cell, kpts).get_jk(dm, kpts=kpts, exxdiv='ewald')
        vj, vk = GDF(cell, kpts).get_jk(cp.asarray(dm), kpts=kpts, exxdiv='ewald')
        self.assertLess(abs(vj.get() - jref).max(), 1e-7)
        self.assertLess(abs(vk.get() - kref).max(), 1e-7)

    def test_numint_nr_rks(self):
        dm = random_dm(cell, len(kpts))
        grids = gen_grid.UniformGrids(cell)
        ni = numint.KNumInt()
        for xc in ('lda,', 'pbe,', 'm06l,'):
            n1, e1, v1 = ni.nr_rks(cell, grids, xc, cp.asarray(dm), hermi=1, kpts=kpts)
            n0, e0, v0 = ni.to_cpu().nr_rks(cell, grids.to_cpu(), xc, dm, hermi=1, kpts=kpts)
            self.assertAlmostEqual(float(n1), n0, 8)
            self.assertAlmostEqual(float(e1), e0, 8)
            self.assertLess(abs(v1.get() - v0).max(), 1e-8, xc)

    def _check_multigrid(self, c, enable_aft):
        dm = random_dm(c)
        grids_cpu = gen_grid_cpu.UniformGrids(c)
        for xc in ('lda,', 'pbe,', 'm06l,'):
            # Uniform-grid numint on CPU as the reference
            n0, e0, v0 = numint_cpu.NumInt().nr_rks(c, grids_cpu, xc, dm)
            ni = multigrid.MultiGridNumInt(c)
            ni.enable_aft = enable_aft
            n1, e1, v1 = ni.nr_rks(c, None, xc, cp.asarray(dm), with_j=False)
            self.assertAlmostEqual(float(n1), n0, 7)
            self.assertAlmostEqual(float(e1), e0, 7)
            self.assertLess(abs(v1.get() - v0).max(), 1e-7, xc)
            if ni.aft_buckets:
                ang = ni.bvkcell._bas[:,1]
                for bucket in ni.aft_buckets:
                    ish, jsh = divmod(bucket['bas_ij_idx'].get(), multigrid.NBAS_MAX)
                    self.assertTrue((ang[ish] <= multigrid.AFT_LMAX).all())
                    self.assertTrue((ang[jsh] <= multigrid.AFT_LMAX).all())
        return ni

    def test_multigrid_fft(self):
        self._check_multigrid(cell, enable_aft=False)

    def test_multigrid_aft(self):
        # A larger cubic cell has more lattice images, enabling the AFT path for
        # the shell pairs without h functions
        c = make_cell(a=6.)
        ni = self._check_multigrid(c, enable_aft=True)
        self.assertTrue(ni.fft_buckets)

if __name__ == '__main__':
    print('Full Tests for h functions in PBC')
    unittest.main()
