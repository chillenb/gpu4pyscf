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

'''Auxiliary basis functions up to l = 7 (k) in PBC GDF, e.g. the cc-pVQZ-fit sets for
h-containing orbital bases. Synthetic small cell with h orbital shells, compared against pyscf on
CPU. The (h h|k) triplets need the wide fill_int3c2e kernel instance.'''

import unittest
import numpy as np
import cupy as cp
import pyscf
from pyscf.pbc.df import df as df_cpu, ft_ao as ft_ao_cpu, rsdf_builder
from pyscf.pbc.df.df import make_auxcell
from gpu4pyscf.pbc.df import ft_ao, int3c2e
from gpu4pyscf.pbc.df.int2c2e import sr_int2c2e
from gpu4pyscf.pbc.df.int3c2e import sr_aux_e2
from gpu4pyscf.pbc.df.df import GDF

BASIS = [[0, (1.2, .6), (.5, .5)],
         [1, (.8, 1)],
         [2, (.6, 1)],
         [5, (1.1, 1)]]
AUXBASIS = [[0, (2.4, 1)],
            [1, (1.6, 1)],
            [3, (1.4, 1)],
            [6, (2.2, 1)],
            [7, (2.6, 1)], [7, (1.3, 1)]]

def setUpModule():
    global cell, auxcell
    cell = pyscf.M(atom='He .2 .3 .4; He 1.9 1.6 2.3', basis=BASIS,
                   a=np.eye(3)*4., unit='B', precision=1e-10, verbose=0)
    auxcell = make_auxcell(cell, {'He': AUXBASIS})

def tearDownModule():
    global cell, auxcell
    del cell, auxcell

def random_dm(nao, nkpts=None, seed=1):
    rng = np.random.default_rng(seed)
    c = rng.random((nao, nao)) - .5
    dm = c.dot(c.T) / nao
    if nkpts is not None:
        dm = np.array([dm] * nkpts)
    return dm

class KnownValues(unittest.TestCase):
    def test_lmax(self):
        self.assertEqual(cell._bas[:,1].max(), 5)
        self.assertEqual(auxcell._bas[:,1].max(), 7)
        self.assertTrue(int3c2e._int3c2e_high_l(7, 5))
        self.assertFalse(int3c2e._int3c2e_high_l(6, 5))

    def test_ft_ao(self):
        Gv = cell.get_Gv([7,5,5])
        ref = ft_ao_cpu.ft_ao(auxcell, Gv)
        dat = ft_ao.ft_ao(auxcell, Gv).get()
        self.assertLess(abs(dat - ref).max() / abs(ref).max(), 1e-10)
        kpt = cell.get_abs_kpts([.3, .1, .2])
        ref = ft_ao_cpu.ft_ao(auxcell, Gv, kpt=kpt)
        dat = ft_ao.ft_ao(auxcell, Gv, kpt=kpt).get()
        self.assertLess(abs(dat - ref).max() / abs(ref).max(), 1e-10)

    def test_sr_int2c2e(self):
        omega = 0.2
        kmesh = [2, 1, 1]
        kpts = auxcell.make_kpts(kmesh)
        a = auxcell.copy()
        a.precision = 1e-14
        a.rcut = 50
        with a.with_short_range_coulomb(omega):
            ref = a.pbc_intor('int2c2e', hermi=1, kpts=kpts)
        dat = sr_int2c2e(auxcell, -omega, kpts=kpts, bvkmesh=kmesh).get()
        self.assertLess(abs(dat - ref).max(), 1e-10)

    def test_sr_int3c2e(self):
        omega = 0.2
        dat = sr_aux_e2(cell, auxcell, -omega).get()
        # the h shells need a tighter lattice sum for a converged CPU reference
        c = cell.copy()
        c.precision = 1e-12
        c.build()
        b = rsdf_builder._RSGDFBuilder(c, auxcell).build(omega=omega)
        ref = b.gen_int3c_kernel('int3c2e', aosym='s1', return_complex=True)().reshape(dat.shape)
        self.assertLess(abs(dat - ref).max(), 1e-8)

    def test_gdf_jk_gamma_point(self):
        dm = random_dm(cell.nao)
        mydf = df_cpu.GDF(cell)
        mydf.auxbasis = {'He': AUXBASIS}
        jref, kref = mydf.get_jk(dm, exxdiv='ewald')
        mydf = GDF(cell)
        mydf.auxbasis = {'He': AUXBASIS}
        mydf.is_gamma_point = True
        vj, vk = mydf.get_jk(cp.asarray(dm), exxdiv='ewald')
        self.assertLess(abs(vj.get() - jref).max(), 1e-7)
        self.assertLess(abs(vk.get() - kref).max(), 1e-7)

    def test_gdf_jk_kpts(self):
        kpts = cell.make_kpts([2, 1, 1])
        dm = random_dm(cell.nao, len(kpts))
        mydf = df_cpu.GDF(cell, kpts)
        mydf.auxbasis = {'He': AUXBASIS}
        jref, kref = mydf.get_jk(dm, kpts=kpts, exxdiv='ewald')
        mydf = GDF(cell, kpts)
        mydf.auxbasis = {'He': AUXBASIS}
        vj, vk = mydf.get_jk(cp.asarray(dm), kpts=kpts, exxdiv='ewald')
        self.assertLess(abs(vj.get() - jref).max(), 1e-7)
        self.assertLess(abs(vk.get() - kref).max(), 1e-7)

    def test_contract_dm_not_supported(self):
        # contract_int3c2e.cu holds at most l_aux = 6 (NF_AUX_MAX)
        opt = int3c2e.SRInt3c2eOpt(cell, auxcell, -0.2).build()
        with self.assertRaises(NotImplementedError):
            opt.contract_dm(cp.asarray(random_dm(opt.cell.nao)))

if __name__ == "__main__":
    print("Full Tests for PBC GDF with l = 7 auxiliary functions")
    unittest.main()
