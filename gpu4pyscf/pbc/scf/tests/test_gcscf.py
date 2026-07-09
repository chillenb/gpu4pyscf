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

import unittest

import numpy as np
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.scf as cpu_pscf
from pyscf.pbc.scf import gcscf as cpu_gcscf
from gpu4pyscf.pbc import scf as gpu_pscf
from gpu4pyscf.pbc.scf import gcscf as gpu_gcscf


SIGMA = .1


def setUpModule():
    global cell, kpts
    cell = pbcgto.Cell()
    cell.atom = '''
    He 0 0 1
    He 1 0 1
    '''
    cell.basis = [[0, [1., 1.]], [0, [.5, 1.]]]
    cell.a = np.eye(3) * 3
    cell.mesh = [10] * 3
    cell.verbose = 7
    cell.output = '/dev/null'
    cell.build()
    kpts = cell.make_kpts([2, 1, 1])


def tearDownModule():
    global cell, kpts
    cell.stdout.close()
    del cell, kpts


def _set_common_flags(mf):
    mf.verbose = 0
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    return mf


def _run_cpu_gcscf(mf, **kwargs):
    mf = cpu_gcscf.gcscf(mf, sigma=SIGMA, **kwargs)
    return _set_common_flags(mf).run()


def _run_gpu_gcscf(mf, **kwargs):
    mf = gpu_gcscf.gcscf(mf, sigma=SIGMA, **kwargs)
    return _set_common_flags(mf).run()


class KnownValues(unittest.TestCase):
    def test_krhf_matches_cpu_gcscf(self):
        gpu_mf = _run_gpu_gcscf(gpu_pscf.KRHF(cell, kpts=kpts))
        cpu_mf = _run_cpu_gcscf(cpu_pscf.KRHF(cell, kpts=kpts))

        self.assertTrue(gpu_mf.converged)
        self.assertTrue(cpu_mf.converged)
        self.assertAlmostEqual(gpu_mf.e_tot, cpu_mf.e_tot, 8)
        self.assertAlmostEqual(gpu_mf.e_free, cpu_mf.e_free, 8)
        self.assertAlmostEqual(gpu_mf.entropy, cpu_mf.entropy, 7)
        self.assertAlmostEqual(gpu_mf.nelectron, cpu_mf.nelectron, 8)
        self.assertGreater(gpu_mf.n_haux_eval, 1)
        self.assertLess(gpu_mf.auxh_residual_norm, 1e-7)

    def test_krhf_fixed_mu0_matches_cpu_gcscf(self):
        mu0 = .3
        gpu_mf = _run_gpu_gcscf(gpu_pscf.KRHF(cell, kpts=kpts), mu0=mu0)
        cpu_mf = _run_cpu_gcscf(cpu_pscf.KRHF(cell, kpts=kpts), mu0=mu0)

        self.assertTrue(gpu_mf.converged)
        self.assertTrue(cpu_mf.converged)
        self.assertAlmostEqual(gpu_mf.e_tot, cpu_mf.e_tot, 8)
        self.assertAlmostEqual(gpu_mf.e_free, cpu_mf.e_free, 8)
        self.assertAlmostEqual(gpu_mf.e_grand, cpu_mf.e_grand, 8)
        self.assertAlmostEqual(gpu_mf.entropy, cpu_mf.entropy, 7)
        self.assertAlmostEqual(gpu_mf.nelectron, cpu_mf.nelectron, 8)
        self.assertGreater(gpu_mf.n_haux_eval, 1)
        self.assertLess(gpu_mf.auxh_residual_norm, 1e-7)

    def test_kuhf_fix_spin_matches_cpu_gcscf(self):
        gpu_mf = _run_gpu_gcscf(gpu_pscf.KUHF(cell, kpts=kpts),
                                fix_spin=True)
        cpu_mf = _run_cpu_gcscf(cpu_pscf.KUHF(cell, kpts=kpts),
                                fix_spin=True)

        self.assertTrue(gpu_mf.converged)
        self.assertTrue(cpu_mf.converged)
        self.assertAlmostEqual(gpu_mf.e_tot, cpu_mf.e_tot, 8)
        self.assertAlmostEqual(gpu_mf.e_free, cpu_mf.e_free, 8)
        self.assertAlmostEqual(gpu_mf.entropy, cpu_mf.entropy, 7)
        self.assertAlmostEqual(gpu_mf.nelectron, cpu_mf.nelectron, 8)
        self.assertEqual(np.asarray(gpu_mf.mu).shape, (2,))
        self.assertGreater(gpu_mf.n_haux_eval, 1)
        self.assertLess(gpu_mf.auxh_residual_norm, 1e-7)

    def test_to_cpu_and_to_gpu(self):
        gpu_mf = gpu_gcscf.gcscf(gpu_pscf.KRHF(cell, kpts=kpts), sigma=SIGMA)
        self.assertIsInstance(gpu_mf, gpu_gcscf._GCKSCF)

        cpu_mf = gpu_mf.to_cpu()
        self.assertIsInstance(cpu_mf, cpu_gcscf._GCKSCF)
        self.assertEqual(cpu_mf.sigma, SIGMA)

        cpu_mf = cpu_gcscf.gcscf(cpu_pscf.KRHF(cell, kpts=kpts), sigma=SIGMA)
        gpu_mf = gpu_gcscf.from_cpu(cpu_mf)
        self.assertIsInstance(gpu_mf, gpu_gcscf._GCKSCF)
        self.assertEqual(gpu_mf.sigma, SIGMA)


if __name__ == "__main__":
    print("Basic Tests for GPU PBC GC-SCF")
    unittest.main()
