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
import pyscf
from pyscf.scf import addons as cpu_addons
from pyscf.scf import hf as cpu_hf
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf.scf import gcscf


SIGMA = .1


def setUpModule():
    global mol
    atom = """
    O       0.0000000000    -0.0000000000     0.1174000000
    H      -0.7570000000    -0.0000000000    -0.4696000000
    H       0.7570000000     0.0000000000    -0.4696000000
    """
    mol = pyscf.M(atom=atom, basis='6-31g', verbose=7, output='/dev/null')


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


def _run_gcscf(mf, **kwargs):
    mf = gcscf.gcscf(mf, sigma=SIGMA, **kwargs)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    return mf.run()


def _run_reference(mf, **kwargs):
    mf = cpu_addons.smearing(mf, sigma=SIGMA, method='fermi', **kwargs)
    mf.conv_tol = 1e-10
    return mf.run()


class KnownValues(unittest.TestCase):
    def test_rhf_matches_cpu_smearing(self):
        gpu_mf = _run_gcscf(gpu_scf.RHF(mol))
        cpu_mf = _run_reference(cpu_hf.RHF(mol))

        self.assertTrue(gpu_mf.converged)
        self.assertTrue(cpu_mf.converged)
        self.assertAlmostEqual(gpu_mf.e_tot, cpu_mf.e_tot, 8)
        self.assertAlmostEqual(gpu_mf.e_free, cpu_mf.e_free, 8)
        self.assertAlmostEqual(gpu_mf.entropy, cpu_mf.entropy, 8)
        self.assertAlmostEqual(gpu_mf.nelectron, mol.nelectron, 8)
        self.assertGreater(gpu_mf.n_haux_eval, 0)
        self.assertLess(gpu_mf.auxh_residual_norm, 1e-7)

    def test_rhf_fixed_mu0_matches_cpu_smearing(self):
        mu0 = -.3
        gpu_mf = _run_gcscf(gpu_scf.RHF(mol), mu0=mu0)
        cpu_mf = _run_reference(cpu_hf.RHF(mol), mu0=mu0)
        nelectron = np.sum(cpu_mf.mo_occ)
        e_grand = cpu_mf.e_free - mu0 * nelectron

        self.assertTrue(gpu_mf.converged)
        self.assertTrue(cpu_mf.converged)
        self.assertAlmostEqual(gpu_mf.e_tot, cpu_mf.e_tot, 8)
        self.assertAlmostEqual(gpu_mf.e_free, cpu_mf.e_free, 8)
        self.assertAlmostEqual(gpu_mf.e_grand, e_grand, 8)
        self.assertAlmostEqual(gpu_mf.nelectron, nelectron, 8)
        self.assertGreater(gpu_mf.n_haux_eval, 0)
        self.assertLess(gpu_mf.auxh_residual_norm, 1e-7)

    def test_uhf_fix_spin_matches_cpu_smearing(self):
        gpu_mf = _run_gcscf(gpu_scf.UHF(mol), fix_spin=True)
        cpu_mf = _run_reference(cpu_hf.UHF(mol), fix_spin=True)

        self.assertTrue(gpu_mf.converged)
        self.assertTrue(cpu_mf.converged)
        self.assertAlmostEqual(gpu_mf.e_tot, cpu_mf.e_tot, 8)
        self.assertAlmostEqual(gpu_mf.e_free, cpu_mf.e_free, 8)
        self.assertAlmostEqual(gpu_mf.entropy, cpu_mf.entropy, 8)
        self.assertAlmostEqual(gpu_mf.nelectron, np.sum(cpu_mf.mo_occ), 8)
        self.assertEqual(np.asarray(gpu_mf.mu).shape, (2,))
        self.assertGreater(gpu_mf.n_haux_eval, 0)
        self.assertLess(gpu_mf.auxh_residual_norm, 1e-7)

    def test_method_hook_and_to_cpu(self):
        gpu_mf = gpu_scf.RHF(mol).gcscf(sigma=SIGMA)
        self.assertIsInstance(gpu_mf, gcscf._GCSCF)

        cpu_mf = gpu_mf.to_cpu()
        from pyscf.scf import gcscf as cpu_gcscf
        self.assertIsInstance(cpu_mf, cpu_gcscf._GCSCF)
        self.assertEqual(cpu_mf.sigma, SIGMA)

        cpu_mf = cpu_gcscf.gcscf(cpu_hf.RHF(mol), sigma=SIGMA)
        gpu_mf = cpu_mf.to_gpu()
        self.assertIsInstance(gpu_mf, gcscf._GCSCF)
        self.assertEqual(gpu_mf.sigma, SIGMA)


if __name__ == "__main__":
    print("Basic Tests for GPU GC-SCF")
    unittest.main()
