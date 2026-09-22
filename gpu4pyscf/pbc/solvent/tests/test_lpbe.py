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

import unittest
import numpy as np
import cupy as cp
import cupyx.scipy.fft as fft

import pyscf

from pyscf.pbc import gto

from gpu4pyscf.pbc.tools import pbc as pbc_tools

from gpu4pyscf.pbc.solvent.lpbe_as_numint import lpbe_inner as lpbe_inner_v3
from gpu4pyscf.pbc.solvent.lpbe_as_numint import (vasp_dens_to_pyscf_dens, vasp_tau_to_pyscf_tau, debye_length_au, molar_to_au, _get_Gv_bases)

from gpu4pyscf.pbc.solvent.lpbe_as_numint import LPBEMultiGridNumInt as LPBEMultiGridNumInt_v3

from gpu4pyscf.pbc.solvent.lpbe_as_numint_v2 import lpbe_inner as lpbe_inner_v2
from gpu4pyscf.pbc.solvent.lpbe_as_numint_v2 import LPBEMultiGridNumInt as LPBEMultiGridNumInt_v2




#import pytest

def setUpModule():
    global cell_orth, cell_he
    global kpts, dm, dm1
    global lpbe_options
    np.random.seed(2)
    cell_orth = gto.M(
        verbose = 7,
        output = '/dev/null',
        a = np.diag([3.6, 3.2, 10.5]),
        atom = '''C     0.      0.      0.
                  C     1.8     1.8     1.8   ''',
        basis = 'gth-dzvp',
        pseudo = 'gth-pbe',
        precision = 1e-10,
        unit = 'Bohr',
    )

    kpts = np.random.random((2,3)).round(1)
    kpts[1] = -kpts[0]
    kpts = cell_orth.get_abs_kpts(kpts)
    nao = cell_orth.nao_nr()
    dm = np.random.random((len(kpts),nao,nao)) * .2
    dm1 = dm + np.eye(nao)
    dm = dm1 + dm1.transpose(0,2,1)

    cell_he = pyscf.M(atom='He 0 0 0',
                      basis='gth-dzvp',
                      pseudo='gth-pbe',
                      unit='B',
                      precision = 1e-10,
                      a=np.eye(3)*10)


    lpbe_options = {
        'tol': 1e-8,
        'cav_smear': 0.6,
        'eps': 1e-10,
        'rel_permittivity': 78.4,
        'cav_dens_cutoff': vasp_dens_to_pyscf_dens(0.0025),
        'cav_tension': vasp_tau_to_pyscf_tau(5.25e-4),
        'has_electrolyte': True,
        'temperature': 298.15,
        'ionic_strength': 1.0,
        'debye_length': debye_length_au(molar_to_au(1.0), 298.15, eps_r=78.4),
    }


def tearDownModule():
    global cell_orth, cell_nonorth, cell_he, lpbe_options
    cell_orth.stdout.close()
    del cell_orth, cell_he
    del lpbe_options


class KnownValues(unittest.TestCase):


    def test_lpbe_gga(self):
        pcell = cell_orth.copy().to_gpu()
        pcell.precision = 1e-10
        pcell = pcell.build()

        xc = 'pbe,'
        kmesh = [1, 1, 1]
        mesh = pcell.mesh
        kpts = pcell.make_kpts(kmesh)

        weight = pcell.vol / np.prod(mesh)

        mf = pcell.KRKS(xc=xc, kpts=kpts)

        dm = mf.get_init_guess(key='minao')

        Gv_bases = _get_Gv_bases(mesh, pcell.reciprocal_vectors())


        numint_v3 = LPBEMultiGridNumInt_v3(pcell, mesh=mesh, lpbe_options=lpbe_options)
        rhoR = numint_v3.get_rho(dm, kpts=kpts)
        rhoG = fft.fftn(rhoR.reshape(*mesh), axes=(0,1,2)) * weight

        numint_v2 = LPBEMultiGridNumInt_v2(pcell, mesh=mesh, lpbe_options=lpbe_options)
        Gv = pbc_tools.get_Gv(pcell, mesh)
        coulomb_kernel_on_g_mesh = pbc_tools.get_coulG(pcell, Gv=Gv)


        v3_result = lpbe_inner_v3(numint_v3, rhoG.copy().reshape(-1), Gv_bases, options=lpbe_options)
        v2_result = lpbe_inner_v2(numint_v2, rhoG.copy().reshape(-1), coulomb_kernel_on_g_mesh, Gv)

        for qname in ('Eion', 'Ediel', 'Ecav', 'E_coul_corr', 'cavity_r', 'solvation_potentialR', 'vcav_r', 'vdiel_r', 'vion_r', 'vcorr_g', 'pseudocore_densityR'):
            self.assertAlmostEqual((v3_result[qname].get() - v2_result[qname].get()).max(), 0.0, 6, msg=f"Failed for {qname} in test_lpbe_gga")


    def test_lpbe_gga_kpts(self):
        pcell = cell_orth.copy().to_gpu()
        pcell.precision = 1e-10
        pcell = pcell.build()

        xc = 'pbe,'
        kmesh = [3, 2, 1]
        mesh = pcell.mesh
        kpts = pcell.make_kpts(kmesh)

        weight = pcell.vol / np.prod(mesh)

        mf = pcell.KRKS(xc=xc, kpts=kpts)

        dm = mf.get_init_guess(key='minao')

        Gv_bases = _get_Gv_bases(mesh, pcell.reciprocal_vectors())


        numint_v3 = LPBEMultiGridNumInt_v3(pcell, mesh=mesh, lpbe_options=lpbe_options)
        rhoR = numint_v3.get_rho(dm, kpts=kpts)
        rhoG = fft.fftn(rhoR.reshape(*mesh), axes=(0,1,2)) * weight

        numint_v2 = LPBEMultiGridNumInt_v2(pcell, mesh=mesh, lpbe_options=lpbe_options)
        Gv = pbc_tools.get_Gv(pcell, mesh)
        coulomb_kernel_on_g_mesh = pbc_tools.get_coulG(pcell, Gv=Gv)


        v3_result = lpbe_inner_v3(numint_v3, rhoG.copy().reshape(-1), Gv_bases, options=lpbe_options)
        v2_result = lpbe_inner_v2(numint_v2, rhoG.copy().reshape(-1), coulomb_kernel_on_g_mesh, Gv)

        for qname in ('Eion', 'Ediel', 'Ecav', 'E_coul_corr', 'cavity_r', 'solvation_potentialR', 'vcav_r', 'vdiel_r', 'vion_r', 'vcorr_g', 'pseudocore_densityR'):
            self.assertAlmostEqual((v3_result[qname].get() - v2_result[qname].get()).max(), 0.0, 6, msg=f"Failed for {qname} in test_lpbe_gga")


if __name__ == '__main__':
    print("Full Tests for LPBE")
    unittest.main()
