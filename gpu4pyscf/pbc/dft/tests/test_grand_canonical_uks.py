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

'''GrandCanonicalKUKS and the KUKS path of the LPBE multigrid NumInt.'''

import unittest
import numpy as np
import cupy as cp
from pyscf.pbc import gto
from pyscf.pbc.dft import kuks as kuks_cpu
from pyscf.pbc.scf import addons as addons_cpu

from gpu4pyscf.pbc.dft import multigrid_v3
from gpu4pyscf.pbc.dft.grand_canonical import (
    GrandCanonicalKRKS, GrandCanonicalKUKS, GrandCanonicalKS)
from gpu4pyscf.pbc.solvent.lpbe_as_numint import multigrid_lpbe


def setUpModule():
    global slab, slab_kpts, hlat, hlat_kpts
    # Small cell with vacuum along z for the solvent.
    slab = gto.M(
        verbose=0,
        output='/dev/null',
        a=np.diag([3.6, 3.2, 14.0]),
        atom='''C     0.      0.      6.
                C     1.8     1.8     7.8''',
        basis='gth-dzvp',
        pseudo='gth-pbe',
        precision=1e-10,
        unit='Bohr',
    )
    slab_kpts = slab.make_kpts([2, 1, 1])
    # Dilute simple-cubic H lattice; ferromagnetic in LSDA.
    hlat = gto.M(
        verbose=0,
        output='/dev/null',
        a=np.eye(3)*3.2,
        atom='H 0 0 0',
        basis='gth-dzv',
        pseudo='gth-pade',
        spin=0,
    )
    hlat_kpts = hlat.make_kpts([2, 2, 2])


def tearDownModule():
    global slab, hlat
    slab.stdout.close()
    hlat.stdout.close()
    del slab, hlat


def _lpbe(mf):
    return multigrid_lpbe(mf, tol=1e-12, ionic_strength=1.0)


class KnownValues(unittest.TestCase):
    def test_nr_uks_lpbe_spin_symmetric(self):
        mf_r = _lpbe(slab.KRKS(xc='pbe', kpts=slab_kpts).to_gpu())
        mf_u = _lpbe(slab.KUKS(xc='pbe', kpts=slab_kpts).to_gpu())
        dm = cp.asarray(mf_r.get_init_guess(slab, 'minao'))
        vr = mf_r.get_veff(slab, dm)
        vu = mf_u.get_veff(slab, cp.stack([dm*.5, dm*.5]))
        self.assertEqual(vu.shape, (2,) + vr.shape)
        self.assertAlmostEqual(float(abs(vu[0]-vr).max()), 0, 7)
        self.assertAlmostEqual(float(abs(vu[1]-vr).max()), 0, 7)
        self.assertAlmostEqual(vu.exc, vr.exc, 8)
        self.assertAlmostEqual(vu.ecoul, vr.ecoul, 8)

    def test_nr_uks_lpbe_spin_polarized(self):
        '''The solvent responds to the total density only: the LPBE
        contributions to veff and the energy are those of KRKS with the
        same total density, and the XC part is that of plain multigrid.'''
        def build(kind, lpbe):
            mf = getattr(slab, kind)(xc='pbe', kpts=slab_kpts).to_gpu()
            if lpbe:
                return _lpbe(mf)
            mf._numint = multigrid_v3.MultiGridNumInt(slab)
            return mf
        dm = cp.asarray(build('KRKS', False).get_init_guess(slab, 'minao'))
        dms = cp.stack([dm*.6, dm*.4])
        vr_lpbe = build('KRKS', True).get_veff(slab, dm)
        vr_plain = build('KRKS', False).get_veff(slab, dm)
        vu_lpbe = build('KUKS', True).get_veff(slab, dms)
        vu_plain = build('KUKS', False).get_veff(slab, dms)
        dv_r = vr_lpbe - vr_plain
        self.assertGreater(float(abs(dv_r).max()), 1e-4)
        for s in range(2):
            self.assertAlmostEqual(
                float(abs(vu_lpbe[s] - vu_plain[s] - dv_r).max()), 0, 7)
        self.assertAlmostEqual(vu_lpbe.exc, vu_plain.exc, 8)
        self.assertAlmostEqual(vu_lpbe.ecoul - vu_plain.ecoul,
                               vr_lpbe.ecoul - vr_plain.ecoul, 8)

    def test_gc_kuks_symmetric_matches_krks(self):
        def compare(**kwargs):
            mf_r = _lpbe(slab.KRKS(xc='pbe', kpts=slab_kpts).to_gpu())
            mf_u = _lpbe(slab.KUKS(xc='pbe', kpts=slab_kpts).to_gpu())
            # KRHF.get_init_guess does not renormalize its guess (~11 e instead
            # of 8 in this small cell), which derails a fixed-mu KRKS run.  Start
            # both from the same atom guess scaled to the neutral electron count.
            dm0 = mf_r.get_init_guess(slab, key='atom')
            s1e = mf_r.get_ovlp()
            ne = float(cp.einsum('kij,kji->', dm0, s1e).real) / len(slab_kpts)
            dm0 *= slab.tot_electrons(1) / ne
            sr = GrandCanonicalKRKS(mf_r, sigma=1e-2, **kwargs)
            sr.kernel(dm0=dm0)
            su = GrandCanonicalKS(mf_u, sigma=1e-2, **kwargs)
            self.assertIsInstance(su, GrandCanonicalKUKS)
            su.kernel(dm0=dm0)
            self.assertTrue(sr.converged and su.converged)
            self.assertAlmostEqual(su.free_energy, sr.free_energy, 7)
            # Omega = A - mu*N inherits the error of mu (small DOS at E_F here:
            # ~5e-8 at conv_tol 1e-8), unlike the variational A, so it gets
            # the same precision as mu below.
            self.assertAlmostEqual(su.grand_potential, sr.grand_potential, 5)
            self.assertAlmostEqual(su.electron_number, sr.electron_number, 6)
            self.assertAlmostEqual(su.mu, sr.mu, 5)
            self.assertAlmostEqual(su.magnetization, 0, 5)
            # one entry per spin; a list rather than an array for ragged k-points
            self.assertEqual(len(su.mo_occ), 2)
            return sr
        sr = compare(nelec=slab.tot_electrons(1) + .02)
        compare(mu=sr.mu)

    def test_gc_kuks_magnetic_vs_smearing(self):
        '''Fixed-N GC-KUKS reproduces Fermi-smeared KUKS (common mu) for a
        ferromagnetic state, which init_magnetization selects.'''
        sigma = 5e-3
        mf = hlat.KUKS(xc='lda,vwn', kpts=hlat_kpts)
        dm = hlat.KRKS(xc='lda,vwn', kpts=hlat_kpts).get_init_guess(hlat, 'minao')
        ref = addons_cpu.smearing_(kuks_cpu.KUKS(hlat, kpts=hlat_kpts, xc='lda,vwn'),
                                   sigma=sigma, method='fermi')
        ref.conv_tol = 1e-11
        ref.max_cycle = 200
        ref.kernel(np.stack([dm*.75, dm*.25]))
        na, nb = [np.sum(o)/len(hlat_kpts) for o in ref.mo_occ]

        gc = GrandCanonicalKUKS(mf.to_gpu(), nelec=1.0, sigma=sigma)
        gc.init_magnetization = 0.5
        gc.kernel(dm0=dm)
        self.assertTrue(gc.converged)
        self.assertGreater(gc.magnetization, 0.5)
        self.assertAlmostEqual(gc.magnetization, na - nb, 5)
        self.assertAlmostEqual(gc.free_energy, ref.e_free, 6)
        self.assertAlmostEqual(gc.e_tot, ref.e_tot, 6)

        # Fixed mu at the fixed-N mu returns the same state.
        gm = GrandCanonicalKUKS(mf.to_gpu(), mu=gc.mu, sigma=sigma)
        gm.init_magnetization = 0.5
        gm.kernel(dm0=dm)
        self.assertTrue(gm.converged)
        self.assertAlmostEqual(gm.electron_number, 1.0, 5)
        self.assertAlmostEqual(gm.magnetization, gc.magnetization, 4)
        self.assertAlmostEqual(gm.grand_potential,
                               gc.free_energy - gc.mu * gc.electron_number, 7)


if __name__ == '__main__':
    print('Full Tests for GrandCanonicalKUKS')
    unittest.main()
