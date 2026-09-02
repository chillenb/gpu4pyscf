import pyscf
from gpu4pyscf.pbc.solvent.lpbe_as_numint import multigrid_lpbe
import unittest


class KnownValues(unittest.TestCase):
    def test_cu111_2layers(self):
        a = '''2.5526554800834367 0.0 0.0
        1.2763277400417183 2.210664492861818 0.0
        0.0 0.0 14.084234471774549
        '''
        atom = '''
        Cu      -0.00000000       1.47377633       6.00000000        2
        Cu       0.00000000       0.00000000       8.08423447        1
        '''
        cell = pyscf.M(
            a=a,
            atom=atom,
            basis={'Cu': 'DZVP-MOLOPT-PBE-GTH'},
            pseudo='GTH-PBE',
            verbose=4,
            nelec_frac=True,
        )
        cell.build()
        kpts = cell.make_kpts([6,6,1])

        mf = cell.KRKS(xc='pbe', kpts=kpts).to_gpu()
        mf = multigrid_lpbe(
            mf,
            tol=1e-12,
            ionic_strength=1.0,
        )
        mf = mf.smearing(method='fermi', sigma=5e-3)
        mf.init_guess = 'atom'
        mf.run()

        self.assertAlmostEqual(mf.e_tot, -96.2055783894392, 8)