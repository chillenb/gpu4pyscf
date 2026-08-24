#!/usr/bin/env python
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

import cupy as cp
import numpy as np
import pytest

from pyscf.pbc import gto
from gpu4pyscf.pbc import df, scf
from gpu4pyscf.pbc.gw.krpa import KRPA


# GPU GDF uses a different auxiliary-metric factorization from PySCF RSGDF.
DIAMOND_NO_FC = (-10.694342003057113, -0.18527462954683174)
DIAMOND_FC = (-10.716296661695937, -0.20722928818566372)


@pytest.fixture(scope='module')
def diamond_krhf():
    cell = gto.Cell()
    cell.build(
        unit='angstrom',
        a='''
            0.000000     1.783500     1.783500
            1.783500     0.000000     1.783500
            1.783500     1.783500     0.000000
        ''',
        atom='C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375',
        dimension=3,
        verbose=0,
        output='/dev/null',
        pseudo='gth-pbe',
        basis='gth-dzv',
        precision=1e-12,
    )

    kpts = cell.make_kpts([3, 1, 1], scaled_center=[0, 0, 0])
    gdf = df.GDF(cell, kpts)
    gdf.build()

    kmf = scf.KRHF(cell, kpts)
    kmf.with_df = gdf
    kmf.conv_tol = 1e-12
    kmf.kernel()

    yield kmf
    cell.stdout.close()


def test_krpa_no_fc(diamond_krhf):
    rpa = KRPA(diamond_krhf)
    rpa.fc = False
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(DIAMOND_NO_FC[1], abs=1e-6)
    assert rpa.e_tot == pytest.approx(DIAMOND_NO_FC[0], abs=1e-6)


def test_krpa_acfd_exx_high_cost(diamond_krhf):
    rpa = KRPA(diamond_krhf)
    rpa.fc = False
    rpa.acfd_exx = True
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(DIAMOND_NO_FC[1], abs=1e-6)
    assert rpa.e_tot == pytest.approx(DIAMOND_NO_FC[0], abs=1e-6)


def test_krpa_with_fc(diamond_krhf):
    rpa = KRPA(diamond_krhf)
    rpa.fc = True
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(DIAMOND_FC[1], abs=1e-6)
    assert rpa.e_tot == pytest.approx(DIAMOND_FC[0], abs=1e-6)


def test_krpa_get_idx_metal():
    from gpu4pyscf.pbc.gw.krpa import get_idx_metal
    cases = [
        ([2.0, 1.5, 0.5, 0.0], ([0], [1, 2], [3])),
        ([1.9, 0.7, 0.0], ([], [0, 1], [2])),
        ([2.0, 1.2, 0.1], ([0], [1, 2], [])),
        ([1.9, 1.0, 0.1], ([], [0, 1, 2], [])),
    ]
    for mo_occ, expected in cases:
        result = tuple(list(idx) for idx in get_idx_metal(np.asarray(mo_occ)))
        assert result == expected


def test_krpa_get_rho_response_metal_all_fractional():
    from gpu4pyscf.pbc.gw.krpa import get_rho_response_metal
    omega = 0.7
    mo_energy = cp.array([[-1.0, -0.2, 0.8]])
    mo_occ = cp.array([[1.8, 1.0, 0.2]])
    Lpq = [cp.arange(18).reshape(2, 3, 3).astype(cp.complex128) / 20]

    eia = mo_energy[0, :, None] - mo_energy[0, None, :]
    fia = mo_occ[0, :, None] - mo_occ[0, None, :]
    weight = eia * fia / (omega**2 + eia**2)
    expected = cp.einsum('Pia,ia,Qia->PQ',
                         Lpq[0], weight, Lpq[0].conj())

    result = get_rho_response_metal(
        omega, mo_energy, mo_occ, Lpq, [0])
    cp.testing.assert_allclose(result, expected)


def test_krpa_rho_accum_real_into_complex():
    """A real response contribution accumulates through the complex real view."""
    from gpu4pyscf.pbc.gw.krpa import rho_accum_inner

    omega = 0.7
    alpha = 1.3
    eia = cp.array([[-1.0, -1.8], [-0.6, -1.4]])
    Lov = cp.arange(8, dtype=cp.float64).reshape(2, 2, 2) / 20
    Pi = cp.full((2, 2), 2j, dtype=cp.complex128)

    weight = eia / (omega**2 + eia**2)
    expected = Pi + alpha * cp.einsum(
        'Pia,ia,Qia->PQ', Lov, weight, Lov)
    rho_accum_inner(Pi, eia, omega, Lov, alpha=alpha)

    cp.testing.assert_allclose(Pi, expected)


@pytest.mark.parametrize('complex_lia', [False, True])
def test_krpa_get_rho_response_batched(complex_lia):
    """The k-batched response agrees with an explicit per-k contraction."""
    from gpu4pyscf.pbc.gw.krpa import get_rho_response

    omega = 0.7
    mo_energy = cp.array([
        [-1.2, -0.7, 0.3, 0.9],
        [-1.0, -0.5, 0.4, 1.1],
        [-1.1, -0.6, 0.2, 1.0],
    ])
    Lia = cp.arange(24, dtype=cp.float64).reshape(3, 2, 2, 2) / 20
    if complex_lia:
        Lia = Lia + 0.1j * Lia[::-1]
    kidx = np.array([2, 0, 1])

    expected = cp.zeros((2, 2), dtype=cp.complex128)
    for k, a in enumerate(kidx):
        eia = mo_energy[k, :2, None] - mo_energy[a, None, 2:]
        weight = eia / (omega**2 + eia**2)
        expected += 4.0 / len(kidx) * cp.einsum(
            'Pia,ia,Qia->PQ', Lia[k], weight, Lia[k].conj())

    result = get_rho_response(omega, mo_energy, Lia, kidx)
    cp.testing.assert_allclose(result, expected)


def test_krpa_kconserv_shifted_kmesh():
    """The RPA transfer table is invariant under a rigid k-mesh shift."""
    from gpu4pyscf.pbc.gw.krpa import get_kconserv_ria_efficient

    cell = gto.Cell()
    cell.build(
        a=np.eye(3) * 3,
        atom='H 0 0 0',
        basis='sto-3g',
        spin=1,
        verbose=0,
    )
    kmesh = [2, 2, 2]
    kpts = cell.make_kpts(kmesh, scaled_center=[0, 0, 0])
    shifted_kpts = cell.make_kpts(
        kmesh, scaled_center=[0.6223 / 2, 0.2953 / 2, 0])

    reference = get_kconserv_ria_efficient(cell, kpts)
    result = get_kconserv_ria_efficient(cell, shifted_kpts)
    np.testing.assert_array_equal(result, reference)


@pytest.fixture(scope='module')
def water_krhf():
    cell = gto.Cell()
    cell.build(
        unit='angstrom',
        atom='''
        O          0.00000        0.00000        0.11779
        H          0.00000        0.75545       -0.47116
        H          0.00000       -0.75545       -0.47116
        ''',
        a=np.eye(3) * 5,
        verbose=0,
        output='/dev/null',
        pseudo=None,
        basis='cc-pvdz',
        precision=1e-12,
    )

    kpts = cell.make_kpts([1, 1, 1], scaled_center=[0, 0, 0])
    gdf = df.GDF(cell, kpts)
    gdf.build()

    kmf = scf.KRHF(cell, kpts)
    kmf.with_df = gdf
    kmf.conv_tol = 1e-12

    yield kmf
    cell.stdout.close()


def test_krpa_exx_with_frozen(water_krhf):
    """KRPA exchange agrees with the GPU mean-field exchange matrix."""
    from gpu4pyscf.pbc.gw.krpa import get_rpa_exx

    kmf = water_krhf
    for sigma_ev in [0.0, 1.0]:
        if sigma_ev > 1e-4:
            kmf = kmf.smearing(
                sigma=sigma_ev / 27.211399, method='fermi')
        kmf.kernel()

        rpa = KRPA(kmf, frozen=0)
        mf = rpa._scf
        dm = mf.make_rdm1()
        vk = mf.get_k(mf.cell, dm, kpts=mf.kpts)
        e_x_ref = _as_float(
            cp.einsum('kij,kji->', vk, dm).real *
            (-0.25 / len(mf.kpts)))
        e_x = get_rpa_exx(rpa)
        assert e_x == pytest.approx(e_x_ref, abs=1e-6)

        rpa = KRPA(kmf, frozen=2)
        e_x = get_rpa_exx(rpa)
        assert e_x == pytest.approx(e_x_ref, abs=1e-6)


def _as_float(value):
    return float(value.item()) if isinstance(value, cp.ndarray) else float(value)
