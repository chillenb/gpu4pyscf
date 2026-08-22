import struct

import cupy as cp
import numpy as np
import pytest

from pyscf.pbc import gto

from gpu4pyscf.pbc.tools.vestaio import cell_params, write_vesta_pgrid


@pytest.fixture(scope='module')
def cell():
    cell = gto.Cell()
    cell.atom = 'He 0 0 0'
    cell.a = np.asarray([
        [3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 2.5, 2.5 * np.sqrt(3.0)],
    ])
    cell.basis = 'sto-3g'
    cell.unit = 'Angstrom'
    cell.verbose = 0
    cell.build()
    return cell


def read_pgrid(filepath):
    contents = filepath.read_bytes()
    header_size = struct.calcsize('<4i80s8i6f')
    return contents, np.frombuffer(contents, dtype='<f4', offset=header_size)


def test_cell_params(cell):
    np.testing.assert_allclose(
        cell_params(cell),
        [3.0, 4.0, 5.0, 60.0, 90.0, 90.0],
        rtol=0.0, atol=1e-6)


def test_write_vesta_pgrid_header_and_3d_data(cell, tmp_path):
    mesh = (2, 3, 4)
    data = cp.arange(np.prod(mesh), dtype=cp.float64).reshape(mesh)
    filepath = tmp_path / 'field.pgrid'

    write_vesta_pgrid(cell, mesh, filepath, data)
    contents, values = read_pgrid(filepath)

    assert len(contents) == 152 + 4 * np.prod(mesh)
    assert struct.unpack_from('<4i', contents) == (3, 0, 0, 0)
    assert contents[16:96].split(b'\0', 1)[0] == b'GPU4PySCF periodic grid'
    assert struct.unpack_from('<8i', contents, 96) == (
        1, 0, 1, 3, 2, 3, 4, 24)
    np.testing.assert_allclose(
        struct.unpack_from('<6f', contents, 128),
        [3.0, 4.0, 5.0, 60.0, 90.0, 90.0],
        rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(
        values, cp.asnumpy(data.ravel(order='F')).astype(np.float32))


def test_write_vesta_pgrid_flat_pyscf_grid_order(cell, tmp_path):
    mesh = (2, 3, 4)
    data = cp.arange(np.prod(mesh), dtype=cp.float64)
    filepath = tmp_path / 'flat-field.pgrid'

    write_vesta_pgrid(cell, mesh, filepath, data)
    _, values = read_pgrid(filepath)

    expected = cp.asnumpy(
        data.reshape(mesh).ravel(order='F')).astype(np.float32)
    np.testing.assert_array_equal(values, expected)


def test_write_vesta_pgrid_rejects_wrong_data_size(cell, tmp_path):
    with pytest.raises(AssertionError, match='data size'):
        write_vesta_pgrid(
            cell, (2, 2, 2), tmp_path / 'field.pgrid', cp.zeros(7))
