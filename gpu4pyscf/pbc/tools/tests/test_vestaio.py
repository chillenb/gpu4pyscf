import struct

import numpy as np
import pytest

from pyscf.lib import param
import pyscf

from gpu4pyscf.pbc.tools.vestaio import write_vesta_pgrid



def test_write_vesta_pgrid_header_and_data_order(tmp_path):
    cell = pyscf.M(
        atom = '''
        O   0.000    0.    0.1174
        H   1.757    0.    0.4696
        H   0.757    0.    0.4696
        ''',
        a=np.eye(3)*7.,
        basis=('ccpvdz', [[3, [.5, 1]]]),
    )
    mesh = (2, 2, 2)
    data = np.arange(8.0).reshape(mesh)
    filepath = tmp_path / 'field.pgrid'

    write_vesta_pgrid(cell, mesh, filepath, data)
    contents = filepath.read_bytes()

    assert len(contents) == 152 + 4 * np.prod(mesh)
    assert struct.unpack_from('<4i', contents) == (3, 0, 0, 0)
    assert contents[16:96].split(b'\0', 1)[0] == b'GPU4PySCF periodic grid'
    assert struct.unpack_from('<8i', contents, 96) == (
        1, 0, 1, 3, 2, 2, 2, 8)
    np.testing.assert_allclose(
        struct.unpack_from('<6f', contents, 128),
        (2.0, 3.0, 4.0, 90.0, 90.0, 90.0),
        rtol=0.0, atol=1e-6)

    values = np.frombuffer(contents, dtype='<f4', offset=152)
    np.testing.assert_array_equal(values, data.ravel(order='F'))


def test_write_vesta_pgrid_rejects_incompatible_data(tmp_path):
    cell = pyscf.M(
        atom = '''
        O   0.000    0.    0.1174
        H   1.757    0.    0.4696
        H   0.757    0.    0.4696
        ''',
        a=np.eye(3)*7.,
        basis=('ccpvdz', [[3, [.5, 1]]]),
    )
    with pytest.raises(ValueError, match='mesh requires 8'):
        write_vesta_pgrid(cell, (2, 2, 2), tmp_path / 'field.pgrid',
                          np.zeros(7))


def test_write_vesta_pgrid_rejects_complex_data(tmp_path):
    cell = pyscf.M(
        atom = '''
        O   0.000    0.    0.1174
        H   1.757    0.    0.4696
        H   0.757    0.    0.4696
        ''',
        a=np.eye(3)*7.,
        basis=('ccpvdz', [[3, [.5, 1]]]),
    )
    with pytest.raises(ValueError, match='must be real'):
        write_vesta_pgrid(cell, (1, 1, 1), tmp_path / 'field.pgrid',
                          np.asarray([1.0j]))
