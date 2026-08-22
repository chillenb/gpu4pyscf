import struct

import numpy as np
import cupy as cp

def cell_params(cell):
    lattice = cell.lattice_vectors(unit='Angstrom')

    lengths = np.linalg.norm(lattice, axis=1)
    def angle(i, j):
        cosine = np.dot(lattice[i], lattice[j]) / (lengths[i] * lengths[j])
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    alpha = angle(1, 2)
    beta = angle(0, 2)
    gamma = angle(0, 1)
    return np.asarray([*lengths, alpha, beta, gamma], dtype=np.float32)


def write_vesta_pgrid(cell, mesh, filepath, data):
    """Write scalar volumetric data in VESTA's binary ``.pgrid`` format.

    Args:
        cell : :class:`pyscf.pbc.gto.Cell`
            Three-dimensional periodic cell. Its lattice vectors are converted
            from PySCF's internal Bohr units to Angstrom.
        mesh : sequence of 3 integers
            Numbers of periodic voxels along the ``a``, ``b``, and ``c`` axes.
        filepath : path-like
            Output file path.
        data : array-like
            Real scalar data with ``prod(mesh)`` elements. A three-dimensional
            input is interpreted with shape ``mesh``; a flat input follows the
            PySCF uniform-grid convention (the ``c`` index varies fastest).

    The file uses VESTA format version 3.0.0.0, a raw periodic grid, and one
    32-bit floating-point value per voxel. NumPy and CuPy arrays are accepted.
    """
    mesh_tuple = tuple(int(n) for n in mesh)

    assert len(mesh_tuple) == 3, 'mesh must have three dimensions'
    assert all(n > 0 for n in mesh_tuple), 'mesh dimensions must be positive'
    assert all(n < np.iinfo(np.int32).max for n in mesh_tuple), \
        'mesh dimensions must fit in a 32-bit integer'

    nvox = mesh_tuple[0] * mesh_tuple[1] * mesh_tuple[2]
    if nvox > np.iinfo(np.int32).max:
        raise ValueError('number of voxels exceeds the VESTA int32 limit')

    values = cp.asnumpy(data.ravel(order='F').astype(cp.float32))

    assert values.size == nvox, 'data size does not match mesh dimensions'

    # VESTA records a first, then b, then c. This is Fortran order for an
    # array whose axes are (a, b, c), whereas PySCF flattens that array in C
    # order.

    cell_parameters = cell_params(cell)

    title = b'GPU4PySCF periodic grid'
    title = title + b'\0' * (80 - len(title))
    header = struct.pack(
        '<4i80s8i6f',
        3, 0, 0, 0,             # version
        title,
        1, 0, 1, 3,             # periodic, raw, one value, 3D
        *mesh_tuple, nvox,
        *cell_parameters,
    )

    with open(filepath, 'wb') as output:
        output.write(header)
        values.tofile(output)
