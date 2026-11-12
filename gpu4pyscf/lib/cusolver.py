# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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


import numpy as np
import cupy
import ctypes
from cupy_backends.cuda.libs import cusolver
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device

from gpu4pyscf.lib.utils import load_library

from ctypes.util import find_library
libcusolver = find_library('cusolver')
if libcusolver is not None:
    libcusolver = ctypes.CDLL(libcusolver)
else:
    libcusolver = load_library('libcusolver', path='nvidia/cusolver/lib')

# workspace size (lwork) provided by the cusolver*_bufferSize is an 32-bit
# integer. For arrays above this dimension, the workspace size would overflow.
MAX_EIGH_DIM = 23150

CUSOLVER_EIG_TYPE_1 = 1
CUSOLVER_EIG_TYPE_2 = 2
CUSOLVER_EIG_TYPE_3 = 3

CUSOLVER_EIG_MODE_NOVECTOR = 0
CUSOLVER_EIG_MODE_VECTOR = 1

libcusolver.cusolverDnDsygvd_bufferSize.restype = int
libcusolver.cusolverDnDsygvd.restype = int

_buffersize = {}

# https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-sygvd
libcusolver.cusolverDnDsygvd_bufferSize.argtypes = [
    ctypes.c_void_p, # handle
    ctypes.c_int,    # itype
    ctypes.c_int,    # jobz
    ctypes.c_int,    # uplo
    ctypes.c_int,    # n
    ctypes.c_void_p, # *A
    ctypes.c_int,    # lda
    ctypes.c_void_p, # *B
    ctypes.c_int,    # ldb
    ctypes.c_void_p, # *w
    ctypes.c_void_p  # *lwork
]

libcusolver.cusolverDnDsygvd.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # itype
    ctypes.c_int,     # jobz
    ctypes.c_int,     # uplo
    ctypes.c_int,     # n
    ctypes.c_void_p,  # *A
    ctypes.c_int,     # lda
    ctypes.c_void_p,  # *B
    ctypes.c_int,     # ldb
    ctypes.c_void_p,  # *w
    ctypes.c_void_p,  # *work
    ctypes.c_int,     # lwork
    ctypes.c_void_p   # *devInfo
]

# https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-sygvd
libcusolver.cusolverDnZhegvd_bufferSize.argtypes = [
    ctypes.c_void_p, # handle
    ctypes.c_int,    # itype
    ctypes.c_int,    # jobz
    ctypes.c_int,    # uplo
    ctypes.c_int,    # n
    ctypes.c_void_p, # *A
    ctypes.c_int,    # lda
    ctypes.c_void_p, # *B
    ctypes.c_int,    # ldb
    ctypes.c_void_p, # *w
    ctypes.c_void_p  # *lwork
]

libcusolver.cusolverDnZhegvd.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # itype
    ctypes.c_int,     # jobz
    ctypes.c_int,     # uplo
    ctypes.c_int,     # n
    ctypes.c_void_p,  # *A
    ctypes.c_int,     # lda
    ctypes.c_void_p,  # *B
    ctypes.c_int,     # ldb
    ctypes.c_void_p,  # *w
    ctypes.c_void_p,  # *work
    ctypes.c_int,     # lwork
    ctypes.c_void_p   # *devInfo
]

def eigh(h, s, overwrite=False):
    '''Solve the generalized Hermitian eigenproblem with positive-definite s.
    '''
    assert h.dtype == s.dtype
    assert h.dtype in (np.float64, np.complex128)
    assert h.flags.c_contiguous and s.flags.c_contiguous
    n = h.shape[0]
    if h.dtype == np.complex128 and h.flags.c_contiguous:
        # zhegvd requires the matrices in F-order. For hermitian matrices,
        # .T.copy() is equivalent to .conj()
        A = h.conj()
        B = s.conj()
    elif overwrite:
        A = h
        B = s
    else:
        A = h.copy()
        B = s.copy()
    _handle = device.get_cusolver_handle()
    cusolver.setStream(_handle, cupy.cuda.get_current_stream().ptr)
    w = cupy.zeros(n)

    # TODO: reuse workspace
    if (h.dtype, n) in _buffersize:
        lwork = _buffersize[h.dtype, n]
    else:
        lwork = ctypes.c_int(0)
        if h.dtype == np.float64:
            fn = libcusolver.cusolverDnDsygvd_bufferSize
        else:
            fn = libcusolver.cusolverDnZhegvd_bufferSize
        status = fn(
            _handle,
            CUSOLVER_EIG_TYPE_1,
            CUSOLVER_EIG_MODE_VECTOR,
            cublas.CUBLAS_FILL_MODE_LOWER,
            n,
            A.data.ptr,
            n,
            B.data.ptr,
            n,
            w.data.ptr,
            ctypes.byref(lwork)
        )
        lwork = lwork.value
        _buffersize[h.dtype, n] = lwork

        if status != 0:
            raise LinAlgError("failed in buffer size")

    if h.dtype == np.float64:
        fn = libcusolver.cusolverDnDsygvd
    else:
        fn = libcusolver.cusolverDnZhegvd
    work = cupy.empty(lwork, dtype=h.dtype)
    devInfo = cupy.empty(1, dtype=np.int32)
    status = fn(
        _handle,
        CUSOLVER_EIG_TYPE_1,
        CUSOLVER_EIG_MODE_VECTOR,
        cublas.CUBLAS_FILL_MODE_LOWER,
        n,
        A.data.ptr,
        n,
        B.data.ptr,
        n,
        w.data.ptr,
        work.data.ptr,
        lwork,
        devInfo.data.ptr,
    )

    if status != 0:
        raise LinAlgError("failed in eigh kernel")
    # API success only confirms execution; devInfo reports numerical failure.
    info = int(devInfo.get()[0])
    if info < 0:
        raise LinAlgError(f'eigh failed: cuSOLVER argument {-info} is invalid (devInfo={info})')
    if info > n:
        raise LinAlgError('eigh failed: overlap matrix is not positive definite '
                          f'(leading minor {info-n}, devInfo={info})')
    if info != 0:
        raise LinAlgError(f'eigh failed to converge (devInfo={info})')
    return w, A.T

def cholesky(A):
    n = len(A)
    if A.flags.f_contiguous:
        A = A.T
    x = A.copy(order='C')
    handle = device.get_cusolver_handle()
    cusolver.setStream(handle, cupy.cuda.get_current_stream().ptr)
    if A.dtype == np.float64:
        potrf = cusolver.dpotrf
        potrf_bufferSize = cusolver.dpotrf_bufferSize
    else:
        potrf = cusolver.zpotrf
        potrf_bufferSize = cusolver.zpotrf_bufferSize
    buffersize = potrf_bufferSize(handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n)
    workspace = cupy.empty(buffersize, dtype=A.dtype)
    dev_info = cupy.empty(1, dtype=np.int32)
    potrf(handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n,
        workspace.data.ptr, buffersize, dev_info.data.ptr)

    if dev_info[0] != 0:
        raise LinAlgError('failed to perform Cholesky Decomposition')
    cupy.linalg._util._tril(x,k=0)
    return x

class LinAlgError(RuntimeError):
    pass


# cusolverDnXsyevBatched (CUDA >= 12.6): batched symmetric/Hermitian eigensolver for
# any n.  Much faster than looping over cusolverDn<t>syevd for many small/medium
# matrices (e.g. 196 complex 598x598 blocks: 0.19 s vs 1.33 s on an H200).
_CUDA_R_64F, _CUDA_C_64F = 1, 5
_FILL_LOWER = 0
HAS_SYEV_BATCHED = hasattr(libcusolver, 'cusolverDnXsyevBatched')
# Upper bound on the bytes of one batched call (input copy); larger stacks are split.
SYEV_BATCHED_MAX_BYTES = 4 * 1024**3
if HAS_SYEV_BATCHED:
    libcusolver.cusolverDnCreateParams.argtypes = [ctypes.c_void_p]
    libcusolver.cusolverDnXsyevBatched_bufferSize.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p,
        ctypes.c_int, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int64]
    libcusolver.cusolverDnXsyevBatched.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_int64]
_dn_params = {}

def _get_dn_params():
    dev = cupy.cuda.Device().id
    if dev not in _dn_params:
        p = ctypes.c_void_p()
        status = libcusolver.cusolverDnCreateParams(ctypes.byref(p))
        if status != 0:
            raise RuntimeError('cusolverDnCreateParams status %d' % status)
        _dn_params[dev] = p
    return _dn_params[dev]

def _syev_batched(a, vectors):
    nb, n = a.shape[:2]
    cplx = a.dtype == np.complex128
    # cuSOLVER is column-major: the C-ordered Hermitian block a reads as conj(a), so
    # hand it conj(a), which it sees as a.  Eigenvectors come back as columns.
    buf = a.conj() if cplx else a.copy()
    buf = cupy.ascontiguousarray(buf)
    w = cupy.empty((nb, n), dtype=np.float64)
    info = cupy.zeros(nb, dtype=np.int32)
    dtype_a = _CUDA_C_64F if cplx else _CUDA_R_64F
    jobz = CUSOLVER_EIG_MODE_VECTOR if vectors else CUSOLVER_EIG_MODE_NOVECTOR
    handle = device.get_cusolver_handle()
    params = _get_dn_params()
    dsize, hsize = ctypes.c_size_t(), ctypes.c_size_t()
    status = libcusolver.cusolverDnXsyevBatched_bufferSize(
        handle, params, jobz, _FILL_LOWER, n, dtype_a, buf.data.ptr, n, _CUDA_R_64F,
        w.data.ptr, dtype_a, ctypes.byref(dsize), ctypes.byref(hsize), nb)
    if status != 0:
        raise RuntimeError('cusolverDnXsyevBatched_bufferSize status %d' % status)
    dwork = cupy.empty(max(1, dsize.value), dtype=np.uint8)
    hwork = ctypes.create_string_buffer(max(1, hsize.value))
    status = libcusolver.cusolverDnXsyevBatched(
        handle, params, jobz, _FILL_LOWER, n, dtype_a, buf.data.ptr, n, _CUDA_R_64F,
        w.data.ptr, dtype_a, dwork.data.ptr, dsize.value, hwork, hsize.value,
        info.data.ptr, nb)
    if status != 0:
        raise RuntimeError('cusolverDnXsyevBatched status %d' % status)
    bad = int(cupy.count_nonzero(info).item())
    if bad:
        raise LinAlgError('cusolverDnXsyevBatched failed for %d of %d matrices' % (bad, nb))
    if not vectors:
        return w
    return w, cupy.ascontiguousarray(buf.transpose(0, 2, 1))

def eigh_batched(a, vectors=True):
    '''Eigen-decomposition of a stack of Hermitian (complex128) or real symmetric
    (float64) matrices a[b], shape (nb, n, n); only the lower triangle is read.
    Returns w (nb, n) in ascending order and, if vectors, v (nb, n, n) with
    a[b] @ v[b] = v[b] * w[b], like cupy.linalg.eigh on each block.  Falls back to a
    loop over cupy.linalg.eigh(/eigvalsh) when cusolverDnXsyevBatched is unavailable.'''
    a = cupy.asarray(a)
    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError('eigh_batched expects a stack of square matrices')
    if a.dtype not in (np.complex128, np.float64):
        a = a.astype(np.complex128 if a.dtype.kind == 'c' else np.float64)
    nb = a.shape[0]
    if not HAS_SYEV_BATCHED or nb == 0:
        if vectors:
            out = [cupy.linalg.eigh(x) for x in a]
            return (cupy.stack([x[0] for x in out]), cupy.stack([x[1] for x in out]))
        return cupy.stack([cupy.linalg.eigvalsh(x) for x in a])
    chunk = max(1, SYEV_BATCHED_MAX_BYTES // max(1, a[0].nbytes))
    if nb <= chunk:
        return _syev_batched(a, vectors)
    parts = [_syev_batched(a[i:i+chunk], vectors) for i in range(0, nb, chunk)]
    if vectors:
        return (cupy.concatenate([p[0] for p in parts]),
                cupy.concatenate([p[1] for p in parts]))
    return cupy.concatenate(parts)
