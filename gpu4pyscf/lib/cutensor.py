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

import ctypes
import ctypes.util
import numpy as np
import cupy
from gpu4pyscf.lib import logger

try:
    import cupy_backends.cuda.libs.cutensor  # NOQA
    from cupyx import cutensor
    from cupy_backends.cuda.libs import cutensor as cutensor_backend
    ALGO_DEFAULT = cutensor_backend.ALGO_DEFAULT
    OP_IDENTITY = cutensor_backend.OP_IDENTITY
    JIT_MODE_NONE = cutensor_backend.JIT_MODE_NONE
    WORKSPACE_RECOMMENDED = cutensor_backend.WORKSPACE_MIN
    #WORKSPACE_RECOMMENDED = cutensor_backend.WORKSPACE_RECOMMENDED
    _tensor_descriptors = {}

    _libcutensor = ctypes.CDLL(
        ctypes.util.find_library('cutensor') or 'libcutensor.so.2')
    _c_void_p = ctypes.c_void_p
    _libcutensor.cutensorCreateContractionTrinary.argtypes = [
        _c_void_p, ctypes.POINTER(_c_void_p),
        _c_void_p, _c_void_p, ctypes.c_int,
        _c_void_p, _c_void_p, ctypes.c_int,
        _c_void_p, _c_void_p, ctypes.c_int,
        _c_void_p, _c_void_p, ctypes.c_int,
        _c_void_p, _c_void_p, _c_void_p,
    ]
    _libcutensor.cutensorCreateContractionTrinary.restype = ctypes.c_int
    _libcutensor.cutensorContractTrinary.argtypes = [
        _c_void_p, _c_void_p,
        _c_void_p, _c_void_p, _c_void_p, _c_void_p,
        _c_void_p, _c_void_p, _c_void_p,
        _c_void_p, ctypes.c_uint64, _c_void_p,
    ]
    _libcutensor.cutensorContractTrinary.restype = ctypes.c_int
    _libcutensor.cutensorGetErrorString.argtypes = [ctypes.c_int]
    _libcutensor.cutensorGetErrorString.restype = ctypes.c_char_p
except (ImportError, AttributeError, OSError):
    cutensor = None
    _libcutensor = None
    ALGO_DEFAULT = None
    OP_IDENTITY = None
    JIT_MODE_NONE = None
    WORKSPACE_RECOMMENDED = None


def _check_cutensor_status(status):
    if status != 0:
        error = _libcutensor.cutensorGetErrorString(status).decode()
        raise RuntimeError(f'cuTENSOR error: {error}')


def _compute_descriptor_ptr(compute_desc, dtype):
    if compute_desc == 0:
        if np.dtype(dtype) in (np.dtype(np.float64), np.dtype(np.complex128)):
            compute_desc = cutensor_backend.COMPUTE_DESC_64F
        else:
            compute_desc = cutensor_backend.COMPUTE_DESC_32F

    symbols = {
        cutensor_backend.COMPUTE_DESC_16F: 'CUTENSOR_COMPUTE_DESC_16F',
        cutensor_backend.COMPUTE_DESC_16BF: 'CUTENSOR_COMPUTE_DESC_16BF',
        cutensor_backend.COMPUTE_DESC_32F: 'CUTENSOR_COMPUTE_DESC_32F',
        cutensor_backend.COMPUTE_DESC_64F: 'CUTENSOR_COMPUTE_DESC_64F',
        cutensor_backend.COMPUTE_DESC_TF32: 'CUTENSOR_COMPUTE_DESC_TF32',
        cutensor_backend.COMPUTE_DESC_3xTF32: 'CUTENSOR_COMPUTE_DESC_3XTF32',
    }
    try:
        symbol = symbols[compute_desc]
    except KeyError as err:
        raise ValueError(f'unsupported compute descriptor: {compute_desc}') from err
    return ctypes.c_void_p.in_dll(_libcutensor, symbol).value

def _auto_create_mode(array, mode):
    if not isinstance(mode, cutensor.Mode):
        mode = cutensor.create_mode(*mode)
    if array.ndim != mode.ndim:
        raise ValueError(
            'ndim mismatch: {} != {}'.format(array.ndim, mode.ndim))
    return mode

def _create_tensor_descriptor(a):
    if any(x == 0 for x in a.strides):
        strides = list(a.strides)
        if strides[0] == 0:
            strides[0] = a.nbytes
        for i, x in enumerate(strides[1:]):
            if x == 0:
                strides[i+1] = strides[i]
        a = cupy.ndarray(a.shape, a.dtype, a.data, strides)
    return cutensor.create_tensor_descriptor(a)

def _contract_einsum(pattern, a, b, alpha, beta, out=None, einsum=cupy.einsum):
    if out is None:
        out = einsum(pattern, a, b)
        out *= alpha
    elif beta == 0.:
        out[:] = einsum(pattern, a, b)
        out *= alpha
    else:
        out *= beta
        tmp = einsum(pattern, a, b)
        tmp *= alpha
        out += tmp
    return out


def _contract_trinary_einsum(
        pattern, a, b, c, alpha, beta, out=None, einsum=cupy.einsum):
    if out is None:
        out = einsum(pattern, a, b, c)
        out *= alpha
    elif beta == 0.:
        out[:] = einsum(pattern, a, b, c)
        out *= alpha
    else:
        out *= beta
        tmp = einsum(pattern, a, b, c)
        tmp *= alpha
        out += tmp
    return out

def contraction(
    pattern, a, b, alpha, beta,
    out=None,
    op_a=OP_IDENTITY,
    op_b=OP_IDENTITY,
    op_c=OP_IDENTITY,
    algo=ALGO_DEFAULT,
    jit_mode=JIT_MODE_NONE,
    compute_desc=0,
    ws_pref=WORKSPACE_RECOMMENDED
):
    if a.size == 0 or b.size == 0:
        # cutensor does not support the 0-sized operands
        return _contract_einsum(pattern, a, b, alpha, beta, out)

    pattern = pattern.replace(" ", "")
    str_a, rest = pattern.split(',')
    str_b, str_c = rest.split('->')
    key = str_a + str_b
    val = list(a.shape) + list(b.shape)
    shape = {k:v for k, v in zip(key, val)}

    mode_a = list(str_a)
    mode_b = list(str_b)
    mode_c = list(str_c)
    if len(mode_c) != len(set(mode_c)):
        raise ValueError('Output subscripts string includes the same subscript multiple times.')

    dtype = np.result_type(a.dtype, b.dtype)
    a = cupy.asarray(a, dtype=dtype)
    b = cupy.asarray(b, dtype=dtype)
    if out is None:
        out = cupy.empty([shape[k] for k in str_c], order='C', dtype=dtype)
    c = out

    if a.size == 0 or b.size == 0 or c.size == 0:
        raise ValueError(f"cutensor contraction doesn't support zero-sized array (a.shape = {a.shape}, b.shape = {b.shape}, expected c.shape = {c.shape})")

    desc_a = _create_tensor_descriptor(a)
    desc_b = _create_tensor_descriptor(b)
    desc_c = _create_tensor_descriptor(c)

    mode_a = _auto_create_mode(a, mode_a)
    mode_b = _auto_create_mode(b, mode_b)
    mode_c = _auto_create_mode(c, mode_c)
    operator = cutensor.create_contraction(
        desc_a, mode_a, op_a, desc_b, mode_b, op_b, desc_c, mode_c, op_c,
        compute_desc)
    plan_pref = cutensor.create_plan_preference(algo=algo, jit_mode=jit_mode)
    ws_size = cutensor_backend.estimateWorkspaceSize(
        cutensor._get_handle().ptr, operator.ptr, plan_pref.ptr, ws_pref)
    plan = cutensor.create_plan(operator, plan_pref, ws_limit=ws_size)
    ws = cupy.empty(ws_size, dtype=np.int8)
    out = c

    alpha = np.asarray(alpha, dtype=dtype)
    beta = np.asarray(beta, dtype=dtype)

    handler = cutensor._get_handle()
    cutensor_backend.contract(handler.ptr, plan.ptr,
                             alpha.ctypes.data, a.data.ptr, b.data.ptr,
                             beta.ctypes.data, c.data.ptr, out.data.ptr,
                             ws.data.ptr, ws_size)
    return out


def contraction_trinary(
    pattern, a, b, c, alpha, beta,
    out=None,
    op_a=OP_IDENTITY,
    op_b=OP_IDENTITY,
    op_c=OP_IDENTITY,
    op_d=OP_IDENTITY,
    algo=ALGO_DEFAULT,
    jit_mode=JIT_MODE_NONE,
    compute_desc=0,
    ws_pref=WORKSPACE_RECOMMENDED
):
    """Contract three tensors using cuTENSOR's trinary contraction API.

    Computes ``out = alpha * einsum(pattern, a, b, c) + beta * out``.
    ``pattern`` must use explicit-output einsum notation with three operands.
    """
    if _libcutensor is None:
        raise RuntimeError('cuTENSOR trinary contraction is not available')

    pattern = pattern.replace(' ', '')
    try:
        inputs, str_out = pattern.split('->')
        str_a, str_b, str_c = inputs.split(',')
    except ValueError as err:
        raise ValueError(
            'pattern must be explicit-output einsum notation with three operands'
        ) from err
    if len(str_out) != len(set(str_out)):
        raise ValueError(
            'Output subscripts string includes the same subscript multiple times.')

    dtype = np.result_type(a.dtype, b.dtype, c.dtype)
    a = cupy.asarray(a, dtype=dtype)
    b = cupy.asarray(b, dtype=dtype)
    c = cupy.asarray(c, dtype=dtype)

    shape = {}
    for subscripts, operand in ((str_a, a), (str_b, b), (str_c, c)):
        if len(subscripts) != operand.ndim:
            raise ValueError(
                f'ndim mismatch for {subscripts}: '
                f'{operand.ndim} != {len(subscripts)}')
        for subscript, extent in zip(subscripts, operand.shape):
            if subscript in shape and shape[subscript] != extent:
                raise ValueError(
                    f'inconsistent extent for mode {subscript}: '
                    f'{shape[subscript]} != {extent}')
            shape[subscript] = extent

    try:
        out_shape = tuple(shape[subscript] for subscript in str_out)
    except KeyError as err:
        raise ValueError(f'output mode {err.args[0]} does not appear in an input') from err

    if out is None:
        out = cupy.empty(out_shape, order='C', dtype=dtype)
        beta = 0.0
    elif out.shape != out_shape:
        raise ValueError(f'output shape mismatch: {out.shape} != {out_shape}')
    elif out.dtype != dtype:
        raise ValueError(f'output dtype mismatch: {out.dtype} != {dtype}')

    if a.size == 0 or b.size == 0 or c.size == 0 or out.size == 0:
        return _contract_trinary_einsum(
            pattern, a, b, c, alpha, beta, out=out)

    desc_a = _create_tensor_descriptor(a)
    desc_b = _create_tensor_descriptor(b)
    desc_c = _create_tensor_descriptor(c)
    desc_out = _create_tensor_descriptor(out)
    mode_a = _auto_create_mode(a, list(str_a))
    mode_b = _auto_create_mode(b, list(str_b))
    mode_c = _auto_create_mode(c, list(str_c))
    mode_out = _auto_create_mode(out, list(str_out))

    handler = cutensor._get_handle()
    operator = ctypes.c_void_p()
    plan = None
    status = _libcutensor.cutensorCreateContractionTrinary(
        handler.ptr, ctypes.byref(operator),
        desc_a.ptr, mode_a.data, op_a,
        desc_b.ptr, mode_b.data, op_b,
        desc_c.ptr, mode_c.data, op_c,
        desc_out.ptr, mode_out.data, op_d,
        desc_out.ptr, mode_out.data,
        _compute_descriptor_ptr(compute_desc, dtype))
    _check_cutensor_status(status)

    try:
        plan_pref = cutensor.create_plan_preference(
            algo=algo, jit_mode=jit_mode)
        ws_size = cutensor_backend.estimateWorkspaceSize(
            handler.ptr, operator.value, plan_pref.ptr, ws_pref)
        plan = cutensor_backend.createPlan(
            handler.ptr, operator.value, plan_pref.ptr, ws_size)
        workspace = cupy.empty(ws_size, dtype=np.int8)
        alpha = np.asarray(alpha, dtype=dtype)
        beta = np.asarray(beta, dtype=dtype)
        stream = cupy.cuda.get_current_stream()
        status = _libcutensor.cutensorContractTrinary(
            handler.ptr, plan,
            alpha.ctypes.data, a.data.ptr, b.data.ptr, c.data.ptr,
            beta.ctypes.data, out.data.ptr, out.data.ptr,
            workspace.data.ptr, ws_size, stream.ptr)
        _check_cutensor_status(status)
    finally:
        if plan is not None:
            cutensor_backend.destroyPlan(plan)
        cutensor_backend.destroyOperationDescriptor(operator.value)
    return out

import os
contract_engine = None
if cutensor is None:
    contract_engine = 'cupy'  # default contraction engine
contract_engine = os.environ.get('CONTRACT_ENGINE', contract_engine)

# override the 'contract' function if einsum is customized or cutensor is not found
if contract_engine is not None:
    einsum = None
    if contract_engine == 'opt_einsum':
        import opt_einsum
        einsum = opt_einsum.contract
    elif contract_engine == 'cuquantum':
        from cuquantum import contract as einsum # type: ignore
    elif contract_engine == 'cupy':
        einsum = cupy.einsum
    else:
        raise RuntimeError('unknown tensor contraction engine.')

    import warnings
    warnings.warn(f'using {contract_engine} as the tensor contraction engine.')
    def contract(pattern, a, b, alpha=1.0, beta=0.0, out=None):
        try:
            return _contract_einsum(pattern, a, b, alpha, beta, out, einsum)
        except cupy.cuda.memory.OutOfMemoryError:
            print('Out of memory error caused by cupy.einsum. '
                  'It is recommended to install cutensor to resolve this.')
            raise
else:
    def contract(pattern, a, b, alpha=1.0, beta=0.0, out=None):
        '''
        a wrapper for general tensor contraction
        pattern has to be a standard einsum notation
        '''
        return contraction(pattern, a, b, alpha, beta, out=out)


def contract_trinary(pattern, a, b, c, alpha=1.0, beta=0.0, out=None):
    """Einsum-style wrapper for a three-operand tensor contraction."""
    if contract_engine is not None or _libcutensor is None:
        fallback = einsum if contract_engine is not None else cupy.einsum
        return _contract_trinary_einsum(
            pattern, a, b, c, alpha, beta, out=out, einsum=fallback)
    return contraction_trinary(
        pattern, a, b, c, alpha, beta, out=out)
