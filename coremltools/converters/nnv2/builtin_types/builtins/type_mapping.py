# -*- coding: utf-8 -*-
from .type_bool import bool as builtins_bool
from .type_double import is_float, fp16 as builtins_fp16, fp32 as builtins_fp32, fp64 as builtins_fp64
from .type_int import is_int, int8 as builtins_int8, int16 as builtins_int16, int32 as builtins_int32, \
    int64 as builtins_int64, uint8 as builtins_uint8, uint16 as builtins_uint16, uint32 as builtins_uint32, \
    uint64 as builtins_uint64
from .type_str import str as builtins_str
import numpy as np
import six
from .get_type_info import get_type_info

_NPTYPES_TO_BUILTINS = {
    np.dtype(np.bool_): builtins_bool,
    np.dtype(np.int8): builtins_int8,
    np.dtype(np.int16): builtins_int16,
    np.dtype(np.int32): builtins_int32,
    np.dtype(np.int64): builtins_int64,
    np.dtype(np.uint8): builtins_uint8,
    np.dtype(np.uint16): builtins_uint16,
    np.dtype(np.uint32): builtins_uint32,
    np.dtype(np.uint64): builtins_uint64,
    np.dtype(np.float16): builtins_fp16,
    np.dtype(np.float32): builtins_fp32,
    np.dtype(np.float64): builtins_fp64
}

_BUILTINS_TO_NPTYPES = {
    builtins_bool: np.bool_,
    builtins_int8: np.int8,
    builtins_int16: np.int16,
    builtins_int32: np.int32,
    builtins_int64: np.int64,
    builtins_uint8: np.uint8,
    builtins_uint16: np.uint16,
    builtins_uint32: np.uint32,
    builtins_uint64: np.uint64,
    builtins_fp16: np.float16,
    builtins_fp32: np.float32,
    builtins_fp64: np.float64
}


def builtin_from_nptype(nptype):
    """
    Given a numpy dtype, return its corresponding primitive builtin type.
    """
    return _NPTYPES_TO_BUILTINS.get(np.dtype(nptype), None)


def nptype_from_builtin(btype):
    """
    Given a Nitro builtin type, return its corresponding Numpy dtype.
    """
    return _BUILTINS_TO_NPTYPES.get(btype, None)


def promote_types(dtype1, dtype2):
    """
    Get the smallest type to which the given scalar types can be cast.

    Args:
        dtype1 (apple_nitro.builtin):
        dtype2 (apple_nitro.builtin):
    
    Returns:
        A Nitro builtin datatype or None.
    """
    nptype1 = nptype_from_builtin(dtype1)
    nptype2 = nptype_from_builtin(dtype2)
    nppromoted = np.promote_types(nptype1, nptype2)
    return builtin_from_nptype(nppromoted)


def is_primitive(btype):
    """
    Is the indicated Nitro builtin type a primitive?
    """
    return btype is builtins_bool or btype is builtins_str or is_float(
        btype) or is_int(btype)


def is_scalar(btype):
    """
    Is the given builtin type a scalar integer, float, or boolean?
    """
    return btype is builtins_bool or is_int(btype) or is_float(btype)


def is_tensor(tensor_type):
    if tensor_type is None:
        return False
    return get_type_info(tensor_type).name == 'tensor'


def is_str(t):
    if t is None:
        return False
    return get_type_info(t).name == 'str'


def is_tuple(t):
    if t is None:
        return False
    return get_type_info(t).name == 'tuple'


def is_builtin(t):
    return is_scalar(t) or is_tensor(t) or is_str(t) or is_tuple(t)


def numpy_primitive_type_to_builtin_type(nptype):
    if np.issubclass_(nptype, np.bool) or np.issubclass_(nptype, np.bool_):
        # numpy as 2 bool types it looks like. what is the difference?
        return builtins_bool
    elif np.issubclass_(nptype, np.int8):
        return builtins_int8
    elif np.issubclass_(nptype, np.int16):
        return builtins_int16
    elif np.issubclass_(nptype, np.int32):
        return builtins_int32
    elif np.issubclass_(nptype, np.int64):
        return builtins_int64
    elif np.issubclass_(nptype, np.uint8):
        return builtins_int8
    elif np.issubclass_(nptype, np.uint16):
        return builtins_int16
    elif np.issubclass_(nptype, np.uint32):
        return builtins_int32
    elif np.issubclass_(nptype, np.uint64):
        return builtins_int64
    elif np.issubclass_(nptype, np.int):
        # Catch all int
        return builtins_int32
    elif np.issubclass_(nptype, np.object_):
        # symbolic shape is considered int32
        return builtins_int32
    elif np.issubclass_(nptype, np.float16):
        return builtins_fp16
    elif np.issubclass_(nptype, np.float32):
        return builtins_fp32
    elif np.issubclass_(nptype, np.float64):
        return builtins_fp64
    elif issubclass(nptype, six.string_types):
        return builtins_str
    else:
        raise TypeError("Not supported numpy type: %s" % (nptype))


def numpy_val_to_builtin_val(npval):
    if np.isscalar(npval):
        ret_type = numpy_primitive_type_to_builtin_type(type(npval))
        ret = ret_type()
        ret.val = npval
        return ret, ret_type
    else:
        builtintype = numpy_primitive_type_to_builtin_type(npval.dtype.type)
        from . import tensor as builtins_tensor
        ret_type = builtins_tensor(builtintype, npval.shape)
        ret = ret_type()
        ret.val = npval
        return ret, ret_type