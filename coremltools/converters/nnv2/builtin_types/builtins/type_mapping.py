# -*- coding: utf-8 -*-
from .type_bool import bool as builtins_bool
from .type_double import is_float, fp16 as builtins_fp16, fp32 as builtins_fp32, fp64 as builtins_fp64
from .type_list import is_list
from .type_int import is_int, int8 as builtins_int8, int16 as builtins_int16, int32 as builtins_int32, \
    int64 as builtins_int64, uint8 as builtins_uint8, uint16 as builtins_uint16, uint32 as builtins_uint32, \
    uint64 as builtins_uint64
from .type_str import str as builtins_str
from .type_unknown import unknown
import numpy as np
import six
from .get_type_info import get_type_info

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
    builtins_fp64: np.float64,
    builtins_str: np.str_
}

_BUILTINS_TO_STRINGS = {
    builtins_bool: 'bool',
    builtins_int8: 'i8',
    builtins_int16: 'i16',
    builtins_int32: 'i32',
    builtins_int64: 'i64',
    builtins_uint8: 'u8',
    builtins_uint16: 'u16',
    builtins_uint32: 'u32',
    builtins_uint64: 'u64',
    builtins_fp16: 'fp16',
    builtins_fp32: 'fp32',
    builtins_fp64: 'fp64',
    builtins_str: 'str'
}

_STRINGS_TO_BUILTINS = {v: k for k, v in _BUILTINS_TO_STRINGS.items()}

def string_to_builtin(s):
    """
    Given a str, return its corresponding builtin type.
    """
    return _STRINGS_TO_BUILTINS.get(s, None)

def builtin_to_string(builtin_type):
    """
    Given a builtin type, return its corresponding string representation.
    """
    return _BUILTINS_TO_STRINGS.get(builtin_type, None)

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
    # Circumvent the undesriable np type promotion:
    # >> np.promote_types(np.float32, np.int)
    # dtype('float64')
    if np.issubdtype(nptype1, np.float) and np.issubdtype(nptype2, np.int):
        nppromoted = nptype1
    elif np.issubdtype(nptype2, np.float) and np.issubdtype(nptype1, np.int):
        nppromoted = nptype2
    else:
        nppromoted = np.promote_types(nptype1, nptype2)
    return numpy_type_to_builtin_type(nppromoted)


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


# Converts a numpy type to its builtins equivalent.
# Supports both dtypes and numpy primitive types.
def numpy_type_to_builtin_type(nptype):
    if (type(nptype) == np.dtype):
        nptype = nptype.type

    if np.issubclass_(nptype, np.bool) or \
       np.issubclass_(nptype, np.bool_):
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
    elif np.issubclass_(nptype, np.float32) or \
         np.issubclass_(nptype, np.single):
        return builtins_fp32
    elif np.issubclass_(nptype, np.float64) or \
         np.issubclass_(nptype, np.double):
        return builtins_fp64
    elif np.issubclass_(nptype, six.string_types) or \
         np.issubclass_(nptype, np.string_) or \
         np.issubclass_(nptype, np.str_):
        return builtins_str
    else:
        raise TypeError("Unsupported numpy type: %s" % (nptype))


# Tries to get the equivalent builtin type of a
# numpy or python type.
def type_to_builtin_type(type):
    # Infer from numpy type if it is one
    if type.__module__ == np.__name__:
        return numpy_type_to_builtin_type(type)

    # Otherwise, try to infer from a few generic python types
    if np.issubclass_(type, bool):
        return builtins_bool
    elif np.issubclass_(type, six.integer_types):
        return builtins_int32
    elif np.issubclass_(type, six.string_types):
        return builtins_str
    elif np.issubclass_(type, float):
        return builtins_fp32
    else:
        raise TypeError("Could not determine builtin type for " \
                        + str(scalar))


def numpy_val_to_builtin_val(npval):
    if np.isscalar(npval):
        ret_type = type_to_builtin_type(type(npval))
        ret = ret_type()
        ret.val = npval
        return ret, ret_type
    else:
        builtintype = numpy_type_to_builtin_type(npval.dtype)
        from . import tensor as builtins_tensor
        ret_type = builtins_tensor(builtintype, npval.shape)
        ret = ret_type()
        ret.val = npval
        return ret, ret_type

def is_subtype(type1, type2):
    """
    Return True if type1 is a subtype of type2. False otherwise.
    """
    if type2 == unknown:
        return True  # any class is a subclass of unknown (None) type.
    if is_list(type2):
        return is_list(type1) and is_subtype(type1.T[0], type2.T[0])

    # simplistic handling of types is sufficient for now. Handling compatible
    # tensor shape requires using builtins.is_tensor_and_is_compatible
    return type1 == type2
