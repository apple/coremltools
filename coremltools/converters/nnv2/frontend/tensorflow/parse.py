# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

from ...builtin_types import builtins
from .parsed_tf_node import ParsedTFNode
from tensorflow.core.framework.types_pb2 import DataType

import logging
import numpy as np

def parse_type(t):
    mapping = {
        DataType.DT_FLOAT: builtins.float,
        DataType.DT_DOUBLE: builtins.double,
        DataType.DT_INT32: builtins.int32,
        DataType.DT_UINT8: builtins.uint8,
        DataType.DT_INT16: builtins.int16,
        DataType.DT_INT8: builtins.int8,
        DataType.DT_STRING: builtins.str,
        DataType.DT_INT64: builtins.int64,
        DataType.DT_BOOL: builtins.bool,
        DataType.DT_UINT16: builtins.uint16,
        DataType.DT_UINT32: builtins.uint32,
        DataType.DT_UINT64: builtins.uint64
    }
    t = int(t)
    if t in mapping:
        return mapping[t]
    else:
        logging.warning("Type %d cannot be mapped", t)
        return None


def parse_shape(t):
    if t.unknown_rank:
        return None
    ret = [d.size for d in t.dim]
    return ret


def parse_tensor(t):
    typ = parse_type(t.dtype)
    shape = parse_shape(t.tensor_shape)

    retval = None
    if len(t.half_val) > 0:
        retval = np.array(t.half_val)
    elif len(t.float_val) > 0:
        retval = np.array(t.float_val)
    elif len(t.double_val) > 0:
        retval = np.array(t.double_val)
    elif len(t.int_val) > 0:
        retval = np.array(t.int_val)
    elif len(t.int64_val) > 0:
        retval = np.array(t.int64_val)
    elif len(t.bool_val) > 0:
        retval = np.array(t.bool_val)
    elif hasattr(t, 'uint32_val') and len(t.uint32_val) > 0:
        retval = np.array(t.uint32_val)
    elif hasattr(t, 'uint64_val') and len(t.uint64_val) > 0:
        retval = np.array(t.uint64_val)

    if not t.tensor_shape.unknown_rank and len(shape) == 0:
        retobj = typ()
        if retval is not None:
            retobj.val = retval[0]
    else:
        rettype = builtins.tensor(typ, tuple(shape))
        retobj = rettype()
        retobj.shape = shape
        if retval is not None:
            retobj.val = retval

    return retobj


def parse_string(s):
    if isinstance(s, bytes):
        return s.decode('utf-8')
    else:
        return s


def parse_list(t):
    if len(t.s) > 0:
        return list(parse_string(s) for s in t.s)
    elif len(t.i) > 0:
        return list(t.i)
    elif len(t.f) > 0:
        return list(t.f)
    elif len(t.b) > 0:
        return list(t.b)
    elif len(t.type) > 0:
        return list(parse_type(z) for z in t.type)
    elif len(t.shape) > 0:
        return list(parse_shape(z) for z in t.shape)
    elif len(t.tensor) > 0:
        return list(parse_tensor(z) for z in t.tensor)
    else:
        return []


def parse_func(f):
    return f.name


def parse_attr(attr):
    if attr.HasField('s'):
        return parse_string(attr.s)
    elif attr.HasField('i'):
        return attr.i
    elif attr.HasField('f'):
        return attr.f
    elif attr.HasField('b'):
        return attr.b
    elif attr.HasField('type'):
        return parse_type(attr.type)
    elif attr.HasField('shape'):
        return parse_shape(attr.shape)
    elif attr.HasField('tensor'):
        return parse_tensor(attr.tensor)
    elif attr.HasField('list'):
        return parse_list(attr.list)
    elif attr.HasField('func'):
        return parse_func(attr.func)
    elif attr.HasField('placeholder'):
        raise NotImplementedError("placeholder not yet implemented")
    raise ValueError('unintelligible TFNode attributes')
