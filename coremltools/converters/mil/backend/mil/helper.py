#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import os
import re

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.types import builtin_to_proto_types
from coremltools.models.model import _WEIGHTS_DIR_NAME, _WEIGHTS_FILE_NAME
import coremltools.proto.FeatureTypes_pb2 as ft
import coremltools.proto.MIL_pb2 as pm

from coremltools.converters.mil.mil.types import (
    type_to_builtin_type,
    numpy_type_to_builtin_type,
    builtin_to_string
)
from coremltools.converters.mil.backend.nn.op_mapping import to_py_type


def create_valuetype_scalar(data_type):
    """
    Return pm.ValueType with DataType set
    """
    v_type = pm.ValueType()
    update_tensortype(v_type.tensorType, (), data_type)
    return v_type


def update_listtype(l_type, length, elem_shape, dtype):
    """
    Update in-place of l_type (ListType) to length and type.
    """

    elem_type = create_valuetype_tensor(elem_shape, dtype)
    l_type.type.CopyFrom(elem_type)

    l_dim = l_type.length
    set_proto_dim(l_dim, length)

def create_valuetype_list(length, elem_shape, dtype):
    """
    Return pm.ValueType with List (ListType) set.
    length: length of list (int)
    """
    v_type = pm.ValueType()
    update_listtype(v_type.listType, length, elem_shape, dtype)
    return v_type

def create_valuetype_dict(key_type, value_type):
    """
    Return pm.ValueType with dict (dictionaryType) set
    """
    v_type = pm.ValueType()
    v_type.dictionaryType.keyType.CopyFrom(types_to_proto(key_type))
    v_type.dictionaryType.valueType.CopyFrom(types_to_proto(value_type))
    return v_type


def create_valuetype_tensor(shape, data_type):
    """
    Return pm.ValueType with tensor (TensorType) set.
    shape: list of ints
    """
    v_type = pm.ValueType()
    update_tensortype(v_type.tensorType, shape, data_type)
    return v_type


def set_proto_dim(proto_dim, dim):
    if isinstance(dim, (int, np.integer)):
        proto_dim.constant.size = dim
    else:
        dim_str = str(dim)
        if len(dim_str) > 0:
            if dim_str[0] == "*" or (len(dim_str) >= 3 and dim_str[0:3] == "..."):
                proto_dim.unknown.variadic = True
                return
            proto_dim.unknown.variadic = False


def update_tensortype(t_type, shape, data_type):
    """
    Update in-place of t_type (TensorType) to shape and data_type.
    """
    t_type.dataType = data_type
    t_type.rank = len(shape)
    t_type.ClearField("dimensions")
    for s in shape:
        t_dim = t_type.dimensions.add()
        set_proto_dim(t_dim, s)

def _tensor_field_by_type(tensor_val, builtin_type):
    if builtin_type == types.bool:
        return tensor_val.bools.values
    elif types.is_int(builtin_type):
        if (builtin_type == types.int64 or builtin_type == types.uint64):
            return tensor_val.longInts.values
        return tensor_val.ints.values
    elif types.is_float(builtin_type):
        if (builtin_type == types.fp64):
            return tensor_val.doubles.values
        elif (builtin_type == types.fp32):
            return tensor_val.floats.values
        elif (builtin_type == types.fp16):
            return tensor_val.bytes.values
        else:
            raise TypeError(
                "Unsupported float dtype for MIL proto serialization: {}".format(builtin_to_string(builtin_type)))
    elif builtin_type == types.str:
        return tensor_val.strings.values
    else:
        raise NotImplementedError("Unimplemented tensor type for: " + str(builtin_type))

def _set_empty_tensor_field_by_type(tensor_val, builtin_type):
    if builtin_type == types.bool:
        tensor_val.bools.SetInParent()
    elif types.is_int(builtin_type):
        if (builtin_type == types.int64 or builtin_type == types.uint64):
            tensor_val.longInts.SetInParent()
        else:
            tensor_val.ints.SetInParent()
    elif types.is_float(builtin_type):
        if (builtin_type == types.fp64):
            tensor_val.doubles.SetInParent()
        elif (builtin_type == types.fp32):
            tensor_val.floats.SetInParent()
        elif (builtin_type == types.fp16):
            tensor_val.bytes.SetInParent()
        else:
            raise TypeError("Unsupported float dtype for MIL proto serialization: {}".format(builtin_to_string(builtin_type)))
    elif builtin_type == types.str:
        tensor_val.strings.SetInParent()
    else:
        raise NotImplementedError("Unimplemented tensor type for: " + str(builtin_type))

def create_tensor_value(np_tensor):
    """
    Return TensorValue.
    """
    builtin_type = numpy_type_to_builtin_type(np_tensor.dtype)

    value_type = create_valuetype_tensor(np_tensor.shape, types_to_proto_primitive(builtin_type))
    val = pm.Value(type=value_type)
    t_val = val.immediateValue.tensor

    # Copy the tensor values from the input tensor
    t_field = _tensor_field_by_type(t_val, builtin_type)

    if 0 not in np_tensor.shape:
        if builtin_type == types.str:
            for x in np.nditer(np_tensor):
                t_field.append(x.encode("utf-8"))
        elif builtin_type == types.fp16:
            bytevals = bytes()
            for x in np_tensor.flatten():
                bytevals += to_py_type(x)
            val.immediateValue.tensor.bytes.values = bytevals
        else:
            for x in np_tensor.flatten():
                t_field.append(to_py_type(x))
    else:  # This is an "empty" tensor (tensor with a dimension being size 0)
        _set_empty_tensor_field_by_type(t_val, builtin_type)
    return val


def create_scalar_value(py_scalar):
    """
    Return TensorValue (since there's no ScalarValue)
    """
    # Create the "scalar" (rank 0) tensor
    builtin_type = type_to_builtin_type(type(py_scalar))
    value_type = create_valuetype_scalar(types_to_proto_primitive(builtin_type))
    val = pm.Value(type=value_type)
    t_val = val.immediateValue.tensor

    # Set the tensor value
    t_field = _tensor_field_by_type(t_val, builtin_type)
    if builtin_type == types.fp16:
        val.immediateValue.tensor.bytes.values = to_py_type(py_scalar)
    else:
        if builtin_type == types.str:
            py_scalar = py_scalar.encode("utf-8")
        t_field.append(to_py_type(py_scalar))

    return val


def create_tuple_value(py_tuple):
    """
    Return type of Tuple
    """
    tp_val = pm.TupleValue()
    for t in py_tuple:
        item_val = tp_val.values.add()
        item_type = item_val.type  # ValueType
        if isinstance(t, int):
            v = create_scalar_value(t)
            item_val.immediateValue.i = t
            item_type = v.type
        elif isinstance(t, np.ndarray):
            v = create_tensor_value(t)
            item_val.immediateValue.tensor.CopyFrom(v.immediateValue.tensor)
            item_type.tensorType.CopyFrom(v.type.tensorType)
        else:
            raise NotImplementedError()
    return tp_val

def create_list_scalarvalue(py_list, np_type):
    """
    Return a Value of type List, which holds scalar values
    """
    builtin_type = numpy_type_to_builtin_type(np_type)
    value_type = create_valuetype_list(length=len(py_list),
                                       elem_shape=(),
                                       dtype=types_to_proto_primitive(builtin_type))
    val = pm.Value(type=value_type)

    list_val = val.immediateValue.list
    for v in py_list:
        item_val = list_val.values.add()
        item_val.CopyFrom(create_scalar_value(v))

    return val

def create_file_value_tensor(file_name, offset, dim, data_type):
    """
    Create a Value Type to store File Value
    """
    val = pm.Value(
        blobFileValue=pm.Value.BlobFileValue(fileName=file_name, offset=offset),
        type=create_valuetype_tensor(dim, data_type),
    )
    return val


def types_to_proto_primitive(valuetype):
    if valuetype not in builtin_to_proto_types:
        raise ValueError(
            "Unknown type {} to map from SSA types to Proto types".format(
                valuetype)
        )
    return builtin_to_proto_types[valuetype]


def types_to_proto(valuetype):
    if types.is_tensor(valuetype):
        primitive = types_to_proto_primitive(valuetype.get_primitive())
        return create_valuetype_tensor(valuetype.get_shape(), primitive)
    elif types.is_tuple(valuetype):
        v_type = pm.ValueType()
        t_type = v_type.tupleType
        for t in valuetype.T:
            new_v_type = t_type.types.add()
            new_v_type.CopyFrom(types_to_proto(t))
        return v_type
    elif types.is_list(valuetype):
        elem = valuetype.T[0]
        length = valuetype.T[1]
        if types.is_tensor(elem):
            dtype = types_to_proto_primitive(elem.get_primitive())
            elem_shape = elem.get_shape()
        elif types.is_scalar(elem):
            dtype = types_to_proto_primitive(valuetype)
            elem_shape = ()
        elif types.is_str(elem):
            dtype = types_to_proto_primitive(elem)
            elem_shape = ()
        else:
            raise NotImplementedError("Only list of either tensors or scalars supported. "
                                      "Got element of type {}".format(elem.__type_info__()))
        return create_valuetype_list(length=length, elem_shape=elem_shape, dtype=dtype)
    elif types.is_dict(valuetype):
        return create_valuetype_dict(valuetype.T[0], valuetype.T[1])
    else:
        return create_valuetype_scalar(types_to_proto_primitive(valuetype))


def create_file_value(output_var, blob_writer):
    if output_var.val.dtype.kind == 'f' and output_var.val.dtype.itemsize == 4:
        offset = blob_writer.write_float_data(output_var.val.flatten())
    elif output_var.val.dtype.kind == 'f' and output_var.val.dtype.itemsize == 2:
        output_var_fp16_to_bytes_to_uint16 = np.frombuffer(output_var.val.flatten().tobytes(), np.uint16)
        offset = blob_writer.write_fp16_data(output_var_fp16_to_bytes_to_uint16)
    else:
        raise TypeError("Unsupported type, {}, for net buffer serialization.".format(output_var.val.dtype))

    return create_file_value_tensor(
        file_name=os.path.join(os.path.join('@model_path', _WEIGHTS_DIR_NAME), _WEIGHTS_FILE_NAME),
        offset=offset,
        dim=output_var.val.shape,
        data_type=types_to_proto_primitive(output_var.sym_type.get_primitive()),
    )

def create_immediate_value(var):
    if types.is_tensor(var.sym_type):
        return create_tensor_value(var.val)
    elif types.is_list(var.sym_type):
        if var.elem_type == types.str:
            return create_list_scalarvalue(var.val, np.str)
        elif var.elem_type == types.int64:
            return create_list_scalarvalue(var.val, np.int64)
        else:
            raise NotImplementedError("List element type, {}, not supported yet.".format(var.sym_type.__type_info__()))
    else:
        return create_scalar_value(var.val)

def cast_to_framework_io_dtype(var, is_output):
    if var.dtype == types.fp32:
        return ft.ArrayFeatureType.ArrayDataType.FLOAT32
    elif var.dtype == types.int32:
        return ft.ArrayFeatureType.ArrayDataType.INT32
    else:
        ioname = "Output " if is_output else "Input "
        ioname2 = "outputs" if is_output else "inputs"
        raise NotImplementedError(ioname + var.name + " has data type " + builtin_to_string(var.dtype) + \
                                  ". ML Program models only support fp32 and int32 " + ioname2 + ".")

