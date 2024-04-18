#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import numpy as np

from coremltools import proto
from coremltools.converters.mil.mil import types

# For immediate values, those types are stored in bytes (MIL parser reads those types from bytes).
IMMEDIATE_VALUE_TYPES_IN_BYTES = (types.fp16, types.int8, types.uint8, types.uint32)


def create_valuetype_scalar(data_type):
    """
    Return proto.MIL_pb2.ValueType with DataType set
    """
    v_type = proto.MIL_pb2.ValueType()
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
    Return proto.MIL_pb2.ValueType with List (ListType) set.
    length: length of list (int)
    """
    v_type = proto.MIL_pb2.ValueType()
    update_listtype(v_type.listType, length, elem_shape, dtype)
    return v_type

def create_valuetype_tensor(shape, data_type):
    """
    Return proto.MIL_pb2.ValueType with tensor (TensorType) set.
    shape: list of ints
    """
    v_type = proto.MIL_pb2.ValueType()
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
    """
    Pick the field based on the builtin_type.

    The field is defined in TensorValue in ``mlmodel/format/MIL.proto``.
    The picked field need to be consistent with how it will be read by MIL.
    For example, int8 is serialized to ``bytes`` field while int16 is serialized to ``ints`` field.
    """
    if builtin_type == types.bool:
        return tensor_val.bools.values
    elif types.is_int(builtin_type):
        if builtin_type == types.int64 or builtin_type == types.uint64:
            return tensor_val.longInts.values
        if builtin_type in IMMEDIATE_VALUE_TYPES_IN_BYTES:
            return tensor_val.bytes.values
        if builtin_type == types.int16 or builtin_type == types.uint16:
            # TODO (rdar://111797203): Serialize to byte after MIL changes to read from byte field.
            return tensor_val.ints.values
        return tensor_val.ints.values
    elif types.is_float(builtin_type):
        if builtin_type == types.fp64:
            return tensor_val.doubles.values
        elif builtin_type == types.fp32:
            return tensor_val.floats.values
        elif builtin_type == types.fp16:
            return tensor_val.bytes.values
        else:
            raise TypeError(
                "Unsupported float dtype for MIL proto serialization: {}".format(
                    types.builtin_to_string(builtin_type)
                )
            )
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
        elif builtin_type in IMMEDIATE_VALUE_TYPES_IN_BYTES:
            tensor_val.bytes.SetInParent()
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
            raise TypeError(
                "Unsupported float dtype for MIL proto serialization: {}".format(
                    types.builtin_to_string(builtin_type)
                )
            )
    elif builtin_type == types.str:
        tensor_val.strings.SetInParent()
    else:
        raise NotImplementedError("Unimplemented tensor type for: " + str(builtin_type))

def create_tensor_value(np_tensor):
    """
    Return TensorValue.
    """
    builtin_type = types.numpy_type_to_builtin_type(np_tensor.dtype)

    value_type = create_valuetype_tensor(np_tensor.shape, types_to_proto_primitive(builtin_type))
    val = proto.MIL_pb2.Value(type=value_type)
    t_val = val.immediateValue.tensor

    # Copy the tensor values from the input tensor
    t_field = _tensor_field_by_type(t_val, builtin_type)

    if 0 not in np_tensor.shape:
        if builtin_type == types.str:
            for x in np.nditer(np_tensor):
                t_field.append(x.encode("utf-8"))
        elif builtin_type in IMMEDIATE_VALUE_TYPES_IN_BYTES:
            val.immediateValue.tensor.bytes.values = types.type_mapping.np_val_to_py_type(np_tensor)
        else:
            for x in np_tensor.flatten():
                t_field.append(types.type_mapping.np_val_to_py_type(x))
    else:  # This is an "empty" tensor (tensor with a dimension being size 0)
        _set_empty_tensor_field_by_type(t_val, builtin_type)
    return val


def create_scalar_value(py_scalar):
    """
    Return TensorValue (since there's no ScalarValue)
    """
    # Create the "scalar" (rank 0) tensor
    builtin_type = types.type_to_builtin_type(type(py_scalar))
    value_type = create_valuetype_scalar(types_to_proto_primitive(builtin_type))
    val = proto.MIL_pb2.Value(type=value_type)
    t_val = val.immediateValue.tensor

    # Set the tensor value
    t_field = _tensor_field_by_type(t_val, builtin_type)
    if builtin_type in IMMEDIATE_VALUE_TYPES_IN_BYTES:
        # Serialize to bytes because MIL read them from the "bytes" field in TensorValue.
        val.immediateValue.tensor.bytes.values = types.type_mapping.np_val_to_py_type(py_scalar)
    else:
        if builtin_type == types.str:
            py_scalar = py_scalar.encode("utf-8")
        t_field.append(types.type_mapping.np_val_to_py_type(py_scalar))

    return val


def create_tuple_value(py_tuple):
    """
    Return type of Tuple
    """
    tp_val = proto.MIL_pb2.TupleValue()
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
    builtin_type = types.numpy_type_to_builtin_type(np_type)
    value_type = create_valuetype_list(length=len(py_list),
                                       elem_shape=(),
                                       dtype=types_to_proto_primitive(builtin_type))
    val = proto.MIL_pb2.Value(type=value_type)

    list_val = val.immediateValue.list
    for v in py_list:
        item_val = list_val.values.add()
        item_val.CopyFrom(create_scalar_value(v))

    return val

def create_file_value_tensor(file_name, offset, dim, data_type):
    """
    Create a Value Type to store File Value
    """
    val = proto.MIL_pb2.Value(
        blobFileValue=proto.MIL_pb2.Value.BlobFileValue(fileName=file_name, offset=offset),
        type=create_valuetype_tensor(dim, data_type),
    )
    return val


def types_to_proto_primitive(valuetype):
    if valuetype not in types.BUILTIN_TO_PROTO_TYPES:
        additional_error_msg = ""
        if valuetype in (types.complex64, types.complex128):
            additional_error_msg = (
                "(MIL doesn't support complex data as model's output, please extract real and "
                "imaginary parts explicitly.) "
            )
        raise ValueError(
            f"Unknown map from SSA type {valuetype} to Proto type. {additional_error_msg}"
        )
    return types.BUILTIN_TO_PROTO_TYPES[valuetype]

def _get_offset_by_writing_data(output_var, blob_writer):
    if output_var.val.dtype.kind == 'f' and output_var.val.dtype.itemsize == 4:
        offset = blob_writer.write_float_data(np.ascontiguousarray(output_var.val.flatten()))
    elif output_var.val.dtype.kind == "f" and output_var.val.dtype.itemsize == 2:
        output_var_fp16_to_bytes_to_uint16 = np.frombuffer(
            output_var.val.flatten().tobytes(), np.uint16
        )
        offset = blob_writer.write_fp16_data(
            np.ascontiguousarray(output_var_fp16_to_bytes_to_uint16)
        )
    elif output_var.val.dtype.kind == "u" and output_var.val.dtype.itemsize == 1:
        offset = blob_writer.write_uint8_data(np.ascontiguousarray(output_var.val.flatten()))
    elif output_var.val.dtype.kind == "i" and output_var.val.dtype.itemsize == 1:
        offset = blob_writer.write_int8_data(np.ascontiguousarray(output_var.val.flatten()))
    elif output_var.val.dtype.kind == "u" and output_var.val.dtype.itemsize == 2:
        offset = blob_writer.write_uint16_data(np.ascontiguousarray(output_var.val.flatten()))
    elif output_var.val.dtype.kind == "i" and output_var.val.dtype.itemsize == 2:
        offset = blob_writer.write_int16_data(np.ascontiguousarray(output_var.val.flatten()))
    else:
        raise TypeError("Unsupported type, {}, for net buffer serialization.".format(output_var.val.dtype))

    return offset

def create_immediate_value(var):
    if types.is_tensor(var.sym_type):
        return create_tensor_value(var.val)
    elif types.is_list(var.sym_type):
        if var.elem_type == types.str:
            return create_list_scalarvalue(var.val, str)
        elif var.elem_type == types.int64:
            return create_list_scalarvalue(var.val, np.int64)
        else:
            raise NotImplementedError("List element type, {}, not supported yet.".format(var.sym_type.__type_info__()))
    else:
        return create_scalar_value(var.val)

def cast_to_framework_io_dtype(var, is_output):
    if var.dtype == types.fp32:
        return proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.FLOAT32
    elif var.dtype == types.int32:
        return proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.INT32
    elif var.dtype == types.fp16:
        return proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.FLOAT16
    else:
        ioname = "Output " if is_output else "Input "
        ioname2 = "outputs" if is_output else "inputs"
        raise NotImplementedError(
            ioname
            + var.name
            + " has data type "
            + types.builtin_to_string(var.dtype)
            + ". ML Program models only support fp32 and int32 "
            + ioname2
            + "."
        )
