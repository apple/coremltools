import six
import numpy as np
import coremltools.proto.Program_pb2 as pm

from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2.nnv2_program.program.type_utils import builtin_to_proto_types

def np_type_to_scalar_type(np_type):
    if np_type == np.int32:
        return pm.ScalarType.INT32
    elif np_type == np.float32:
        return pm.ScalarType.FLOAT32
    raise NotImplementedError()

def create_scalartype(scalar_type):
    """
    Return pm.ValueType with ScalarType set
    """
    v_type = pm.ValueType()
    v_type.scalarType = scalar_type
    return v_type

def create_valuetype_tensor(shape, scalar_type):
    """
    Return pm.ValueType with tensor (TensorType) set.
    shape: list of ints
    """
    v_type = pm.ValueType()
    update_tensortype(v_type.tensorType, shape, scalar_type)
    return v_type

def update_tensortype(t_type, shape, scalar_type):
    """
    Update in-place of t_type (TensorType) to shape and scalar_type.
    """
    t_type.scalarType = scalar_type
    t_type.rank = len(shape)
    t_type.ClearField("dimension")
    for s in shape:
        t_dim = t_type.dimension.add()
        if isinstance(s, (six.integer_types, np.integer)):
            t_dim.size = s
        else:
            t_dim.symbol = str(s)

def create_tensor_value(np_tensor):
    """
    Return TensorValue.
    """
    scalar_t = np_type_to_scalar_type(np_tensor.dtype)
    t_type = create_valuetype_tensor(np_tensor.shape, scalar_t)
    # update_tensortype(t_type, np_tensor.shape, scalar_t)
    val = pm.Value(type=t_type)
    t_val = val.immediateValue.tensor
    if np_tensor.dtype in [np.int64, np.int32]:
        for x in np.nditer(np_tensor):
            t_val.ints.append(x)
    elif np_tensor.dtype == np.float32:
        for x in np.nditer(np_tensor):
            t_val.floats.append(x)
    else:
        raise NotImplementedError()

    return val

def create_scalar_value(py_scalar):
    """
    Return Value (since there's no ScalarValue)
    """
    val = pm.Value()
    if isinstance(py_scalar, np.bool):
        val.immediateValue.b = py_scalar
        val.type.scalarType = pm.ScalarType.BOOL
    elif isinstance(py_scalar, (np.integer, six.integer_types)):
        val.immediateValue.i = py_scalar
        if isinstance(py_scalar, np.int8):
            val.type.scalarType = pm.ScalarType.INT8
        elif isinstance(py_scalar, np.int16):
            val.type.scalarType = pm.ScalarType.INT16
        elif isinstance(py_scalar, np.int64):
            val.type.scalarType = pm.ScalarType.INT64
        elif isinstance(py_scalar, np.uint8):
            val.type.scalarType = pm.ScalarType.UINT8
        elif isinstance(py_scalar, np.uint16):
            val.type.scalarType = pm.ScalarType.UINT16
        elif isinstance(py_scalar, np.uint32):
            val.type.scalarType = pm.ScalarType.UINT32
        elif isinstance(py_scalar, np.uint64):
            val.type.scalarType = pm.ScalarType.UINT64
        else: # default is INT32
            val.type.scalarType = pm.ScalarType.INT32
    elif isinstance(py_scalar, (float, np.float, np.float16,
        np.float32, np.float64)):
        val.immediateValue.f = py_scalar
        if isinstance(py_scalar, np.float16):
            val.type.scalarType = pm.ScalarType.FLOAT16
        elif isinstance(py_scalar, np.float64):
            val.type.scalarType = pm.ScalarType.FLOAT64
        else: # default is FLOAT32
            val.type.scalarType = pm.ScalarType.FLOAT32
    elif isinstance(py_scalar, six.string_types):
        val.immediateValue.s = py_scalar.encode('utf-8')
        val.type.scalarType = pm.ScalarType.STRING
    else:
        raise NotImplementedError("value: " + str(py_scalar))
    return val

def create_tuple_value(py_tuple):
    """
    Return type of Tuple
    """
    tp_val = pm.TupleValue()
    for t in py_tuple:
        item_val = tp_val.value.add()
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

def create_file_value_tensor(file_name, offset, dim, scalar_type):
    """
    Create a Value Type to store File Value
    """
    val = pm.Value(fileValue=pm.Value.FileValue(fileName=file_name, offset=offset),
                   type=create_valuetype_tensor(dim, scalar_type))
    return val

def builtins_to_proto_primitive(valuetype):
    if valuetype not in builtin_to_proto_types:
        raise ValueError("Unknown type {} to map from SSA builtins to Proto types".format(primitive))
    return builtin_to_proto_types[valuetype]

def builtins_to_proto(valuetype):
    if builtins.is_tensor(valuetype):
        primitive = builtins_to_proto_primitive(valuetype.get_primitive())
        return create_valuetype_tensor(valuetype.get_shape(), primitive)
    elif builtins.is_tuple(valuetype):
        v_type = pm.ValueType()
        t_type = v_type.tupleType
        for t in valuetype.T:
            new_v_type = t_type.values.add()
            new_v_type.CopyFrom(builtins_to_proto(t))
        return v_type
    elif builtins.is_list(valuetype):
        # TODO rdar://57897440
        raise NotImplementedError()
    else:
        return create_scalartype(builtins_to_proto_primitive(valuetype))

