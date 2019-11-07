import numpy as np
import coremltools.proto.Program_pb2 as pm

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
        t_dim.size = s

def create_tensor_value(np_tensor):
    """
    Return TensorValue.
    """
    scalar_t = np_type_to_scalar_type(np_tensor.dtype)
    t_type = create_valuetype_tensor(np_tensor.shape, scalar_t)
    # update_tensortype(t_type, np_tensor.shape, scalar_t)
    val = pm.Value(type=t_type)
    t_val = val.immediateValue.tensor
    for x in np.nditer(np_tensor):
        t_val.ints.append(x)
    return val

def create_scalar_value(py_scalar):
    """
    Return Value (since there's no ScalarValue)
    """
    val = pm.Value()
    if isinstance(py_scalar, int):
        val.immediateValue.i = py_scalar
        val.type.scalarType = pm.ScalarType.INT32  # fix this.
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

def create_load_from_file_value(file_name, offset, dim, scalar_type):
    """
    Create a Value Type to store File Value
    """
    val = pm.Value(fileValue=pm.Value.FileValue(fileName=file_name, offset=offset),
                   type=create_valuetype_tensor(dim, scalar_type))
    return val