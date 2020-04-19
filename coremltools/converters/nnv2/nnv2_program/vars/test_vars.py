from coremltools.converters.nnv2.nnv2_program.vars import ScalarVar, TensorVar
from coremltools.converters.nnv2.nnv2_program.program import Operation
from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2.builtin_types.builtins import int as bint, tensor as btensor
from coremltools.converters.nnv2.builtin_types.builtins import numpy_type_to_builtin_type

from functools import reduce
from sympy import symbols

import numpy as np
import mock
import pytest

op = mock.Mock()
op.__class__ = Operation

def test_scalar_var():
    # Concrete value
    x = ScalarVar("x", op, bint, bint(8))
    assert x.type == bint
    assert x.val.__class__ == bint
    assert x.val == bint(8)
    assert x.get_value() == 8
    assert x.get_value(allow_symbolic=True) == 8
    assert x.get_value(allow_symbolic=False) == 8
    assert x.has_symbolic() == False
    assert x.__str__() == "%x: (i64)*"

    # Missing value
    x = ScalarVar("x", op, bint, val=None)
    assert x.type == bint
    assert x.val == None
    assert x.get_value() == None
    assert x.get_value(allow_symbolic=False) == None
    assert x.get_value(allow_symbolic=True) == None
    assert x.has_symbolic() == False
    assert x.__str__() == "%x: (i64)"

    # Symbolic value
    x = ScalarVar("x", op, bint, val=symbols("x1"))
    assert x.has_symbolic()
    assert x.val == symbols("x1")
    assert x.get_value() == None
    assert x.get_value(allow_symbolic=False) == None
    assert x.get_value(allow_symbolic=True) == symbols("x1")
    assert x.__str__() == "%x: (i64)^"

    # Failure cases: int is not a builtin primitive type
    with pytest.raises(TypeError):
        x = ScalarVar("x", op, int, val=bint(8))

    # Failure cases: value is builtins.float but type is builtins.int
    with pytest.raises(TypeError):
        x = ScalarVar("x", op, bint, val=builtins.float(8.))


def _make_builtin_tensor(np_dtype, shape, symbolic=False):
    if symbolic:
        np_tensor = np.zeros(shape, dtype=object)
        np_tensor[0][0] = symbols("d")
    elif shape == ():
        # make a 0d tensor
        np_tensor = np.asarray(0).astype(np_dtype)
    else:
        size = reduce(lambda x, y: x*y, shape)
        np_tensor = np.asarray([range(size)]).reshape(shape).astype(np_dtype)
    dtype = numpy_type_to_builtin_type(np_dtype)
    tensor_t = btensor(dtype, shape)
    tensor_val = tensor_t()
    tensor_val.val = np_tensor
    return tensor_t, tensor_val, np_tensor

def test_tensor_var():
    # Concrete value
    np_dtype = np.float32
    shape = (2,2)
    tensor_t, tensor_val, np_tensor = _make_builtin_tensor(np_dtype, shape)
    x = TensorVar("x", op, tensor_t, tensor_val)
    assert x.type == tensor_t
    assert x.val == tensor_val
    assert x.shape == shape
    assert x.dtype == tensor_t.get_primitive()
    assert x.rank == 2
    assert x.__str__() == "%x: (2, 2, fp32)*"

    assert x.has_symbolic() == False
    np.testing.assert_array_equal(x.get_value(), np_tensor)
    np.testing.assert_array_equal(x.get_value(allow_symbolic=False), np_tensor)
    np.testing.assert_array_equal(x.get_value(allow_symbolic=True), np_tensor)

    # Missing value
    np_dtype = np.float32
    shape = (2,2)
    tensor_t = btensor(numpy_type_to_builtin_type(np_dtype), shape)
    x = TensorVar("x", op, tensor_t, val=None)
    assert x.type == tensor_t
    assert x.val == None
    assert x.shape == shape
    assert x.dtype == tensor_t.get_primitive()
    assert x.rank == 2
    assert x.__str__() == "%x: (2, 2, fp32)"

    assert x.has_symbolic() == False
    assert x.get_value() == None
    assert x.get_value(allow_symbolic=True) == None
    assert x.get_value(allow_symbolic=False) == None

    # Missing value with variadic rank
    np_dtype = np.float32
    shape = (3, symbols("*L"))
    tensor_t = btensor(numpy_type_to_builtin_type(np_dtype), shape)
    x = TensorVar("x", op, tensor_t, val=None)
    assert x.type == tensor_t
    assert x.val == None
    assert x.shape == shape
    assert x.dtype == tensor_t.get_primitive()
    assert x.rank == None
    assert x.__str__() == "%x: (3, *L, fp32)"

    # Symbolic value
    # Note: Symbolic values are treated as int32, more detail in builtins/type_mapping.py
    np_dtype = np.int32
    shape = (2,2)
    tensor_t, tensor_val, np_tensor = _make_builtin_tensor(np_dtype, shape,
                                                           symbolic=True)
    x = TensorVar("x", op, tensor_t, val=tensor_val)
    assert x.type == tensor_t
    assert x.val == tensor_val
    assert x.has_symbolic() == True
    assert x.shape == shape
    assert x.dtype == tensor_t.get_primitive()
    assert x.rank == 2
    assert x.__str__() == "%x: (2, 2, i32)^"

    assert x.has_symbolic() == True
    assert x.get_value() == None
    assert x.get_value(allow_symbolic=False) == None
    np.testing.assert_array_equal(x.get_value(allow_symbolic=True), np_tensor)

    # Zero D tensor
    np_dtype = np.float32
    shape = ()
    tensor_t, tensor_val, np_tensor = _make_builtin_tensor(np_dtype, shape)
    x = TensorVar("x", op, tensor_t, tensor_val)
    assert x.type == tensor_t
    assert x.val == tensor_val
    assert x.shape == ()
    assert x.dtype == builtins.fp32
    assert x.rank == 0
    assert x.__str__() == "%x: (fp32)*"
