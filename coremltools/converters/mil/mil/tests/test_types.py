#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pytest

from coremltools import ImageType, StateType, TensorType
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.types import type_mapping
from coremltools.optimize.coreml import _utils as optimize_utils


class TestTypes:
    def test_sub_byte_type(self):
        assert types.is_int(types.int4)
        assert types.is_int(types.uint1)
        assert types.is_int(types.uint2)
        assert types.is_int(types.uint3)
        assert types.is_int(types.uint4)
        assert types.is_int(types.uint6)
        assert types.is_int(types.int8)

        assert types.is_sub_byte(types.int4)
        assert types.is_sub_byte(types.uint1)
        assert types.is_sub_byte(types.uint2)
        assert types.is_sub_byte(types.uint3)
        assert types.is_sub_byte(types.uint4)
        assert types.is_sub_byte(types.uint6)
        assert not types.is_sub_byte(types.int8)
        assert not types.is_sub_byte(types.uint8)

        int4_instance = types.int4()
        uint1_instance = types.uint1()
        uint2_instance = types.uint2()
        uint3_instance = types.uint3()
        uint4_instance = types.uint4()
        uint6_instance = types.uint6()
        int8_instance = types.int8()
        assert types.is_sub_byte(int4_instance)
        assert types.is_sub_byte(uint1_instance)
        assert types.is_sub_byte(uint2_instance)
        assert types.is_sub_byte(uint3_instance)
        assert types.is_sub_byte(uint4_instance)
        assert types.is_sub_byte(uint6_instance)
        assert not types.is_sub_byte(int8_instance)

    def test_state_type_with_tensor(self):
        state_wrapped_type = types.tensor(types.int32, (2, 3))
        state_type = types.state(state_wrapped_type)
        assert types.is_state(state_type)
        assert state_type.wrapped_type() == state_wrapped_type

    def test_numpy_type_to_builtin_type(self):
        assert types.numpy_type_to_builtin_type(np.float32) == types.fp32
        assert types.numpy_type_to_builtin_type(np.float16) == types.fp16
        assert types.numpy_type_to_builtin_type(np.int32) == types.int32
        assert types.numpy_type_to_builtin_type(np.int16) == types.int16
        assert types.numpy_type_to_builtin_type(np.int8) == types.int8
        assert types.numpy_type_to_builtin_type(types.np_int4_dtype) == types.int4
        assert types.numpy_type_to_builtin_type(types.np_uint4_dtype) == types.uint4
        assert types.numpy_type_to_builtin_type(types.np_uint3_dtype) == types.uint3


class TestTypeMapping:
    def test_promote_dtypes_basic(self):
        assert type_mapping.promote_dtypes([types.int32, types.int32]) == types.int32
        assert type_mapping.promote_dtypes([types.int32, types.int64, types.int16]) == types.int64
        assert type_mapping.promote_dtypes([types.fp16, types.fp32, types.fp64]) == types.fp64
        assert type_mapping.promote_dtypes([types.fp16, types.int32, types.int64]) == types.fp16

    @pytest.mark.parametrize(
        "input_size",
        [10, 10000],
    )
    def test_promote_dtypes_different_input_sizes(self, input_size):
        assert (
            type_mapping.promote_dtypes([types.int32, types.int64, types.int16] * input_size)
            == types.int64
        )

    def test_np_val_to_py_type(self):
        assert types.type_mapping.np_val_to_py_type(np.array([True, False])) == (True, False)
        assert types.type_mapping.np_val_to_py_type(np.array(32, dtype=np.int32)) == 32

        # Sub-byte conversion.
        int4_array = np.array([1, 2]).reshape([1, 2, 1]).astype(types.np_int4_dtype)
        py_bytes = types.type_mapping.np_val_to_py_type(int4_array)
        assert len(py_bytes) == 1  # Two 4-bit elements should only take 1 byte.
        restored_array = optimize_utils.restore_elements_from_packed_bits(
            np.frombuffer(py_bytes, dtype=np.uint8),
            nbits=4,
            element_num=2,
            are_packed_values_signed=True,
        )
        np.testing.assert_array_equal(restored_array.reshape([1, 2, 1]), int4_array)


class TestInputTypes:
    def test_state_type(self):
        state_type = StateType(name="x", wrapped_type=TensorType(shape=(2, 3), dtype=np.float32))
        assert state_type.name == "x"
        assert state_type.shape.shape == (2, 3)

    def test_state_type_invalid_wrapped_type(self):
        wrapped_type = ImageType(shape=(1, 3, 3, 3))
        with pytest.raises(ValueError, match="StateType only supports"):
            StateType(wrapped_type=wrapped_type)

        with pytest.raises(ValueError, match="name cannot be set in the state wrapped_type"):
            StateType(wrapped_type=TensorType(name="x", shape=(2, 3)))

        with pytest.raises(
            ValueError, match="default_value cannot be set in the state wrapped_type"
        ):
            StateType(wrapped_type=TensorType(shape=(3,), default_value=np.array([0.0, 0.0, 0.0])))
