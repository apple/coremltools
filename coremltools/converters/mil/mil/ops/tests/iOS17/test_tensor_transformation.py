#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.tests.iOS14.test_tensor_transformation import (
    TestSliceByIndex as _TestSliceByIndexIos14,
)
from coremltools.converters.mil.mil.ops.tests.iOS14.test_tensor_transformation import (
    TestSliceBySize as _TestSliceBySizeIos14,
)
from coremltools.converters.mil.mil.ops.tests.iOS16.test_tensor_transformation import (
    TestReshapeLike as _TestReshapeLike_iOS16,
)
from coremltools.converters.mil.mil.ops.tests.iOS17 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.mil.types.type_mapping import numpy_type_to_builtin_type
from coremltools.converters.mil.testing_reqs import compute_units


class TestReshape:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_reshape_with_zero_different_len_iOS17(self, compute_unit, backend):
        t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return [mb.reshape(x=x, shape=[1, 0, -1, 0])]

        # In IOS17 it accepts different length.
        expected_output_types = [(1, 1, 2, 3, types.fp32)]
        expected_outputs = [np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.float32)]
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_reshape_invalid_with_zero(self, compute_unit, backend):
        t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return [mb.reshape(x=x, shape=[4, 0, -1, 0])]

        with pytest.raises(ValueError, match="Invalid target shape in `reshape` op"):
            run_compare_builder(
                build,
                input_placeholders,
                input_values,
                compute_unit=compute_unit,
                backend=backend,
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, x_dtype, shape_dtype",
        itertools.product(
            compute_units,
            backends,
            [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.float16, np.float32],
            [np.int8, np.int16, np.int32],
        ),
    )
    def test_reshape_ios17_different_data_types(self, compute_unit, backend, x_dtype, shape_dtype):
        x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=x_dtype)
        target_shape = np.array([1, 6], dtype=shape_dtype)
        x_builtin_dtype = numpy_type_to_builtin_type(x_dtype)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype)}
        input_values = {"x": x_val}

        def build(x):
            return mb.reshape(x=x, shape=target_shape)

        expected_output_types = (1, 6, x_builtin_dtype)
        expected_outputs = np.array([[1, 2, 3, 4, 5, 6]], dtype=x_dtype)
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestReshapeLike(_TestReshapeLike_iOS16):
    @pytest.mark.parametrize(
        "compute_unit, backend, InputShape_RefShapes_Begins_Ends_EndMasks, x_dtype, ref_dtype",
        itertools.product(
            compute_units,
            backends,
            [
                [(4, 3), ((2, 2, 3), (1, 3)), (0, 1), (2, 2), (False, False)],
                [(32,), ((1, 2, 2, 2), (3, 2, 2)), (1, 1), (0, 0), (True, True)],
                [(72, 1), ((1, 2, 3, 4, 1), (3,)), (1, 0), (0, 1), (True, False)],
            ],
            [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.float16, np.float32, bool],
            [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.float16, np.float32, bool],
        ),
    )
    def test_builder_to_backend_smoke(
        self,
        compute_unit,
        backend,
        InputShape_RefShapes_Begins_Ends_EndMasks,
        x_dtype,
        ref_dtype,
    ):
        super().test_builder_to_backend_smoke(
            compute_unit, backend, InputShape_RefShapes_Begins_Ends_EndMasks, x_dtype, ref_dtype
        )


class TestExpandDims:
    @pytest.mark.parametrize(
        "compute_unit, backend, x_dtype",
        itertools.product(
            compute_units,
            backends,
            [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.float16, np.float32],
        ),
    )
    def test_expand_dims_different_data_types(self, compute_unit, backend, x_dtype):
        axis = 1
        x_val = np.random.randint(low=2, high=6, size=(2, 3, 4)).astype(x_dtype)
        x_builtin_dtype = numpy_type_to_builtin_type(x_dtype)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype)}
        input_values = {"x": x_val}

        def build(x):
            return mb.expand_dims(x=x, axes=[axis])

        x_shape = list(x_val.shape)
        out_shape = x_shape[:axis] + [1] + x_shape[axis:]
        expected_output_types = tuple(out_shape[:]) + (x_builtin_dtype,)
        expected_outputs = np.expand_dims(input_values["x"], axis)
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestReverse:
    @pytest.mark.parametrize(
        "compute_unit, backend, x_dtype",
        itertools.product(
            compute_units,
            backends,
            [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.float16, np.float32],
        ),
    )
    def test_reverse_different_data_types(self, compute_unit, backend, x_dtype):
        def build(x):
            return [mb.reverse(x=x), mb.reverse(x=x, axes=[0])]

        x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=x_dtype)
        x_builtin_dtype = numpy_type_to_builtin_type(x_dtype)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype)}
        input_values = {"x": x_val}
        expected_output_types = [(2, 3, x_builtin_dtype), (2, 3, x_builtin_dtype)]
        expected_outputs = [
            np.array([[6, 5, 4], [3, 2, 1]], dtype=x_dtype),
            np.array([[4, 5, 6], [1, 2, 3]], dtype=x_dtype),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestReverseSequence:
    @pytest.mark.parametrize(
        "compute_unit, backend, x_dtype, length_dtype",
        itertools.product(
            compute_units,
            backends,
            [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.float16, np.float32],
            [np.int8, np.int16, np.int32],
        ),
    )
    def test_reverse_sequence_different_data_types(
        self, compute_unit, backend, x_dtype, length_dtype
    ):
        def build(x, length):
            return mb.reverse_sequence(x=x, lengths=length, seq_axis=1, batch_axis=0)

        x_val = np.array(
            [
                [1, 2, 3, 4, 5, 0, 0, 0],
                [1, 2, 0, 0, 0, 0, 0, 0],
                [1, 2, 3, 4, 0, 0, 0, 0],
                [1, 2, 3, 4, 5, 6, 7, 8],
            ],
            dtype=x_dtype,
        )
        length_val = np.array([7, 2, 3, 5], dtype=length_dtype)
        x_builtin_dtype = numpy_type_to_builtin_type(x_dtype)
        length_builtin_dtype = numpy_type_to_builtin_type(length_dtype)

        input_placeholders = {
            "x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype),
            "length": mb.placeholder(shape=length_val.shape, dtype=length_builtin_dtype),
        }
        input_values = {"x": x_val, "length": length_val}
        expected_output_types = (4, 8, x_builtin_dtype)
        expected_outputs = np.array(
            [
                [0, 0, 5, 4, 3, 2, 1, 0],
                [2, 1, 0, 0, 0, 0, 0, 0],
                [3, 2, 1, 4, 0, 0, 0, 0],
                [5, 4, 3, 2, 1, 6, 7, 8],
            ],
            dtype=x_dtype,
        )

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestSqueeze:
    @pytest.mark.parametrize(
        "compute_unit, backend, x_dtype",
        itertools.product(
            compute_units,
            backends,
            [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.float16, np.float32],
        ),
    )
    def test_squeeze_different_data_types(self, compute_unit, backend, x_dtype):
        def build(x):
            return mb.squeeze(x=x, axes=(-1,))

        x_val = np.array([[[[1], [2], [3]]]], dtype=x_dtype)
        x_builtin_dtype = numpy_type_to_builtin_type(x_dtype)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype)}
        input_values = {"x": x_val}
        expected_outputs = np.squeeze(x_val, -1)
        expected_output_types = tuple(expected_outputs.shape) + (x_builtin_dtype,)
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestTranspose:
    @pytest.mark.parametrize(
        "compute_unit, backend, x_dtype",
        itertools.product(
            compute_units,
            backends,
            [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.float16, np.float32],
        ),
    )
    def test_transpose_different_data_types(self, compute_unit, backend, x_dtype):
        def build(x):
            return mb.transpose(x=x, perm=(-1, 0))

        x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=x_dtype)
        x_builtin_dtype = numpy_type_to_builtin_type(x_dtype)
        run_compare_builder(
            build,
            input_placeholders={"x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype)},
            input_values={"x": x_val},
            expected_output_types=(3, 2, types.fp32),
            expected_outputs=x_val.T,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestSlidingWindows:
    @pytest.mark.parametrize(
        "compute_unit, backend, x_dtype",
        itertools.product(
            compute_units,
            backends,
            [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.float16, np.float32],
        ),
    )
    def test_ios17_different_data_types(self, compute_unit, backend, x_dtype):
        def build(x):
            return mb.sliding_windows(x=x, axis=1, size=2)

        x_val = np.array([[[[9.0]], [[5.0]], [[1.0]], [[3.0]]]], dtype=x_dtype)
        x_builtin_dtype = numpy_type_to_builtin_type(x_dtype)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype)}
        input_values = {"x": x_val}
        expected_output_types = (1, 3, 2, 1, 1, x_builtin_dtype)
        expected_outputs = np.array(
            [[[[[9.0]], [[5.0]]], [[[5.0]], [[1.0]]], [[[1.0]], [[3.0]]]]],
            dtype=x_dtype,
        )

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestSliceByIndex(_TestSliceByIndexIos14):
    @pytest.mark.parametrize(
        "compute_unit, backend, x_dtype, idx_dtype",
        itertools.product(
            compute_units,
            backends,
            (np.float16, np.float32, np.int8, np.int16, np.int32, np.uint8, np.uint16),
            (np.int8, np.int16, np.int32),
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, x_dtype, idx_dtype):
        super().test_builder_to_backend_smoke(compute_unit, backend, x_dtype, idx_dtype)


class TestSliceBySize(_TestSliceBySizeIos14):
    @pytest.mark.parametrize(
        "compute_unit, backend, size_val, x_dtype, idx_dtype",
        itertools.product(
            compute_units,
            backends,
            ([1, 2, 3], [-1, 2, -1]),
            (np.float16, np.float32, np.int8, np.int16, np.int32, np.uint8, np.uint16),
            (np.int8, np.int16, np.int32),
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, size_val, x_dtype, idx_dtype):
        super().test_builder_to_backend_smoke(compute_unit, backend, size_val, x_dtype, idx_dtype)
