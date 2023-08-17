#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.tests.iOS17 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.testing_reqs import compute_units


class TestTopK:
    @pytest.mark.parametrize(
        "compute_unit, backend, x_dtype, k_dtype",
        itertools.product(
            compute_units,
            backends,
            [np.float16, np.float32, np.int8, np.int16, np.int32, np.uint8, np.uint16],
            [np.int8, np.int16, np.int32],
        ),
    )
    def test_ios17_different_dtypes(self, compute_unit, backend, x_dtype, k_dtype):
        def build(x):
            return mb.topk(x=x, k=k_dtype(2), axis=1)

        val = np.array([[2, 3, 1], [5, 4, 6]], dtype=x_dtype)
        x_mb_dtype = types.numpy_type_to_builtin_type(x_dtype)
        input_placeholders = {"x": mb.placeholder(shape=val.shape, dtype=x_mb_dtype)}
        input_values = {"x": val}
        # As int16 is not in CoreML I/O supported dtypes, it will be cast to int32.
        expected_output_types = [(2, 2, x_mb_dtype), (2, 2, types.int32)]
        expected_outputs = [
            np.array([[3, 2], [6, 5]], dtype=x_dtype),
            np.array([[1, 0], [2, 0]], dtype=np.int32),
        ]

        mlmodel = run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )
        prog = mlmodel._mil_program
        topk_op = prog["main"].find_ops(op_type="topk")[0]
        expected_x_dtype = x_mb_dtype
        if backend.precision == "fp16" and types.is_float(x_mb_dtype):
            expected_x_dtype = types.fp16
        assert types.builtin_to_string(topk_op.x.dtype) == types.builtin_to_string(expected_x_dtype)

    @pytest.mark.parametrize(
        "compute_unit, backend, output_indices_dtype",
        itertools.product(
            compute_units,
            backends,
            ["int32", "uint16", None],
        ),
    )
    def test_ios17_output_indices_dtype(self, compute_unit, backend, output_indices_dtype):
        def build(x):
            return mb.topk(x=x, k=2, axis=1, output_indices_dtype=output_indices_dtype)

        val = np.array([[2, 3, 1], [5, 4, 6]], dtype=np.int32)
        input_placeholders = {"x": mb.placeholder(shape=val.shape, dtype=types.int32)}
        input_values = {"x": val}
        expected_output_types = [(2, 2, types.int32), (2, 2, types.int32)]
        expected_outputs = [
            np.array([[3, 2], [6, 5]], dtype=np.int32),
            np.array([[1, 0], [2, 0]], dtype=np.int32),
        ]

        mlmodel = run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )
        prog = mlmodel._mil_program
        topk_op = prog["main"].find_ops(op_type="topk")[0]

        # If output_indices_dtype is not set, the output should be in type int32
        expected_output_indices_dtype = "int32"
        if output_indices_dtype is not None:
            expected_output_indices_dtype = output_indices_dtype

        assert types.builtin_to_string(topk_op.outputs[1].dtype) == expected_output_indices_dtype

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_ios17_invalid_output_indices_dtype(self, compute_unit, backend):
        def build(x):
            return mb.topk(x=x, k=2, axis=1, output_indices_dtype="dummy")

        val = np.array([[2, 3, 1], [5, 4, 6]], dtype=np.int32)
        with pytest.raises(ValueError, match="invalid output_indices_dtype"):
            run_compare_builder(
                build,
                input_placeholders={"x": mb.placeholder(shape=val.shape, dtype=types.int32)},
                input_values={"x": val},
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
    def test_ios17_redundant_output_indices_dtype_early_error_out(self, compute_unit, backend):
        def build(x):
            return mb.topk(x=x, k=2, axis=1, return_indices=False, output_indices_dtype="int32")

        val = np.array([[2, 3, 1], [5, 4, 6]], dtype=np.int32)
        with pytest.raises(
            ValueError, match='"output_indices_dtype" can only be set when "return_indices=True"'
        ):
            run_compare_builder(
                build,
                input_placeholders={"x": mb.placeholder(shape=val.shape, dtype=types.int32)},
                input_values={"x": val},
                compute_unit=compute_unit,
                backend=backend,
            )
