#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.tests.iOS17 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.mil.types import (
    builtin_to_string,
    nptype_from_builtin,
    numpy_type_to_builtin_type,
)
from coremltools.converters.mil.testing_reqs import compute_units


class TestLinear:
    @pytest.mark.parametrize(
        "compute_unit, backend, x_dtype, weight_bias_dtype",
        itertools.product(
            compute_units,
            backends,
            [np.float16, np.float32, np.int32],
            [np.float16, np.float32, np.int32],
        ),
    )
    def test_linear_ios17_mixed_precision(self, compute_unit, backend, x_dtype, weight_bias_dtype):
        if x_dtype == np.int32:
            pytest.xfail("Linear op doesn't work with int32 input (rdar://111421695)")

        out_channels = 3
        x_shape = np.random.randint(low=1, high=3, size=(3,))
        w_shape = np.array([out_channels, x_shape[-1]])
        b_shape = np.array([out_channels])

        x_val = np.random.randint(low=0, high=10, size=x_shape).astype(x_dtype)
        weight_val = np.random.randint(low=0, high=10, size=w_shape).astype(weight_bias_dtype)
        bias_val = np.random.randint(low=0, high=10, size=b_shape).astype(weight_bias_dtype)

        x_builtin_dtype = numpy_type_to_builtin_type(x_dtype)
        input_placeholders = {
            "x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype),
        }

        def build(x):
            return mb.linear(x=x, weight=weight_val, bias=bias_val)

        expected_outputs = np.matmul(x_val, np.transpose(weight_val)) + bias_val
        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_outputs.shape + (x_builtin_dtype,),
            expected_outputs=expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, x_input_type, weight_input_type",
        itertools.product(
            compute_units,
            backends,
            [types.int32, types.fp16, types.fp32],
            [types.int32, types.fp16, types.fp32],
        ),
    )
    def test_default_bias_type_ios17(self, compute_unit, backend, x_input_type, weight_input_type):
        # Start from iOS17, x and weight can have different dtype.
        # Test the default bias matches the dtype of weight.
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 2), dtype=types.fp32)],
            opset_version=backend.opset_version,
        )
        def prog(x):
            x = mb.cast(x=x, dtype=builtin_to_string(x_input_type))
            weight = np.random.rand(3, 2).astype(nptype_from_builtin(weight_input_type))
            res = mb.linear(x=x, weight=weight)
            assert res.op.bias.val.dtype == nptype_from_builtin(weight_input_type)
            return res


class TestMatMul:
    @pytest.mark.parametrize(
        "compute_unit, backend, x_dtype, y_dtype",
        itertools.product(
            compute_units,
            backends,
            [np.float32, np.float16, np.int32],
            [np.float32, np.float16, np.int32],
        ),
    )
    def test_ios17_mixed_precision(self, compute_unit, backend, x_dtype, y_dtype):
        x_val = np.random.randint(low=0, high=10, size=(2, 5)).astype(x_dtype)
        y_val = np.random.randint(low=0, high=10, size=(5, 10)).astype(y_dtype)
        x_mb_dtype = numpy_type_to_builtin_type(x_dtype)
        y_mb_dtype = numpy_type_to_builtin_type(y_dtype)
        expected_outputs = np.matmul(x_val, y_val)

        def build_x_const(y):
            return mb.matmul(x=x_val, y=y, transpose_x=False, transpose_y=False)

        def build_y_const(x):
            return mb.matmul(x=x, y=y_val, transpose_x=False, transpose_y=False)

        mlmodel = run_compare_builder(
            build_y_const,
            input_placeholders={"x": mb.placeholder(shape=x_val.shape, dtype=x_mb_dtype)},
            input_values={"x": x_val},
            expected_output_types=expected_outputs.shape + (x_mb_dtype,),
            expected_outputs=expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            pass_pipeline=ct.PassPipeline.EMPTY,
        )
        prog = mlmodel._mil_program
        matmul_op = prog["main"].find_ops(op_type="matmul")[0]
        # When x is non-const and y is const, the output should have the same dtype as x.
        assert types.builtin_to_string(matmul_op.outputs[0].dtype) == types.builtin_to_string(
            x_mb_dtype
        )

        mlmodel = run_compare_builder(
            build_x_const,
            input_placeholders={"y": mb.placeholder(shape=y_val.shape, dtype=y_mb_dtype)},
            input_values={"y": y_val},
            expected_output_types=expected_outputs.shape + (y_mb_dtype,),
            expected_outputs=expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            pass_pipeline=ct.PassPipeline.EMPTY,
        )
        prog = mlmodel._mil_program
        matmul_op = prog["main"].find_ops(op_type="matmul")[0]
        # When x is const and y is non-const, the output should have the same dtype as y.
        assert types.builtin_to_string(matmul_op.outputs[0].dtype) == types.builtin_to_string(
            y_mb_dtype
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_ios17_invalid_mixed_precision(self, compute_unit, backend):
        """When x and y are both const or both non-const, mixed precision is not allowed."""
        x_val = np.random.rand(2, 5).astype(np.float32)
        y_val = np.random.randint(low=0, high=10, size=(5, 10)).astype(np.int32)

        def build_both_const():
            return mb.matmul(x=x_val, y=y_val, transpose_x=False, transpose_y=False)

        def build_both_not_const(x, y):
            return mb.matmul(x=x, y=y, transpose_x=False, transpose_y=False)

        with pytest.raises(
            ValueError, match="when x and y are both const, their dtype need to match"
        ):
            run_compare_builder(
                build_both_const,
                input_placeholders={},
                input_values={},
                compute_unit=compute_unit,
                backend=backend,
            )

        with pytest.raises(
            ValueError, match="when x and y are both non-const, their dtype need to match"
        ):
            run_compare_builder(
                build_both_not_const,
                input_placeholders={
                    "x": mb.placeholder(shape=x_val.shape, dtype=types.fp32),
                    "y": mb.placeholder(shape=y_val.shape, dtype=types.int32),
                },
                input_values={"x": x_val, "y": y_val},
                compute_unit=compute_unit,
                backend=backend,
            )
