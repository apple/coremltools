#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
from unittest.mock import patch

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.mil.ops.tests.iOS14.test_elementwise_unary import (
    TestElementwiseUnary as _TestElementwiseUnary_iOS14,
)
from coremltools.converters.mil.mil.ops.tests.iOS17 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipeline
from coremltools.converters.mil.mil.types.type_mapping import numpy_type_to_builtin_type
from coremltools.converters.mil.mil.var import Var
from coremltools.converters.mil.testing_reqs import compute_units
from coremltools.converters.mil.testing_utils import get_op_types_in_program


class TestElementwiseUnary:
    @pytest.mark.parametrize(
        "compute_unit, backend, src_dtype, dst_dtype",
        itertools.product(
            compute_units,
            backends,
            [
                np.float16,
                np.float32,
                np.int32,
                np.int16,
                np.uint16,
                np.int8,
                np.uint8,
            ],
            [
                np.float16,
                np.float32,
                np.int32,
                np.int16,
                np.uint16,
                np.int8,
                np.uint8,
            ],
        ),
    )
    def test_builder_eval_cast_ios17(self, compute_unit, backend, src_dtype, dst_dtype):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=src_dtype)
        dst_dtype_str = types.builtin_to_string(numpy_type_to_builtin_type(dst_dtype))
        expected_res = x.astype(dtype=np.float16)

        @mb.program(input_specs=[], opset_version=backend.opset_version)
        def prog():
            return mb.cast(x=x, dtype=dst_dtype_str)

        main_func = prog.functions["main"]
        cast_op = main_func.find_ops(op_type="cast")[0]
        np.testing.assert_allclose(expected_res, cast_op.outputs[0].val, atol=1e-04, rtol=1e-05)

    @pytest.mark.parametrize(
        "backend, dtype",
        itertools.product(
            backends,
            ["int8", "uint8", "int16", "uint16"],
        ),
    )
    def test_cast_with_symbolic_value_iOS17(self, backend, dtype):
        s1 = get_new_symbol()

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(s1, 1))],
            opset_version=backend.opset_version,
        )
        def prog(x):
            shape = mb.shape(x=x)
            out = mb.cast(x=shape, dtype=dtype)
            assert out.val is None
            sym_val = out.sym_val
            assert sym_val.tolist() == [s1, 1]
            return out

    @pytest.mark.parametrize(
        "compute_unit, backend, src_dtype, dst_dtype",
        itertools.product(
            compute_units,
            backends,
            [np.float16, np.float32, np.int16, np.int32, np.uint16, np.int8, np.uint8],
            [np.float16, np.float32, np.int16, np.int32, np.uint16, np.int8, np.uint8],
        ),
    )
    def test_builder_to_backend_cast_ios17(self, compute_unit, backend, src_dtype, dst_dtype):
        _SUPPORTED_IO_DTYPES = {types.fp16, types.fp32, types.int32}
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=src_dtype)
        src_builtin_dtype = numpy_type_to_builtin_type(src_dtype)
        dst_builtin_dtype = numpy_type_to_builtin_type(dst_dtype)
        expected_res = x.astype(dtype=np.float16)

        expected_cast_num = 1
        if src_builtin_dtype not in _SUPPORTED_IO_DTYPES:
            # A cast will be inserted for unsupported dtypes inputs.
            expected_cast_num += 1

        # As CoreML IO only allows fp16/32 and int32, the output will be further cast.
        expected_res_builtin_dtype = dst_builtin_dtype
        if dst_builtin_dtype not in _SUPPORTED_IO_DTYPES:
            expected_res_builtin_dtype = (
                types.int32 if types.is_int(dst_builtin_dtype) else types.fp32
            )
            expected_cast_num += 1

        def build(x):
            return mb.cast(x=x, dtype=types.builtin_to_string(dst_builtin_dtype))

        with patch.object(Var, "_is_nonreplaceable_var") as mocked_is_nonreplaceable_var:
            # Mock that the cast is non-replaceable, to make sure it's kept in the graph.
            mocked_is_nonreplaceable_var.side_effect = (
                lambda var: var.op and var.op.op_type == "cast"
            )
            # Remove the cast optimization pass to make sure all cast are kept in the graph.
            pass_pipeline: PassPipeline = PassPipeline.DEFAULT
            pass_pipeline.remove_passes(
                ["common::cast_optimization", "common::topological_reorder"]
            )
            mlmodel = run_compare_builder(
                build,
                {"x": mb.placeholder(shape=x.shape, dtype=src_builtin_dtype)},
                input_values={"x": x},
                expected_output_types=x.shape + (expected_res_builtin_dtype,),
                expected_outputs=expected_res,
                compute_unit=compute_unit,
                backend=backend,
                pass_pipeline=pass_pipeline,
            )
            prog = mlmodel._mil_program
            cast_ops = prog["main"].find_ops(op_type="cast")
            assert len(cast_ops) == expected_cast_num

    @pytest.mark.parametrize(
        "compute_unit, backend, op_name, epsilon_val, x_eps_dtype",
        itertools.product(
            compute_units,
            backends,
            ["inverse", "log", "rsqrt"],
            [1e-3, 1e-1, 1.0],
            [(np.float32, np.float16), (np.float16, np.float32)],
        ),
    )
    def test_builder_to_backend_stress_with_epsilon(
        self,
        compute_unit,
        backend,
        op_name,
        epsilon_val,
        x_eps_dtype,
    ):
        # From iOS17, epsilon and have different dtype than x
        _TestElementwiseUnary_iOS14._test_builder_to_backend_stress_with_epsilon(
            compute_unit, backend, op_name, epsilon_val, x_eps_dtype
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            [ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.ALL],
            backends,
        ),
    )
    def test_cast_fp16_output_bug_smoke(self, compute_unit, backend):
        """
        Since a fp16 output bug in Core ML can only be reproduced by non-CPU backends,
        for this test, we hardcode the compute_unit.
        """

        def build(x):
            return mb.cast(x=x, dtype="fp16")

        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        expected_res = x.astype(dtype=np.float16)

        mlmodel = run_compare_builder(
            build,
            {"x": mb.placeholder(shape=x.shape, dtype=types.int32)},
            input_values={"x": x},
            expected_output_types=x.shape + (types.fp16,),
            expected_outputs=expected_res,
            compute_unit=compute_unit,
            backend=backend,
        )

        prog = mlmodel._mil_program
        assert get_op_types_in_program(prog) == ["cast"]
        cast_op = prog.find_ops(op_type="cast", exactly_one=True)[0]
        assert cast_op.dtype.val == "fp16"
