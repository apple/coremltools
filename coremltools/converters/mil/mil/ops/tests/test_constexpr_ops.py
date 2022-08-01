#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.testing_utils import get_op_types_in_program, ssa_fn

backends = [("mlprogram", "fp32"), ("mlprogram", "fp16")]

@pytest.mark.skipif(
    ct.utils._macos_version() < (13, 0),
    reason="ConstExpr ops available from macOS13 onwards.",
)
class TestConstexprAffineDequantize:
    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend", itertools.product([True, False], backends)
    )
    def test_builder_to_backend_smoke(self, use_cpu_for_conversion, backend):

        t = np.array(range(4)).reshape(1, 1, 2, 2).astype(np.float32)
        decompressed_constant = (
            np.array([1, 2, 3, 4]).reshape(1, 1, 2, 2).astype(np.float32)
        )
        input_placeholders = {
            "x": mb.placeholder(shape=t.shape),
        }
        input_values = {"x": t}

        def build(x):
            quantized_data = np.array([3, 5, 5, 6]).reshape(1, 1, 2, 2).astype(np.uint8)
            scale = np.array([1, 2]).astype(np.float32)
            zero_point = np.array([2, 4]).astype(np.uint8)
            axis = 3
            y = mb.constexpr_affine_dequantize(
                quantized_data=quantized_data,
                zero_point=zero_point,
                scale=scale,
                axis=axis,
            )
            return mb.add(x=x, y=y)

        expected_output_types = (1, 1, 2, 2, types.fp32)
        expected_outputs = t + decompressed_constant.astype(np.float32)

        mlmodel = run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            use_cpu_only=use_cpu_for_conversion,
            frontend_only=False,
            backend=backend,
            converter=ct.convert,
            minimum_deployment_target=ct.target.iOS16,
        )

        # validate that the constexpr op is not removed by any graph pass
        prog = mlmodel._mil_program
        assert "constexpr_affine_dequantize" in get_op_types_in_program(prog)

    @ssa_fn
    def test_builder_eval(self):
        v = mb.constexpr_affine_dequantize(
            quantized_data=np.array([[1, 2, 3], [1, 2, 3]]).astype(np.uint8),
            zero_point=np.uint8(1),
            scale=np.float32(2),
            axis=0
        )
        np.testing.assert_allclose(np.float32([[0, 2, 4], [0, 2, 4]]), v.val)

@pytest.mark.skipif(
    ct.utils._macos_version() < (13, 0),
    reason="ConstExpr ops available from macOS13 onwards.",
)
class TestConstexprCast:
    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend", itertools.product([True, False], backends)
    )
    def test_builder_to_backend_smoke(self, use_cpu_for_conversion, backend):

        t = np.array(range(4)).reshape(4, 1).astype(np.float32)
        decompressed_constant = np.array([1, 2, 3, 4]).reshape(4, 1).astype(np.float32)
        input_placeholders = {
            "x": mb.placeholder(shape=t.shape),
        }
        input_values = {"x": t}

        def build(x):
            source_val = np.array([1, 2, 3, 4]).reshape(4, 1).astype(np.float16)
            y = mb.constexpr_cast(source_val=source_val, output_dtype="fp32")
            return mb.add(x=x, y=y)

        expected_output_types = (4, 1, types.fp32)
        expected_outputs = t + decompressed_constant.astype(np.float32)

        mlmodel = run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            use_cpu_only=use_cpu_for_conversion,
            frontend_only=False,
            backend=backend,
            converter=ct.convert,
            minimum_deployment_target=ct.target.iOS16,
        )

        # validate that the constexpr op is not removed by any graph pass
        prog = mlmodel._mil_program
        assert "constexpr_cast" in get_op_types_in_program(prog)

    @ssa_fn
    def test_builder_eval(self):
        v = mb.constexpr_cast(source_val=np.float16([1, 2]), output_dtype="fp32")
        np.testing.assert_allclose(np.float32([1, 2]), v.val)

@pytest.mark.skipif(
    ct.utils._macos_version() < (13, 0),
    reason="ConstExpr ops available from macOS13 onwards.",
)
class TestConstexprLutToDense:
    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend", itertools.product([True, False], backends)
    )
    def test_builder_to_backend_smoke(self, use_cpu_for_conversion, backend):

        t = np.array(range(4)).reshape(4, 1).astype(np.float32)
        decompressed_constant = np.array([1, 2, 3, 4]).reshape(4, 1).astype(np.float32)
        input_placeholders = {
            "x": mb.placeholder(shape=t.shape),
        }
        input_values = {"x": t}

        def build(x):
            lut_data = np.array(
                [
                    -19.0,
                    4.0,
                    0.0,
                    -1.0,
                    1.0,
                    3.0,
                    5.0,
                    -8.0,
                    19,
                    13,
                    42,
                    4.5,
                    5.4,
                    2.0,
                    -6,
                    -7,
                ]
            ).astype(np.float32)
            indices = np.array([212, 21]).astype(np.uint8)
            shape = np.array([4, 1]).astype(np.uint32)
            y = mb.constexpr_lut_to_dense(lut=lut_data, indices=indices, shape=shape)
            return mb.add(x=x, y=y)

        expected_output_types = (4, 1, types.fp32)
        expected_outputs = t + decompressed_constant.astype(np.float32)

        mlmodel = run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            use_cpu_only=use_cpu_for_conversion,
            frontend_only=False,
            backend=backend,
            converter=ct.convert,
            minimum_deployment_target=ct.target.iOS16,
        )

        # validate that the constexpr op is not removed by any graph pass
        prog = mlmodel._mil_program
        assert "constexpr_lut_to_dense" in get_op_types_in_program(prog)

    @ssa_fn
    def test_builder_eval(self):
        v = mb.constexpr_lut_to_dense(
            lut=np.array([1., 2., 3., 4.]),
            indices=np.array([10, 4]).astype(np.uint8),
            shape=np.array([5,]).astype(np.uint32),
        )
        np.testing.assert_allclose(np.float32([3, 3, 1, 1, 1]).astype(np.float32), v.val)

@pytest.mark.skipif(
    ct.utils._macos_version() < (13, 0),
    reason="ConstExpr ops available from macOS13 onwards.",
)
class TestConstexprSparseToDense:
    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend", itertools.product([True, False], backends)
    )
    def test_builder_to_backend_smoke(self, use_cpu_for_conversion, backend):

        t = np.array(range(4)).reshape(4, 1).astype(np.float32)
        decompressed_constant = np.array([1, 2, 0, 4]).reshape(4, 1).astype(np.float32)
        input_placeholders = {
            "x": mb.placeholder(shape=t.shape),
        }
        input_values = {"x": t}

        def build(x):
            nonzero_data = np.array([1, 2, 4]).astype(np.float32)
            mask = np.array([11]).astype(np.uint8)
            shape = np.array([4, 1]).astype(np.uint32)
            y = mb.constexpr_sparse_to_dense(
                nonzero_data=nonzero_data, mask=mask, shape=shape
            )
            return mb.add(x=x, y=y)

        expected_output_types = (4, 1, types.fp32)
        expected_outputs = t + decompressed_constant.astype(np.float32)

        mlmodel = run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            use_cpu_only=use_cpu_for_conversion,
            frontend_only=False,
            backend=backend,
            converter=ct.convert,
            minimum_deployment_target=ct.target.iOS16,
        )

        # validate that the constexpr op is not removed by any graph pass
        prog = mlmodel._mil_program
        assert "constexpr_sparse_to_dense" in get_op_types_in_program(prog)

    @ssa_fn
    def test_builder_eval(self):
        v = mb.constexpr_sparse_to_dense(
            nonzero_data=np.array([1., 2., 4.]),
            mask=np.array([11]).astype(np.uint8),
            shape=np.array([4,]).astype(np.uint32),
        )
        np.testing.assert_allclose(np.float32([1., 2., 0., 4.]), v.val)
