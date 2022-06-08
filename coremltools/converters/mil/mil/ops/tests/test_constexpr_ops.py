#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder

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

        run_compare_builder(
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

        run_compare_builder(
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

        run_compare_builder(
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

        run_compare_builder(
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
