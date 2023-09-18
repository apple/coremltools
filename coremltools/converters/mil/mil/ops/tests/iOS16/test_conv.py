#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.tests.iOS16 import backends
from coremltools.converters.mil.testing_utils import get_op_types_in_program

compute_units = testing_reqs.compute_units


class TestConvolution:
    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_type_inference_with_constexpr_ops(self, compute_unit, backend):
        # Test the type inference of the conv op doesn't error out for constexpr bias
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 3, 4, 4), dtype=types.fp32)],
            opset_version=backend.opset_version,
        )
        def prog(x):
            weight = np.random.rand(2, 3, 2, 2)
            bias = mb.constexpr_affine_dequantize(
                quantized_data=np.array([1, 2]).astype(np.uint8),
                zero_point=np.uint8(1),
                scale=np.float32(2),
                axis=0,
            )
            return mb.conv(x=x, weight=weight, bias=bias)

        assert get_op_types_in_program(prog) == ["constexpr_affine_dequantize", "conv"]

        # Test conv op can have dilations with constexpr weight
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 3, 4, 4), dtype=types.fp32)],
            opset_version=backend.opset_version,
        )
        def prog(x):
            weight = mb.constexpr_affine_dequantize(
                quantized_data=np.array(range(24)).astype(np.uint8).reshape(2, 3, 2, 2),
                zero_point=np.uint8(1),
                scale=np.float32(2),
                axis=0,
            )
            return mb.conv(x=x, weight=weight, dilations=[2, 2])

        assert get_op_types_in_program(prog) == ["constexpr_affine_dequantize", "conv"]

        # Test conv op can have dilations with constexpr weight with casts
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 3, 4, 4), dtype=types.fp16)],
            opset_version=backend.opset_version,
        )
        def prog(x):
            weight = mb.constexpr_affine_dequantize(
                quantized_data=np.array(range(24)).astype(np.uint8).reshape(2, 3, 2, 2),
                zero_point=np.uint8(1),
                scale=np.float16(2),
                axis=0,
            )
            cast_weight = mb.cast(x=weight, dtype="fp32")
            cast_weight = mb.cast(x=weight, dtype="fp16")
            return mb.conv(x=x, weight=cast_weight, dilations=[2, 2])

        assert get_op_types_in_program(prog) == [
            "constexpr_affine_dequantize",
            "cast",
            "cast",
            "conv",
        ]
