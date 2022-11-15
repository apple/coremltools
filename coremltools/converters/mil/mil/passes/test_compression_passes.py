#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import get_op_types_in_program

from .compression_passes import (WeightAffineQuantizer, WeightPalettizer,
                                 WeightSparsifier)

np.random.seed(1984)

def _get_conv_program():
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 30, 10, 10))], opset_version=ct.target.iOS16)
    def prog(x):
        conv_weight = np.random.rand(90, 30, 2, 2).astype(np.float32)
        x =  mb.conv(x=x, weight=conv_weight)
        return x
    return prog

class TestBasicCompressionGraphPass:
    # Most of the numerical tests are already convered in coremltools.tests.ml_program.test_compression_utils
    # This test is checking the basic behavior of the graph pass classes

    @staticmethod
    @pytest.mark.parametrize(
        "fake_compression",
        [True, False],
    )
    def test_affine_quantizer(fake_compression):
        quantizer = WeightAffineQuantizer(fake_compression=fake_compression, op_selector=lambda const: True)
        prog = _get_conv_program()
        quantizer.apply(prog)
        expected_ops = ["constexpr_affine_dequantize", "conv"] if not fake_compression else ["conv"]
        assert get_op_types_in_program(prog) == expected_ops

    @staticmethod
    @pytest.mark.parametrize(
        "fake_compression",
        [True, False],
    )
    def test_weight_sparsifier(fake_compression):
        quantizer = WeightSparsifier(
            fake_compression=fake_compression, 
            op_selector=lambda const: True, 
            mode="percentile_based",
            target_percentile=0.75)
        prog = _get_conv_program()
        quantizer.apply(prog)
        expected_ops = ["constexpr_sparse_to_dense", "conv"] if not fake_compression else ["conv"]
        assert get_op_types_in_program(prog) == expected_ops

    @staticmethod
    @pytest.mark.parametrize(
        "fake_compression",
        [True, False],
    )
    def test_weight_palettization(fake_compression):
        quantizer = WeightPalettizer(
            fake_compression=fake_compression, 
            op_selector=lambda const: True, 
            mode="uniform",
            nbits=4,
        )
        prog = _get_conv_program()
        quantizer.apply(prog)
        expected_ops = ["constexpr_lut_to_dense", "conv"] if not fake_compression else ["conv"]
        assert get_op_types_in_program(prog) == expected_ops
