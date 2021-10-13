#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    assert_op_count_match,
    assert_model_is_valid,
    get_op_types_in_program,
    apply_pass_and_basic_check,
)
import unittest
import pytest

import numpy as np

np.random.seed(1984)


class PadConvOptimizationPass(unittest.TestCase):
    """
    Input graph:
    input -----> pad -----> transpose -----> conv -----> transpose ---> out

    Output graph:
    input -----> transpose -----> pad ----> conv -----> transpose ----> out
    """

    def test_simple_direct_output(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 20, 24))])
        def prog(x):
            x = mb.pad(x=x, pad=[0,0,1,1,1,1,0,0])
            x = mb.transpose(x=x, perm=[0, 3, 1, 2])
            x = mb.conv(x=x, weight=np.random.random([24,24,3,3]), pad_type="valid")
            x = mb.transpose(x=x, perm=[0, 2, 3, 1])
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::pad_conv_connect"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["pad", "transpose", "conv", "transpose"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["transpose", "pad", "conv", "transpose"])
        assert_model_is_valid(
            prog,
            {"x": (1, 16, 20, 24)},
            expected_output_shapes={block.outputs[0].name: (1, 16, 20, 24)},
        )

    """
    Input graph:
    input -----> pad -----> transpose -----> conv -----> transpose ---> out
                  |
                  |
                  --------> transpose -----> conv -----> transpose ---> out

    Output graph:
    input ---------> transpose -----> pad -----> conv -----> transpose ---> out
             |
             |
             ------> transpose -----> pad -----> conv -----> transpose ---> out

    """

    def test_pad_transposed_forked_conv(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 20, 24))])
        def prog(x):
            pad = mb.pad(x=x, pad=[0,0,1,1,1,1,0,0])
            x = mb.transpose(x=pad, perm=[0, 3, 1, 2])
            x = mb.conv(x=x, weight=np.random.random([24,24,3,3]), pad_type="valid")
            x = mb.transpose(x=x, perm=[0, 2, 3, 1])
            y = mb.transpose(x=pad, perm=[0, 3, 1, 2])
            y = mb.conv(x=y, weight=np.random.random([24,24,3,3]), pad_type="valid")
            y = mb.transpose(x=y, perm=[0, 2, 3, 1])
            return x, y

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::pad_conv_connect"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["pad", "transpose", "conv", "transpose", "transpose", "conv", "transpose"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["transpose", "pad", "conv", "transpose", "transpose", "pad", "conv", "transpose"])
        assert_model_is_valid(
            prog,
            {"x": (1, 16, 20, 24)},
            expected_output_shapes={block.outputs[0].name: (1, 16, 20, 24),
                                    block.outputs[1].name: (1, 16, 20, 24)},
        )

    """
    Input graph:
    input -----> pad -----> transpose -----> conv -----> transpose ---> out
                  |
                  |
                  ---------> out

    Output graph:
    No change.
    """

    def test_pad_output(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 20, 24))])
        def prog(x):
            pad = mb.pad(x=x, pad=[0,0,1,1,1,1,0,0])
            x = mb.transpose(x=pad, perm=[0, 3, 1, 2])
            x = mb.conv(x=x, weight=np.random.random([24,24,3,3]), pad_type="valid")
            x = mb.transpose(x=x, perm=[0, 2, 3, 1])
            return x, pad

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::pad_conv_connect"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["pad", "transpose", "conv", "transpose"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["pad", "transpose", "conv", "transpose"])
        assert_model_is_valid(
            prog,
            {"x": (1, 16, 20, 24)},
            expected_output_shapes={block.outputs[0].name: (1, 16, 20, 24),
                                    block.outputs[1].name: (1, 18, 22, 24)},
        )

