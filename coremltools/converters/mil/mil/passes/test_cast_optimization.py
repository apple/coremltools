#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import unittest

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    assert_op_count_match,
    assert_model_is_valid,
    get_op_types_in_program,
    apply_pass_and_basic_check,
)


np.random.seed(1984)


class CastOptimizationPass(unittest.TestCase):
    """"""

    """
    Input graph:
    input -----> cast(dtype="fp32") -----> square -----> cast(dtype="fp32") ---> out

    Output graph:
    input -----> square -----> out
    """

    def test_remove_redundant_casts(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp32")
            x = mb.square(x=x)
            x = mb.cast(x=x, dtype="fp32")
            return x

        self.assertEqual(get_op_types_in_program(prog), ['cast', 'square', 'cast'])

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _,_,block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        self.assertEqual(get_op_types_in_program(prog), ["square"])

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (10, 20)},
        )

    """
    Input graph:
    input -----> cast(dtype="fp16") -----> cast(dtype="fp32") ----> square ---> out

    Output graph:
    input -----> square -----> out
    """

    def test_linear_consecutive_cast_ops_cancellation(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16")
            x = mb.cast(x=x, dtype="fp32")
            x = mb.square(x=x)
            return x

        self.assertEqual(get_op_types_in_program(prog), ['cast', 'cast', 'square'])

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _,_,block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        self.assertEqual(get_op_types_in_program(prog), ["square"])

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (10, 20)},
        )

    """
    Input graph:
    input---->cast(dtype="int32")---->cast(dtype="fp16")--->square--->out

    Output graph:
    input----->cast(dtype="fp16")----->square--->out
    """

    def test_linear_consecutive_cast_ops_fusion(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="int32")
            x = mb.cast(x=x, dtype="fp16")
            x = mb.square(x=x)
            return x

        self.assertEqual(get_op_types_in_program(prog), ['cast', 'cast', 'square'])

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _,_,block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        self.assertEqual(get_op_types_in_program(prog), ["cast", "square"])
        self.assertEqual(block.find_ops(op_type="cast")[0].dtype.val, "fp16")

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (10, 20)},
        )

    """
    Input graph:
    input-->cast(dtype="fp16")-->cast(dtype="fp16")-->cast(dtype="int32")-->cast(dtype="int64")-->cast(dtype="fp32")-->cast(dtype="fp16")-->square->out

    Output graph:
    input---->cast(dtype="fp16")----->square--->out
    """

    def test_linear_multiple_consecutive_cast_ops(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16")
            x = mb.cast(x=x, dtype="fp16")
            x = mb.cast(x=x, dtype="int32")
            x = mb.cast(x=x, dtype="int64")
            x = mb.cast(x=x, dtype="fp32")
            x = mb.cast(x=x, dtype="fp16")
            x = mb.square(x=x)
            return x

        self.assertEqual(get_op_types_in_program(prog), ['cast', 'cast', 'cast', 'cast', 'cast', 'cast', 'square'])

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _,_,block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        self.assertEqual(get_op_types_in_program(prog), ["cast", "square"])
        self.assertEqual(block.find_ops(op_type="cast")[0].dtype.val, "fp16")

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (10, 20)},
        )

    """
    Input graph:
                               |---->cast(dtype="fp32")---->square--->out_1
                               |
    input---->cast(dtype="fp16")---->cast(dtype="fp32")---->relu--->out_2
                               |
                               |---->cast(dtype="fp32")---->log--->out_3

    Output graph:
    
         |---->square--->out_1
         |
    input---->relu--->out_2
         |
         |---->log--->out_3
    """

    def test_same_consecutive_cancelling_casts_on_all_branches(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16")
            x1 = mb.cast(x=x, dtype="fp32")
            x2 = mb.cast(x=x, dtype="fp32")
            x3 = mb.cast(x=x, dtype="fp32")
            x4 = mb.square(x=x1)
            x5 = mb.relu(x=x2)
            x6 = mb.log(x=x3)
            return x4, x5, x6

        self.assertEqual(get_op_types_in_program(prog), ['cast', 'cast', 'cast', 'cast', 'square', 'relu', 'log'])

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _,_,block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        self.assertEqual(get_op_types_in_program(prog), ["square", "relu", "log"])

        assert_model_is_valid(
            prog,
            {"x": (10,20)},
            expected_output_shapes={
                block.outputs[0].name: (10, 20),
                block.outputs[1].name: (10, 20),
                block.outputs[2].name: (10, 20),
            },
        )

    """
    Input graph:
                                |---->cast(dtype="fp16")---->square--->out_1
                                |
    input---->cast(dtype="int32")---->cast(dtype="fp16")---->relu--->out_2
                                |
                                |---->cast(dtype="fp16")---->log--->out_3

    Output graph:

                                |---->square--->out_1
                                |
    input---->cast(dtype="fp16")---->relu--->out_2
                                |
                                |---->log--->out_3
    """

    def test_consecutive_fusable_casts_on_all_branches(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="int32")
            x1 = mb.cast(x=x, dtype="fp16")
            x2 = mb.cast(x=x, dtype="fp16")
            x3 = mb.cast(x=x, dtype="fp16")
            x4 = mb.square(x=x1)
            x5 = mb.relu(x=x2)
            x6 = mb.log(x=x3)
            return x4, x5, x6

        self.assertEqual(get_op_types_in_program(prog), ['cast', 'cast', 'cast', 'cast', 'square', 'relu', 'log'])

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        self.assertEqual(get_op_types_in_program(prog), ["cast", "square", "relu", "log"])
        self.assertEqual(block.find_ops(op_type="cast")[0].dtype.val, "fp16")

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (10, 20),
                block.outputs[1].name: (10, 20),
                block.outputs[2].name: (10, 20),
            },
        )

    """
    Input graph:
    
                                |---->cast(dtype="fp32")---->square--->out_1
                                |
                                |---->cast(dtype="fp16")---->square--->out_2
                                |
    input---->cast(dtype="int32")---->cast(dtype="fp16")---->relu--->out_3
                                |
                                |---->cast(dtype="fp16")---->log--->out_4
                                |
                                |---->cast(dtype="fp32")---->log--->out_5

    Output graph:
    
         |---->square--->out_1
         |
         |                      |---->square--->out_2
         |                      |
    input---->cast(dtype="fp16")---->relu--->out_3
         |                      |
         |                      |---->log--->out_4
         |                      
         |
         |---->log--->out_5

    """

    def test_mixed_consecutive_casts_on_different_branches(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="int32")
            x1 = mb.cast(x=x, dtype="fp32")
            x2 = mb.cast(x=x, dtype="fp16")
            x3 = mb.cast(x=x, dtype="fp16")
            x4 = mb.cast(x=x, dtype="fp16")
            x5 = mb.cast(x=x, dtype="fp32")
            x6 = mb.square(x=x1)
            x7 = mb.square(x=x2)
            x8 = mb.relu(x=x3)
            x9 = mb.log(x=x4)
            x10 = mb.log(x=x5)
            return x6, x7, x8, x9, x10

        self.assertEqual(get_op_types_in_program(prog), ['cast', 'cast', 'cast', 'cast', 'cast', 'cast', 'square', 'square', 'relu', 'log', "log"])

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        self.assertEqual(get_op_types_in_program(prog), ["cast", "square", "square", "relu", "log", "log"])
        self.assertEqual(block.find_ops(op_type="cast")[0].dtype.val, "fp16")

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (10, 20),
                block.outputs[1].name: (10, 20),
                block.outputs[2].name: (10, 20),
            },
        )

    """
    Input graph:

                                |---->cast(dtype="fp32")---->square--->out_1
                                |
    input---->cast(dtype="int32")---->cast(dtype="fp16")---->relu--->out_2
                                |
                                |---->log--->out_3


    Output graph:

         |---->square--->out_1
         |
         |
         |
    input---->cast(dtype="fp16")---->relu--->out_2
         |
         |
         |
         |
         |---->cast(dtype="int32")---->log--->out_3

    """

    def test_different_consecutive_casts__config_on_different_branches(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="int32")
            x1 = mb.cast(x=x, dtype="fp32")
            x2 = mb.cast(x=x, dtype="fp16")
            x3 = mb.square(x=x1)
            x4 = mb.relu(x=x2)
            x5 = mb.log(x=x)
            return x3, x4, x5

        self.assertEqual(get_op_types_in_program(prog),
                         ['cast', 'cast', 'cast', 'square', 'relu', 'log'])

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        self.assertEqual(get_op_types_in_program(prog), ['cast', 'cast', 'square', 'relu', 'log'])

        # Asserting first cast configuration
        cast_1 = block.find_ops(op_type="cast")[0]
        self.assertEqual(cast_1.dtype.val, "int32")
        self.assertEqual(len(cast_1.outputs), 1)
        self.assertEqual(len(cast_1.outputs[0].child_ops), 1)
        self.assertEqual(cast_1.outputs[0].child_ops[0].op_type, "log")

        # Asserting second cast configuration
        cast_2 = block.find_ops(op_type="cast")[1]
        self.assertEqual(cast_2.dtype.val, "fp16")
        self.assertEqual(len(cast_2.outputs), 1)
        self.assertEqual(len(cast_2.outputs[0].child_ops), 1)
        self.assertEqual(cast_2.outputs[0].child_ops[0].op_type, "relu")

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (10, 20),
                block.outputs[1].name: (10, 20),
                block.outputs[2].name: (10, 20),
            },
        )
