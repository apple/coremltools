#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools._deps import _IS_MACOS
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes import quantization_passes as transform
from coremltools.converters.mil.testing_utils import (
    assert_model_is_valid,
    get_op_types_in_program,
    apply_pass_and_basic_check,
)
import unittest
import numpy as np
import coremltools as ct

np.random.seed(1984)


class FP16CastTransform(unittest.TestCase):
    """"""

    """
    Input graph:
        input -----> square -----> out

    Output graph:
        input -----> cast(dtype="fp16") -----> square -----> cast(dtype="fp32") ---> out
    """

    def test_single_input_to_single_operation(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.square(x=x)
            return x

        self.assertEqual(get_op_types_in_program(prog), ['square'])

        apply_pass_and_basic_check(prog, transform.FP16ComputePrecision(op_selector=lambda op: True))
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        self.assertEqual(get_op_types_in_program(prog), ["cast", "square", "cast"])

        # Asserting first cast configuration
        cast_1 = block.find_ops(op_type="cast")[0]
        self.assertEqual(cast_1.dtype.val, "fp16")
        self.assertEqual(len(cast_1.outputs), 1)
        self.assertEqual(len(cast_1.outputs[0].child_ops), 1)
        self.assertEqual(cast_1.outputs[0].child_ops[0].op_type, "square")

        # Asserting second cast configuration
        cast_2 = block.find_ops(op_type="cast")[1]
        self.assertEqual(cast_2.dtype.val, "fp32")
        self.assertEqual(len(cast_2.outputs), 1)
        self.assertEqual(len(cast_2.outputs[0].child_ops), 0)

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (10, 20)},
        )

    """
    Input graph:
        input -----> div -----> out
                      ^
        const(eps) ---|

    Output graph:
        input --------> cast(dtype="fp16") -----> div -----> cast(dtype="fp32") ---> out
                                                   ^
        const(eps) ---> cast(dtype="fp16") --------|
    """

    def test_divide_by_zero_operation(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            eps = mb.const(val=1e-10)
            x = mb.real_div(x=x, y=eps)
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, transform.FP16ComputePrecision(op_selector=lambda op: True)
        )

        mlmodel = ct.convert(prog, source="milinternal")
        input_dict = {"x": np.random.rand(10, 20)}

        if _IS_MACOS:
            prediction = mlmodel.predict(input_dict, useCPUOnly=True)
            assert(not np.isnan(prediction['real_div_0']).any())
            assert(np.isfinite(prediction['real_div_0']).all())

    """
    Input graph:
        input1 ----->|
                     concat -----> out
        input2 ----->|

    Output graph:
        input1 -----> cast(dtype="fp16") ----->|
                                               concat -----> cast(dtype="fp32") ---> out
        input2 -----> cast(dtype="fp16") ----->|

    """

    def test_multiple_inputs_to_single_operation(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20)), mb.TensorSpec(shape=(10, 20))])
        def prog(x, y):
            x = mb.concat(values= (x,y), axis=0)
            return x

        self.assertEqual(get_op_types_in_program(prog), ['concat'])

        apply_pass_and_basic_check(prog, transform.FP16ComputePrecision(op_selector=lambda op: True))
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        self.assertEqual(get_op_types_in_program(prog), ["cast", "cast", "concat", "cast"])

        # Asserting first cast configuration
        cast_1 = block.find_ops(op_type="cast")[0]
        self.assertEqual(cast_1.dtype.val, "fp16")
        self.assertEqual(len(cast_1.outputs), 1)
        self.assertEqual(len(cast_1.outputs[0].child_ops), 1)
        self.assertEqual(cast_1.outputs[0].child_ops[0].op_type, "concat")

        # Asserting second cast configuration
        cast_2 = block.find_ops(op_type="cast")[1]
        self.assertEqual(cast_2.dtype.val, "fp16")
        self.assertEqual(len(cast_2.outputs), 1)
        self.assertEqual(len(cast_2.outputs[0].child_ops), 1)
        self.assertEqual(cast_2.outputs[0].child_ops[0].op_type, "concat")


        # Asserting third cast configuration
        cast_3 = block.find_ops(op_type="cast")[2]
        self.assertEqual(cast_3.dtype.val, "fp32")
        self.assertEqual(len(cast_3.outputs), 1)
        self.assertEqual(len(cast_3.outputs[0].child_ops), 0)

        assert_model_is_valid(
            prog,
            {"x": (10, 20), "y": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (20, 20)},
        )


    """
    Input graph:
                            |-----> output_1
          input -----> split
                            |-----> output_2

    Output graph:

                                                     |-----> cast(dtype="fp32") ---> output_1
          input -----> cast(dtype="fp16") -----> split
                                                     |-----> cast(dtype="fp32") ---> output_2

    """


    def test_multiple_outputs_from_single_operation(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.split(x=x, axis=0, num_splits=2)
            return x

        self.assertEqual(get_op_types_in_program(prog), ['split'])

        apply_pass_and_basic_check(prog, transform.FP16ComputePrecision(op_selector=lambda op: True))
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        self.assertEqual(get_op_types_in_program(prog), ["cast", "split", "cast", "cast"])

        # Asserting first cast configuration
        cast_1 = block.find_ops(op_type="cast")[0]
        self.assertEqual(cast_1.dtype.val, "fp16")
        self.assertEqual(len(cast_1.outputs), 1)
        self.assertEqual(len(cast_1.outputs[0].child_ops), 1)
        self.assertEqual(cast_1.outputs[0].child_ops[0].op_type, "split")

        # Asserting second cast configuration
        cast_2 = block.find_ops(op_type="cast")[1]
        self.assertEqual(cast_2.dtype.val, "fp32")
        self.assertEqual(len(cast_2.outputs), 1)
        self.assertEqual(len(cast_2.outputs[0].child_ops), 0)

        # Asserting third cast configuration
        cast_3 = block.find_ops(op_type="cast")[2]
        self.assertEqual(cast_3.dtype.val, "fp32")
        self.assertEqual(len(cast_3.outputs), 1)
        self.assertEqual(len(cast_3.outputs[0].child_ops), 0)

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (5, 20), block.outputs[1].name: (5, 20)},
        )

    """
    Input graph:

         |----> square ---> output_1
    input|
         |----> relu   ---> output_2

    Output graph:

                                        |---->square-----> cast(dtype="fp32") ---> output_1
          input -----> cast(dtype="fp16")
                                        |----> relu -----> cast(dtype="fp32") ---> output_2

    """

    def test_single_input_to_multiple_operations(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            y = mb.square(x=x)
            z = mb.relu(x=x)
            return y,z

        self.assertEqual(get_op_types_in_program(prog), ['square', 'relu'])

        apply_pass_and_basic_check(prog, transform.FP16ComputePrecision(op_selector=lambda op: True))
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        self.assertEqual(get_op_types_in_program(prog), ["cast", "square", "cast", "relu", "cast"])

        # Asserting first cast configuration
        cast_1 = block.find_ops(op_type="cast")[0]
        self.assertEqual(cast_1.dtype.val, "fp16")
        self.assertEqual(len(cast_1.outputs), 1)
        self.assertEqual(len(cast_1.outputs[0].child_ops), 2)
        self.assertEqual(cast_1.outputs[0].child_ops[0].op_type, "square")
        self.assertEqual(cast_1.outputs[0].child_ops[1].op_type, "relu")

        # Asserting second cast configuration
        cast_2 = block.find_ops(op_type="cast")[1]
        self.assertEqual(cast_2.dtype.val, "fp32")
        self.assertEqual(len(cast_2.outputs), 1)
        self.assertEqual(len(cast_2.outputs[0].child_ops), 0)

        # Asserting third cast configuration
        cast_3 = block.find_ops(op_type="cast")[2]
        self.assertEqual(cast_3.dtype.val, "fp32")
        self.assertEqual(len(cast_3.outputs), 1)
        self.assertEqual(len(cast_3.outputs[0].child_ops), 0)

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (10, 20), block.outputs[1].name: (10, 20)},
        )
