#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools._deps import _IS_MACOS
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    assert_op_count_match,
    assert_model_is_valid,
    get_op_types_in_program,
    apply_pass_and_basic_check,
)
from coremltools.converters.mil.testing_reqs import ct

import unittest
import pytest

import numpy as np

np.random.seed(1984)

class ReplaceStackReshapePass(unittest.TestCase):
    
    def test_with_interleave(self):
        """
        input1(1, 5, 3, 4) -----> stack(axis=2) -----> reshape(shape=(1, 10, 3, 4)) ---> out(1, 10, 3, 4)
                                    ^
                                    |
        input2(1, 5, 3, 4) ----------

        Output graph:
        input -----> concat ----> out

        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))])
        def prog(x1, x2):
            x = mb.stack(values=[x1, x2], axis=2)
            x = mb.reshape(x=x, shape=[1, 10, 3, 4])
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["stack", "reshape"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["concat"])

        inputs = {"x1": (1, 5, 3, 4), "x2": (1, 5, 3, 4)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 10, 3, 4)},
        )

        concat_ops = [op for op in block.operations if op.op_type == 'concat']
        concat_op = concat_ops[0]
        assert concat_op.interleave.val == True

        output_name = block.outputs[0].name

        mlmodel = ct.convert(prog, source="milinternal", convert_to="neuralnetwork")

        if not _IS_MACOS:
            # Can not get predictions unless on macOS.
            return

        input_dict = dict()
        for name, shape in inputs.items():
            input_dict[name] = np.random.rand(*shape)

        old_prediction = np.reshape(np.stack([input_dict["x1"], input_dict["x2"]], axis=2), newshape=[1, 10, 3, 4])

        prediction = mlmodel.predict(input_dict, useCPUOnly=True)

        np.testing.assert_allclose(old_prediction, prediction[output_name], atol=1e-04, rtol=1e-05)

    def test_without_interleave(self):
        """
        Input graph:
        input1(1, 5, 3, 4) -----> stack(axis=1) -----> reshape(shape=(1, 10, 3, 4)) ---> out(1, 10, 3, 4)
                                    ^
                                    |
        input2(1, 5, 3, 4) ----------

        Output graph:
        input -----> concat ----> out
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))])
        def prog(x1, x2):
            x = mb.stack(values=[x1, x2], axis=1)
            x = mb.reshape(x=x, shape=[1, 10, 3, 4])
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["stack", "reshape"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["concat"])

        inputs = {"x1": (1, 5, 3, 4), "x2": (1, 5, 3, 4)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 10, 3, 4)},
        )

        concat_ops = [op for op in block.operations if op.op_type == 'concat']
        concat_op = concat_ops[0]
        assert concat_op.interleave.val == False

        output_name = block.outputs[0].name

        mlmodel = ct.convert(prog, source="milinternal", convert_to="neuralnetwork")

        if not _IS_MACOS:
            # Can not get predictions unless on macOS.
            return

        input_dict = dict()
        for name, shape in inputs.items():
            input_dict[name] = np.random.rand(*shape)

        old_prediction = np.reshape(np.stack([input_dict["x1"], input_dict["x2"]], axis=1), newshape=[1, 10, 3, 4])

        prediction = mlmodel.predict(input_dict, useCPUOnly=True)
        np.testing.assert_allclose(old_prediction, prediction[output_name], atol=1e-04, rtol=1e-05)

    def test_multiple(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4)), 
                                 mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x1, x2, x3, x4):
            a = mb.stack(values=[x1, x2], axis=1)
            a = mb.reshape(x=a, shape=[1, 4, 3, 4])

            b = mb.stack(values=[x3, x4], axis=1)
            b = mb.reshape(x=b, shape=[1, 4, 3, 4])

            c = mb.stack(values=[a, b], axis=2) 
            c = mb.reshape(x=c, shape=[1, 4, 6, 4])

            return c

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["stack", "reshape", "stack", "reshape", "stack", "reshape"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])

        inputs = {"x1": (1, 2, 3, 4), "x2": (1, 2, 3, 4), "x3": (1, 2, 3, 4), "x4": (1, 2, 3, 4)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 4, 6, 4)},
        )

        output_name = block.outputs[0].name

        mlmodel = ct.convert(prog, source="milinternal", convert_to="neuralnetwork")

        if not _IS_MACOS:
            # Can not get predictions unless on macOS.
            return

        input_dict = dict()
        for name, shape in inputs.items():
            input_dict[name] = np.random.rand(*shape)

        branch_1 = np.reshape(np.stack([input_dict['x1'], input_dict['x2']], axis=1), newshape=[1, 4, 3, 4])
        branch_2 = np.reshape(np.stack([input_dict['x3'], input_dict['x4']], axis=1), newshape=[1, 4, 3, 4])
        old_prediction = np.reshape(np.stack([branch_1, branch_2], axis=2), newshape=[1, 4, 6, 4])

        prediction = mlmodel.predict(input_dict, useCPUOnly=True)

        np.testing.assert_allclose(old_prediction, prediction[output_name], atol=1e-04, rtol=1e-05)

    def test_negative_1(self):
        """
        Input graph:
        input1(1, 5, 3, 4) -----> stack(axis=1) -----> reshape(shape=(-1, 5, 6, 4)) ---> out(1, 5, 6, 4)
                                    ^
                                    |
        input2(1, 5, 3, 4) ----------

        Output graph:
        Unchanged -- this graph is not equivalent to a concat. 
        """        
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))])
        def prog(x1, x2):
            a = mb.stack(values=[x1, x2], axis=1)
            a = mb.reshape(x=a, shape=[-1, 5, 6, 4])
            return a

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )

        self.assertEqual(
            get_op_types_in_program(prev_prog), ["stack", "reshape"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["stack", "reshape"])
    
    def test_negative_2(self):
        """
        Input graph:
        input1(1, 5, 3, 4) -----> stack(axis=1) -----> reshape(shape=(-1, 5, 12, 2)) ---> out(1, 5, 6, 4)
                                    ^
                                    |
        input2(1, 5, 3, 4) ----------

        Output graph:
        Unchanged -- this graph is not equivalent to a concat. 
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))])
        def prog(x1, x2):
            a = mb.stack(values=[x1, x2], axis=1)
            a = mb.reshape(x=a, shape=[-1, 5, 12, 2])
            return a

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )

        self.assertEqual(
            get_op_types_in_program(prev_prog), ["stack", "reshape"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["stack", "reshape"])

    def test_negative_3(self):
        """
        Input graph:
        input1(1, 5, 3, 4) -----> stack(axis=1) -----> reshape(shape=(-1, 2, 5, 4, 3)) ---> out(1, 5, 6, 4)
                                    ^
                                    |
        input2(1, 5, 3, 4) ----------

        Output graph:
        Unchanged -- this graph is not equivalent to a concat. 
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))])
        def prog(x1, x2):
            a = mb.stack(values=[x1, x2], axis=1)
            a = mb.reshape(x=a, shape=[-1, 2, 5, 4, 3])
            return a

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )

        self.assertEqual(
            get_op_types_in_program(prev_prog), ["stack", "reshape"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["stack", "reshape"])

    def test_negative_4(self):
        """
        More than two inputs to the stack op -- can't be transformed. 
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))])
        def prog(x1, x2, x3):
            a = mb.stack(values=[x1, x2, x3], axis=1)
            a = mb.reshape(x=a, shape=[-1, 15, 4, 3])
            return a

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )

        self.assertEqual(
            get_op_types_in_program(prev_prog), ["stack", "reshape"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["stack", "reshape"])

    def test_negative_5(self):
        """
        The stack and reshape are not adjacent, so the graph is not transformed. 
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))])
        def prog(x1, x2):
            a = mb.stack(values=[x1, x2], axis=1)
            a = mb.relu(x=a)
            a = mb.reshape(x=a, shape=[-1, 10, 4, 3])
            return a

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )

        self.assertEqual(
            get_op_types_in_program(prev_prog), ["stack", "relu", "reshape"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["stack", "relu", "reshape"])

    def test_negative_6(self):
        """
        The stack op's output is used elsewhere in the graph, so it can't be removed
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))])
        def prog(x1, x2):
            a = mb.stack(values=[x1, x2], axis=1)
            b = mb.reshape(x=a, shape=[-1, 10, 4, 3])
            c = mb.relu(x=a)
            c = mb.reshape(x=c, shape=[-1, 10, 4, 3])
            d = mb.add(x=b, y=c)
            return d

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )

        self.assertEqual(
            get_op_types_in_program(prev_prog), ["stack", "reshape", "relu", "reshape", "add"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["stack", "reshape", "relu", "reshape", "add"])

    def test_negative_7(self):
        """
        The stack op is not followed by any other ops.
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))])
        def prog(x1, x2):
            a = mb.stack(values=[x1, x2], axis=1)
            return a

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )

        self.assertEqual(
            get_op_types_in_program(prev_prog), ["stack"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["stack"])
