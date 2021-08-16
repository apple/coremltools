#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    assert_model_is_valid,
    get_op_types_in_program,
    get_op_names_in_program,
    apply_pass_and_basic_check,
)
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
import unittest

class OpNameDeduplicationPass(unittest.TestCase):

    def test_unchanged(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            x = mb.reshape(x=x, shape=(1, 8), name="reshape")
            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog,
                "common::dedup_op_and_var_names")

        self.assertEqual(get_op_types_in_program(prev_prog), ['reshape'])
        self.assertEqual(get_op_names_in_program(prev_prog), ['reshape'])

        self.assertEqual(get_op_types_in_program(prog), ['reshape'])
        self.assertEqual(get_op_names_in_program(prog), ['reshape'])

        assert_model_is_valid(
            prog,
            {"x": (2, 4)},
            expected_output_shapes={block.outputs[0].name: (1, 8)},
        )

    def test_op_name_duplicated_once(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16", name="castop")
            x = mb.cast(x=x, dtype="fp32", name="castop")
            x = mb.square(x=x, name="square_last")
            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog,
                "common::dedup_op_and_var_names")

        self.assertEqual(get_op_types_in_program(prev_prog),
                ['cast', 'cast', 'square'])
        self.assertEqual(get_op_names_in_program(prev_prog),
                ['castop', 'castop', 'square_last'])

        self.assertEqual(get_op_types_in_program(prog),
                ['cast', 'cast', 'square'])
        self.assertEqual(get_op_names_in_program(prog),
                ['castop', 'castop_1', 'square_last'])

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (10, 20)},
        )

    def test_op_name_duplicated_many(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16", name="castop")
            x = mb.cast(x=x, dtype="fp16", name="castop")
            x = mb.cast(x=x, dtype="int32", name="castop_2")
            x = mb.cast(x=x, dtype="int64", name="castop")
            x = mb.cast(x=x, dtype="fp32", name="castop_2")
            x = mb.square(x=x, name="square")
            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog,
                "common::dedup_op_and_var_names")

        self.assertEqual(get_op_types_in_program(prev_prog),
                ['cast', 'cast', 'cast', 'cast', 'cast', 'square'])
        self.assertEqual(get_op_names_in_program(prev_prog),
                ['castop', 'castop', 'castop_2', 'castop', 'castop_2', 'square'])

        self.assertEqual(get_op_types_in_program(prog),
                ['cast', 'cast', 'cast', 'cast', 'cast', 'square'])
        self.assertEqual(get_op_names_in_program(prog),
                ['castop', 'castop_1', 'castop_2', 'castop_3',
                    'castop_2_1', 'square'])

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (10, 20)},
        )


    def test_input_name_shadow(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            # op name "x" results in output var name "x", which shadows prog
            # input var name "x"
            x = mb.transpose(x=x, perm=[1, 0], name="x")
            x = mb.relu(x=x, name="relu")
            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog,
                "common::dedup_op_and_var_names")
        self.assertEqual(get_op_types_in_program(prev_prog),
                ['transpose', 'relu'])
        self.assertEqual(get_op_names_in_program(prev_prog),
                ['x', 'relu'])

        self.assertEqual(get_op_types_in_program(prog),
                ['transpose', 'relu'])
        self.assertEqual(get_op_names_in_program(prog),
                ['x', 'relu'])

        op = prog['main'].find_ops(op_type='transpose')[0]
        self.assertEqual("x_1", op.outputs[0].name)

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (20, 10)},
        )

    def test_nested_block(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1,))])
        def prog(x):
            def true_fn():
                # returns var with name x shadows input 'x'
                return mb.add(x=x, y=1, name='x')

            def false_fn():
                # two ops with name "x"
                return mb.add(x=x, y=-1, name='x')

            pred = mb.equal(x=mb.squeeze(x=x), y=1)
            return mb.cond(pred=pred, _true_fn=true_fn, _false_fn=false_fn)

        cond_op = prog.functions['main'].operations[-1]
        assert cond_op.blocks[0].outputs[0].name == 'x'
        assert cond_op.blocks[1].outputs[0].name == 'x'
        prev_prog, _, block = apply_pass_and_basic_check(prog,
                "common::dedup_op_and_var_names")
        cond_op = prog.functions['main'].operations[-1]
        assert cond_op.blocks[0].outputs[0].name == 'x_1'
        assert cond_op.blocks[1].outputs[0].name == 'x_2'

        assert_model_is_valid(
            prog,
            {"x": (1,)},
            expected_output_shapes={block.outputs[0].name: (1,)},
        )
