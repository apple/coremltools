#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pytest

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check, assert_model_is_valid, get_op_types_in_program)

np.random.seed(1984)


class TestMergeConsecutivePaddings:
    
    def test_success_reflect(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):
            pad1 = mb.pad(x=x1, pad=[0, 0, 1, 1], mode='reflect')
            pad2 = mb.pad(x=pad1, pad=[1, 1, 0, 0], mode='reflect')

            return pad2

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::merge_consecutive_paddings"
        )
        assert get_op_types_in_program(prev_prog) == ["pad", "pad"]
        assert get_op_types_in_program(prog) == ["pad"]

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 8, 10)},
        )

    @pytest.mark.parametrize("swap_axes", [False, True])
    def test_success_different_rank1(self, swap_axes):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):
            if swap_axes: 
                pad1 = mb.pad(x=x1, pad=[1, 1], mode='reflect')
                pad2 = mb.pad(x=pad1, pad=[1, 1, 0, 0], mode='reflect')
            else: 
                pad1 = mb.pad(x=x1, pad=[1, 1, 0, 0], mode='reflect')
                pad2 = mb.pad(x=pad1, pad=[1, 1], mode='reflect')

            return pad2

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::merge_consecutive_paddings"
        )
        assert get_op_types_in_program(prev_prog) == ["pad", "pad"]
        assert get_op_types_in_program(prog) == ["pad"]

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 8, 10)},
        )

    def test_success_constant(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):
            pad1 = mb.pad(x=x1, pad=[0, 0, 1, 1], mode='constant', constant_val=3.0)
            pad2 = mb.pad(x=pad1, pad=[1, 1, 0, 0], mode='constant', constant_val=3.0)

            return pad2

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::merge_consecutive_paddings"
        )
        assert get_op_types_in_program(prev_prog) == ["pad", "pad"]
        assert get_op_types_in_program(prog) == ["pad"]

        pad_ops = [op for op in prog["main"].operations if op.op_type == "pad"]
        assert pad_ops[0].inputs["constant_val"].val == 3.0

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 8, 10)},
        )

    def test_success_3_layers(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):
            pad1 = mb.pad(x=x1, pad=[0, 0, 1, 1], mode='constant', constant_val=3.0)
            pad2 = mb.pad(x=pad1, pad=[1, 1, 0, 0], mode='constant', constant_val=3.0)
            pad3 = mb.pad(x=pad2, pad=[1, 1, 0, 0], mode='constant', constant_val=3.0)

            return pad3

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::merge_consecutive_paddings"
        )
        assert get_op_types_in_program(prev_prog) == ["pad", "pad", "pad"]
        assert get_op_types_in_program(prog) == ["pad"]

        pad_ops = [op for op in prog["main"].operations if op.op_type == "pad"]
        assert pad_ops[0].inputs["constant_val"].val == 3.0

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 10, 10)},
        )

    def test_failure_different_mode(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):

            pad1 = mb.pad(x=x1, pad=[0, 0, 1, 1], mode='reflect')
            pad2 = mb.pad(x=pad1, pad=[1, 1, 0, 0], mode='constant')

            return pad2

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::merge_consecutive_paddings"
        )
        assert get_op_types_in_program(prev_prog) == ["pad", "pad"]
        assert get_op_types_in_program(prog) == ["pad", "pad"]

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 8, 10)},
        )

    def test_failure_different_constants(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):
            
            pad1 = mb.pad(x=x1, pad=[0, 0, 1, 1], mode='constant', constant_val=1.0)
            pad2 = mb.pad(x=pad1, pad=[1, 1, 0, 0], mode='constant', constant_val=2.0)

            return pad2

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::merge_consecutive_paddings"
        )
        assert get_op_types_in_program(prev_prog) == ["pad", "pad"]
        assert get_op_types_in_program(prog) == ["pad", "pad"]

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 8, 10)},
        )

    def test_failure_repeat_on_same_axis(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):
            
            pad1 = mb.pad(x=x1, pad=[1, 1], mode='reflect')
            pad2 = mb.pad(x=pad1, pad=[1, 1], mode='reflect')

            return pad2

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::merge_consecutive_paddings"
        )
        assert get_op_types_in_program(prev_prog) == ["pad", "pad"]
        assert get_op_types_in_program(prog) == ["pad", "pad"]

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 6, 12)},
        )
