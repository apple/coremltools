#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools._deps import _IS_MACOS
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    assert_model_is_valid,
    get_op_types_in_program,
    apply_pass_and_basic_check,
)
from coremltools.converters.mil.testing_reqs import ct

import pytest

import numpy as np

np.random.seed(1984)


class TestUseReflectionPadding:
    
    def test_success_w_axis(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):
            left = mb.slice_by_index(x=x1, begin=[0, 0, 0, 1], end=[0, 0, 0, 2], end_mask=[True, True, True, False])
            right = mb.slice_by_index(x=x1, begin=[0, 0, 0, -2], end=[0, 0, 0, -1], end_mask=[True, True, True, False])
            x = mb.concat(values=[left, x1, right], axis=3)

            return x

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::use_reflection_padding"
        )
        assert get_op_types_in_program(prev_prog) == ["slice_by_index", "slice_by_index", "concat"]
        assert get_op_types_in_program(prog) == ["pad"]

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 6, 10)},
        )

    def test_success_w_axis_multiple(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):
            left0 = mb.slice_by_index(x=x1, begin=[0, 0, 0, 2], end=[0, 0, 0, 3], end_mask=[True, True, True, False])
            left1 = mb.slice_by_index(x=x1, begin=[0, 0, 0, 1], end=[0, 0, 0, 2], end_mask=[True, True, True, False])
            right0 = mb.slice_by_index(x=x1, begin=[0, 0, 0, -2], end=[0, 0, 0, -1], end_mask=[True, True, True, False])
            right1 = mb.slice_by_index(x=x1, begin=[0, 0, 0, -3], end=[0, 0, 0, -2], end_mask=[True, True, True, False])
            x = mb.concat(values=[left0, left1, x1, right0, right1], axis=3)

            return x

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::use_reflection_padding"
        )
        assert get_op_types_in_program(prev_prog) == ["slice_by_index", "slice_by_index", "slice_by_index", "slice_by_index", "concat"]
        assert get_op_types_in_program(prog) == ["pad"]

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 6, 12)},
        )

    def test_success_h_axis(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):
            left = mb.slice_by_index(x=x1, begin=[0, 0, 1, 0], end=[0, 0, 2, 0], end_mask=[True, True, False, True])
            right = mb.slice_by_index(x=x1, begin=[0, 0, -2, 0], end=[0, 0, -1, 0], end_mask=[True, True, False, True])
            x = mb.concat(values=[left, x1, right], axis=2)

            return x

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::use_reflection_padding"
        )
        assert get_op_types_in_program(prev_prog) == ["slice_by_index", "slice_by_index", "concat"]
        assert get_op_types_in_program(prog) == ["pad"]

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 8, 8)},
        )        

    def test_failure_wrong_concat_order(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):
            left = mb.slice_by_index(x=x1, begin=[0, 0, 1, 0], end=[0, 0, 2, 0], end_mask=[True, True, False, True])
            right = mb.slice_by_index(x=x1, begin=[0, 0, -2, 0], end=[0, 0, -1, 0], end_mask=[True, True, False, True])
            # Concat is not in correct order 
            x = mb.concat(values=[left, right, x1], axis=2)

            return x

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::use_reflection_padding"
        )
        assert get_op_types_in_program(prev_prog) == ["slice_by_index", "slice_by_index", "concat"]
        assert get_op_types_in_program(prog) == ["slice_by_index", "slice_by_index", "concat"]

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 8, 8)},
        )        

    def test_failure_wrong_concat_order_2(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):
            left0 = mb.slice_by_index(x=x1, begin=[0, 0, 0, 1], end=[0, 0, 0, 2], end_mask=[True, True, True, False])
            left1 = mb.slice_by_index(x=x1, begin=[0, 0, 0, 2], end=[0, 0, 0, 3], end_mask=[True, True, True, False])
            right0 = mb.slice_by_index(x=x1, begin=[0, 0, 0, -3], end=[0, 0, 0, -2], end_mask=[True, True, True, False])
            right1 = mb.slice_by_index(x=x1, begin=[0, 0, 0, -2], end=[0, 0, 0, -1], end_mask=[True, True, True, False])
            # concat args are out of order
            x = mb.concat(values=[left0, left1, x1, right1, right0], axis=3)

            return x

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::use_reflection_padding"
        )
        assert get_op_types_in_program(prev_prog) == ["slice_by_index", "slice_by_index", "slice_by_index", "slice_by_index", "concat"]
        assert get_op_types_in_program(prog) == ["slice_by_index", "slice_by_index", "slice_by_index", "slice_by_index", "concat"]

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 6, 12)},
        )

    def test_failure_wrong_slice_size(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):
            # slice is too big
            left = mb.slice_by_index(x=x1, begin=[0, 0, 1, 0], end=[0, 0, 3, 0], end_mask=[True, True, False, True])
            right = mb.slice_by_index(x=x1, begin=[0, 0, -2, 0], end=[0, 0, -1, 0], end_mask=[True, True, False, True])
            x = mb.concat(values=[left, x1, right], axis=2)

            return x

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::use_reflection_padding"
        )
        assert get_op_types_in_program(prev_prog) == ["slice_by_index", "slice_by_index", "concat"]
        assert get_op_types_in_program(prog) == ["slice_by_index", "slice_by_index", "concat"]

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 9, 8)},
        )        

    def test_failure_not_all_same_input(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8)), mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1, x2):
            left0 = mb.slice_by_index(x=x1, begin=[0, 0, 0, 1], end=[0, 0, 0, 2], end_mask=[True, True, True, False])
            left1 = mb.slice_by_index(x=x1, begin=[0, 0, 0, 2], end=[0, 0, 0, 3], end_mask=[True, True, True, False])
            right0 = mb.slice_by_index(x=x1, begin=[0, 0, 0, -3], end=[0, 0, 0, -2], end_mask=[True, True, True, False])
            # one of the slices consumes a different input from the others
            right1 = mb.slice_by_index(x=x2, begin=[0, 0, 0, -2], end=[0, 0, 0, -1], end_mask=[True, True, True, False])
            x = mb.concat(values=[left0, left1, x1, right0, right1], axis=3)

            return x

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::use_reflection_padding"
        )
        assert get_op_types_in_program(prev_prog) == ["slice_by_index", "slice_by_index", "slice_by_index", "slice_by_index", "concat"]
        assert get_op_types_in_program(prog) == ["slice_by_index", "slice_by_index", "slice_by_index", "slice_by_index", "concat"]

        inputs = {"x1": (1, 2, 6, 8), "x2": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 6, 12)},
        )

    def test_failure_slice_output(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):
            left = mb.slice_by_index(x=x1, begin=[0, 0, 0, 1], end=[0, 0, 0, 2], end_mask=[True, True, True, False])
            right = mb.slice_by_index(x=x1, begin=[0, 0, 0, -2], end=[0, 0, 0, -1], end_mask=[True, True, True, False])
            x = mb.concat(values=[left, x1, right], axis=3)

            # slice is an output
            return x, right

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::use_reflection_padding"
        )
        assert get_op_types_in_program(prev_prog) == ["slice_by_index", "slice_by_index", "concat"]
        assert get_op_types_in_program(prog) == ["slice_by_index", "slice_by_index", "concat"]

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 6, 10), block.outputs[1].name: (1, 2, 6, 1)},
        )
