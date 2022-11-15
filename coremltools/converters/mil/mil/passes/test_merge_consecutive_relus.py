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


class TestMergeConsecutiveRelus:

    @pytest.mark.parametrize(
        "relu_num",
        [2, 3, 4],
    )
    def test_success_reduce_consecutive_relus(self, relu_num):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3))])
        def prog(x):
            for _ in range(relu_num):
                x = mb.relu(x=x)
            x = mb.add(x=x, y=1.0)
            return x

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::merge_consecutive_relus"
        )
        assert get_op_types_in_program(prev_prog) == ["relu"] * relu_num + ["add"]
        assert get_op_types_in_program(prog) == ["relu", "add"]

        inputs = {"x": (1, 2, 3)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 3)},
        )

    @pytest.mark.parametrize(
        "relu_num",
        [2, 3, 4],
    )
    def test_keep_not_consecutive_relus(self, relu_num):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3))])
        def prog(x):
            for _ in range(relu_num):
                x = mb.relu(x=x)
                x = mb.add(x=x, y=1.0)
            return x

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::merge_consecutive_relus"
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "add"] * relu_num
        assert get_op_types_in_program(prog) == get_op_types_in_program(prev_prog)

        inputs = {"x": (1, 2, 3)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 3)},
        )

    def test_mix_situation(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3))])
        def prog(x):
            relu1 = mb.relu(x=x)
            relu_after_add = mb.add(x=relu1, y=1.0)
            relu2 = mb.relu(x=relu_after_add)
            relu3 = mb.relu(x=relu2)
            return relu3

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::merge_consecutive_relus"
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "add", "relu", "relu"]
        assert get_op_types_in_program(prog) == ["relu", "add", "relu"]

        inputs = {"x": (1, 2, 3)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 3)},
        )

    def test_name_change_depend_on_output(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3))])
        def prog_output_transpose_2(x):
            transpose_1 = mb.relu(x=x, name="transpose_1")
            transpose_2 = mb.relu(x=transpose_1, name="transpose_2")
            transpose_3 = mb.transpose(x=transpose_2, perm=[0, 2, 1], name="transpose_3")
            return transpose_2, transpose_3

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3))])
        def prog_output_transpose_3(x):
            transpose_1 = mb.relu(x=x, name="transpose_1")
            transpose_2 = mb.relu(x=transpose_1, name="transpose_2")
            transpose_3 = mb.transpose(x=transpose_2, perm=[0, 2, 1], name="transpose_3")
            return transpose_3

        prev_prog_output_transpose_2, _, block = apply_pass_and_basic_check(
            prog_output_transpose_2, "common::merge_consecutive_relus"
        )
        assert get_op_types_in_program(prev_prog_output_transpose_2) == ["relu", "relu", "transpose"]
        assert get_op_types_in_program(prog_output_transpose_2) == ["relu", "transpose"]
        assert prog_output_transpose_2['main'].operations[0].name == "transpose_1"
        # As the block's output has transpose_2, the original output name of the first operation
        # is replaced.
        assert prog_output_transpose_2['main'].operations[0].outputs[0].name == "transpose_2"

        prev_prog_output_transpose_3, _, block = apply_pass_and_basic_check(
            prog_output_transpose_3, "common::merge_consecutive_relus"
        )
        assert get_op_types_in_program(prev_prog_output_transpose_3) == ["relu", "relu", "transpose"]
        assert get_op_types_in_program(prog_output_transpose_3) == ["relu", "transpose"]
        assert prog_output_transpose_3['main'].operations[0].name == "transpose_1"
        # As the block's output only has transpose_3, the entire transpose_2 gets removed.
        assert prog_output_transpose_3['main'].operations[0].outputs[0].name == "transpose_1"

        inputs = {"x": (1, 2, 3)}
        assert_model_is_valid(
            prog_output_transpose_2,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 3, 2)},
        )

        inputs = {"x": (1, 2, 3)}
        assert_model_is_valid(
            prog_output_transpose_3,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 3, 2)},
        )
