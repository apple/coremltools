#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    assert_model_is_valid,
    get_op_types_in_program,
    apply_pass_and_basic_check,
)
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
import copy
import pytest
import itertools

import numpy as np


@pytest.mark.parametrize("op_type, pos, val", itertools.product(['add', 'mul', 'floor_div', 'pow', 'real_div', 'sub'], ['x', 'y'], [0, 1, [0, 0, 0, 0], [1, 1, 1, 1]]))
def test_elementwise_elimination(op_type, pos, val):
    if 'div' in op_type and np.prod(val) == 0:
        return
    if 'pow' in op_type and (val != 0 or val != 1):
        return

    test_op = getattr(mb, op_type)

    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
    def prog(x):
        if pos == "x":
            r1 = test_op(x=val, y=x)
        else:
            r1 = test_op(x=x, y=val)
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    original_program = [op_type, "relu"]
    new_program = original_program
    if op_type in {'add'}:
        if val == 0 or val == [0, 0, 0, 0]:
            new_program = ["relu"]
    elif op_type in {'mul'}:
        if val == 1 or val == [1, 1, 1, 1]:
            new_program = ["relu"]
    elif op_type in {'pow', 'real_div', 'floor_div'}:
        if pos == 'y' and (val == 1 or val == [1, 1, 1, 1]):
            new_program = ["relu"]
    elif op_type in {'sub'}:
        if pos == 'y' and (val == 0 or val == [0, 0, 0, 0]):
            new_program = ["relu"]
            
    assert get_op_types_in_program(prev_prog) == original_program
    assert get_op_types_in_program(prog) == new_program
    assert_model_is_valid(
        prog,
        {"x": (2, 4)},
        expected_output_shapes={block.outputs[0].name: (2, 4)},
    )

def test_elementwise_broadcast():

    @mb.program(input_specs=[mb.TensorSpec(shape=[4])])
    def prog(x):
        r1 = mb.add(x=x, y=[[0, 0, 0, 0], [0, 0, 0, 0]])
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    original_program = ["add", "relu"]

    assert get_op_types_in_program(prev_prog) == original_program
    assert get_op_types_in_program(prog) == original_program
    assert_model_is_valid(
        prog,
        {"x": [4]},
        expected_output_shapes={block.outputs[0].name: (2, 4)},
    )

def test_reshape_elimination():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
    def prog(x):
        r1 = mb.reshape(x=x, shape=[1, 8])
        r2 = mb.reshape(x=r1, shape=[1, 8])
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    assert get_op_types_in_program(prev_prog) == ["reshape", "reshape", "relu"]
    assert get_op_types_in_program(prog) == ["reshape", "relu"]
    assert_model_is_valid(
        prog,
        {"x": (2, 4)},
        expected_output_shapes={block.outputs[0].name: (1, 8)},
    )


def test_oneway_split_elimination():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
    def prog(x):
        r1 = mb.split(x=x, num_splits=1, axis=-1) 
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    assert get_op_types_in_program(prev_prog) == ["split", "relu"]
    assert get_op_types_in_program(prog) == ["relu"]
    assert_model_is_valid(
        prog,
        {"x": (2, 4)},
        expected_output_shapes={block.outputs[0].name: (2, 4)},
    )


def test_full_split_elimination():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
    def prog(x):
        r1 = mb.split(x=x, split_sizes=[4], axis=-1) 
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    assert get_op_types_in_program(prev_prog) == ["split", "relu"]
    assert get_op_types_in_program(prog) == ["relu"]
    assert_model_is_valid(
        prog,
        {"x": (2, 4)},
        expected_output_shapes={block.outputs[0].name: (2, 4)},
    )


def test_slicebysize_full_elimination():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
    def prog(x):
        r1 = mb.slice_by_size(x=x, begin=[0, 0], size=[2, 4]) 
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    assert get_op_types_in_program(prev_prog) == ["slice_by_size", "relu"]
    assert get_op_types_in_program(prog) == ["relu"]
    assert_model_is_valid(
        prog,
        {"x": (2, 4)},
        expected_output_shapes={block.outputs[0].name: (2, 4)},
    )


def test_slicebysize_to_end_elimination():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
    def prog(x):
        r1 = mb.slice_by_size(x=x, begin=[0, 0], size=[-1, -1]) 
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    assert get_op_types_in_program(prev_prog) == ["slice_by_size", "relu"]
    assert get_op_types_in_program(prog) == ["relu"]
    assert_model_is_valid(
        prog,
        {"x": (2, 4)},
        expected_output_shapes={block.outputs[0].name: (2, 4)},
    )


def test_slicebyindex_full_elimination():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
    def prog(x):
        r1 = mb.slice_by_index(x=x, begin=[0, 0], end=[2, 4]) 
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    assert get_op_types_in_program(prev_prog) == ["slice_by_index", "relu"]
    assert get_op_types_in_program(prog) == ["relu"]
    assert_model_is_valid(
        prog,
        {"x": (2, 4)},
        expected_output_shapes={block.outputs[0].name: (2, 4)},
    )


@pytest.mark.parametrize("begin_mask, end_mask",
                         itertools.product(itertools.product([True, False],[True, False]),
                                           itertools.product([True, False],[True, False])))
def test_slicebyindex_mask_elimination(begin_mask, end_mask):
    @mb.program(input_specs=[mb.TensorSpec(shape=(4, 4))])
    def prog(x):
        begin = [1, 1]
        end = [1, 1]
        for i in range(2):
            if not begin_mask[i]:
                begin[i] = 0
            if not end_mask[i]:
                end[i] = 4
        r1 = mb.slice_by_index(x=x, begin=begin, end=end, begin_mask=begin_mask, end_mask=end_mask) 
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    assert get_op_types_in_program(prev_prog) == ["slice_by_index", "relu"]
    assert get_op_types_in_program(prog) == ["relu"]
    assert_model_is_valid(
        prog,
        {"x": (4, 4)},
        expected_output_shapes={block.outputs[0].name: (4, 4)},
    )


def test_pad_elimination():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
    def prog(x):
        r1 = mb.pad(x=x, pad=[0, 0, 0, 0]) 
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    assert get_op_types_in_program(prev_prog) == ["pad", "relu"]
    assert get_op_types_in_program(prog) == ["relu"]
    assert_model_is_valid(
        prog,
        {"x": (2, 4)},
        expected_output_shapes={block.outputs[0].name: (2, 4)},
    )


def test_keep_pad():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
    def prog(x):
        r1 = mb.pad(x=x, pad=[4, 4, 2, 2]) 
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    assert get_op_types_in_program(prev_prog) == ["pad", "relu"]
    assert get_op_types_in_program(prog) == ["pad", "relu"]
    assert_model_is_valid(
        prog,
        {"x": (2, 4)},
        expected_output_shapes={block.outputs[0].name: (10, 8)},
    )


def test_tile_elimination():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
    def prog(x):
        r1 = mb.tile(x=x, reps=[1, 1]) 
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    assert get_op_types_in_program(prev_prog) == ["tile", "relu"]
    assert get_op_types_in_program(prog) == ["relu"]
    assert_model_is_valid(
        prog,
        {"x": (2, 4)},
        expected_output_shapes={block.outputs[0].name: (2, 4)},
    )


def test_keep_tile():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
    def prog(x):
        r1 = mb.tile(x=x, reps=[2, 2]) 
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    assert get_op_types_in_program(prev_prog) == ["tile", "relu"]
    assert get_op_types_in_program(prog) == ["tile", "relu"]
    assert_model_is_valid(
        prog,
        {"x": (2, 4)},
        expected_output_shapes={block.outputs[0].name: (4, 8)},
    )


def test_upsample_nearest_neighbor_elimination():
    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 2, 4))])
    def prog(x):
        r1 = mb.upsample_nearest_neighbor(x=x) 
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    assert get_op_types_in_program(prev_prog) == ["upsample_nearest_neighbor", "relu"]
    assert get_op_types_in_program(prog) == ["relu"]
    assert_model_is_valid(
        prog,
        {"x": (3, 2, 4)},
        expected_output_shapes={block.outputs[0].name: (3, 2, 4)},
    )


def test_upsample_bilinear_elimination():
    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 2, 4))])
    def prog(x):
        r1 = mb.upsample_bilinear(x=x) 
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    assert get_op_types_in_program(prev_prog) == ["upsample_bilinear", "relu"]
    assert get_op_types_in_program(prog) == ["relu"]
    assert_model_is_valid(
        prog,
        {"x": (3, 2, 4)},
        expected_output_shapes={block.outputs[0].name: (3, 2, 4)},
    )


def test_resize_bilinear_elimination():
    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 2, 4))])
    def prog(x):
        r1 = mb.resize_bilinear(x=x, target_size_height=2, target_size_width=4) 
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    assert get_op_types_in_program(prev_prog) == ["resize_bilinear", "relu"]
    assert get_op_types_in_program(prog) == ["relu"]
    assert_model_is_valid(
        prog,
        {"x": (3, 2, 4)},
        expected_output_shapes={block.outputs[0].name: (3, 2, 4)},
    )


def test_crop_elimination():
    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 2, 4))])
    def prog(x):
        r1 = mb.crop(x=x, crop_height=[0, 0], crop_width=[0, 0]) 
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    assert get_op_types_in_program(prev_prog) == ["crop", "relu"]
    assert get_op_types_in_program(prog) == ["relu"]
    assert_model_is_valid(
        prog,
        {"x": (3, 2, 4)},
        expected_output_shapes={block.outputs[0].name: (3, 2, 4)},
    )


def test_linear_elimination():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
    def prog(x):
        r1 = mb.linear_activation(x=x, alpha=1.0, beta=0.0) 
        return mb.relu(x=r1)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::noop_elimination"
    )
    assert get_op_types_in_program(prev_prog) == ["linear_activation", "relu"]
    assert get_op_types_in_program(prog) == ["relu"]
    assert_model_is_valid(
        prog,
        {"x": (2, 4)},
        expected_output_shapes={block.outputs[0].name: (2, 4)},
    )


