#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    assert_op_count_match,
    assert_model_is_valid,
    assert_same_output_names,
    get_op_types_in_program,
    apply_pass_and_basic_check,
)
from coremltools.converters.mil.mil import Symbol, types
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
import copy
import pytest
import itertools

import numpy as np

np.random.seed(1984)
validate_model = True



def test_const_elimination():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
    def prog(x):
        a = np.random.rand(2, 4).astype(np.float32)
        double_a = mb.add(x=a, y=a)
        return mb.add(x=x, y=double_a)

    assert_op_count_match(prog, expect=2, op="const")
    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY["common::const_elimination"](prog)
    assert_same_output_names(prev_prog, prog)
    assert_op_count_match(prog, expect=3, op="const")

    if validate_model:
        assert_model_is_valid(prog, {"x": (2, 4)})


def test_divide_to_multiply():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
    def prog(x):
        div_val = np.random.rand(2, 4).astype(np.float32)
        div_const = mb.const(val=div_val)

        div_val_1 = np.random.rand(2, 4).astype(np.float32)
        div_const_1 = mb.const(val=div_val_1)

        real_div = mb.real_div(x=x, y=div_const)

        return mb.real_div(x=real_div, y=div_const_1)

    assert_op_count_match(prog, expect=2, op="real_div")
    assert_op_count_match(prog, expect=0, op="mul")
    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY["common::divide_to_multiply"](prog)
    assert_same_output_names(prev_prog, prog)
    assert_op_count_match(prog, expect=0, op="real_div")
    assert_op_count_match(prog, expect=2, op="mul")

    if validate_model:
        assert_model_is_valid(prog, {"x": (2, 4)})


def test_fuse_matmul_weight_bias():
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
    def prog(x):
        weights_val = np.random.rand(2, 4).T.astype(np.float32)
        weights = mb.const(val=weights_val)
        bias_val = np.random.rand(2).astype(np.float32)
        bias = mb.const(val=bias_val)

        matmul = mb.matmul(x=x, y=weights)
        return mb.add(x=matmul, y=bias)

    assert_op_count_match(prog, expect=1, op="matmul")
    assert_op_count_match(prog, expect=0, op="linear")
    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY["common::fuse_matmul_weight_bias"](prog)
    assert_same_output_names(prev_prog, prog)
    assert_op_count_match(prog, expect=0, op="matmul")
    assert_op_count_match(prog, expect=1, op="linear")

    if validate_model:
        assert_model_is_valid(prog, {"x": (2, 4)})


def test_dead_code_elimination():
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(2, 4)), mb.TensorSpec(shape=(2, 4)),]
    )
    def program0(x, y):
        # following three unused op should be eliminated
        a = mb.const(val=np.zeros(shape=(1,)))
        b = mb.const(val=np.zeros(shape=(1,)))
        _ = mb.add(x=a, y=b)
        return mb.add(x=x, y=y)

    assert_op_count_match(program0, expect=4)
    prev_prog = copy.deepcopy(program0)
    PASS_REGISTRY["common::dead_code_elimination"](program0)
    assert_same_output_names(prev_prog, program0)
    assert_op_count_match(program0, expect=1)

    if validate_model:
        assert_model_is_valid(program0, {"x": (2, 4), "y": (2, 4)})

    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
    def program1(x):
        weights_val = np.random.rand(2, 4).T.astype(np.float32)
        weights = mb.const(val=weights_val)
        bias_val = np.random.rand(4).astype(np.float32)
        bias = mb.const(val=bias_val)

        # unused op and its inputs should be eliminated
        mb.matmul(x=x, y=weights)

        return mb.linear(x=x, weight=weights, bias=bias)

    assert_op_count_match(program1, expect=6)
    prev_prog = copy.deepcopy(program1)
    PASS_REGISTRY["common::dead_code_elimination"](program1)
    assert_same_output_names(prev_prog, program1)
    assert_op_count_match(program1, expect=3)

    if validate_model:
        assert_model_is_valid(program1, {"x": (2, 4)})


def test_remove_symbolic_reshape():
    sym_b = Symbol("s0")
    original_shape = (sym_b, Symbol("s1"), 2)
    reshape_name = "reshape"

    @mb.program(input_specs=[mb.TensorSpec(shape=(sym_b, 4))])
    def prog(x):
        # const cannot represent symbolic values. Use _const_symbolic
        shape = mb._const_symbolic(val=original_shape)
        return mb.reshape(x=x, shape=shape, name=reshape_name)

    reshape_op = prog.find_ops(
        prefix=reshape_name, op_type="reshape", exactly_one=True
    )[0]
    shape_var = reshape_op.shape
    reshaped_var = reshape_op.outputs[0]
    assert np.all(shape_var.sym_val == original_shape)
    assert np.all(reshaped_var.shape == (sym_b, 2, 2))

    # Note: we cannot deepcopy prog with symbol.
    prev_outputs = [o.name for o in prog["main"].outputs]
    PASS_REGISTRY["common::remove_symbolic_reshape"](prog)
    curr_outputs = [o.name for o in prog["main"].outputs]
    assert curr_outputs == prev_outputs

    reshape_op = prog.find_ops(
        prefix=reshape_name, op_type="reshape", exactly_one=True
    )[0]
    shape_var = reshape_op.shape
    reshaped_var = reshape_op.outputs[0]
    # shape param cannot be symbolic after the pass
    assert np.all(shape_var.sym_val == (-1, 2, 2))
    # output shape is still symbolic
    assert np.all(reshaped_var.shape == (sym_b, 2, 2))

    if validate_model:
        assert_model_is_valid(prog, {"x": (3, 4)})


def test_loop_invariant_elimination1():
    """
    Invariant pattern: Block input vars are returned as block output vars.
    """

    def body(a, b):
        return mb.add(x=a, y=b), b

    def cond(a, b):
        a_mean = mb.reduce_mean(x=a, axes=[0, 1])
        b_mean = mb.reduce_mean(x=b, axes=[0, 1])
        return mb.less(x=a_mean, y=b_mean)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 2)), mb.TensorSpec(shape=(1, 2)),]
    )
    def prog(a, b):
        # b is loop invariant
        return mb.while_loop(_cond=cond, _body=body, loop_vars=(a, b))

    while_op = prog.find_ops(op_type="while_loop", exactly_one=True)[0]
    assert len(while_op.blocks[0].inputs) == 2
    assert len(while_op.outputs) == 2
    assert len(while_op.loop_vars) == 2
    assert while_op.blocks[0].inputs[0].name == "a_x1"
    assert while_op.blocks[0].inputs[1].name == "b_x1"

    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY["common::loop_invariant_elimination"](prog)
    assert_same_output_names(prev_prog, prog)

    while_op = prog.find_ops(op_type="while_loop", exactly_one=True)[0]
    assert len(while_op.blocks[0].inputs) == 1
    assert len(while_op.outputs) == 1
    assert len(while_op.loop_vars) == 1
    assert while_op.blocks[0].inputs[0].name == "a_x1"

    if validate_model:
        assert_model_is_valid(prog, {"a": (1, 2), "b": (1, 2)})


def test_loop_invariant_elimination2():
    """
    Invariant pattern: Block outputs var from outside of the block
    """

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 2)), mb.TensorSpec(shape=(1, 2)),]
    )
    def prog(a, b):
        def body(a, bx):
            return mb.add(x=a, y=b), b

        def cond(a, bx):
            a_mean = mb.reduce_mean(x=a, axes=[0, 1])
            b_mean = mb.reduce_mean(x=bx, axes=[0, 1])
            return mb.less(x=a_mean, y=b_mean)

        # b is loop invariant
        return mb.while_loop(_cond=cond, _body=body, loop_vars=(a, b))

    while_op = prog.find_ops(op_type="while_loop", exactly_one=True)[0]
    assert len(while_op.blocks[0].inputs) == 2
    assert len(while_op.outputs) == 2
    assert len(while_op.loop_vars) == 2
    assert while_op.blocks[0].inputs[0].name == "a_x1"
    assert while_op.blocks[0].inputs[1].name == "b_x1"

    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY["common::loop_invariant_elimination"](prog)
    assert_same_output_names(prev_prog, prog)

    while_op = prog.find_ops(op_type="while_loop", exactly_one=True)[0]
    assert len(while_op.blocks[0].inputs) == 1
    assert len(while_op.outputs) == 1
    assert len(while_op.loop_vars) == 1
    assert while_op.blocks[0].inputs[0].name == "a_x1"

    if validate_model:
        assert_model_is_valid(prog, {"a": (1, 2), "b": (1, 2)})


def test_gelu_tanh_approximation():
    """
    Detect gelu tanh approx pattern, found in the TF bert model.
    y = ( tanh((.0447)x^3 + x ) * (sqrt(2/pi)) + 1 ) * 0.5 * x
    """

    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
    def prog(x):
        x1 = mb.pow(x=x, y=3)
        x1 = mb.mul(x=0.044715, y=x1)
        x1 = mb.add(x=x1, y=x)
        x1 = mb.mul(x=x1, y=np.sqrt(2 / np.pi))
        x1 = mb.tanh(x=x1)
        x1 = mb.add(x=1, y=x1)
        x1 = mb.mul(x=0.5, y=x1)
        x1 = mb.mul(x=x, y=x1)
        return x1

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::fuse_gelu_tanh_approximation"
    )
    assert get_op_types_in_program(prev_prog) == [
        "pow",
        "mul",
        "add",
        "mul",
        "tanh",
        "add",
        "mul",
        "mul",
    ]
    assert get_op_types_in_program(prog) == ["gelu"]
    assert_model_is_valid(
        prog,
        {"x": (3, 5, 6)},
        expected_output_shapes={block.outputs[0].name: (3, 5, 6)},
    )

@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_onehot_matmul_to_gather_fusion(rank):
    """
    Input:
        %2 = one_hot(%1, on_value=1, off_value=0, axis=-1)
        %3 = const() # rank 2
        %4  = matmul(%2, %3)

    Output:
        %4 = gather(%3, %2, axis=0)
    """
    rank4_shape = (10, 3, 6, 7)
    input_shape = rank4_shape[-rank:]
    vocab_size = 15
    embedding_size = 12

    @mb.program(input_specs=[mb.TensorSpec(shape=input_shape, dtype=types.int32)])
    def prog(x):
        x = mb.one_hot(
            indices=x, on_value=1, off_value=0, axis=-1, one_hot_vector_size=vocab_size
        )
        x = mb.matmul(x=x, y=np.random.rand(vocab_size, embedding_size))
        return x

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::fuse_onehot_matmul_to_gather"
    )
    assert get_op_types_in_program(prev_prog) == ["one_hot", "matmul"]
    assert get_op_types_in_program(prog) == ["gather"]
    assert_model_is_valid(
        prog,
        {"x": input_shape},
        expected_output_shapes={block.outputs[0].name: input_shape + (embedding_size,)},
    )

def test_concat_interleave_fusion_pass():
    """
        Given:
        %3 = concat(%1.a, %1.b, axis=-3, interleave=False) #shape = (B, n*C, H, W)
        %4 = reshape(%3) #shape = (B, n, C, H, W)
        %5 = transpose(%4, perm=[0, 2, 1, 3, 4]) # shape = (B, C, n, H, W)
        %6 = reshape(%5) # shape = (B, C*n, H, W)

    Result:
        %6 = concat(%1.a, %1.b, axis=-3, interleave=True)
    """
    B, C, H, W = 1, 10, 20, 20
    @mb.program(input_specs=[mb.TensorSpec(shape=(B,C,H,W)), mb.TensorSpec(shape=(B,C,H,W))])
    def prog(x, y):
        z = mb.concat(values=[x,y], axis=1)
        z = mb.reshape(x=z, shape=(B, 2, C, H, W))
        z = mb.transpose(x=z, perm=[0, 2, 1, 3, 4])
        z = mb.reshape(x=z, shape=(B, -1, H, W))
        return z

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::detect_concat_interleave"
    )
    assert get_op_types_in_program(prev_prog) == ["concat", "reshape", "transpose", "reshape"]
    assert get_op_types_in_program(prog) == ["concat"]
    concat_op = prog.find_ops(op_type="concat", exactly_one=True)[0]
    assert concat_op.interleave.val
    assert_model_is_valid(
        prog,
        {"x": (B, C, H, W), "y": (B, C, H, W)},
        expected_output_shapes={block.outputs[0].name: (B, 2*C, H, W)},
    )

def test_add_conv_transpose_output_shape():
    """
    Given:
      %1: (1, 5, 39, fp32) = conv_transpose(...) # no output_shape input.

    Result:
      %2: (3, i32) = const(val=[1,5,39])
      %3: (1, 5, 39, fp32) = conv_transpose(..., output_shape=%2)
    """
    N, C_in, C_out, D1 = 1, 3, 5, 20

    @mb.program(input_specs=[mb.TensorSpec(shape=(N, C_in, D1))])
    def prog(x):
        weight = np.random.rand(C_in, C_out, D1).astype(np.float32)
        return mb.conv_transpose(x=x, weight=weight)

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::add_conv_transpose_output_shape"
    )
    assert get_op_types_in_program(prev_prog) == ["conv_transpose"]
    assert get_op_types_in_program(prog) == ["conv_transpose"]
    prev_conv_transpose_op = prev_prog.find_ops(op_type="conv_transpose",
        exactly_one=True)[0]
    conv_transpose_op = prog.find_ops(op_type="conv_transpose",
        exactly_one=True)[0]
    assert np.all(conv_transpose_op.output_shape.val ==
        prev_conv_transpose_op.outputs[0].shape)

@pytest.mark.parametrize(
    "op_type, is_first_op1, is_first_op2, is_first_op3, is_first_op4, const_mul_first",
    itertools.product(
        ["real_div", "mul"],
        [True, False],
        [True, False],
        [True ,False],
        [True, False],
        [True, False],
        )
    )
def test_gelu_exact_approximation(op_type, is_first_op1, is_first_op2, is_first_op3, is_first_op4, const_mul_first):
    """
    Detect gelu exact pattern.
    y = 0.5 * x * ( 1 + erf ( x / srqt(2)))
    """

    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
    def prog(x):
        if op_type == "real_div":
            x1 = mb.real_div(x=x, y=2**0.5)
        elif op_type == "mul":
            x1 = mb.mul(x=x, y=2**-0.5) if is_first_op1 else mb.mul(x=2**-0.5, y=x)

        x2 = mb.erf(x=x1)
        x3 = mb.add(x=x2, y=1) if is_first_op2 else mb.add(x=1, y=x2)

        if const_mul_first:
            y1 = mb.const(val=0.5)
            y2 = x
        else:
            y1 = x
            y2 = mb.const(val=0.5)

        x4 = mb.mul(x=x3, y=y1) if is_first_op3 else mb.mul(x=y1, y=x3)
        x5 = mb.mul(x=x4, y=y2) if is_first_op4 else mb.mul(x=y2, y=x4)

        return x5

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::fuse_gelu_exact"
    )

    assert get_op_types_in_program(prev_prog) == [
        op_type,
        "erf",
        "add",
        "mul",
        "mul",
    ]
    assert get_op_types_in_program(prog) == ["gelu"]
    assert_model_is_valid(
        prog,
        {"x": (3, 5, 6)},
        expected_output_shapes={block.outputs[0].name: (3, 5, 6)},
    )

class TestTopologicalReorder:

    def test_move_sink_casts_to_the_end(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16")
            x1 = mb.square(x=x)
            x2 = mb.cast(x=x1, dtype="fp32")
            x3 = mb.log(x=x)
            x4 = mb.cast(x=x3, dtype="fp32")
            x5 = mb.relu(x=x)
            x6 = mb.cast(x=x5, dtype="fp32")
            x7 = mb.relu(x=x6)
            return x2, x4, x7

        assert get_op_types_in_program(prog) == ['cast', 'square', 'cast', 'log', 'cast', 'relu', 'cast', 'relu']

        apply_pass_and_basic_check(prog, "common::topological_reorder")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["cast", "square", "log", "relu", "cast", "cast", "cast", "relu"]

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (10, 20),
                block.outputs[1].name: (10, 20),
                block.outputs[2].name: (10, 20),
            },
        )
