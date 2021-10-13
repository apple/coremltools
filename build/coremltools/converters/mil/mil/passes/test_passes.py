#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    assert_op_count_match,
    assert_model_is_valid,
    assert_same_output_names,
    get_op_types_in_program,
    apply_pass_and_basic_check,
)
from coremltools.converters.mil.mil import Function, get_new_symbol, Program, Symbol, types
from coremltools.converters.mil.experimental.passes.generic_pass_infrastructure import register_generic_pass
from coremltools.converters.mil.mil.passes.helper import _check_var_scalar_value
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
import copy
import pytest
import itertools
import os

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

@pytest.mark.parametrize(
    "first_op_1, first_op_2, first_op_3, first_op_4, first_op_5, first_op_6",
     itertools.product(
         [True, False],
         [True, False],
         [True, False],
         [True, False],
         [True, False],
         [True, False]
     )
)
def test_gelu_tanh_approximation2(first_op_1, first_op_2, first_op_3, first_op_4, first_op_5, first_op_6):
    """
    Detect gelu tanh approx pattern, found in the TF Sanitized GPT2 model.
    y = ( tanh((.0447)x^3 + x ) * (sqrt(2/pi)) + 1 ) * 0.5 * x
    """

    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
    def prog(x):
        firstmul = mb.mul(x=x, y=0.5) if first_op_1 else mb.mul(x=0.5, y=x)
        x1 = mb.pow(x=x, y=3)
        x1 = mb.mul(x=0.044715, y=x1) if first_op_2 else mb.mul(x=x1, y=0.044715)
        x1 = mb.add(x=x1, y=x) if first_op_3 else mb.add(x=x, y=x1)
        x1 = mb.mul(x=x1, y=np.sqrt(2 / np.pi)) if first_op_4 else mb.mul(x=np.sqrt(2 / np.pi), y=x1)
        x1 = mb.tanh(x=x1)
        x1 = mb.add(x=1, y=x1) if first_op_5 else mb.add(x=x1, y=1)
        x1 = mb.mul(x=firstmul, y=x1) if first_op_6 else mb.mul(x=x1, y=firstmul)
        return x1

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::fuse_gelu_tanh_approximation"
    )
    assert get_op_types_in_program(prev_prog) == [
        "mul",
        "pow",
        "mul",
        "add",
        "mul",
        "tanh",
        "add",
        "mul",
    ]

    if os.getenv('ENABLE_EXPERIMENTAL_PASSES') == '1':
        assert get_op_types_in_program(prog) == ["gelu"]
        assert_model_is_valid(
            prog,
            {"x": (3, 5, 6)},
            expected_output_shapes={block.outputs[0].name: (3, 5, 6)},
        )

def test_generic_child_ordering():
    """
    Checks that the new generic pattern matching infrastructure works
    regardless of the ordering of an operation's children
    """

    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
    def prog(x):
        power = mb.pow(x=x, y=3, name="thepowerop")
        add_0 = mb.add(x=power, y=5, name="add_0")
        sub_0 = mb.sub(x=power, y=5, name="sub_0")
        mul_0 = mb.mul(x=power, y=5, name="mul_0")
        add_1 = mb.add(x=add_0, y=mul_0, name="add_1")
        add_2 = mb.add(x=sub_0, y=add_1, name="add_2")
        return add_2

    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
    def ops_arrangement(x):
        power = mb.pow(x=x, y=3, name="thepowerop")
        sub_0 = mb.sub(x=power, y=5, name="sub_0")
        add_0 = mb.add(x=power, y=5, name="add_0")
        mul_0 = mb.mul(x=power, y=5, name="mul_0")
        add_1 = mb.add(x=add_0, y=mul_0, name="add_1")
        add_2 = mb.add(x=sub_0, y=add_1,name="add_2")
        return add_2

    def var_constraints(pattern):
        constraints_passed = True
        constraints_passed &= _check_var_scalar_value(pattern.thepowerop.y, 3)
        constraints_passed &= _check_var_scalar_value(pattern.sub_0.y, 5)
        constraints_passed &= _check_var_scalar_value(pattern.add_0.x, 5) or _check_var_scalar_value(pattern.add_0.y, 5)
        constraints_passed &=  _check_var_scalar_value(pattern.mul_0.x, 5) or _check_var_scalar_value(pattern.mul_0.y, 5)
        return constraints_passed

    def transform_pattern(pattern):
        out_name = "new operation"
        x = mb.gelu(x=pattern.root_var, mode="TANH_APPROXIMATION", name=out_name, before_op=pattern.thepowerop)

        pattern.add_2.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=pattern.add_2, old_var=pattern.add_2.outputs[0], new_var=x
        )

        pattern.block.remove_ops(pattern.op_list())

    register_generic_pass(ops_arrangement=ops_arrangement, var_constraints=var_constraints,
                          transform_pattern=transform_pattern, pass_name="test_generic_child_ordering",
                          namespace="common")

    prev_prog, prev_block, block = apply_pass_and_basic_check(
        prog, "common::test_generic_child_ordering"
    )
    assert get_op_types_in_program(prev_prog) == [
        "pow",
        "add",
        "sub",
        "mul",
        "add",
        "add",
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



class TestLeakyReluFusionPass:

    @pytest.mark.parametrize(
        "swap_mul_input_order, swap_max_input_order",
        itertools.product(
            [True, False],
            [True, False],
            )
        )
    def test_valid_leaky_relu_pattern(self, swap_mul_input_order, swap_max_input_order):
        """
        Input graph:

                const (val = 0.3)
                    |
        input ----> mul ---------------> maximum -----------> output
          |                                 |
          |----------------------------------

        Output graph:

        input --------> leaky_relu ---------> output
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            if swap_mul_input_order:
                x1 = mb.mul(x=x, y=0.3)
            else:
                x1 = mb.mul(x=0.3, y=x)
            if swap_max_input_order:
                x1 = mb.maximum(x=x1, y=x)
            else:
                x1 = mb.maximum(x=x, y=x1)
            return x1

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_leaky_relu"
        )
        assert get_op_types_in_program(prev_prog) == ["mul", "maximum"]
        assert get_op_types_in_program(prog) == ["leaky_relu"]
        assert_model_is_valid(
            prog,
            {"x": (3, 5, 6)},
            expected_output_shapes={block.outputs[0].name: (3, 5, 6)},
        )

    def test_invalid_leaky_relu_pattern1(self):
        """
        Invalid because alpha value greater than 1

        Input graph:

                const (val = 1.3)
                    |
        input ----> mul ---------------> maximum -----------> output
          |                                 |
          |----------------------------------

        Output graph: same as input graph
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            x1 = mb.mul(x=x, y=1.3)
            x1 = mb.maximum(x=x1, y=x)
            return x1

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_leaky_relu"
        )
        assert get_op_types_in_program(prev_prog) == ["mul", "maximum"]
        assert get_op_types_in_program(prog) == ["mul", "maximum"]


    def test_invalid_leaky_relu_pattern2(self):
        """
        Invalid because input to the "maximum" op is not same as the input of the "mul" op

        Input graph:

                const (val = 0.3)
                    |
        input ----> mul ---------------> maximum -----------> output
                                           |
                                         const


        Output graph: same as input graph
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            x1 = mb.mul(x=x, y=0.3)
            x1 = mb.maximum(x=x1, y=0.4)
            return x1

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_leaky_relu"
        )
        assert get_op_types_in_program(prev_prog) == ["mul", "maximum"]
        assert get_op_types_in_program(prog) == ["mul", "maximum"]

class TestReduceMeanFusionPass:

    def test_valid_pattern1(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            x1 = mb.reduce_sum(x=x, axes=[-1, -2], keep_dims=True)
            x1 = mb.mul(x=1.0 / 30, y=x1)
            return x1

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_reduce_mean"
        )
        assert get_op_types_in_program(prev_prog) == ["reduce_sum", "mul"]
        assert get_op_types_in_program(prog) == ["reduce_mean"]
        assert_model_is_valid(
            prog,
            {"x": (3, 5, 6)},
            expected_output_shapes={block.outputs[0].name: (3, 1, 1)},
        )

    def test_valid_pattern2(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 5))])
        def prog(x):
            x1 = mb.reduce_sum(x=x, axes=[0], keep_dims=False)
            x1 = mb.real_div(x=x1, y=4.0)
            return x1

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_reduce_mean"
        )
        assert get_op_types_in_program(prev_prog) == ["reduce_sum", "real_div"]
        assert get_op_types_in_program(prog) == ["reduce_mean"]
        assert_model_is_valid(
            prog,
            {"x": (4, 5)},
            expected_output_shapes={block.outputs[0].name: (5,)},
        )

    def test_invalid_pattern1(self):
        '''
        The mul does not correspond to "1/count"
        '''
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            x1 = mb.reduce_sum(x=x, axes=[-1, -2], keep_dims=True)
            x1 = mb.mul(x=5.0, y=x1)
            return x1

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_reduce_mean"
        )
        assert get_op_types_in_program(prog) == ["reduce_sum", "mul"]

    def test_invalid_pattern2(self):
        '''
        The div does not correspond to "count"
        '''
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            x1 = mb.reduce_sum(x=x, axes=[-1, -2], keep_dims=True)
            x1 = mb.real_div(x=x1, y=31.0)
            return x1

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_reduce_mean"
        )
        assert get_op_types_in_program(prog) == ["reduce_sum", "real_div"]

    def test_invalid_pattern3(self):
        '''
        One of the reduction dim is symbolic
        '''
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, get_new_symbol(), 6))])
        def prog(x):
            x1 = mb.reduce_sum(x=x, axes=[-1, -2], keep_dims=True)
            x1 = mb.real_div(x=x1, y=30.0)
            return x1

        pass_name = "common::fuse_reduce_mean"
        PASS_REGISTRY[pass_name](prog)
        assert get_op_types_in_program(prog) == ["reduce_sum", "real_div"]

    def test_invalid_pattern4(self):
        '''
        output of reduce_sum is model output
        '''
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            x1 = mb.reduce_sum(x=x, axes=[-1, -2], keep_dims=True)
            y1 = mb.real_div(x=x1, y=30.0)
            return y1, x1

        pass_name = "common::fuse_reduce_mean"
        PASS_REGISTRY[pass_name](prog)
        assert get_op_types_in_program(prog) == ["reduce_sum", "real_div"]

    def test_invalid_pattern5(self):
        '''
        output of reduce_sum is feeding into another op
        '''
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            x1 = mb.reduce_sum(x=x, axes=[-1, -2], keep_dims=True)
            y1 = mb.real_div(x=x1, y=30.0)
            y2 = mb.mul(x=x1, y=10.0)
            y3 = mb.add(x=y1, y=y2)
            return y3

        pass_name = "common::fuse_reduce_mean"
        PASS_REGISTRY[pass_name](prog)
        assert get_op_types_in_program(prog) == ["reduce_sum", "real_div", "mul", "add"]


class TestTopologicalReorder:

    def test_move_sink_casts_to_the_end(self):
        '''
        Input graph:
            x (input) ---> square ---> cast (output)
            |
            | -----------> log ------> cast (output)
            |
            | -----------> relu -----> cast ----> relu (output)
        '''
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

        assert get_op_types_in_program(prog) == ['cast', 'square', 'log', 'relu', 'cast', 'relu', 'cast', 'cast',]

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (10, 20),
                block.outputs[1].name: (10, 20),
                block.outputs[2].name: (10, 20),
            },
        )

    def test_move_sink_cast_transpose_to_the_end(self):
        '''
        Input graph:
            x (input) ---> square ---> transpose ---> cast (output)
            |
            | -----------> log ------> transpose ---> cast (output)
            |
            | -----------> relu -----> cast ----> relu (output)
            |
            | -----------> relu (output)
        '''
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16")
            x1 = mb.square(x=x)
            x1_t = mb.transpose(x=x1, perm=[1, 0])
            x2 = mb.cast(x=x1_t, dtype="fp32")
            x3 = mb.log(x=x)
            x3_t = mb.transpose(x=x3, perm=[1, 0])
            x4 = mb.cast(x=x3_t, dtype="fp32")
            x5 = mb.relu(x=x)
            x6 = mb.cast(x=x5, dtype="fp32")
            x7 = mb.relu(x=x6)
            x8 = mb.relu(x=x)
            return x2, x4, x7, x8

        assert get_op_types_in_program(prog) == ['cast', 'square', 'transpose', 'cast', 'log', 'transpose', 'cast', 'relu', 'cast', 'relu', 'relu']

        apply_pass_and_basic_check(prog, "common::topological_reorder")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ['cast', 'square', 'log', 'relu', 'cast', 'relu', 'relu', 'transpose', 'cast', 'transpose', 'cast']

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (20, 10),
                block.outputs[1].name: (20, 10),
                block.outputs[2].name: (10, 20),
                block.outputs[3].name: (10, 20),
            },
        )

    def test_move_multiple_uses_overlapping(self):
        '''
        Input graph:
            x (input) ---> cast ---> cast (output)
                           |
                           |-------> transpose ---> transpose (output)
        '''
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x1 = mb.cast(x=x, dtype="fp16")
            x2 = mb.cast(x=x1, dtype="fp32")
            x3 = mb.transpose(x=x1, perm=[1, 0])
            x4 = mb.transpose(x=x3, perm=[1, 0])
            return x2, x4

        assert get_op_types_in_program(prog) == ['cast', 'cast', 'transpose', 'transpose']

        apply_pass_and_basic_check(prog, "common::topological_reorder")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ['cast', 'transpose', 'transpose', 'cast']

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (10, 20),
                block.outputs[1].name: (10, 20)
            },
        )

    def test_move_split_to_first_use(self):
        '''
        Input graph:
            x (input) ---> split ---> square ---> add (output)
            |                |                     |
            |                | --------------------|
            |
            | -----------> square --------------> relu (output)
        '''
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            s1, s2 = mb.split(x=x, num_splits=2, axis=0)
            x2 = mb.square(x=x)
            x3 = mb.relu(x=x2)
            s1_1 = mb.square(x=s1)
            s3 = mb.add(x=s1_1, y=s2)
            return x3, s3

        assert get_op_types_in_program(prog) == ['split', 'square', 'relu', 'square', 'add']

        block = prog.functions["main"]
        # Reorder `split` op to test op with multiple output case
        from .topological_reorder import  move_operations_to_the_end_block
        move_operations_to_the_end_block(block, ['split'])

        assert get_op_types_in_program(prog) == ['square', 'relu', 'split', 'square', 'add']

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (10, 20),
                block.outputs[1].name: (5, 20),
            },
        )

    def test_move_transpose_before_subblock(self):
        '''
        Input graph:
            x (input) ---> cast ---> transpose ---> cast (output)
            |
            | -----------> square ------> transpose (x1_t) ---> cast (output)
            |
            | -----------> squeeze ----> equal ----> squeeze
                                                        |
                                          (true) <--- /  \ ---> (false)
                                            |                        |
                                            |      /<-(x1_t)->\      |
                                           add  <-/            \--> add
                                            |---------> | <---------|
                                                        |
                                                       add ---> cast (output)
        '''
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16")
            x1 = mb.square(x=x)
            x1_t = mb.transpose(x=x1, perm=[1, 0])

            def true_fn():
                return mb.add(x=x1_t, y=1, name='x2')

            def false_fn():
                return mb.add(x=x1_t, y=2, name='x2')

            is_one = mb.equal(x=mb.squeeze(x=x), y=1)
            pred = mb.squeeze(x=is_one)
            x3 = mb.cond(pred=pred, _true_fn=true_fn, _false_fn=false_fn)
            x4 = mb.add(x=x1_t, y=x3)
            x5 = mb.cast(x=x4, dtype="fp32")
            return x5

        apply_pass_and_basic_check(prog, "common::topological_reorder")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ['cast', 'square', 'squeeze', 'equal', 'squeeze', 'transpose', 'cond', 'add', 'cast']

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (20, 10)
            },
        )

    def test_cast_transpose_already_at_the_end(self):
        '''
        Input graph:
            x (input) ---> square ---> transpose ---> cast (output)
            |
            | -----------> log ------> transpose ---> cast (output)
            |
            | -----------> relu -----> cast ----> relu (output)
            |
            | -----------> relu (output)
        '''
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16")
            x1 = mb.square(x=x)
            x3 = mb.log(x=x)
            x5 = mb.relu(x=x)
            x6 = mb.cast(x=x5, dtype="fp32")
            x7 = mb.relu(x=x6)
            x8 = mb.relu(x=x)
            x1_t = mb.transpose(x=x1, perm=[1, 0])
            x2 = mb.cast(x=x1_t, dtype="fp32")
            x3_t = mb.transpose(x=x3, perm=[1, 0])
            x4 = mb.cast(x=x3_t, dtype="fp32")
            return x2, x4, x7, x8

        assert get_op_types_in_program(prog) == ['cast', 'square', 'log', 'relu', 'cast', 'relu', 'relu', 'transpose', 'cast', 'transpose', 'cast']

        apply_pass_and_basic_check(prog, "common::topological_reorder")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ['cast', 'square', 'log', 'relu', 'cast', 'relu', 'relu', 'transpose', 'cast', 'transpose', 'cast']

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (20, 10),
                block.outputs[1].name: (20, 10),
                block.outputs[2].name: (10, 20),
                block.outputs[3].name: (10, 20),
            },
        )


class TestInputOutputSanitization:

    def test_nn_backend_style_sanitization(self):
        '''
        Test that intermediate var names are unchanged, and
        only model input and output names are modified, i.e.
        sanitized (adhering to the format [a-zA-Z_][a-zA-Z0-9_]*)
        for the NN backend.
        '''

        prog = Program()
        func_inputs = {"x/0": mb.placeholder(shape=[2, 3]),
                       "y": mb.placeholder(shape=[2, 3])}
        with Function(func_inputs) as ssa_fun:
            x, y = ssa_fun.inputs["x/0"], ssa_fun.inputs["y"]
            x = mb.relu(x=x, name="relu/1")
            z = mb.add(x=x, y=y, name="out/1")
            ssa_fun.set_outputs([z])
        prog.add_function("main", ssa_fun)

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::sanitize_input_output_names",
            skip_output_name_check=True
        )

        relu_op = prog.find_ops(op_type="relu", exactly_one=True)[0]
        assert relu_op.inputs["x"].name == "x_0" # input name: sanitized
        assert relu_op.outputs[0].name == "relu/1" # intermediate name: unchanged
        assert block.outputs[0].name == "out_1" # output name: sanitized

        # convert prev_prog to NN backend
        mlmodel = ct.convert(prev_prog)
        spec = mlmodel._spec
        assert spec.description.input[0].name == "x_0"
        assert spec.description.output[0].name == "out_1"
        relu_layer = spec.neuralNetwork.layers[0]
        assert relu_layer.output[0] == "relu/1"


class TestPassRank0ExpandDimsSwap:
    """
    Input graph:
                                 2.0
                                  |
                                  v
    input --> slice_by_index --> sub --> expand_dims --> output

    Output graph:
                                                [2.0]
                                                  |
                                                  v
    input --> slice_by_index --> expand_dims --> sub --> output
    """

    @pytest.mark.skipif(ct.utils._macos_version() < (12, 0), reason="mlprogram predict available only on macOS12+")
    @pytest.mark.parametrize(
        "reverse_order, elem_op",
        itertools.product(
            [True, False],
            ["add", "sub", "mul", "real_div", "floor_div"],
        ),
    )
    def test(self, reverse_order, elem_op):
        x_shape = [1,]

        @mb.program(input_specs=[mb.TensorSpec(shape=x_shape)])
        def program(x):
            x = mb.slice_by_index(x=x, begin=[0], end=[1], squeeze_mask=[True])
            func = getattr(mb, elem_op)

            if reverse_order:
                x = func(x=2.0, y=x)
            else:
                x = func(x=x, y=2.0)

            expand = mb.expand_dims(x=x, axes=[0])
            other_1 = mb.add(x=x, y=[1, 2, 3])
            other_2 = mb.sub(x=x, y=[1, 2, 3])
            return expand, other_1, other_2

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            program, "common::rank0_expand_dims_swap"
        )
        assert get_op_types_in_program(prev_prog) == ["slice_by_index", elem_op, "expand_dims", "add", "sub"]
        assert get_op_types_in_program(program) == ["slice_by_index", "expand_dims", "expand_dims", elem_op, "squeeze", "add", "sub"]
        assert_model_is_valid(
            program=program,
            inputs={"x": x_shape},
            expected_output_shapes={
                block.outputs[0].name: tuple(x_shape),
                block.outputs[1].name: (3,),
                block.outputs[2].name: (3,),
            },
        )
