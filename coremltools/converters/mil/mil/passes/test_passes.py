#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil.experimental.passes.generic_pass_infrastructure import \
    register_generic_pass
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import (Function, Program, Symbol,
                                            get_new_symbol, types)
from coremltools.converters.mil.mil.passes.helper import \
    _check_var_scalar_value
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check, assert_model_is_valid, assert_op_count_match,
    assert_same_output_names, get_op_types_in_program)

from .compression_passes import (WeightAffineQuantizer, WeightPalettizer,
                                 WeightSparsifier)

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
        weights_val = np.random.rand(4, 2).T.astype(np.float32)
        weights = mb.const(val=weights_val)
        bias_val = np.random.rand(2).astype(np.float32)
        bias = mb.const(val=bias_val)

        # unused op and its inputs should be eliminated
        weights_for_matmul = mb.transpose(x=weights, perm=[1, 0])
        mb.matmul(x=x, y=weights_for_matmul)

        return mb.linear(x=x, weight=weights, bias=bias)

    assert_op_count_match(program1, expect=8)
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
    assert while_op.blocks[0].inputs[0].name == "a_x0"
    assert while_op.blocks[0].inputs[1].name == "b_x0"

    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY["common::loop_invariant_elimination"](prog)
    assert_same_output_names(prev_prog, prog)

    while_op = prog.find_ops(op_type="while_loop", exactly_one=True)[0]
    assert len(while_op.blocks[0].inputs) == 1
    assert len(while_op.outputs) == 1
    assert len(while_op.loop_vars) == 1
    assert while_op.blocks[0].inputs[0].name == "a_x0"

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
    assert while_op.blocks[0].inputs[0].name == "a_x0"
    assert while_op.blocks[0].inputs[1].name == "b_x0"

    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY["common::loop_invariant_elimination"](prog)
    assert_same_output_names(prev_prog, prog)

    while_op = prog.find_ops(op_type="while_loop", exactly_one=True)[0]
    assert len(while_op.blocks[0].inputs) == 1
    assert len(while_op.outputs) == 1
    assert len(while_op.loop_vars) == 1
    assert while_op.blocks[0].inputs[0].name == "a_x0"

    if validate_model:
        assert_model_is_valid(prog, {"a": (1, 2), "b": (1, 2)})

def test_generic_child_ordering():
    """
    Checks that the new generic pattern matching infrastructure works
    regardless of the ordering of an operation's children
    """

    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
    def prog(x):
        power = mb.pow(x=x, y=3., name="thepowerop")
        add_0 = mb.add(x=power, y=5., name="add_0")
        sub_0 = mb.sub(x=power, y=5., name="sub_0")
        mul_0 = mb.mul(x=power, y=5., name="mul_0")
        add_1 = mb.add(x=add_0, y=mul_0, name="add_1")
        add_2 = mb.add(x=sub_0, y=add_1, name="add_2")
        return add_2

    @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
    def ops_arrangement(x):
        power = mb.pow(x=x, y=3., name="thepowerop")
        sub_0 = mb.sub(x=power, y=5., name="sub_0")
        add_0 = mb.add(x=power, y=5., name="add_0")
        mul_0 = mb.mul(x=power, y=5., name="mul_0")
        add_1 = mb.add(x=add_0, y=mul_0, name="add_1")
        add_2 = mb.add(x=sub_0, y=add_1,name="add_2")
        return add_2

    def var_constraints(pattern):
        constraints_passed = True
        constraints_passed &= _check_var_scalar_value(pattern.thepowerop.y, 3)
        constraints_passed &= _check_var_scalar_value(pattern.sub_0.y, 5)
        constraints_passed &= _check_var_scalar_value(pattern.add_0.x, 5) or _check_var_scalar_value(pattern.add_0.y, 5)
        constraints_passed &= _check_var_scalar_value(pattern.mul_0.x, 5) or _check_var_scalar_value(pattern.mul_0.y, 5)
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
            indices=x, on_value=1., off_value=0., axis=-1, one_hot_vector_size=vocab_size
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


def test_prelu_to_lrelu():
    @mb.program(input_specs=[mb.TensorSpec(shape=(4, 2, 3, 1))])
    def prog(x):
        # Not a common leakage factor.
        alpha_0 = np.array([1.0, 2.0], dtype=np.float32)
        x = mb.prelu(x=x, alpha=alpha_0)

        add_val = np.random.rand(4, 2, 3, 1).astype(np.float32)
        x = mb.add(x=x, y=add_val)

        # Common leakage factor.
        alpha_1 = np.array([1.5, 1.5], dtype=np.float32)
        x = mb.prelu(x=x, alpha=alpha_1)

        return x

    assert_op_count_match(prog, expect=2, op="prelu")
    assert_op_count_match(prog, expect=0, op="leaky_relu")
    prev_prog, _, _ = apply_pass_and_basic_check(
        prog, "common::prelu_to_lrelu")
    assert_same_output_names(prev_prog, prog)
    # The prelu with a common leakage factor becomes leaky_relu.
    assert_op_count_match(prog, expect=1, op="prelu")
    assert_op_count_match(prog, expect=1, op="leaky_relu")

    if validate_model:
        assert_model_is_valid(prog, {"x": (4, 2, 3, 1)})


class TestGeluFusionPass:

    def test_gelu_tanh_approximation1(self):
        """
        Detect gelu tanh approx pattern, found in the TF bert model.
        y = ( tanh((.0447)x^3 + x ) * (sqrt(2/pi)) + 1 ) * 0.5 * x
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            x1 = mb.pow(x=x, y=3.)
            x1 = mb.mul(x=0.044715, y=x1)
            x1 = mb.add(x=x1, y=x)
            x1 = mb.mul(x=x1, y=np.sqrt(2 / np.pi))
            x1 = mb.tanh(x=x1)
            x1 = mb.add(x=1., y=x1)
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
    def test_gelu_tanh_approximation2(self, first_op_1, first_op_2, first_op_3, first_op_4, first_op_5, first_op_6):
        """
        Detect gelu tanh approx pattern, found in the TF Sanitized GPT2 model.
        y = ( tanh((.0447)x^3 + x ) * (sqrt(2/pi)) + 1 ) * 0.5 * x
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            firstmul = mb.mul(x=x, y=0.5) if first_op_1 else mb.mul(x=0.5, y=x)
            x1 = mb.pow(x=x, y=3.)
            x1 = mb.mul(x=0.044715, y=x1) if first_op_2 else mb.mul(x=x1, y=0.044715)
            x1 = mb.add(x=x1, y=x) if first_op_3 else mb.add(x=x, y=x1)
            x1 = mb.mul(x=x1, y=np.sqrt(2 / np.pi)) if first_op_4 else mb.mul(x=np.sqrt(2 / np.pi), y=x1)
            x1 = mb.tanh(x=x1)
            x1 = mb.add(x=1., y=x1) if first_op_5 else mb.add(x=x1, y=1.)
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

        assert get_op_types_in_program(prog) == ["gelu"]
        assert_model_is_valid(
            prog,
            {"x": (3, 5, 6)},
            expected_output_shapes={block.outputs[0].name: (3, 5, 6)},
        )

    @pytest.mark.parametrize(
        "op_type, is_first_op1, is_first_op2, is_first_op3, is_first_op4, const_mul_first",
        itertools.product(
            ["real_div", "mul"],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
            )
        )
    def test_gelu_exact(self, op_type, is_first_op1, is_first_op2, is_first_op3, is_first_op4, const_mul_first):
        """
        Detect gelu exact pattern.
        y = 0.5 * (x * ( 1 + erf ( x / srqt(2))))
         or
        y = x * (0.5 * ( 1 + erf ( x / srqt(2))))
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            if op_type == "real_div":
                x1 = mb.real_div(x=x, y=2**0.5)
            elif op_type == "mul":
                x1 = mb.mul(x=x, y=2**-0.5) if is_first_op1 else mb.mul(x=2**-0.5, y=x)

            x2 = mb.erf(x=x1)
            x3 = mb.add(x=x2, y=1.) if is_first_op2 else mb.add(x=1., y=x2)

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

    @pytest.mark.parametrize(
        "is_first_op0, is_first_op4",
        itertools.product(
            [True, False],
            [True, False],
            )
        )
    def test_gelu_exact_pattern_2(self, is_first_op0, is_first_op4):
        """
        Detect gelu exact pattern.
        y = (0.5 * x) * ( 1 + erf ( x / srqt(2)))
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            x0 = mb.mul(x=x, y=0.5) if is_first_op0 else mb.mul(x=0.5, y=x)
            x1 = mb.mul(x=x, y=2**-0.5)
            x2 = mb.erf(x=x1)
            x3 = mb.add(x=x2, y=1.)
            x4 = mb.mul(x=x0, y=x3) if is_first_op4 else mb.mul(x=x3, y=x0)
            return x4

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_gelu_exact"
        )

        assert get_op_types_in_program(prev_prog) == [
            "mul",
            "mul",
            "erf",
            "add",
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
        from .topological_reorder import _move_operations_to_the_end_block
        _move_operations_to_the_end_block(block, ['split'])

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
                return mb.add(x=x1_t, y=np.float16(1), name='x2')

            def false_fn():
                return mb.add(x=x1_t, y=np.float16(2), name='x2')

            is_one = mb.equal(x=mb.squeeze(x=x), y=np.float16(1.))
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
            other_1 = mb.add(x=x, y=[1., 2., 3.])
            other_2 = mb.sub(x=x, y=[1., 2., 3.])
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


class TestRemoveRedundantOpsPass:

    def test_redundant_ops_just_after_input_valid_pattern_1(self):
        """
        Input graph:
        input----->transpose(perm=[0, 2, 1])--->add---> add ---> out
               |                                 ^       ^
               |                                 |       |
               |---->transpose(perm=[0, 2, 1])----       |
               |                                         |
               |                                         |
               |---->transpose(perm=[0, 2, 1])------------

        Output graph:
        input----->transpose(perm=[0, 2, 1])--->add---> add ----> out
                                    |            ^       ^
                                    |            |       |
                                    |-------------       |
                                    |                    |
                                    |--------------------
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 5))])
        def prog(x):
            x1 = mb.transpose(x=x, perm=[0, 2, 1])
            x2 = mb.transpose(x=x, perm=[0, 2, 1])
            x3 = mb.transpose(x=x, perm=[0, 2, 1])
            z = mb.add(x=x1, y=x2)
            z = mb.add(x=z, y=x3)
            return z

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::remove_redundant_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["transpose", "transpose", "transpose", "add", "add"]
        assert get_op_types_in_program(prog) == ["transpose", "add", "add"]
        assert_model_is_valid(
            prog,
            {"x": (2, 3, 5)},
            expected_output_shapes={block.outputs[0].name: (2, 5, 3)},
        )

    def test_redundant_ops_just_after_input_valid_pattern_2(self):
        """
        Input graph:
        input----->leaky_relu(alpha=0.3)--->add---> add ---> out
               |                             ^       ^
               |                             |       |
               |----->leaky_relu(alpha=0.3)---       |
               |                                     |
               |                                     |
               |---->leaky_relu(alpha=0.3)------------

        Output graph:
        input--------->leaky_relu(alpha=0.3)--->add---> add ----> out
                                    |            ^       ^
                                    |            |       |
                                    |-------------       |
                                    |                    |
                                    |---------------------
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 5))])
        def prog(x):
            x1 = mb.leaky_relu(x=x, alpha=0.3)
            x2 = mb.leaky_relu(x=x, alpha=0.3)
            x3 = mb.leaky_relu(x=x, alpha=0.3)
            z = mb.add(x=x1, y=x2)
            z = mb.add(x=z, y=x3)
            return z

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::remove_redundant_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["leaky_relu", "leaky_relu", "leaky_relu", "add", "add"]
        assert get_op_types_in_program(prog) == ["leaky_relu", "add", "add"]
        assert_model_is_valid(
            prog,
            {"x": (2, 3, 5)},
            expected_output_shapes={block.outputs[0].name: (2, 3, 5)},
        )

    def test_redundant_ops_just_after_input_invalid_pattern_1(self):
        """
        input----->transpose(perm=[0, 2, 1])---> reshape(shape=[-1]) -----> add ---> out
               |                                                             ^
               |                                                             |
               |---->transpose(perm=[1, 0, 2])----> reshape(shape=[-1])------
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 5))])
        def prog(x):
            x1 = mb.transpose(x=x, perm=[0, 2, 1])
            x2 = mb.transpose(x=x, perm=[1, 0, 2])
            x1 = mb.reshape(x=x1, shape=[-1])
            x2 = mb.reshape(x=x2, shape=[-1])
            z = mb.add(x=x1, y=x2)
            return z

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::remove_redundant_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["transpose", "transpose", "reshape", "reshape", "add"]
        assert get_op_types_in_program(prog) == ["transpose", "transpose", "reshape", "reshape", "add"]
        assert_model_is_valid(
            prog,
            {"x": (2, 3, 5)},
            expected_output_shapes={block.outputs[0].name: (30,)},
        )

    def test_redundant_ops_just_after_input_invalid_pattern_2(self):
        """
        input----->leaky_relu(alpha=0.3) -----> add ---> out
               |                                 ^
               |                                 |
               |---->leaky_relu(alpha=0.4)-------

        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 5))])
        def prog(x):
            x1 = mb.leaky_relu(x=x, alpha=0.3)
            x2 = mb.leaky_relu(x=x, alpha=0.4)
            z = mb.add(x=x1, y=x2)
            return z

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::remove_redundant_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["leaky_relu", "leaky_relu", "add"]
        assert get_op_types_in_program(prog) == ["leaky_relu", "leaky_relu", "add"]
        assert_model_is_valid(
            prog,
            {"x": (2, 3, 5)},
            expected_output_shapes={block.outputs[0].name: (2, 3, 5)},
        )

    def test_redundant_ops_just_after_input_invalid_pattern_3(self):
        """
        test case, when inputs of 1 op is a subset of the inputs of the other op

        input----->layer_norm1 -----> add ---> out
               |                       ^
               |                       |
               |---->layer_norm2-------

        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 2))])
        def prog(x):
            x1 = mb.layer_norm(x=x, axes=[2], epsilon=1e-4)
            gamma_val = np.array([1.0, 1.0], dtype=np.float32)
            beta_val = np.array([1.0, 0.0], dtype=np.float32)
            x2 = mb.layer_norm(x=x, axes=[2], epsilon=1e-4, gamma=gamma_val, beta=beta_val)
            z = mb.add(x=x1, y=x2)
            return z

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::remove_redundant_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["layer_norm", "layer_norm", "add"]
        assert get_op_types_in_program(prog) == ["layer_norm", "layer_norm", "add"]
        assert_model_is_valid(
            prog,
            {"x": (1, 3, 2)},
            expected_output_shapes={block.outputs[0].name: (1, 3, 2)},
        )

    @staticmethod
    def _make_repeated_conv_prog(redundant_conv=True):
        prog = Program()
        func_inputs = {"x": mb.placeholder(shape=[1, 4, 5, 5])}
        with Function(func_inputs) as ssa_fun:
            x = ssa_fun.inputs["x"]
            x = mb.relu(x=x)
            W = np.random.rand(8, 4, 3, 3)
            if redundant_conv:
                bias = np.random.rand(8)
                x1 = mb.conv(x=x, weight=W, bias=bias, pad_type="same", strides=[1, 1])
                x2 = mb.conv(x=x, weight=W, bias=bias, pad_type="same", strides=[1, 1])
            else:
                x1 = mb.conv(x=x, weight=W, bias=np.random.rand(8), pad_type="same", strides=[1, 1])
                x2 = mb.conv(x=x, weight=W, bias=np.random.rand(8), pad_type="same", strides=[1, 1])
            x1 = mb.relu(x=x1)
            x2 = mb.relu(x=x2)
            x1 = mb.avg_pool(x=x1, kernel_sizes=[2, 2], strides=[1, 1], pad_type="same")
            z = mb.concat(values=(x1, x2), axis=-3)
            ssa_fun.set_outputs([z])
        prog.add_function("main", ssa_fun)
        return prog

    def test_redundant_ops_inside_graph_valid_pattern(self):
        """
        Input graph:
        input--> relu--------->conv------>relu----> pool ---> concat ---> out
                 |                                              ^
                 |                                              |
                 |---->conv---->relu----------------------------

        Output graph:
        input-> relu--->conv------>relu----> pool ---> concat ---> out
                                    |                   ^
                                    |                   |
                                    |-------------------
        """
        prog = self._make_repeated_conv_prog(redundant_conv=True)

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::remove_redundant_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "conv", "conv", "relu", "relu", "avg_pool", "concat"]
        assert get_op_types_in_program(prog) == ["relu", "conv", "relu", "avg_pool", "concat"]
        assert_model_is_valid(
            prog,
            {"x": (1, 4, 5, 5)},
            expected_output_shapes={block.outputs[0].name: (1, 16, 5, 5)},
        )

    def test_redundant_ops_inside_graph_invalid_pattern(self):
        """
        input--->relu--------->conv1------>relu----> pool ---> concat ---> out
                  |                                              ^
                  |                                              |
                  |---->conv2---->relu---------------------------
        """
        prog = self._make_repeated_conv_prog(redundant_conv=False)

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::remove_redundant_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "conv", "conv", "relu", "relu", "avg_pool", "concat"]
        assert get_op_types_in_program(prog) == ["relu", "conv", "conv", "relu", "relu", "avg_pool", "concat"]
        assert_model_is_valid(
            prog,
            {"x": (1, 4, 5, 5)},
            expected_output_shapes={block.outputs[0].name: (1, 16, 5, 5)},
        )

    def test_redundant_op_as_output_valid_pattern_1(self):
        """
        Input graph:
        input--------->relu------> out1
               |
               |
               |---->relu---->tanh---> out2

        Output graph:
        input--------->relu------> out1
                             |
                             |
                             |---->tanh---> out2
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 5))])
        def prog(x):
            x1 = mb.relu(x=x)
            x2 = mb.relu(x=x)
            return x1, mb.tanh(x=x2)

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::remove_redundant_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "relu", "tanh"]
        assert get_op_types_in_program(prog) == ["relu", "tanh"]
        assert_model_is_valid(
            prog,
            {"x": (2, 3, 5)},
            expected_output_shapes={block.outputs[0].name: (2, 3, 5), block.outputs[1].name: (2, 3, 5)},
        )

    def test_redundant_op_as_output_invalid_pattern_1(self):
        """
        Input graph:
        input--------->relu------> out1
               |
               |
               |---->relu---> out2

        "common::remove_redundant_ops" pass does not remove ops if their outputs
        are block outputs.
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 5))])
        def prog(x):
            x1 = mb.relu(x=x)
            x2 = mb.relu(x=x)
            return x1, x2

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::remove_redundant_ops",
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "relu"]
        assert get_op_types_in_program(prog) == ["relu", "relu"]
        assert_model_is_valid(
            prog,
            {"x": (2, 3, 5)},
            expected_output_shapes={block.outputs[0].name: (2, 3, 5), block.outputs[1].name: (2, 3, 5)},
        )

    def test_cond_block_program(self):
        # to test:
        # - identical ops within different blocks are not removed. The "relu" op inside true and false blocks
        #   are not removed since they are in different blocks
        # - ops that have blocks inside them are not removed. There are two cond ops here, with identical inputs
        #   but they are not removed, since they are ops that have nested block inside them
        @mb.program(input_specs=[mb.TensorSpec(shape=(1,))])
        def prog(x):
            x1 = mb.cast(x=x, dtype="bool")
            def true_fn():
                x = mb.shape(x=x1)
                x = mb.cast(x=x, dtype="fp32")
                return mb.add(x=x, y=1.)

            def false_fn():
                x = mb.shape(x=x1)
                x = mb.cast(x=x, dtype="fp32")
                return mb.add(x=x, y=-1.)

            z1 = mb.cond(pred=x1, _true_fn=true_fn, _false_fn=false_fn)
            z2 = mb.cond(pred=x1, _true_fn=true_fn, _false_fn=false_fn)
            z = mb.add(x=z1, y=z2)
            return z

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::remove_redundant_ops",
        )
        assert get_op_types_in_program(prev_prog) == ["cast", "cond", "cond", "add"]
        assert get_op_types_in_program(prog) == ["cast", "cond", "cond", "add"]
        cond_op = prog.find_ops(op_type="cond")[0]
        assert cond_op.blocks[0].operations[0].op_type == "shape"
        assert cond_op.blocks[1].operations[0].op_type == "shape"
        assert_model_is_valid(
            prog,
            {"x": (1,)},
            expected_output_shapes={block.outputs[0].name: (1,)},
        )

    def test_concat_op_pattern(self):
        '''
        Input graph:
                          ---------------> concat ------> log ------> out1
                         |                   ^
                         |                   |
        input--------->relu------> concat ------> relu----> out2
                 |                  ^        |
                 |                  |        |
                 |---->tanh--------------------

        Output graph:
                                     |------>log ------> out1
                                     |
                                     |
        input--------->relu------> concat ------> relu----> out2
                 |                  ^
                 |                  |
                 |---->tanh---------
        '''

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 5))])
        def prog(x):
            x1 = mb.relu(x=x)
            x2 = mb.tanh(x=x)
            c1 = mb.concat(values=(x1, x2), axis=0)
            c2 = mb.concat(values=(x1, x2), axis=0)
            z1 = mb.log(x=c1)
            z2 = mb.relu(x=c2)
            return z1, z2

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::remove_redundant_ops",
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "tanh", "concat", "concat", "log", "relu"]
        assert get_op_types_in_program(prog) == ["relu", "tanh", "concat", "log", "relu"]
        assert_model_is_valid(
            prog,
            {"x": (10, 5)},
            expected_output_shapes={block.outputs[0].name: (20, 5), block.outputs[1].name: (20, 5)},
        )

    def test_multiple_redundant_child_ops_pattern(self):
        '''
        Input graph

        input -------------> reshape ----------> add ---------> out1
                  |                               ^
                  |                               |
                  |-------> reshape ---------------
                  |
                  |------> slice_by_size-----> add ----------> out2
                  |                             ^
                  |                             |
                  |------> slice_by_size -------

        Output graph

        input -------------> reshape ----------> add ------------> out1
          |                              |        ^
          |                              |        |
          |                              |---------
          |
          |------> slice_by_size----------> add -----------------> out2
                        |                    ^
                        |                    |
                        |---------------------

        '''

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 5, 4))])
        def prog(x):
            x1 = mb.reshape(x=x, shape=[5, 2, -1])
            x2 = mb.reshape(x=x, shape=[5, 2, -1])
            x3 = mb.slice_by_size(x=x, begin=[0, 0, 1], size=[2, 4, 3])
            x4 = mb.slice_by_size(x=x, begin=[0, 0, 1], size=[2, 4, 3])
            z1 = mb.add(x=x1, y=x2)
            z2 = mb.add(x=x3, y=x4)
            return z1, z2

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::remove_redundant_ops",
        )
        assert get_op_types_in_program(prev_prog) == ["reshape", "reshape", "slice_by_size", "slice_by_size", "add", "add"]
        assert get_op_types_in_program(prog) == ["reshape", "slice_by_size", "add", "add"]
        assert_model_is_valid(
            prog,
            {"x": (10, 5, 4)},
            expected_output_shapes={block.outputs[0].name: (5, 2, 20), block.outputs[1].name: (2, 4, 3)},
        )

    def test_random_distribution_op_invalid_pattern(self):
        """
        Identical random ops are not removed

        input----->cast---->random_uniform------> add ---> out
                    |                              ^
                    |                              |
                    |---->random_uniform------------
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(3,))])
        def prog(shape):
            shape = mb.cast(x=shape, dtype="int32")
            x1 = mb.random_uniform(shape=shape, low=0.0, high=1.0, seed=11)
            x2 = mb.random_uniform(shape=shape, low=0.0, high=1.0, seed=11)
            return mb.add(x=x1, y=x2)

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::remove_redundant_ops",
        )
        assert get_op_types_in_program(prev_prog) == ["cast", "random_uniform", "random_uniform", "add"]
        assert get_op_types_in_program(prog) == ["cast", "random_uniform", "random_uniform", "add"]


class TestPreluFusion:

    @pytest.mark.parametrize(
        "swap_input_order, alpha_rank",
        itertools.product(
            [True, False],
            [3, 4],
        )
    )
    def test_channel_first_pattern(self, swap_input_order, alpha_rank):
        """
        Input:
                          | ------------> relu --------------------|
                          |                                        V
           x (BCHW) ------|                                       add -----> y (BCHW)
                          |                                        ^
                          --------> mul -------> relu -----> mul---|
                                    ^                         ^
                                    |                         |
                                Const(val=-1)               Const(name=a, shape=(1,C,1,1))

        Output:
            x (BCHW) ------> prelu(alpha=a, shape=(C,)) ---------> y (BCHW)
        """
        B, C, H, W = 2, 3, 5, 6

        if alpha_rank == 3:
            alpha = np.random.rand(C, 1, 1)
        elif alpha_rank == 4:
            alpha = np.random.rand(1, C, 1, 1)
        else:
            raise NotImplementedError("alpha rank must be 3 or 4")

        @mb.program(input_specs=[mb.TensorSpec(shape=(B, C, H, W))])
        def prog(x):
            if swap_input_order:
                neg = mb.mul(x=x, y=-1.)
            else:
                neg = mb.mul(x=-1., y=x)
            relu1 = mb.relu(x=neg)
            if swap_input_order:
                mul = mb.mul(x=relu1, y=alpha)
            else:
                mul = mb.mul(x=alpha, y=relu1)
            relu2 = mb.relu(x=x)
            if swap_input_order:
                out = mb.add(x=relu2, y=mul)
            else:
                out = mb.add(x=mul, y=relu2)
            return out

        prev_prog, _, _ = apply_pass_and_basic_check(
            prog, "common::fuse_prelu",
        )
        assert get_op_types_in_program(prev_prog) == ["mul", "relu", "mul", "relu", "add"]
        assert get_op_types_in_program(prog) == ["prelu"]


    @pytest.mark.parametrize(
        "swap_input_order, alpha_rank",
        itertools.product(
            [True, False],
            [1, 2, 3],
        )
    )
    def test_channel_last_transpose_pattern(self, swap_input_order, alpha_rank):
        """
        Input:

                                                        | ------------> relu --------------------|
                                                        |                                        V
        x (shappe=BCHW)-->transpose(out_shape=BHWC)---->|                                       add -----> y (BHWC)
                                                        |                                        ^
                                                        --------> mul -------> relu -----> mul---|
                                                                   ^                        ^
                                                                   |                        |
                                                           Const(val=-1)             Const(shape=(1,1,C))

        Output:
            x (BCHW) ------> prelu ---------> transpose ------> y (BHWC)
        """
        B, C, H, W = 2, 3, 5, 6
        if alpha_rank == 1:
            alpha = np.random.rand(C)
        elif alpha_rank == 2:
            alpha = np.random.rand(1, C)
        elif alpha_rank == 3:
            alpha = np.random.rand(1, 1, C)
        else:
            raise NotImplementedError("alpha rank must be 1 or 2 or 3")

        @mb.program(input_specs=[mb.TensorSpec(shape=(B, C, H, W))])
        def prog(x):
            x = mb.transpose(x=x, perm=[0,2,3,1])
            if swap_input_order:
                neg = mb.mul(x=x, y=-1.)
            else:
                neg = mb.mul(x=-1., y=x)
            relu1 = mb.relu(x=neg)
            if swap_input_order:
                mul = mb.mul(x=relu1, y=alpha)
            else:
                mul = mb.mul(x=alpha, y=relu1)
            relu2 = mb.relu(x=x)
            if swap_input_order:
                out = mb.add(x=relu2, y=mul)
            else:
                out = mb.add(x=mul, y=relu2)
            return out

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::fuse_prelu",
        )
        assert get_op_types_in_program(prev_prog) == ["transpose", "mul", "relu", "mul", "relu", "add"]
        assert get_op_types_in_program(prog) == ["prelu", "transpose"]
        assert_model_is_valid(
            prog,
            {"x": (B, C, H, W)},
            expected_output_shapes={block.outputs[0].name: (B, H, W, C)},
        )

class TestUpdateOutputDtypePass:

    def test_single_output(self):
        """
        Given:
        ------
        main(%input: (1, 20, int32)(Tensor)) {
          block0() {
            %abs: (1, 20, int32)(Tensor) = abs(x=%input, name="abs")
            %output_square: (1, 20, int32)(Tensor) = square(x=%input, name="output_square")
          } -> (%output_square)
        }
        prog.main_output_types = [ct.TensorType(dtype=np.float16)]

        Result:
        ------
        main(%input: (1, 20, int32)(Tensor)) {
          block0() {
            %abs: (1, 20, int32)(Tensor) = abs(x=%input, name="abs")
            %output_square_type_int32: (1, 20, int32)(Tensor) = square(x=%input, name="output_square")
            %output_square: (1, 20, fp16)(Tensor) = cast(x=%output_square_type_int32, dtype="fp16", name="cast_0")
          } -> (%output_square)
        }
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 20), dtype=types.int32)])
        def prog(input):
            x = mb.abs(x=input, name="abs")
            x = mb.square(x=input, name="output_square")
            return x

        prog.set_main_output_types([ct.TensorType(dtype=np.float16)])
        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::update_output_dtypes"
        )
        assert get_op_types_in_program(prev_prog) == ["abs", "square"]
        assert prev_block.outputs[0].dtype == types.int32
        assert get_op_types_in_program(prog) == ["abs", "square", "cast"]
        assert block.outputs[0].dtype == types.fp16
        assert block.outputs[0].name == "output_square"

    def test_multiple_outputs(self):
        """
        Given:
        -----
        main(%input: (1, 20, int32)(Tensor)) {
          block0() {
            %split_0: (1, 10, int32)(Tensor), %split_1: (1, 10, int32)(Tensor) = split(x=%input, num_splits=2, axis=1, name="split")
          } -> (%split_0, %split_1)
        }
        prog.main_output_types = [ct.TensorType(), ct.TensorType(dtype=np.float16)]

        Result:
        ------
        main(%input: (1, 20, int32)(Tensor)) {
          block0() {
            %split_0: (1, 10, int32)(Tensor), %split_1_type_int32: (1, 10, int32)(Tensor) = split(x=%input, num_splits=2, axis=1, name="split")
            %split_1: (1, 10, fp16)(Tensor) = cast(x=%split_1_type_int32, dtype="fp16", name="cast_0")
          } -> (%split_0, %split_1)
        }

        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 20), dtype=types.int32)])
        def prog(input):
            x1, x2 = mb.split(x=input, num_splits=2, axis=1, name="split")
            return x1, x2

        prog.set_main_output_types([ct.TensorType(), ct.TensorType(dtype=np.float16)])
        _, _, block = apply_pass_and_basic_check(
            prog, "common::update_output_dtypes"
        )
        assert get_op_types_in_program(prog) == ["split", "cast"]
        assert block.outputs[1].dtype == types.fp16
        assert block.outputs[1].name == "split_1"


def _get_constexpr_cast(shape):
    val = np.random.rand(*shape).astype(np.float16)
    return mb.constexpr_cast(source_val=val, output_dtype="fp32")

def _get_constexpr_sparse_to_dense(shape):
    val = np.random.rand(*shape)
    sparse_params = WeightSparsifier.compress(val=val, mode="PERCENTILE_MODE", target_percentile=0.4)
    return mb.constexpr_sparse_to_dense(
            nonzero_data=sparse_params.nonzero_data,
            mask=sparse_params.mask,
            shape=np.uint32(sparse_params.shape),)

def _get_constexpr_lut_to_dense(shape):
    val = np.random.rand(*shape)
    lut_params = WeightPalettizer.compress(val=val, nbits=4, mode="UNIFORM")
    return mb.constexpr_lut_to_dense(
            indices=lut_params.indices,
            lut=lut_params.lut,
            shape=np.uint32(lut_params.shape),)

def _get_constexpr_affine_dequantize(shape):
    val = np.random.rand(*shape)
    quant_params = WeightAffineQuantizer.compress(val=val, axis=0, mode="LINEAR_SYMMETRIC")
    return mb.constexpr_affine_dequantize(
            quantized_data=quant_params.quantized_data,
            zero_point=quant_params.zero_point,
            scale=quant_params.scale,
            axis=quant_params.axis,)

CONSTEXPR_FUNCS = {
    "constexpr_cast": _get_constexpr_cast,
    "constexpr_sparse_to_dense": _get_constexpr_sparse_to_dense,
    "constexpr_lut_to_dense": _get_constexpr_lut_to_dense,
    "constexpr_affine_dequantize": _get_constexpr_affine_dequantize,
}

CONSTEXPR_OPS = [
    "constexpr_cast",
    "constexpr_sparse_to_dense",
    "constexpr_lut_to_dense",
    "constexpr_affine_dequantize"
]

class TestSkipConstexprOps:

    @staticmethod
    @pytest.mark.parametrize(
        "constexpr_op",
        CONSTEXPR_OPS,
    )
    def test_skip_const_elimination(constexpr_op):
        """
                           constexpr_op
                                 |
                                 v
                      const -> linear 
                                 |
                                 v
          input --------------> add -> output 
        
        We are testing that:
        1. constexpr_op can serve as a const input weight for linear op
        2. linear op shoudn't be removed by the const_elimination pass
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(4,))])
        def prog(x):
            a = np.random.rand(2,)
            constexpr = CONSTEXPR_FUNCS[constexpr_op]((4, 2))
            linear = mb.linear(x=a, weight=constexpr)
            return mb.add(x=x, y=linear)

        PASS_REGISTRY["common::const_elimination"](prog)
        assert get_op_types_in_program(prog) == [constexpr_op, "linear", "add"]

    @staticmethod
    @pytest.mark.parametrize(
        "constexpr_op, weight_constexpr, bias_constexpr",
        itertools.product(
            CONSTEXPR_OPS,
            [True, False],
            [True, False],
        )
    )
    def test_skip_fuse_matmul_weight_bias(constexpr_op, weight_constexpr, bias_constexpr):
        """
                    const_1       const_2
                       |            |
                       v            v
        input -----> matmul -----> add ---> out

        In this case, if either const_1 or const_2 is constexpr op, they should be not fused into a single linear op
        """

        def get_matmul(x, weight_constexpr):
            weight = CONSTEXPR_FUNCS[constexpr_op]((3, 2))
            if not weight_constexpr:
                weight = weight.val
            return mb.matmul(x=x, y=weight)

        def get_add(x, bias_constexpr):
            bias = CONSTEXPR_FUNCS[constexpr_op]((2,))
            if not bias_constexpr:
                bias = bias.val
            return mb.add(x=x, y=bias)
        
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3))])
        def prog(x):
            x = get_matmul(x, weight_constexpr)
            x = get_add(x, bias_constexpr)
            return x
        
        apply_pass_and_basic_check(prog, "common::fuse_matmul_weight_bias")
        apply_pass_and_basic_check(prog, "common::const_elimination")
        apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        
        if not weight_constexpr and not bias_constexpr:
            expected_ops = ["linear"]
        else:
            expected_ops = []
            if weight_constexpr:
                expected_ops.append(constexpr_op)
            expected_ops.append("matmul")
            if bias_constexpr:
                expected_ops.append(constexpr_op)
            expected_ops.append("add")
        
        assert get_op_types_in_program(prog) == expected_ops

    @staticmethod
    @pytest.mark.parametrize(
        "constexpr_op, op, weight_constexpr, const_constexpr",
        itertools.product(
            CONSTEXPR_OPS,
            ["mul", "add"],
            [True, False],
            [True, False],
        )
    )
    def test_skip_fuse_conv(constexpr_op, op, weight_constexpr, const_constexpr):

        """
                    const_1       const_2
                       |            |
                       v            v
        input -----> conv -----> mul/add ---> out

        This pattern shouldn't be fused into a single conv layer if one of const_1 or const_2 is a constexpr op.
        """
        Cin, Cout = 3, 3
        input_shape = (2, Cin, 5, 5)
        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            conv_weight = CONSTEXPR_FUNCS[constexpr_op]((Cout, Cin, 2, 2)) 
            if not weight_constexpr:
                conv_weight = conv_weight.val
            x =  mb.conv(x=x, weight=conv_weight)
            const = CONSTEXPR_FUNCS[constexpr_op]((Cout, 1, 1))
            if not const_constexpr:
                const = const.val
            return getattr(mb, op)(x=x, y=const)
            
        apply_pass_and_basic_check(prog, "common::fuse_conv_scale")
        apply_pass_and_basic_check(prog, "common::fuse_conv_bias")
        apply_pass_and_basic_check(prog, "common::const_elimination")
        apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        expected_ops = []
        if not weight_constexpr and not const_constexpr:
            expected_ops = ["conv"]
        else:
            if weight_constexpr:
                expected_ops.append(constexpr_op)
            expected_ops.append("conv")
            if const_constexpr:
                expected_ops.append(constexpr_op)
            if op != "add" or const_constexpr:
                expected_ops.append(op)

        assert get_op_types_in_program(prog) == expected_ops

    @staticmethod
    @pytest.mark.parametrize(
        "constexpr_op, weight_constexpr, bias_constexpr",
        itertools.product(
            CONSTEXPR_OPS,
            [True, False],
            [True, False],
        )
    )
    def test_skip_fuse_linear_bias(constexpr_op, weight_constexpr, bias_constexpr):
        """
                     const_1      const_2
                       |            |
                       v            V
        input -----> linear -----> add ---> out

        This pattern shouldn't be fused into a single linear layer if one of const_1 or const_2 is a constexpr op.
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(2,))])
        def prog(x):
            weight = CONSTEXPR_FUNCS[constexpr_op]((4, 2))
            if not weight_constexpr:
                weight = weight.val
            linear = mb.linear(x=x, weight=weight)
            bias = CONSTEXPR_FUNCS[constexpr_op]((4,))
            if not bias_constexpr:
                bias = bias.val
            return mb.add(x=linear, y=bias)

        apply_pass_and_basic_check(prog, "common::fuse_linear_bias")
        apply_pass_and_basic_check(prog, "common::const_elimination")
        apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        expected_ops = []
        if not weight_constexpr and not bias_constexpr:
            expected_ops = ["linear"]
        else:
            if weight_constexpr:
                expected_ops.append(constexpr_op)
            expected_ops.append("linear")
            if bias_constexpr:
                expected_ops.append(constexpr_op)
            expected_ops.append("add")

        assert get_op_types_in_program(prog) == expected_ops


    @staticmethod
    @pytest.mark.parametrize(
        "constexpr_op, weight_constexpr, bias_constexpr",
        itertools.product(
            CONSTEXPR_OPS,
            [True, False],
            [True, False],
        )
    )
    def test_skip_fuse_conv_batchnorm(constexpr_op, weight_constexpr, bias_constexpr):
        """
              weight        bias
                |            |
                |_____   ____|
                      | |
                      v v
        input -----> conv -----> batch_norm ---> out

        This pattern shouldn't be fused into a single conv layer if one of the weight / bias is a constexpr op.
        """
        Cin, Cout = 2, 3
        input_shape = (2, Cin, 5, 5)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            # conv layer
            weight = CONSTEXPR_FUNCS[constexpr_op]((Cout, Cin, 2, 2))
            if not weight_constexpr:
                weight = weight.val
            bias = CONSTEXPR_FUNCS[constexpr_op]((Cout, ))
            if not bias_constexpr:
                bias = bias.val

            x = mb.conv(
                    x=x,
                    weight=weight,
                    bias=bias,
                )

            # batch_norm layer
            gamma = np.random.rand(Cout)
            beta = np.random.rand(Cout)
            mean = np.random.rand(Cout)
            variance = np.random.rand(Cout)
            epsilon = 1e-2
            return mb.batch_norm(
                    x=x,
                    mean=mean,
                    variance=variance,
                    gamma=gamma,
                    beta=beta,
                    epsilon=epsilon,
                    )

        apply_pass_and_basic_check(prog, "common::fuse_conv_batchnorm")
        apply_pass_and_basic_check(prog, "common::const_elimination")
        apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        expected_ops = []
        if not weight_constexpr and not bias_constexpr:
            expected_ops = ["conv"]
        else:
            expected_ops = [constexpr_op] * sum([weight_constexpr, bias_constexpr]) + ["conv", "batch_norm"]

        assert get_op_types_in_program(prog) == expected_ops
