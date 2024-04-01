#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
import itertools
import unittest

import numpy as np
import pytest

import coremltools as ct
import coremltools.optimize as cto
from coremltools._deps import _IS_MACOS
from coremltools.converters.mil import mil
from coremltools.converters.mil.experimental.passes.generic_pass_infrastructure import (
    register_generic_pass,
)
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, types
from coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_unary import cast as _cast_iOS14
from coremltools.converters.mil.mil.ops.defs.iOS17.elementwise_unary import cast as _cast_iOS17
from coremltools.converters.mil.mil.passes.defs.optimize_repeat_ops import cast_optimization
from coremltools.converters.mil.mil.passes.helper import _check_var_scalar_value
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.converters.mil.mil.scope import ScopeInfo, ScopeSource
from coremltools.converters.mil.mil.types import numpy_type_to_builtin_type
from coremltools.converters.mil.mil.types.type_mapping import builtin_to_string
from coremltools.converters.mil.testing_reqs import backends
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    assert_op_count_match,
    assert_same_output_names,
    get_op_types_in_block,
    get_op_types_in_program,
)
from coremltools.models.utils import _macos_version

np.random.seed(1984)
_VALIDATE_MODEL = True


def _get_constexpr_cast(shape, seed=None):
    if seed is not None:
        np.random.seed(seed)
    val = np.random.rand(*shape).astype(np.float16)
    return mb.constexpr_cast(source_val=val, output_dtype="fp32")


def _get_constexpr_sparse_to_dense(shape, seed=None):
    if seed is not None:
        np.random.seed(seed)
    val = np.random.rand(*shape)
    sparse_params = cto.coreml._quantization_passes.prune_weights.compress_by_magnitude(
        val=val, target_sparsity=0.4
    )
    return mb.constexpr_sparse_to_dense(
        nonzero_data=sparse_params.nonzero_data,
        mask=sparse_params.mask,
        shape=np.uint32(sparse_params.shape),
    )


def _get_constexpr_lut_to_dense(shape, seed=None):
    if seed is not None:
        np.random.seed(seed)
    val = np.random.rand(*shape)
    lut_params = cto.coreml._quantization_passes.palettize_weights.compress(
        val=val, nbits=4, mode="UNIFORM"
    )
    return mb.constexpr_lut_to_dense(
        indices=lut_params.indices,
        lut=lut_params.lut,
        shape=np.uint32(lut_params.shape),
    )


def _get_constexpr_affine_dequantize(shape, seed=None):
    if seed is not None:
        np.random.seed(seed)
    val = np.random.rand(*shape)
    quant_params = cto.coreml._quantization_passes.linear_quantize_weights.compress(
        val=val, axis=0, mode="LINEAR_SYMMETRIC", dtype=types.uint8
    )
    return mb.constexpr_affine_dequantize(
        quantized_data=quant_params.quantized_data,
        zero_point=quant_params.zero_point,
        scale=quant_params.scale,
        axis=quant_params.axis,
    )


def _get_constexpr_val(constexpr_var):
    assert "constexpr" in constexpr_var.op.op_type
    if constexpr_var.val is not None:
        return constexpr_var.val
    return constexpr_var.op.materialized_val_inference()


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
    "constexpr_affine_dequantize",
]


class TestFuseSqueezeExpandDims:
    @pytest.mark.parametrize(
        "rank",
        [1, 5],
    )
    def test_fuse_squeeze_expand_dims_basic(self, rank):
        """
        Given:
          %1 = squeeze(%x)
          %2 = expand_dims(%1)
          %3 = relu(%2)

        Result:
          %3 = relu(%x)
        """
        if rank == 1:
            input_shape = (1,)
            axes = (0,)
        else:
            assert rank == 5
            input_shape = (3, 1, 4, 1, 1)
            axes = (1, 3, 4)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            x = mb.squeeze(x=x, axes=axes)
            x = mb.expand_dims(x=x, axes=axes)
            return mb.relu(x=x)

        # fuse_squeeze_expand_dims fused squeeze + expand_dims into identity
        apply_pass_and_basic_check(prog, "common::fuse_squeeze_expand_dims")
        assert get_op_types_in_program(prog) == ["identity", "relu"]

        # noop_elimination can further remove the identity op
        apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prog) == ["relu"]

    def test_fuse_squeeze_expand_dims_negative(self):
        """
        If squeeze and expand_dims cannot cancel each other,
        the graph pass does nothing
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 1, 4, 1, 1))])
        def prog(x):
            x = mb.squeeze(x=x, axes=(1, 2))
            x = mb.expand_dims(x=x, axes=(1, 3))
            return mb.relu(x=x)

        apply_pass_and_basic_check(prog, "common::fuse_squeeze_expand_dims")
        assert get_op_types_in_program(prog) == ["squeeze", "expand_dims", "relu"]

    def test_fuse_squeeze_expand_dims_connected_output(self):
        """
        If squeeze is connected to block output, it cannot be removed.
        However, the expand_dims can be a block output.
        """
        # squeeze connected to output. Nothing happens.
        @mb.program(input_specs=[mb.TensorSpec(shape=(1,))])
        def prog(x):
            squeeze = mb.squeeze(x=x, axes=(0,))
            expand_dims = mb.expand_dims(x=squeeze, axes=(0,))
            return mb.relu(x=expand_dims), squeeze

        apply_pass_and_basic_check(prog, "common::fuse_squeeze_expand_dims")
        assert get_op_types_in_program(prog) == ["squeeze", "expand_dims", "relu"]

        # expand_dims connected to output. Still good to fuse.
        @mb.program(input_specs=[mb.TensorSpec(shape=(1,))])
        def prog(x):
            squeeze = mb.squeeze(x=x, axes=(0,))
            expand_dims = mb.expand_dims(x=squeeze, axes=(0,))
            return mb.relu(x=expand_dims), expand_dims

        apply_pass_and_basic_check(prog, "common::fuse_squeeze_expand_dims")
        assert get_op_types_in_program(prog) == ["identity", "relu"]


class TestAddConvTransposeOutputShape:
    def test_add_conv_transpose_output_shape(self):
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
        prev_conv_transpose_op = prev_prog.find_ops(op_type="conv_transpose", exactly_one=True)[0]
        conv_transpose_op = prog.find_ops(op_type="conv_transpose", exactly_one=True)[0]
        assert np.all(conv_transpose_op.output_shape.val == prev_conv_transpose_op.outputs[0].shape)


class TestChildOrdering:
    def test_generic_child_ordering(self):
        """
        Checks that the new generic pattern matching infrastructure works
        regardless of the ordering of an operation's children
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            power = mb.pow(x=x, y=3.0, name="thepowerop")
            add_0 = mb.add(x=power, y=5.0, name="add_0")
            sub_0 = mb.sub(x=power, y=5.0, name="sub_0")
            mul_0 = mb.mul(x=power, y=5.0, name="mul_0")
            add_1 = mb.add(x=add_0, y=mul_0, name="add_1")
            add_2 = mb.add(x=sub_0, y=add_1, name="add_2")
            return add_2

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def ops_arrangement(x):
            power = mb.pow(x=x, y=3.0, name="thepowerop")
            sub_0 = mb.sub(x=power, y=5.0, name="sub_0")
            add_0 = mb.add(x=power, y=5.0, name="add_0")
            mul_0 = mb.mul(x=power, y=5.0, name="mul_0")
            add_1 = mb.add(x=add_0, y=mul_0, name="add_1")
            add_2 = mb.add(x=sub_0, y=add_1, name="add_2")
            return add_2

        def var_constraints(pattern):
            constraints_passed = True
            constraints_passed &= _check_var_scalar_value(pattern.thepowerop.y, 3)
            constraints_passed &= _check_var_scalar_value(pattern.sub_0.y, 5)
            constraints_passed &= _check_var_scalar_value(
                pattern.add_0.x, 5
            ) or _check_var_scalar_value(pattern.add_0.y, 5)
            constraints_passed &= _check_var_scalar_value(
                pattern.mul_0.x, 5
            ) or _check_var_scalar_value(pattern.mul_0.y, 5)
            return constraints_passed

        def transform_pattern(pattern):
            out_name = "new operation"
            x = mb.gelu(
                x=pattern.root_var,
                mode="TANH_APPROXIMATION",
                name=out_name,
                before_op=pattern.thepowerop,
            )

            pattern.add_2.enclosing_block.replace_uses_of_var_after_op(
                anchor_op=pattern.add_2, old_var=pattern.add_2.outputs[0], new_var=x
            )

            pattern.block.remove_ops(pattern.op_list())

        register_generic_pass(
            ops_arrangement=ops_arrangement,
            var_constraints=var_constraints,
            transform_pattern=transform_pattern,
            pass_name="test_generic_child_ordering",
            namespace="common",
        )

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


class TestGeluFusion:
    def test_gelu_tanh_approximation1(self):
        """
        Detect gelu tanh approx pattern, found in the TF bert model.
        y = ( tanh((.0447)x^3 + x ) * (sqrt(2/pi)) + 1 ) * 0.5 * x
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            x1 = mb.pow(x=x, y=3.0)
            x1 = mb.mul(x=0.044715, y=x1)
            x1 = mb.add(x=x1, y=x)
            x1 = mb.mul(x=x1, y=np.sqrt(2 / np.pi))
            x1 = mb.tanh(x=x1)
            x1 = mb.add(x=1.0, y=x1)
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
            [True, False], [True, False], [True, False], [True, False], [True, False], [True, False]
        ),
    )
    def test_gelu_tanh_approximation2(
        self, first_op_1, first_op_2, first_op_3, first_op_4, first_op_5, first_op_6
    ):
        """
        Detect gelu tanh approx pattern, found in the TF Sanitized GPT2 model.
        y = ( tanh((.0447)x^3 + x ) * (sqrt(2/pi)) + 1 ) * 0.5 * x
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            firstmul = mb.mul(x=x, y=0.5) if first_op_1 else mb.mul(x=0.5, y=x)
            x1 = mb.pow(x=x, y=3.0)
            x1 = mb.mul(x=0.044715, y=x1) if first_op_2 else mb.mul(x=x1, y=0.044715)
            x1 = mb.add(x=x1, y=x) if first_op_3 else mb.add(x=x, y=x1)
            x1 = (
                mb.mul(x=x1, y=np.sqrt(2 / np.pi))
                if first_op_4
                else mb.mul(x=np.sqrt(2 / np.pi), y=x1)
            )
            x1 = mb.tanh(x=x1)
            x1 = mb.add(x=1.0, y=x1) if first_op_5 else mb.add(x=x1, y=1.0)
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

    def test_gelu_tanh_multiple_final_operations(self):
        """
        The generic pattern matching only supports one final output operation. For multiple final
        operations, we want to make sure it just skip the pattern matching instead of failing the
        whole conversion.
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            x_1 = mb.mul(x=x, y=0.5)
            x_2 = mb.pow(x=x, y=3.0)
            x_2 = mb.mul(x=x_2, y=0.044715)
            x_2 = mb.add(x=x, y=x_2)
            x_2 = mb.mul(x=x_2, y=np.sqrt(2 / np.pi))
            x_2 = mb.tanh(x=x_2)
            x_2 = mb.add(x=x_2, y=1.0)
            x_2 = mb.mul(x=x_1, y=x_2)
            x_2 = mb.mul(x=x_2, y=1.0)
            return x_2

        with pytest.warns(
            UserWarning,
            match="User defined pattern matched to more than one final operation. "
            "Skipped the pattern matching.",
        ):
            apply_pass_and_basic_check(prog, "common::fuse_gelu_tanh_approximation")

    @pytest.mark.parametrize(
        "op_type, is_first_op1, is_first_op2, is_first_op3, is_first_op4, const_mul_first",
        itertools.product(
            ["real_div", "mul"],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
        ),
    )
    def test_gelu_exact(
        self, op_type, is_first_op1, is_first_op2, is_first_op3, is_first_op4, const_mul_first
    ):
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
            x3 = mb.add(x=x2, y=1.0) if is_first_op2 else mb.add(x=1.0, y=x2)

            if const_mul_first:
                y1 = mb.const(val=0.5)
                y2 = x
            else:
                y1 = x
                y2 = mb.const(val=0.5)

            x4 = mb.mul(x=x3, y=y1) if is_first_op3 else mb.mul(x=y1, y=x3)
            x5 = mb.mul(x=x4, y=y2) if is_first_op4 else mb.mul(x=y2, y=x4)

            return x5

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_gelu_exact")

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
        ),
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
            x3 = mb.add(x=x2, y=1.0)
            x4 = mb.mul(x=x0, y=x3) if is_first_op4 else mb.mul(x=x3, y=x0)
            return x4

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_gelu_exact")

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


class TestLeakyReluFusion:
    @pytest.mark.parametrize(
        "swap_mul_input_order, swap_max_input_order",
        itertools.product(
            [True, False],
            [True, False],
        ),
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

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_leaky_relu")
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

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_leaky_relu")
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

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_leaky_relu")
        assert get_op_types_in_program(prev_prog) == ["mul", "maximum"]
        assert get_op_types_in_program(prog) == ["mul", "maximum"]


class TestPreluFusion:
    @pytest.mark.parametrize(
        "swap_input_order, alpha_rank",
        itertools.product(
            [True, False],
            [3, 4],
        ),
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
                neg = mb.mul(x=x, y=-1.0)
            else:
                neg = mb.mul(x=-1.0, y=x)
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
            prog,
            "common::fuse_prelu",
        )
        assert get_op_types_in_program(prev_prog) == ["mul", "relu", "mul", "relu", "add"]
        assert get_op_types_in_program(prog) == ["prelu"]

    @pytest.mark.parametrize(
        "swap_input_order, alpha_rank",
        itertools.product(
            [True, False],
            [1, 2, 3],
        ),
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
            x = mb.transpose(x=x, perm=[0, 2, 3, 1])
            if swap_input_order:
                neg = mb.mul(x=x, y=-1.0)
            else:
                neg = mb.mul(x=-1.0, y=x)
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
            prog,
            "common::fuse_prelu",
        )
        assert get_op_types_in_program(prev_prog) == [
            "transpose",
            "mul",
            "relu",
            "mul",
            "relu",
            "add",
        ]
        assert get_op_types_in_program(prog) == ["prelu", "transpose"]
        assert_model_is_valid(
            prog,
            {"x": (B, C, H, W)},
            expected_output_shapes={block.outputs[0].name: (B, H, W, C)},
        )


class TestPreluToLrelu:
    def test_prelu_to_lrelu(self):
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
        prev_prog, _, _ = apply_pass_and_basic_check(prog, "common::prelu_to_lrelu")
        assert_same_output_names(prev_prog, prog)
        # The prelu with a common leakage factor becomes leaky_relu.
        assert_op_count_match(prog, expect=1, op="prelu")
        assert_op_count_match(prog, expect=1, op="leaky_relu")

        if _VALIDATE_MODEL:
            assert_model_is_valid(prog, {"x": (4, 2, 3, 1)})


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
        2. linear op shouldn't be removed by the const_elimination pass
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(4,))])
        def prog(x):
            a = np.random.rand(
                2,
            )
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
        ),
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
                weight = _get_constexpr_val(weight)
            return mb.matmul(x=x, y=weight)

        def get_add(x, bias_constexpr):
            bias = CONSTEXPR_FUNCS[constexpr_op]((2,))
            if not bias_constexpr:
                bias = _get_constexpr_val(bias)
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
        ),
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
                conv_weight = _get_constexpr_val(conv_weight)
            x = mb.conv(x=x, weight=conv_weight)
            const = CONSTEXPR_FUNCS[constexpr_op]((Cout, 1, 1))
            if not const_constexpr:
                const = _get_constexpr_val(const)
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
        ),
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
                weight = _get_constexpr_val(weight)
            linear = mb.linear(x=x, weight=weight)
            bias = CONSTEXPR_FUNCS[constexpr_op]((4,))
            if not bias_constexpr:
                bias = _get_constexpr_val(bias)
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
        ),
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
                weight = _get_constexpr_val(weight)
            bias = CONSTEXPR_FUNCS[constexpr_op]((Cout,))
            if not bias_constexpr:
                bias = _get_constexpr_val(bias)

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
            expected_ops = [constexpr_op] * sum([weight_constexpr, bias_constexpr]) + [
                "conv",
                "batch_norm",
            ]

        assert get_op_types_in_program(prog) == expected_ops


class TestMergeConsecutivePaddings:
    def test_success_reflect(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):
            pad1 = mb.pad(x=x1, pad=[0, 0, 1, 1], mode="reflect")
            pad2 = mb.pad(x=pad1, pad=[1, 1, 0, 0], mode="reflect")

            return pad2

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_paddings")
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
                pad1 = mb.pad(x=x1, pad=[1, 1], mode="reflect")
                pad2 = mb.pad(x=pad1, pad=[1, 1, 0, 0], mode="reflect")
            else:
                pad1 = mb.pad(x=x1, pad=[1, 1, 0, 0], mode="reflect")
                pad2 = mb.pad(x=pad1, pad=[1, 1], mode="reflect")

            return pad2

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_paddings")
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
            pad1 = mb.pad(x=x1, pad=[0, 0, 1, 1], mode="constant", constant_val=3.0)
            pad2 = mb.pad(x=pad1, pad=[1, 1, 0, 0], mode="constant", constant_val=3.0)

            return pad2

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_paddings")
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
            pad1 = mb.pad(x=x1, pad=[0, 0, 1, 1], mode="constant", constant_val=3.0)
            pad2 = mb.pad(x=pad1, pad=[1, 1, 0, 0], mode="constant", constant_val=3.0)
            pad3 = mb.pad(x=pad2, pad=[1, 1, 0, 0], mode="constant", constant_val=3.0)

            return pad3

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_paddings")
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
            pad1 = mb.pad(x=x1, pad=[0, 0, 1, 1], mode="reflect")
            pad2 = mb.pad(x=pad1, pad=[1, 1, 0, 0], mode="constant")

            return pad2

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_paddings")
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
            pad1 = mb.pad(x=x1, pad=[0, 0, 1, 1], mode="constant", constant_val=1.0)
            pad2 = mb.pad(x=pad1, pad=[1, 1, 0, 0], mode="constant", constant_val=2.0)

            return pad2

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_paddings")
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
            pad1 = mb.pad(x=x1, pad=[1, 1], mode="reflect")
            pad2 = mb.pad(x=pad1, pad=[1, 1], mode="reflect")

            return pad2

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_paddings")
        assert get_op_types_in_program(prev_prog) == ["pad", "pad"]
        assert get_op_types_in_program(prog) == ["pad", "pad"]

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 6, 12)},
        )

class TestMergeConsecutiveTransposes:
    def test_success_reduce_consecutive_transposes(self):
        """
        Input:
              |--> transpose_1 -> transpose_2 -> output_1
            x -
              |--> transpose_3 -> tranpose_4 -> transpose_5 -> output_2

        Output:
              |--> transpose_6 -> output_1
            x -
              |--> transpose_7 -> output_2
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x):
            x1 = mb.transpose(x=x, perm=[0, 2, 1, 3])
            x1 = mb.transpose(x=x1, perm=[3, 2, 0, 1])
            x2 = mb.transpose(x=x, perm=[3, 2, 1, 0])
            x2 = mb.transpose(x=x2, perm=[2, 3, 0, 1])
            x2 = mb.transpose(x=x2, perm=[0, 2, 1, 3])

            return x1, x2

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_transposes")
        assert get_op_types_in_program(prev_prog) == ["transpose"] * 5
        assert get_op_types_in_program(prog) == ["transpose"] * 2

        inputs = {"x": (1, 2, 3, 4)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={
                block.outputs[0].name: (4, 2, 1, 3),
                block.outputs[1].name: (2, 4, 1, 3),
            },
        )

    def test_success_reduce_consecutive_transposes_with_output_constrain(self):
        """
        Input:
            x --> transpose_1 -> transpose_2 -> transpose_3 -> transpose_4 -> transpose_5 -> add -> output_3
                       |                            |
                       v                            v
                    output_1                     output_2

        Output:
            x --> transpose_1 -> transpose_6 -> transpose_7-> add -> output_1
                       |             |
                       v             v
                    output_2       output_3
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x):
            x1 = mb.transpose(x=x, perm=[3, 2, 1, 0], name="output_1")
            x2 = mb.transpose(x=x1, perm=[1, 3, 2, 0])
            x2 = mb.transpose(x=x2, perm=[2, 3, 0, 1], name="output_2")
            x3 = mb.transpose(x=x2, perm=[0, 2, 1, 3])
            x3 = mb.transpose(x=x3, perm=[3, 2, 1, 0])
            x3 = mb.add(x=x3, y=1., name="output_3")
            return x1, x2, x3

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_transposes")
        assert get_op_types_in_program(prev_prog) == ["transpose"] * 5 + ["add"]
        assert get_op_types_in_program(prog) == ["transpose"] * 3 + ["add"]

        inputs = {"x": (1, 2, 3, 4)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={
                block.outputs[0].name: (4, 3, 2, 1),
                block.outputs[1].name: (2, 4, 3, 1),
                block.outputs[2].name: (1, 4, 3, 2),
            },
        )

        assert block.outputs[0].name == "output_1"
        assert block.outputs[1].name == "output_2"
        assert block.outputs[2].name == "output_3"

    def test_not_merge_transposes(self):
        """
        Input:
            x --> transpose_1 -> add -> transpose_2 -> output

        Output:
            x --> transpose_1 -> add -> transpose_2 -> output
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x):
            x = mb.transpose(x=x, perm=[3, 2, 1, 0])
            x = mb.add(x=x, y=1.)
            x = mb.transpose(x=x, perm=[1, 3, 2, 0])
            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_transposes")
        assert get_op_types_in_program(prev_prog) == ["transpose", "add", "transpose"]
        assert get_op_types_in_program(prog) == ["transpose", "add", "transpose"]

        inputs = {"x": (1, 2, 3, 4)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (3, 1, 2, 4),},
        )

class TestExpandHighRankReshapeAndTranspose:
    @staticmethod
    def _test_numerical(prog, input_shape, reshape_shape, perm, output_shape):
        x = np.random.rand(*input_shape)
        coreml_input = {"x": x}
        mlmodel = ct.convert(prog, source="milinternal")
        coreml_output = list(mlmodel.predict(coreml_input).values())[0]

        gt = np.reshape(x, reshape_shape)
        gt = np.transpose(gt, perm)
        gt = np.reshape(gt, output_shape)
        np.testing.assert_allclose(gt, coreml_output, rtol=1e-03, atol=1e-05)

    def test_rank6(self):
        input_shape = (1, 2, 3, 4, 5)
        reshape_shape = (1, 2, 3, 2, 2, 5)
        perm = (4, 5, 3, 2, 0, 1)
        output_shape = (5, 24)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            x = mb.reshape(x=x, shape=reshape_shape)
            x = mb.transpose(x=x, perm=perm)
            x = mb.reshape(x=x, shape=output_shape)
            return x
        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::expand_high_rank_reshape_and_transpose")

        prog._check_early_error_out_for_invalid_program()
        assert get_op_types_in_program(prog) == ["reshape", "transpose", "reshape"]
        TestExpandHighRankReshapeAndTranspose._test_numerical(prev_prog, input_shape, reshape_shape, perm, output_shape)

    def test_rank10(self):
        input_shape = (2, 3, 4, 5, 6)
        reshape_shape = (1, 2, 1, 3, 2, 2, 1, 5, 2, 3)
        perm = (0, 1, 2, 3, 4, 9, 5, 6, 7, 8)
        output_shape = (30, 24)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            x = mb.reshape(x=x, shape=reshape_shape)
            x = mb.transpose(x=x, perm=perm)
            x = mb.reshape(x=x, shape=output_shape)
            return x
        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::expand_high_rank_reshape_and_transpose")

        prog._check_early_error_out_for_invalid_program()
        assert get_op_types_in_program(prog) == ["reshape", "transpose", "reshape"]
        TestExpandHighRankReshapeAndTranspose._test_numerical(prev_prog, input_shape, reshape_shape, perm, output_shape)

    def test_rank20(self):
        input_shape = (4, 6, 8, 20, 40)
        reshape_shape = (1, 2, 2, 1, 2, 3, 2, 2, 2, 2, 2, 1, 1, 1, 5, 2, 2, 2, 1, 5)
        perm = (19, 14, 13, 12, 0, 3, 1, 2, 10, 5, 4, 6, 15, 11, 17, 18, 7, 8, 9, 16)
        output_shape = (24, 160, 40)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            x = mb.reshape(x=x, shape=reshape_shape)
            x = mb.transpose(x=x, perm=perm)
            x = mb.reshape(x=x, shape=output_shape)
            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::expand_high_rank_reshape_and_transpose")

        prog._check_early_error_out_for_invalid_program()
        assert get_op_types_in_program(prog) == ["reshape", "transpose"] * 16 + ["reshape"]
        TestExpandHighRankReshapeAndTranspose._test_numerical(prev_prog, input_shape, reshape_shape, perm, output_shape)

    def test_negative_case(self):
        input_shape = (4, 6, 8, 20, 40)
        reshape_shape = (1, 2, 2, 1, 2, 3, 2, 2, 2, 2, 2, 1, 1, 1, 5, 2, 2, 2, 1, 5)
        perm = (19, 14, 13, 12, 0, 3, 1, 2, 10, 5, 4, 6, 15, 11, 17, 18, 7, 8, 9, 16)
        output_shape = (24, 160, 40)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            x1 = mb.reshape(x=x, shape=reshape_shape)
            x2 = mb.transpose(x=x1, perm=perm)
            x3 = mb.reshape(x=x2, shape=output_shape)
            return x, x1

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::expand_high_rank_reshape_and_transpose")

        with pytest.raises(ValueError, match="Core ML only supports tensors with rank <= 5"):
            prog._check_early_error_out_for_invalid_program()


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

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_relus")
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

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_relus")
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

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_relus")
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
        assert get_op_types_in_program(prev_prog_output_transpose_2) == [
            "relu",
            "relu",
            "transpose",
        ]
        assert get_op_types_in_program(prog_output_transpose_2) == ["relu", "transpose"]
        assert prog_output_transpose_2["main"].operations[0].name == "transpose_1"
        # As the block's output has transpose_2, the original output name of the first operation
        # is replaced.
        assert prog_output_transpose_2["main"].operations[0].outputs[0].name == "transpose_2"

        prev_prog_output_transpose_3, _, block = apply_pass_and_basic_check(
            prog_output_transpose_3, "common::merge_consecutive_relus"
        )
        assert get_op_types_in_program(prev_prog_output_transpose_3) == [
            "relu",
            "relu",
            "transpose",
        ]
        assert get_op_types_in_program(prog_output_transpose_3) == ["relu", "transpose"]
        assert prog_output_transpose_3["main"].operations[0].name == "transpose_1"
        # As the block's output only has transpose_3, the entire transpose_2 gets removed.
        assert prog_output_transpose_3["main"].operations[0].outputs[0].name == "transpose_1"

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


class TestMergeConsecutiveReshapes:
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_merge_consecutive_2reshapes(self, backend):
        INPUT_SHAPE = (2, 3)
        OUTPUT_SHAPE = (3, 2)

        @mb.program(input_specs=[mb.TensorSpec(shape=INPUT_SHAPE)])
        def prog(x):
            y1 = mb.reshape(x=x, shape=(-1,))
            y2 = mb.reshape(x=y1, shape=OUTPUT_SHAPE)
            return y2

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_reshapes")
        assert get_op_types_in_program(prev_prog) == ["reshape"] * 2
        assert get_op_types_in_program(prog) == ["reshape"]

        assert_model_is_valid(
            prog,
            {"x": INPUT_SHAPE},
            expected_output_shapes={block.outputs[0].name: OUTPUT_SHAPE},
            backend=backend,
        )

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_merge_consecutive_4reshapes(self, backend):
        INPUT_SHAPE = (2, 3, 5)
        OUTPUT_SHAPE = (10, 3)

        @mb.program(input_specs=[mb.TensorSpec(shape=INPUT_SHAPE)])
        def prog(x):
            y1 = mb.reshape(x=x, shape=(15, 2))
            y2 = mb.reshape(x=y1, shape=(2, 5, 3))
            y3 = mb.reshape(x=y2, shape=(6, 5))
            y4 = mb.reshape(x=y3, shape=OUTPUT_SHAPE)
            return y4

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_reshapes")
        assert get_op_types_in_program(prev_prog) == ["reshape"] * 4
        assert get_op_types_in_program(prog) == ["reshape"]

        assert_model_is_valid(
            prog,
            {"x": INPUT_SHAPE},
            expected_output_shapes={block.outputs[0].name: OUTPUT_SHAPE},
            backend=backend,
        )

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_keep_separate_reshapes(self, backend):
        INPUT_SHAPE = (3, 5, 7)
        OUTPUT_SHAPE = (7, 3, 5)

        @mb.program(input_specs=[mb.TensorSpec(shape=INPUT_SHAPE)])
        def prog(x):
            y1 = mb.reshape(x=x, shape=(21, 5))

            # Note [elementwise op and reshape]
            # In principle, elementwise ops can be swapped with the reshapes, e.g.
            #     in -> reshape1 -> elementwise1 -> reshape2 -> elementwise2 -> reshape3 -> out
            # is equivalent to
            #     in -> elementwise1 -> elementwise2 -> reshape1 -> reshape2 -> reshape3 -> out
            # which can then be optimized to
            #     in -> elementwise1 -> elementwise2 -> reshape3 -> out
            #
            # so here we divide the reshape sequence with something non-elementwise
            bias = np.random.rand(5) * 2.0 - 1.0
            y2 = mb.add(x=y1, y=bias)

            y3 = mb.reshape(x=y2, shape=OUTPUT_SHAPE)
            return y3

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_reshapes")
        assert get_op_types_in_program(prev_prog) == ["reshape", "add", "reshape"]
        assert get_op_types_in_program(prog) == ["reshape", "add", "reshape"]

        assert_model_is_valid(
            prog,
            {"x": INPUT_SHAPE},
            expected_output_shapes={block.outputs[0].name: OUTPUT_SHAPE},
            backend=backend,
        )

    @pytest.mark.parametrize("backend", backends)
    def test_merge_2consecutive_keep_1separate(self, backend):
        INPUT_SHAPE = (5, 7, 11)
        OUTPUT_SHAPE = (11, 5, 7)

        @mb.program(input_specs=[mb.TensorSpec(shape=(INPUT_SHAPE))])
        def prog(x):
            # these 2 reshapes will be merged
            y1 = mb.reshape(x=x, shape=(35, 11))
            y2 = mb.reshape(x=y1, shape=(55, 7))

            # see Note [elementwise op and reshape]
            bias = np.random.rand(7) * 2.0 - 1.0
            y3 = mb.sub(x=y2, y=bias)

            # this reshape is separated, so it will be kept
            y4 = mb.reshape(x=y3, shape=OUTPUT_SHAPE)
            return y4

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_reshapes")
        assert get_op_types_in_program(prev_prog) == ["reshape", "reshape", "sub", "reshape"]
        assert get_op_types_in_program(prog) == ["reshape", "sub", "reshape"]

        assert_model_is_valid(
            prog,
            {"x": INPUT_SHAPE},
            expected_output_shapes={block.outputs[0].name: OUTPUT_SHAPE},
            backend=backend,
        )

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_keep_block_outputs(self, backend):
        INPUT_SHAPE = (5, 6)
        OUTPUT0_SHAPE = (15, 2)
        OUTPUT1_SHAPE = (3, 10)

        @mb.program(input_specs=[mb.TensorSpec(shape=INPUT_SHAPE)])
        def prog(x):
            y1 = mb.reshape(x=x, shape=OUTPUT0_SHAPE)
            y2 = mb.reshape(x=y1, shape=OUTPUT1_SHAPE)
            return y1, y2

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_reshapes")
        assert get_op_types_in_program(prev_prog) == ["reshape", "reshape"]
        assert get_op_types_in_program(prog) == ["reshape", "reshape"]

        assert len(block.outputs) == 2
        expected_output_shapes = {
            block.outputs[0].name: OUTPUT0_SHAPE,
            block.outputs[1].name: OUTPUT1_SHAPE,
        }
        assert_model_is_valid(
            prog,
            {"x": INPUT_SHAPE},
            expected_output_shapes=expected_output_shapes,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_keep_nonreshape_child(self, backend):
        INPUT_SHAPE = (6, 7)
        OUTPUT_SHAPE = (14, 3)

        @mb.program(input_specs=[mb.TensorSpec(shape=INPUT_SHAPE)])
        def prog(x):
            y1 = mb.reshape(x=x, shape=(21, 2))
            y2 = mb.reshape(x=y1, shape=OUTPUT_SHAPE)
            # the 1st reshape creating y1 has a non-reshape child op (matmul),
            # so it will not be merged
            y3 = mb.matmul(x=y1, y=np.random.rand(2, 5))
            return y2, y3

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::merge_consecutive_reshapes")
        assert get_op_types_in_program(prev_prog) == ["reshape", "reshape", "matmul"]
        assert get_op_types_in_program(prog) == ["reshape", "reshape", "matmul"]

        assert len(block.outputs) == 2
        assert_model_is_valid(
            prog,
            {"x": INPUT_SHAPE},
            expected_output_shapes={block.outputs[0].name: OUTPUT_SHAPE},
            backend=backend,
        )

class TestCastOptimizationReduendantCastRemoval:
    """
    Test single cast op removal.
    """

    def test_time_complexity(self):
        """
        This test makes sure the cast_optimization's time complexity is O(N) for most of the cases.

        In this test case, the program consists of 1000 relu ops followed by 100 cast ops.

            input -> relu -> relu -> ... -> relu -> cast -> cast -> ... -> cast

        The algorithm goes through the first pass to eliminate all cast ops:

            input -> relu -> ... -> relu

        Note that, the total number of visited op is 1000 (relu) + 100 (cast) + 100 (const for the dtype) = 1200.

        Because the fusion happens, the algorithm goes through the program again.
        This time, the number of visited op is 1000 (relu) + 100 (const) = 1100.

        Overally, the number of visited op is 2300.
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)])
        def prog(x):
            for _ in range(1000):
                x = mb.relu(x=x)
            for _ in range(100):
                x = mb.cast(x=x, dtype="fp32")
            return x

        graph_pass = cast_optimization()
        graph_pass.apply(prog)
        assert (
            graph_pass._num_of_visited_ops == 2_300
        )  # Please refer to the doc string for how 2300 comes from.

    def test_remove_redundant_cast_smoke(self):
        """
        Input graph:
        input(fp32) -> cast(dtype=fp32) -> output

        Output graph:
        input -> output
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)])
        def prog(x):
            x = mb.cast(x=x, dtype="fp32")
            return x

        assert get_op_types_in_program(prog) == ["cast"]

        _, _, block = apply_pass_and_basic_check(prog, "common::cast_optimization")

        assert len(block.find_ops(op_type="cast")) == 0
        assert block.outputs[0].dtype == types.fp32

    def test_remove_redundant_cast_negative_smoke(self):
        """
        Input graph:
        input(fp32) -> cast(dtype=fp16) -> output

        Output graph:
        input -> cast -> output
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16")
            return x

        assert get_op_types_in_program(prog) == ["cast"]

        _, _, block = apply_pass_and_basic_check(prog, "common::cast_optimization")

        assert len(block.find_ops(op_type="cast")) == 1
        assert block.outputs[0].dtype == types.fp16

    @pytest.mark.parametrize(
        "opset_version",
        [ct.target.iOS14, ct.target.iOS17],
    )
    def test_remove_redundant_cast_stress(self, opset_version):
        """
        Test all possible dtype combination for each iOS version of cast.

        Input graph:
        input(dtype=dtype_a) -> cast(dtype=dtype_b) -> out

        Output graph:
        if dtype_a == dtype_b, the cast op can be eliminated
            input -> out

        if dtype_a != dtype_b, the cast op should be preserved
            input -> cast -> out
        """

        def _test_cast_op_cancellation(dtype_a, dtype_b):
            @mb.program(
                input_specs=[mb.TensorSpec(shape=(1,), dtype=dtype_a)], opset_version=opset_version
            )
            def prog(x):
                x = mb.cast(x=x, dtype=builtin_to_string(dtype_b))
                return x

            assert get_op_types_in_program(prog) == ["cast"]

            _, _, block = apply_pass_and_basic_check(prog, "common::cast_optimization")
            cast_ops = block.find_ops(op_type="cast")
            if dtype_a == dtype_b:
                assert len(cast_ops) == 0
            else:
                assert len(cast_ops) == 1
            assert block.outputs[0].dtype == dtype_b

        opset_version_to_cast_op = {
            ct.target.iOS14: _cast_iOS14,
            ct.target.iOS17: _cast_iOS17,
        }
        cast_op = opset_version_to_cast_op[opset_version]
        for dtype_a in cast_op.type_domains["T"]:
            for dtype_b in cast_op.type_domains["T"]:
                _test_cast_op_cancellation(dtype_a, dtype_b)


class TestCastOptimizationCastFusion:
    """
    Test consecutive cast ops funsion
    """
    def test_cast_ops_fusion_smoke(self):
        """
        Input graph:
        input(fp16) --> cast(dtype="fp32") --> cast(dtype="fp16") --> out

        Output graph:
        input --> identity --> out

        This pattern should be fused, since it doesn't affect the computation precision
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp16)])
        def prog(x):
            x = mb.cast(x=x, dtype="fp32")
            x = mb.cast(x=x, dtype="fp16")
            return x

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prog) == ["identity"]
        assert block.outputs[0].dtype == types.fp16

    def test_cast_ops_fusion_smoke_2(self):
        """
        Input graph:
        input(int8) --> cast(dtype="fp16") --> cast(dtype="fp32") --> out

        Output graph:
        input --> cast(dtype="fp32") --> out

        This pattern should be fused, since it doesn't affect the computation precision, given that the precision is limited by the program int8 input.
        """

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1,), dtype=types.int8)], opset_version=ct.target.iOS17
        )
        def prog(x):
            x = mb.cast(x=x, dtype="fp16")
            x = mb.cast(x=x, dtype="fp32")
            return x

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["cast"]
        assert block.find_ops(op_type="cast")[0].outputs[0].dtype == types.fp32
        assert block.outputs[0].dtype == types.fp32

    def test_cast_ops_fusion_smoke_3(self):
        """
        Input graph:
        input(fp32) --> cast(dtype="fp16") --> cast(dtype="fp16") --> out

        Output graph:
        input --> cast(dtype="fp16") --> out

        Two identical cast ops can be fused into one.
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16")
            x = mb.cast(x=x, dtype="fp16")
            return x

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["cast"]
        assert block.find_ops(op_type="cast")[0].outputs[0].dtype == types.fp16
        assert block.outputs[0].dtype == types.fp16

    def test_cast_ops_fusion_smoke_4(self):
        """
        Input graph:
        input(int8) --> cast(dtype="fp32") --> cast(dtype="int8") --> out

        Output graph:
        input --> identity --> out

        There will be two staged of optimization:
        1. cast(dtype=fp32) + cast(dtype=int8) fused into a single cast(dtype=int8)
        2. cast(dtype=int8) is further removed
        """

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1,), dtype=types.int8)], opset_version=ct.target.iOS17
        )
        def prog(x):
            x = mb.cast(x=x, dtype="fp32")
            x = mb.cast(x=x, dtype="int8")
            return x

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["identity"]
        assert block.outputs[0].dtype == types.int8

    def test_cast_ops_fusion_negative_smoke(self):
        """
        Input graph:
        input(fp32) --> cast(dtype="fp16") --> cast(dtype="fp32") --> out

        Output graph:
        input --> cast --> cast --> out

        This pattern should not be fused, since the precision is lowered.
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16")
            x = mb.cast(x=x, dtype="fp32")
            return x

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["cast", "cast"]
        cast_ops = block.find_ops(op_type="cast")
        assert cast_ops[0].outputs[0].dtype == types.fp16
        assert cast_ops[1].outputs[0].dtype == types.fp32
        assert block.outputs[0].dtype == types.fp32

    def test_cast_ops_fusion_negative_smoke_2(self):
        """
        Input graph:
        input(int32) --> cast(dtype="uint8") --> cast(dtype="int8") --> out

        Output graph:
        input --> cast --> cast --> out

        This pattern should not be fused, since the data range results from uint8 -> int8
        is [0, 127], while a single cast(int8) produces [-128, 127]. The data point between [-128, 0] will have wrong numerical result.
        """

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1,), dtype=types.int32)],
            opset_version=ct.target.iOS17,
        )
        def prog(x):
            x = mb.cast(x=x, dtype="uint8")
            x = mb.cast(x=x, dtype="int8")
            return x

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["cast", "cast"]
        cast_ops = block.find_ops(op_type="cast")
        assert cast_ops[0].outputs[0].dtype == types.uint8
        assert cast_ops[1].outputs[0].dtype == types.int8
        assert block.outputs[0].dtype == types.int8

    @pytest.mark.parametrize(
        "opset_version",
        [ct.target.iOS14, ct.target.iOS17],
    )
    def test_cast_ops_fusion_stress(self, opset_version):
        """
        Test all possible dtype combination for each iOS version of cast.

        Input graph:
        input(dtype=dtype_a) -> cast(dtype=dtype_b) -> cast(dtype=dtype_c) -> out

        Output graph:
        The output graph can have cast ops with number from 0 to 2
        """

        def _test_cast_op_fusion(dtype_a, dtype_b, dtype_c):
            @mb.program(
                input_specs=[mb.TensorSpec(shape=(1,), dtype=dtype_a)], opset_version=opset_version
            )
            def prog(x):
                x = mb.cast(x=x, dtype=builtin_to_string(dtype_b))
                x = mb.cast(x=x, dtype=builtin_to_string(dtype_c))
                return x

            _, _, block = apply_pass_and_basic_check(prog, "common::cast_optimization")
            assert block.outputs[0].dtype == dtype_c
            return
            cast_ops = block.find_ops(op_type="cast")
            if dtype_a == dtype_b:
                assert len(cast_ops) == 0
            else:
                assert len(cast_ops) == 1

        opset_version_to_cast_op = {
            ct.target.iOS14: _cast_iOS14,
            ct.target.iOS17: _cast_iOS17,
        }
        cast_op = opset_version_to_cast_op[opset_version]
        supported_dtypes = cast_op.type_domains["T"]
        for dtype_a in supported_dtypes:
            for dtype_b in supported_dtypes:
                for dtype_c in supported_dtypes:
                    _test_cast_op_fusion(dtype_a, dtype_b, dtype_c)

class TestCastOptimizationComplexPatterns:
    """
    Test cast ops fusion / romoval in some complex graph examples.
    """
    def test_linear_consecutive_cast_ops_cancellation(self):
        """Test the cast optimization pass with more complicated patterns."""

        """
        Input graph:
        input(fp16) -----> cast(dtype="fp32") -----> cast(dtype="fp16") ----> square ---> out

        Output graph:
        input -----> square -----> out
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20), dtype=types.fp16)])
        def prog(x):
            x = mb.cast(x=x, dtype="fp32")
            x = mb.cast(x=x, dtype="fp16")
            x = mb.square(x=x)
            return x

        assert get_op_types_in_program(prog) == ["cast", "cast", "square"]

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["square"]

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (10, 20)},
        )

    def test_linear_consecutive_cast_ops_fusion(self):
        """
        Input graph:
        input(fp32)---->cast(dtype="fp16")---->cast(dtype="bool")--->identity--->out

        Output graph:
        input(fp32)----->cast(dtype="bool")----->identity--->out
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16")
            x = mb.cast(x=x, dtype="bool")
            x = mb.identity(x=x)
            return x

        assert get_op_types_in_program(prog) == ["cast", "cast", "identity"]

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["cast", "identity"]
        assert block.find_ops(op_type="cast")[0].dtype.val == "bool"

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (10, 20)},
        )

    def test_linear_multiple_consecutive_cast_ops(self):
        """
        Input graph:
        input(fp16)-->cast(dtype="fp32")-->cast(dtype="fp32")-->cast(dtype="int32")-->cast(dtype="fp32")-->cast(dtype="fp16")-->square->out

        Output graph:
        input(fp16)-->cast(dtype="int32")-->cast(dtype="fp16")-->square--->out
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20), dtype=types.fp16)])
        def prog(x):
            x = mb.cast(x=x, dtype="fp32")
            x = mb.cast(x=x, dtype="fp32")
            x = mb.cast(x=x, dtype="int32")
            x = mb.cast(x=x, dtype="fp32")
            x = mb.cast(x=x, dtype="fp16")
            x = mb.square(x=x)
            return x

        assert get_op_types_in_program(prog) == [
            "cast",
            "cast",
            "cast",
            "cast",
            "cast",
            "square",
        ]

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prog) == ["cast", "cast", "square"]
        assert block.find_ops(op_type="cast")[0].dtype.val == "int32"
        assert block.find_ops(op_type="cast")[1].dtype.val == "fp16"

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (10, 20)},
        )

    def test_same_consecutive_cancelling_casts_on_all_branches(self):
        """
        Input graph:
                                      |---->cast(dtype="fp16")---->square--->out_1
                                      |
        input(fp16)---->cast(dtype="fp32")---->cast(dtype="fp16")---->relu--->out_2
                                      |
                                      |---->cast(dtype="fp16")---->log--->out_3

        Output graph:

             |---->square--->out_1
             |
        input---->relu--->out_2
             |
             |---->log--->out_3
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20), dtype=types.fp16)])
        def prog(x):
            x = mb.cast(x=x, dtype="fp32")
            x1 = mb.cast(x=x, dtype="fp16")
            x2 = mb.cast(x=x, dtype="fp16")
            x3 = mb.cast(x=x, dtype="fp16")
            x4 = mb.square(x=x1)
            x5 = mb.relu(x=x2)
            x6 = mb.log(x=x3)
            return x4, x5, x6

        assert get_op_types_in_program(prog) == [
            "cast",
            "cast",
            "cast",
            "cast",
            "square",
            "relu",
            "log",
        ]

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["square", "relu", "log"]

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (10, 20),
                block.outputs[1].name: (10, 20),
                block.outputs[2].name: (10, 20),
            },
        )

    def test_consecutive_fusable_casts_on_all_branches(self):
        """
        Input graph:
                                         |---->cast(dtype="int32")---->square--->out_1
                                         |
        input(fp16)---->cast(dtype="fp32")---->cast(dtype="int32")---->abs--->out_2
                                         |
                                         |---->cast(dtype="int32")---->identity--->out_3

        Output graph:

                                          |-->square-->out_1
                                          |
        input(fp16)---->cast(dtype="int32")-->abs-->out_2
                                          |
                                          |-->identity->out_3

        Note that, this result needs the assistant of another pass remove_redundant_ops
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20), dtype=types.fp16)])
        def prog(x):
            x = mb.cast(x=x, dtype="fp32")
            x1 = mb.cast(x=x, dtype="int32")
            x2 = mb.cast(x=x, dtype="int32")
            x3 = mb.cast(x=x, dtype="int32")
            x4 = mb.square(x=x1)
            x5 = mb.abs(x=x2)
            x6 = mb.identity(x=x3)
            return x4, x5, x6

        assert get_op_types_in_program(prog) == [
            "cast",
            "cast",
            "cast",
            "cast",
            "square",
            "abs",
            "identity",
        ]

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prog) == [
            "cast",
            "cast",
            "cast",
            "square",
            "abs",
            "identity",
        ]
        cast_ops = block.find_ops(op_type="cast")
        assert all([v.dtype.val == "int32" for v in cast_ops])

        apply_pass_and_basic_check(prog, "common::remove_redundant_ops")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prog) == [
            "cast",
            "square",
            "abs",
            "identity",
        ]
        assert block.find_ops(op_type="cast")[0].dtype.val == "int32"

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (10, 20),
                block.outputs[1].name: (10, 20),
                block.outputs[2].name: (10, 20),
            },
        )

    def test_mixed_consecutive_casts_on_different_branches(self):
        """
        Input graph:

                                    |---->cast(dtype="fp16")---->square--->out_1
                                    |
                                    |---->cast(dtype="int32")---->square--->out_2
                                    |
        input(fp16)---->cast(dtype="fp32")---->cast(dtype="int32")---->identity--->out_3
                                    |
                                    |---->cast(dtype="int32")---->abs--->out_4
                                    |
                                    |---->cast(dtype="fp16")---->abs--->out_5

        Output graph:

                 |---->square--->out_1
                 |
                 |                      |---->square--->out_2
                 |                      |
        input(fp16)---->cast(dtype="int32")---->identity--->out_3
                 |                      |
                 |                      |---->abs--->out_4
                 |
                 |
                 |---->abs--->out_5

        Note that, this result needs the assistant of another pass remove_redundant_ops
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20), dtype=types.fp16)])
        def prog(x):
            x = mb.cast(x=x, dtype="fp32")
            x1 = mb.cast(x=x, dtype="fp16")
            x2 = mb.cast(x=x, dtype="int32")
            x3 = mb.cast(x=x, dtype="int32")
            x4 = mb.cast(x=x, dtype="int32")
            x5 = mb.cast(x=x, dtype="fp16")
            x6 = mb.square(x=x1)
            x7 = mb.square(x=x2)
            x8 = mb.identity(x=x3)
            x9 = mb.abs(x=x4)
            x10 = mb.abs(x=x5)
            return x6, x7, x8, x9, x10

        assert get_op_types_in_program(prog) == [
            "cast",
            "cast",
            "cast",
            "cast",
            "cast",
            "cast",
            "square",
            "square",
            "identity",
            "abs",
            "abs",
        ]

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prog) == [
            "cast",
            "cast",
            "cast",
            "square",
            "square",
            "identity",
            "abs",
            "abs",
        ]
        cast_ops = block.find_ops(op_type="cast")
        assert all([v.dtype.val == "int32" for v in cast_ops])

        apply_pass_and_basic_check(prog, "common::remove_redundant_ops")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prog) == [
            "cast",
            "square",
            "square",
            "identity",
            "abs",
            "abs",
        ]
        assert block.find_ops(op_type="cast")[0].dtype.val == "int32"
        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (10, 20),
                block.outputs[1].name: (10, 20),
                block.outputs[2].name: (10, 20),
            },
        )

    def test_different_consecutive_casts_config_on_different_branches(self):
        """
        Input graph:

                                        |---->cast(dtype="fp16")---->square--->out_1
                                        |
        input(fp16)---->cast(dtype="fp32")---->cast(dtype="int32")---->exp2--->out_2
                                        |
                                        |---->abs--->out_3


        Output graph:

                |---->square--->out_1
                |
                |
                |
        input(fp16)---->cast(dtype="int32")---->exp2--->out_2
                |
                |
                |
                |
                |---->cast(dtype="fp32")---->abs--->out_3

        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20), dtype=types.fp16)])
        def prog(x):
            x = mb.cast(x=x, dtype="fp32")
            x1 = mb.cast(x=x, dtype="fp16")
            x2 = mb.cast(x=x, dtype="int32")
            x3 = mb.square(x=x1)
            x4 = mb.exp2(x=x2)
            x5 = mb.abs(x=x)
            return x3, x4, x5

        assert get_op_types_in_program(prog) == ["cast", "cast", "cast", "square", "exp2", "abs"]

        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["cast", "cast", "square", "exp2", "abs"]

        # Asserting first cast configuration
        cast_1 = block.find_ops(op_type="cast")[0]
        assert cast_1.dtype.val == "fp32"
        assert len(cast_1.outputs) == 1
        assert len(cast_1.outputs[0].child_ops) == 1
        assert cast_1.outputs[0].child_ops[0].op_type == "abs"

        # Asserting second cast configuration
        cast_2 = block.find_ops(op_type="cast")[1]
        assert cast_2.dtype.val == "int32"
        assert len(cast_2.outputs) == 1
        assert len(cast_2.outputs[0].child_ops) == 1
        assert cast_2.outputs[0].child_ops[0].op_type == "exp2"

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (10, 20),
                block.outputs[1].name: (10, 20),
                block.outputs[2].name: (10, 20),
            },
        )

    def test_two_casts_at_the_end(self):
        """
        Input graph:
        input(dtype="fp16")---->relu----->relu
                                          |
                                  --------|
                                  |
                                  V
                                 cast(dtype="fp32")---->cast(dtype="fp16")
                                                          |
                                    ----------------------|
                                    |
                                    V
                                 cast(dtype="fp32")---->cast(dtype="fp16")---->output(dtype="fp16")

        Output graph:
        input(dtype="fp16")---->relu----->relu---->output(dtype="fp16")
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20), dtype=types.fp16)])
        def prog(x):
            x = mb.relu(x=x)
            x = mb.relu(x=x)
            x = mb.cast(x=x, dtype="fp32")
            x = mb.cast(x=x, dtype="fp16")
            x = mb.cast(x=x, dtype="fp32")
            x = mb.cast(x=x, dtype="fp16", name="original_output_name")
            return x

        assert get_op_types_in_program(prog) == ["relu", "relu", "cast", "cast", "cast", "cast"]
        apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, prev_block, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prog) == ["relu", "relu"]
        assert prev_block.outputs[0].name == "original_output_name"
        assert block.outputs[0].name == "original_output_name"
        assert block.outputs[0].dtype == types.fp16

    def test_mixed_consecutive_casts_on_different_branches_complex(self):
        """
        Input graph:

                                    |->cast(dtype="fp16")->cast(dtype="fp16")->out_1
                                    |
        input(fp16)---->cast(dtype="fp32")->cast(dtype="uint8")->cast(dtype="int8")->out_2
                                    |
                                    |->cast(dtype="int32")->out_3
                                    |
                                    |->cast(dtype="int32")->cast(dtype="float32")->out_4

        Output graph:

                    |-->out_1
                    |
        input(fp16)-->cast(dtype="uint8")-->cast(dtype="int8")-->out_2
                    |
                    .-->cast(dtype="int32")-->out_3
                                           |
                                           .-->cast(dtype="float32")-->out_4

        Note that, this result needs the assistant of another pass remove_redundant_ops
        """

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp16)], opset_version=ct.target.iOS17
        )
        def prog(x):
            x = mb.cast(x=x, dtype="fp32")
            x1 = mb.cast(x=x, dtype="fp16")
            x1 = mb.cast(x=x1, dtype="fp16")
            x2 = mb.cast(x=x, dtype="uint8")
            x2 = mb.cast(x=x2, dtype="int8")
            x3 = mb.cast(x=x, dtype="int32")
            x4 = mb.cast(x=x, dtype="int32")
            x4 = mb.cast(x=x4, dtype="fp32")
            return x2, x3, x4

        assert get_op_types_in_program(prog) == ["cast"] * 8
        apply_pass_and_basic_check(prog, "common::cast_optimization")
        apply_pass_and_basic_check(prog, "common::remove_redundant_ops")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prog) == ["cast"] * 4

        expected_cast_dtype = ["uint8", "int8", "int32", "fp32"]
        cast_ops = block.find_ops(op_type="cast")
        assert [v.dtype.val for v in cast_ops] == expected_cast_dtype


class TestCastOptimizationAcrossBlocks:
    """
    Test the cast optimization for cast ops at the boundary of inner and outer block.
    """
    def test_cast_ops_fuse_across_block_smoke_1(self):
        """
        Input graph:
        main[CoreML3](%x: (1,int32)(Tensor)) {
        main[CoreML3](%x: (1,int32)(Tensor)) {
          block0() {
            %cast_0: (1,fp32)(Tensor) = cast(x=%x, dtype="fp32", name="cast_0")
            %cond_0: (1,fp32)(Tensor) = cond(pred=True, name="cond_0")
              cond_0_true() {
                %cast_1: (1,fp32)(Tensor) = cast(x=%cast_0, dtype="fp32", name="cast_1")
              } -> (%cast_1)
              cond_0_false() {
                %cast_2: (1,fp32)(Tensor) = cast(x=%cast_0, dtype="fp32", name="cast_2")
              } -> (%cast_2)
          } -> (%cond_0)
        }

        Output graph:
        main[CoreML3](%x: (1,int32)(Tensor)) {
          block0() {
            %cast_0: (1,fp32)(Tensor) = cast(x=%x, dtype="fp32", name="cast_0")
            %cond_0: (1,fp32)(Tensor) = cond(pred=True, name="cond_0")
              cond_0_true() {
              } -> (%cast_0)
              cond_0_false() {
              } -> (%const_0)
          } -> (%cond_0)
        }
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.int32)])
        def prog(x):
            x = mb.cast(x=x, dtype="fp32")
            def _true_fn():
                return mb.cast(x=x, dtype="fp32")

            def _false_fn():
                return mb.cast(x=x, dtype="fp32")

            return mb.cond(pred=True, _true_fn=_true_fn, _false_fn=_false_fn)

        _, _, block = apply_pass_and_basic_check(prog, "common::cast_optimization")
        assert get_op_types_in_program(prog) == ["cast", "cond"]

        cast_op = block.find_ops(op_type="cast")[0]
        assert cast_op.dtype.val == "fp32"

        cond_op = block.find_ops(op_type="cond")[0]
        true_block, false_block = cond_op.blocks
        assert get_op_types_in_block(true_block) == []
        assert get_op_types_in_block(false_block) == []
        assert true_block.outputs[0] == cast_op.outputs[0]
        assert false_block.outputs[0] == cast_op.outputs[0]

    def test_cast_ops_fuse_across_block_smoke_2(self):
        """
        Input graph:
        main[CoreML3](%x: (1,fp32)(Tensor)) {
          block0() {
            %cast_0: (1,fp32)(Tensor) = cast(x=%x, dtype="fp32", name="cast_0")
            %cond_0: (1,fp32)(Tensor) = cond(pred=True, name="cond_0")
              cond_0_true() {
                %cast_1: (1,fp32)(Tensor) = cast(x=%cast_0, dtype="fp32", name="cast_1")
              } -> (%cast_1)
              cond_0_false() {
                %cast_2: (1,fp32)(Tensor) = cast(x=%cast_0, dtype="fp32", name="cast_2")
              } -> (%cast_2)
          } -> (%cond_0)
        }

        Output graph:
        main[CoreML3](%x: (1,fp32)(Tensor)) {
          block0() {
            %cond_0: (1,fp32)(Tensor) = cond(pred=True, name="cond_0")
              cond_0_true() {
              } -> (%x)
              cond_0_false() {
              } -> (%x)
          } -> (%cond_0)
        }
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)])
        def prog(x):
            x = mb.cast(x=x, dtype="fp32")

            def _true_fn():
                return mb.cast(x=x, dtype="fp32")

            def _false_fn():
                return mb.cast(x=x, dtype="fp32")

            return mb.cond(pred=True, _true_fn=_true_fn, _false_fn=_false_fn)

        _, _, block = apply_pass_and_basic_check(prog, "common::cast_optimization")
        assert get_op_types_in_program(prog) == ["cond"]

        cond_op = block.find_ops(op_type="cond")[0]
        true_block, false_block = cond_op.blocks
        assert get_op_types_in_block(true_block) == []
        assert get_op_types_in_block(false_block) == []
        assert true_block.outputs[0] == block.inputs["x"]
        assert false_block.outputs[0] == block.inputs["x"]

    def test_cast_ops_fuse_across_block_smoke_3(self):
        """
        Input graph:
        main[CoreML7](%x: (1,int32)(Tensor)) {
          block0() {
            %cast_0: (1,fp32)(Tensor) = cast(x=%x, dtype="fp32", name="cast_0")
            %cond_0: (1,uint8)(Tensor) = cond(pred=True, name="cond_0")
              cond_0_true() {
                %cast_1: (1,int32)(Tensor) = cast(x=%cast_0, dtype="int32", name="cast_1")
                %cast_2: (1,uint8)(Tensor) = cast(x=%cast_1, dtype="uint8", name="cast_2")
                %cast_3: (1,fp32)(Tensor) = cast(x=%cast_2, dtype="fp32", name="cast_3")
                %cast_4: (1,uint8)(Tensor) = cast(x=%cast_3, dtype="uint8", name="cast_4")
              } -> (%cast_4)
              cond_0_false() {
                %cast_5: (1,int8)(Tensor) = cast(x=%cast_0, dtype="int8", name="cast_5")
                %cast_6: (1,bool)(Tensor) = cast(x=%cast_5, dtype="bool", name="cast_6")
                %cast_7: (1,uint8)(Tensor) = cast(x=%cast_6, dtype="uint8", name="cast_7")
              } -> (%cast_7)
          } -> (%cond_0)
        }

        Output graph:
        main[CoreML7](%x: (1,int32)(Tensor)) {
          block0() {
            %cond_0: (1,uint8)(Tensor) = cond(pred=True, name="cond_0")
              cond_0_true() {
                %x_to_uint8: (1,uint8)(Tensor) = cast(x=%x, dtype="uint8", name="x_to_uint8")
              } -> (%x_to_uint8)
              cond_0_false() {
                %x_to_bool: (1,bool)(Tensor) = cast(x=%x, dtype="bool", name="x_to_bool")
                %cast_7: (1,uint8)(Tensor) = cast(x=%x_to_bool, dtype="uint8", name="cast_7")
              } -> (%cast_7)
          } -> (%cond_0)
        }

        This is a more complex example:
        First, in the true branch, 4 ``cast`` ops are optimized into a single ``cast(dtype="uint8")``. In the false branch, 3 ``cast`` ops are optimized to ``cast(dtype="bool")->cast(dtype="uint8")``
        Second, the first ``cast`` op in each inner block is fused with the outer ``cast_0`` op, resulting in the above output graph.
        """

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1,), dtype=types.int32)],
            opset_version=ct.target.iOS17,
        )
        def prog(x):
            x = mb.cast(x=x, dtype="fp32")

            def _true_fn():
                x1 = mb.cast(x=x, dtype="int32")
                x1 = mb.cast(x=x1, dtype="uint8")
                x1 = mb.cast(x=x1, dtype="fp32")
                return mb.cast(x=x1, dtype="uint8")

            def _false_fn():
                x2 = mb.cast(x=x, dtype="int8")
                x2 = mb.cast(x=x2, dtype="bool")
                return mb.cast(x=x2, dtype="uint8")

            return mb.cond(pred=True, _true_fn=_true_fn, _false_fn=_false_fn)

        _, _, block = apply_pass_and_basic_check(prog, "common::cast_optimization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["cond"]

        cond_op = block.find_ops(op_type="cond")[0]
        true_block, false_block = cond_op.blocks
        assert get_op_types_in_block(true_block) == ["cast"]
        assert get_op_types_in_block(false_block) == ["cast"] * 2

        expected_true_branch_types = ["uint8"]
        expected_false_branch_types = ["bool", "uint8"]

        assert expected_true_branch_types == [
            v.dtype.val for v in true_block.find_ops(op_type="cast")
        ]
        assert expected_false_branch_types == [
            v.dtype.val for v in false_block.find_ops(op_type="cast")
        ]


class TestConv1dCompositionPasses:
    @pytest.mark.parametrize(
        "backend, has_strides, pad_type, has_pad, has_dilations, has_bias",
        itertools.product(
            backends,
            (True, False),
            ("valid", "custom", "same"),
            (True, False),
            (True, False),
            (True, False),
        ),
    )
    def test_conv1d_composition(
        self, backend, has_strides, pad_type, has_pad, has_dilations, has_bias
    ):
        """
        Input graph:
        input -> expand_dims -> conv2d -> squeeze -> out

        Output graph:
        input -> conv1d -> out
        """
        N, L = 2, 8
        C_in, C_out = 3, 4
        K = 3

        conv_kwargs = {"weight": np.random.rand(C_out, C_in, 1, K), "pad_type": pad_type}
        if has_strides:
            conv_kwargs["strides"] = (2, 2)
        if has_pad:
            # The pad is specially designed to make sure the output of conv has dim_size=1 at axis 1.
            conv_kwargs["pad"] = (0, 0, 1, 1) if pad_type == "custom" else (1, 1, 1, 1)
        if has_dilations:
            conv_kwargs["dilations"] = (2, 2)
        if has_bias:
            conv_kwargs["bias"] = np.random.rand(C_out)

        @mb.program(input_specs=[mb.TensorSpec(shape=(N, C_in, L))])
        def prog(x):
            y_expand = mb.expand_dims(x=x, axes=(2,))
            y_conv = mb.conv(x=y_expand, **conv_kwargs)
            y_squeeze = mb.squeeze(x=y_conv, axes=(2,))
            return y_squeeze

        assert get_op_types_in_program(prog) == ["expand_dims", "conv", "squeeze"]

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::compose_conv1d")
        assert get_op_types_in_program(prog) == ["squeeze", "conv"]

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::const_elimination")
        assert get_op_types_in_program(prog) == ["conv"]

        # infer output shape
        strides = conv_kwargs["strides"] if has_strides else (1, 1)
        pad = conv_kwargs["pad"] if has_pad else (0, 0, 0, 0)
        dilations = conv_kwargs["dilations"] if has_dilations else (1, 1)
        L_out = None
        if pad_type == "valid":
            L_out = (L - dilations[-1] * (K - 1) - 1) // strides[-1] + 1
        elif pad_type == "custom":
            L_out = (L + pad[-2] + pad[-1] - dilations[-1] * (K - 1) - 1) // strides[-1] + 1
        elif pad_type == "same":
            L_out = np.ceil(L / strides[-1])
        else:
            raise Exception("unsupported pad type")
        output_shape = (N, C_out, L_out)

        assert_model_is_valid(
            prog,
            {"x": (N, C_in, L)},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=backend,
        )

    @pytest.mark.parametrize("backend", backends)
    def test_conv1d_composotion_dynamic_weight(self, backend):
        """
        Input graph:
        input -> expand_dims -> conv2d -> squeeze -> out

        Output graph:
        input -> conv1d -> out
        """
        N, L = 2, 9
        C_in, C_out = 4, 3
        K = 4

        strides = (1, 2)
        pad = (0, 0, 1, 1)
        # MIL convolution with dynamic weights does not support dilations != 1
        # see coremltools/coremltools/converters/mil/mil/ops/defs/iOS15/conv.py
        dilations = (1, 1)

        # infer L_out with pad_type fixed to custom
        L_out = (L + pad[-2] + pad[-1] - dilations[-1] * (K - 1) - 1) // strides[-1] + 1

        conv_kwargs = {
            "strides": strides,
            "pad_type": "custom",
            "pad": pad,
            "dilations": dilations,
        }

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(N, C_in, L)),
                mb.TensorSpec(shape=(C_out, C_in, 1, K)),
            ]
        )
        def prog(x, weight):
            y_expand = mb.expand_dims(x=x, axes=(-2,))
            y_conv = mb.conv(x=y_expand, weight=weight, **conv_kwargs)
            y_squeeze = mb.squeeze(x=y_conv, axes=(-2,))
            return y_squeeze

        assert get_op_types_in_program(prog) == ["expand_dims", "conv", "squeeze"]

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::compose_conv1d")
        assert get_op_types_in_program(prog) == ["squeeze", "conv"]

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::const_elimination")
        assert get_op_types_in_program(prog) == ["squeeze", "conv"]

        output_shape = (N, C_out, L_out)
        assert_model_is_valid(
            prog,
            {"x": (N, C_in, L), "weight": (C_out, C_in, 1, K)},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, has_bias, bias_op_type",
        itertools.product(
            backends,
            (True, False),
            ("add", "sub"),
        ),
    )
    def test_conv1d_bias_fusion(self, backend, has_bias, bias_op_type):
        """
        After recomposing the shattered conv1d, conv1d optimization passes should work

        Input graph:
        input -> expand_dims -> conv2d -> squeeze -> add/sub a constant -> out

        Output graph:
        input -> conv1d -> out
        """
        N, L = 2, 8
        C_in, C_out = 3, 5
        K = 3

        strides = (1, 2)
        pad = (0, 0, 0, 1)
        dilations = (1, 2)

        # infer L_out with pad_type fixed to custom
        L_out = (L + pad[-2] + pad[-1] - dilations[-1] * (K - 1) - 1) // strides[-1] + 1

        conv_kwargs = {
            "weight": np.random.rand(C_out, C_in, 1, K),
            "strides": strides,
            "pad_type": "custom",
            "pad": pad,
            "dilations": dilations,
        }
        if has_bias:
            conv_kwargs["bias"] = np.random.rand(C_out)

        bias2 = np.random.rand(C_out, 1)

        @mb.program(input_specs=[mb.TensorSpec(shape=(N, C_in, L))])
        def prog(x):
            y_expand = mb.expand_dims(x=x, axes=(-2,))
            y_conv = mb.conv(x=y_expand, **conv_kwargs)
            y_squeeze = mb.squeeze(x=y_conv, axes=(-2,))
            y_bias2 = (
                mb.add(x=y_squeeze, y=bias2)
                if bias_op_type == "add"
                else mb.sub(x=y_squeeze, y=bias2)
            )
            return y_bias2

        assert get_op_types_in_program(prog) == ["expand_dims", "conv", "squeeze", bias_op_type]

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::compose_conv1d")
        assert get_op_types_in_program(prog) == ["squeeze", "conv", bias_op_type]

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_conv_bias")
        assert get_op_types_in_program(prog) == ["squeeze", "conv"]

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::const_elimination")
        assert get_op_types_in_program(prog) == ["conv"]

        output_shape = (N, C_out, L_out)
        assert_model_is_valid(
            prog,
            {"x": (N, C_in, L)},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=backend,
        )


class TestConv1dChannellastCompositionPasses:
    @pytest.mark.parametrize(
        "backend, has_strides, pad_type, has_pad, has_dilations, has_bias",
        itertools.product(
            backends,
            (True, False),
            ("valid", "custom", "same"),
            (True, False),
            (True, False),
            (True, False),
        ),
    )
    def test_conv1d_channellast_composition(
        self, backend, has_strides, pad_type, has_pad, has_dilations, has_bias
    ):
        """
        Input graph:
        input -> expand_dims -> transpose -> conv2d -> transpose -> squeeze -> out

        Output graph:
        input -> transpose -> conv1d -> transpose -> out
        """
        N, L = 2, 8
        C_in, C_out = 5, 3
        K = 3

        conv_kwargs = {
            "weight": np.random.rand(C_out, C_in, 1, K),
            "pad_type": pad_type,
        }
        if has_strides:
            conv_kwargs["strides"] = (2, 2)
        if has_pad:
            # The pad is specially designed to make sure the output of conv has dim_size=1 at axis 1.
            conv_kwargs["pad"] = (0, 0, 1, 1) if pad_type == "custom" else (1, 1, 1, 1)
        if has_dilations:
            conv_kwargs["dilations"] = (2, 2)
        if has_bias:
            conv_kwargs["bias"] = np.random.rand(C_out)

        @mb.program(input_specs=[mb.TensorSpec(shape=(N, L, C_in))])
        def prog(x):
            y_expand = mb.expand_dims(x=x, axes=(1,))
            y_transpose1 = mb.transpose(x=y_expand, perm=(0, 3, 1, 2))
            y_conv = mb.conv(x=y_transpose1, **conv_kwargs)
            y_transpose2 = mb.transpose(x=y_conv, perm=(0, 2, 3, 1))
            y_squeeze = mb.squeeze(x=y_transpose2, axes=(1,))
            return y_squeeze

        assert get_op_types_in_program(prog) == [
            "expand_dims",
            "transpose",
            "conv",
            "transpose",
            "squeeze",
        ]

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::compose_conv1d")
        assert get_op_types_in_program(prog) == ["transpose", "squeeze", "conv", "transpose"]

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::const_elimination")
        assert get_op_types_in_program(prog) == ["transpose", "conv", "transpose"]

        # infer output shape
        strides = conv_kwargs["strides"] if has_strides else (1, 1)
        pad = conv_kwargs["pad"] if has_pad else (0, 0, 0, 0)
        dilations = conv_kwargs["dilations"] if has_dilations else (1, 1)
        L_out = None
        if pad_type == "valid":
            L_out = (L - dilations[-1] * (K - 1) - 1) // strides[-1] + 1
        elif pad_type == "custom":
            L_out = (L + pad[-2] + pad[-1] - dilations[-1] * (K - 1) - 1) // strides[-1] + 1
        elif pad_type == "same":
            L_out = np.ceil(L / strides[-1])
        else:
            raise Exception("unsupported pad type")
        output_shape = (N, L_out, C_out)

        assert_model_is_valid(
            prog,
            {"x": (N, L, C_in)},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=backend,
        )

    @pytest.mark.parametrize("backend", backends)
    def test_conv1d_channellast_composotion_dynamic_weight(self, backend):
        """
        Input graph:
        input -> expand_dims -> transpose -> conv2d -> transpose -> squeeze -> out

        Output graph:
        input -> transpose -> conv1d -> transpose -> out
        """
        N, L = 2, 9
        C_in, C_out = 4, 5
        K = 4

        strides = (1, 2)
        # The pad is specially designed to make sure the output of conv has dim_size=1 at axis 1.
        pad = (0, 0, 0, 1)
        # MIL convolution with dynamic weights does not support dilations != 1
        # see coremltools/coremltools/converters/mil/mil/ops/defs/iOS15/conv.py
        dilations = (1, 1)

        # infer L_out with pad_type fixed to custom
        L_out = (L + pad[-2] + pad[-1] - dilations[-1] * (K - 1) - 1) // strides[-1] + 1

        conv_kwargs = {
            "strides": strides,
            "pad_type": "custom",
            "pad": pad,
            "dilations": dilations,
        }

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(N, L, C_in)),
                mb.TensorSpec(shape=(C_out, C_in, 1, K)),
            ]
        )
        def prog(x, weight):
            y_expand = mb.expand_dims(x=x, axes=(1,))
            y_transpose1 = mb.transpose(x=y_expand, perm=(0, 3, 1, 2))
            y_conv = mb.conv(x=y_transpose1, weight=weight, **conv_kwargs)
            y_transpose2 = mb.transpose(x=y_conv, perm=(0, 2, 3, 1))
            y_squeeze = mb.squeeze(x=y_transpose2, axes=(1,))
            return y_squeeze

        assert get_op_types_in_program(prog) == [
            "expand_dims",
            "transpose",
            "conv",
            "transpose",
            "squeeze",
        ]

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::compose_conv1d")
        assert get_op_types_in_program(prog) == ["transpose", "squeeze", "conv", "transpose"]

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::const_elimination")
        assert get_op_types_in_program(prog) == ["transpose", "squeeze", "conv", "transpose"]

        output_shape = (N, L_out, C_out)
        assert_model_is_valid(
            prog,
            {"x": (N, L, C_in), "weight": (C_out, C_in, 1, K)},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, has_bias, bias_op_type",
        itertools.product(
            backends,
            (True, False),
            ("add", "sub"),
        ),
    )
    def test_conv1d_channellast_bias_fusion(self, backend, has_bias, bias_op_type):
        """
        After recomposing the shattered conv1d, conv1d optimization passes should work

        Input graph:
        input -> expand_dims -> transpose -> conv2d -> transpose -> squeeze -> add/sub a constant -> out

        Output graph:
        input -> transpose -> conv1d -> transpose -> out
        """
        N, L = 2, 8
        C_in, C_out = 5, 4
        K = 4

        strides = (1, 2)
        pad = (0, 0, 1, 0)
        dilations = (1, 2)

        # infer L_out with pad_type fixed to custom
        L_out = (L + pad[-2] + pad[-1] - dilations[-1] * (K - 1) - 1) // strides[-1] + 1

        conv_kwargs = {
            "weight": np.random.rand(C_out, C_in, 1, K),
            "strides": strides,
            "pad_type": "custom",
            "pad": pad,
            "dilations": dilations,
        }
        if has_bias:
            conv_kwargs["bias"] = np.random.rand(C_out)

        bias2 = np.random.rand(C_out)

        @mb.program(input_specs=[mb.TensorSpec(shape=(N, L, C_in))])
        def prog(x):
            y_expand = mb.expand_dims(x=x, axes=(-3,))
            y_transpose1 = mb.transpose(x=y_expand, perm=(0, 3, 1, 2))
            y_conv = mb.conv(x=y_transpose1, **conv_kwargs)
            y_transpose2 = mb.transpose(x=y_conv, perm=(0, 2, 3, 1))
            y_squeeze = mb.squeeze(x=y_transpose2, axes=(-3,))
            y_bias2 = (
                mb.add(x=y_squeeze, y=bias2)
                if bias_op_type == "add"
                else mb.sub(x=y_squeeze, y=bias2)
            )
            return y_bias2

        assert get_op_types_in_program(prog) == [
            "expand_dims",
            "transpose",
            "conv",
            "transpose",
            "squeeze",
            bias_op_type,
        ]

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::compose_conv1d")
        assert get_op_types_in_program(prog) == [
            "transpose",
            "squeeze",
            "conv",
            "transpose",
            bias_op_type,
        ]

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_conv_bias")
        assert get_op_types_in_program(prog) == ["transpose", "squeeze", "conv", "transpose"]

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::const_elimination")
        assert get_op_types_in_program(prog) == ["transpose", "conv", "transpose"]

        output_shape = (N, L_out, C_out)
        assert_model_is_valid(
            prog,
            {"x": (N, L, C_in)},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=backend,
        )


class TestConvBatchNormFusion:
    @staticmethod
    def _apply_weight_transform(inputs, is_deconv, dtype=np.float32):
        """
        Utility function to test the weight transform function in conv batch_norm fusion pass.
        """
        Cin, _, groups = 10, 20, 10
        input_shape = (1, Cin, 2, 2)

        @mb.program(
            input_specs=[mb.TensorSpec(shape=input_shape, dtype=numpy_type_to_builtin_type(dtype))]
        )
        def prog(x):

            if is_deconv:
                x = mb.conv_transpose(
                    x=x,
                    weight=inputs["conv_weight"],
                    bias=inputs["conv_bias"],
                    groups=groups,
                )
            else:
                x = mb.conv(
                    x=x,
                    weight=inputs["conv_weight"],
                    bias=inputs["conv_bias"],
                    groups=groups,
                )

            x = mb.batch_norm(
                x=x,
                mean=inputs["mean"],
                variance=inputs["variance"],
                gamma=inputs["gamma"],
                beta=inputs["beta"],
                epsilon=inputs["epsilon"],
            )
            return x

        apply_pass_and_basic_check(prog, "common::fuse_conv_batchnorm")

        # get the updated weight from the prog
        conv_op = []
        for op in prog["main"].operations:
            if op.op_type == "const":
                continue
            conv_op.append(op)
        assert len(conv_op) == 1, "should only have one conv / conv_transpose layer."

        return conv_op[0].weight.val, conv_op[0].bias.val

    @pytest.mark.parametrize(
        "conv_type",
        ["conv", "conv_transpose"],
    )
    def test_weight_transform_conv_identity(self, conv_type):
        """
        Test the weight transform function with an identity batchnorm layer.
        """
        # parameters for conv
        is_deconv = conv_type == "conv_transpose"
        conv_weight = np.arange(20).astype(np.float32)
        conv_weight = (
            np.reshape(conv_weight, (10, 2, 1, 1))
            if is_deconv
            else np.reshape(conv_weight, (20, 1, 1, 1))
        )
        conv_bias = np.arange(20).astype(np.float32)

        # parameters for batch_norm
        gamma = np.ones(20).astype(np.float32)
        beta = np.zeros(20).astype(np.float32)
        mean = np.zeros(20).astype(np.float32)
        variance = np.ones(20).astype(np.float32)
        epsilon = 0.0

        inputs = {
            "conv_weight": conv_weight,
            "conv_bias": conv_bias,
            "gamma": gamma,
            "beta": beta,
            "mean": mean,
            "variance": variance,
            "epsilon": epsilon,
        }

        new_conv_weight, new_conv_bias = self._apply_weight_transform(inputs, is_deconv)

        np.testing.assert_equal(new_conv_weight, conv_weight)
        np.testing.assert_equal(new_conv_bias, conv_bias)

    @pytest.mark.parametrize(
        "conv_type, dtype",
        itertools.product(
            ["conv", "conv_transpose"],
            [np.float16, np.float32],
        ),
    )
    def test_weight_transform_conv_type(self, conv_type, dtype):
        """
        The weight transform function should return an updated conv weight with correct data type
        """
        # parameters for conv
        is_deconv = conv_type == "conv_transpose"
        conv_weight = np.arange(20).astype(dtype)
        conv_weight = (
            np.reshape(conv_weight, (10, 2, 1, 1))
            if is_deconv
            else np.reshape(conv_weight, (20, 1, 1, 1))
        )
        conv_bias = np.arange(20).astype(dtype)

        # parameters for batch_norm
        gamma = np.ones(20).astype(dtype)
        beta = np.zeros(20).astype(dtype)
        mean = np.zeros(20).astype(dtype)
        variance = np.ones(20).astype(dtype)
        epsilon = dtype(0.1)

        inputs = {
            "conv_weight": conv_weight,
            "conv_bias": conv_bias,
            "gamma": gamma,
            "beta": beta,
            "mean": mean,
            "variance": variance,
            "epsilon": epsilon,
        }

        new_conv_weight, _ = self._apply_weight_transform(inputs, is_deconv, dtype)

        assert (
            new_conv_weight.dtype == dtype
        ), "the weight transform function should retain the weight's original dtype."

    @pytest.mark.parametrize(
        "rank, groups, has_bias, backend",
        itertools.product([3, 4, 5], [1, 2, 10], [False, True], backends),
    )
    def test_conv(self, rank, groups, has_bias, backend):
        """
        Input graph:
        input -----> conv -----> batch_norm ---> out

        Output graph:
        input -----> conv ----> out

        Different `rank` represents different conv dimensions: rank=3 for Conv1d, rank=4 for Conv2d, rank=5 for Conv3d.
        """
        Cin, Cout = 10, 30
        rank_to_input_shape = {3: (2, Cin, 20), 4: (2, Cin, 20, 24), 5: (2, Cin, 20, 24, 24)}
        rank_to_conv_weight_shape = {
            3: (Cout, Cin // groups, 2),
            4: (Cout, Cin // groups, 2, 3),
            5: (Cout, Cin // groups, 2, 3, 3),
        }
        rank_to_output_shape = {3: (2, Cout, 19), 4: (2, Cout, 19, 22), 5: (2, Cout, 19, 22, 22)}

        input_shape = rank_to_input_shape[rank]

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            # conv layer
            conv_weight = np.random.rand(*rank_to_conv_weight_shape[rank])
            conv_bias = np.random.rand(Cout) if has_bias else None
            x = mb.conv(
                x=x,
                weight=conv_weight,
                bias=conv_bias,
                groups=groups,
            )

            # batch_norm layer
            gamma = np.random.rand(Cout)
            beta = np.random.rand(Cout)
            mean = np.random.rand(Cout)
            variance = np.random.rand(Cout)
            epsilon = 1e-2
            x = mb.batch_norm(
                x=x,
                mean=mean,
                variance=variance,
                gamma=gamma,
                beta=beta,
                epsilon=epsilon,
            )
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_conv_batchnorm"
        )

        assert get_op_types_in_program(prev_prog) == ["conv", "batch_norm"]
        assert get_op_types_in_program(prog) == ["conv"]

        # validate graph pass
        output_shape = rank_to_output_shape[rank]
        assert_model_is_valid(
            prog,
            {"x": input_shape},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=backend,
        )

    @pytest.mark.parametrize(
        "rank, groups, has_bias, backend",
        itertools.product([3, 4, 5], [1, 2, 10], [False, True], backends),
    )
    def test_conv_transpose(self, rank, groups, has_bias, backend):
        """
        Input graph:
        input -----> conv_transpose -----> batch_norm ---> out

        Output graph:
        input -----> conv_transpose ----> out
        """
        Cin, Cout = 10, 30
        rank_to_input_shape = {3: (2, Cin, 20), 4: (2, Cin, 20, 24), 5: (2, Cin, 20, 24, 24)}
        rank_to_conv_weight_shape = {
            3: (Cin, Cout // groups, 2),
            4: (Cin, Cout // groups, 2, 3),
            5: (Cin, Cout // groups, 2, 3, 3),
        }
        rank_to_output_shape = {3: (2, Cout, 21), 4: (2, Cout, 21, 26), 5: (2, Cout, 21, 26, 26)}

        input_shape = rank_to_input_shape[rank]

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            # conv layer
            conv_weight = np.random.rand(*rank_to_conv_weight_shape[rank])
            conv_bias = np.random.rand(Cout) if has_bias else None
            x = mb.conv_transpose(
                x=x,
                weight=conv_weight,
                bias=conv_bias,
                groups=groups,
            )

            # batch_norm layer
            gamma = np.random.rand(Cout)
            beta = np.random.rand(Cout)
            mean = np.random.rand(Cout)
            variance = np.random.rand(Cout)

            epsilon = 1e-5
            x = mb.batch_norm(
                x=x,
                mean=mean,
                variance=variance,
                gamma=gamma,
                beta=beta,
                epsilon=epsilon,
            )
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_conv_batchnorm"
        )

        assert get_op_types_in_program(prev_prog) == ["conv_transpose", "batch_norm"]
        assert get_op_types_in_program(prog) == ["conv_transpose"]

        # validate graph pass
        output_shape = rank_to_output_shape[rank]
        assert_model_is_valid(
            prog,
            {"x": input_shape},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=backend,
        )


class TestConvBiasFusion:
    @staticmethod
    def get_conv(x, name, Cin=3, Cout=3):
        conv_weight = np.random.rand(Cout, Cin, 2, 2)
        x = mb.conv(x=x, weight=conv_weight, name=name)
        return x

    @staticmethod
    def get_linear(x, name, linear_op, C=3):
        bias = np.arange(C).astype(np.float32)
        bias = np.reshape(bias, (C, 1, 1))
        x = getattr(mb, linear_op)(x=x, y=bias, name=name)
        return x

    @pytest.mark.parametrize(
        "rank, linear_op",
        itertools.product([4], ["add", "sub"]),
    )
    def test_conv(self, rank, linear_op):
        """
        Input graph:
        input -----> conv -----> add/sub ---> out

        Output graph:
        If the linear op is trainable, the program is not modified.
        Otherwise, conv and the linear op will be fused:
        input -----> conv ----> out
        """
        Cin, Cout = 3, 3
        input_shape = (2, Cin, 100, 100)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            x = self.get_conv(x, "conv")
            x = self.get_linear(x, "linear", linear_op)
            return x

        apply_pass_and_basic_check(prog, "common::fuse_conv_bias")
        apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prog) == ["conv"]

    def test_scope_back_propagation(self):
        Cin, Cout = 3, 3
        input_shape = (2, Cin, 100, 100)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            with mb.scope(ScopeInfo(source=ScopeSource.COREMLTOOLS_GRAPH_PASS, data=["pass_1"])):
                x = self.get_conv(x, "conv1")

            with mb.scope(ScopeInfo(source=ScopeSource.COREMLTOOLS_GRAPH_PASS, data=["pass_2"])):
                x = self.get_linear(x, "linear1", "add")

            with mb.scope(ScopeInfo(source=ScopeSource.COREMLTOOLS_GRAPH_PASS, data=["pass_3"])):
                x = self.get_conv(x, "conv2")

            with mb.scope(ScopeInfo(source=ScopeSource.COREMLTOOLS_GRAPH_PASS, data=["pass_4"])):
                x = self.get_linear(x, "linear2", "add")
            return x

        apply_pass_and_basic_check(prog, "common::fuse_conv_bias")
        assert get_op_types_in_program(prog) == ["conv", "conv"]

        conv_ops = prog.functions["main"].find_ops(op_type="conv")
        assert conv_ops[0].scopes == {
            ScopeSource.COREMLTOOLS_GRAPH_PASS: ["pass_2", "fuse_conv_bias"]
        }
        assert conv_ops[1].scopes == {
            ScopeSource.COREMLTOOLS_GRAPH_PASS: ["pass_4", "fuse_conv_bias"]
        }

    """
    Input graph:
                                    Const
                                      |
                                      V
    input -----> convolution -----> add/sub  ----> relu ---> out

    Output graph:
    input -----> convolution -----> relu ----> out
    """

    @pytest.mark.parametrize(
        "conv_dim, \
        flip_add_input_order, \
        add_batch_dim_to_const, \
        use_sub_instead, \
        prebuilt_bias, \
        scalar_elementwise, \
        use_conv_transpose",
        itertools.product(
            [2, 3],  # 1D conv conversion broken even without the pass: rdar://problem/62960720
            [True, False],  # flip_add_input_order
            [True, False],  # add_batch_dim_to_const
            [True, False],  # use_sub_instead
            [True, False],  # prebuilt_bias
            [True, False],  # scalar_elementwise
            [True, False],  # use_conv_transpose
        ),
    )
    def test_fuse_conv_bias(
        self,
        conv_dim,
        flip_add_input_order,
        add_batch_dim_to_const,
        use_sub_instead,
        prebuilt_bias,
        scalar_elementwise,
        use_conv_transpose,
    ):

        if flip_add_input_order and use_sub_instead:
            return

        if use_conv_transpose and conv_dim != 2:
            return

        input_shape = None
        W = None
        Cout = 8
        Cin = 3
        D = 10
        const = np.random.rand(Cout) if add_batch_dim_to_const else np.random.rand(1, Cout)
        const = np.expand_dims(const, axis=-1)

        if conv_dim == 1:
            input_shape = (1, Cin, D)
            W = np.random.rand(Cout, Cin, 1)
        elif conv_dim == 2:
            input_shape = (1, Cin, D, D)
            W = np.random.rand(Cout, Cin, 1, 1)
            const = np.expand_dims(const, axis=-1)
        elif conv_dim == 3:
            input_shape = (1, Cin, D, D, D)
            W = np.random.rand(Cout, Cin, 1, 1, 1)
            const = np.expand_dims(const, axis=-1)
            const = np.expand_dims(const, axis=-1)

        if use_conv_transpose:
            W = np.swapaxes(W, 0, 1)
        output_shape = list(input_shape)
        output_shape[1] = Cout

        if scalar_elementwise:
            const = np.random.uniform(0)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            kwargs = {
                "x": x,
                "weight": W,
                "pad_type": "valid",
                "dilations": [1] * conv_dim,
                "strides": [1] * conv_dim,
            }
            if prebuilt_bias:
                kwargs["bias"] = np.random.rand(Cout)

            x = mb.conv_transpose(**kwargs) if use_conv_transpose else mb.conv(**kwargs)

            if use_sub_instead:
                x = mb.sub(x=x, y=const)
            else:
                x = mb.add(
                    x=const if flip_add_input_order else x,
                    y=x if flip_add_input_order else const,
                )
            x = mb.relu(x=x)
            return x

        element_op = "sub" if use_sub_instead else "add"
        conv_op = "conv" if not use_conv_transpose else "conv_transpose"

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_conv_bias")
        assert get_op_types_in_program(prev_prog) == [conv_op, element_op, "relu"]
        assert get_op_types_in_program(prog) == [conv_op, "relu"]

        old_bias = prev_block.find_ops(op_type=conv_op)[0].inputs.get("bias", None)
        old_bias_val = 0 if old_bias is None else old_bias.val
        assert old_bias_val is not None
        assert block.find_ops(op_type=conv_op)[0].inputs["bias"] is not None
        new_bias_val = block.find_ops(op_type=conv_op)[0].inputs["bias"].val
        assert new_bias_val is not None
        if use_sub_instead:
            np.testing.assert_almost_equal(old_bias_val - np.squeeze(const), new_bias_val)
        else:
            np.testing.assert_almost_equal(old_bias_val + np.squeeze(const), new_bias_val)

        assert_model_is_valid(
            prog,
            {"x": input_shape},
            expected_output_shapes={block.outputs[0].name: tuple(output_shape)},
        )

    """
    Input graph:
                                                      Const
                                                        |
                                                        V
    input -----> convolution -----> transpose -----> add/sub ---> out

    Output graph:
    input -----> convolution -----> transpose -----> out
    """

    @pytest.mark.parametrize(
        "conv_dim, has_bias, is_sub, is_conv_first_input, is_bias_scalar, is_deconv, is_all_1s",
        itertools.product(
            [1, 2, 3],  # conv_dim
            [True, False],  # has_bias
            [True, False],  # is_sub
            [True, False],  # is_conv_first_input
            [True, False],  # is_bias_scalar
            [True, False],  # is_deconv
            [True, False],  # is_all_1s
        ),
    )
    def test_fuse_conv_bias_transpose_pattern(
        self,
        conv_dim,
        has_bias,
        is_sub,
        is_conv_first_input,
        is_bias_scalar,
        is_deconv,
        is_all_1s,
    ):
        if is_all_1s and is_bias_scalar:
            return

        # construct the conv weight/bias
        input_shape = None
        Cout = 8
        Cin = 3
        D = 10
        conv_weight = None
        conv_bias = (
            np.arange(Cout).astype(np.float32) if has_bias else np.zeros(Cout).astype(np.float32)
        )
        rank = conv_dim + 2

        if conv_dim == 1:
            input_shape = (1, Cin, D)
            conv_weight = np.random.rand(Cout, Cin, 1)
        elif conv_dim == 2:
            input_shape = (1, Cin, D, D)
            conv_weight = np.random.rand(Cout, Cin, 1, 1)
        elif conv_dim == 3:
            input_shape = (1, Cin, D, D, D)
            conv_weight = np.random.rand(Cout, Cin, 1, 1, 1)

        if is_deconv:
            conv_weight = np.swapaxes(conv_weight, 0, 1)

        output_shape = list(input_shape)
        output_shape[1] = Cout
        output_shape = np.array(output_shape)

        # generate the perm for the transpose op
        perm = np.arange(rank)
        np.random.shuffle(perm)
        output_shape = output_shape[perm]
        cout_index = np.where(perm == 1)[0][0]

        # generate the const bias, and reshape it to a random broadcasable shape
        bias = np.arange(Cout).astype(np.float32)
        bias_shape = [1] * rank
        bias_shape[cout_index] = Cout
        if cout_index != 0:
            crop_index = np.random.randint(low=0, high=cout_index + 1)
            bias_shape = bias_shape[crop_index:]
        bias = np.reshape(bias, bias_shape)

        # for the scalar case, random generate a number
        if is_bias_scalar:
            bias = np.random.uniform(0)

        # for the all 1s case, random generate a number and reshape it to (1, 1, ..., 1)
        if is_all_1s:
            bias = np.array([np.random.uniform(0)])
            bias_rank = np.random.randint(low=1, high=rank + 1)
            bias_shape = [1] * bias_rank
            bias = np.reshape(bias, bias_shape)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            # conv or conv_transpose
            kwargs = {
                "x": x,
                "weight": conv_weight,
                "pad_type": "valid",
                "dilations": [1] * conv_dim,
                "strides": [1] * conv_dim,
            }
            if has_bias:
                kwargs["bias"] = conv_bias
            x = mb.conv_transpose(**kwargs) if is_deconv else mb.conv(**kwargs)

            # transpose
            x = mb.transpose(x=x, perm=perm)

            # elementwise op
            element_args = {"x": x, "y": bias} if is_conv_first_input else {"x": bias, "y": x}
            element_op = mb.sub if is_sub else mb.add
            x = element_op(**element_args)
            return x

        element_op = "sub" if is_sub else "add"
        conv_op = "conv" if not is_deconv else "conv_transpose"

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_conv_bias")
        assert get_op_types_in_program(prev_prog) == [conv_op, "transpose", element_op]
        assert get_op_types_in_program(prog) == [conv_op, "transpose"]

        # get the value of new weight/bias
        new_bias_val = block.find_ops(op_type=conv_op)[0].inputs["bias"].val
        assert new_bias_val is not None

        new_weight_val = block.find_ops(op_type=conv_op)[0].inputs["weight"].val
        assert new_weight_val is not None

        # compare the weight
        if is_sub and not is_conv_first_input:
            np.testing.assert_almost_equal(new_weight_val, -conv_weight)
        else:
            np.testing.assert_almost_equal(new_weight_val, conv_weight)

        # compare the bias
        if is_sub:
            if is_conv_first_input:
                bias = -bias
            else:
                conv_bias = -conv_bias
        expected_conv_bias_val = conv_bias + np.squeeze(bias)
        np.testing.assert_almost_equal(expected_conv_bias_val, new_bias_val)

        # run the model
        assert_model_is_valid(
            prog,
            {"x": input_shape},
            expected_output_shapes={block.outputs[0].name: tuple(output_shape)},
        )


class TestConvScaleFusion:
    @staticmethod
    def _apply_weight_transform(inputs, is_deconv, is_real_div, is_conv_first_input, const_type):
        """
        Utility function to test the weight transform function in conv scale fusion pass.
        """
        Cin, _, groups = 10, 20, 10
        input_shape = (1, Cin, 2, 2)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            # create conv or deconv op
            if is_deconv:
                conv = mb.conv_transpose(
                    x=x,
                    weight=inputs["conv_weight"],
                    bias=inputs["conv_bias"],
                    groups=groups,
                )
            else:
                conv = mb.conv(
                    x=x,
                    weight=inputs["conv_weight"],
                    bias=inputs["conv_bias"],
                    groups=groups,
                )

            # create const op based on different mode
            scale = inputs["scale"]

            if const_type == "python_scale":
                scale = mb.const(val=scale)
            elif const_type == "numpy_scale":
                if type(scale) == int:
                    np_value = np.int32(scale)
                elif type(scale) == float:
                    np_value = np.float32(scale)
                scale = mb.const(val=np_value)
            elif const_type == "numpy_0d_array":
                scale = mb.const(val=np.array(scale))
            elif const_type == "numpy_1d_array":
                scale = mb.const(val=np.array([scale]))
            else:
                scale = mb.const(val=scale)

            # do the scale operation
            if is_real_div:
                x = mb.real_div(
                    x=conv,
                    y=scale,
                )
            else:
                if is_conv_first_input:
                    x = mb.mul(
                        x=conv,
                        y=scale,
                    )
                else:
                    x = mb.mul(
                        x=scale,
                        y=conv,
                    )

            return x

        apply_pass_and_basic_check(prog, "common::fuse_conv_scale")

        # get the updated weight from the prog
        conv_op = []
        for op in prog["main"].operations:
            if op.op_type == "const":
                continue
            conv_op.append(op)
        assert len(conv_op) == 1, "should only have one conv / conv_transpose layer."

        return conv_op[0].weight.val, conv_op[0].bias.val

    @pytest.mark.parametrize(
        "conv_type, is_real_div, is_conv_first_input, const_type",
        itertools.product(
            ["conv", "conv_transpose"],
            [True, False],
            [True, False],
            [
                "python_scale",
                "numpy_scale",
                "numpy_0d_array",
                "numpy_1d_array",
                "numpy_3d_array",
                "numpy_4d_array",
            ],
        ),
    )
    def test_weight_transform_conv(self, conv_type, is_real_div, is_conv_first_input, const_type):
        """
        Test the weight transform function in the conv scale fusion pass
        """
        # parameters for conv
        is_deconv = conv_type == "conv_type"
        conv_weight = np.arange(20).astype(np.float32)
        conv_weight = (
            np.reshape(conv_weight, (10, 2, 1, 1))
            if is_deconv
            else np.reshape(conv_weight, (20, 1, 1, 1))
        )
        conv_bias = np.arange(20).astype(np.float32)

        if const_type == "numpy_3d_array":
            scale = np.reshape(np.arange(20).astype(np.float32), (20, 1, 1))
        elif const_type == "numpy_4d_array":
            scale = np.reshape(np.arange(20).astype(np.float32), (1, 20, 1, 1))
        else:
            scale = 12.7

        inputs = {
            "conv_weight": conv_weight,
            "conv_bias": conv_bias,
            "scale": scale,
        }

        new_conv_weight, new_conv_bias = self._apply_weight_transform(
            inputs, is_deconv, is_real_div, is_conv_first_input, const_type
        )

        if is_real_div:
            scale = 1.0 / scale

        if const_type != "numpy_3d_array" and const_type != "numpy_4d_array":
            expected_bias = conv_bias * scale
            expected_weight = conv_weight * scale
        else:
            scale = np.reshape(scale, (20))
            expected_bias = conv_bias * scale
            if is_deconv:
                scale = np.reshape(scale, (20, 1, 1))
                expected_weight = np.reshape(np.arange(20), (20, 1, 1))
                expected_weight = expected_weight * scale
                expected_weight = np.reshape(expected_weight, (10, 2, 1, 1)).astype(np.float32)
            else:
                scale = np.reshape(scale, (20, 1, 1, 1))
                expected_weight = conv_weight * scale

        np.testing.assert_almost_equal(new_conv_weight, expected_weight)
        np.testing.assert_almost_equal(new_conv_bias, expected_bias)

        assert (
            new_conv_weight.dtype == conv_weight.dtype
        ), "weight data type should not be changed after conv_scale_fusion pass."
        assert (
            new_conv_bias.dtype == conv_weight.dtype
        ), "bias data type should be the same as the weight for conv layer."

    @pytest.mark.parametrize(
        "rank, groups, has_bias, scale_op, scale_type, backend",
        itertools.product(
            [3, 4], [1, 10], [False, True], ["mul", "real_div"], ["scalar", "vector"], backends
        ),
    )
    def test_conv(self, rank, groups, has_bias, scale_op, scale_type, backend):
        """
        Input graph:
        input -----> conv -----> mul/real_div ---> out

        Output graph:
        input -----> conv ----> out
        """
        Cin, Cout = 10, 30
        input_shape = (2, Cin, 20) if rank == 3 else (2, Cin, 20, 24)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            # conv layer
            conv_weight = (
                np.random.rand(Cout, Cin // groups, 2)
                if rank == 3
                else np.random.rand(Cout, Cin // groups, 2, 3)
            )
            conv_bias = np.random.rand(Cout) if has_bias else None
            x = mb.conv(
                x=x,
                weight=conv_weight,
                bias=conv_bias,
                groups=groups,
            )
            if scale_type == "scalar":
                scale = np.array([2.3])
            else:
                scale = np.arange(Cout).astype(np.float32)
                scale = np.reshape(scale, (1, Cout, 1) if rank == 3 else (Cout, 1, 1))

            # scale layer
            if scale_op == "mul":
                x = mb.mul(x=x, y=scale)
            elif scale_op == "real_div":
                x = mb.real_div(x=x, y=scale)
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_conv_scale")

        assert get_op_types_in_program(prev_prog) == ["conv", scale_op]
        assert get_op_types_in_program(prog) == ["conv"]

        # validate graph pass
        output_shape = (2, Cout, 19) if rank == 3 else (2, Cout, 19, 22)
        assert_model_is_valid(
            prog,
            {"x": input_shape},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=backend,
        )

    @pytest.mark.parametrize(
        "rank, groups, has_bias, scale_op, scale_type, backend",
        itertools.product(
            [3, 4], [1, 10], [False, True], ["mul", "real_div"], ["scalar", "vector"], backends
        ),
    )
    def test_conv_transpose(self, rank, groups, has_bias, scale_op, scale_type, backend):
        """
        Input graph:
        input -----> conv_transpose -----> mul/real_div ---> out

        Output graph:
        input -----> conv_transpose ----> out
        """
        Cin, Cout = 10, 30
        input_shape = (2, Cin, 20) if rank == 3 else (2, Cin, 20, 24)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            # conv layer
            conv_weight = (
                np.random.rand(Cin, Cout // groups, 2)
                if rank == 3
                else np.random.rand(Cin, Cout // groups, 2, 3)
            )
            conv_bias = np.random.rand(Cout) if has_bias else None
            x = mb.conv_transpose(
                x=x,
                weight=conv_weight,
                bias=conv_bias,
                groups=groups,
            )

            if scale_type == "scalar":
                scale = np.array([2.3])
            else:
                scale = np.arange(Cout).astype(np.float32)
                scale = np.reshape(scale, (Cout, 1) if rank == 3 else (1, Cout, 1, 1))

            # scale layer
            if scale_op == "mul":
                x = mb.mul(x=x, y=scale)
            elif scale_op == "real_div":
                x = mb.real_div(x=x, y=scale)
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_conv_scale")

        assert get_op_types_in_program(prev_prog) == ["conv_transpose", scale_op]
        assert get_op_types_in_program(prog) == ["conv_transpose"]

        # validate graph pass
        output_shape = (2, Cout, 21) if rank == 3 else (2, Cout, 21, 26)
        assert_model_is_valid(
            prog,
            {"x": input_shape},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=backend,
        )


class TestFusePadConv(unittest.TestCase):
    """
    Input graph:
    input -----> pad -----> transpose -----> conv -----> transpose ---> out

    Output graph:
    input -----> transpose -----> pad ----> conv -----> transpose ----> out
    """

    def test_simple_direct_output(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 20, 24))])
        def prog(x):
            x = mb.pad(x=x, pad=[0, 0, 1, 1, 1, 1, 0, 0])
            x = mb.transpose(x=x, perm=[0, 3, 1, 2])
            x = mb.conv(x=x, weight=np.random.random([24, 24, 3, 3]), pad_type="valid")
            x = mb.transpose(x=x, perm=[0, 2, 3, 1])
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_pad_conv")
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["pad", "transpose", "conv", "transpose"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["transpose", "pad", "conv", "transpose"])
        assert_model_is_valid(
            prog,
            {"x": (1, 16, 20, 24)},
            expected_output_shapes={block.outputs[0].name: (1, 16, 20, 24)},
        )

    """
    Input graph:
    input -----> pad -----> transpose -----> conv -----> transpose ---> out
                  |
                  |
                  --------> transpose -----> conv -----> transpose ---> out

    Output graph:
    input ---------> transpose -----> pad -----> conv -----> transpose ---> out
             |
             |
             ------> transpose -----> pad -----> conv -----> transpose ---> out

    """

    def test_pad_transposed_forked_conv(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 20, 24))])
        def prog(x):
            pad = mb.pad(x=x, pad=[0, 0, 1, 1, 1, 1, 0, 0])
            x = mb.transpose(x=pad, perm=[0, 3, 1, 2])
            x = mb.conv(x=x, weight=np.random.random([24, 24, 3, 3]), pad_type="valid")
            x = mb.transpose(x=x, perm=[0, 2, 3, 1])
            y = mb.transpose(x=pad, perm=[0, 3, 1, 2])
            y = mb.conv(x=y, weight=np.random.random([24, 24, 3, 3]), pad_type="valid")
            y = mb.transpose(x=y, perm=[0, 2, 3, 1])
            return x, y

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_pad_conv")
        self.assertEqual(
            get_op_types_in_program(prev_prog),
            ["pad", "transpose", "conv", "transpose", "transpose", "conv", "transpose"],
        )
        self.assertEqual(
            get_op_types_in_program(prog),
            ["transpose", "pad", "conv", "transpose", "transpose", "pad", "conv", "transpose"],
        )
        assert_model_is_valid(
            prog,
            {"x": (1, 16, 20, 24)},
            expected_output_shapes={
                block.outputs[0].name: (1, 16, 20, 24),
                block.outputs[1].name: (1, 16, 20, 24),
            },
        )

    """
    Input graph:
    input -----> pad -----> transpose -----> conv -----> transpose ---> out
                  |
                  |
                  ---------> out

    Output graph:
    No change.
    """

    def test_pad_output(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 20, 24))])
        def prog(x):
            pad = mb.pad(x=x, pad=[0, 0, 1, 1, 1, 1, 0, 0])
            x = mb.transpose(x=pad, perm=[0, 3, 1, 2])
            x = mb.conv(x=x, weight=np.random.random([24, 24, 3, 3]), pad_type="valid")
            x = mb.transpose(x=x, perm=[0, 2, 3, 1])
            return x, pad

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_pad_conv")
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["pad", "transpose", "conv", "transpose"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["pad", "transpose", "conv", "transpose"])
        assert_model_is_valid(
            prog,
            {"x": (1, 16, 20, 24)},
            expected_output_shapes={
                block.outputs[0].name: (1, 16, 20, 24),
                block.outputs[1].name: (1, 18, 22, 24),
            },
        )


class TestConcatToPixelShuffle(unittest.TestCase):
    def test_success(self):
        """
        Input graph:
        input1(1, 2, 3, 4) -----> concat(axis=2, interleave=True) -----> concat(axis=3, interleave=True) ---> out(1, 2, 6, 8)
                                             ^                                           ^
                                             |                                           |
        input2(1, 2, 3, 4) -------------------                                           |
                                                                                         |
        input3(1, 2, 3, 4) -----> concat(axis=2, interleave=True) -----------------------|
                                             ^
                                             |
        input4(1, 2, 3, 4) ------------------|

        Output graph:
        input1(1, 2, 3, 4) -----> concat(axis=1) ---> pixel_shuffle(upsample_factor=2) ----> out(1, 2, 6, 8)
                                     ^
        input2(1, 2, 3, 4) ----------|
                                     |
        input3(1, 2, 3, 4) ----------|
                                     |
        input4(1, 2, 3, 4) ----------|
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
            ]
        )
        def prog(x1, x2, x3, x4):
            ab = mb.concat(values=[x1, x2], axis=2, interleave=True)
            cd = mb.concat(values=[x3, x4], axis=2, interleave=True)
            x = mb.concat(values=[ab, cd], axis=3, interleave=True)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(get_op_types_in_program(prev_prog), ["concat", "concat", "concat"])
        self.assertEqual(get_op_types_in_program(prog), ["concat", "pixel_shuffle"])

        inputs = {"x1": (1, 2, 3, 4), "x2": (1, 2, 3, 4), "x3": (1, 2, 3, 4), "x4": (1, 2, 3, 4)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 6, 8)},
        )

        mlmodel = ct.convert(
            prog,
            source="milinternal",
            convert_to="neuralnetwork",
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )

        if not _IS_MACOS:
            # Can not get predictions unless on macOS.
            return

        input_dict = dict()
        input_dict["x1"] = np.ones(inputs["x1"])
        input_dict["x2"] = np.ones(inputs["x2"]) * 2
        input_dict["x3"] = np.ones(inputs["x3"]) * 3
        input_dict["x4"] = np.ones(inputs["x4"]) * 4

        output_name = block.outputs[0].name

        ab = np.reshape(
            np.stack((input_dict["x1"], input_dict["x2"]), axis=3), newshape=[1, 2, 6, 4]
        )
        cd = np.reshape(
            np.stack((input_dict["x3"], input_dict["x4"]), axis=3), newshape=[1, 2, 6, 4]
        )
        old_prediction = np.reshape(np.stack((ab, cd), axis=4), newshape=[1, 2, 6, 8])

        prediction = mlmodel.predict(input_dict)
        np.testing.assert_allclose(old_prediction, prediction[output_name], atol=1e-04, rtol=1e-05)

    def test_nested(self):
        """
        Two nested blocks that will each be transformed.
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
            ]
        )
        def prog(x1, x2, x3, x4, x5, x6, x7, x8):
            ab = mb.concat(values=[x1, x2], axis=2, interleave=True)
            cd = mb.concat(values=[x3, x4], axis=2, interleave=True)
            x = mb.concat(values=[ab, cd], axis=3, interleave=True)

            ef = mb.concat(values=[x5, x6], axis=2, interleave=True)
            gh = mb.concat(values=[x7, x8], axis=2, interleave=True)
            y = mb.concat(values=[ef, gh], axis=3, interleave=True)

            z = mb.concat(values=[x, y], axis=1)

            return z

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog),
            ["concat", "concat", "concat", "concat", "concat", "concat", "concat"],
        )
        self.assertEqual(
            get_op_types_in_program(prog),
            ["concat", "pixel_shuffle", "concat", "pixel_shuffle", "concat"],
        )

        inputs = {
            "x1": (1, 2, 3, 4),
            "x2": (1, 2, 3, 4),
            "x3": (1, 2, 3, 4),
            "x4": (1, 2, 3, 4),
            "x5": (1, 2, 3, 4),
            "x6": (1, 2, 3, 4),
            "x7": (1, 2, 3, 4),
            "x8": (1, 2, 3, 4),
        }
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 4, 6, 8)},
        )

        input_dict = dict()
        for name, shape in inputs.items():
            input_dict[name] = np.random.rand(*shape)

        output_name = block.outputs[0].name

        ab = np.reshape(
            np.stack((input_dict["x1"], input_dict["x2"]), axis=3), newshape=[1, 2, 6, 4]
        )
        cd = np.reshape(
            np.stack((input_dict["x3"], input_dict["x4"]), axis=3), newshape=[1, 2, 6, 4]
        )
        x = np.reshape(np.stack((ab, cd), axis=4), newshape=[1, 2, 6, 8])

        ef = np.reshape(
            np.stack((input_dict["x5"], input_dict["x6"]), axis=3), newshape=[1, 2, 6, 4]
        )
        gh = np.reshape(
            np.stack((input_dict["x7"], input_dict["x8"]), axis=3), newshape=[1, 2, 6, 4]
        )
        y = np.reshape(np.stack((ef, gh), axis=4), newshape=[1, 2, 6, 8])

        old_prediction = np.concatenate((x, y), axis=1)

        mlmodel = ct.convert(
            prog,
            source="milinternal",
            convert_to="neuralnetwork",
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )

        if _IS_MACOS:
            prediction = mlmodel.predict(input_dict)
            np.testing.assert_allclose(
                old_prediction, prediction[output_name], atol=1e-04, rtol=1e-05
            )

    def test_failure_0(self):
        """
        The h_concat has three inputs, so the pattern won't match.
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
            ]
        )
        def prog(x1, x2, x3, x4):
            ab = mb.concat(values=[x1, x2, x3], axis=2, interleave=True)
            cd = mb.concat(values=[x3, x4, x1], axis=2, interleave=True)
            x = mb.concat(values=[ab, cd], axis=3, interleave=True)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(get_op_types_in_program(prev_prog), ["concat", "concat", "concat"])
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])

    def test_failure_1(self):
        """
        The first concat is on the wrong axis, so the pattern won't match.
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
            ]
        )
        def prog(x1, x2, x3, x4):
            ab = mb.concat(values=[x1, x2], axis=3, interleave=True)
            cd = mb.concat(values=[x3, x4], axis=3, interleave=True)
            x = mb.concat(values=[ab, cd], axis=3, interleave=True)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(get_op_types_in_program(prev_prog), ["concat", "concat", "concat"])
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])

    def test_failure_2(self):
        """
        The last concat is on the wrong axis, so the pattern won't match.
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
            ]
        )
        def prog(x1, x2, x3, x4):
            ab = mb.concat(values=[x1, x2], axis=2, interleave=True)
            cd = mb.concat(values=[x3, x4], axis=2, interleave=True)
            x = mb.concat(values=[ab, cd], axis=2, interleave=True)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(get_op_types_in_program(prev_prog), ["concat", "concat", "concat"])
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])

    def test_failure_3(self):
        """
        The first concat is not interleaved, so the pattern won't match.
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
            ]
        )
        def prog(x1, x2, x3, x4):
            ab = mb.concat(values=[x1, x2], axis=2, interleave=False)
            cd = mb.concat(values=[x3, x4], axis=2, interleave=True)
            x = mb.concat(values=[ab, cd], axis=3, interleave=True)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(get_op_types_in_program(prev_prog), ["concat", "concat", "concat"])
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])

    def test_failure_4(self):
        """
        The second concat is not interleaved, so the pattern won't match.
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
            ]
        )
        def prog(x1, x2, x3, x4):
            ab = mb.concat(values=[x1, x2], axis=2, interleave=True)
            cd = mb.concat(values=[x3, x4], axis=2, interleave=False)
            x = mb.concat(values=[ab, cd], axis=3, interleave=True)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(get_op_types_in_program(prev_prog), ["concat", "concat", "concat"])
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])

    def test_failure_5(self):
        """
        The last concat is not interleaved, so the pattern won't match.
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
            ]
        )
        def prog(x1, x2, x3, x4):
            ab = mb.concat(values=[x1, x2], axis=2, interleave=True)
            cd = mb.concat(values=[x3, x4], axis=2, interleave=True)
            x = mb.concat(values=[ab, cd], axis=3, interleave=False)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(get_op_types_in_program(prev_prog), ["concat", "concat", "concat"])
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])

    def test_failure_6(self):
        """
        The inputs are the wrong rank, so the pattern won't match.
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 2, 3, 4, 5)),
                mb.TensorSpec(shape=(1, 2, 3, 4, 5)),
                mb.TensorSpec(shape=(1, 2, 3, 4, 5)),
                mb.TensorSpec(shape=(1, 2, 3, 4, 5)),
            ]
        )
        def prog(x1, x2, x3, x4):
            ab = mb.concat(values=[x1, x2], axis=2, interleave=True)
            cd = mb.concat(values=[x3, x4], axis=2, interleave=True)
            x = mb.concat(values=[ab, cd], axis=3, interleave=True)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(get_op_types_in_program(prev_prog), ["concat", "concat", "concat"])
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])

    def test_failure_7(self):
        """
        Extra input to the w_concats means the pattern won't match.
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 2, 4, 4)),
                mb.TensorSpec(shape=(1, 2, 4, 4)),
                mb.TensorSpec(shape=(1, 2, 4, 4)),
                mb.TensorSpec(shape=(1, 2, 4, 4)),
                mb.TensorSpec(shape=(1, 2, 8, 4)),
            ]
        )
        def prog(x1, x2, x3, x4, x5):
            ab = mb.concat(values=[x1, x2], axis=2, interleave=True)
            cd = mb.concat(values=[x3, x4], axis=2, interleave=True)
            x = mb.concat(values=[ab, cd, x5], axis=3, interleave=True)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(get_op_types_in_program(prev_prog), ["concat", "concat", "concat"])
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])


class TestConcatInterleave:
    def test_concat_interleave_fusion_pass(self):
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

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(B, C, H, W)), mb.TensorSpec(shape=(B, C, H, W))]
        )
        def prog(x, y):
            z = mb.concat(values=[x, y], axis=1)
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
            expected_output_shapes={block.outputs[0].name: (B, 2 * C, H, W)},
        )


class TestFuseOnehotMatmulToGather:
    @pytest.mark.parametrize(
        "backend, rank, opset_version",
        itertools.product(backends, [1, 2, 3, 4], [None, ct.target.iOS17]),
    )
    def test_fuse_onehot_matmul_to_gather(self, backend, rank, opset_version):
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

        @mb.program(
            input_specs=[mb.TensorSpec(shape=input_shape, dtype=types.int32)],
            opset_version=opset_version,
        )
        def prog(x):
            x = mb.one_hot(
                indices=x, on_value=1.0, off_value=0.0, axis=-1, one_hot_vector_size=vocab_size
            )
            x = mb.matmul(x=x, y=np.random.rand(vocab_size, embedding_size))
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_onehot_matmul_to_gather"
        )
        assert get_op_types_in_program(prev_prog) == ["one_hot", "matmul"]
        if opset_version == ct.target.iOS17:
            # Several ops added to make sure indices in iOS17 gather is non-negative.
            assert get_op_types_in_program(prog) == [
                "greater_equal",
                "shape",
                "slice_by_index",
                "add",
                "select",
                "gather",
            ]
        else:
            assert get_op_types_in_program(prog) == ["gather"]

        if opset_version == ct.target.iOS17:
            if backend[0] != "mlprogram" or _macos_version() < (14, 0):
                pytest.skip("IOS17 target available only on macOS 14+ with mlprogram.")

        assert_model_is_valid(
            prog,
            {"x": input_shape},
            backend=backend,
            expected_output_shapes={block.outputs[0].name: input_shape + (embedding_size,)},
            minimum_deployment_target=opset_version,
        )


class TestReplaceStackReshape(unittest.TestCase):
    def test_with_interleave(self):
        """
        input1(1, 5, 3, 4) -----> stack(axis=2) -----> reshape(shape=(1, 10, 3, 4)) ---> out(1, 10, 3, 4)
                                    ^
                                    |
        input2(1, 5, 3, 4) ----------

        Output graph:
        input -----> concat ----> out

        """

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))]
        )
        def prog(x1, x2):
            x = mb.stack(values=[x1, x2], axis=2)
            x = mb.reshape(x=x, shape=[1, 10, 3, 4])
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )
        self.assertEqual(get_op_types_in_program(prev_prog), ["stack", "reshape"])
        self.assertEqual(get_op_types_in_program(prog), ["concat"])

        inputs = {"x1": (1, 5, 3, 4), "x2": (1, 5, 3, 4)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 10, 3, 4)},
        )

        concat_ops = [op for op in block.operations if op.op_type == "concat"]
        concat_op = concat_ops[0]
        assert concat_op.interleave.val == True  # noqa: E712

        output_name = block.outputs[0].name

        mlmodel = ct.convert(
            prog,
            source="milinternal",
            convert_to="neuralnetwork",
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )

        if not _IS_MACOS:
            # Can not get predictions unless on macOS.
            return

        input_dict = dict()
        for name, shape in inputs.items():
            input_dict[name] = np.random.rand(*shape)

        old_prediction = np.reshape(
            np.stack([input_dict["x1"], input_dict["x2"]], axis=2), newshape=[1, 10, 3, 4]
        )

        prediction = mlmodel.predict(input_dict)

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

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))]
        )
        def prog(x1, x2):
            x = mb.stack(values=[x1, x2], axis=1)
            x = mb.reshape(x=x, shape=[1, 10, 3, 4])
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )
        self.assertEqual(get_op_types_in_program(prev_prog), ["stack", "reshape"])
        self.assertEqual(get_op_types_in_program(prog), ["concat"])

        inputs = {"x1": (1, 5, 3, 4), "x2": (1, 5, 3, 4)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 10, 3, 4)},
        )

        concat_ops = [op for op in block.operations if op.op_type == "concat"]
        concat_op = concat_ops[0]
        assert concat_op.interleave.val == False  # noqa: E712

        output_name = block.outputs[0].name

        mlmodel = ct.convert(
            prog,
            source="milinternal",
            convert_to="neuralnetwork",
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )

        if not _IS_MACOS:
            # Can not get predictions unless on macOS.
            return

        input_dict = dict()
        for name, shape in inputs.items():
            input_dict[name] = np.random.rand(*shape)

        old_prediction = np.reshape(
            np.stack([input_dict["x1"], input_dict["x2"]], axis=1), newshape=[1, 10, 3, 4]
        )

        prediction = mlmodel.predict(input_dict)
        np.testing.assert_allclose(old_prediction, prediction[output_name], atol=1e-04, rtol=1e-05)

    def test_multiple(self):
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
                mb.TensorSpec(shape=(1, 2, 3, 4)),
            ]
        )
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
            get_op_types_in_program(prev_prog),
            ["stack", "reshape", "stack", "reshape", "stack", "reshape"],
        )
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])

        inputs = {"x1": (1, 2, 3, 4), "x2": (1, 2, 3, 4), "x3": (1, 2, 3, 4), "x4": (1, 2, 3, 4)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 4, 6, 4)},
        )

        output_name = block.outputs[0].name

        mlmodel = ct.convert(
            prog,
            source="milinternal",
            convert_to="neuralnetwork",
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )

        if not _IS_MACOS:
            # Can not get predictions unless on macOS.
            return

        input_dict = dict()
        for name, shape in inputs.items():
            input_dict[name] = np.random.rand(*shape)

        branch_1 = np.reshape(
            np.stack([input_dict["x1"], input_dict["x2"]], axis=1), newshape=[1, 4, 3, 4]
        )
        branch_2 = np.reshape(
            np.stack([input_dict["x3"], input_dict["x4"]], axis=1), newshape=[1, 4, 3, 4]
        )
        old_prediction = np.reshape(np.stack([branch_1, branch_2], axis=2), newshape=[1, 4, 6, 4])

        prediction = mlmodel.predict(input_dict)

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

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))]
        )
        def prog(x1, x2):
            a = mb.stack(values=[x1, x2], axis=1)
            a = mb.reshape(x=a, shape=[-1, 5, 6, 4])
            return a

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )

        self.assertEqual(get_op_types_in_program(prev_prog), ["stack", "reshape"])
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

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))]
        )
        def prog(x1, x2):
            a = mb.stack(values=[x1, x2], axis=1)
            a = mb.reshape(x=a, shape=[-1, 5, 12, 2])
            return a

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )

        self.assertEqual(get_op_types_in_program(prev_prog), ["stack", "reshape"])
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

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))]
        )
        def prog(x1, x2):
            a = mb.stack(values=[x1, x2], axis=1)
            a = mb.reshape(x=a, shape=[-1, 2, 5, 4, 3])
            return a

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )

        self.assertEqual(get_op_types_in_program(prev_prog), ["stack", "reshape"])
        self.assertEqual(get_op_types_in_program(prog), ["stack", "reshape"])

    def test_negative_4(self):
        """
        More than two inputs to the stack op -- can't be transformed.
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 5, 3, 4)),
                mb.TensorSpec(shape=(1, 5, 3, 4)),
                mb.TensorSpec(shape=(1, 5, 3, 4)),
            ]
        )
        def prog(x1, x2, x3):
            a = mb.stack(values=[x1, x2, x3], axis=1)
            a = mb.reshape(x=a, shape=[-1, 15, 4, 3])
            return a

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )

        self.assertEqual(get_op_types_in_program(prev_prog), ["stack", "reshape"])
        self.assertEqual(get_op_types_in_program(prog), ["stack", "reshape"])

    def test_negative_5(self):
        """
        The stack and reshape are not adjacent, so the graph is not transformed.
        """

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))]
        )
        def prog(x1, x2):
            a = mb.stack(values=[x1, x2], axis=1)
            a = mb.relu(x=a)
            a = mb.reshape(x=a, shape=[-1, 10, 4, 3])
            return a

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )

        self.assertEqual(get_op_types_in_program(prev_prog), ["stack", "relu", "reshape"])
        self.assertEqual(get_op_types_in_program(prog), ["stack", "relu", "reshape"])

    def test_negative_6(self):
        """
        The stack op's output is used elsewhere in the graph, so it can't be removed
        """

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))]
        )
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
        self.assertEqual(
            get_op_types_in_program(prog), ["stack", "reshape", "relu", "reshape", "add"]
        )

    def test_negative_7(self):
        """
        The stack op is not followed by any other ops.
        """

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 5, 3, 4)), mb.TensorSpec(shape=(1, 5, 3, 4))]
        )
        def prog(x1, x2):
            a = mb.stack(values=[x1, x2], axis=1)
            return a

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::replace_stack_reshape"
        )

        self.assertEqual(get_op_types_in_program(prev_prog), ["stack"])
        self.assertEqual(get_op_types_in_program(prog), ["stack"])


class TestUseReflectionPadding:
    def test_success_w_axis(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):
            left = mb.slice_by_index(
                x=x1, begin=[0, 0, 0, 1], end=[0, 0, 0, 2], end_mask=[True, True, True, False]
            )
            right = mb.slice_by_index(
                x=x1, begin=[0, 0, 0, -2], end=[0, 0, 0, -1], end_mask=[True, True, True, False]
            )
            x = mb.concat(values=[left, x1, right], axis=3)

            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::use_reflection_padding")
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
            left0 = mb.slice_by_index(
                x=x1, begin=[0, 0, 0, 2], end=[0, 0, 0, 3], end_mask=[True, True, True, False]
            )
            left1 = mb.slice_by_index(
                x=x1, begin=[0, 0, 0, 1], end=[0, 0, 0, 2], end_mask=[True, True, True, False]
            )
            right0 = mb.slice_by_index(
                x=x1, begin=[0, 0, 0, -2], end=[0, 0, 0, -1], end_mask=[True, True, True, False]
            )
            right1 = mb.slice_by_index(
                x=x1, begin=[0, 0, 0, -3], end=[0, 0, 0, -2], end_mask=[True, True, True, False]
            )
            x = mb.concat(values=[left0, left1, x1, right0, right1], axis=3)

            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::use_reflection_padding")
        assert get_op_types_in_program(prev_prog) == [
            "slice_by_index",
            "slice_by_index",
            "slice_by_index",
            "slice_by_index",
            "concat",
        ]
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
            left = mb.slice_by_index(
                x=x1, begin=[0, 0, 1, 0], end=[0, 0, 2, 0], end_mask=[True, True, False, True]
            )
            right = mb.slice_by_index(
                x=x1, begin=[0, 0, -2, 0], end=[0, 0, -1, 0], end_mask=[True, True, False, True]
            )
            x = mb.concat(values=[left, x1, right], axis=2)

            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::use_reflection_padding")
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
            left = mb.slice_by_index(
                x=x1, begin=[0, 0, 1, 0], end=[0, 0, 2, 0], end_mask=[True, True, False, True]
            )
            right = mb.slice_by_index(
                x=x1, begin=[0, 0, -2, 0], end=[0, 0, -1, 0], end_mask=[True, True, False, True]
            )
            # Concat is not in correct order
            x = mb.concat(values=[left, right, x1], axis=2)

            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::use_reflection_padding")
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
            left0 = mb.slice_by_index(
                x=x1, begin=[0, 0, 0, 1], end=[0, 0, 0, 2], end_mask=[True, True, True, False]
            )
            left1 = mb.slice_by_index(
                x=x1, begin=[0, 0, 0, 2], end=[0, 0, 0, 3], end_mask=[True, True, True, False]
            )
            right0 = mb.slice_by_index(
                x=x1, begin=[0, 0, 0, -3], end=[0, 0, 0, -2], end_mask=[True, True, True, False]
            )
            right1 = mb.slice_by_index(
                x=x1, begin=[0, 0, 0, -2], end=[0, 0, 0, -1], end_mask=[True, True, True, False]
            )
            # concat args are out of order
            x = mb.concat(values=[left0, left1, x1, right1, right0], axis=3)

            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::use_reflection_padding")
        assert get_op_types_in_program(prev_prog) == [
            "slice_by_index",
            "slice_by_index",
            "slice_by_index",
            "slice_by_index",
            "concat",
        ]
        assert get_op_types_in_program(prog) == [
            "slice_by_index",
            "slice_by_index",
            "slice_by_index",
            "slice_by_index",
            "concat",
        ]

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
            left = mb.slice_by_index(
                x=x1, begin=[0, 0, 1, 0], end=[0, 0, 3, 0], end_mask=[True, True, False, True]
            )
            right = mb.slice_by_index(
                x=x1, begin=[0, 0, -2, 0], end=[0, 0, -1, 0], end_mask=[True, True, False, True]
            )
            x = mb.concat(values=[left, x1, right], axis=2)

            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::use_reflection_padding")
        assert get_op_types_in_program(prev_prog) == ["slice_by_index", "slice_by_index", "concat"]
        assert get_op_types_in_program(prog) == ["slice_by_index", "slice_by_index", "concat"]

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 9, 8)},
        )

    def test_failure_not_all_same_input(self):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8)), mb.TensorSpec(shape=(1, 2, 6, 8))]
        )
        def prog(x1, x2):
            left0 = mb.slice_by_index(
                x=x1, begin=[0, 0, 0, 1], end=[0, 0, 0, 2], end_mask=[True, True, True, False]
            )
            left1 = mb.slice_by_index(
                x=x1, begin=[0, 0, 0, 2], end=[0, 0, 0, 3], end_mask=[True, True, True, False]
            )
            right0 = mb.slice_by_index(
                x=x1, begin=[0, 0, 0, -3], end=[0, 0, 0, -2], end_mask=[True, True, True, False]
            )
            # one of the slices consumes a different input from the others
            right1 = mb.slice_by_index(
                x=x2, begin=[0, 0, 0, -2], end=[0, 0, 0, -1], end_mask=[True, True, True, False]
            )
            x = mb.concat(values=[left0, left1, x1, right0, right1], axis=3)

            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::use_reflection_padding")
        assert get_op_types_in_program(prev_prog) == [
            "slice_by_index",
            "slice_by_index",
            "slice_by_index",
            "slice_by_index",
            "concat",
        ]
        assert get_op_types_in_program(prog) == [
            "slice_by_index",
            "slice_by_index",
            "slice_by_index",
            "slice_by_index",
            "concat",
        ]

        inputs = {"x1": (1, 2, 6, 8), "x2": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 6, 12)},
        )

    def test_failure_slice_output(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x1):
            left = mb.slice_by_index(
                x=x1, begin=[0, 0, 0, 1], end=[0, 0, 0, 2], end_mask=[True, True, True, False]
            )
            right = mb.slice_by_index(
                x=x1, begin=[0, 0, 0, -2], end=[0, 0, 0, -1], end_mask=[True, True, True, False]
            )
            x = mb.concat(values=[left, x1, right], axis=3)

            # slice is an output
            return x, right

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::use_reflection_padding")
        assert get_op_types_in_program(prev_prog) == ["slice_by_index", "slice_by_index", "concat"]
        assert get_op_types_in_program(prog) == ["slice_by_index", "slice_by_index", "concat"]

        inputs = {"x1": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={
                block.outputs[0].name: (1, 2, 6, 10),
                block.outputs[1].name: (1, 2, 6, 1),
            },
        )

    def test_concat_input_only(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 6, 8))])
        def prog(x):
            x = mb.concat(values=[x, x, x], axis=0)
            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::use_reflection_padding")
        assert get_op_types_in_program(prog) == ["concat"]

        inputs = {"x": (1, 2, 6, 8)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (3, 2, 6, 8)},
        )


class TestDivideToMultiply:
    def test_divide_to_multiply(self):
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

        if _VALIDATE_MODEL:
            assert_model_is_valid(prog, {"x": (2, 4)})


class TestSelectOptimization:
    @pytest.mark.parametrize(
        "cond_val, is_cond_scalar, need_broadcast, is_block_output",
        itertools.product(
            (True, False),
            (True, False),
            (True, False),
            (True, False),
        ),
    )
    def test_const_scalar_cond(self, cond_val, is_cond_scalar, need_broadcast, is_block_output):
        """
        Input graph:

            const(cond) -|
                         |
            a -----------|-> select -> (add 1.0 if not is_block_output) -> output
                         |
            b -----------|

        If a and b need broadcast, then nothing is changed; else output graph becomes:

            if cond:
                if is_block_output:
                    a -> identity -> output
                else:
                    a -> add 1.0 -> output
            else:
                if is_block_output:
                    b -> identity -> output
                else:
                    b -> add 1.0 -> output
        """
        SHAPE = (5, 2, 3)

        if need_broadcast:
            a_shape = (5, 2, 1)
            b_shape = (5, 1, 3)
        else:
            a_shape = SHAPE
            b_shape = SHAPE

        if is_cond_scalar:
            cond = cond_val
        else:
            cond_shape = (5, 1, 1)
            cond = np.full(cond_shape, cond_val)

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=a_shape),
                mb.TensorSpec(shape=b_shape),
            ]
        )
        def prog(a, b):
            c = mb.select(cond=cond, a=a, b=b)
            if not is_block_output:
                c = mb.add(x=c, y=1.0)
            return c

        prev_prog, _, _ = apply_pass_and_basic_check(prog, "common::select_optimization")
        apply_pass_and_basic_check(prog, "common::noop_elimination")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        # check previous program
        if is_block_output:
            assert get_op_types_in_program(prev_prog) == ["select"]
        else:
            assert get_op_types_in_program(prev_prog) == ["select", "add"]
        # check passed program
        if is_block_output:
            if need_broadcast:
                assert get_op_types_in_program(prog) == ["select"]
            else:
                assert get_op_types_in_program(prog) == ["identity"]
        else:
            if need_broadcast:
                assert get_op_types_in_program(prog) == ["select", "add"]
            else:
                assert get_op_types_in_program(prog) == ["add"]

        output_name = block.outputs[0].name
        assert_model_is_valid(
            prog,
            {"a": a_shape, "b": b_shape},
            expected_output_shapes={output_name: SHAPE},
        )

        prev_model = ct.convert(
            prev_prog,
            pass_pipeline=ct.PassPipeline.EMPTY,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        model = ct.convert(
            prog,
            pass_pipeline=ct.PassPipeline.EMPTY,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )

        a = np.random.rand(*a_shape)
        b = np.random.rand(*b_shape)
        input_dict = {"a": a, "b": b}
        prev_output = prev_model.predict(input_dict)[output_name]
        output = model.predict(input_dict)[output_name]
        np.testing.assert_allclose(prev_output, output, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        "is_a_const, is_fill_scalar",
        itertools.product((True, False), (True, False)),
    )
    def test_inf_const_selection(self, is_a_const, is_fill_scalar):
        """
        Input graph if is_a_const (else input and fill are swapped):

            const(cond) ------|
                              |
            input ------------|-> select -> tanh -> output
                              |
            const(±inf fill) -|

        Output graph:

            input -> add -> tanh -> output
        """
        INPUT_SHAPE = (5, 2, 3)

        cond_shape = (2, 3)

        while True:
            cond = np.random.randint(0, 2, size=cond_shape) == 0
            if not np.all(cond) and not np.all(np.logical_not(cond)):
                break

        if is_fill_scalar:
            fill = np.float16(-np.inf)
        else:
            fill_shape = (5, 2, 1)
            fill = np.empty(fill_shape, dtype=np.float16)
            neg_pos = np.random.randint(0, 2, size=fill_shape)
            fill[np.where(neg_pos == 0)] = -np.inf
            fill[np.where(neg_pos == 1)] = np.inf

        output_shape = INPUT_SHAPE

        @mb.program(input_specs=[mb.TensorSpec(shape=INPUT_SHAPE, dtype=types.fp16)])
        def prog(x):
            if is_a_const:
                y = mb.select(cond=cond, a=fill, b=x)
            else:
                y = mb.select(cond=cond, a=x, b=fill)
            return mb.tanh(x=y)

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::select_optimization")
        assert get_op_types_in_program(prev_prog) == ["select", "tanh"]
        assert get_op_types_in_program(prog) == ["add", "tanh"]

        output_name = block.outputs[0].name
        assert_model_is_valid(
            prog,
            {"x": INPUT_SHAPE},
            expected_output_shapes={output_name: output_shape},
        )

        prev_model = ct.convert(
            prev_prog,
            pass_pipeline=ct.PassPipeline.EMPTY,
            convert_to="mlprogram",
        )
        model = ct.convert(
            prog,
            pass_pipeline=ct.PassPipeline.EMPTY,
            convert_to="mlprogram",
        )

        a = 65500.0 * np.random.rand(*INPUT_SHAPE)
        input_dict = {"x": a}
        prev_output = prev_model.predict(input_dict)[output_name]
        output = model.predict(input_dict)[output_name]
        np.testing.assert_allclose(prev_output, output, rtol=0.0, atol=0.0)


class TestFuseElementwiseToBatchNorm:
    """
    Input graph:
                                 Const     Const
                                   |         |
                                   V         V
    input -----> transpose -----> mul ----> add ---> out

    Output graph:
    input -----> transpose -----> batchnorm ----> out
    """

    @pytest.mark.parametrize(
        "flip_mul_input_order, flip_add_input_order, rank_3_const_input",
        itertools.product([False, True], [False, True], [False, True]),
    )
    def test_mul_add_fusion_to_batchnorm(
        self, flip_mul_input_order, flip_add_input_order, rank_3_const_input
    ):

        C = 3
        gamma = np.random.rand(1, C, 1, 1)
        beta = np.random.rand(1, C, 1, 1)
        if rank_3_const_input:
            gamma = np.squeeze(gamma, axis=0)
            beta = np.squeeze(beta, axis=0)

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 10, 10, C))])
        def prog(x):
            x = mb.transpose(x=x, perm=[0, 3, 1, 2])
            if flip_mul_input_order:
                x = mb.mul(x=gamma, y=x)
            else:
                x = mb.mul(x=x, y=gamma)
            if flip_add_input_order:
                x = mb.add(x=beta, y=x)
            else:
                x = mb.add(x=x, y=beta)
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_elementwise_to_batchnorm"
        )
        assert get_op_types_in_program(prev_prog) == ["transpose", "mul", "add"]
        assert get_op_types_in_program(prog) == ["transpose", "batch_norm"]
        assert_model_is_valid(
            prog,
            {"x": (1, 10, 10, C)},
            expected_output_shapes={block.outputs[0].name: (1, C, 10, 10)},
        )


class TestRank0ExpandDimsSwap:
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

    @pytest.mark.skipif(
        ct.utils._macos_version() < (12, 0), reason="mlprogram predict available only on macOS12+"
    )
    @pytest.mark.parametrize(
        "reverse_order, elem_op",
        itertools.product(
            [True, False],
            ["add", "sub", "mul", "real_div", "floor_div"],
        ),
    )
    def test(self, reverse_order, elem_op):
        x_shape = [
            1,
        ]

        @mb.program(input_specs=[mb.TensorSpec(shape=x_shape)])
        def program(x):
            x = mb.slice_by_index(x=x, begin=[0], end=[1], squeeze_mask=[True])
            func = getattr(mb, elem_op)

            if reverse_order:
                x = func(x=2.0, y=x)
            else:
                x = func(x=x, y=2.0)

            expand = mb.expand_dims(x=x, axes=[0])
            other_1 = mb.add(x=x, y=[1.0, 2.0, 3.0])
            other_2 = mb.sub(x=x, y=[1.0, 2.0, 3.0])
            return expand, other_1, other_2

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            program, "common::rank0_expand_dims_swap"
        )
        assert get_op_types_in_program(prev_prog) == [
            "slice_by_index",
            elem_op,
            "expand_dims",
            "add",
            "sub",
        ]
        assert get_op_types_in_program(program) == [
            "slice_by_index",
            "expand_dims",
            "expand_dims",
            elem_op,
            "squeeze",
            "add",
            "sub",
        ]
        assert_model_is_valid(
            program=program,
            inputs={"x": x_shape},
            expected_output_shapes={
                block.outputs[0].name: tuple(x_shape),
                block.outputs[1].name: (3,),
                block.outputs[2].name: (3,),
            },
        )


class TestImageInputPreprocess(unittest.TestCase):
    """
    Input graph:
    input (format=NHWC) ------> transpose(axis=[0, 3, 1, 2]) ---------> add ----> relu ---> out
                           |                                             ^
                           |                                             |
                           ---> relu ---> transpose(axis=[0, 3, 1, 2]) ---

    Intermediate graph:
    input (format=NCHW) -----> transpose(axis=[0, 2, 3, 1]) ----> transpose(axis=[0, 3, 1, 2]) ---------> add ----> relu ---> out
                                                              |                                             ^
                                                              |                                             |
                                                              ---> relu ---> transpose(axis=[0, 3, 1, 2]) ---


    Output graph:
    input (format=NCHW) -----> relu -----> add -----> relu -----> out
                          |                 ^
                          |                 |
                          -------------------
    """

    def test_fusion_with_image_intermediate_graph(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20, 30, 3))])
        def prog(x):
            x1 = mb.transpose(x=x, perm=[0, 3, 1, 2])
            x2 = mb.relu(x=x)
            x3 = mb.transpose(x=x2, perm=[0, 3, 1, 2])
            x4 = mb.add(x=x1, y=x3)
            return mb.relu(x=x4)

        prog.functions["main"].input_types = [
            ct.ImageType(name="x", shape=(10, 20, 30, 3), channel_first=False)
        ]
        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::image_input_preprocess"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["transpose", "relu", "transpose", "add", "relu"]
        )
        self.assertEqual(
            get_op_types_in_program(prog),
            ["transpose", "transpose", "relu", "transpose", "add", "relu"],
        )

    def test_fusion_with_image_full(self):
        # Avoid circular import
        from coremltools import convert

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20, 30, 3))])
        def prog(x):
            x1 = mb.transpose(x=x, perm=[0, 3, 1, 2])
            x2 = mb.relu(x=x)
            x3 = mb.transpose(x=x2, perm=[0, 3, 1, 2])
            x4 = mb.add(x=x1, y=x3)
            return mb.relu(x=x4)

        mlmodel = convert(
            prog,
            inputs=[ct.ImageType(name="x", shape=(10, 20, 30, 3), channel_first=False)],
            source="milinternal",
            convert_to="neuralnetwork",
        )
        assert mlmodel is not None
        assert len(mlmodel.get_spec().neuralNetwork.layers) == 3


class TestSanitizeInputOutputNames:
    def test_nn_backend_style_sanitization(self):
        """
        Test that intermediate var names are unchanged, and
        only model input and output names are modified, i.e.
        sanitized (adhering to the format [a-zA-Z_][a-zA-Z0-9_]*)
        for the NN backend.
        """

        prog = mil.Program()
        func_inputs = {"x/0": mb.placeholder(shape=[2, 3]), "y": mb.placeholder(shape=[2, 3])}
        with Function(func_inputs) as ssa_fun:
            x, y = ssa_fun.inputs["x/0"], ssa_fun.inputs["y"]
            x = mb.relu(x=x, name="relu/1")
            z = mb.add(x=x, y=y, name="out/1")
            ssa_fun.set_outputs([z])
        prog.add_function("main", ssa_fun)

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog,
            "common::sanitize_input_output_names",
            skip_output_name_check=True,
            skip_input_name_check=True,
        )

        relu_op = prog.find_ops(op_type="relu", exactly_one=True)[0]
        assert relu_op.inputs["x"].name == "x_0"  # input name: sanitized
        assert relu_op.outputs[0].name == "relu/1"  # intermediate name: unchanged
        assert block.outputs[0].name == "out_1"  # output name: sanitized

        # convert prev_prog to NN backend
        mlmodel = ct.convert(prev_prog, convert_to="neuralnetwork")
        spec = mlmodel._spec
        assert spec.description.input[0].name == "x_0"
        assert spec.description.output[0].name == "out_1"
        relu_layer = spec.neuralNetwork.layers[0]
        assert relu_layer.output[0] == "relu/1"


class TestUpdateOutputDtypes:
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

        prog.functions["main"].set_output_types([ct.TensorType(dtype=np.float16)])
        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::update_output_dtypes", skip_output_type_check=True
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

        prog.functions["main"].set_output_types([ct.TensorType(), ct.TensorType(dtype=np.float16)])
        _, _, block = apply_pass_and_basic_check(
            prog, "common::update_output_dtypes", skip_output_type_check=True
        )
        assert get_op_types_in_program(prog) == ["split", "cast"]
        assert block.outputs[1].dtype == types.fp16
        assert block.outputs[1].name == "split_1"

    def test_output_as_input(self, caplog):
        """
        Given:
        -----
        main(%input: (3, fp32)(Tensor)) {
          block0() {
          } -> (input)
        }
        prog.main_output_types = [ct.TensorType(dtype=np.float16)]

        Result:
        Since the output var is also an input var, the dtype is not changed, and a warning message is thrown
        ------
        main(%input: (3, fp32)(Tensor)) {
          block0() {
          } -> (input)
        }

        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3,), dtype=types.fp32)])
        def prog(input):
            return input

        prog.functions["main"].set_output_types([ct.TensorType(dtype=np.float16)])
        _, _, block = apply_pass_and_basic_check(
            prog,
            "common::update_output_dtypes",
        )
        warning_msg = "Output var 'input' is also an input var, hence the dtype cannot be changed: output var 'input' remains dtype fp32"
        assert any([warning_msg in rec.message for rec in caplog.records])
        assert get_op_types_in_program(prog) == []
        assert block.outputs[0].dtype == types.fp32

class TestFuseLayerNormOrInstanceNorm:
    @pytest.mark.parametrize("axes_size", [1, 2, 3])
    def test_layer_norm(self, axes_size):
        """
        Detect layer norm pattern, found in the TF bert model.
        y = x * [gamma * rsqrt(variance + eps)] + (beta - mean * [gamma * rsqrt(variance + eps)])

        where mean and variance are computed along axes [-1] or [-1,-2] and so on
        and gamma and beta are constants with rank equal to the length of the axes parameter.
        """
        shape = (3, 5, 6)
        rank = len(shape)
        axes = list(range(rank - axes_size, rank))

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            x1 = mb.reduce_mean(x=x, axes=axes, keep_dims=True)
            x2 = mb.sub(x=x, y=x1)
            x2 = mb.square(x=x2)
            x2 = mb.reduce_mean(x=x2, axes=axes, keep_dims=True)
            x2 = mb.add(x=x2, y=1e-5)
            x2 = mb.rsqrt(x=x2)
            x3 = mb.mul(x=np.random.rand(*shape[-len(axes) :]), y=x2)
            x4 = mb.mul(x=x3, y=x1)
            x5 = mb.mul(x=x, y=x3)
            x4 = mb.sub(x=np.random.rand(*shape[-len(axes) :]), y=x4)
            y = mb.add(x=x4, y=x5)
            return y

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_layernorm_or_instancenorm"
        )
        assert get_op_types_in_program(prev_prog) == [
            "reduce_mean",
            "sub",
            "square",
            "reduce_mean",
            "add",
            "rsqrt",
            "mul",
            "mul",
            "mul",
            "sub",
            "add",
        ]
        assert get_op_types_in_program(prog) == ["layer_norm"]
        assert_model_is_valid(
            prog, {"x": shape}, expected_output_shapes={block.outputs[0].name: shape}
        )

    @pytest.mark.parametrize("with_affine", [True, False])
    def test_ane_layer_norm(self, with_affine):
        """
        Detect layer norm pattern, found in models based on ml-ane-transformers.

        ``y = (x - mean(x)) * rsqrt(mean((x - mean(x))^2) + eps)``
        ``y = [(x - mean(x)) * rsqrt(mean((x - mean(x))^2) + eps) + beta] * gamma``

        Note that beta and gamma in these equations are applied in
        opposite order compared to the MIL.

        Only applies when mean and variance are computed along axes [1]
        """
        shape = (3, 5, 1, 6)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            x1 = mb.reduce_mean(x=x, axes=[1], keep_dims=True) # mean
            x2 = mb.sub(x=x, y=x1) # x - mean
            x3 = mb.mul(x=x2, y=x2) # (x - mean)^2
            x4 = mb.reduce_mean(x=x3, axes=[1], keep_dims=True) # variance
            x5 = mb.add(x=x4, y=1e-5) # variance + eps
            x6 = mb.rsqrt(x=x5) # rsqrt(variance + eps)
            y = mb.mul(x=x2, y=x6) # (x - mean) * rsqrt(variance + eps)

            if with_affine:
                y = mb.add(x=y, y=np.random.rand(1, shape[1], 1, 1))
                y = mb.mul(x=y, y=np.random.rand(1, shape[1], 1, 1))

            return y

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_layernorm_or_instancenorm"
        )
        assert get_op_types_in_program(prev_prog) == [
            "reduce_mean",
            "sub",
            "mul",
            "reduce_mean",
            "add",
            "rsqrt",
            "mul",
        ] + (["add", "mul"] if with_affine else [])
        assert get_op_types_in_program(prog) == ["layer_norm"]
        assert_model_is_valid(
            prog, {"x": shape}, expected_output_shapes={block.outputs[0].name: shape}
        )

    @pytest.mark.parametrize("with_affine", [True, False])
    def test_ane_layer_norm_root_var_reuse(self, with_affine):
        """
        Detect layer norm pattern, found in models based on ml-ane-transformers.

        Cover cases where the input to the layer norm is used as input to an
        op that occurs after the layer norm.

        ``y = (x - mean(x)) * rsqrt(mean((x - mean(x))^2) + eps)``
        ``y = [(x - mean(x)) * rsqrt(mean((x - mean(x))^2) + eps) + beta] * gamma``

        Note that beta and gamma in these equations are applied in
        opposite order compared to the MIL.

        Only applies when mean and variance are computed along axes [1]

        """
        shape = (3, 5, 1, 6)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            x1 = mb.add(x=x, y=np.random.rand(*shape))
            x2 = mb.reduce_mean(x=x1, axes=[1], keep_dims=True) # mean
            x3 = mb.sub(x=x1, y=x2) # x - mean
            x4 = mb.mul(x=x3, y=x3) # (x - mean)^2
            x5 = mb.reduce_mean(x=x4, axes=[1], keep_dims=True) # variance
            x6 = mb.add(x=x5, y=1e-5) # variance + eps
            x7 = mb.rsqrt(x=x6) # rsqrt(variance + eps)
            y = mb.mul(x=x3, y=x7) # (x - mean) * rsqrt(variance + eps)

            if with_affine:
                y = mb.add(x=y, y=np.random.rand(1, shape[1], 1, 1))
                y = mb.mul(x=y, y=np.random.rand(1, shape[1], 1, 1))

            y = mb.sub(x=x, y=y) # use x for something after the norm
            return y

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_layernorm_or_instancenorm"
        )
        assert get_op_types_in_program(prev_prog) == [
            "add",
            "reduce_mean",
            "sub",
            "mul",
            "reduce_mean",
            "add",
            "rsqrt",
            "mul",
        ] + (["add", "mul"] if with_affine else []) + ["sub"]
        assert get_op_types_in_program(prog) == ["add", "layer_norm", "sub"]
        assert_model_is_valid(
            prog, {"x": shape}, expected_output_shapes={block.outputs[0].name: shape}
        )

    @pytest.mark.parametrize(
        "with_affine, reused_name",
        itertools.chain(
            itertools.product(
                [True, False],
                [None, "x2", "x3", "x4", "x8"],
            ),
            [[True, "x9"]]
        ),
    )
    def test_ane_layer_norm_intermediate_var_reuse(self, with_affine, reused_name):
        """
        Avoid false positive detection of ml-ane-transformers layer norm pattern.

        In cases where an intermediate value is used after the layer norm
        is computed, the pattern should not be fused.
        """
        shape = (3, 5, 1, 6)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            x1 = mb.add(x=x, y=np.random.rand(*shape))
            x2 = mb.reduce_mean(x=x1, axes=[1], keep_dims=True) # mean
            x3 = mb.sub(x=x1, y=x2) # x - mean
            x4 = mb.mul(x=x3, y=x3) # (x - mean)^2
            x5 = mb.reduce_mean(x=x4, axes=[1], keep_dims=True) # variance
            x6 = mb.add(x=x5, y=1e-5) # variance + eps
            x7 = mb.rsqrt(x=x6) # rsqrt(variance + eps)
            x8 = mb.mul(x=x3, y=x7) # (x - mean) * rsqrt(variance + eps)
            y = x8

            if with_affine:
                x9 = mb.add(x=y, y=np.random.rand(1, shape[1], 1, 1))
                y = x9
                y = mb.mul(x=y, y=np.random.rand(1, shape[1], 1, 1))

            # All the same shape (3,5,1,6)
            reused = None
            if reused_name == "x2":
                reused = x2
            elif reused_name == "x3":
                reused = x3
            elif reused_name == "x4":
                reused = x4
            elif reused_name == "x8":
                reused = x8
            elif reused_name == "x9":
                reused = x9

            if reused:
                y = mb.sub(x=reused, y=y) # reuse an intermediate variable

            return y

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_layernorm_or_instancenorm"
        )
        if reused_name == "x8" and not with_affine:
            # This can be fused since without affine, x8 is the final op in the layer norm.
            assert get_op_types_in_program(prog) == ["add", "layer_norm", "sub"]
        elif reused_name:
            assert "layer_norm" not in get_op_types_in_program(prog)
            assert get_op_types_in_program(prog)[-1] ==  "sub"
        else:
            # Fusion should still work when nothing is reused.
            assert get_op_types_in_program(prog) == ["add", "layer_norm"]

        assert_model_is_valid(
            prog, {"x": shape}, expected_output_shapes={block.outputs[0].name: shape}
        )

    def test_ane_layer_norm_ambiguous_add(self):
        """
        Avoid false positive detection of ml-ane-transformers layer norm pattern.

        In cases where the pattern has an add that could be beta,
        but it does not feed into a gamma mul, it is ambiguous if the
        pattern should be fused so don't.
        """
        shape = (3, 5, 1, 6)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            x1 = mb.add(x=x, y=np.random.rand(*shape))
            x2 = mb.reduce_mean(x=x1, axes=[1], keep_dims=True) # mean
            x3 = mb.sub(x=x1, y=x2) # x - mean
            x4 = mb.mul(x=x3, y=x3) # (x - mean)^2
            x5 = mb.reduce_mean(x=x4, axes=[1], keep_dims=True) # variance
            x6 = mb.add(x=x5, y=1e-5) # variance + eps
            x7 = mb.rsqrt(x=x6) # rsqrt(variance + eps)
            x8 = mb.mul(x=x3, y=x7) # (x - mean) * rsqrt(variance + eps)
            y = mb.add(x=x8, y=np.random.rand(1, shape[1], 1, 1)) # ambiguous add (maybe beta)
            return y

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_layernorm_or_instancenorm"
        )
        assert get_op_types_in_program(prev_prog) == [
            "add",
            "reduce_mean",
            "sub",
            "mul",
            "reduce_mean",
            "add",
            "rsqrt",
            "mul",
            "add",
        ]
        assert "layer_norm" not in get_op_types_in_program(prog)
        assert_model_is_valid(
            prog, {"x": shape}, expected_output_shapes={block.outputs[0].name: shape}
        )

    @pytest.mark.parametrize(
        "first_axes, second_axes",
        [
            [[0], [1]],
            [[1], [0]],
            [[2], [1]],
            [[1], [2]],
            [[1,2], [1]],
            [[1], [1,2]],
        ]
    )
    def test_ane_layer_norm_wrong_axes(self, first_axes, second_axes):
        """
        Avoid false positive detection of ml-ane-transformers layer norm pattern.

        In cases where the axes != [1] the pattern should not be fused.
        """
        shape = (1, 1, 1, 6)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            x1 = mb.add(x=x, y=np.random.rand(*shape))
            x2 = mb.reduce_mean(x=x1, axes=first_axes, keep_dims=True) # mean
            x3 = mb.sub(x=x1, y=x2) # x - mean
            x4 = mb.mul(x=x3, y=x3) # (x - mean)^2
            x5 = mb.reduce_mean(x=x4, axes=second_axes, keep_dims=True) # variance
            x6 = mb.add(x=x5, y=1e-5) # variance + eps
            x7 = mb.rsqrt(x=x6) # rsqrt(variance + eps)
            y = mb.mul(x=x3, y=x7) # (x - mean) * rsqrt(variance + eps)
            return y

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_layernorm_or_instancenorm"
        )
        assert get_op_types_in_program(prev_prog) == [
            "add",
            "reduce_mean",
            "sub",
            "mul",
            "reduce_mean",
            "add",
            "rsqrt",
            "mul",
        ]
        assert "layer_norm" not in get_op_types_in_program(prog)
        assert_model_is_valid(
            prog, {"x": shape}, expected_output_shapes={block.outputs[0].name: shape}
        )

    def test_instance_norm_pattern_1(self):
        """
        Detect instance norm pattern
        y = x * [gamma * rsqrt(variance + eps)] + (beta - mean * [gamma * rsqrt(variance + eps)])

        where input is rank 4, (N,C,H,W), axis=[2, 3], along which reduction happens,
        and gamma and beta are of shape (1,C,1,1)
        """
        shape = (3, 5, 6, 7)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            x1 = mb.reduce_mean(x=x, axes=[2, 3], keep_dims=True)
            x2 = mb.sub(x=x, y=x1)
            x2 = mb.square(x=x2)
            x2 = mb.reduce_mean(x=x2, axes=[2, 3], keep_dims=True)
            x2 = mb.add(x=x2, y=1e-5)
            x2 = mb.rsqrt(x=x2)
            x3 = mb.mul(x=np.random.rand(1, shape[1], 1, 1), y=x2)
            x4 = mb.mul(x=x3, y=x1)
            x5 = mb.mul(x=x, y=x3)
            x4 = mb.sub(x=np.random.rand(1, shape[1], 1, 1), y=x4)
            y = mb.add(x=x4, y=x5)
            return y

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_layernorm_or_instancenorm"
        )
        assert get_op_types_in_program(prev_prog) == [
            "reduce_mean",
            "sub",
            "square",
            "reduce_mean",
            "add",
            "rsqrt",
            "mul",
            "mul",
            "mul",
            "sub",
            "add",
        ]
        assert get_op_types_in_program(prog) == ["instance_norm"]
        assert_model_is_valid(
            prog, {"x": shape}, expected_output_shapes={block.outputs[0].name: shape}
        )

    def test_instance_norm_pattern_1_rank_1_gamma_beta(self):
        """
        Detect instance norm pattern
        y = x * [gamma * rsqrt(variance + eps)] + (beta - mean * [gamma * rsqrt(variance + eps)])

        where input is rank 4, (N,C,H,W), axis=[2, 3], along which reduction happens,
        and gamma and beta are of shape (C,)
        """
        shape = (3, 5, 6, 7)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            x1 = mb.reduce_mean(x=x, axes=[1, 2], keep_dims=True)
            x2 = mb.sub(x=x, y=x1)
            x2 = mb.square(x=x2)
            x2 = mb.reduce_mean(x=x2, axes=[1, 2], keep_dims=True)
            x2 = mb.add(x=x2, y=1e-5)
            x2 = mb.rsqrt(x=x2)
            x3 = mb.mul(x=np.random.rand(shape[3]), y=x2)
            x4 = mb.mul(x=x3, y=x1)
            x5 = mb.mul(x=x, y=x3)
            x4 = mb.sub(x=np.random.rand(shape[3]), y=x4)
            y = mb.add(x=x4, y=x5)
            return y

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_layernorm_or_instancenorm"
        )
        assert get_op_types_in_program(prev_prog) == [
            "reduce_mean",
            "sub",
            "square",
            "reduce_mean",
            "add",
            "rsqrt",
            "mul",
            "mul",
            "mul",
            "sub",
            "add",
        ]
        assert get_op_types_in_program(prog) == ["transpose", "instance_norm", "transpose"]
        assert_model_is_valid(
            prog, {"x": shape}, expected_output_shapes={block.outputs[0].name: shape}
        )

    def test_instance_norm_pattern_1_with_channel_last_data_format(self):
        """
        Detect instance norm pattern with channel last data format
        x = transpose(x) # channel first to channel last, NCHW -> NHWC
        x = x * [gamma * rsqrt(variance + eps)] + (beta - mean * [gamma * rsqrt(variance + eps)])
        x = transpose(x) # channel last to channel first, NHWC -> NCHW

        The input is rank 4 (N, C, H, W) and the input for fused "instance_norm" op is
        rank 4 (N, H, W, C), and axis=[1, 2] or [-3, -2], along which reduction happens.

        This is common in TensorFlow model when data format is channel last.
        PyMIL inserts transposes around "conv" layer to make "conv" channel first.
        "fuse_layernorm_or_instancenorm" pass is expected to fuse this pattern as well.
        """
        shape = (1, 3, 5, 5)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            x = mb.transpose(x=x, perm=[0, 2, 3, 1])
            x1 = mb.reduce_mean(x=x, axes=[1, 2], keep_dims=True)
            x2 = mb.sub(x=x, y=x1)
            x2 = mb.square(x=x2)
            x2 = mb.reduce_mean(x=x2, axes=[1, 2], keep_dims=True)
            x2 = mb.add(x=x2, y=1e-5)
            x2 = mb.rsqrt(x=x2)
            x3 = mb.mul(x=np.random.rand(1, 1, 1, shape[1]), y=x2)
            x4 = mb.mul(x=x3, y=x1)
            x5 = mb.mul(x=x, y=x3)
            x4 = mb.sub(x=np.random.rand(1, 1, 1, shape[1]), y=x4)
            x6 = mb.add(x=x4, y=x5)
            y = mb.transpose(x=x6, perm=[0, 3, 1, 2])
            return y

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_layernorm_or_instancenorm"
        )
        assert get_op_types_in_program(prev_prog) == [
            "transpose",
            "reduce_mean",
            "sub",
            "square",
            "reduce_mean",
            "add",
            "rsqrt",
            "mul",
            "mul",
            "mul",
            "sub",
            "add",
            "transpose",
        ]
        assert get_op_types_in_program(prog) == [
            "transpose",
            "transpose",
            "instance_norm",
            "transpose",
            "transpose",
        ]
        assert_model_is_valid(
            prog,
            {"x": shape},
            expected_output_shapes={block.outputs[0].name: shape},
        )
        # reduce transpose pass should remove extra ones
        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::reduce_transposes")
        assert get_op_types_in_program(prog) == ["instance_norm"]
        assert_model_is_valid(
            prog,
            {"x": shape},
            expected_output_shapes={block.outputs[0].name: shape},
        )

    def test_instance_norm_pattern_2(self):
        """
        Detect instance norm pattern 2 and fusion.

        |----> sub0 ----|                            const (0.5)
        |       ^       |                                |
        |       |       V                                V
        x ---> mean0  square --> mean1 --> add_eps ---> pow       const_gamma   const_beta
        |       |                                        |             |            |
        |       V                                        V             V            V
        |----> sub1 --------------------------------> real_div --> mul_gamma --> add_beta --> ...
        """
        shape = (3, 5, 6, 7)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            mean0 = mb.reduce_mean(x=x, axes=[2, 3], keep_dims=True)
            sub0 = mb.sub(x=x, y=mean0)
            sub1 = mb.sub(x=x, y=mean0)
            square = mb.square(x=sub0)
            mean1 = mb.reduce_mean(x=square, axes=[2, 3], keep_dims=True)
            add_eps = mb.add(x=mean1, y=1e-5)  # epsilon
            pow = mb.pow(x=add_eps, y=0.5)
            div = mb.real_div(x=sub1, y=pow)
            mul_gamma = mb.mul(x=np.random.rand(1, shape[1], 1, 1), y=div)  #
            add_beta = mb.add(x=np.random.rand(1, shape[1], 1, 1), y=mul_gamma)
            return add_beta

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_layernorm_or_instancenorm"
        )
        assert get_op_types_in_program(prev_prog) == [
            "reduce_mean",
            "sub",
            "sub",
            "square",
            "reduce_mean",
            "add",
            "pow",
            "real_div",
            "mul",
            "add",
        ]
        assert get_op_types_in_program(prog) == ["instance_norm"]
        assert_model_is_valid(
            prog, {"x": shape}, expected_output_shapes={block.outputs[0].name: shape}
        )

    def test_instance_norm_pattern_3(self):
        """
        Detect and fuse instance norm pattern 3 (pattern in TensorFlow-Addons).

               |-------------------------------------------------|
               |                                                 |
               |                                                 V
        x --> mean   square --> mean1 --> add_eps --> rsqrt --> mul2 --> mul_sub
        |      |       ^                                |                   |
        |      V       |                                |                   |
        | --> sub -----|                                |                   |
        |                                               V                   V
        |--------------------------------------------> mul1 -------------> add --> ...
        """
        shape = (3, 5, 6, 7)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            mean0 = mb.reduce_mean(x=x, axes=[2, 3], keep_dims=True)
            sub = mb.sub(x=x, y=mean0)
            square = mb.square(x=sub)
            mean1 = mb.reduce_mean(x=square, axes=[2, 3], keep_dims=True)
            add_eps = mb.add(x=mean1, y=1e-5)  # epsilon
            rsqrt = mb.rsqrt(x=add_eps)
            mul1 = mb.mul(x=rsqrt, y=x)
            mul2 = mb.mul(x=mean0, y=rsqrt)
            mul_sub = mb.mul(x=mul2, y=-1.0)
            add = mb.add(x=mul1, y=mul_sub)
            return add

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_layernorm_or_instancenorm"
        )
        assert get_op_types_in_program(prev_prog) == [
            "reduce_mean",
            "sub",
            "square",
            "reduce_mean",
            "add",
            "rsqrt",
            "mul",
            "mul",
            "mul",
            "add",
        ]
        assert get_op_types_in_program(prog) == ["instance_norm"]
        assert_model_is_valid(
            prog, {"x": shape}, expected_output_shapes={block.outputs[0].name: shape}
        )

    def test_instance_norm_pattern_4(self):
        """
        Detect and fuse instance norm pattern 4.

        |-----------|
        |           V
        |------> mul_square1 -----> sum1 -----> mul_mean1
        |                                           |
        |                                           V
        x --> sum --> mul_mean ==> mul_square --> sub_variance --> add_eps --> rsqrt
        |                |                                                      |
        |                |                                                      V
        |                |                                                  mul_gamma
        |                |                                                      |
        |                |                                            |----------------|
        |                |                                            |                V
        |                |--------------------------------------------+-------------> mul2
        |                                                             V                |
        |----------------------------------------------------------> mul1              |
                                                                      |                V
                                                                      |             sub_beta --> add --> [...]
                                                                      |                           ^
                                                                      |---------------------------|
        """
        shape = (3, 5, 6, 7)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            mul_square1 = mb.mul(x=x, y=x)
            sum = mb.reduce_sum(x=x, axes=[2, 3], keep_dims=True)
            mul_mean = mb.mul(x=sum, y=3.3333334e-05)  # dummy value here
            mul_square = mb.mul(x=mul_mean, y=mul_mean)
            sum1 = mb.reduce_sum(x=mul_square1, axes=[2, 3], keep_dims=True)
            mul_mean1 = mb.mul(x=sum1, y=8.333333e-06)  # dummy value here
            sub_variance = mb.sub(x=mul_mean1, y=mul_square)
            add_eps = mb.add(x=sub_variance, y=1e-5)  # epsilon
            rsqrt = mb.rsqrt(x=add_eps)
            mul_gamma = mb.mul(x=rsqrt, y=np.random.rand(1, shape[1], 1, 1))
            mul1 = mb.mul(x=mul_gamma, y=x)
            mul2 = mb.mul(x=mul_mean, y=mul_gamma)
            sub_beta = mb.sub(x=np.random.rand(1, shape[1], 1, 1), y=mul2)
            add = mb.add(x=mul1, y=sub_beta)
            return add

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_layernorm_or_instancenorm"
        )
        assert get_op_types_in_program(prev_prog) == [
            "mul",
            "reduce_sum",
            "mul",
            "mul",
            "reduce_sum",
            "mul",
            "sub",
            "add",
            "rsqrt",
            "mul",
            "mul",
            "mul",
            "sub",
            "add",
        ]
        assert get_op_types_in_program(prog) == ["instance_norm"]
        assert_model_is_valid(
            prog, {"x": shape}, expected_output_shapes={block.outputs[0].name: shape}
        )

class TestFuseLinearBias:
    @staticmethod
    def _apply_transform(inputs, func, is_first_input, has_bias):
        """
        Utility function to test the weight/bias transform function in linear bias fusion pass.
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4))])
        def prog(x):

            if has_bias:
                linear = mb.linear(
                    x=x,
                    weight=inputs["linear_weight"],
                    bias=inputs["linear_bias"],
                )
            else:
                linear = mb.linear(
                    x=x,
                    weight=inputs["linear_weight"],
                )

            if is_first_input:
                kwargs = {
                    "x": linear,
                    "y": inputs["bias"],
                }
            else:
                kwargs = {
                    "x": inputs["bias"],
                    "y": linear,
                }

            x = func(**kwargs)
            return x

        apply_pass_and_basic_check(
            prog,
            "common::fuse_linear_bias",
        )

        # get the updated weight from the prog
        linear_op = []
        for op in prog["main"].operations:
            if op.op_type == "const":
                continue
            linear_op.append(op)
        assert len(linear_op) == 1, "should only have one linear layer."

        return linear_op[0].weight.val, linear_op[0].bias.val

    @pytest.mark.parametrize(
        "op_type, is_first_input, has_bias, broadcast",
        itertools.product(
            ["add", "sub"],
            [True, False],
            [True, False],
            [True, False],
        ),
    )
    def test_transform_linear(self, op_type, is_first_input, has_bias, broadcast):
        """
        Test the weight / bias transform function in the linear bias fusion pass
        """
        weight = np.reshape(np.arange(8), (2, 4)).astype(np.float32)
        linear_bias = (
            np.array([1, 2]).astype(np.float32) if has_bias else np.array([0, 0]).astype(np.float32)
        )
        bias = np.array([3, 4]).astype(np.float32)
        if broadcast:
            bias = np.reshape(bias, (1, 2))

        inputs = {
            "linear_weight": weight,
            "linear_bias": linear_bias,
            "bias": bias,
        }

        if op_type == "add":
            func = mb.add
        elif op_type == "sub":
            func = mb.sub

        new_weight, new_bias = self._apply_transform(
            inputs,
            func,
            is_first_input,
            has_bias,
        )
        if broadcast:
            bias = np.reshape(bias, (2,))

        if op_type == "sub" and not is_first_input:
            expected_weight = -weight
        else:
            expected_weight = weight

        if op_type == "sub":
            if is_first_input:
                expected_bias = linear_bias - bias
            else:
                expected_bias = bias - linear_bias
        else:
            expected_bias = linear_bias + bias

        np.testing.assert_almost_equal(new_weight, expected_weight)
        np.testing.assert_almost_equal(new_bias, expected_bias)

    @pytest.mark.parametrize(
        "rank, op_type, is_first_input, broadcast, backend",
        itertools.product([1, 2, 3], ["add", "sub"], [True, False], [True, False], backends),
    )
    def test_linear_bias_fusion(self, rank, op_type, is_first_input, broadcast, backend):
        """
        Input graph:
                                    Const
                                      |
                                      V
        input -----> linear -----> add/sub ---> out

        Output graph:
        input -----> linear ----> out
        """
        input_shape = [1, 2, 3]
        input_shape = input_shape[-rank:]
        input_shape = tuple(input_shape)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
        def prog(x):
            linear_weight = np.reshape(np.arange(6), (2, 3)).astype(np.float32)
            linear_bias = np.array([1.0, 2.0])
            bias = np.array([3.0, 4.0])
            if broadcast:
                if rank >= 2:
                    bias = np.reshape(bias, (1, 2))

            x = mb.linear(
                x=x,
                weight=linear_weight,
                bias=linear_bias,
            )

            func = mb.add if op_type == "add" else mb.sub
            if is_first_input:
                kwargs = {
                    "x": x,
                    "y": bias,
                }
            else:
                kwargs = {
                    "x": bias,
                    "y": x,
                }
            x = func(**kwargs)
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_linear_bias")

        assert get_op_types_in_program(prev_prog) == ["linear", op_type]
        assert get_op_types_in_program(prog) == ["linear"]

        # validate graph pass
        output_shape = [1, 2, 2]
        output_shape = tuple(output_shape[-rank:])
        assert_model_is_valid(
            prog,
            {"x": input_shape},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=backend,
        )


class TestFuseMatmulWeightBias:
    def test_fuse_matmul_weight_bias(self):
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

        if _VALIDATE_MODEL:
            assert_model_is_valid(prog, {"x": (2, 4)})


class TestGraphPassScopePreservation:
    def test_single_pass(self):
        """
        Input:

            x
            -> relu(torch_scope="module_1")
            -> transpose_1(torch_scope="module_1")
            -> transpose_2(torch_scope="module_2")
            -> output

        Output:

            x
            -> relu(torch_scope="module_1")
            -> transpose_3(
                    torch_scope="module_2",
                    pass_scope="merge_consecutive_transposes"
                )
            -> output

        In the above case, the relu op preserves its original scope information.
        Since transpose_3 is created by the "merge_consecutive_transposes" pass, the COREMLTOOLS_GRAPH_PASS scope
        information will be saved in the op.
        Also, the TORCHSCRIPT_MODULE_TYPE scope info of transpose_2 is back propagated to transpose_3,
        when the use of output of transpose_2 is replaced by the output of transpose_3.
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x):
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="module_1"),
            ):
                x = mb.relu(x=x)
                x = mb.transpose(x=x, perm=[0, 2, 1, 3])

            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="module_2"),
            ):
                return mb.transpose(x=x, perm=[3, 2, 0, 1])

        prog._add_essential_scope_source(ScopeSource.TORCHSCRIPT_MODULE_TYPE)

        apply_pass_and_basic_check(prog, "common::merge_consecutive_transposes")
        assert get_op_types_in_program(prog) == ["relu", "transpose"]

        # the scope info in the relu op is not affected
        relu_op = prog.find_ops(op_type="relu")[0]
        assert len(relu_op.scopes) == 1
        assert relu_op.scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_1"]

        # the new transpose op has the scope information from the graph pass
        transpose_op = prog.find_ops(op_type="transpose")[0]
        assert len(transpose_op.scopes) == 2
        assert transpose_op.scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS] == [
            "merge_consecutive_transposes"
        ]
        assert transpose_op.scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_2"]

    def test_single_pass_without_creating_new_var(self):
        """
        Input:

            x
            -> relu(torch_scope="module_1")
            -> relu(torch_scope="module_2")
            -> relu(torch_scope="module_3")
            -> output

        Output:

            x
            -> relu(torch_scope="module_1")
            -> output

        In the above case, the relu op preserves its original scope information, since the graph pass only reconnects the graph.
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x):
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="module_1"),
            ):
                x = mb.relu(x=x)

            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="module_2"),
            ):
                x = mb.relu(x=x)

            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="module_3"),
            ):
                return mb.relu(x=x)

        prog._add_essential_scope_source(ScopeSource.TORCHSCRIPT_MODULE_TYPE)

        apply_pass_and_basic_check(prog, "common::merge_consecutive_relus")
        assert get_op_types_in_program(prog) == ["relu"]

        # the scope info in the relu op is not affected
        relu_op = prog.find_ops(op_type="relu")[0]
        assert len(relu_op.scopes) == 1
        assert relu_op.scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_1"]

    def test_multiple_passes(self):
        """
        In this case, a program goes through two graph passes.
        And the resulting program should have scope information from both passes.
        """
        shape = (3, 5, 6, 7)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="module_1"),
            ):
                # dummy op
                x = mb.relu(x=x)

                # pattern for "merge_consecutive_transposes"
                x = mb.transpose(x=x, perm=[0, 2, 1, 3])

            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="module_2"),
            ):
                x = mb.transpose(x=x, perm=[3, 2, 0, 1])

                # pattern for "fuse_layernorm_or_instancenorm"
                mean0 = mb.reduce_mean(x=x, axes=[2, 3], keep_dims=True)
                sub0 = mb.sub(x=x, y=mean0)
                sub1 = mb.sub(x=x, y=mean0)
                square = mb.square(x=sub0)
                mean1 = mb.reduce_mean(x=square, axes=[2, 3], keep_dims=True)
                add_eps = mb.add(x=mean1, y=1e-5)  # epsilon
                pow = mb.pow(x=add_eps, y=0.5)
                div = mb.real_div(x=sub1, y=pow)

            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="module_3"),
            ):
                mul_gamma = mb.mul(x=np.random.rand(1, shape[1], 1, 1), y=div)
                return mb.add(x=np.random.rand(1, shape[1], 1, 1), y=mul_gamma)

        prog._add_essential_scope_source(ScopeSource.TORCHSCRIPT_MODULE_TYPE)

        apply_pass_and_basic_check(prog, "common::fuse_layernorm_or_instancenorm")
        apply_pass_and_basic_check(prog, "common::merge_consecutive_transposes")
        assert get_op_types_in_program(prog) == ["relu", "transpose", "instance_norm"]

        # the scope info in the relu op is not affected
        relu_op = prog.find_ops(op_type="relu")[0]
        assert len(relu_op.scopes) == 1
        assert relu_op.scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_1"]

        # the new transpose op has the scope information from the graph pass
        transpose_op = prog.find_ops(op_type="transpose")[0]
        assert len(transpose_op.scopes) == 2
        assert transpose_op.scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS] == [
            "merge_consecutive_transposes"
        ]
        assert transpose_op.scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_2"]

        # the new instance_norm op has the scope information from the graph pass
        instance_norm_op = prog.find_ops(op_type="instance_norm")[0]
        assert len(instance_norm_op.scopes) == 2
        assert instance_norm_op.scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS] == [
            "fuse_layernorm_or_instancenorm"
        ]
        assert instance_norm_op.scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_3"]

    def test_fp16_scope_preservation(self):
        """
        This test explains step-by-step how the scope information preservation works in the fp32 -> fp16 pass.

        Input graph:

            x(fp32)
            -> relu(torch_scope="module_1")
            -> sin(torch_scope="module_2")
            -> output(fp32)

        (1) "common::add_fp16_cast"

            First, in the add_fp16_cast graph pass, multiple cast ops are injected in the graph:

                x(fp32)
                -> cast(dtype="fp16", torch_scope="module_1", pass_scope="add_fp16_cast")
                -> relu(torch_scope="module_1", pass_scope="add_fp16_cast")
                -> cast(dtype="fp32", torch_scope="module_1", pass_scope="add_fp16_cast")
                -> cast(dtype="fp16", torch_scope="module_2", pass_scope="add_fp16_cast")
                -> sin(torch_scope="module_2", pass_scope="add_fp16_cast")
                -> cast(dtype="fp32, torch_scope="module_2", pass_scope="add_fp16_cast")
                -> output

            There are 4 cast ops in the graph who has pass_scope = "add_fp16_cast", which indicates they are added by the "add_fp16_cast" pass.

            Note that, the first cast -> relu -> cast pattern has the same torch scope information as
            the original relu(torch_scope="module_1"). This is due to the fact that when we replace
            the use of the original relu output with the output of the second cast op, the scope information is back propagated.

            The same reason applied for why the cast -> sin -> cast patterns has the torch scope as
            the original sin op.

        (2) "common::cast_optimization" + "dead-code_elimination"

            After the cleanup, the graph becomes:

                x(fp32)
                -> cast(
                        dtype="fp16",
                        torch_scope="module_1",
                        pass_scope="add_fp16_cast"
                    )
                -> relu(
                        torch_scope="module_1",
                        pass_scope="add_fp16_cast]
                    )
                -> sin(
                        torch_scope="module_2",
                        pass_scope="add_fp16_cast"
                    )
                -> cast(
                        dtype="fp32,
                        torch_scope="module_2",
                        pass_scope="add_fp16_cast"
                    )
                -> output

            We can see that, the fp16 version of relu / sin preserves the original torch scope information.
        """
        shape = (3, 5, 6, 7)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="module_1"),
            ):
                x = mb.relu(x=x)
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="module_2"),
            ):
                return mb.sin(x=x)

        prog._add_essential_scope_source(ScopeSource.TORCHSCRIPT_MODULE_TYPE)

        # fp16 cast pass
        apply_pass_and_basic_check(prog, "common::add_fp16_cast")
        assert get_op_types_in_program(prog) == ["cast", "relu", "cast", "cast", "sin", "cast"]

        cast_ops = prog.find_ops(op_type="cast")
        assert len(cast_ops) == 4
        assert len(cast_ops[0].scopes) == 2
        assert cast_ops[0].scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS] == ["add_fp16_cast"]
        assert cast_ops[0].scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_1"]
        assert len(cast_ops[1].scopes) == 2
        assert cast_ops[1].scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS] == ["add_fp16_cast"]
        assert cast_ops[1].scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_1"]
        assert len(cast_ops[2].scopes) == 2
        assert cast_ops[2].scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS] == [
            "add_fp16_cast",
        ]
        assert cast_ops[2].scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_2"]
        assert len(cast_ops[3].scopes) == 2
        assert cast_ops[3].scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS] == ["add_fp16_cast"]
        assert cast_ops[3].scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_2"]

        relu_op = prog.find_ops(op_type="relu")[0]
        assert len(relu_op.scopes) == 2
        assert relu_op.scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS] == ["add_fp16_cast"]
        assert relu_op.scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_1"]

        sin_op = prog.find_ops(op_type="sin")[0]
        assert len(sin_op.scopes) == 2
        assert sin_op.scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS] == ["add_fp16_cast"]
        assert sin_op.scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_2"]

        # clean up with cast optimization and dead code elimination
        apply_pass_and_basic_check(prog, "common::cast_optimization")
        apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["cast", "relu", "sin", "cast"]
        cast_ops = prog.find_ops(op_type="cast")
        assert len(cast_ops) == 2
        assert len(cast_ops[0].scopes) == 2
        assert cast_ops[0].scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS] == ["add_fp16_cast"]
        assert cast_ops[0].scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_1"]
        assert len(cast_ops[1].scopes) == 2
        assert cast_ops[1].scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS] == ["add_fp16_cast"]
        assert cast_ops[1].scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_2"]

        relu_op = prog.find_ops(op_type="relu")[0]
        assert len(relu_op.scopes) == 2
        assert relu_op.scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS] == [
            "add_fp16_cast",
        ]
        assert relu_op.scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_1"]

        sin_op = prog.find_ops(op_type="sin")[0]
        assert len(sin_op.scopes) == 2
        assert sin_op.scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS] == ["add_fp16_cast"]
        assert sin_op.scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_2"]

    def test_pass_followed_by_fp16(self):
        """
        Input:

            x
            -> transpose_1(torch_scope="module_1")
            -> transpose_2(torch_scope="module_2")
            -> output

        Output:

            x
            -> cast(
                    dtype="fp16",
                    torch_scope="module_2",
                    pass_scope=["merge_consecutive_transposes", "add_fp16_cast"]
               )
            -> transpose_3_fp16(
                    torch_scope="module_2",
                    pass_scope=["merge_consecutive_transposes", "add_fp16_cast"]
               )
            -> cast(dtype="fp32",
                    torch_scope="module_2",
                    pass_scope=["merge_consecutive_transposes", "add_fp16_cast"]
               )
            -> output

        In the above case, two transpose ops first merged into a single transpose op,
        and the graph is transformed into fp16.

        Hence, the final transpose op should have scope information from both graph passes.
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x):
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="module_1"),
            ):
                x = mb.transpose(x=x, perm=[0, 2, 1, 3])
            with mb.scope(
                ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data="module_2"),
            ):
                return mb.transpose(x=x, perm=[3, 2, 0, 1])

        prog._add_essential_scope_source(ScopeSource.TORCHSCRIPT_MODULE_TYPE)

        apply_pass_and_basic_check(prog, "common::merge_consecutive_transposes")
        apply_pass_and_basic_check(prog, "common::add_fp16_cast")
        apply_pass_and_basic_check(prog, "common::cast_optimization")
        apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["cast", "transpose", "cast"]

        cast_ops = prog.find_ops(op_type="cast")
        assert len(cast_ops) == 2
        assert len(cast_ops[0].scopes) == 2
        assert cast_ops[0].scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS] == [
            "merge_consecutive_transposes",
            "add_fp16_cast",
        ]
        assert cast_ops[0].scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == ["module_2"]
        assert len(cast_ops[1].scopes) == 2
        assert cast_ops[1].scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS] == [
            "merge_consecutive_transposes",
            "add_fp16_cast",
        ]
        assert cast_ops[1].scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == [
            "module_2",
        ]

        transpose_op = prog.find_ops(op_type="transpose")[0]
        assert len(transpose_op.scopes) == 2
        assert transpose_op.scopes[ScopeSource.COREMLTOOLS_GRAPH_PASS] == [
            "merge_consecutive_transposes",
            "add_fp16_cast",
        ]
        assert transpose_op.scopes[ScopeSource.TORCHSCRIPT_MODULE_TYPE] == [
            "module_2",
        ]
