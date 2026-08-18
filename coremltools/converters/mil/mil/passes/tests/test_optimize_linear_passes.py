#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
import itertools

import numpy as np
import pytest

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.converters.mil.testing_reqs import backends
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    assert_op_count_match,
    assert_same_output_names,
    get_op_types_in_program,
)

from .test_passes import _VALIDATE_MODEL


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

    @staticmethod
    def _evaluate(prog, x_val):
        """Interpret the handful of op types these programs are made of."""
        env = {}
        func = prog.functions["main"]
        for var in func.inputs.values():
            env[var] = x_val

        def value_of(var):
            return env[var] if var in env else np.asarray(var.val)

        for op in func.operations:
            if op.op_type == "const":
                continue
            elif op.op_type == "matmul":
                a, b = value_of(op.x), value_of(op.y)
                if op.transpose_x.val:
                    a = np.swapaxes(a, -1, -2)
                if op.transpose_y.val:
                    b = np.swapaxes(b, -1, -2)
                env[op.outputs[0]] = a @ b
            elif op.op_type == "linear":
                env[op.outputs[0]] = value_of(op.x) @ value_of(op.weight).T + value_of(op.bias)
            elif op.op_type == "transpose":
                env[op.outputs[0]] = np.transpose(value_of(op.x), list(op.perm.val))
            elif op.op_type == "add":
                env[op.outputs[0]] = value_of(op.x) + value_of(op.y)
            elif op.op_type == "sub":
                env[op.outputs[0]] = value_of(op.x) - value_of(op.y)
            else:
                raise AssertionError(f"unhandled op type {op.op_type}")
        return env[func.outputs[0]]

    @pytest.mark.parametrize(
        "transpose_x, transpose_y", itertools.product([False, True], [False, True])
    )
    def test_fuse_matmul_weight_bias_sub_with_matmul_second(self, transpose_x, transpose_y):
        """
        sub is not commutative: ``bias - x @ w`` is ``linear(x, -w, bias)``, not
        ``linear(x, w, -bias)``.
        """
        weights_val = ((np.arange(9, dtype=np.float32).reshape(3, 3) + 1.0) / 7.0)
        bias_val = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        x_val = np.arange(9, dtype=np.float32).reshape(3, 3)

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 3))])
        def prog(x):
            matmul = mb.matmul(
                x=x, y=weights_val, transpose_x=transpose_x, transpose_y=transpose_y
            )
            return mb.sub(x=bias_val, y=matmul)

        expected = self._evaluate(prog, x_val)
        PASS_REGISTRY["common::fuse_matmul_weight_bias"](prog)
        assert get_op_types_in_program(prog).count("linear") == 1
        np.testing.assert_allclose(self._evaluate(prog, x_val), expected, atol=1e-04, rtol=1e-05)

    @pytest.mark.parametrize(
        "op_type, bias_first", itertools.product(["add", "sub"], [True, False])
    )
    def test_no_fuse_matmul_weight_bias_const_on_the_left(self, op_type, bias_first):
        """
        ``w @ x`` becomes ``transpose(linear(transpose(x), w))``, and linear's bias would
        land on the wrong axis of the result, so this pattern must be left alone.
        """
        weights_val = ((np.arange(9, dtype=np.float32).reshape(3, 3) + 1.0) / 7.0)
        bias_val = np.array([100.0, 200.0, 300.0], dtype=np.float32)

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 3))])
        def prog(x):
            matmul = mb.matmul(x=weights_val, y=x)
            builder_op = getattr(mb, op_type)
            if bias_first:
                return builder_op(x=bias_val, y=matmul)
            return builder_op(x=matmul, y=bias_val)

        PASS_REGISTRY["common::fuse_matmul_weight_bias"](prog)
        assert get_op_types_in_program(prog) == ["matmul", op_type]


class TestFuseTransposeMatmul:
    def test_fuse_transposes(self):
        X_SHAPE = (3, 2)
        Y_SHAPE = (5, 2)

        output_shape = (X_SHAPE[0], Y_SHAPE[0])

        @mb.program(input_specs=[mb.TensorSpec(shape=X_SHAPE), mb.TensorSpec(shape=Y_SHAPE)])
        def prog(x, y):
            transposed_x = mb.transpose(x=x, perm=(1, 0))
            transposed_y = mb.transpose(x=y, perm=(1, 0))
            z = mb.matmul(x=transposed_x, y=transposed_y, transpose_x=True, transpose_y=False)
            return z

        prev_prog, _, _ = apply_pass_and_basic_check(prog, "common::fuse_transpose_matmul")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prev_prog) == ["transpose", "transpose", "matmul"]
        assert get_op_types_in_program(prog) == ["matmul"]

        matmul = prog.find_ops(op_type="matmul")[0]
        assert not matmul.transpose_x.val
        assert matmul.transpose_y.val

        assert_model_is_valid(
            prog,
            {"x": X_SHAPE, "y": Y_SHAPE},
            expected_output_shapes={block.outputs[0].name: output_shape},
        )

    def test_fuse_transpose_y(self):
        X_SHAPE = (3, 2)
        Y_SHAPE = (2, 5)

        output_shape = (X_SHAPE[0], Y_SHAPE[1])

        @mb.program(input_specs=[mb.TensorSpec(shape=X_SHAPE), mb.TensorSpec(shape=Y_SHAPE)])
        def prog(x, y):
            transposed_y = mb.transpose(x=y, perm=(1, 0))
            z = mb.matmul(x=x, y=transposed_y, transpose_y=True)
            return z

        prev_prog, _, _ = apply_pass_and_basic_check(prog, "common::fuse_transpose_matmul")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prev_prog) == ["transpose", "matmul"]
        assert get_op_types_in_program(prog) == ["matmul"]

        matmul = prog.find_ops(op_type="matmul")[0]
        assert not matmul.transpose_x.val
        assert not matmul.transpose_y.val

        assert_model_is_valid(
            prog,
            {"x": X_SHAPE, "y": Y_SHAPE},
            expected_output_shapes={block.outputs[0].name: output_shape},
        )

    def test_fuse_transpose_x_but_unfuseable_transpose_y(self):
        X_SHAPE = (4, 2, 5, 3)
        Y_SHAPE = (4, 5, 2, 7)

        output_shape = (X_SHAPE[0], X_SHAPE[1], X_SHAPE[3], Y_SHAPE[3])

        @mb.program(input_specs=[mb.TensorSpec(shape=X_SHAPE), mb.TensorSpec(shape=Y_SHAPE)])
        def prog(x, y):
            transposed_x = mb.transpose(x=x, perm=(0, 1, 3, 2))
            transposed_y = mb.transpose(x=y, perm=(0, 2, 1, 3))
            z = mb.matmul(x=transposed_x, y=transposed_y)
            return z

        prev_prog, _, _ = apply_pass_and_basic_check(prog, "common::fuse_transpose_matmul")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prev_prog) == ["transpose", "transpose", "matmul"]
        assert get_op_types_in_program(prog) == ["transpose", "matmul"]

        assert_model_is_valid(
            prog,
            {"x": X_SHAPE, "y": Y_SHAPE},
            expected_output_shapes={block.outputs[0].name: output_shape},
        )

    def test_unfuseable_transposes(self):
        X_SHAPE = (3, 2, 5)
        Y_SHAPE = (5, 2, 7)

        output_shape = (X_SHAPE[1], X_SHAPE[0], Y_SHAPE[2])

        @mb.program(input_specs=[mb.TensorSpec(shape=X_SHAPE), mb.TensorSpec(shape=Y_SHAPE)])
        def prog(x, y):
            transposed_x = mb.transpose(x=x, perm=(1, 0, 2))
            transposed_y = mb.transpose(x=y, perm=(1, 0, 2))
            z = mb.matmul(x=transposed_x, y=transposed_y)
            return z

        prev_prog, _, _ = apply_pass_and_basic_check(prog, "common::fuse_transpose_matmul")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prev_prog) == ["transpose", "transpose", "matmul"]
        assert get_op_types_in_program(prev_prog) == get_op_types_in_program(prog)

        assert_model_is_valid(
            prog,
            {"x": X_SHAPE, "y": Y_SHAPE},
            expected_output_shapes={block.outputs[0].name: output_shape},
        )
