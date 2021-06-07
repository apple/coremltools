#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    assert_model_is_valid,
    get_op_types_in_program,
    apply_pass_and_basic_check,
)
from coremltools.converters.mil import testing_reqs

import pytest
import numpy as np
import itertools

np.random.seed(1984)

backends = testing_reqs.backends

def _apply_transform(inputs, func, is_first_input, has_bias):
    """
    Utility funtion to test the weight/bias transform function in linear bias fusion pass.
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
            prog, "common::fuse_linear_bias",
    )

    # get the updated weight from the prog
    linear_op = []
    for op in prog["main"].operations:
        if op.op_type == "const":
            continue
        linear_op.append(op)
    assert len(linear_op) == 1, "should only have one linear layer."

    return linear_op[0].weight.val, linear_op[0].bias.val


class TestLinearBiasOptimizationPasses:

    @pytest.mark.parametrize(
        "op_type, is_first_input, has_bias, broadcast",
        itertools.product(
            ["add", "sub"],
            [True, False],
            [True, False],
            [True, False],
        )
    )
    def test_transform_linear(self, op_type, is_first_input, has_bias, broadcast):
        """
        Test the weight / bias transform function in the linear bias fusion pass
        """
        weight = np.reshape(np.arange(8), (2, 4)).astype(np.float32)
        linear_bias = np.array([1, 2]).astype(np.float32) if has_bias else np.array([0, 0]).astype(np.float32)
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

        new_weight, new_bias = _apply_transform(
            inputs, func, is_first_input, has_bias,
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
        itertools.product(
            [1, 2, 3],
            ["add", "sub"],
            [True, False],
            [True, False],
            backends),
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
            linear_bias = np.array([1., 2.])
            bias = np.array([3., 4.])
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

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_linear_bias"
        )

        assert get_op_types_in_program(prev_prog) == ["linear", op_type]
        assert get_op_types_in_program(prog) == ["linear"]

        # validate graph pass
        input_dict = {
            "x": np.random.rand(*input_shape),
        }
        output_shape = [1, 2, 2]
        output_shape = tuple(output_shape[-rank:])
        assert_model_is_valid(
            prog,
            {"x": input_shape},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=backend,
        )
