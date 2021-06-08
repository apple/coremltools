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

def _apply_weight_transform(inputs, is_deconv, is_real_div, is_conv_first_input, const_type):
    """
    Utility funtion to test the weight transform function in conv scale fusion pass.
    """
    Cin, Cout, groups = 10, 20, 10
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

    apply_pass_and_basic_check(
            prog, "common::fuse_conv_scale"
    )

    # get the updated weight from the prog
    conv_op = []
    for op in prog["main"].operations:
        if op.op_type == "const":
            continue
        conv_op.append(op)
    assert len(conv_op) == 1, "should only have one conv / conv_transpose layer."

    return conv_op[0].weight.val, conv_op[0].bias.val


class TestConvScaleOptimizationPasses:

    @pytest.mark.parametrize(
        "conv_type, is_real_div, is_conv_first_input, const_type",
        itertools.product(
            ["conv", "conv_transpose"],
            [True, False],
            [True, False],
            ["python_scale", "numpy_scale", "numpy_0d_array", "numpy_1d_array", "numpy_3d_array", "numpy_4d_array"],
        )
    )
    def test_weight_transform_conv(self, conv_type, is_real_div, is_conv_first_input, const_type):
        """
        Test the weight transform function in the conv scale fusion pass
        """
        # parameters for conv
        is_deconv = conv_type == "conv_type"
        conv_weight = np.arange(20).astype(np.float32)
        conv_weight = np.reshape(conv_weight, (10, 2, 1, 1)) if is_deconv else np.reshape(conv_weight, (20, 1, 1, 1))
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

        new_conv_weight, new_conv_bias = _apply_weight_transform(
            inputs, is_deconv, is_real_div, is_conv_first_input, const_type
            )

        if is_real_div:
            scale = 1./scale

        if const_type != "numpy_3d_array" and const_type != "numpy_4d_array":
            expected_bias = conv_bias * scale
            expected_weight = conv_weight * scale
        else:
            scale = np.reshape(scale, (20))
            expected_bias = conv_bias * scale
            if is_deconv:
                scale = np.reshape(scale, (20, 1, 1))
                expect_weight = np.reshape(np.arange(20), (20, 1, 1))
                expected_weight = expected_weight * scale
                expected_weight = np.reshape(expected_weight, (10, 2, 1, 1)).astype(np.float32)
            else:
                scale = np.reshape(scale, (20, 1, 1, 1))
                expected_weight = conv_weight * scale

        np.testing.assert_almost_equal(new_conv_weight, expected_weight)
        np.testing.assert_almost_equal(new_conv_bias, expected_bias)

        assert new_conv_weight.dtype == conv_weight.dtype, "weight data type should not be changed after conv_scale_fusion pass."
        assert new_conv_bias.dtype == conv_weight.dtype, "bias data type should be the same as the weight for conv layer."

    @pytest.mark.parametrize(
        "rank, groups, has_bias, scale_op, scale_type, backend",
        itertools.product([3, 4], [1, 10], [False, True], ["mul", "real_div"], ["scalar", "vector"], backends),
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
            conv_weight = np.random.rand(Cout, Cin // groups, 2) if rank == 3 else np.random.rand(Cout, Cin // groups, 2, 3)
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

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_conv_scale"
        )

        assert get_op_types_in_program(prev_prog) == ["conv", scale_op]
        assert get_op_types_in_program(prog) == ["conv"]

        # validate graph pass
        input_dict = {
            "x": np.random.rand(*input_shape),
        }
        output_shape = (2, Cout, 19) if rank == 3 else (2, Cout, 19, 22)
        assert_model_is_valid(
            prog,
            {"x": input_shape},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=backend,
        )


    @pytest.mark.parametrize(
        "rank, groups, has_bias, scale_op, scale_type, backend",
        itertools.product([3, 4], [1, 10], [False, True], ["mul", "real_div"], ["scalar", "vector"], backends),
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
            conv_weight = np.random.rand(Cin, Cout // groups, 2) if rank == 3 else np.random.rand(Cin, Cout // groups, 2, 3)
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


        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_conv_scale"
        )

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