#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    assert_op_count_match,
    assert_model_is_valid,
    get_op_types_in_program,
    apply_pass_and_basic_check,
)

import pytest
import numpy as np
import itertools

np.random.seed(1984)


class TestElementwiseOptimizationPasses:
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
            [
                2,
                3,
            ],  # 1D conv conversion broken even without the pass: rdar://problem/62960720
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
        const = (
            np.random.rand(Cout) if add_batch_dim_to_const else np.random.rand(1, Cout)
        )
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

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_conv_bias"
        )
        assert get_op_types_in_program(prev_prog) == [conv_op, element_op, "relu"]
        assert get_op_types_in_program(prog) == [conv_op, "relu"]

        old_bias = prev_block.find_ops(op_type=conv_op)[0].inputs.get("bias", None)
        old_bias_val = 0 if old_bias is None else old_bias.val
        assert old_bias_val is not None
        assert block.find_ops(op_type=conv_op)[0].inputs["bias"] is not None
        new_bias_val = block.find_ops(op_type=conv_op)[0].inputs["bias"].val
        assert new_bias_val is not None
        if use_sub_instead:
            np.testing.assert_almost_equal(
                old_bias_val - np.squeeze(const), new_bias_val
            )
        else:
            np.testing.assert_almost_equal(
                old_bias_val + np.squeeze(const), new_bias_val
            )

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
            [1, 2, 3], # conv_dim
            [True, False], # has_bias
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
        conv_bias = np.arange(Cout).astype(np.float32) if has_bias else np.zeros(Cout).astype(np.float32)
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

        # generate the perm for the tranpose op
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
            bias_rank = np.random.randint(low=1, high=rank+1)
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

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_conv_bias"
        )
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
