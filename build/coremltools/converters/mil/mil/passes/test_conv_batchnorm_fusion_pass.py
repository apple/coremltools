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
from coremltools.converters.mil import testing_reqs

import pytest
import numpy as np
import itertools

np.random.seed(1984)

backends = testing_reqs.backends

def _apply_weight_transform(inputs, is_deconv):
    """
    Utility funtion to test the weight transform function in conv batch_norm fusion pass.
    """
    Cin, Cout, groups = 10, 20, 10
    input_shape = (1, Cin, 2, 2)
    @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)])
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

    apply_pass_and_basic_check(
            prog, "common::fuse_conv_batchnorm"
    )

    # get the updated weight from the prog
    conv_op = []
    for op in prog["main"].operations:
        if op.op_type == "const":
            continue
        conv_op.append(op)
    assert len(conv_op) == 1, "should only have one conv / conv_transpose layer."

    return conv_op[0].weight.val, conv_op[0].bias.val


class TestConvBatchNormOptimizationPasses:

    @pytest.mark.parametrize(
        "conv_type",
        ["conv", "conv_transpose"],
    )
    def test_weight_transform_conv_identity(self, conv_type):
        """
        Test the weight transform function with an identity batchnorm layer.
        """
        # parameters for conv
        is_deconv = conv_type == "conv_type"
        conv_weight = np.arange(20).astype(np.float32)
        conv_weight = np.reshape(conv_weight, (10, 2, 1, 1)) if is_deconv else np.reshape(conv_weight, (20, 1, 1, 1))
        conv_bias = np.arange(20).astype(np.float32)

        # parameters for batch_norm
        gamma = np.ones(20)
        beta = np.zeros(20)
        mean = np.zeros(20)
        variance = np.ones(20)
        epsilon = 0.

        inputs = {
            "conv_weight": conv_weight,
            "conv_bias": conv_bias,
            "gamma": gamma,
            "beta": beta,
            "mean": mean,
            "variance": variance,
            "epsilon": epsilon,
        }

        new_conv_weight, new_conv_bias = _apply_weight_transform(inputs, is_deconv)

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
        is_deconv = conv_type == "conv_type"
        conv_weight = np.arange(20).astype(dtype)
        conv_weight = np.reshape(conv_weight, (10, 2, 1, 1)) if is_deconv else np.reshape(conv_weight, (20, 1, 1, 1))
        conv_bias = np.arange(20).astype(np.float)

        # parameters for batch_norm
        gamma = np.ones(20).astype(np.int)
        beta = np.zeros(20).astype(np.int)
        mean = np.zeros(20).astype(np.int)
        variance = np.ones(20).astype(np.int)
        epsilon = 0.1

        inputs = {
            "conv_weight": conv_weight,
            "conv_bias": conv_bias,
            "gamma": gamma,
            "beta": beta,
            "mean": mean,
            "variance": variance,
            "epsilon": epsilon,
        }

        new_conv_weight, _ = _apply_weight_transform(inputs, is_deconv)

        assert new_conv_weight.dtype == dtype, "the weight transform function should retain the weight's original dtype."


    @pytest.mark.parametrize(
        "rank, groups, has_bias, backend",
        itertools.product([3, 4], [1, 2, 10], [False, True], backends),
    )
    def test_conv(self, rank, groups, has_bias, backend):
        """
        Input graph:
        input -----> conv -----> batch_norm ---> out

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
        "rank, groups, has_bias, backend",
        itertools.product([3, 4], [1, 2, 10], [False, True], backends),
    )
    def test_conv_transpose(self, rank, groups, has_bias, backend):
        """
        Input graph:
        input -----> conv_transpose -----> batch_norm ---> out

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
        output_shape = (2, Cout, 21) if rank == 3 else (2, Cout, 21, 26)
        assert_model_is_valid(
            prog,
            {"x": input_shape},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=backend,
        )
