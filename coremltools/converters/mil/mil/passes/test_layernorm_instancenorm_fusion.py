#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    assert_model_is_valid,
    get_op_types_in_program,
    apply_pass_and_basic_check,
)

np.random.seed(6174)


class TestLayerNormOrInstanceNormFusionPass:
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
            prog, {"x": shape}, expected_output_shapes={block.outputs[0].name: shape},
        )
        # reduce transpose pass should remove extra ones
        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::reduce_transposes"
        )
        assert get_op_types_in_program(prog) == ["instance_norm"]
        assert_model_is_valid(
            prog, {"x": shape}, expected_output_shapes={block.outputs[0].name: shape},
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
            mul_sub = mb.mul(x=mul2, y=-1)
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
