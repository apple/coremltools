# -*- coding: utf-8 -*-

#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import operator
import os
import numpy as np

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.passes.helper import _check_var_scalar_value
from coremltools.converters.mil.experimental.passes.generic_pass_infrastructure import register_generic_pass
from coremltools.converters.mil.mil import get_new_symbol

if os.getenv("ENABLE_EXPERIMENTAL_PASSES") == "1":
    shape = (get_new_symbol(), get_new_symbol(), get_new_symbol(), get_new_symbol())

def _check_reduce_op(reduce_op, mode="reduce_mean") -> bool:
    """
    Check whether or not the reduction op satisfy following conditions:
    - Mode is expected.
    - Does not change rank (keep_dims is True).
    - Axes are known at compile time.

    :param reduce_op: reduce op to check on
    :param mode: reduce mode
    """
    if reduce_op is None:
        return False
    if reduce_op.op_type != mode:
        return False
    if reduce_op.keep_dims.val is False:
        return False
    if reduce_op.axes is None or reduce_op.axes.val is None:
        return False
    return True

if os.getenv("ENABLE_EXPERIMENTAL_PASSES") == "1":
    @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
    def instancenorm_or_layernorm(x):
        """
        Identify the pattern:

        y = gamma * (x - mean) / sqrt(variance + epsilon) + beta

        y = x * [gamma * rsqrt(variance + eps)] + (beta - mean * [gamma * rsqrt(variance + eps)])

        x --> main_reduce --> sub --> square --> reduce_mean_2 --> add(epsilon) --> rsqrt
        |             |        ^                                                      |
        |             |        |                                                      V
        |-----------------------                                                mul (gamma)
        |             |                                                             |
        |             |                                                     --------|---------
        |             |                                                     |                |
        |             |                                                     |                V
        |             |------------------------------------------------------------------>  mul_3
        |                                                                   |                |
        |                                                                   V                |
        |----------------------------------------------------------------> mul_2             |
                                                                            |                V
                                                                            |              sub (beta) --> add_2 --> [...]
                                                                            |                              ^
                                                                            |-------------------------------

        This pattern corresponds to either layer_norm or instance_norm.

        It is instance_norm if all of the following are true:
            - input is rank 4
            - axes of reduce_mean is [-2, -1] or [-3, -2]
              (when [-3, -2], a channel first to channel last transpose would be inserted)
            - gamma and beta are rank 1, after squeeze

        It is layer_norm if all of the following are true:
            - axes is either [-1] or [-1, -2] or [-1, -2, -3] and so on
            - rank of gamma and beta is equal to the length of the axes
        """
        main_reduce = mb.reduce_mean(x=x, axes=[2, 3], keep_dims=True, name="main_reduce")
        sub = mb.sub(x=x, y=main_reduce, name="sub")
        square = mb.square(x=sub, name="square")
        reduce_mean_2 = mb.reduce_mean(x=square, axes=[2, 3], keep_dims=True, name="reduce_mean_2")
        add_epsilon = mb.add(x=reduce_mean_2, y=1e-5, name="add_epsilon")
        rsqrt = mb.rsqrt(x=add_epsilon, epsilon=1e-12, name="rsqrt")
        mul_gamma = mb.mul(x=rsqrt, y=np.random.rand(1, 5, 1, 1), name="mul_gamma")
        mul_2 = mb.mul(x=x, y=mul_gamma, name="mul_2")
        mul_3 = mb.mul(x=main_reduce, y=mul_gamma, name="mul_3")
        sub_beta = mb.sub(x=np.random.rand(1, 5, 1, 1), y=mul_3, name="sub_beta")
        add_2 = mb.add(x=sub_beta, y=mul_2, name="add_2")
        return add_2

if os.getenv("ENABLE_EXPERIMENTAL_PASSES") == "1":
    @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
    def instancenorm_2(x):
        """
        Identify the pattern:
        y = (x - mean) / pow(variance + epsilon) * gamma + beta

        This pattern corresponds to, should be fused as instance_norm.
        All of the following must be satisty:
        1) Input is rank 4 tensor
        2) Reduce operates on spatial dimensions axes=[-2, -1], or axes=[-3, -2] (a
           channel first to channel last transpose would be inserted in such case)
        3) Gamma and beta are both shape (C,) after squeeze, where C is number of channels


        |----> sub0 ----------|                            const (0.5)
        |       ^             |                                |
        |       |             V                                V
        x ---> main_reduce  square --> mean1 --> add_eps ---> pow       const_gamma   const_beta
        |       |                                              |             |            |
        |       V                                              V             V            V
        |----> sub1 --------------------------------------> real_div --> mul_gamma --> add_beta --> ...
        """

        main_reduce = mb.reduce_mean(x=x, axes=[2, 3], keep_dims=True, name="main_reduce")
        sub0 = mb.sub(x=x, y=main_reduce, name="sub0")
        sub1 = mb.sub(x=x, y=main_reduce, name="sub1")
        square = mb.square(x=sub0, name="square")
        mean1 = mb.reduce_mean(x=square, axes=[2, 3], keep_dims=True, name="mean1")
        add_epsilon = mb.add(x=mean1, y=1e-5, name="add_epsilon")
        pow = mb.pow(x=add_epsilon, y=0.5, name="pow")
        real_div = mb.real_div(x=sub1, y=pow, name="real_div")
        mul_gamma = mb.mul(x=np.random.rand(1, 5, 1, 1), y=real_div, name="mul_gamma")
        add_beta = mb.add(x=np.random.rand(1, 5, 1, 1), y=mul_gamma, name="add_beta")
        return add_beta


if os.getenv("ENABLE_EXPERIMENTAL_PASSES") == "1":
    @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
    def instancenorm_3(x):
        """
        Detect InstanceNorm pattern in TensorFlow-Addons.

        This pattern corresponds to, should be fused as instance_norm.
        All of the following must be satisty:
        1) Input is rank 4 tensor
        2) Reduce operates on spatial dimensions axes=[-2, -1], or axes=[-3, -2] (a
           channel first to channel last transpose would be inserted in such case)
        3) Gamma and beta are absent. Default values for gamma and beta would be used.

               |-------------------------------------------------------|
               |                                                       |
               |                                                       V
        x --> main_reduce   square --> mean1 --> add_eps --> rsqrt --> mul2 --> mul_sub
        |      |             ^                                |                   |
        |      V             |                                |                   |
        | --> sub -----------|                                |                   |
        |                                                     V                   V
        |--------------------------------------------------> mul1 -------------> add --> ...
        """

        main_reduce = mb.reduce_mean(x=x, axes=[2, 3], keep_dims=True, name="main_reduce")
        sub = mb.sub(x=x, y=main_reduce, name="sub")
        square = mb.square(x=sub, name="square")
        mean1 = mb.reduce_mean(x=square, axes=[2, 3], keep_dims=True, name="mean1")
        add_epsilon = mb.add(x=mean1, y=1e-5, name="add_epsilon")  # epsilon
        rsqrt = mb.rsqrt(x=add_epsilon, name="rsqrt")
        mul1 = mb.mul(x=rsqrt, y=x, name="mul1")
        mul2 = mb.mul(x=main_reduce, y=rsqrt, name="mul2")
        mul_sub = mb.mul(x=mul2, y=-1, name="mul_sub")
        add = mb.add(x=mul1, y=mul_sub, name="add")
        return add


if os.getenv("ENABLE_EXPERIMENTAL_PASSES") == "1":
    @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
    def instancenorm_4(x):
        """
        Identify the pattern:
        y = x * [gamma * rsqrt(variance + eps)] + (beta - mean * [gamma * rsqrt(variance + eps)])

        This pattern corresponds to, should be fused as instance_norm.
        All of the following must be satisty:
        1) Input is rank 4 tensor
        2) Reduce operates on spatial dimensions axes=[-2, -1], or axes=[-3, -2] (a
           channel first to channel last transpose would be inserted in such case)
        3) Gamma and beta are both shape (C,) after squeeze, where C is number of channels

        |-----------|
        |           V
        |------> mul_square1 -------------> sum1 -----> mul_mean1
        |                                                   |
        |                                                   V
        x --> main_reduce --> mul_mean ==> mul_square --> sub_variance --> add_eps --> rsqrt
        |                        |                                                      |
        |                        |                                                      V
        |                        |                                                  mul_gamma
        |                        |                                                      |
        |                        |                                            |----------------|
        |                        |                                            |                V
        |                        |--------------------------------------------+-------------> mul2
        |                                                                     V                |
        |------------------------------------------------------------------> mul1              |
                                                                              |                V
                                                                              |             sub_beta --> add --> [...]
                                                                              |                           ^
                                                                              |---------------------------|
        """
        mul_square1 = mb.mul(x=x, y=x, name="mul_square1")
        main_reduce = mb.reduce_sum(x=x, axes=[2, 3], keep_dims=True, name="main_reduce")
        mul_mean = mb.mul(x=main_reduce, y=3.3333334e-05, name="mul_mean")  # dummy value here
        mul_square = mb.mul(x=mul_mean, y=mul_mean, name="mul_square")
        sum1 = mb.reduce_sum(x=mul_square1, axes=[2, 3], keep_dims=True, name="sum1")
        mul_mean1 = mb.mul(x=sum1, y=8.333333e-06, name="mul_mean1")  # dummy value here
        sub_variance = mb.sub(x=mul_mean1, y=mul_square, name="sub_variance")
        add_epsilon = mb.add(x=sub_variance, y=1e-5, name="add_epsilon")  # epsilon
        rsqrt = mb.rsqrt(x=add_epsilon, name="rsqrt")
        mul_gamma = mb.mul(x=rsqrt, y=np.random.rand(1, 5, 1, 1), name="mul_gamma")
        mul1 = mb.mul(x=mul_gamma, y=x, name="mul1")
        mul2 = mb.mul(x=mul_mean, y=mul_gamma, name="mul2")
        sub_beta = mb.sub(x=np.random.rand(1, 5, 1, 1), y=mul2, name="sub_beta")
        add = mb.add(x=mul1, y=sub_beta, name="add")
        return add

def instancenorm_1_constraints(pattern):
    passed = True
    passed = passed and _common_pattern1_constraints(pattern)
    passed = passed and _instancenorm_constraints(pattern)
    return passed


def layernorm_1_constraints(pattern):
    passed = True
    passed = passed and _common_pattern1_constraints(pattern)
    passed = passed and _layernorm_constraints(pattern)
    return passed


def instancenorm_2_constraints(pattern):
    epsilon_var = _get_var(pattern.add_epsilon, pattern.mean1)
    gamma_var = _get_var(pattern.mul_gamma, pattern.real_div)
    beta_var = _get_var(pattern.add_beta, pattern.mul_gamma)

    passed = True
    passed = passed and _check_reduce_op(pattern.main_reduce)
    passed = passed and pattern.sub0.x == pattern.root_var and pattern.sub0.y == pattern.main_reduce.outputs[0]
    passed = passed and pattern.sub1.x == pattern.root_var and pattern.sub1.y == pattern.main_reduce.outputs[0]
    passed = passed and _check_reduce_op(pattern.mean1)
    passed = passed and pattern.pow.y.val is not None and np.isclose(pattern.pow.y.val, 0.5)
    passed = passed and pattern.real_div.x == pattern.sub1.outputs[0] and pattern.real_div.y == pattern.pow.outputs[0]

    passed = passed and _general_constraints(pattern, epsilon_var, gamma_var, beta_var)
    passed = passed and _instancenorm_constraints(pattern)

    return passed


def instancenorm_3_constraints(pattern):
    epsilon_var = _get_var(pattern.add_epsilon, pattern.mean1)

    gamma_var = mb.const(
        val=np.ones(shape=(1, pattern.root_var.shape[1], 1, 1)), name="gamma_var"
    )
    beta_var = mb.const(
        val=np.zeros(shape=(1, pattern.root_var.shape[1], 1, 1)),
        name="_fuse_layernorm_or_instancenorm_beta",
    )
    passed = True
    passed = passed and _check_reduce_op(pattern.main_reduce)
    passed = passed and pattern.sub.x == pattern.root_var and pattern.sub.y == pattern.main_reduce.outputs[0]
    passed = passed and _check_reduce_op(pattern.mean1)
    passed = passed and pattern.mul_sub.y.val is not None and pattern.mul_sub.y.val == -1

    passed = passed and _general_constraints(pattern, epsilon_var, gamma_var, beta_var)
    passed = passed and _instancenorm_constraints(pattern)

    return passed


def instancenorm_4_constraints(pattern):
    epsilon_var = _get_var(pattern.add_epsilon, pattern.sub_variance)
    gamma_var = _get_var(pattern.mul_gamma, pattern.rsqrt)
    beta_var = pattern.sub_beta.x

    passed = True
    passed = passed and _check_reduce_op(pattern.main_reduce, mode="reduce_sum")
    passed = passed and pattern.mul_mean.y.shape == ()
    passed = passed and _check_reduce_op(pattern.sum1, "reduce_sum")
    passed = passed and pattern.mul_mean1.y.shape == ()
    passed = passed and pattern.sub_variance.y == pattern.mul_square.outputs[0]
    passed = passed and pattern.sub_beta.y == pattern.mul2.outputs[0]

    passed = passed and _general_constraints(pattern, epsilon_var, gamma_var, beta_var)
    passed = passed and _instancenorm_constraints(pattern)

    return passed


def _general_constraints(pattern, epsilon_var, gamma_var, beta_var):
    passed = True
    passed = passed and pattern.root_var.shape is not None
    passed = passed and epsilon_var.val is not None and len(epsilon_var.val.shape) == 0
    passed = passed and gamma_var.val is not None
    passed = passed and beta_var.val is not None

    pattern.add_attribute("epsilon_var", epsilon_var)
    pattern.add_attribute("gamma_var", gamma_var)
    pattern.add_attribute("beta_var", beta_var)
    return passed


def _common_pattern1_constraints(pattern):
    epsilon_var = _get_var(pattern.add_epsilon, pattern.reduce_mean_2)
    gamma_var = _get_var(pattern.mul_gamma, pattern.rsqrt)
    beta_var = pattern.sub_beta.x

    passed = True
    passed = passed and _check_reduce_op(pattern.main_reduce)
    passed = passed and _check_reduce_op(pattern.reduce_mean_2)
    passed = passed and pattern.sub.x == pattern.root_var and pattern.sub.y == pattern.main_reduce.outputs[0]
    passed = passed and pattern.sub_beta.y == pattern.mul_3.outputs[0]

    passed = passed and _general_constraints(pattern, epsilon_var, gamma_var, beta_var)

    return passed

def _layernorm_constraints(pattern):
    rank, axes, negative_axes = _rank_and_axes(pattern)

    passed = True
    passed = passed and len(pattern.gamma_var.val.shape) == len(axes)
    passed = passed and len(pattern.beta_var.val.shape) == len(axes)
    passed = passed and negative_axes == list(range(-len(negative_axes), 0))
    requires_rank4_transpose = False

    if rank == 4 and negative_axes == [-3, -2]:
        requires_rank4_transpose = True

    pattern.add_attribute("requires_rank4_transpose", requires_rank4_transpose)
    pattern.add_attribute("is_instancenorm", False)
    return passed


def _instancenorm_constraints(pattern):
    rank, axes, negative_axes = _rank_and_axes(pattern)

    passed = True
    passed = passed and rank == 4
    passed = passed and _check_axes_and_var_shape(negative_axes, pattern.gamma_var.shape)
    passed = passed and _check_axes_and_var_shape(negative_axes, pattern.beta_var.shape)

    requires_rank4_transpose = False
    if negative_axes == [-3, -2]: requires_rank4_transpose = True
    pattern.add_attribute("requires_rank4_transpose", requires_rank4_transpose)
    pattern.add_attribute("is_instancenorm", True)
    return passed


def _rank_and_axes(pattern):
    rank = len(pattern.root_var.shape)
    axes = pattern.main_reduce.axes.val
    negative_axes = [a - rank if a >= 0 else a for a in axes]
    negative_axes.sort()
    return rank, axes, negative_axes


def _get_var(operation1, operation2):
    return operation1.y if operation1.x == operation2.outputs[0] else operation1.x

def _check_axes_and_var_shape(negative_axes, shape):
    if len(shape) == 1:
        return True
    if negative_axes == [-2, -1]:
        return shape[0] == 1 and shape[2] == 1 and shape[3] == 1
    if negative_axes == [-3, -2]:
        return shape[0] == 1 and shape[1] == 1 and shape[2] == 1
    return False

def transform_pattern(pattern):
    """
    Insert instance_norm / layer_norm and delete all ops.
    :param pattern: A pattern object that contains all relevant information.
    """
    out_name = pattern.final_op.outputs[0].name
    axes = pattern.main_reduce.axes.val

    if pattern.requires_rank4_transpose:
        x = mb.transpose(
            x=pattern.main_reduce.x,
            perm=[0, 3, 1, 2],
            name=out_name + "_transpose_nhwc_nchw",
            before_op=pattern.final_op,
        )
    if pattern.is_instancenorm:
        x = mb.instance_norm(
            x=x if pattern.requires_rank4_transpose else pattern.main_reduce.x,
            gamma=np.squeeze(pattern.gamma_var.val),
            beta=np.squeeze(pattern.beta_var.val),
            epsilon=pattern.epsilon_var,
            name=out_name + "_instancenorm" if pattern.requires_rank4_transpose else out_name,
            before_op=pattern.final_op,
        )
    else:  # is_layernorm
        x = mb.layer_norm(
            x=x if pattern.requires_rank4_transpose else pattern.main_reduce.x,
            axes=axes,
            gamma=pattern.gamma_var,
            beta=pattern.beta_var,
            epsilon=pattern.epsilon_var,
            name=out_name + "_layernorm" if pattern.requires_rank4_transpose else out_name,
            before_op=pattern.final_op,
        )
    if pattern.requires_rank4_transpose:
        x = mb.transpose(
            x=x,
            perm=[0, 2, 3, 1],
            name=out_name + "_transpose_nchw_nhwc",
            before_op=pattern.final_op,
        )

    pattern.final_op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=pattern.final_op, old_var=pattern.final_op.outputs[0], new_var=x
    )
    # Remove all the ops at once
    pattern.block.remove_ops(pattern.op_list())


if os.getenv("ENABLE_EXPERIMENTAL_PASSES") == "1":
    register_generic_pass(
        ops_arrangement=instancenorm_or_layernorm,
        var_constraints=layernorm_1_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_layernorm_or_instancenorm",
        namespace="common",
    )

    register_generic_pass(
        ops_arrangement=instancenorm_or_layernorm,
        var_constraints=instancenorm_1_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_layernorm_or_instancenorm",
        namespace="common",
    )

    register_generic_pass(
        ops_arrangement=instancenorm_2,
        var_constraints=instancenorm_2_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_layernorm_or_instancenorm",
        namespace="common",
    )

    register_generic_pass(
        ops_arrangement=instancenorm_3,
        var_constraints=instancenorm_3_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_layernorm_or_instancenorm",
        namespace="common",
    )

    register_generic_pass(
        ops_arrangement=instancenorm_4,
        var_constraints=instancenorm_4_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_layernorm_or_instancenorm",
        namespace="common",
    )
