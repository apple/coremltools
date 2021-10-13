# -*- coding: utf-8 -*-

#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import os
import numpy as np

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.experimental.passes.generic_pass_infrastructure import register_generic_pass

"""
Fuse the following batch_norm layer into conv and conv_transpose
That is, convert conv + batch_norm to conv, by modifying the weight and bias in the conv layer
Given:
    %2 = conv(%1)
    ...
    %3 = batch_norm(%2)
    ...

Result:
    %3 = conv(%1)
    ...
"""

arbitrary_cin = 5
arbitrary_cout = 8
np.random.seed()
arbitrary_input = (3, arbitrary_cin, 224, 224)
arbitrary_weight = np.random.rand(arbitrary_cout, arbitrary_cin, 10, 10)
arbitrary_mean= np.random.rand(arbitrary_cout)
arbitrary_variance = np.random.rand(arbitrary_cout)

@mb.program(input_specs=[mb.TensorSpec(shape=arbitrary_input)])
def conv_batchnorm(x):
    conv = mb.conv(x=x, weight=arbitrary_weight, pad_type="valid", name="conv")
    batch_norm = mb.batch_norm(x=conv, mean=arbitrary_mean, variance=arbitrary_variance, name="batchnorm")
    return batch_norm


@mb.program(input_specs=[mb.TensorSpec(shape=arbitrary_input)])
def conv_transpose_batchorm(x):
    conv = mb.conv_transpose(x=x, weight=arbitrary_weight, pad_type="valid", name="conv")
    batch_norm = mb.batch_norm(x=conv, mean=arbitrary_mean, variance=arbitrary_variance, name="batchnorm")
    return batch_norm


def var_constraints(pattern):
    return pattern.conv.weight.val is not None


def transform_pattern(pattern):
    # get parameters from batch_norm layer
    gamma = pattern.batchnorm.gamma.val
    beta = pattern.batchnorm.beta.val
    mean = pattern.batchnorm.mean.val
    variance = pattern.batchnorm.variance.val
    epsilon = pattern.batchnorm.epsilon.val
    # get weight, bias and groups from conv layer

    conv_weight = pattern.conv.weight.val
    conv_bias = pattern.conv.bias
    groups = pattern.conv.groups.val

    # get type of the conv layer
    is_deconv = pattern.conv.op_type == 'conv_transpose'
    is_conv_1d  = len(conv_weight.shape) == 3

    # D_in denotes the spatial dimensions for conv kernel weight
    # for conv_transpose, conv_weight has shape [Cin, Cout / groups, *D_in]
    # for conv, conv_weight has shape [Cout, Cin / groups, *D_in]
    if is_deconv:
        Cout = conv_weight.shape[1] * groups
        Cin = conv_weight.shape[0]
    else:
        Cout = conv_weight.shape[0]
        Cin = conv_weight.shape[1] * groups

    # get the type of the conv weight
    conv_weight_type = conv_weight.dtype

    # create bias for conv if not exist
    if conv_bias is None:
        conv_bias = np.zeros(Cout)
    else:
        conv_bias = conv_bias.val
    conv_bias = conv_bias.astype(conv_weight_type)

    # get the original shape of weight and bias
    origin_weight_shape = conv_weight.shape
    origin_bias_shape = conv_bias.shape

    # update the weight for conv layer
    new_conv_weight = []
    new_conv_bias = []

    if is_deconv:
        conv_weight = np.transpose(conv_weight, [1, 0, 2] if is_conv_1d else [1, 0, 2, 3])
        conv_weight = np.reshape(conv_weight, [Cout, Cin // groups] + list(conv_weight.shape[2:]))

    for i in range(Cout):
        # get batch norm parameters for each channel
        _gamma = gamma[i]
        _beta = beta[i]
        _mean = mean[i]
        _variance = variance[i]
        _scale = _gamma / np.sqrt(_variance + epsilon)

        # get conv weight and bias for each channel
        _conv_weight = conv_weight[i]
        _conv_bias = conv_bias[i]

        # update the conv weight and bias
        _conv_weight = _conv_weight * _scale
        _conv_bias = _scale * (_conv_bias - _mean) + _beta
        new_conv_weight.append(_conv_weight)
        new_conv_bias.append(_conv_bias)

    new_conv_weight = np.array(new_conv_weight).astype(conv_weight_type)
    new_conv_bias = np.array(new_conv_bias).astype(conv_weight_type)

    if is_deconv:
        new_conv_weight = np.reshape(new_conv_weight, [Cout // groups, Cin] + list(new_conv_weight.shape[2:]))
        new_conv_weight = np.transpose(new_conv_weight, [1, 0, 2] if is_conv_1d else [1, 0, 2, 3])

    # make sure the updated weight and bias have the same shape as the original ones
    assert new_conv_weight.shape == origin_weight_shape, "conv weight should have the same shape before and after the fuse_conv_batchnorm pass."
    assert new_conv_bias.shape == origin_bias_shape, "conv bias should have the same shape before and after the fuse_conv_batchnorm pass."

    # create a new conv op with the new bias value, copying rest of the attributes
    out_name = pattern.batchnorm.outputs[0].name
    conv_kargs = {"weight": new_conv_weight, "bias": new_conv_bias, "name": out_name, "before_op": pattern.conv}

    for k, v in pattern.conv.inputs.items():
        if k in ["weight", "bias"]:
            continue
        conv_kargs[k] = v

    if is_deconv:
        x = mb.conv_transpose(**conv_kargs)
    else:
        x = mb.conv(**conv_kargs)

    pattern.batchnorm.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=pattern.batchnorm, old_var=pattern.batchnorm.outputs[0], new_var=x
    )
    # Remove all the ops at once
    pattern.block.remove_ops(pattern.op_list())


if os.getenv('ENABLE_EXPERIMENTAL_PASSES') == '1':
    register_generic_pass(
        ops_arrangement=conv_batchnorm,
        var_constraints=var_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_conv_batchnorm",
        namespace="common",
    )

    register_generic_pass(
        ops_arrangement=conv_transpose_batchorm,
        var_constraints=var_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_conv_batchnorm",
        namespace="common",
    )