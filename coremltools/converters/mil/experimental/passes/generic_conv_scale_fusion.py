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
Fold mul/div into conv/conv_transpose by updating the weight/bias of the convolution layers.

The scale const can be a single number (scalar) or a vector with a broacasable shape,
for instance, if the output of the conv/deconv layer is (B, Cout, H, W),
const of shape (Cout, 1, 1) and (1, Cout, 1, 1) are allowed.

Given:
    %2 = conv(%1)
    ...
    %3 = mul(%2, constant) # where constant is the scale constant
    ...

Result:
    %3 = conv(%1)
    ...
"""

arbitrary_cin = 5
arbitrary_cout = 8
arbitrary_scalar = 5
np.random.seed()
arbitrary_input = (3, arbitrary_cin, 224, 224)
arbitrary_weight = np.random.rand(arbitrary_cout, arbitrary_cin, 10, 10)

if os.getenv("ENABLE_EXPERIMENTAL_PASSES") == "1":
    @mb.program(input_specs=[mb.TensorSpec(shape=arbitrary_input)])
    def conv_scale_mul(x):
        conv = mb.conv(x=x, weight=arbitrary_weight, pad_type="valid", name="conv")
        mul = mb.mul(x=conv, y=arbitrary_scalar, name="scale")
        return mul

if os.getenv("ENABLE_EXPERIMENTAL_PASSES") == "1":
    @mb.program(input_specs=[mb.TensorSpec(shape=arbitrary_input)])
    def conv_transpose_scale_mul(x):
        conv = mb.conv_transpose(x=x, weight=arbitrary_weight, pad_type="valid", name="conv")
        mul = mb.mul(x=conv, y=arbitrary_scalar, name="scale")
        return mul

if os.getenv("ENABLE_EXPERIMENTAL_PASSES") == "1":
    @mb.program(input_specs=[mb.TensorSpec(shape=arbitrary_input)])
    def conv_scale_div(x):
        conv = mb.conv(x=x, weight=arbitrary_weight, pad_type="valid", name="conv")
        real_div = mb.real_div(x=conv, y=arbitrary_scalar, name="scale")
        return real_div

if os.getenv("ENABLE_EXPERIMENTAL_PASSES") == "1":
    @mb.program(input_specs=[mb.TensorSpec(shape=arbitrary_input)])
    def conv_transpose_scale_div(x):
        conv = mb.conv_transpose(x=x, weight=arbitrary_weight, pad_type="valid", name="conv")
        real_div = mb.real_div(x=conv, y=arbitrary_scalar, name="scale")
        return real_div


def _cin_cout(pattern):
    # D_in denotes the spatial dimensions for conv kernel weight
    # for conv_transpose, conv_weight has shape [Cin, Cout / groups, *D_in]
    # for conv, conv_weight has shape [Cout, Cin / groups, *D_in]
    is_deconv = pattern.conv.op_type == "conv_transpose"
    groups = pattern.conv.groups.val
    conv_weight = pattern.conv.weight.val
    if is_deconv:
        Cout = conv_weight.shape[1] * groups
        Cin = conv_weight.shape[0]
    else:
        Cout = conv_weight.shape[0]
        Cin = conv_weight.shape[1] * groups

    return Cin, Cout


def _is_scalar(pattern):
    # for the scalar case, the scalar can be either
    # 1. a python int/float
    # 2. a 0d numpy array
    # 3. a 1d numpy array with shape (1,)
    scale_var = pattern.scale.x if pattern.scale.x.val is not None else pattern.scale.y
    scale = scale_var.val
    is_scalar = True
    if isinstance(scale, np.ndarray):
        if scale.shape == ():
            scale = scale.tolist()
        elif scale.shape == (1) or scale.shape == (1,):
            scale = scale[0]
        else:
            is_scalar = False

    return is_scalar


def var_constraints(pattern):
    passed = True
    passed = passed and pattern.scale.x.val is not None or pattern.scale.y.val is not None
    passed = passed and pattern.conv.weight.val is not None

    is_scalar = _is_scalar(pattern)
    Cin, Cout = _cin_cout(pattern)
    scale_var = pattern.scale.x if pattern.scale.x.val is not None else pattern.scale.y
    scale = scale_var.val

    # for the vector scale case, check if the shape is broacastable
    if not is_scalar:
        conv_weight = pattern.conv.weight.val
        passed = passed and (
            np.product(scale.shape) == Cout
            or (len(scale.shape) == len(conv_weight.shape) and scale.shape[1] == Cout)
            or (len(scale.shape) == len(conv_weight.shape) - 1 and scale.shape[0] == Cout)
        )

    return passed


def transform_pattern(pattern):
    # get the scale
    scale_var = pattern.scale.x if pattern.scale.x.val is not None else pattern.scale.y
    scale = scale_var.val
    is_scalar = _is_scalar(pattern)

    # get weight and bias and groups from conv layer
    conv_weight = pattern.conv.weight.val
    conv_bias = pattern.conv.bias
    groups = pattern.conv.groups.val

    # get type of the conv layer
    is_deconv = pattern.conv.op_type == "conv_transpose"
    is_conv_1d = len(conv_weight.shape) == 3

    Cin, Cout = _cin_cout(pattern)

    # transform the scale to 1./scale for the real_div case
    if pattern.scale.op_type == "real_div":
        scale = 1.0 / scale

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

    # update the weight/bias for conv layer
    if is_scalar:
        new_conv_bias = np.array(conv_bias * scale).astype(conv_weight_type)
        new_conv_weight = np.array(conv_weight * scale).astype(conv_weight_type)

    else:
        scale = np.reshape(scale, (Cout))
        new_conv_bias = np.array(conv_bias * scale).astype(conv_weight_type)
        new_conv_weight = []
        if is_deconv:
            conv_weight = np.transpose(conv_weight, [1, 0, 2] if is_conv_1d else [1, 0, 2, 3])
            conv_weight = np.reshape(conv_weight, [Cout, Cin // groups] + list(conv_weight.shape[2:]))

        for i in range(Cout):
            _conv_weight = conv_weight[i] * scale[i]
            new_conv_weight.append(_conv_weight)
        new_conv_weight = np.array(new_conv_weight).astype(conv_weight_type)

        if is_deconv:
            new_conv_weight = np.reshape(new_conv_weight, [Cout // groups, Cin] + list(new_conv_weight.shape[2:]))
            new_conv_weight = np.transpose(new_conv_weight, [1, 0, 2] if is_conv_1d else [1, 0, 2, 3])

    # make sure the updated weight and bias have the same shape as the original ones
    assert new_conv_weight.shape == origin_weight_shape, "conv weight should have the same shape before and after the fuse_conv_scale pass."
    assert new_conv_bias.shape == origin_bias_shape, "conv bias should have the same shape before and after the fuse_conv_scale pass."

    # create a new conv op with the new weight, bias value, copying rest of the attributes
    out_name = pattern.scale.outputs[0].name
    conv_kargs = {
        "weight": new_conv_weight,
        "bias": new_conv_bias,
        "name": out_name,
        "before_op": pattern.conv,
    }

    for k, v in pattern.conv.inputs.items():
        if k in ["weight", "bias"]:
            continue
        conv_kargs[k] = v

    if is_deconv:
        x = mb.conv_transpose(**conv_kargs)
    else:
        x = mb.conv(**conv_kargs)

    pattern.scale.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=pattern.scale, old_var=pattern.scale.outputs[0], new_var=x
    )
    # Remove all the ops at once
    pattern.block.remove_ops(pattern.op_list())


if os.getenv("ENABLE_EXPERIMENTAL_PASSES") == "1":
    register_generic_pass(
        ops_arrangement=conv_scale_mul,
        var_constraints=var_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_conv_scale",
        namespace="common",
    )

    register_generic_pass(
        ops_arrangement=conv_transpose_scale_mul,
        var_constraints=var_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_conv_scale",
        namespace="common",
    )

    register_generic_pass(
        ops_arrangement=conv_scale_div,
        var_constraints=var_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_conv_scale",
        namespace="common",
    )

    register_generic_pass(
        ops_arrangement=conv_transpose_scale_div,
        var_constraints=var_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_conv_scale",
        namespace="common",
    )
