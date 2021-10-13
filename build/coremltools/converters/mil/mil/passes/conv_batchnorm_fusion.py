# -*- coding: utf-8 -*-

#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil import Builder as mb
import numpy as np


def _try_to_transform(conv_op, bn_op, block):

    # get parameters from batch_norm layer
    gamma = bn_op.gamma.val
    beta = bn_op.beta.val
    mean = bn_op.mean.val
    variance = bn_op.variance.val
    epsilon = bn_op.epsilon.val

    # get weight, bias and groups from conv layer
    if conv_op.weight.val is None:
        return False
    conv_weight = conv_op.weight.val
    conv_bias = conv_op.bias
    groups = conv_op.groups.val

    # get type of the conv layer
    is_deconv = conv_op.op_type == 'conv_transpose'
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
    out_name = bn_op.outputs[0].name
    conv_kargs = {"weight": new_conv_weight, "bias": new_conv_bias, "name": out_name, "before_op": conv_op}

    for k, v in conv_op.inputs.items():
        if k in ["weight", "bias"]:
            continue
        conv_kargs[k] = v

    if is_deconv:
        x = mb.conv_transpose(**conv_kargs)
    else:
        x = mb.conv(**conv_kargs)

    bn_op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=bn_op, old_var=bn_op.outputs[0], new_var=x
    )
    # Remove all the ops at once
    block.remove_ops([conv_op, bn_op])
    return True


def _fuse_conv_batchnorm_block(block):

    def _match_pattern(op):
        if op.op_type == "conv" or op.op_type == "conv_transpose":
            # abort fusion if op output is also a block output
            if op.outputs[0] in op.enclosing_block.outputs:
                return None
            # find batch_norm op
            child_ops = op.outputs[0].child_ops
            if len(child_ops) == 1:
                bn_op_candidate = list(child_ops)[0]
                if bn_op_candidate.op_type == "batch_norm":
                    return bn_op_candidate
        return None

    fusion_occurred = False
    for op in list(block.operations):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = _fuse_conv_batchnorm_block(b)
        if len(op.blocks) > 0:
            # This op can't be conv or conv_transpose
            continue

        bn_op = _match_pattern(op)
        if bn_op is not None:
            with block:
                fusion_occurred = _try_to_transform(op, bn_op, block)
            # has to break as the downstream iterator is affected.
            if fusion_occurred:
                return fusion_occurred
    return fusion_occurred


@register_pass(namespace="common")
def fuse_conv_batchnorm(prog):
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
    for f in prog.functions.values():
        block_changed = True
        while block_changed:
            block_changed = _fuse_conv_batchnorm_block(f)
