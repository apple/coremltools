# -*- coding: utf-8 -*-

#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil import Builder as mb
import numpy as np


def _try_to_transform(conv_op, scale_op, block):

    # get the scale
    if scale_op.x.val is None and scale_op.y.val is None:
        return False
    scale_var = scale_op.x if scale_op.x.val is not None else scale_op.y
    scale = scale_var.val

    # for the scalar case, the scalar can be either
    # 1. a python int/float
    # 2. a 0d numpy array
    # 3. a 1d numpy array with shape (1,)

    is_scalar = True
    if isinstance(scale, np.ndarray):
        if scale.shape == ():
            scale = scale.tolist()
        elif scale.shape == (1) or scale.shape == (1,):
            scale = scale[0]
        else:
            is_scalar = False

    # get weight and bias and groups from conv layer
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

    # for the vector scale case, check if the shape is broacastable
    if not is_scalar:
        if not np.product(scale.shape) == Cout:
            return False
        if len(scale.shape) == len(conv_weight.shape):
            if not scale.shape[1] == Cout:
                return False
        elif len(scale.shape) == len(conv_weight.shape) - 1:
            if not scale.shape[0] == Cout:
                return False
        else:
            return False

    # transform the scale to 1./scale for the real_div case
    if scale_op.op_type == "real_div":
        scale = 1./scale

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
    out_name = scale_op.outputs[0].name
    conv_kargs = {"weight": new_conv_weight, "bias": new_conv_bias, "name": out_name, "before_op": conv_op}

    for k, v in conv_op.inputs.items():
        if k in ["weight", "bias"]:
            continue
        conv_kargs[k] = v

    if is_deconv:
        x = mb.conv_transpose(**conv_kargs)
    else:
        x = mb.conv(**conv_kargs)

    scale_op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=scale_op, old_var=scale_op.outputs[0], new_var=x
    )
    # Remove all the ops at once
    block.remove_ops([conv_op, scale_op])
    return True


def _fuse_conv_scale_block(block):

    def _match_pattern(op):
        if op.op_type == "conv" or op.op_type == "conv_transpose":
            # abort fusion if op output is also a block output
            if op.outputs[0] in op.enclosing_block.outputs:
                return None
            # find batch_norm op
            child_ops = op.outputs[0].child_ops
            if len(child_ops) == 1:
                scale_op_candidate = list(child_ops)[0]
                if scale_op_candidate.op_type in ["mul", "real_div"]:
                    return scale_op_candidate
        return None

    fusion_occurred = False
    for op in list(block.operations):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = _fuse_conv_scale_block(b)
        if len(op.blocks) > 0:
            # This op can't be conv or conv_transpose
            continue

        scale_op = _match_pattern(op)
        if scale_op is not None:
            with block:
                fusion_occurred = _try_to_transform(op, scale_op, block)
            # has to break as the downstream iterator is affected.
            if fusion_occurred:
                return fusion_occurred
    return fusion_occurred


@register_pass(namespace="common")
def fuse_conv_scale(prog):
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
    for f in prog.functions.values():
        block_changed = True
        while block_changed:
            block_changed = _fuse_conv_scale_block(f)
