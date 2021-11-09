# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from .helper import _check_child_op_type
import numpy as np
import logging

child_op_types = ["add", "sub"]

def _match_pattern(op):
    if op.op_type == "conv" or op.op_type == "conv_transpose":
        # abort fusion if op output is also a block output
        if op.outputs[0] in op.enclosing_block.outputs:
            return None
        # find add
        child_ops = op.outputs[0].child_ops
        if len(child_ops) == 1:
            add_op_candidate = list(child_ops)[0]
            if add_op_candidate.op_type in child_op_types:
                return add_op_candidate
    return None

def _try_to_transform_transpose_pattern(conv_op, block):
    ops_to_remove = []

    # conv layer
    if conv_op.op_type != "conv" and conv_op.op_type != "conv_transpose":
        return False
    is_deconv = conv_op.op_type == "conv_transpose"
    ops_to_remove.append(conv_op)

    # transpose layer
    if not _check_child_op_type(conv_op, "transpose"):
        return False
    transpose_op = list(conv_op.outputs[0].child_ops)[0]
    ops_to_remove.append(transpose_op)

    # add/sub layer
    if not _check_child_op_type(transpose_op, "add") and not _check_child_op_type(transpose_op, "sub"):
        return False
    add_or_sub_op = list(transpose_op.outputs[0].child_ops)[0]
    ops_to_remove.append(add_or_sub_op)

    # get the bias
    if add_or_sub_op.x.val is None and add_or_sub_op.y.val is None:
        return False
    bias = add_or_sub_op.x.val if add_or_sub_op.x.val is not None else add_or_sub_op.y.val
    is_first_input = add_or_sub_op.y.val is not None
    is_sub = add_or_sub_op.op_type == "sub"


    # get the conv bias/weight
    conv_shape = conv_op.outputs[0].shape
    Cout = conv_shape[1]
    conv_weight = conv_op.weight.val
    conv_weight_type = conv_weight.dtype
    conv_bias = np.zeros(Cout).astype(conv_weight_type) if conv_op.bias is None else conv_op.bias.val

    # check if the bias is compatible for fusion
    is_bias_scalar = True
    if isinstance(bias, np.ndarray):
        if bias.shape == ():
            bias = bias.tolist()
        elif np.prod(bias.shape) == 1:
            bias = np.squeeze(bias).tolist()
        else:
            is_bias_scalar = False

    if not is_bias_scalar:
        if np.prod(bias.shape) != Cout:
            return False
        rank = transpose_op.outputs[0].rank
        cout_dim = transpose_op.perm.val.tolist().index(1) - rank
        if bias.shape[cout_dim] != Cout:
            return False
        bias = np.reshape(bias, (Cout))

    # compute the new bias
    if is_sub:
        if is_first_input:
            bias = -bias
        else:
            conv_bias = -conv_bias

    new_bias = conv_bias + bias

    # compute the new weight
    if is_sub and not is_first_input:
        new_weight = -conv_weight
    else:
        new_weight = conv_weight

    # check that none of the op in this pattern is connected to the output
    # (except the last op)
    for op in ops_to_remove[:-1]:
        for out in op.outputs:
            if out in block.outputs:
                return False

    # create a new conv op with the new weight, bias value, copying rest of the attributes
    conv_kargs = {"weight": new_weight, "bias": new_bias, "before_op": conv_op}

    for k, v in conv_op.inputs.items():
        if k in ["weight", "bias"]:
            continue
        conv_kargs[k] = v

    if is_deconv:
        x = mb.conv_transpose(**conv_kargs)
    else:
        x = mb.conv(**conv_kargs)

    # create a new transpose op
    out_name = add_or_sub_op.outputs[0].name
    tranpose_kargs = {"x": x, "name": out_name, "before_op": transpose_op}
    for k, v in transpose_op.inputs.items():
        if k == "x":
            continue
        tranpose_kargs[k] = v
    x = mb.transpose(**tranpose_kargs)

    add_or_sub_op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=add_or_sub_op, old_var=add_or_sub_op.outputs[0], new_var=x
    )

    # Remove all the ops at once
    block.remove_ops(ops_to_remove)
    return True


def _try_to_transform(conv_op, add_op, block):
    if add_op.op_type == "sub":
        bias_var = add_op.y
    else:
        bias_var = add_op.x if add_op.x.val is not None else add_op.y
    bias_value = bias_var.val

    is_conv_op = (conv_op.op_type == "conv")

    # check that the bias value is a constant array or a scalar constant
    if not isinstance(bias_value, (np.ndarray, np.generic)):
        return False

    is_bias_scalar = False
    if not isinstance(bias_value, np.ndarray):
        is_bias_scalar = True

    # find rank of the conv input
    rank = conv_op.x.rank
    if rank is None:
        return False
    if not (rank == 3 or rank == 4 or rank == 5):
        return False

    # check compatibility of bias value with the rank of the conv op
    # either bias value should be a scalar or:
    # rank=3 ==> (B,C,D), which means bias must be (1,C,1) or (C,1)
    # rank=4 ==> (B,C,D1,D2), which means bias must be (1,C,1,1) or (C,1,1)
    # rank=5 ==> (B,C,D1,D2,D3), which means bias must be (1,C,1,1,1) or (C,1,1,1)

    if is_bias_scalar:
        bias_value = np.array([bias_value])
    else:
        # check that there is at most one dimension in the shape that is not 1
        if len(np.squeeze(bias_value).shape) > 1:
            return False
        # check that addition is not happening on the batch dimension
        if len(bias_value) == rank:
            if bias_value.shape[0] != 1:
                return False
        # check that last rank-2 entries in the shape vector are all 1s
        if np.prod(bias_value.shape[-(rank - 2) :]) != 1:
            return False
        bias_value = np.squeeze(bias_value)

    if add_op.op_type == "sub":
        bias_value *= -1

    # everything looks good, now find the new updated bias
    old_bias = conv_op.inputs.get("bias", None)
    old_bias_value = None
    if old_bias is not None and old_bias.val is not None:
        old_bias_value = old_bias.val
    if old_bias is None:
        # need to create a fresh numpy array for bias
        if np.prod(bias_value.shape) == 1:
            # its a scalar bias
            # need to find the value of Cout to form a new bias
            if conv_op.weight.val is None:
                return False
            # conv_transpose has weight format [K, C_out, spatial dims]
            # conv has weight format [C_out, K, spatial dims]
            Cout = conv_op.weight.val.shape[0 if is_conv_op else 1]
            new_bias_value = np.broadcast_to(bias_value, (Cout,))
        else:
            new_bias_value = bias_value
    else:
        # just need to update the existing bias array
        try:
            new_bias_value = old_bias_value + bias_value
        except:
            return False

    # create a new conv op with the new bias value, copying rest of the attributes
    out_name = add_op.outputs[0].name
    if new_bias_value.dtype != np.float32 and new_bias_value.dtype != np.float16:
        # cast the bias to match the weight type
        weight_np_type = types.nptype_from_builtin(conv_op.inputs["weight"].sym_type.get_primitive())
        logging.warning("conv_bias_fusion pass: casting bias "
                        "from {} to {} to match the dtype of the weight of the conv layer".format(
                        new_bias_value.dtype, weight_np_type
                        )
        )
        new_bias_value = new_bias_value.astype(weight_np_type)
    new_bias_var = mb.const(val=new_bias_value, before_op=conv_op)

    conv_kargs = {"bias": new_bias_var, "name": out_name, "before_op": conv_op}

    for k, v in conv_op.inputs.items():
        if k == "bias":
            continue
        conv_kargs[k] = v

    if is_conv_op:
        x = mb.conv(**conv_kargs)
    else:
        x = mb.conv_transpose(**conv_kargs)

    add_op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=add_op, old_var=add_op.outputs[0], new_var=x
    )
    # Remove all the ops at once
    block.remove_ops([conv_op, add_op])
    return True


def _fuse_conv_bias_block(block):
    fusion_status = False
    for op in list(block.operations):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = _fuse_conv_bias_block(b)
        if len(op.blocks) > 0:
            # This op can't be conv or conv_transpose
            continue

        # pattern 1 : conv + add/sub
        add_op = _match_pattern(op)
        if add_op is not None:
            with block:
                fusion_status = _try_to_transform(op, add_op, block)
            # has to break as the downstream iterator is affected.
            if fusion_status:
                return fusion_status

        # pattern 2 : conv + transpose + add/sub
        with block:
            fusion_status = _try_to_transform_transpose_pattern(op, block)
            if fusion_status:
                return fusion_status

    return fusion_status


@register_pass(namespace="common")
class fuse_conv_bias(AbstractGraphPass):
    """
    Fold add/sub into bias of conv and conv_transpose
    That is, convert conv + add/sub to conv, when add/sub is adding a constant

    There are two patterns supported now:

    Pattern 1:
    Given:
        %2 = conv(%1)
        ...
        %3 = add(%2, constant) # where constant has shape (1,C,1)/(C,1) for 1d conv, (1,C,1,1)/(C,1,1) for 2d conv etc
        ...

    Result:
        %3 = conv(%1)
        ...


    Pattern 2:
    Given:
        %2 = conv(%1)
        %3 = transpose(%2)
        ...
        %4 = add(%3, constant) # where constant has a broacasable shape
        ...

    Result:
        %2 = conv(%1)
        %4 = transpose(%2)
        ...

    """
    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = _fuse_conv_bias_block(f)
