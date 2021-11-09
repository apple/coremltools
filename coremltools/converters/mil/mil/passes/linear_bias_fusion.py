# -*- coding: utf-8 -*-

#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil import Builder as mb
import numpy as np

def _try_to_transform(linear_op, add_or_sub_op, block):

    if add_or_sub_op.x.val is None and add_or_sub_op.y.val is None:
        return False

    is_sub = add_or_sub_op.op_type == "sub"
    is_first_input = add_or_sub_op.x == linear_op.outputs[0]

    # compute the new bias
    linear_bias = linear_op.bias.val
    bias = add_or_sub_op.y.val if is_first_input else add_or_sub_op.x.val

    # check if the shape is broadcasable
    if np.prod(linear_bias.shape) != np.prod(bias.shape):
        return False
    Dout = linear_bias.shape[0]
    if bias.shape[-1] != Dout:
        return False
    bias = np.reshape(bias, (Dout,))

    if is_sub:
        if is_first_input:
            bias = -bias
        else:
            linear_bias = -linear_bias

    new_bias = linear_bias + bias

    # compute the new weight
    if is_sub and not is_first_input:
        new_weight = -linear_op.weight.val
    else:
        new_weight = linear_op.weight.val

    # create a new linear op with the new weight, bias value, copying rest of the attributes
    out_name = add_or_sub_op.outputs[0].name
    linear_kargs = {"weight": new_weight, "bias": new_bias, "name": out_name, "before_op": linear_op}

    for k, v in linear_op.inputs.items():
        if k in ["weight", "bias"]:
            continue
        linear_kargs[k] = v

    x = mb.linear(**linear_kargs)

    add_or_sub_op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=add_or_sub_op, old_var=add_or_sub_op.outputs[0], new_var=x
    )
    # Remove all the ops at once
    block.remove_ops([linear_op, add_or_sub_op])
    return True


def _fuse_linear_bias_block(block):

    def _match_pattern(op):
        if op.op_type == "linear":
            # abort fusion if op output is also a block output
            if op.outputs[0] in op.enclosing_block.outputs:
                return None
            # find add/sub op
            child_ops = op.outputs[0].child_ops
            if len(child_ops) == 1:
                op_candidate = list(child_ops)[0]
                if op_candidate.op_type in ["add", "sub"]:
                    return op_candidate
        return None

    fusion_occurred = False
    for op in list(block.operations):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = _fuse_linear_bias_block(b)
        if len(op.blocks) > 0:
            # This op can't be conv or conv_transpose
            continue

        add_or_sub_op = _match_pattern(op)
        if add_or_sub_op is not None:
            with block:
                fusion_occurred = _try_to_transform(op, add_or_sub_op, block)
            # has to break as the downstream iterator is affected.
            if fusion_occurred:
                return fusion_occurred
    return fusion_occurred

@register_pass(namespace="common")
class fuse_linear_bias(AbstractGraphPass):
    """
    Convert linear + add/sub to a single linear by updating the weight and bias of the linear layer.

    Example 1:
        Original:
            %4 = linear(x=%1, weight=%2, bias=%3) # %2 is a rank-2 const tensor (weight)
                                                  # %3 is a rank-1 const tensor (bias)
            ...
            %6 = add(x=%4, y=%5) # %5 is a const tensor with same shape as %3

        Result:
            %8 = linear(x=%1, weight=%2, bias=%7) # where %7 is a new const tensor with value
                                                  # %7 = %3 + %6

    Example 2:
        Original:
            %4 = linear(x=%1, weight=%2, bias=%3) # %2 is a rank-2 const tensor (weight)
                                                  # %3 is a rank-1 const tensor (bias)
            ...
            %6 = sub(x=%5, y=%4) # %5 is a const tensor with a broacasable shape with %3.
                                   i.e. if %3 has shape (Dout), %5 could be (1, Dout).

        Result:
            %9 = linear(x=%1, weight=%7, bias=%8) # where %7 is a new const tensor with value %7 = -%2
                                                  # %8 = %5 - %3
    Inputs:

        prog: Program
    """
    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = _fuse_linear_bias_block(f)
