# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil import Builder as mb
import numpy as np
from coremltools.converters.mil.mil.types.symbolic import is_symbolic, any_symbolic

def match_pattern(op):
    if op.outputs[0] in op.enclosing_block.outputs:
        return None

    if op.op_type == "concat":
        if op.interleave.val:
            return None

        # check that axis is -3 and rank is 4
        rank = op.values[0].rank
        if rank != 4:
            return None
        axis = op.axis.val
        if axis > 0:
            axis = axis - rank
        if axis != -3:
            return None

        # check that all inputs to concat have fully defined shapes
        for in_ in op.values:
            if any_symbolic(in_.shape):
                return None

        # check that all inputs to concat have the same shape
        inshape = list(op.values[0].shape)
        for v in op.values[1:]:
            for i in range(rank):
                if inshape[i] != v.shape[i]:
                    return None

        # check that this concat is connected to exactly 1 reshape op
        child_ops = list(op.outputs[0].child_ops)
        if len(child_ops) == 1:
            if list(child_ops)[0].op_type == "reshape":
                return op
    return None

def try_to_transform(concat_op, add_op, block):
    all_ops = [concat_op]
    B, C, H, W = list(concat_op.values[0].shape)
    n = len(concat_op.values)

    # check that reshape shapes the input to (B, n, C, H, W)
    reshape_op1 = concat_op.outputs[0].child_ops[0]
    reshape_shape1 = reshape_op1.shape.val
    if reshape_shape1 is None:
        return False
    if not isinstance(reshape_shape1, np.ndarray):
        return False
    reshape_shape1 = list(reshape_shape1)
    if reshape_shape1 != [B, n, C, H, W]:
        return False
    all_ops.append(reshape_op1)

    # check that after reshape is a transpose op with perm=[0, 2, 1, 3, 4]
    if len(list(reshape_op1.outputs[0].child_ops)) != 1:
        return False
    transpose_op = list(reshape_op1.outputs[0].child_ops)[0]
    if transpose_op.op_type != 'transpose':
        return False
    perm = transpose_op.perm.val
    if perm is None:
        return
    if list(perm) != [0, 2, 1, 3, 4]:
        return False
    all_ops.append(transpose_op)

    # check that after transpose is another reshape with [B, . , H, W]
    if len(list(transpose_op.outputs[0].child_ops)) != 1:
        return False
    reshape_op2 = list(transpose_op.outputs[0].child_ops)[0]
    if reshape_op2.op_type != 'reshape':
        return False
    reshape_shape2 = reshape_op2.shape.val
    if reshape_shape2 is None:
        return False
    if not isinstance(reshape_shape2, np.ndarray):
        return False
    reshape_shape2 = list(reshape_shape2)
    if len(reshape_shape2) != 4:
        return False
    if [reshape_shape2[0], reshape_shape2[-2], reshape_shape2[-1]] != [B, H, W]:
        return False
    all_ops.append(reshape_op2)

    # check that none of the op in this pattern is connected to the output
    # (except the last mul op)
    for i, op in enumerate(all_ops):
        if i == len(all_ops) - 1:
            continue
        for out in op.outputs:
            if out in block.outputs:
                return False

    # add a new concat op
    out_name = reshape_op2.outputs[0].name
    x = mb.concat(values=concat_op.values, axis=concat_op.axis.val, interleave=True, name=out_name, before_op=concat_op)

    reshape_op2.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=reshape_op2, old_var=reshape_op2.outputs[0], new_var=x
    )

    # Remove all the ops at once
    block.remove_ops(all_ops)
    return True

def fuse_concat_interleave(block):
    fusion_status = False
    for op in list(block.operations):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = fuse_concat_interleave(b)
        if len(op.blocks) > 0:
            continue

        concat_op = match_pattern(op)
        if concat_op is not None:
            with block:
                fusion_status = try_to_transform(op, concat_op, block)
            # has to break as the downstream iterator is affected.
            if fusion_status:
                return fusion_status
    return fusion_status

@register_pass(namespace="common")
def detect_concat_interleave(prog):
    """
    Detect the pattern "concat-->reshape--->transpose--->reshape", where concat is
    along the channel axis (axis=-3), and map this pattern to the concat interleave op.

    This pattern occurs, for example, in the torchvision's shufflenet model.

    Given:
        %3 = concat(%1.a, %1.b, ..., axis=-3, interleave=False) #shape = (B, n*C, H, W)
        %4 = reshape(%3) #shape = (B, n, C, H, W)
        %5 = transpose(%4, perm=[0, 2, 1, 3, 4]) # shape = (B, C, n, H, W)
        %6 = reshape(%5) # shape = (B, C*n, H, W)

    Result:
        %6 = concat(%1.a, %1.b, ..., axis=-3, interleave=True)
    """
    for f_name, f in prog.functions.items():
        block_changed = True
        while block_changed:
            block_changed = fuse_concat_interleave(f)