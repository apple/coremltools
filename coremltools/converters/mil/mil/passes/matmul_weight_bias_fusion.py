# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import numpy as np
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil import Builder as mb

child_op_types = ["add", "sub"]

def _match_pattern(op):

    if op.op_type == "matmul":
        # find add
        child_ops = op.outputs[0].child_ops
        if len(child_ops) == 1:
            add_op_candidate = list(child_ops)[0]
            if add_op_candidate.op_type in child_op_types:
                return add_op_candidate
    return None

def _transpose(v, before_op):
    """
    Transpose the last 2 dims.
    v: Var (must be a tensor)
    """
    perm = list(range(v.rank))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return mb.transpose(x=v, perm=perm, before_op=before_op)

def _try_to_transform(matmul_op, add_op, block):
    if matmul_op.x.val is None and matmul_op.y.val is None:
        # This is a dynamic matmul.
        return False
    if add_op.x.val is None and add_op.y.val is None:
        # This is a dynamic add.
        return False

    x_is_weight = matmul_op.x.val is not None
    if x_is_weight:
        weight, linear_x = matmul_op.x, matmul_op.y
        transpose_weight = matmul_op.transpose_x.val
        transpose_x = matmul_op.transpose_y.val
    else:
        weight, linear_x = matmul_op.y, matmul_op.x
        transpose_weight = matmul_op.transpose_y.val
        transpose_x = matmul_op.transpose_x.val

    if linear_x.rank < 2 or weight.rank != 2:
        # We don't support these cases yet.
        return False

    # For those weights which are the input for more than one op,
    # we don't do the fusion.
    # The reason is that it might cause memory explosion by adding
    # those weight as a numpy array in the inner product or
    # the batch_mat_mul kernel.
    if len(weight.child_ops) > 1:
        return False

    d_out = weight.shape[1] if not transpose_weight else weight.shape[0]
    bias = add_op.x.val if add_op.x.val is not None else add_op.y.val
    if len(bias.shape) > 1:
        if any([d != 1 for d in bias.shape[:-1]]):
            return  # cannot transform

        # squeeze leading dims of size 1
        bias = np.squeeze(bias)

    if len(bias.shape) != 1 or bias.shape[0] != d_out:
        return  # cannot transform

    if add_op.op_type == "sub":
        bias = -bias
    out_name = add_op.outputs[0].name

    if x_is_weight:
        # If transpose_x == transpose_weight == False:
        # w*x = (x^T w^T)^T = linear(x^T, w)^T
        x_transposed = (
            _transpose(linear_x, before_op=matmul_op) if not transpose_x else linear_x
        )
        w_no_transpose = (
            weight if not transpose_weight else _transpose(weight, before_op=matmul_op)
        )
        x = mb.linear(
            x=x_transposed, weight=w_no_transpose, bias=bias, before_op=matmul_op
        )
        x = _transpose(x, before_op=matmul_op, name=out_name)
    else:
        # If transpose_x == transpose_weight == False
        # x*w = x*(w^T)^T = linear(x, w^T)
        x_no_transpose = (
            _transpose(linear_x, before_op=matmul_op) if transpose_x else linear_x
        )
        w_transposed = (
            weight if transpose_weight else _transpose(weight, before_op=matmul_op)
        )
        x = mb.linear(
            x=x_no_transpose,
            weight=w_transposed,
            bias=bias,
            before_op=matmul_op,
            name=out_name,
        )

    add_op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=add_op, old_var=add_op.outputs[0], new_var=x
    )
    # Remove all the ops at once
    block.remove_ops([matmul_op, add_op])
    return True


def _fuse_matmul_weight_bias_block(block):
    fusion_status = False
    for op in list(block.operations):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = _fuse_matmul_weight_bias_block(b)
        if len(op.blocks) > 0:
            # This op can't be matmul
            continue

        add_op = _match_pattern(op)
        if add_op is not None:
            with block:
                fusion_status = _try_to_transform(op, add_op, block)
            # has to break as the downstream iterator is affected.
            if fusion_status:
                return fusion_status
    return fusion_status

@register_pass(namespace="common")
class fuse_matmul_weight_bias(AbstractGraphPass):
    """
    Convert matmul + add/sub to linear whenever possible.

    Given:
        %3 = matmul(x=%1, y=%2)  # %1 or %2 is const and rank 2 (weight)
        ...
        %5 = add(x=%3, y=%4) # %4 is const. add(x=%4, y=%3) is equivalent
                             # sub is similar.

    Result:
        # assuming %2 above is const and rank 2
        %5 = linear(x=%1, weight=%2, bias=%4)

    Inputs:

        prog: Program
    """
    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = _fuse_matmul_weight_bias_block(f)
