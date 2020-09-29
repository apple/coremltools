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
import copy

def match_pattern(op):
    ret = set([])
    child_ops = op.outputs[0].child_ops

    for child_op in child_ops:
        if child_op.op_type != "transpose":
            continue
        skip_ops = child_op.outputs[0].child_ops
        for skip_op in skip_ops:
            if "conv" not in skip_op.op_type: 
                continue
            ret.update([child_op])

    return ret if len(ret) != 0 else None


def try_to_transform(pad_op, transpose_ops, block):

    def _compute_new_pad_values(transpose_op):
        if pad_op.inputs["pad"].val is None:
            return None
        pad_amounts = np.reshape(pad_op.inputs["pad"].val, [-1, 2])
        transpose_axes = transpose_op.inputs["perm"].val
        rank_diff = len(transpose_axes) - pad_amounts.shape[0]
        pad_amounts_new = copy.deepcopy(pad_amounts)
        # append "rank_diff" rows of zeros to the top
        pad_amounts_new = np.concatenate(
            (np.zeros((2 * rank_diff)).reshape(-1, 2), pad_amounts_new)
        )
        pad_amounts_new = pad_amounts_new.astype(pad_amounts.dtype)
        pad_amounts = np.concatenate(
            (np.zeros((2 * rank_diff)).reshape(-1, 2), pad_amounts)
        )
        for i, axis in enumerate(transpose_axes):
            pad_amounts_new[i][0] = pad_amounts[axis][0]
            pad_amounts_new[i][1] = pad_amounts[axis][1]

        # get the top "rank_diff" rows
        top_rows = pad_amounts_new[:rank_diff, :]
        if not np.all(top_rows == 0):
            return False
        # cut "rank_diff" from the top
        pad_amounts_new = pad_amounts_new[rank_diff:, :]
        pad_amounts_new = pad_amounts_new.flatten()
        return pad_amounts_new

    if pad_op.outputs[0] in pad_op.enclosing_block.outputs:
        return False
    if len(set(pad_op.outputs[0].child_ops)) != len(transpose_ops):
        return False

    for transpose_op in transpose_ops:
        pad_amounts_new = _compute_new_pad_values(transpose_op)
        if pad_amounts_new is None:
            continue

        with pad_op.enclosing_block:
            new_transpose_var = mb.transpose(x=pad_op.inputs["x"], perm=transpose_op.inputs["perm"].val, before_op=transpose_op)
            new_pad_inputs = {"x": new_transpose_var, "pad": pad_amounts_new}
            for k, v in pad_op.inputs.items():
                if k not in new_pad_inputs:
                    new_pad_inputs[k] = v
            new_pad_var = mb.pad(before_op=transpose_op, **new_pad_inputs) 
        pad_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=transpose_op, old_var=transpose_op.outputs[0], new_var=new_pad_var
        )

    pad_op.enclosing_block.remove_ops(list(transpose_ops) + [pad_op])

    return True

def pad_conv_connect_block(block):
    fusion_status = False
    for op in list(block.operations):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = pad_conv_connect_block(b)

        if op.op_type != "pad":
            continue

        transpose_ops = match_pattern(op)
        if transpose_ops is not None:
            with block:
                fusion_status = try_to_transform(op, transpose_ops, block)
            # has to break as the downstream iterator is affected.
            if fusion_status:
                return fusion_status
    return fusion_status


@register_pass(namespace="common")
def pad_conv_connect(prog):
    """
    When we observe pad -> transpose -> conv, we move the pad to be next to conv.
    This allows us to meld pad + conv if possible.

    Given:
        %1 = pad(%0, ...)
        %2 = transpose(%1, ...)
        %3 = conv(%2, ...)
        ...

    Result:
        %1.a = transpose(%0, ...)
        $2.a = pad(%1.a, ...)
        %3 = conv(%2.a)
        ...

    """
    for f_name, f in prog.functions.items():
        block_changed = True
        while block_changed:
            block_changed = pad_conv_connect_block(f)
