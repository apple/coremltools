# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil import Builder as mb
import numpy as np


def _remove_elementwise_binary(op, block, x, y):
    # We remove the ops that has op.x == x or op.y == y
    if x is not None and op.x.val is not None and np.all(op.x.val == x):
        input_var = op.y
        input_op = input_var.op
    elif y is not None and op.y.val is not None and np.all(op.y.val == y):
        input_var = op.x
        input_op = input_var.op
    else:
        return False

    input_shape = input_var.sym_type
    output_shape = op.outputs[0].sym_type

    # We might be using elementwise as broadcasting
    if input_shape != output_shape:
        return False

    op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=input_op, old_var=op.outputs[0], new_var=input_var
    )
    block.remove_ops([op])

    return True


def remove_elementwise(op, block):

    if op.op_type in {"add"}:
        return _remove_elementwise_binary(op, block, 0, 0)
    elif op.op_type in {"mul"}:
        return _remove_elementwise_binary(op, block, 1, 1)
    elif op.op_type in {"floor_div", "pow", "real_div"}:
        return _remove_elementwise_binary(op, block, None, 1)
    elif op.op_type in {"sub"}:
        return _remove_elementwise_binary(op, block, None, 0)
    else:
        return False


def remove_same_shape(op, block):
    input_shape = op.x.sym_type
    output_shape = op.outputs[0].sym_type

    if input_shape != output_shape:
        return False

    input_var = op.x
    input_op = input_var.op

    op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=input_op, old_var=op.outputs[0], new_var=input_var
    )

    # Remove all the ops at once
    block.remove_ops([op])
    return True


def remove_linear(op, block):
    if op.alpha.val != 1 or op.beta.val != 0:
        return False

    input_var = op.x
    input_op = input_var.op

    op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=input_op, old_var=op.outputs[0], new_var=input_var
    )

    # Remove all the ops at once
    block.remove_ops([op])
    return True


_SUPPORTED_OPS = {
    "add",
    "mul",
    "floor_div",
    "pow",
    "real_div",
    "sub",
    "reshape",
    "split",
    "slice_by_index",
    "slice_by_size",
    "pad",
    "tile",
    "upsample_nearest_neighbor",
    "upsample_bilinear",
    "resize_bilinear",
    "crop",
    "linear_activation"
}
op_to_removal_fn = {
    "add": remove_elementwise,
    "mul": remove_elementwise,
    "floor_div": remove_elementwise,
    "pow": remove_elementwise,
    "real_div": remove_elementwise,
    "sub": remove_elementwise,
    "reshape": remove_same_shape,
    "split": remove_same_shape,
    "slice_by_index": remove_same_shape,
    "slice_by_size": remove_same_shape,
    "pad": remove_same_shape,
    "tile": remove_same_shape,
    "upsample_nearest_neighbor": remove_same_shape,
    "upsample_bilinear": remove_same_shape,
    "resize_bilinear": remove_same_shape,
    "crop": remove_same_shape,
    "linear_activation": remove_linear,
}


def match_pattern(op):
    # abort if op output is a block output
    if op.outputs[0] in op.enclosing_block.outputs:
        return None

    if op.op_type in _SUPPORTED_OPS:

        if len(op.outputs) != 1:
            return None
        return op_to_removal_fn[op.op_type]

    return None


def noop_elimination_block(block):
    for op in list(block.operations):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = noop_elimination_block(b)
        if len(op.blocks) > 0:
            continue

        remove_fn = match_pattern(op)
        if remove_fn is not None:
            with block:
                status = remove_fn(op, block)
            # has to break as the downstream iterator is affected.
            if status:
                return status
    return False


@register_pass(namespace="common")
def noop_elimination(prog):
    """
    We remove ops that has no effect.

    Given:
        %1 (1, 96, 128, 64, fp32) = ...
        %2 (1, 96, 128, 64, fp32) = reshape(%1)
        ...
        %3 (1, 96, 128, 64, fp32) = add(%2, constant)
        ...

    Result:
        %1 (1, 96, 128, 64, fp32) = ...
        %3 (1, 96, 128, 64, fp32) = add(%1, constant)
        ...

    """
    for f_name, f in prog.functions.items():
        block_changed = True
        while block_changed:
            block_changed = noop_elimination_block(f)
