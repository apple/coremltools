#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from .helper import _check_var_scalar_value, _check_child_op_type

def _try_to_transform(reduce_sum_op, block):

    ops_to_remove = []

    # check that the dimensions in the shape of the input to the reduce_sum op,
    # over which the reduction operation is being performed, are known
    input_shape = reduce_sum_op.x.shape
    if input_shape is None:
        return False
    axes = None
    if reduce_sum_op.axes is not None:
        axes = reduce_sum_op.axes.val
    if axes is None:
        return False
    count = 1
    for dim in axes:
        if is_symbolic(input_shape[dim]):
            return False
        count *= input_shape[dim]

    # check that output of reduce_sum is not a block output
    if reduce_sum_op.outputs[0] in block.outputs:
        return False
    ops_to_remove.append(reduce_sum_op)

    # check that reduce_sum op is followed by either:
    # - mul op with scalar value 1/count
    # or
    # - real_div op with scalar value count
    if _check_child_op_type(reduce_sum_op, "mul"):
        child_op = list(reduce_sum_op.outputs[0].child_ops)[0]
        other_input = child_op.x if child_op.y == reduce_sum_op.outputs[0] else child_op.y
        if not _check_var_scalar_value(other_input, 1.0 / count, 1e-6):
            return False
    elif _check_child_op_type(reduce_sum_op, "real_div"):
        child_op = list(reduce_sum_op.outputs[0].child_ops)[0]
        if child_op.x != reduce_sum_op.outputs[0]:
            return False
        other_input = child_op.y
        if not _check_var_scalar_value(other_input, count, 1e-2):
            return False
    else:
        return False

    ops_to_remove.append(child_op)

    # remove all the ops, and replace with a reduce_mean op
    out_name = child_op.outputs[0].name
    x = mb.reduce_mean(x=reduce_sum_op.x,
                       axes = reduce_sum_op.axes.val,
                       keep_dims = reduce_sum_op.keep_dims.val,
                       name=out_name,
                       before_op=child_op)
    child_op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=child_op, old_var=child_op.outputs[0], new_var=x
    )
    block.remove_ops(ops_to_remove)
    return True


def _fuse_reduce_mean_block(block):
    fusion_status = False
    for i, op in enumerate(list(block.operations)):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = _fuse_reduce_mean_block(b)
        if len(op.blocks) > 0:
            continue

        # start pattern match if mul op is encountered
        if op.op_type == "reduce_sum":
            with block:
                fusion_status = _try_to_transform(op, block)
            # has to break as the downstream iterator is affected.
            if fusion_status:
                return fusion_status
    return fusion_status

@register_pass(namespace="common")
class fuse_reduce_mean(AbstractGraphPass):
    """
    Detect the "reduce_sum--->mul/real_div" pattern than can be mapped to reduce_mean.
    That is, the operation "reduce_sum/count == reduce_mean"

    Input graph:

                                const (scalar)
                                    |
    input ----> reduce_sum ----> mul/real_div -----------> output

    Output graph:

    input --------> reduce_mean ---------> output

    """
    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = _fuse_reduce_mean_block(f)

