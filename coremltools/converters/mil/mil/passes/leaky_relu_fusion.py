#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil import Builder as mb
from .helper import _check_var_scalar_value_in_interval, _check_child_op_type

def _try_to_transform(mul_op, block):

    ops_to_remove = []

    # check that one of the inputs of the mul op is a constant that is between 0 and 1
    if _check_var_scalar_value_in_interval(mul_op.x, 0, 1):
        alpha_input_var = mul_op.x
        parent_var = mul_op.y
    elif _check_var_scalar_value_in_interval(mul_op.y, 0, 1):
        alpha_input_var = mul_op.y
        parent_var = mul_op.x
    else:
        return False

    # check that output of mul is not a block output
    if mul_op.outputs[0] in block.outputs:
        return False
    ops_to_remove.append(mul_op)

    # check if the child op of the mul op is maximum
    if not _check_child_op_type(mul_op, "maximum"):
        return False

    # check that the other input of the max op is same as the parent of the mul op
    max_op = list(mul_op.outputs[0].child_ops)[0]
    if not (
        (max_op.x == mul_op.outputs[0] and max_op.y == parent_var)
        or (max_op.y == mul_op.outputs[0] and max_op.x == parent_var)
    ):
        return False
    ops_to_remove.append(max_op)

    # remove all the ops, and replace with a leaky relu op
    out_name = max_op.outputs[0].name
    x = mb.leaky_relu(x=parent_var, alpha=alpha_input_var.val, name=out_name, before_op=max_op)
    max_op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=max_op, old_var=max_op.outputs[0], new_var=x
    )
    block.remove_ops(ops_to_remove)
    return True


def _fuse_leaky_relu_block(block):
    fusion_status = False
    for i, op in enumerate(list(block.operations)):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = _fuse_leaky_relu_block(b)
        if len(op.blocks) > 0:
            continue

        # start pattern match if mul op is encountered
        if op.op_type == "mul":
            with block:
                fusion_status = _try_to_transform(op, block)
            # has to break as the downstream iterator is affected.
            if fusion_status:
                return fusion_status
    return fusion_status

@register_pass(namespace="common")
class fuse_leaky_relu(AbstractGraphPass):
    """
    Detect the "mul--->max" pattern than can be mapped to leaky relu

    In code form:
    ------------

    Input:
        %2 = const(value = alpha) # where 0 <= alpha <= 1
        %3 = mul(%1, %2) # alpha * x
        %4 = max(%3, %1) # max(alpha * x, x)

    Output:
        %4 = leaky_relu(x=%1, alpha=%2)


    In graphical form:
    -----------------

    Input graph:

            const (val = alpha)
                |
    input ----> mul ---------------> maximum -----------> output
      |                                 |
      |----------------------------------

    Output graph:

    input --------> leaky_relu ---------> output

    """
    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = _fuse_leaky_relu_block(f)
