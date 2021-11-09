#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.helper import _check_child_op_type

def _try_to_transform(op, block):
    op_type = op.op_type
    ops_to_remove = []
    if op.x.rank != 0 or op.y.rank != 0:
      return False

    # One and only one input is a scalar const
    if (op.x.val is None) == (op.y.val is None):
        return False

    var_1, var_2 = op.x, op.y
    ops_to_remove.append(op)

    # check if the output is consumed by exact one expand_dims op and other ops
    expand_dims_ops = []
    other_ops = []
    child_ops = list(op.outputs[0].child_ops)
    for child_op in child_ops:
        if child_op.op_type == "expand_dims":
            expand_dims_ops.append(child_op)
        else:
            other_ops.append(child_op)
    if len(expand_dims_ops) != 1:
        return False

    # check the expand_dim op has axes = [0]
    expand_dims_op = expand_dims_ops[0]
    if expand_dims_op.axes.val != [0]:
        return False
    ops_to_remove.append(expand_dims_op)
    ops_to_remove += other_ops

    for out in op.outputs:
        if out in block.outputs:
            return False

    # add a expand_dims op after each rank-0 tensor
    var_1_expand = mb.expand_dims(x=var_1, axes=[0], before_op=op)
    var_2_expand = mb.expand_dims(x=var_2, axes=[0], before_op=op)

    # add a new elementwise binary op
    elem_op = getattr(mb, op_type)

    # replace var for the expand_dims op
    x = elem_op(x=var_1_expand, y=var_2_expand, name=expand_dims_op.outputs[0].name, before_op=op)
    expand_dims_op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=expand_dims_op, old_var=expand_dims_op.outputs[0], new_var=x
    )

    # replace var for other ops
    if len(other_ops) >= 1:
        elem_op_output = op.outputs[0]
        squeeze = mb.squeeze(x=x, before_op=op)
        for other_op in other_ops:
            new_op = getattr(mb, other_op.op_type)
            kargs = {}
            for k, v in other_op.inputs.items():
                if v == elem_op_output:
                    kargs[k] = squeeze
                else:
                    kargs[k] = v
            kargs["name"] = other_op.name
            kargs["before_op"] = other_op
            new_var = new_op(**kargs)
            other_op.enclosing_block.replace_uses_of_var_after_op(
                anchor_op=other_op, old_var=other_op.outputs[0], new_var=new_var
            )

    # Remove all the ops at once
    block.remove_ops(ops_to_remove)
    return True


def _rank0_expand_dims_swap(block):
    fusion_occurred = False
    for op in list(block.operations):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = _rank0_expand_dims_swap(b)
        if len(op.blocks) > 0:
            # This op can't be elementwise binary ops
            continue

        if op.op_type in ["add", "sub", "mul", "real_div", "floor_div"]:
            with block:
                fusion_occurred = _try_to_transform(op, block)
                # has to break as the downstream iterator is affected.
                if fusion_occurred:
                    return fusion_occurred
    return fusion_occurred

@register_pass(namespace="common")
class rank0_expand_dims_swap(AbstractGraphPass):
    """
    Identify the pattern that a rank-0 binary elementwise operation followed by an expand_dims op.
    In the MIL backend, the output of the elementwise op becomes rank 1. Hence an expand_dims op
    should be added after both of the rank-0 tensor, and the final expand_dims should be removed.
    If the output var of the binary elementwise op is consumed by more then one ops, a squeeze op
    is inserted.

    Input:

        [...](rank-0) --> sub --> expand_dims (axes=[0]) --> [...]
                           ^   |
                           |   |--> op2
                           |   |
                           |   |--> op3
                           |
                     [scalar const]

    Output:
        [...](rank-0) --> expand_dims (axes=[0]) --> sub --> [...]
                                                      ^   |
                                                      |   |--> squeeze ---> op2
                                                      |                |
                                                      |                |--> op3
                                                      |
                                                expand_dims (axes=[0])
                                                      ^
                                                      |
                                                      |
                                                [scalar const]
    """

    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = _rank0_expand_dims_swap(f)
