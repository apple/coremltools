#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

def _match_operation(stack_op):

    # Identify if this is an op we can transform
    if stack_op.op_type != "stack":
        return None, None

    child_ops = stack_op.outputs[0].child_ops
    if len(child_ops) != 1:
        return None, None

    if child_ops[0].op_type != "reshape":
        return None, None

    stack_axis = stack_op.inputs["axis"]
    if not stack_axis:
        return None, None
    stack_axis_val = stack_axis.val

    reshape_op = child_ops[0]

    # Now, op is a stack op followed by a reshape op
    # So we need to check that the stack really gets eliminated
    stack_output_rank = len(stack_op.outputs[0].shape)
    reshape_output_rank = len(reshape_op.outputs[0].shape)

    if stack_output_rank != (reshape_output_rank + 1):
        return None, None

    # Compare the input to stack to the output from reshape
    # These shapes should differ in either the stack_axis_val place (by a factor of 2), 
    # or in the stack_axis_val-1 place by the same factor
    input_shape = list(stack_op.inputs["values"][0].shape) 
    concat_axis = [idx for idx, (x, y) in enumerate(zip(input_shape, reshape_op.outputs[0].shape)) if x != y]
    if len(concat_axis) != 1:
        return None, None

    concat_axis = concat_axis[0]

    if input_shape[concat_axis] * 2 != reshape_op.outputs[0].shape[concat_axis]:
        return None, None

    if concat_axis != stack_axis_val and concat_axis != stack_axis_val - 1:
        return None, None

    return stack_op, reshape_op


def _replace_stack_reshape_ops(block, stack_op, reshape_op):

    stack_axis = stack_op.inputs["axis"]
    if not stack_axis:
        return None, None
    stack_axis_val = stack_axis.val

    input_shape = list(stack_op.outputs[0].shape)
    input_shape.pop(stack_axis_val)

    concat_axis = [idx for idx, (x, y) in enumerate(zip(input_shape, reshape_op.outputs[0].shape)) if x != y]
    if len(concat_axis) != 1:
        return
    concat_axis = concat_axis[0]

    interleave = (concat_axis == stack_axis_val - 1)

    with block:
        x = mb.concat(values=stack_op.values, axis=concat_axis, before_op=stack_op, interleave=interleave)

        reshape_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=stack_op, old_var=reshape_op.outputs[0], new_var=x
        )
        block.remove_ops([stack_op, reshape_op])


def _replace_stack_reshape_block(block):
    for op in list(block.operations):

        stack_op, reshape_op = _match_operation(op)

        if stack_op:
            _replace_stack_reshape_ops(block, stack_op, reshape_op)


@register_pass(namespace="common")
def replace_stack_reshape(prog):
    """
    A stack followed by a reshape layer can be replaced by a concat if the reshape 
    simply removes the new axis and doubles the size of one of the axes next to it.

    If the new axis is reshaped to the "right" (i.e. the axis just after it is 
    doubled), then we can use a concat. If it is reshaped to the "left" (the axis
    just before it is doubled), then the concat needs to set the "interleaved" flag.

    Examples: 
        Given:
            %1 = tensor(1, 5, 3, 4)
            %2 = tensor(1, 5, 3, 4)
            %3 = stack((%1,%2), axis=2) # shape = (1, 5, 2, 3, 4)
            %4 = reshape(%3, shape=[1, 10, 3, 4]) 

        Result:
            %1 = tensor(1, 5, 3, 4)
            %2 = tensor(1, 5, 3, 4)
            %4 = concat((%1,%2), axis=1, interleave=True) # shape = (1, 10, 3, 4)

        Given:
            %1 = tensor(1, 5, 3, 4)
            %2 = tensor(1, 5, 3, 4)
            %3 = stack((%1, %2), axis=1) # shape = (1, 2, 5, 3, 4)
            %4 = reshape(%3, shape=[1, 10, 3, 4]) 

        Result:
            %1 = tensor(1, 5, 3, 4)
            %2 = tensor(1, 5, 3, 4)
            %4 = concat((%1, %2), axis = 1) # shape = (1, 10, 3, 4)
    """
    for f in prog.functions.values():
        _replace_stack_reshape_block(f)
