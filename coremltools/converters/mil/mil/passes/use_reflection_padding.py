#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


def _match_pattern(concat_op, block):
    if concat_op.op_type != "concat":
        return False

    concat_inputs = list(concat_op.inputs["values"])
    # There need to be an odd number of inputs, and at least one model has a concat input of length 1
    if len(concat_inputs) % 2 != 1 or len(concat_inputs) == 1:
        return False

    # The original input will need to be in the middle of the concatenated inputs
    original_input = concat_inputs[len(concat_inputs)//2]

    slice_ops_out = []

    end_mask = None

    begin_index = len(concat_inputs)//2

    for slice_op in concat_inputs:

        # one of the concat inputs is the original input (to the slices)
        if slice_op == original_input:
            # We'll now start checking indices from the end
            begin_index = begin_index - 2
            continue

        slice_op = slice_op.op
        if not slice_op:
            return False

        if slice_op.op_type != 'slice_by_index':
            return False

        # check that the input to slice op is the original input
        if slice_op.inputs["x"] != original_input:
            return False

        # If the slice is an output
        if slice_op.outputs[0] in block.outputs:
            return False

        if end_mask is None:
            end_mask = slice_op.inputs["end_mask"].val
            axis = list(end_mask).index(False, 0, len(end_mask))

        if end_mask is None:
            return False

        if axis != list(end_mask).index(False, 0, len(end_mask)):
            return False

        # Check that we're only taking a slice of size 1
        end = slice_op.inputs["end"].val
        begin = slice_op.inputs["begin"].val
        if end[axis] - begin[axis] != 1:
            return False

        input_shape = original_input.shape
        # Check that the slices are in order
        if begin[axis] != begin_index and begin[axis] != begin_index + input_shape[axis]:
            return False
        begin_index = begin_index - 1

        slice_ops_out.append(slice_op)

    return _replace_ops(block, concat_op, slice_ops_out, axis - len(end_mask))


def _replace_ops(block, concat_op, slice_ops, axis):

    with block:

        pad_size = len(slice_ops) // 2
        if axis == -1:
            pad = [pad_size, pad_size]
        elif axis == -2:
            pad = [pad_size, pad_size, 0, 0]
        else:
            return False

        x = mb.pad(x=slice_ops[0].inputs["x"], pad=pad, mode='reflect', before_op=concat_op)
        concat_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=concat_op, old_var=concat_op.outputs[0], new_var=x
        )

        block.remove_ops([concat_op] + slice_ops)
        return True


def _reflection_padding_block(block):
    for op in list(block.operations):
        _match_pattern(op, block)


@register_pass(namespace="common")
class use_reflection_padding(AbstractGraphPass):
    """
    Identify a reflection padding layer composed out of slices and concats.

    Input graph:
            ------------------------------------------------------------------------------------- |
            |                                                                                     v
    input(1, 2, 6, 8) ------> slice_by_index(begin=[0, 0, 0, 1], end=[0, 0, 0, 2]) -----> concat(axis=3) ---> out(1, 2, 6, 10)
            |                                                                                     ^
            ----------------> slice_by_index(begin=[0, 0, 0, -2], end=[0, 0, 0, -1]) -------------|

    Output graph:
    input(1, 2, 6, 8) -----0> pad(mode=reflect, size=[0, 0, 1, 1]) -----> out(1, 2, 6, 10)
    """

    def apply(self, prog):
        for f in prog.functions.values():
            _reflection_padding_block(f)
