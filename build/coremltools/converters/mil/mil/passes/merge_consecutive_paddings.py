#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from .helper import _check_child_op_type
import numpy as np


def _match_pattern(block, padding_op):

    if padding_op.op_type != "pad":
        return False

    if not _check_child_op_type(padding_op, "pad"):
        return False

    child_padding_op = list(padding_op.outputs[0].child_ops)[0]

    if padding_op.inputs["mode"].val != child_padding_op.inputs["mode"].val:
        return False

    # Ensure the paddings have the same length by prepending zeros to the shorter one 
    first_pad = padding_op.inputs["pad"].val
    child_pad = child_padding_op.inputs["pad"].val
    if len(first_pad) > len(child_pad):
        child_pad = np.insert(child_pad, 0, [0] * (len(first_pad) - len(child_pad)))
    elif len(child_pad) > len(first_pad):
        first_pad = np.insert(first_pad, 0, [0] * (len(child_pad) - len(first_pad)))
    final_pad = child_pad + first_pad
    
    if padding_op.inputs["mode"].val == "constant":
        # if the padding is constant, then the values need to be equal
        if padding_op.inputs["constant_val"].val != child_padding_op.inputs["constant_val"].val:
            return False
    else: 
        # if the padding is not constant, then we can't merge if both pads affected the same side of the image
        if any(i != 0 and j != 0 for (i,j) in zip(first_pad, child_pad)):
            return False

    return _replace_ops(block, padding_op, child_padding_op, final_pad)


def _replace_ops(block, padding_op, child_padding_op, final_pad):

    with block:

        mode = padding_op.inputs["mode"].val
        x = mb.pad(x=padding_op.inputs["x"], pad=final_pad, mode=mode, constant_val=padding_op.inputs["constant_val"].val, 
                   before_op=padding_op)
        padding_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=padding_op, old_var=child_padding_op.outputs[0], new_var=x
        )

        block.remove_ops([padding_op, child_padding_op])

    return True

def _merge_padding_block(block):
    for op in list(block.operations):
        result = _match_pattern(block, op)
        if result:
            return True
    return False           

@register_pass(namespace="common")
def merge_consecutive_paddings(prog):
    """
    Identify two consecutive 'pad' layers which could be merged into a single 'pad' layer. This is possible when 
    - the paddings are "constant" and have the same "constant_val"
    - OR, the paddings act along different axes. 

    Input graph: 
    input(1, 2, 6, 8) ------> pad([1, 1], mode='reflect) -----> pad([1, 1, 0, 0], mode='reflect') ---> out(1, 2, 8, 10)

    Output graph: 
    input(1, 2, 6, 8) ------> pad([1, 1, 1, 1], mode='reflect) ---> out(1, 2, 8, 10)
    """
    for f in prog.functions.values():
        block_changed = True 
        while block_changed:
            block_changed = _merge_padding_block(f)
