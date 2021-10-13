#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


def _match_pattern(op):

    # Identify if this is an op we can transform
    if op.op_type != "concat":
        return None

    w_concat = op
    if w_concat.inputs["values"][0].rank != 4:
        return None

    if w_concat.inputs["axis"].val != 3:
        return None
    if not w_concat.inputs["interleave"].val:
        return None

    inputs = list(w_concat.inputs["values"])
    if len(inputs) != 2:
        return None
    
    if not inputs[0].op or not inputs[1].op:
        return None

    if inputs[0].op.op_type != "concat" or inputs[1].op.op_type != "concat":
        return None

    h_concat_0 = inputs[0].op
    if not h_concat_0.inputs["interleave"].val:
        return None

    h_concat_0_inputs = list(h_concat_0.inputs["values"])
    if len(h_concat_0_inputs) != 2:
        return None

    h_concat_1 = inputs[1].op
    if not h_concat_1.inputs["interleave"].val:
        return None

    h_concat_1_inputs = list(h_concat_1.inputs["values"])
    if len(h_concat_1_inputs) != 2:
        return None

    if h_concat_0.inputs["axis"].val != 2 or h_concat_1.inputs["axis"].val != 2:
        return None

    return w_concat, h_concat_0, h_concat_1


def _replace_ops(block, w_concat, h_concat_0, h_concat_1):

    with block:
        h_concat_0_inputs = list(h_concat_0.inputs["values"])
        h_concat_1_inputs = list(h_concat_1.inputs["values"])

        all_inputs = [h_concat_0_inputs[0], h_concat_1_inputs[0], h_concat_0_inputs[1], h_concat_1_inputs[1]]

        # Concatenate all 4 inputs on the channel axis
        x = mb.concat(values=all_inputs, axis=1, before_op=h_concat_0, interleave=True)
        # Shuffle into place
        x = mb.pixel_shuffle(x=x, upscale_factor=2, before_op=h_concat_0)

        w_concat.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=h_concat_0, old_var=w_concat.outputs[0], new_var=x
        )

        block.remove_ops([w_concat, h_concat_0, h_concat_1])


def _concat_to_pixel_shuffle_block(block):
    for op in list(block.operations):

        layers = _match_pattern(op)
        if layers:
            _replace_ops(block, layers[0], layers[1], layers[2])
           

@register_pass(namespace="common")
def concat_to_pixel_shuffle(prog):
    """
    Identify nested, interleaved concats which can be replaced by a single concat and a pixel shuffle layer. 

    This pattern occurs with the faster up-convolution from the FCRN model (Laina et al., 2016). 

    input(N, C, H, W) -------------------
                                        |
                                        v
    input(N, C, H, W) -----> concat(axis=2, interleave=True) -----> concat(axis=3, interleave=True) ----> output
                                                                                ^
                                                                                |
    input(N, C, H, W) -----> concat(axis=2, interleave=True) --------------------
                |                       ^
                |                       |
    input(N, C, H, W) -------------------



    input(N, C, H, W) ---------------
                                    |
                                    v
    input(N, C, H, W) -----> concat(axis=1, interleave=True) -----> pixel_shuffle(upscale_factor=2) ----> output
                                    ^                
                                    |                
    input(N, C, H, W) --------------|
                                    |
                                    |
    input(N, C, H, W) ---------------
    """
    for f in prog.functions.values():
        _concat_to_pixel_shuffle_block(f)
