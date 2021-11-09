# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
import logging


@register_pass(namespace="tensorflow")
class tf_lstm_to_core_lstm(AbstractGraphPass):
    """
    Try to map TF dialect ops `tf_lstm_block` and `tf_lstm_block_cell` to
    `lstm` in the core op set if compatible. They are compatible if all of the
    followings are satisfied:

    - If tf_lstm_block: only h output is consumed. tf_lstm_block has 7
      sequence outputs: [i, cs, f, o, ci, co, h]. Each of them (e.g., i) has
      shape [seq_len, batch, hidden_dim] (see tf_lstm_block op doc string).
      core lstm only supports sequence output for hidden state h, and thus if
      any outputs other than `h` is consumed, we cannot convert to lstm in the
      core op set.

    - If tf_lstm_block_cell: only cs, h output (outputs[1], outputs[6])
      are consumed. Similar to above.

    - batch size == 1

    Inputs:

        prog: Program
    """
    def apply(self, prog):
        for f in prog.functions.values():
            _tf_lstm_to_core_lstm_block(f)


def _tf_lstm_to_core_lstm_block(block):
    # shallow copy hides changes on f.operations during the loop
    for op in block.operations:
        for b in op.blocks:
            _tf_lstm_to_core_lstm_block(b)

        if op.op_type in ["tf_lstm_block_cell", "tf_lstm_block"]:
            if _try_replace_with_core_lstm(op):
                logging.info("Successfully map {} to lstm".format(op.op_type))
            else:
                logging.info("Unable to map {} to lstm".format(op.op_type))


def _try_replace_with_core_lstm(op):
    """
    Inputs:

    op (Operation): op.op_type must be 'tf_lstm_block_cell' or `tf_lstm_block`

    Returns:

    True if op can be represented by mb.lstm op in SSA. False otherwise
    """
    def _check_unsupported_outputs(unsupported_outputs):
        for ov in unsupported_outputs:
            if len(ov.child_ops) > 0 or len(ov.consuming_blocks) > 0:
                return False
        return True


    if op.op_type == "tf_lstm_block_cell":
        batch = op.x.shape[0]
    else:  # tf_lstm_block
        batch = op.x.shape[1]

    # Check for unsupported configuration : When peephole is present
    if op.use_peephole.val:
        return False

    # Check if tf_lstm_block_cell can be replaced with lstm op
    i, cs, f, o, ci, co, h = op.outputs
    if op.op_type == "tf_lstm_block_cell":
        unsupported_outputs = [i, f, o, ci, co]  # only cs, h are supported
    else:  # tf_lstm_block
        unsupported_outputs = [i, cs, f, o, ci, co]  # only h is supported
    if not _check_unsupported_outputs(unsupported_outputs):
        return False

    # op is compatible with lstm
    hidden_dim = op.c_prev.shape[1]

    mb_peep = None
    if op.use_peephole.val:
        mb_peep = np.stack(
            [op.weight_peep_i.val, op.weight_peep_f.val, op.weight_peep_o.val]
        )

    # Set weights. The layout of the weight in TF1 is icfo (input, cell, forget, output gate).
    # Need to convert to ifoc for coreml
    tf_w = op.weight.val  # [input_dim+hidden_dim, 4*hidden_dim] in icfo layout
    tf_w_i, tf_w_c, tf_w_f, tf_w_o = np.split(tf_w, 4, axis=1)
    w = np.concatenate([tf_w_i, tf_w_f, tf_w_o, tf_w_c], axis=1)
    w = np.transpose(w, [1, 0])
    hidden_dim = w.shape[0] // 4
    input_dim = w.shape[1] - hidden_dim
    # Split input and hidden weights
    w_ih, w_hh = np.split(w, [input_dim], axis=1)

    # Bias is icfo. Convert to ssa LSTM's ifoc layout
    tf_b = op.bias.val
    tf_b_i, tf_b_c, tf_b_f, tf_b_o = np.split(tf_b, 4, axis=0)
    tf_b_f += op.forget_bias.val  # add forget bias to bias
    bias = np.concatenate([tf_b_i, tf_b_f, tf_b_o, tf_b_c], axis=0)

    cell_clip = None if op.cell_clip is None else op.cell_clip.val

    output_sequence = op.op_type == "tf_lstm_block"

    block = op.enclosing_block
    with block:
        # x: [seq_len, batch, input_dim]
        if op.op_type == "tf_lstm_block_cell":
            x = mb.expand_dims(x=op.x, axes=[0], before_op=op)
        else:  # tf_lstm_block
            x = op.x
        new_h_all, new_h, new_cs = mb.lstm(
            x=x,
            initial_c=op.c_prev,
            initial_h=op.h_prev,
            weight_ih=w_ih,
            weight_hh=w_hh,
            bias=bias,
            recurrent_activation="sigmoid",
            cell_activation="tanh",
            activation="tanh",
            peephole=mb_peep,
            clip=cell_clip,
            output_sequence=output_sequence,
            name=op.name,
            before_op=op,
        )

    if op.op_type == "tf_lstm_block_cell":
        block.replace_uses_of_var_after_op(anchor_op=op, old_var=cs, new_var=new_cs)
        block.replace_uses_of_var_after_op(anchor_op=op, old_var=h, new_var=new_h)
    else:  # 'tf_lstm_block'
        block.replace_uses_of_var_after_op(anchor_op=op, old_var=h, new_var=new_h_all)
    block.remove_ops([op])
    return True
