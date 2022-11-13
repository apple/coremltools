#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools import _logger as logger
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.symbolic import is_symbolic


SUPPORTED_TF_LSTM_OPS = ["tf_lstm_block_cell", "tf_lstm_block"]


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

    Inputs:

        prog: Program
    """
    def apply(self, prog):
        for f in prog.functions.values():
            _tf_lstm_to_core_lstm_block(f)

@block_context_manager
def _tf_lstm_to_core_lstm_block(block):
    # shallow copy hides changes on f.operations during the loop
    for op in block.operations:
        for b in op.blocks:
            _tf_lstm_to_core_lstm_block(b)

        if op.op_type in SUPPORTED_TF_LSTM_OPS:
            if _try_replace_with_core_lstm(op):
                logger.info("Successfully map {} to lstm".format(op.op_type))
            else:
                logger.info("Unable to map {} to lstm".format(op.op_type))

def _try_get_last_cell_state_in_tf_lstm_block(op):
    """
    An utility function extracts the last cell state in a tf_lstm_block op

    Parameters
    ----------
    op: Operation
        Must have op type "tf_lstm_block"

    Returns
    -------
    Var, a var representing the last cell state in the lstm. None if check fails.

    We check if we can convert a program in which the cell states (cs) from the tf_lstm_block is used down stream.
    Since Core ML only support returning the "last" cell state, we check if the program is contructed in the pattern of

    ..., cs, ... = tf_lstm_block(...) # [seq, batch, feat]
    extracted_cell_state = slice_by_index(x=cs, ...) # [new_seq, new_batch, new_feat]
    out = op(extracted_cell_state)

    We check if the resulting extracted_cell_state is sliced only from the last time step, that is:

    (1) new_seq == 1
    (2) new_seq comes from the slice of the last time step

    In this case, we do the following transformation:

    ..., cs, ... = tf_lstm_block(...) # [seq, batch, feat]
    last_cell_state = slice_by_index(x=cs, ...) # [batch, feat]
    expand_last_cell_state = expand_dims(x=last_cell_state) # [1, batch, feat]
    extracted_cell_state = slice_by_index(x=expand_last_cell_state, ...) # [1, new_batch, new_feat]
    out = op(extracted_cell_state)

    The function returns last_cell_state, which allows the tf_lstm_core_lstm graph pass to directly use it later
    """
    if op.op_type != "tf_lstm_block":
        raise ValueError("op must have type 'tf_lstm_block'. Got {}".format(op.op_type))

    cs = op.outputs[1]
    if len(cs.child_ops) == 0 and len(cs.consuming_blocks) == 0:
        return cs
    if len(cs.consuming_blocks) > 1:
        return None
    if not all([child_op.op_type == "slice_by_index" for child_op in cs.child_ops]):
        return None

    # extract the last time step of the cell states
    child_ops = cs.child_ops[:]
    last_cs = mb.slice_by_index(
        x=cs,
        begin=[-1, 0, 0],
        end=[-1, 0, 0],
        begin_mask=[False, True, True],
        end_mask=[False, True, True],
        squeeze_mask=[True, False, False],
        before_op=child_ops[0],
    )
    expand_last_cs = mb.expand_dims(x=last_cs, axes=[0], before_op=child_ops[0])

    # for each slice_by_index op, check if the configuration only depends on the last time step
    # if so, we do an equivalent transformation using last_cs
    block = op.enclosing_block
    for slice_op in child_ops:
        # if any of the input arguments is not compile time known, the check fails early
        for input in slice_op.inputs.values():
            if input == slice_op.x:
                continue
            if input is None or input.val is None:
                return None

        # check for valid configuration
        x = slice_op.x
        begin = slice_op.begin.val.tolist()
        end = slice_op.end.val.tolist()
        stride = slice_op.stride.val.tolist()
        begin_mask = slice_op.begin_mask.val.tolist()
        end_mask = slice_op.end_mask.val.tolist()
        squeeze_mask = slice_op.squeeze_mask.val.tolist()

        # make the begin and end index positive, if negative numbers present
        if is_symbolic(x.shape[0]):
            return None
        time = x.shape[0]
        begin = [i + time if i < 0 else i for i in begin]
        end = [i + time if i < 0 else i for i in end]

        # the stride for the first dimension must be 1
        if stride[0] != 1:
            return None

        # check if the first dimension is sliced exactly for the last time step
        begin_time = 0 if begin_mask[0] else begin[0]
        end_time = time if end_mask[0] else end[0]
        if squeeze_mask[0]:
            if begin_time != time - 1:
                return None
        else:
            if end_time - begin_time != 1:
                return None
            if begin_time != time - 1:
                return None

        # add a new slice_by_index layer so that we produce the same slice transformation
        out = mb.slice_by_index(
            x=expand_last_cs,
            begin=[0]+begin[1:],
            end=[1]+end[1:],
            stride=[1]+stride[1:],
            begin_mask=[False]+begin_mask[1:],
            end_mask=[False]+end_mask[1:],
            squeeze_mask=squeeze_mask,
            before_op=slice_op,
        )
        block.replace_uses_of_var_after_op(
            anchor_op=slice_op,
            old_var=slice_op.outputs[0],
            new_var=out,
        )
    block.remove_ops(child_ops)
    return last_cs

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

    # Check for unsupported configuration : When peephole is present
    if op.use_peephole.val:
        return False

    # Check if the tf lstm op can be replaced with coreml lstm op
    # We check the following two conditions
    # (1) The outputs must not be (i, f, o, ci, co), since there is no corresponding outputs with the LSTM in Core ML
    # (2) For the tf_lstm_block op, only the last time step of cell state can be used
    #     Here is an example of valid supported configuration:
    #          _, cell_states, _, _, _, _, _, _ = tf_lstm_block.outputs
    #          output = cell_states[-1, 1:2, :]
    #     And here is an example that coreml cannot handle currently:
    #          _, cell_states, _, _, _, _, _, _ = tf_lstm_block.outputs
    #          output = cell_states[:2, :, :]
    i, cs, f, o, ci, co, h = op.outputs
    unsupported_outputs = [i, f, o, ci, co]
    if not _check_unsupported_outputs(unsupported_outputs):
        return False

    if op.op_type == "tf_lstm_block":
        cs = _try_get_last_cell_state_in_tf_lstm_block(op)
        if cs is None:
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
    # x: [seq_len, batch, input_dim]
    if op.op_type == "tf_lstm_block_cell":
        x = mb.expand_dims(x=op.x, axes=[0], before_op=op)
    elif op.op_type == "tf_lstm_block":
        x = op.x
    else:
        raise ValueError("tf lstm op {} not supported. Only {} supported".format(op.op_type, SUPPORTED_TF_LSTM_OPS))

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

    ops_to_remove = [op]
    block.replace_uses_of_var_after_op(anchor_op=op, old_var=cs, new_var=new_cs)
    if op.op_type == "tf_lstm_block_cell":
        block.replace_uses_of_var_after_op(anchor_op=op, old_var=h, new_var=new_h)
    elif op.op_type == "tf_lstm_block":
        block.replace_uses_of_var_after_op(anchor_op=op, old_var=h, new_var=new_h_all)
        if cs.op != op:
            ops_to_remove.append(cs.op)
    else:
        raise ValueError("tf lstm op {} not supported. Only {} supported".format(op.op_type, SUPPORTED_TF_LSTM_OPS))

    block.remove_ops(ops_to_remove)
    return True
