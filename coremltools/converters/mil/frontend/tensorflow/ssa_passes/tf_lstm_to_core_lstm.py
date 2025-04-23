#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools import _logger as logger
from coremltools.converters.mil.mil import (
    Block,
    Builder as mb,
    Operation,
    Var,
)
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
    following are satisfied:

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
def _tf_lstm_to_core_lstm_block(block: Block):
    # shallow copy hides changes on f.operations during the loop
    for op in list(block.operations):
        for b in op.blocks:
            _tf_lstm_to_core_lstm_block(b)

        if op.op_type in SUPPORTED_TF_LSTM_OPS:
            if _try_replace_with_core_lstm(op):
                logger.info("Successfully map {} to lstm".format(op.op_type))
            else:
                logger.info("Unable to map {} to lstm".format(op.op_type))

def _try_get_last_cell_state_in_tf_lstm_block(op: Operation) -> Var:
    """
    Parameters
    ----------
    op: Operation
        Must have op type "tf_lstm_block"

    Returns
    -------
    Var, a var representing the last cell state in the lstm. None if check fails.

    One of the outputs of the op "tf_lstm_block" is the cell state (cs) which has shape [seq_len, batch, feat].
    That is, it is the cell state tensor of the lstm, which includes all the time steps.
    This, normally, can not be mapped to the MIL lstm op's cell state output, since that op only
    returns the last time step of the cell state, which is a tensor of shape [batch, feat].
    However, if the cell state output of "tf_lstm_block" is being sliced, before being used anywhere else,
    and sliced in such a way that it extracts just the last time step of the seq dimension, then
    it can indeed be mapped to MIL's lstm op.
    This utility function detects this condition. If true, it returns the var
    that corresponds to the rank 2 sliced cell state.

    In particular, the following pattern is detected:

    Input pattern:
    ..., cs, ... = tf_lstm_block(...) # [seq_len, batch, feat]
    extracted_cell_state = slice_by_index(x=cs, ...) # [batch, feat] or [1, batch, feat], such that seq dim. is sliced at the last time step
    out = op(extracted_cell_state)

    The "cs" var can be feeding into multiple "slice_by_index" ops, some of which slice it into [batch, feat] and
    some into [1, batch, feat] shaped tensors. This scenario is handled in the following manner:

    step 1: verify that the output "cs" only feeds into slice_by_index ops
    step 2: add a slice_by_index op to the graph, which slices the last time step and creates a
            tensor, "last_cs", of shape [batch, feat]
    step 3: add an expand_dims op to the graph which takes in "last_cs" and expands it to create
            a tensor, "expanded_last_cs", of shape [1, batch, feat]
    step 4: now, iterate over all the child ops of "cs". Each one of these will be of type "slice_by_index".
            Verify that they are slicing only the last time step. If not, exit out of the function by returning None.
            Once verified, replace its output var with either "last_cs" or "expanded_last_cs", depending on its shape.
    step 5: remove all the child ops of "cs". Return "last_cs"
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
    child_ops = cs.child_ops[:]
    block = op.enclosing_block

    # extract the last time step of the cell states
    last_cs = mb.slice_by_index(
        x=cs,
        begin=[-1, 0, 0],
        end=[-1, 0, 0],
        begin_mask=[False, True, True],
        end_mask=[False, True, True],
        squeeze_mask=[True, False, False],
        before_op=child_ops[0],
    )  # this is of shape [batch, feat]
    expanded_last_cs = mb.expand_dims(
        x=last_cs, axes=[0], before_op=child_ops[0]
    )  # shape: [1, batch, feat]

    # for each child op, which is a "slice_by_index" op, verify the following conditions:
    # - input is a rank 3 tensor, of shape [seq_len, batch, feat]
    # - output is either a rank 2 tensor of shape [batch, feat] or rank 3 of shape [1, batch, feat]
    # - the first dimension is sliced with an index that is the last index,
    #   so if its positive it should be of value, seq-1, or if negative, it should be -1
    for slice_op in child_ops:
        # if any of the input arguments of the slice op is not compile time known, the check fails early
        for input in slice_op.inputs.values():
            if input == slice_op.x:
                continue
            if input is None or input.val is None:
                return None

        x = slice_op.x
        out = slice_op.outputs[0]
        # check input rank
        if x.rank != 3:
            return None
        # check output rank and shape
        if out.rank not in (2, 3):
            return None
        if out.shape[-2:] != x.shape[-2:]:
            return None
        if out.rank == 3 and out.shape[0] != 1:
            return None

        # check that only the last time step is being extracted
        begin = slice_op.begin.val.tolist()
        end = slice_op.end.val.tolist()
        stride = slice_op.stride.val.tolist()
        begin_mask = slice_op.begin_mask.val.tolist()
        end_mask = slice_op.end_mask.val.tolist()
        squeeze_mask = slice_op.squeeze_mask.val.tolist()

        # the stride for the first dimension must be 1
        if stride[0] != 1:
            return None

        # check if the first dimension is sliced exactly for the last time step
        if is_symbolic(x.shape[0]):
            """
            When the first dimension is symbolic, we check for the following condition to be true:
            - begin[0] == -1 and begin_mask[0] == False
            If this condition is not met, we return None and exit
            """
            if begin[0] != -1 or begin_mask[0]:
                return None
        else:
            time = x.shape[0]
            begin = [i + time if i < 0 else i for i in begin]
            end = [i + time if i < 0 else i for i in end]

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

        block.replace_uses_of_var_after_op(
            anchor_op=slice_op,
            old_var=slice_op.outputs[0],
            new_var=last_cs if len(out.shape) == 2 else expanded_last_cs,
        )

    block.remove_ops(child_ops)
    return last_cs


def _try_replace_with_core_lstm(op: Operation) -> bool:
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
