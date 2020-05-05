# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

from coremltools.converters.nnv2.nnv2_program.passes.pass_registry import register_pass
from coremltools.converters.nnv2 import CoremlBuilder as cb
from coremltools.converters.nnv2.builtin_types import builtins
import numpy as np
import logging

@register_pass(namespace='tensorflow')
def expand_tf_lstm(prog):
    """
    Expand tf_lstm_block_cell to fine-grained SSA ops following:

    xh = [x, h_prev]
    [i, ci, f, o] = xh * w + b
    f = f + forget_bias
    if not use_peephole:
      wci = wcf = wco = 0
    i = sigmoid(cs_prev .* wci + i)
    f = sigmoid(cs_prev .* wcf + f)
    ci = tanh(ci)
    cs = ci .* i + cs_prev .* f
    cs = clip(cs, cell_clip)
    o = sigmoid(cs * wco + o)
    co = tanh(cs)
    h = co .* o

    Inputs:

        prog: SsaProgram
    """
    for f_name, f in prog.functions.items():
        expand_tf_lstm_helper(f)

def expand_tf_lstm_helper(block):
    # shallow copy hides changes on f.operations during the loop
    for op in block.operations[:]:
        for b in op.blocks:
            expand_tf_lstm_helper(b)

        if op.op_type == 'tf_lstm_block_cell':
            expand_tf_lstm_block_cell(op)
            logging.info('Expanding {} (op_type: {})'.format(op.name,
                op.op_type))

def expand_tf_lstm_block_cell(op):
    if op.op_type != 'tf_lstm_block_cell':
        raise ValueError()

    with op.enclosing_block as block:
        x = op.x # [b, input_dim]
        h_prev = op.h_prev # [b, hidden_dim]
        cs_prev = op.c_prev # [b, hidden_dim]
        b = op.bias # [4*hidden_dim]
        forget_bias = op.forget_bias.val  # python:float

        # xh = [x, h_prev]
        # xh shape: [b, input_dim+hidden_dim]
        xh = cb.concat(values=[x, h_prev], axis=1, before_op=op)

        # w: [4*hidden_dim, input_dim + hidden_dim] (icfo layout)
        w = np.transpose(op.W.val)
        # [i, ci, f, o] = xh * w + b. Shape is [b, 4*hidden_dim]
        icfo = cb.linear(x=xh, weight=w, bias=b, before_op=op)

        # i, ci, f, o shape: [b, hidden_dim]
        i, ci, f, o = cb.split(x=icfo, num_splits=4, axis=1, before_op=op)
        #i, f, ci, o = cb.split(x=icfo, num_splits=4, axis=1, before_op=op)
        if op.forget_bias.val != 0:
            f = cb.add(x=f, y=forget_bias, before_op=op)

        # i = sigmoid(cs_prev .* wci + i)
        # f = sigmoid(cs_prev .* wcf + f)
        if op.use_peephole.val:
            wci = op.W_peep_i.val # [hidden_dim]
            wcf = op.W_peep_f.val # [hidden_dim]

            x = cb.mul(x=cs_prev, y=wci, before_op=op)
            pre_i = cb.add(x=x, y=i, before_op=op)

            x = cb.mul(x=cs_prev, y=wcf, before_op=op)
            pre_f = cb.add(x=x, y=f, before_op=op)
        else:
            pre_i = i
            pre_f = f

        i = cb.sigmoid(x=pre_i, before_op=op)
        f = cb.sigmoid(x=pre_f, before_op=op)

        # ci = tanh(ci)
        ci = cb.tanh(x=ci, before_op=op)

        # cs = ci .* i + cs_prev .* f
        x = cb.mul(x=ci, y=i, before_op=op)
        y = cb.mul(x=cs_prev, y=f, before_op=op)
        cs = cb.add(x=x, y=y, before_op=op)

        # cs = clip(cs, cell_clip)
        if op.cell_clip is not None:
            clip_val = op.cell_clip.val
            cs = cb.clip(x=cs, alpha=-clip_val, beta=clip_val, before_op=op)

        # o = sigmoid(cs * wco + o)
        if op.use_peephole.val:
            wco = op.W_peep_o.val
            x = cb.mul(x=cs, y=wco, before_op=op)
            pre_o = cb.add(x=x, y=o, before_op=op)
        else:
            pre_o = o
        o = cb.sigmoid(x=pre_o, before_op=op)

        # co = tanh(cs)
        co = cb.tanh(x=cs, before_op=op)

        # h = co .* o
        h = cb.mul(x=co, y=o, before_op=op)

        # Replace all outputs
        new_outputs = [i, cs, f, o, ci, co, h]
        for old_v, new_v in zip(op.outputs, new_outputs):
            block.replace_uses_of_var_after_op(anchor_op=op,
                    old_var=old_v, new_var=new_v)
        block.remove_ops([op])
