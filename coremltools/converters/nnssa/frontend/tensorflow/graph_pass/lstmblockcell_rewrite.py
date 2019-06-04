import numpy as np
import copy

from ....builder import GraphBuilder
from ....commons import builtins
from ....commons.basic_graph_ops import replace_node, delete_node

from ..parsed_tf_node import ParsedTFNode


def Linear(builder, mul1, mul2, add, name=None):
    mul = builder.add_matmul([mul1, mul2], name=name)
    return builder.add_elementwise("Add", [mul, add], name=name)


def expand_lstm_block_cell(graph, node):
    assert (node.op == 'LSTMBlockCell')
    assert (len(node.inputs) == 8)
    peephole = node.attr['use_peephole']
    cell_clip = node.attr['cell_clip']
    forget_bias = node.attr['forget_bias']
    x, cs_prev, h_prev, w, wci, wcf, wco, b = node.inputs
    """
    input_arg {
      name: "x"
      description: "The input to the LSTM cell, shape (batch_size, num_inputs)."
      type_attr: "T"
    }
    input_arg {
      name: "cs_prev"
      description: "Value of the cell state at previous time step."
      type_attr: "T"
    }
    input_arg {
      name: "h_prev"
      description: "Output of the previous cell at previous time step."
      type_attr: "T"
    }
    input_arg {
      name: "w"
      description: "The weight matrix."
      type_attr: "T"
    }
    input_arg {
      name: "wci"
      description: "The weight matrix for input gate peephole connection."
      type_attr: "T"
    }
    input_arg {
      name: "wcf"
      description: "The weight matrix for forget gate peephole connection."
      type_attr: "T"
    }
    input_arg {
      name: "wco"
      description: "The weight matrix for output gate peephole connection."
      type_attr: "T"
    }
    input_arg {
      name: "b"
      description: "The bias vector."
      type_attr: "T"
    }

    output_arg {
      name: "i"
      description: "The input gate."
      type_attr: "T"
    }
    output_arg {
      name: "cs"
      description: "The cell state before the tanh."
      type_attr: "T"
    }
    output_arg {
      name: "f"
      description: "The forget gate."
      type_attr: "T"
    }
    output_arg {
      name: "o"
      description: "The output gate."
      type_attr: "T"
    }
    output_arg {
      name: "ci"
      description: "The cell input."
      type_attr: "T"
    }
    output_arg {
      name: "co"
      description: "The cell after the tanh."
      type_attr: "T"
    }
    output_arg {
      name: "h"
      description: "The output h vector."
      type_attr: "T"
    }

    xh = [x, h_prev]
    [i, ci, f, o] = xh * w + b
    f = f + forget_bias
    if not use_peephole:
        wci = wcf = wco = 0
    i = sigmoid(cs_prev * wci + i)
    f = sigmoid(cs_prev * wcf + f)
    ci = tanh(ci)
    cs = ci .* i + cs_prev .* f
    cs = clip(cs, cell_clip)
    o = sigmoid(cs * wco + o)
    co = tanh(cs)
    h = co .* o
    """
    builder = GraphBuilder(graph, node.name + '/', ParsedTFNode)
    one = builtins.int32()
    one.val = 1
    concat_axis = builder.add_const(one, name='concat_axis')
    xh = builder.add_concat([x, h_prev], concat_axis)
    icifo_presplit = Linear(builder, xh, w, b)
    icifo = builder.add_split(value=icifo_presplit, split_dim=concat_axis, num_split=4)
    i = builder.add_get_tuple(icifo, index=0)
    ci = builder.add_get_tuple(icifo, index=1)
    f = builder.add_get_tuple(icifo, index=2)
    o = builder.add_get_tuple(icifo, index=3)
    if forget_bias is not None and forget_bias != 0.0:
        fb = builtins.fp32()
        fb.val = forget_bias
        bias = builder.add_const(fb, name='forget_bias')
        f = builder.add_elementwise("Add", [f, bias])
    if peephole:
        i = builder.add_activation('Sigmoid', Linear(builder, cs_prev, wci, i))
        f = builder.add_activation('Sigmoid', Linear(builder, cs_prev, wcf, f))
    else:
        i = builder.add_activation('Sigmoid', i)
        f = builder.add_activation('Sigmoid', f)
    ci = builder.add_activation('Tanh', ci)
    cs = builder.add_elementwise(
        "Add",
        [builder.add_elementwise("Mul", [ci, i]),
         builder.add_elementwise("Mul", [cs_prev, f])])
    if cell_clip is not None and cell_clip > 0.0:
        cc = builtins.fp32()
        cc.val = cell_clip
        upper_clip = builder.add_const(cc, name='upper_clip')
        neg_cc = builtins.fp32()
        neg_cc.val = -cell_clip
        lower_clip = builder.add_const(neg_cc, name='lower_clip')
        cs = builder.add_elementwise('Maximum', [cs, lower_clip])
        cs = builder.add_elementwise('Minimum', [cs, upper_clip])
    if peephole:
        o = builder.add_activation('Sigmoid', Linear(builder, cs, wco, o))
    else:
        o = builder.add_activation('Sigmoid', o)
    co = builder.add_activation('Tanh', cs)
    h = builder.add_elementwise("Mul", [co, o])

    outputs = [i, cs, f, o, ci, co, h]
    for o in node.outputs:
        assert (graph[o].op == 'get_tuple')

    original_node_outputs = list(node.outputs)
    for o in original_node_outputs:
        replace_node(graph, o, outputs[graph[o].attr['index']])
        delete_node(graph, o)
    delete_node(graph, node.name)


def lstmblockcell_rewrite(nnssa):
    for i in list(nnssa.functions):
        graph = nnssa.functions[i].graph
        blockcells = [k for k, v in graph.items() if v.op == 'LSTMBlockCell']
        for b in blockcells:
            expand_lstm_block_cell(graph, graph[b])
