from ....builder import GraphBuilder
from ....commons import builtins
from ....commons.basic_graph_ops import replace_node, delete_node

from ..parsed_tf_node import ParsedTFNode


def Linear(builder, mul1, mul2, add, name=None):
    mul = builder.add_matmul([mul1, mul2], name=name)
    return builder.add_elementwise("Add", [mul, add], name=name)


def expand_gru_block_cell(graph, node):
    r"""
Args
    x: Input to the GRU cell.
    h_prev: State input from the previous GRU cell.
    w_ru: Weight matrix for the reset and update gate.
    w_c: Weight matrix for the cell connection gate.
    b_ru: Bias vector for the reset and update gate.
    b_c: Bias vector for the cell connection gate.

Returns
    r: Output of the reset gate.
    u: Output of the update gate.
    c: Output of the cell connection gate.
    h: Current state of the GRU cell.

    x_h_prev = [x, h_prev]
    [r_bar u_bar] = x_h_prev * w_ru + b_ru
    r = sigmoid(r_bar)
    u = sigmoid(u_bar)
    h_prevr = h_prev \circ r
    x_h_prevr = [x h_prevr]
    c_bar = x_h_prevr * w_c + b_c
    c = tanh(c_bar)
    h = (1-u) \circ c + u \circ h_prev
    """
    assert (node.op == 'GRUBlockCell' or node.op == 'GRUBlockCellV2')
    assert (len(node.inputs) == 6)
    x, h_prev, w_ru, w_c, b_ru, b_c = node.inputs

    builder = GraphBuilder(graph, node.name + '/', ParsedTFNode)
    one = builtins.int32()
    one.val = 1
    concat_axis = builder.add_const(one, name='concat_axis')
    x_h_prev = builder.add_concat([x, h_prev], concat_axis)
    r_bar_u_bar = Linear(builder, x_h_prev, w_ru, b_ru)

    r_bar_u_bar_split = builder.add_split(value=r_bar_u_bar, split_dim=concat_axis, num_split=2)
    r_bar = builder.add_get_tuple(r_bar_u_bar_split, index=0)
    u_bar = builder.add_get_tuple(r_bar_u_bar_split, index=1)
    r = builder.add_activation('Sigmoid', r_bar)
    u = builder.add_activation('Sigmoid', u_bar)
    h_prevr = builder.add_elementwise("Mul", [h_prev, r])
    x_h_prevr = builder.add_concat([x, h_prevr], concat_axis)
    c_bar = Linear(builder, x_h_prevr, w_c, b_c)
    c = builder.add_activation('Tanh', c_bar)
    fpone = builtins.fp32()
    fpone.val = 1.0
    fpone = builder.add_const(fpone, name='fpone')
    hl = builder.add_elementwise("Mul", [builder.add_elementwise("Sub", [fpone, u]), c])
    hr = builder.add_elementwise("Mul", [u, h_prev])
    h = builder.add_elementwise("Add", [hl, hr])

    outputs = [r, u, c, h]
    for o in node.outputs:
        assert (graph[o].op == 'get_tuple')

    original_node_outputs = list(node.outputs)
    for o in original_node_outputs:
        replace_node(graph, o, outputs[graph[o].attr['index']])
        delete_node(graph, o)
    delete_node(graph, node.name)


def grublockcell_rewrite(nnssa):
    for i in list(nnssa.functions):
        graph = nnssa.functions[i].graph
        blockcells = [
            k for k, v in graph.items() if v.op == 'GRUBlockCell' or v.op == 'GRUBlockCellV2'
        ]
        for b in blockcells:
            expand_gru_block_cell(graph, graph[b])
