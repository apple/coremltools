# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import numpy as np
from ...commons import builtins
from ...commons.basic_graph_ops import disconnect_edge, connect_edge, \
    delete_node, replace_node, connect_dests
from ...nnssa import ParsedNode

ELEMENTWISE_OPS = [
    'Maximum',
    'Minimum',
    'Add',
    'Sub',
    'BiasAdd',
    'Mul',
    'Sigmoid',
    'Relu',
    'LeakyRelu',
    'Tanh',
    'Identity',
    'Sqrt',
    'Rsqrt',
    'Pow',
]

def _is_NHWC(graph, node):
    if (node.op == 'ResizeBilinear' or node.op == 'ResizeNearestNeighbor'):
        return True
    if (node.op == 'Conv2D' or node.op == 'Pooling' or node.op =='MaxPool' or \
        node.op == 'AvgPool') and node.attr.get('data_format') == 'NHWC':
        return True
    if node.op == 'ConcatV2':
        # ConcatV2's last input is axis
        return all(graph[inp].attr.get('data_format') == 'NHWC' for inp in
            node.inputs[:-1])
    if node in ELEMENTWISE_OPS:
        return all(graph[inp].attr.get('data_format') == 'NHWC' for inp in
            node.inputs)
    return False


def _insert_transpose_to_nchw(graph, src, dst):

    tp_node_name = src.name + "_to_nchw"
    tp_node = ParsedNode()
    tp_node.op = 'Transpose'
    tp_node.name = tp_node_name

    # Adjust type inference
    if builtins.is_tensor(src.datatype):
        s = src.datatype.get_shape()
        tp_shape = tuple([s[0], s[3], s[1], s[2]])
        tp_node.datatype = builtins.tensor(src.datatype.get_primitive(), tp_shape)

    tp_node.inputs = [src.name]
    tp_node.outputs = [dst.name]
    tp_node.attr['dim'] = [0,3,1,2]
    input_shape = src.attr['_output_shapes'][0]
    n,h,w,c = input_shape
    tp_node.attr['_output_shapes'] = [[n,c,h,w]]
    graph[tp_node_name] = tp_node

    # Rename dst's input 'src' to 'tp_node'
    for idx, inp in enumerate(dst.inputs):
        if inp == src.name:
            dst.inputs[idx] = tp_node_name

    # Rename src's output from 'dst' to 'tp_node'
    for idx, outp in enumerate(src.outputs):
        if outp == dst.name:
            src.outputs[idx] = tp_node_name


def _insert_transpose_from_nchw(graph, src, dst):

    tp_node_name = src.name + "_to_nhwc"
    tp_node = ParsedNode()
    tp_node.op = 'Transpose'
    tp_node.name = tp_node_name

    # Adjust type inference
    if builtins.is_tensor(src.datatype):
        s = src.datatype.get_shape()
        tp_shape = tuple([s[0], s[2], s[3], s[1]])
        tp_node.datatype = builtins.tensor(src.datatype.get_primitive(), tp_shape)

    tp_node.inputs = [src.name]
    tp_node.outputs = [dst.name]
    tp_node.attr['dim'] = [0,2,3,1]
    input_shape = src.attr['_output_shapes'][0]
    n,c,h,w = input_shape
    tp_node.attr['_output_shapes'] = [[n,h,w,c]]
    graph[tp_node_name] = tp_node

    # Rename dst's input 'src' to 'tp_node'
    for idx, inp in enumerate(dst.inputs):
        if inp == src.name:
            dst.inputs[idx] = tp_node_name

    # Rename src's output from 'dst' to 'tp_node'
    for idx, outp in enumerate(src.outputs):
        if outp == dst.name:
            src.outputs[idx] = tp_node_name


def transform_nhwc_to_nchw(nnssa):
    """
    Mark each one of the node with "NHWC", so that the conversion process
    could avoid inserting unnecessary transpositions.
    A node's format is "NHWC" if and only if:
    (1) it is a conv or pooling or image_resize layer with "NHWC" data format
    (2) it is a rank-preserving operation whose inputs are all "NHWC"
    """
    for fn_key in list(nnssa.functions.keys()):
        graph = nnssa.functions[fn_key].graph
        node_names = list(graph.keys())

        # Mark all NHWC nodes
        nhwc_nodes = []
        for name in node_names:
            node = graph[name]
            if len(node.outputs) > 0 and len(node.inputs) > 0 and _is_NHWC(graph, node):
                node.attr['data_format'] = 'NHWC'
                nhwc_nodes.append(name)

        for name in nhwc_nodes:
            node = graph[name]
            orig_out_shapes = node.attr['_output_shapes']

            # Adjust type inference
            if builtins.is_tensor(node.datatype):
                s = node.datatype.get_shape()
                new_shape = tuple([s[0], s[3], s[1], s[2]])
                node.datatype = builtins.tensor(node.datatype.get_primitive(), new_shape)

            # Insert NHWC->NCHW tranpose
            for i, inp_node_name in enumerate(node.inputs):
                inp_node_format = graph[inp_node_name].attr.get('data_format')
                if len(node.inputs) == 2 and i == 1 and graph[inp_node_name].op == 'Const':
                    # Const weights and parameters
                    continue
                if inp_node_format != 'NHWC':
                    _insert_transpose_to_nchw(graph, graph[inp_node_name], node)

            # Insert NCHW->NHWC tranpose
            for i, out_node_name in enumerate(node.outputs):
                out_node_format = graph[out_node_name].attr.get('data_format')
                if out_node_format != 'NHWC':
                    _insert_transpose_from_nchw(graph, node, graph[out_node_name])

            # Adjust output shape and concat layer's axis parameter
            node.attr['_output_shapes'] = [[s[0], s[3], s[1], s[2]] for s in orig_out_shapes]
            if node.op == 'ConcatV2' and len(node.inputs) > 1 and graph[node.inputs[-1]].value is not None:
                axis = graph[node.inputs[-1]].value.val
                axis = 4 + axis if axis < 0 else axis
                if axis == 3:
                    node.attr['axis'] = 1
                elif axis == 2 or axis == 1:
                    node.attr['axis'] = axis + 1
                else:
                    node.attr['axis'] = axis


def fuse_bias_add(nnssa):
    # look for 'BiasAdd' nodes following 'MatMul' or 'Conv2D'. If the other input in
    # 'BiasAdd' is coming from a const node, then copy the value of that const
    # in the parent and remove the 'BiasAdd', i.e. connect its children
    # to its parent.
    for fn_key in list(nnssa.functions.keys()):
        f = nnssa.functions[fn_key]
        keys = list(f.graph.keys())
        nodes_fused = []
        for k in keys:
            if k not in f.graph:
                continue
            current_node = f.graph[k]
            if current_node.op == 'BiasAdd' and len(current_node.inputs) == 2:
                parent_node = f.graph[current_node.inputs[0]]
                second_p_node = f.graph[current_node.inputs[1]]
                if (parent_node.op == 'MatMul' or parent_node.op == 'Conv2D' and len(parent_node.outputs) == 1) and \
                    (second_p_node.value is not None and len(second_p_node.outputs) == 1 and second_p_node.outputs[0] == k):

                    parent_node.attr['bias'] = second_p_node.value.val
                    disconnect_edge(f.graph, second_p_node.name, k)  # disconnect the const
                    disconnect_edge(f.graph, parent_node.name, k)  # disconnect the first parent
                    for out_node in current_node.outputs:
                        f.graph[parent_node.name].outputs.append(out_node)
                        if current_node.name in f.graph[out_node].inputs:
                            idx = f.graph[out_node].inputs.index(current_node.name)
                            f.graph[out_node].inputs[idx] = parent_node.name
                        else:
                            raise ValueError('[Op Fusion] fuse_bias_add() cannot identify biasAdd output.')
                    nodes_fused.append(k)
                    nodes_fused.append(second_p_node.name)

        for nf in nodes_fused:
            delete_node(f.graph, nf)
        if len(nodes_fused) > 0:
            print("[Op Fusion] fuse_bias_add() deleted {} nodes.".format(len(nodes_fused)))


def onehot_matmul_to_embedding(nnssa):
    # Look for 'MatMul' whose first input is 'OneHot'
    # and replace it with embedding op
    for fn_key in list(nnssa.functions.keys()):
        f = nnssa.functions[fn_key]
        keys = list(f.graph.keys())

        for k in keys:
            if k not in f.graph:
                continue
            current_node = f.graph[k]
            if len(current_node.inputs) < 1:
                continue
            inp_node = f.graph[current_node.inputs[0]]
            if (current_node.op == 'BatchMatMul' or current_node.op == 'MatMul') and inp_node.op == 'OneHot':
                assert len(inp_node.inputs) == 4, 'OneHot node should have 4 inputs'
                onehot_params = [f.graph[name].attr.get('value') for name in inp_node.inputs[1:]]
                depth_val, on_val, off_val = [x.val[0] for x in onehot_params]
                # Change the current node operation to Embedding
                current_node.op = 'Embedding'
                current_node.attr['depth'] = depth_val
                current_node.attr['on_value'] = on_val
                current_node.attr['off_value'] = off_val
                # Replace OneHot with its first input
                onehot_inp_node_names = inp_node.inputs[:]
                replace_node(f.graph, inp_node.name, onehot_inp_node_names[0])

                # Now delete the OneHot node and other input nodes
                delete_node(f.graph, onehot_inp_node_names[1])
                print('[Op Fusion] Node %s is removed.' %(onehot_inp_node_names[1]))
                delete_node(f.graph, onehot_inp_node_names[2])
                print('[Op Fusion] Node %s is removed.' %(onehot_inp_node_names[2]))
                delete_node(f.graph, onehot_inp_node_names[3])
                print('[Op Fusion] Node %s is removed.' %(onehot_inp_node_names[3]))
                delete_node(f.graph, inp_node.name)
                print('[Op Fusion] Node %s is removed.' %(inp_node.name))


def _search_nodes_by_type(gf, node_names, op_type):
    for name in node_names:
        if gf[name].op == op_type:
            return gf[name]


def _match_layernorm_pattern(gf, entry_node):
    """ Return the nodes that form the subgraph of a LayerNormalization layer
    """
    def _axes_in_range(axes, rank):
        return all([x in range(-rank, rank) for x in axes])

    try:
        params = {}
        mean_1 = _search_nodes_by_type(gf, entry_node.outputs, 'Mean')
        sqdiff_2 = _search_nodes_by_type(gf, entry_node.outputs, 'SquaredDifference')
        mul_3 = _search_nodes_by_type(gf, entry_node.outputs, 'Mul')

        if not (mean_1.op == 'Mean' and sqdiff_2.op == 'SquaredDifference' and
            mul_3.op == 'Mul'):
            return None
        const_4 = gf[mean_1.inputs[1]]
        mean_1_rank = len(mean_1.datatype.get_shape())
        if not (const_4.op == 'Const' and len(const_4.value.val) == 1 and
            _axes_in_range(const_4.value.val, mean_1_rank)):
            return None
        axes = const_4.value.val
        mean_5 = gf[sqdiff_2.outputs[0]]
        if not (mean_5.op == 'Mean'):
            return None
        const_6 = gf[mean_5.inputs[1]]
        mean_5_rank = len(mean_5.datatype.get_shape())
        if not (const_6.op == 'Const' and len(const_6.value.val) == 1 and
            axes == const_6.value.val):
            return None

        axes = sorted([x if x > 0 else mean_1_rank - x for x in
            const_4.value.val])
        ref_axes = list(range(mean_1_rank-len(axes), mean_1_rank))
        if not all([x == y for (x,y) in zip(axes, ref_axes)]):
            return None
        params['axes'] = axes

        add_7 = gf[mean_5.outputs[0]]
        const_8 = gf[add_7.inputs[1]] # epsilon
        params['epsilon'] = const_8.value.val
        rsqrt_9 = gf[add_7.outputs[0]]
        mul_10 = gf[rsqrt_9.outputs[0]]
        if not (add_7.op == 'Add' and const_8.op == 'Const' and
            rsqrt_9.op == 'Rsqrt' and mul_10.op == 'Mul'):
            return None
        const_11 = gf[mul_10.inputs[1]]
        params['gamma'] = const_11.value.val
        if not (mul_3.name in mul_10.outputs and len(mul_10.outputs) == 2):
            return None
        mul_12 = gf[mul_10.outputs[1]] if gf[mul_10.outputs[0]] == mul_3 else \
            gf[mul_10.outputs[0]]

        sub_13 = gf[mul_12.outputs[0]]
        if not (mul_12.op == 'Mul' and sub_13.op == 'Sub'):
            return None
        const_14 = gf[sub_13.inputs[0]]
        if not const_14.op == 'Const':
            return None
        params['beta'] = const_14.value.val
        add_15 = gf[sub_13.outputs[0]]
        if not (gf[add_15.inputs[0]] == mul_3 and add_15.op == 'Add'):
            return None

        layernorm_nodes = [mean_1, sqdiff_2, mul_3, const_4, mean_5, const_6,
            add_7, const_8, rsqrt_9, mul_10, const_11, mul_12, sub_13, const_14,
            add_15]

        return (layernorm_nodes, params)
    except Exception as e:
        return None


def _fuse_layer_norm(graph):
    keys = list(graph.keys())
    count = 0
    for k in keys:
        if k not in graph:
            continue
        current_node = graph[k]
        layernorm_nodes_params = _match_layernorm_pattern(graph, current_node)
        if layernorm_nodes_params is not None:
            ln_nodes, ln_params = layernorm_nodes_params
            out_node = ln_nodes[-1]
            ln_outputs = out_node.outputs[:]

            # Instantiate a new fused node in the graph
            fused_ln_node = ParsedNode()
            fused_ln_node.op = 'LayerNormalization'
            fused_ln_node.name = out_node.name + '_layernorm'
            fused_ln_node.attr = ln_params
            fused_ln_node.datatype = current_node.datatype

            graph[fused_ln_node.name] = fused_ln_node

            # Connect fused node to entry and output nodes
            connect_edge(graph, current_node.name, fused_ln_node.name)
            replace_node(graph, out_node.name, fused_ln_node.name)
            # connect_dests(graph, fused_ln_node.name, ln_outputs)

            # Delete nodes
            ln_node_names = [x.name for x in ln_nodes]
            for name in ln_node_names:
                delete_node(graph, name)

            count += 1

    if count > 0:
        print('[Op Fusion] Fused {} layer normalizations.'.format(count))


def fuse_layer_norm(nnssa):
    for fn_key in list(nnssa.functions.keys()):
        f = nnssa.functions[fn_key]
        _fuse_layer_norm(f.graph)


def _match_gelu_pattern(gf, entry_node):
    """ Return the nodes that form the subgraph of a GELU layer
    """
    try:
        if not len(entry_node.outputs) == 3:
            return None
        pow_1 = _search_nodes_by_type(gf, entry_node.outputs, 'Pow')
        add_2 = _search_nodes_by_type(gf, entry_node.outputs, 'Add')
        mul_3 = _search_nodes_by_type(gf, entry_node.outputs, 'Mul')

        if not (pow_1.op == 'Pow' and add_2.op == 'Add' and mul_3.op == 'Mul'):
            return None
        const_4 = gf[pow_1.inputs[1]]
        if not (const_4.op == 'Const' and int(round(const_4.value.val)) == 3):
            return None
        mul_5 = gf[pow_1.outputs[0]]
        const_6 = gf[mul_5.inputs[0]]
        if not (const_6.op == 'Const' and \
            abs(const_6.value.val - 0.0447) < 1e-3):
            return None
        if not (gf[add_2.inputs[0]] == entry_node and \
            gf[add_2.inputs[1]] == mul_5):
            return None
        mul_7 = gf[add_2.outputs[0]]
        const_8 = gf[mul_7.inputs[0]]
        if not abs(const_8.value.val - np.sqrt(2 / np.pi)) < 1e-3:
            return None
        tanh_9 = gf[mul_7.outputs[0]]
        add_10 = gf[tanh_9.outputs[0]]
        const_11 = gf[add_10.inputs[0]]
        if not (tanh_9.op == 'Tanh' and add_10.op == 'Add' and \
            const_11.op == 'Const' and int(round(const_11.value.val)) == 1):
            return None
        mul_12 = gf[add_10.outputs[0]]
        const_13 = gf[mul_12.inputs[0]]
        if not (mul_12.op == 'Mul' and const_13.op == 'Const' and \
            abs(const_13.value.val - 0.5) < 1e-3):
            return None
        if not (gf[mul_3.inputs[0]] == entry_node and \
            gf[mul_3.inputs[1]] == mul_12):
            return None

        gelu_nodes = [pow_1, add_2, mul_3, const_4, mul_5, const_6, mul_7,
            const_8, tanh_9, add_10, const_11, mul_12, const_13]

        return gelu_nodes

    except:
        return None

def _fuse_gelu(graph):
    keys = list(graph.keys())
    count = 0
    for k in keys:
        if k not in graph:
            continue
        current_node = graph[k]
        gelu_nodes = _match_gelu_pattern(graph, current_node)
        if gelu_nodes is not None:
            out_node = gelu_nodes[2]
            gelu_outputs = out_node.outputs[:]

            # Instantiate a new fused node in the graph
            fused_gelu_node = ParsedNode()
            fused_gelu_node.op = 'GeLU'
            fused_gelu_node.name = out_node.name + '_gelu'
            fused_gelu_node.attr = {}
            fused_gelu_node.datatype = current_node.datatype

            graph[fused_gelu_node.name] = fused_gelu_node

            # Delete nodes
            gelu_node_names = [x.name for x in gelu_nodes]
            for name in gelu_node_names:
                delete_node(graph, name)

            # Connect fused node to entry and output nodes
            connect_edge(graph, current_node.name, fused_gelu_node.name)
            connect_dests(graph, fused_gelu_node.name, gelu_outputs)

            count += 1

    if count > 0:
        print('[Op Fusion] Fused {} GeLUs.'.format(count))


def fuse_gelu(nnssa):
    for fn_key in list(nnssa.functions.keys()):
        f = nnssa.functions[fn_key]
        _fuse_gelu(f.graph)
