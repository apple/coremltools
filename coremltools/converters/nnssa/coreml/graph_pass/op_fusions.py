# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import numpy as np
from ...commons.basic_graph_ops import disconnect_edge, connect_edge, delete_node, replace_node
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
    if (node.op == 'Conv2D' or node.op == 'Pooling' or node.op =='MaxPool' or \
        node.op == 'AvgPool') and node.attr.get('data_format') == 'NHWC':
        return True
    if node.op == 'ConcatV2':
        # ConcatV2's last input is axis
        return all(graph[inp].attr.get('data_format') == 'NHWC' for inp in node.inputs[:-1])
    if node in ELEMENTWISE_OPS:
        return all(graph[inp].attr.get('data_format') == 'NHWC' for inp in node.inputs)
    return False


def _insert_transpose_to_nchw(graph, src, dst):

    tp_node_name = src.name + "_to_nchw"
    tp_node = ParsedNode()
    tp_node.op = 'Transpose'
    tp_node.name = tp_node_name
    tp_node.datatype = src.datatype
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
    tp_node.datatype = src.datatype
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
    (1) it is a conv or pooling layer with "NHWC" data format
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

            # Insert NHWC->NCHW tranpose
            for i, inp_node_name in enumerate(node.inputs):
                inp_node_format = graph[inp_node_name].attr.get('data_format')
                if node.op == 'Conv2D' and i == 1 and graph[inp_node_name].op == 'Const':
                    # Skip constant weights
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

"""
def connect_edge(g, source, dest):
    g[source].outputs.append(dest)
    g[dest].inputs.append(source)
def replace_node(g, original_node, new_node):
    for o in list(g[original_node].outputs):
        replace_source(g, original_node, o, new_node)

"""
def onehot_matmul_to_embedding(nnssa):
    # Look for 'MatMul' whose first input is 'OneHot', and replace it with embedding op
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

