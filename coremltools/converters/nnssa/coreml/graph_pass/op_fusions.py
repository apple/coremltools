# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import numpy as np
from ...commons import builtins
from ...commons.symbolic import *
from ...commons.basic_graph_ops import disconnect_edge, connect_edge, \
    delete_node, replace_node, connect_dests, topsort
from ...nnssa import ParsedNode

ELEMENTWISE_OPS = {
    'Maximum',
    'Minimum',
    'Add',
    'AddV2',
    'Sub',
    'BiasAdd',
    'Mul',
    'RealDiv',
    'Sigmoid',
    'Relu',
    'Relu6',
    'LeakyRelu',
    'Tanh',
    'Identity',
    'Sqrt',
    'Rsqrt',
    'Pow',
    'LRN',
    'Square',
    'SquaredDifference'
}

# Native SSA nodes with data_format attributes of NHWC / NCHW
NATIVE_NHWC_OPS = {
    'Conv2D', 'Conv2DBackpropInput', 'DepthwiseConv2dNative',
    'Pooling', 'MaxPool', 'AvgPool', 'DepthToSpace', 'SpaceToDepth',
}

REDUCTION_OPS = {
    'Mean',
    'Max'
}


def _check_number_inputs(node, n):
    return len(node.inputs) == n


def _check_number_outputs(node, n):
    return len(node.outputs) == n


def _check_single_out_vector_constant_node(node):
    return node.op == 'Const' and len(node.outputs) == 1 and \
           node.value is not None and len(np.squeeze(node.value.val).shape) == 1


def _check_rank_matches(node1, node2):
    rank1 = len(node1.datatype.get_shape())
    rank2 = len(node2.datatype.get_shape())
    return rank1 == rank2


def _update_padding_and_crop_values_2d(pad_values, crop_values, params):
    def _new_pad_crop_1d(p1, p2, c1, c2, k, s, n1):
        n2 = np.floor((n1 + p1 + p2 - k) / s) + 1
        if 1 + c1 * s <= p1:
            p1 -= c1 * s
            c1 = 0
        if k + (n2 - c2 - 1) * s > p1 + n1:
            p2 = k + (n2 - c2 - 1) - (p1 + n1)
            c2 = 0
        return p1, p2, c1, c2

    p1, p2, c1, c2 = _new_pad_crop_1d(pad_values[2], pad_values[3],
                                      crop_values[2], crop_values[3],
                                      params['kh'], params['sh'], params['Hin'])
    pad_values[2:] = np.array([p1, p2], dtype=np.int)
    crop_values[2:] = np.array([c1, c2], dtype=np.int)

    p1, p2, c1, c2 = _new_pad_crop_1d(pad_values[0], pad_values[1],
                                      crop_values[0], crop_values[1],
                                      params['kw'], params['sw'], params['Win'])
    pad_values[:2] = np.array([p1, p2], dtype=np.int)
    crop_values[:2] = np.array([c1, c2], dtype=np.int)


def _is_NHWC(graph, node):
    if node.op == 'ResizeBilinear' or node.op == 'ResizeNearestNeighbor' \
            or node.op == 'MirrorPad':
        return True
    if node.op in NATIVE_NHWC_OPS and node.attr.get('data_format') == 'NHWC':
        return True

    if node.op == 'Concat':  # Concat's first input is axis
        return all(graph[inp].attr.get('data_format') == 'NHWC_format_inserted'
                   for inp in node.inputs[1:])
    if node.op == 'ConcatV2':  # ConcatV2's last input is axis
        return all(graph[inp].attr.get('data_format') == 'NHWC_format_inserted'
                   for inp in node.inputs[:-1])

    if node.op == 'Pad' and len(node.datatype.get_shape()) == 4:
        # adjust constant padding values
        parent_node = graph[node.inputs[1]]
        if parent_node.value is not None:
            val = np.array(parent_node.value.val)
            if len(val) == 4 and builtins.is_tensor(parent_node.datatype) and len(parent_node.outputs) == 1:
                parent_node.value.val = parent_node.value.val[[0, 3, 1, 2]]
        return True

    if node.op in REDUCTION_OPS:
        if not any([graph[inp].attr.get('data_format', '') ==
                    'NHWC_format_inserted' for inp in node.inputs]):
            return False
        if not node.attr.get('keep_dims', True):
            return False
        # adjust axis / dims / reduction_indices values
        for inp in node.inputs:
            parent_node = graph[inp]
            if parent_node.value is not None:
                val = np.array(parent_node.value.val)
                if isinstance(parent_node.value.val, np.int32):
                    val = np.array([parent_node.value.val])
                m_nhwc_to_nchw = {0: 0, 1: 2, 2: 3, 3: 1}
                reduction_indices = np.array([m_nhwc_to_nchw[x if x >= 0 else 4 + x] for x in val], dtype=np.int32)
                parent_node.value.val = np.reshape(reduction_indices, parent_node.value.val.shape)
                node.attr['reduction_indices'] = reduction_indices
        return True

    if node.op in ELEMENTWISE_OPS:
        # if its an element-wise op and if all of its parent(s) are
        # "NHWC_format_inserted" or given that at least one of the parents
        # is "NHWC_format_inserted" and rest are vector constants, then the
        # node is also declared to be "NHWC_format_inserted"

        NHWC_parent = any([graph[inp].attr.get('data_format',
                                               None) == 'NHWC_format_inserted' for inp in node.inputs])

        if NHWC_parent:
            for inp in node.inputs:
                parent_node = graph[inp]
                if parent_node.attr.get('data_format', None) == 'NHWC_format_inserted':
                    continue
                elif parent_node.value is not None:
                    val = np.array(parent_node.value.val)
                    # constant scalar
                    if val.shape == () and not builtins.is_tensor(parent_node.datatype) and len(parent_node.outputs) == 1:
                        continue
                    # constant vector
                    if len(val.shape) == 1 and builtins.is_tensor(parent_node.datatype) and len(parent_node.outputs) == 1:
                        continue
                    else:
                        return False
                else:
                    return False
            return True

    return False


def _insert_transpose_to_or_from_nchw(graph, src, dst, transpose_node_name, transpose_params=None):
    """
    Insert a node called 'transpose_node_name' between src and dst
    This node should be a transpose node with params 'transpose_params'
    """

    if not transpose_params:
        transpose_params = [0, 3, 1, 2]  # channel_last to channel_first

    # First check whether the node already exists in the graph or not.

    if transpose_node_name in graph:
        tp_node = graph[transpose_node_name]
        if dst.name not in tp_node.outputs:
            tp_node.outputs.append(dst.name)
    else:
        # the node does not exist, so create a fresh one
        tp_node = ParsedNode()
        tp_node.op = 'Transpose'
        tp_node.name = transpose_node_name

        # Adjust type inference
        if builtins.is_tensor(src.datatype):
            s = src.datatype.get_shape()
            if len(s) == 4:
                tp_shape = tuple([s[transpose_params[0]], s[transpose_params[1]], s[transpose_params[2]], s[transpose_params[3]]])
                tp_node.datatype = builtins.tensor(src.datatype.get_primitive(), tp_shape)
            else:
                tp_node.datatype = src.datatype

        tp_node.inputs = [src.name]
        tp_node.outputs = [dst.name]
        tp_node.attr['dim'] = transpose_params
        graph[transpose_node_name] = tp_node

    # Rename dst's input 'src' to 'transpose_node_name'
    for idx, inp in enumerate(dst.inputs):
        if inp == src.name:
            dst.inputs[idx] = transpose_node_name
            break

    # Rename src's output from 'dst' to 'transpose_node_name'
    if transpose_node_name in src.outputs:
        # 'transpose_node_name' already exists as an output of the src,
        # we just need to delete dst node from the output list of src, instead of replacing it
        if dst.name in src.outputs:
            src.outputs.remove(dst.name)
    else:
        for idx, outp in enumerate(src.outputs):
            if outp == dst.name:
                src.outputs[idx] = transpose_node_name
                break


def _insert_transpose_to_nchw(graph, src, dst):
    tp_node_name = src.name + "_to_nchw"
    _insert_transpose_to_or_from_nchw(graph, src, dst, tp_node_name, [0, 3, 1, 2])


def _insert_transpose_from_nchw(graph, src, dst):
    tp_node_name = src.name + "_to_nhwc"
    _insert_transpose_to_or_from_nchw(graph, src, dst, tp_node_name, [0, 2, 3, 1])


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
        # this pass needs the ssa to be in the topologically sorted order
        node_names = topsort(graph)

        # Mark all NHWC nodes
        nhwc_nodes = []
        for name in node_names:
            node = graph[name]
            if len(node.outputs) > 0 and len(node.inputs) > 0 and _is_NHWC(graph, node):
                node.attr['data_format'] = 'NHWC_format_inserted'
                nhwc_nodes.append(name)

        for name in nhwc_nodes:
            node = graph[name]

            # Adjust type inference
            if builtins.is_tensor(node.datatype):
                s = node.datatype.get_shape()
                if len(s) == 4:
                    new_shape = tuple([s[0], s[3], s[1], s[2]])
                    node.datatype = builtins.tensor(node.datatype.get_primitive(), new_shape)
                    node.attr['symbolic_datatype'] = node.datatype

            if '_output_shapes' in node.attr:
                orig_out_shapes = node.attr['_output_shapes']
                if len(orig_out_shapes) == 1 and len(orig_out_shapes[0]) == 4:
                    s = orig_out_shapes[0]
                    node.attr['_output_shapes'] = [[s[0], s[3], s[1], s[2]]]

            if node.op in ELEMENTWISE_OPS:
                for inp in node.inputs:
                    parent_node = graph[inp]
                    if parent_node.value is None:
                        continue

                    # if there is a constant vector input
                    val = np.array(parent_node.value.val)
                    if len(val.shape) == 1 and builtins.is_tensor(parent_node.datatype):
                        new_shape = (1, val.shape[0], 1, 1)
                        parent_node.datatype = builtins.tensor(
                            parent_node.datatype.get_primitive(), new_shape
                        )
                        parent_node.value.val = np.reshape(
                            parent_node.value.val, new_shape
                        )

            # Insert NHWC -> NCHW transpose
            for i, inp_node_name in enumerate(node.inputs):
                inp_node_format = graph[inp_node_name].attr.get('data_format')
                symbolic_value = graph[inp_node_name].attr['symbolic_value']
                if (graph[inp_node_name].op == 'Const' or
                        len(graph[inp_node_name].datatype.get_shape()) != 4 or
                        (symbolic_value and not any_symbolic_or_unknown(symbolic_value))):
                    # Const weights and parameters
                    continue

                if inp_node_format != 'NHWC_format_inserted':
                    assert len(graph[inp_node_name].datatype.get_shape()) == 4
                    _insert_transpose_to_nchw(graph, graph[inp_node_name], node)

            # Insert NCHW -> NHWC transpose
            for i, out_node_name in enumerate(node.outputs):
                out_node_format = graph[out_node_name].attr.get('data_format')
                if out_node_format != 'NHWC_format_inserted':
                    _insert_transpose_from_nchw(graph, node, graph[out_node_name])

            # Adjust output shape and concat layer's axis parameter
            if node.op == 'Concat' and len(node.inputs) > 1 and graph[node.inputs[0]].value is not None:
                axis = graph[node.inputs[0]].value.val
                axis = 4 + axis if axis < 0 else axis
                if axis == 3:
                    node.attr['axis'] = 1
                elif axis == 2 or axis == 1:
                    node.attr['axis'] = axis + 1
                else:
                    node.attr['axis'] = axis

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
                ops_to_merge = ['MatMul', 'Conv2D', 'DepthwiseConv2dNative']
                if (
                    (parent_node.op in ops_to_merge
                     and len(parent_node.outputs) == 1)
                    and
                    (second_p_node.value is not None
                     and len(second_p_node.outputs) == 1
                     and second_p_node.outputs[0] == k)
                ):
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
                print('[Op Fusion] Node %s is removed.' % (onehot_inp_node_names[1]))
                delete_node(f.graph, onehot_inp_node_names[2])
                print('[Op Fusion] Node %s is removed.' % (onehot_inp_node_names[2]))
                delete_node(f.graph, onehot_inp_node_names[3])
                print('[Op Fusion] Node %s is removed.' % (onehot_inp_node_names[3]))
                delete_node(f.graph, inp_node.name)
                print('[Op Fusion] Node %s is removed.' % inp_node.name)


def _search_nodes_by_type(gf, node_names, op_types):
    for name in node_names:
        if gf[name].op in op_types:
            return gf[name]


def _match_layernorm_pattern(gf, entry_node):
    """ Return the nodes that form the subgraph of a LayerNormalization layer
    """

    def _axes_in_range(axes, rank):
        return all([x in range(-rank, rank) for x in axes])

    try:
        params = {}
        mean_1 = _search_nodes_by_type(gf, entry_node.outputs, ['Mean'])
        sqdiff_2 = _search_nodes_by_type(gf, entry_node.outputs, ['SquaredDifference'])
        mul_3 = _search_nodes_by_type(gf, entry_node.outputs, ['Mul'])

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
        ref_axes = list(range(mean_1_rank - len(axes), mean_1_rank))
        if not all([x == y for (x, y) in zip(axes, ref_axes)]):
            return None
        params['axes'] = axes

        add_7 = gf[mean_5.outputs[0]]
        const_8 = gf[add_7.inputs[1]]  # epsilon
        params['epsilon'] = const_8.value.val
        rsqrt_9 = gf[add_7.outputs[0]]
        mul_10 = gf[rsqrt_9.outputs[0]]
        if not (add_7.op in ['Add','AddV2'] and const_8.op == 'Const' and
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
        if not (gf[add_15.inputs[0]] == mul_3 and add_15.op in ['Add','AddV2']):
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

            # Delete nodes
            ln_node_names = [x.name for x in ln_nodes]
            for name in ln_node_names:
                delete_node(graph, name)

            count += 1

    if count > 0:
        print('[Op Fusion] Fused {} layer normalizations.'.format(count))


def fuse_layer_norm(nnssa):
    """
    Layernorm op replaces the following sub-graph:

    [...] -----> Mean ---> SquaredDifference ---> Mean ---> Add/AddV2 (epsilon) ---> Rsqrt ---> Mul (gamma) ---->  Mul ----> Sub (beta) ---->  Add/AddV2 -------> [...]
      |            |             ^                                                                  |               ^                              ^
      |            |             |                                                                  |               |                              |
      | --------------------------                                                                  |               |                              |
      |            |------------------------------------------------------------------------------------------------                               |
      |                                                                                             |------------------------------> Mul------------
      |                                                                                                                               ^
      |                                                                                                                               |
      | -------------------------------------------------------------------------------------------------------------------------------

    """
    for fn_key in list(nnssa.functions.keys()):
        f = nnssa.functions[fn_key]
        _fuse_layer_norm(f.graph)


def _match_gelu_pattern(gf, entry_node):
    """ Return the nodes that form the subgraph of a GELU layer
    """
    try:
        if not len(entry_node.outputs) == 3:
            return None
        pow_1 = _search_nodes_by_type(gf, entry_node.outputs, ['Pow'])
        add_2 = _search_nodes_by_type(gf, entry_node.outputs, ['Add', 'AddV2'])
        mul_3 = _search_nodes_by_type(gf, entry_node.outputs, ['Mul'])

        if not (pow_1.op == 'Pow' and add_2.op in ['Add','AddV2'] and mul_3.op == 'Mul'):
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
        if not (tanh_9.op == 'Tanh' and add_10.op in ['Add','AddV2'] and \
                const_11.op == 'Const' and int(round(const_11.value.val)) == 1):
            return None
        mul_12 = gf[add_10.outputs[0]]
        const_13 = gf[mul_12.inputs[0]]
        if not (mul_12.op == 'Mul' and const_13.op == 'Const' and \
                abs(const_13.value.val - 0.5) < 1e-3):
            return None
        if not ([gf[mul_3.inputs[0]], gf[mul_3.inputs[1]]] == [entry_node, mul_12] \
                or [gf[mul_3.inputs[1]], gf[mul_3.inputs[0]]] == [entry_node, mul_12]):
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

            # Instantiate a new fused node in the graph
            fused_gelu_node = ParsedNode()
            fused_gelu_node.op = 'GeLU'
            fused_gelu_node.name = out_node.name + '_gelu'
            fused_gelu_node.attr = {}
            fused_gelu_node.datatype = current_node.datatype

            graph[fused_gelu_node.name] = fused_gelu_node

            # Connect fused node to entry and output nodes
            connect_edge(graph, current_node.name, fused_gelu_node.name)
            replace_node(graph, out_node.name, fused_gelu_node.name)

            # Delete nodes
            gelu_node_names = [x.name for x in gelu_nodes]
            for name in gelu_node_names:
                delete_node(graph, name)

            count += 1

    if count > 0:
        print('[Op Fusion] Fused {} GeLUs.'.format(count))


def fuse_gelu(nnssa):
    """
    This is the Gelu pattern:
    [...] -----> Pow (3) ----> Mul (.0447) -----> Add/AddV2 -----> Mul (sqrt(2/pi)) ---> tanh ----> Add/AddV2 (1) ----> Mul (0.5) -----> Mul ------> [...]
      |                                            ^                                                                                      ^
      |                                            |                                                                                      |
      |------------------------------------------------------------------------------------------------------------------------------------

    y = ( tanh((.0447)x^3 + x ) * (sqrt(2/pi)) + 1 ) * 0.5 * x

    Replace this subgraph with a single "GeLU" op
    """
    for fn_key in list(nnssa.functions.keys()):
        f = nnssa.functions[fn_key]
        _fuse_gelu(f.graph)


def fuse_batch_norm(ssa):
    """
    A graph pass that match and fuses following op patterns into a single BatchNorm op.

    Pattern 1:
             [Const]   [Const]
                |         |
                V         V
    [...] --> [Mul] --> [Add] --> [...] to [...] --> [BatchNorm] --> [...]

    Pattern 2:
             [Const]   [Const]   [Const]
                |         |         |
                V         V         V
    [...] --> [Sub] --> [Mul] --> [Add] --> [...] to [...] --> [BatchNorm] --> [...]

    Pattern 3:
             [Const]   [Const]       [Const]     [Const]
                |         |            |            |
                V         V            V            V
    [...] --> [Sub] --> [RealDiv] --> [Mul] --> [BiasAdd] --> [...] to [...] --> [BatchNorm] --> [...]
    """

    def _match_batch_norm_pattern(graph, entry_node, pattern_ops):
        if not _check_number_outputs(entry_node, 1):
            return None
        nodes_to_merge = list()
        node = graph[entry_node.outputs[0]]
        for i, op in enumerate(pattern_ops):
            if node.op != op:
                return None
            if node.op != pattern_ops[i] and not _check_number_outputs(node, 1):
                return None
            if not _check_number_inputs(node, 2):
                return None
            node_inputs = [graph[n].op.lower() for n in node.inputs]
            try:
                const_node = graph[node.inputs[node_inputs.index('const')]]
            except ValueError:
                return None
            if not _check_single_out_vector_constant_node(const_node):
                return None
            if not _check_rank_matches(const_node, node):
                return None
            nodes_to_merge.extend([const_node, node])
            if len(node.outputs) == 0:  # do not fuse the output layer
                return None
            node = graph[node.outputs[0]]
        if len(nodes_to_merge) != len(pattern_ops) * 2:
            return None
        return nodes_to_merge

    def _merge_batch_norm(graph, nodes, pattern_id=1):
        expected_num_nodes = 4
        if pattern_id == 2:
            expected_num_nodes = 6
        elif pattern_id == 3:
            expected_num_nodes = 8
        assert len(nodes) == expected_num_nodes

        current_node = graph[nodes[1].inputs[0]]
        out_node = nodes[-1]
        bn_outputs = out_node.outputs[:]

        fused_bn_node = ParsedNode()
        fused_bn_node.op = 'BatchNorm'
        fused_bn_node.name = out_node.name + '_batch_norm'

        fused_bn_node.attr = {
            'gamma': np.squeeze(nodes[0].value.val),
            'beta': np.squeeze(nodes[2].value.val),
        }
        if pattern_id == 2:
            fused_bn_node.attr = {
                'mean': np.squeeze(nodes[0].value.val),
                'gamma': np.squeeze(nodes[2].value.val),
                'beta': np.squeeze(nodes[4].value.val),
            }
        elif pattern_id == 3:
            fused_bn_node.attr = {
                'mean': np.squeeze(nodes[0].value.val),
                'gamma': np.squeeze(nodes[4].value.val) / np.squeeze(nodes[2].value.val),
                'beta': np.squeeze(nodes[6].value.val),
            }

        fused_bn_node.datatype = current_node.datatype
        graph[fused_bn_node.name] = fused_bn_node

        # combine control i/o
        control_inputs = []
        control_outputs = []
        bn_node_names = [x.name for x in nodes]

        for name in bn_node_names:
            control_inputs += graph[name].control_inputs
            control_outputs += graph[name].control_outputs

            # Modify control outputs with name of fused batch norm node.
            for control_output_name in graph[name].control_outputs:
                ctrl_node = graph[control_output_name]
                for i, inpt_name in enumerate(ctrl_node.control_inputs):
                    if inpt_name == name:
                        ctrl_node.control_inputs[i] = fused_bn_node.name

        fused_bn_node.control_inputs = control_inputs
        fused_bn_node.control_outputs = control_outputs

        # connect fused node to entry and output nodes
        connect_edge(graph, current_node.name, fused_bn_node.name)
        connect_dests(graph, fused_bn_node.name, bn_outputs)

        # correct output's inputs order
        for out in bn_outputs:
            if len(graph[out].inputs) < 2:
                continue
            out_inputs = graph[out].inputs
            a = out_inputs.index(out_node.name)
            b = out_inputs.index(fused_bn_node.name)
            out_inputs[a], out_inputs[b] = out_inputs[b], out_inputs[a]

        # delete merged nodes
        for name in bn_node_names:
            delete_node(graph, name)

    def _fuse_batch_norm(graph):
        keys = list(graph.keys())
        count = 0
        for k in keys:
            if k not in graph:
                continue
            current_node = graph[k]

            # return nodes order: [Const, Sub, Const, RealDiv, Const, Mul, Const, BiasAdd]
            nodes3 = _match_batch_norm_pattern(graph, current_node, ['Sub', 'RealDiv', 'Mul', 'BiasAdd'])
            # return nodes order: : [Const, Sub, Const, Mul, Const, Add]
            nodes2 = _match_batch_norm_pattern(graph, current_node, ['Sub', 'Mul', 'Add'])
            # return nodes order: : [Const, Mul, Const, Add]
            nodes1 = _match_batch_norm_pattern(graph, current_node, ['Mul', 'Add'])

            if nodes3:
                _merge_batch_norm(graph, nodes=nodes3, pattern_id=3)
                count += len(nodes3)

            if nodes2:
                _merge_batch_norm(graph, nodes=nodes2, pattern_id=2)
                count += len(nodes2)

            if nodes1:
                _merge_batch_norm(graph, nodes=nodes1, pattern_id=1)
                count += len(nodes1)

        if count > 0:
            print('[Op Fusion] Fused {} nodes into BatchNorms.'.format(count))

    for fn_key in list(ssa.functions.keys()):
        f = ssa.functions[fn_key]
        _fuse_batch_norm(f.graph)


def fuse_pad_into_conv(nnssa):
    """
    A graph pass that match and fuses following op patterns into one Conv2D op.

    Pattern 1:
    [Const]
      |
      V
    [Pad] --> [Conv2D] --> [...] to [Conv2D] --> [...]
    """

    def _match_pad_conv2d_pattern(graph, entry_node):
        if not _check_number_outputs(entry_node, 1):
            return None
        conv2d_node = graph[entry_node.outputs[0]]
        if not (conv2d_node.op == 'Conv2D' and _check_number_outputs(conv2d_node, 1) and _check_number_inputs(conv2d_node, 1)):
            return None
        if conv2d_node.attr.get('padding', '').lower() != 'valid':
            return None
        return [entry_node, conv2d_node]

    def _fuse_pad_into_conv(graph):
        keys = list(graph.keys())
        count = 0
        for k in keys:
            if k not in graph:
                continue
            current_node = graph[k]
            if current_node.op != 'Pad':
                continue

            nodes = _match_pad_conv2d_pattern(graph, current_node)  # [Pad, Conv2D]

            if nodes:
                pad_node, conv2d_node = nodes
                previous_node = pad_node.inputs[0]
                paddings = graph[pad_node.inputs[1]].value.val
                pad_h, pad_w = paddings[-2], paddings[-1]

                # fused node in the graph
                conv2d_node.attr.update({
                    'pad_h': pad_h, 'pad_w': pad_w
                })
                graph[conv2d_node.name] = conv2d_node

                # delete pad const node and pad node
                delete_node(graph, pad_node.inputs[1])
                delete_node(graph, pad_node.name)
                connect_edge(graph, previous_node, conv2d_node.name)

                count += 1

        if count > 0:
            print('[Op Fusion] Fused {} Pad nodes into Conv2D.'.format(count))

    for fn_key in list(nnssa.functions.keys()):
        f = nnssa.functions[fn_key]
        _fuse_pad_into_conv(f.graph)


def spatial_reduce_to_global_pool(nnssa):
    """
    A graph pass to translate a spatial reduce op to global pool op for better GPU performance.
    """
    reduce_ops = {'mean', 'max'}

    def _spatial_reduce_to_global_pool(graph):
        keys = list(graph.keys())
        count = 0
        for k in keys:
            if k not in graph:
                continue
            current_node = graph[k]
            if current_node.op.lower() not in reduce_ops:
                continue
            reduction_indices = current_node.attr.get('reduction_indices')
            # reduction on height and weight dimensions
            hw_dims = {(2, 3), (3, 2), (-2, -1), (-1, -2), (2, -1), (-1, 2), (-2, 3), (3, -2)}
            if tuple(reduction_indices) in hw_dims:
                # replace reduce op to global pooling op
                previous_node = current_node.inputs[0]
                output_nodes = current_node.outputs[:]

                pooling_node = ParsedNode()
                pooling_node.op = 'AvgPool' if current_node.op.lower() == 'mean' else 'MaxPool'
                pooling_node.name = current_node.name + '_pooling'
                pooling_node.attr = {
                    'padding': 'valid'.upper(),
                    'global_pooling': True,
                }
                pooling_node.datatype = current_node.datatype
                graph[pooling_node.name] = pooling_node

                delete_node(graph, current_node.name)
                connect_edge(graph, previous_node, pooling_node.name)
                connect_dests(graph, pooling_node.name, output_nodes)

                count += 1

        if count > 0:
            print('[Op Fusion] Tuned {} Reductions.'.format(count))

    for fn_key in list(nnssa.functions.keys()):
        f = nnssa.functions[fn_key]
        _spatial_reduce_to_global_pool(f.graph)


def fuse_batch_to_space_or_space_to_batch(ssa):
    """
    A graph pass to fuse patterns related to space/batch transformations.
    """

    def _match_batch_to_space_nd(graph, entry_node):
        nodes = list()
        prev_node = entry_node
        while len(nodes) < 2:
            if len(prev_node.inputs) > 0 and graph[prev_node.inputs[0]].op:
                prev_node = graph[prev_node.inputs[0]]
                if prev_node.op == 'Transpose':
                    continue
                nodes.append(prev_node.op)
            else:
                break
        if len(nodes) > 1 \
                and (nodes[0] == 'Conv2d' or nodes[0] == 'DepthwiseConv2dNative') \
                and nodes[1] == 'SpaceToBatchND':
            return entry_node
        return None

    def _match_space_to_batch_nd(graph, entry_node):
        nodes = list()
        next_node = entry_node
        while len(nodes) < 2:
            if len(next_node.inputs) > 0 and graph[next_node.outputs[0]].op:
                next_node = graph[next_node.outputs[0]]
                if next_node.op == 'Transpose':
                    continue
                nodes.append(next_node.op)
            else:
                break
        if len(nodes) > 1 \
                and (nodes[0] == 'Conv2d' or nodes[0] == 'DepthwiseConv2dNative') \
                and nodes[1] == 'BatchToSpaceND':
            return entry_node
        return None

    def _fuse_batch_to_space_or_space_to_batch(graph):
        keys = list(graph.keys())
        count = 0
        nodes = list()
        for k in keys:
            if k not in graph:
                continue
            current_node = graph[k]

            if current_node.op == 'BatchToSpaceND' and len(current_node.outputs) == 1:
                node = _match_batch_to_space_nd(graph, current_node)
                nodes += [node] if node is not None else []

            if current_node.op == 'SpaceToBatchND' and len(current_node.outputs) == 1:
                node = _match_space_to_batch_nd(graph, current_node)
                nodes += [node] if node is not None else []

        for n in nodes:
            previous_node = n.inputs[0]
            output_node = n.outputs[0]
            connect_edge(graph, previous_node, output_node)
            # make sure output's inputs is in correct order
            out_inputs = graph[output_node].inputs
            a = out_inputs.index(n.name)
            b = out_inputs.index(previous_node)
            out_inputs[a], out_inputs[b] = out_inputs[b], out_inputs[a]

            if n.op == 'SpaceToBatchND':
                padding_values = [0] * 4
                dilations = list(graph[n.inputs[1]].value.val)
                paddings = graph[n.inputs[2]].value.val
                padding_values[2] = paddings[0, 0]  # top
                padding_values[3] = paddings[0, 1]  # bottom
                padding_values[0] = paddings[1, 0]  # left
                padding_values[1] = paddings[1, 1]  # right
                graph[output_node].attr.update({'dilations': dilations})
                needs_padding_before = True if sum(padding_values) != 0 else False
                if needs_padding_before:
                    graph[output_node].attr.update({'_paddings_before': padding_values})

            elif n.op == 'BatchToSpaceND':
                cropping_values = [0] * 4
                croppings = graph[n.inputs[2]].value.val
                cropping_values[2] = croppings[0, 0]  # top
                cropping_values[3] = croppings[0, 1]  # bottom
                cropping_values[0] = croppings[1, 0]  # left
                cropping_values[1] = croppings[1, 1]  # right
                needs_cropping_after = False
                border_mode = n.attr.get('padding', '').lower()
                if sum(cropping_values) != 0:
                    if border_mode != 'valid':
                        needs_cropping_after = True
                    else:
                        raise NotImplementedError('unhandled BatchToSpaceND case.')
                if needs_cropping_after:
                    graph[output_node].attr.update({'_cropping_after': cropping_values})

            # adjust type inference
            shape = list(graph[previous_node].datatype.get_shape())
            graph[output_node].datatype = builtins.tensor(graph[output_node].datatype.get_primitive(), tuple(shape))

            delete_node(graph, n.name)
            count += 1

        if count > 0:
            print('[Op Fusion] Skipped {} BatchToSpaceND / SpaceToBatchND nodes.'.format(count))

    for fn_key in list(ssa.functions.keys()):
        f = ssa.functions[fn_key]
        _fuse_batch_to_space_or_space_to_batch(f.graph)
