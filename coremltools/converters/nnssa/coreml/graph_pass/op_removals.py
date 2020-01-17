# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import copy
from ...commons.basic_graph_ops import delete_node, disconnect_edge, replace_node, replace_control_dest, connect_edge, connect_dests
from .op_fusions import _check_number_inputs, _check_number_outputs


def remove_no_ops_and_shift_control_dependencies(nnssa):
    for fn_key in list(nnssa.functions.keys()):
        f = nnssa.functions[fn_key]
        for name, node in f.graph.copy().items():
            if node.op == "NoOp":
                for each_control_output in node.control_outputs:
                    f.graph[each_control_output].control_inputs.remove(node.name)

                for each_control_input in node.control_inputs:
                    f.graph[each_control_input].control_outputs.remove(node.name)

                for each_control_output in node.control_outputs:
                    for each_control_input in node.control_inputs:
                        f.graph[each_control_output].control_inputs.append(each_control_input)
                        f.graph[each_control_input].control_outputs.append(each_control_output)

                del f.graph[name]


def constant_weight_link_removal(nnssa):
    # look for constant nodes and if they are feeding into
    # 'MatMul' or 'Conv2D', then copy the value to their attributes and delete the link.
    for fn_key in list(nnssa.functions.keys()):
        f = nnssa.functions[fn_key]
        keys = list(f.graph.keys())
        for k in keys:
            if k not in f.graph:
                continue
            is_const = (f.graph[k].value is not None) and (k not in f.outputs)
            if is_const:
                for o in f.graph[k].outputs:
                    nextnode = f.graph[o]
                    op_type = nextnode.op
                    if op_type == 'MatMul' or op_type == 'Conv2D':
                        if nextnode.inputs[1] == k:
                            nextnode.attr['W'] = f.graph[k].value.val
                            disconnect_edge(f.graph, k, o)


def remove_single_isolated_node(nnssa):
    # remove nodes that do not have any output and input
    delete_count = 0
    for fn_key in list(nnssa.functions.keys()):
        f = nnssa.functions[fn_key]
        keys = list(f.graph.keys())
        for k in keys:
            if k not in f.graph:
                continue
            if len(f.graph[k].outputs) == 0 and len(f.graph[k].inputs) == 0:
                delete_count += 1
                delete_node(f.graph, k)

    print('%d disconnected nodes deleted' % delete_count)


def _remove_internal_identity_nodes(nnssa):
    '''
    remove identity nodes that are not connected to the model outputs
    '''
    delete_count = 0
    for fn_key in list(nnssa.functions.keys()):
        f = nnssa.functions[fn_key]
        keys = list(f.graph.keys())
        for k in keys:
            if k not in f.graph:
                continue
            node = f.graph[k]
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                continue
            inp_node = f.graph[node.inputs[0]]
            if node.op == 'Identity' and inp_node.op != 'get_tuple':
                delete_count += 1
                parent_name = f.graph[k].inputs[0]
                disconnect_edge(f.graph, parent_name, k)
                for control_input in f.graph[k].control_inputs:
                    replace_control_dest(f.graph, control_input, k, parent_name)

                replace_node(f.graph, k, parent_name)  # join parent to children
                delete_node(f.graph, k)

    return delete_count


def _remove_output_identity_nodes(nnssa):
    '''
    remove identity nodes that ARE connected to the model outputs
    '''
    delete_count = 0
    for fn_key in list(nnssa.functions.keys()):
        f = nnssa.functions[fn_key]
        keys = list(f.graph.keys())
        for k in keys:
            if k not in f.graph:
                continue
            node = f.graph[k]

            if node.op != 'Identity' or len(node.inputs) != 1:
                continue

            if len(node.outputs) != 0 or (k not in f.outputs) or k != node.name:
                continue
            # this means node k is the "output-identity" node that nnssa inserts
            # we remove it here
            parent_name = node.inputs[0]
            parent_node = f.graph[parent_name]

            # Continue if parent node has an other outputs than identity node.
            if any([an_output != k for an_output in parent_node.outputs]):
                continue

            delete_count += 1

            # Remove Identity node and copy existing parent node
            parent_node = copy.deepcopy(f.graph[parent_name])
            for control_input_name in node.control_inputs:
                if control_input_name == parent_node.name:
                    continue
                if control_input_name in parent_node.control_inputs:
                    continue

                parent_node.control_inputs.append(control_input_name)

            del f.graph[k]
            del f.graph[parent_name]
            parent_node.name = k
            parent_node.outputs = []
            f.graph[k] = parent_node

            node = f.graph[k]
            for p in node.inputs:
                for idx, out in enumerate(f.graph[p].outputs):
                    if out == parent_name:
                        f.graph[p].outputs[idx] = k

            for p in node.control_inputs:
                for idx, out in enumerate(f.graph[p].control_outputs):
                    if out == parent_name:
                        f.graph[p].control_outputs[idx] = k

    return delete_count


def remove_identity(nnssa):
    '''
    remove node of type 'identity', connect its parent to its child.
    Disable this pass, if ssa contains more than 1 functions. In that case
    a few 'identity' nodes are crucial to get data in/out of body of loops
    '''
    if len(nnssa.functions.keys()) > 1:
        return
    delete_count = _remove_internal_identity_nodes(nnssa)
    delete_count += _remove_output_identity_nodes(nnssa)
    print('%d identity nodes deleted' % delete_count)


def remove_oneway_split(nnssa):
    """ Remove split op with 1 output that splits the input into itself.
    """
    for fn_key in list(nnssa.functions.keys()):
        f = nnssa.functions[fn_key]
        keys = list(f.graph.keys())
        for k in keys:
            if k not in f.graph:
                continue
            node = f.graph[k]
            if not (node.op == 'Split' and node.attr['num_split'] == 1 and
                    len(node.datatype.T) == 1 and len(node.inputs) == 2):
                continue

            if f.graph[node.inputs[0]].op == 'Const':
                axis_name, parent_name = node.inputs
            elif f.graph[node.inputs[1]].op == 'Const':
                parent_name, axis_name = node.inputs
            else:
                continue

            if len(node.outputs) == 1 and f.graph[node.outputs[0]].op == 'get_tuple':
                get_tuple_name = node.outputs[0]
            else:
                continue

            parent_node = f.graph[parent_name]
            get_tuple_node = f.graph[get_tuple_name]
            for out_name in get_tuple_node.outputs:
                out_node = f.graph[out_name]
                out_node.inputs = [parent_name if x == get_tuple_name else x \
                    for x in out_node.inputs]
                out_node.control_inputs = [parent_name if x == get_tuple_name \
                    else x for x in out_node.control_inputs]
            parent_node.outputs = get_tuple_node.outputs[:]
            parent_node.control_outputs = get_tuple_node.control_outputs[:]

            del f.graph[axis_name], f.graph[k], f.graph[get_tuple_name]


def remove_noneffective_transpose(nnssa):
    """
    A graph pass to eliminate extra noneffective consecutive transpose ops.
    """

    def _match(graph, entry_node, pattern_ops):
        if not _check_number_outputs(entry_node, 1):
            return None
        nodes_to_merge = list()
        node = entry_node
        for i, op in enumerate(pattern_ops):
            if node.op.lower() != pattern_ops[i]:
                return None
            if not _check_number_outputs(node, 1) and i == 0:
                return None
            if not _check_number_inputs(node, 2):
                return None
            node_inputs = [graph[n].op.lower() for n in node.inputs]
            try:
                const_node = graph[node.inputs[node_inputs.index('const')]]
            except ValueError:
                return None
            nodes_to_merge.extend([node, const_node])
            # do not fuse the output layer
            if len(node.outputs) == 0:
                return None
            node = graph[node.outputs[0]]
        return nodes_to_merge

    def _remove_noneffective_transpose(graph):
        keys = list(graph.keys())
        count = 0
        for k in keys:
            if k not in graph:
                continue
            current_node = graph[k]
            if current_node.op.lower() not in {'transpose'}:
                continue

            nodes = _match(graph, current_node, ['transpose'])
            if nodes:
                assert len(nodes) == 2
                # remove transpose op that does nothing
                perm = list(nodes[1].value.val)
                if perm == sorted(perm):
                    previous_node_name = current_node.inputs[0]
                    output_nodes = current_node.outputs[:]
                    delete_node(graph, nodes[1].name)
                    delete_node(graph, nodes[0].name)
                    connect_dests(graph, previous_node_name, output_nodes)
                    # make sure output's inputs is in correct order
                    out_inputs = graph[output_nodes[0]].inputs
                    a = out_inputs.index(graph[output_nodes[0]].inputs[0])
                    b = out_inputs.index(previous_node_name)
                    out_inputs[a], out_inputs[b] = out_inputs[b], out_inputs[a]
                    count += 1

        if count > 0:
            print('[Op Removal] Deleted {} transpose ops.'.format(count))

    for fn_key in list(nnssa.functions.keys()):
        f = nnssa.functions[fn_key]
        _remove_noneffective_transpose(f.graph)


def remove_noneffective_reshape(nnssa):
    """
    A graph pass to eliminate extra noneffective consecutive reshape ops.
    """

    def _match(graph, entry_node):
        # currently only merge two consecutive reshape ops
        pattern_ops = ['reshape', 'reshape']
        if not _check_number_outputs(entry_node, 1):
            return None
        nodes_to_merge = list()
        node = entry_node
        for i, op in enumerate(pattern_ops):
            if node.op.lower() != pattern_ops[i]:
                return None
            if not _check_number_outputs(node, 1) or not _check_number_inputs(node, 2):
                return None
            # do not fuse the output layer
            if len(node.outputs) == 0:
                return None
            node_inputs = [graph[n].op.lower() for n in node.inputs]
            try:
                const_node = graph[node.inputs[node_inputs.index('const')]]
            except ValueError:
                return None
            if const_node.op.lower() != 'const':
                return None
            nodes_to_merge.extend([node, const_node])
            node = graph[node.outputs[0]]
        return nodes_to_merge

    def _remove_noneffective_reshape(graph):
        keys = list(graph.keys())
        count = 0
        for k in keys:
            if k not in graph:
                continue
            current_node = graph[k]
            if current_node.op.lower() not in {'reshape'}:
                continue

            nodes = _match(graph, current_node)
            if nodes:
                assert len(nodes) == 4
                # squash consecutive reshape into last one
                previous_node = current_node.inputs[0]
                output_nodes = current_node.outputs[:]
                delete_node(graph, nodes[1].name)
                delete_node(graph, nodes[0].name)
                connect_dests(graph, previous_node, output_nodes)
                # make sure output's inputs is in correct order
                out_inputs = graph[output_nodes[0]].inputs
                a = out_inputs.index(nodes[3].name)
                b = out_inputs.index(previous_node)
                out_inputs[a], out_inputs[b] = out_inputs[b], out_inputs[a]
                count += 1

        if count > 0:
            print('[Op Removal] Deleted {} reshape ops.'.format(count))

    for fn_key in list(nnssa.functions.keys()):
        f = nnssa.functions[fn_key]
        _remove_noneffective_reshape(f.graph)
