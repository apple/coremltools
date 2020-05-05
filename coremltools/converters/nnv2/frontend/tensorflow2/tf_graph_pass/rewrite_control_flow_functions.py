# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import logging
from coremltools.converters.nnv2.frontend.tensorflow.parsed_tf_node import ParsedTFNode
from coremltools.converters.nnv2.frontend.tensorflow.basic_graph_ops import (
    disconnect_edge,
    connect_edge,
    delete_node,
    replace_node,
    replace_dest,
    connect_edge_at_index,
)


def _rename_node_in_fn(node, new_name, fn):
    """
    Rename a node and all it's connections.

    Parameters
    ----------
    node: ParsedTFNode
        Node to rename.
    new_name: str
        New name of the node.
    fn: SSAFunction
        Function that contains graph to operate on.
    """
    old_name = node.name
    node.name = new_name
    for i in node.inputs:
        idx = fn.graph[i].outputs.index(old_name)
        fn.graph[i].outputs[idx] = new_name
        if old_name in fn.graph[i].control_outputs:
            idx = fn.graph[i].control_outputs.index(old_name)
            fn.graph[i].control_outputs[idx] = new_name

    for o in node.outputs:
        idx = fn.graph[o].inputs.index(old_name)
        fn.graph[o].inputs[idx] = new_name
        if old_name in fn.graph[o].control_inputs:
            idx = fn.graph[o].control_inputs.index(old_name)
            fn.graph[o].control_inputs[idx] = new_name

    for i in node.control_inputs:
        if old_name in fn.graph[i].control_outputs:
            idx = fn.graph[i].control_outputs.index(old_name)
            fn.graph[i].control_outputs[idx] = new_name

    for o in node.control_outputs:
        if old_name in fn.graph[o].control_inputs:
            idx = fn.graph[o].control_inputs.index(old_name)
            fn.graph[o].control_inputs[idx] = new_name

    fn.graph[new_name] = fn.graph.pop(old_name)


def _flatten_while_loop_namespaces(tf_ssa, fn_name):
    """
    A pass to flatten namespaces for sub-graphs of the control flow while_loop
    op. For example, the while_loop's has two sub-graphs, "cond" and "body",
    all the nodes in the graph will be prefixing the sub-graph's name. This
    pass is required for converting control flow v2 ops (enabled by default in
    TensorFlow 2.0+) as the original sub-graphs will contain duplicated names.

    Parameters
    ----------
    tf_ssa: NetworkEnsemble
        An object that contains multiple functions / sub-graphs.
    fn_name: str
        Name of the function / sub-graph to operate on.
    """
    count = 0
    fn = tf_ssa.functions.get(fn_name)
    for name, node in fn.graph.copy().items():
        if node.op not in {'StatelessWhile', 'While'}:
            continue

        sub_fn_names = [node.attr.get('cond'), node.attr.get('body')]

        for sf_name in sub_fn_names:
            sf = tf_ssa.functions.get(sf_name)
            prefix = '{}/{}'.format(node.name, sf_name)

            for old_name, n in sf.graph.copy().items():
                _rename_node_in_fn(n, '{}/{}'.format(prefix, old_name), sf)
                count += 1

            sf.inputs = ['{}/{}'.format(prefix, n) for n in sf.inputs]
            sf.outputs = ['{}/{}'.format(prefix, n) for n in sf.outputs]
            _flatten_while_loop_namespaces(tf_ssa, sf_name)

    msg = 'flatten_while_loop_namespaces: {} nodes renamed in {}'
    logging.info(msg.format(count, fn_name))


def _insert_op(fn, op, name, attr=None):
    """
    Create a node with given attributes, then insert to the target graph in
    given function.

    Parameters
    ----------
    fn: SSAFunction
        Function that contains graph to operate on.
    op: str
        Type of the operation for the new node.
    name: str
        Name of the new node.
    attr: dict or None (optional)
        Attributes of the new node.

    Returns
    -------
    node: ParsedTFNode
        New node object.
    """
    node = ParsedTFNode()
    node.op = op
    node.name = name
    if attr is not None:
        node.attr = attr
    fn.graph[node.name] = node
    return node


def _insert_function_entry(fn, name):
    return _insert_op(fn=fn, op='function_entry', name=name)


def _insert_return(fn, name):
    return _insert_op(fn=fn, op='return', name=name)


def _insert_make_tuple(fn, name):
    return _insert_op(fn=fn, op='make_tuple', name=name)


def _insert_get_tuple(fn, name, idx):
    return _insert_op(fn=fn, op='get_tuple', name=name, attr={'index': idx})


def _rewrite_cond_functions(tf_ssa, fn):
    """
    Rewrite tf.cond's sub-graphs with get_tuple, make_tuple, function_entry and
    return ops. This rewrite is required in order to convert functional form
    control flow v2 nodes 'StatelessIf' and 'If'.

    Parameters
    ----------
    tf_ssa: NetworkEnsemble
        An object that contains multiple functions / sub-graphs.
    fn: SSAFunction
        Function that contains graph to operate on.

    Examples
    --------

    Input:

        Before pass "main" graph:

            [const/greater/y] ---------\
            [placeholder/args_0] -> [greater] -> [if] -> [identity]
                              \------------------/  \--> [identity]
            [placeholder/args_1] ----------------/

        Before pass "then" graph:

            [const/sub/y] ---------------\
            [placeholder/sub_args_0] -> [sub]
            [placeholder/sub_args_1] -> [identity]

        Before pass "else" graph:

            [const/add/y] ---------------\
            [placeholder/add_args_0] -> [add]

            [const/mul/y] ---------------\
            [placeholder/add_args_1] -> [mul]

    Output:

        After pass "main" graph:

            [const/greater/y] ---------\
            [placeholder/args_0] -> [greater] -> [make_tuple] -> [if] -> [get_tuple] -> [identity]
                              \---------------------/               \--> [get_tuple] -> [identity]
            [placeholder/args_1] -------------------/

        After pass "then" graph:

                                      [const/sub/y] ---------------\
            [entry] -> [get_tuple] -> [placeholder/sub_args_0] -> [sub] -> [make_tuple] -> [return]
                    -> [get_tuple] -> [placeholder/sub_args_1] -----------------/

        After pass "else" graph:

                                      [const/add/y] ---------------\
            [entry] -> [get_tuple] -> [placeholder/add_args_0] -> [add] -> [make_tuple] -> [return]
                    -> [get_tuple] -> [placeholder/add_args_1] -> [mul] --------/
                                      [const/mul/y] ---------------/

    """
    for name, node in fn.graph.copy().items():
        if node.op not in {'StatelessIf', 'If'}:
            continue

        then_fn_name = node.attr.get('then_branch')
        else_fn_name = node.attr.get('else_branch')
        msg = "Rewriting '{}' ({}) sub-graphs: then '{}', else '{}'"
        logging.info(msg.format(
            node.name, node.op, then_fn_name, else_fn_name))

        then_fn = tf_ssa.functions.get(then_fn_name)
        else_fn = tf_ssa.functions.get(else_fn_name)

        # insert function entry nodes
        then_entry = _insert_function_entry(then_fn, name + '_then_entry')
        else_entry = _insert_function_entry(else_fn, name + '_else_entry')

        # pack node inputs to a single tuple
        cond_input = _insert_make_tuple(fn, name + '_inputs')
        for i in node.inputs:
            disconnect_edge(fn.graph, i, node.name)
            connect_edge(fn.graph, i, cond_input)
        connect_edge(fn.graph, cond_input, node.name)

        # unpack node outputs to multiple get_tuples
        for i, o in enumerate(node.outputs):
            cond_output = _insert_get_tuple(
                fn, name + '_outputs_' + str(i), i)
            idx = fn.graph[o].inputs.index(node.name)
            replace_dest(fn.graph, node, o, cond_output)
            connect_edge_at_index(fn.graph, cond_output, o, idx)

        # fetch inputs using get_tuple for then branch
        for i, n in enumerate(then_fn.inputs):
            then_input = _insert_get_tuple(
                then_fn, name + '_then_input_' + str(i), i + 1)
            connect_edge(then_fn.graph, then_entry, then_input)
            replace_node(then_fn.graph, n, then_input)
            delete_node(then_fn.graph, n)

        # fetch inputs using get_tuple for else branch
        for i, n in enumerate(else_fn.inputs):
            else_input = _insert_get_tuple(
                else_fn, name + '_else_input_' + str(i), i + 1)
            connect_edge(else_fn.graph, else_entry, else_input)
            replace_node(else_fn.graph, n, else_input)
            delete_node(else_fn.graph, n)

        # returns a tuple of value(s) as output for then branch
        then_output = _insert_make_tuple(then_fn, name + '_then_output')
        for o in then_fn.outputs:
            if o not in then_fn.graph.keys():
                # from identity, map back to get_tuple node
                o = '{}_then_input_{}'.format(name, then_fn.inputs.index(o))
            connect_edge(then_fn.graph, o, then_output.name)

        then_return = _insert_return(then_fn, name + '_then_return')
        connect_edge(then_fn.graph, then_output.name, then_return.name)

        # returns a tuple of value(s) as output for else branch
        else_output = _insert_make_tuple(else_fn, name + '_else_output')
        for o in else_fn.outputs:
            if o not in else_fn.graph.keys():
                # from identity, map back to get_tuple node
                o = '{}_else_input_{}'.format(name, else_fn.inputs.index(o))
            connect_edge(else_fn.graph, o, else_output.name)

        else_return = _insert_return(else_fn, name + '_else_return')
        connect_edge(else_fn.graph, else_output.name, else_return.name)


def _eliminate_loop_cond_nodes(tf_ssa, fn):
    """
    Eliminate loop condition nodes, such as loop_counters, max_iterations from
    the cond sub-graph and body sub-graph of tf.while_loop.

    Parameters
    ----------
    tf_ssa: NetworkEnsemble
        An object that contains multiple functions / sub-graphs.
    fn: SSAFunction
        Function that contains graph to operate on.

    Examples
    --------

    Input:

        Before pass "main" graph:

            [while/maximum_iterations] -----\
            [while/loop_counter] -------> [while] --> [identity]
            [placeholder/args_0] ----------/

        Before pass "cond" graph:

            [const/mean] -------\
            [placeholder] --> [mean] --> [greater]
            [const/greater/y] --------------/

            [while_maximum_iterations], [while_loop_counter] (not connected)

        Before pass "body" graph:

            [const/sub/y] ------\
            [placeholder] ---> [sub]

            [const/add/y] ------------\
            [while_loop_counter] --> [add]

            [while_maximum_iterations] (not connected)

    Output:

        After pass "main" graph:

            [placeholder/args_0] --> [while] --> [identity]

        After pass "cond" graph:

            [const/mean] -------\
            [placeholder] --> [mean] --> [greater]
            [const/greater/y] --------------/

        After pass "body" graph:

            [const/sub/y] ------\
            [placeholder] ---> [sub]
    """
    for name, node in fn.graph.copy().items():
        if node.op not in {'StatelessWhile', 'While'}:
            continue

        cond_fn = tf_ssa.functions.get(node.attr.get('cond'))
        body_fn = tf_ssa.functions.get(node.attr.get('body'))

        cond_lc_nodes = {cond_fn.inputs.pop(0), cond_fn.inputs.pop(0)}
        logging.info("Eliminating {} from cond fn".format(cond_lc_nodes))

        for n in cond_lc_nodes:
            delete_node(cond_fn.graph, n)

        body_lc_nodes = {body_fn.inputs.pop(0), body_fn.inputs.pop(0)}
        q = list(body_lc_nodes)

        # delete entire sub-fn
        while len(q) > 0:
            n = body_fn.graph[q.pop(0)]
            for o in n.outputs:
                if o not in body_lc_nodes:
                    q.append(o)
                body_lc_nodes.add(o)
                for i in body_fn.graph[o].inputs:
                    if i not in body_lc_nodes:
                        q.append(i)
                    body_lc_nodes.add(i)

        logging.info("Eliminating {} from body fn".format(body_lc_nodes))

        for n in body_lc_nodes:
            delete_node(body_fn.graph, n)

        # remove if in outputs
        for n in body_lc_nodes:
            if n in body_fn.outputs:
                body_fn.outputs.remove(n)


def _rewrite_while_loop_functions(tf_ssa, fn):
    """
    Rewrite tf.while_loop's sub-graphs with get_tuple, make_tuple,
    function_entry and return ops. This rewrite is required in order to convert
    functional form control flow v2 nodes 'StatelessWhile' and 'While'.

    Parameters
    ----------
    tf_ssa: NetworkEnsemble
        An object that contains multiple functions / sub-graphs.
    fn: SSAFunction
        Function that contains graph to operate on.

    Example
    -------

    Input:

        Before pass "main" graph:

            [placeholder/args_0] --> [while] --> [identity]

        Before pass "cond" graph:

            [const/mean] -------\
            [placeholder] --> [mean] --> [greater]
            [const/greater/y] --------------/

        Before pass "body" graph:

            [const/sub/y] ------\
            [placeholder] ---> [sub]

    Output:

        After pass "main" graph:

            [placeholder/args_0] --> [make_tuple] --> [while] --> [get_tuple] --> [identity]

        After pass "cond" graph:

                                      [const/mean] ------\
            [entry] -> [get_tuple] -> [placeholder] -> [mean] -> [greater] -> [make_tuple] -> [return]
                                      [const/greater/y] ------------/

        After pass "body" graph:

                                      [const/sub/y] ----\
            [entry] -> [get_tuple] -> [placeholder] -> [sub] -> [make_tuple] -> [return]
    """
    for name, node in fn.graph.copy().items():
        if node.op not in {'StatelessWhile', 'While'}:
            continue

        cond_fn_name = node.attr.get('cond')
        body_fn_name = node.attr.get('body')
        msg = "Rewriting '{}' ({}) sub-graphs: cond '{}', body '{}'"
        logging.info(msg.format(
            node.name, node.op, cond_fn_name, body_fn_name))

        cond_fn = tf_ssa.functions.get(cond_fn_name)
        body_fn = tf_ssa.functions.get(body_fn_name)

        # insert function entry nodes
        cond_entry = _insert_function_entry(cond_fn, name + '_cond_entry')
        body_entry = _insert_function_entry(body_fn, name + '_body_entry')

        # pack node inputs to a single tuple
        loop_input = _insert_make_tuple(fn, name=name + '_inputs')
        for i in node.inputs:
            disconnect_edge(fn.graph, i, node.name)
            connect_edge(fn.graph, i, loop_input)
        connect_edge(fn.graph, loop_input, node.name)

        # unpack node outputs to multiple get_tuples
        for i, o in enumerate(node.outputs):
            loop_output = _insert_get_tuple(
                fn, name + '_output_' + str(i), i)
            idx = fn.graph[o].inputs.index(node.name)
            replace_dest(fn.graph, node, o, loop_output)
            connect_edge_at_index(fn.graph, loop_output, o, idx)

        # fetch inputs using get_tuple for cond fn
        for i, n in enumerate(cond_fn.inputs):
            cond_input = _insert_get_tuple(
                cond_fn, name + '_cond_input_' + str(i), i)
            connect_edge(cond_fn.graph, cond_entry, cond_input)
            replace_node(cond_fn.graph, n, cond_input)
            delete_node(cond_fn.graph, n)

        # fetch inputs using get_tuple for body fn
        for i, n in enumerate(body_fn.inputs):
            body_input = _insert_get_tuple(
                body_fn, name + '_body_input_' + str(i), i)
            connect_edge(body_fn.graph, body_entry, body_input)
            replace_node(body_fn.graph, n, body_input)
            delete_node(body_fn.graph, n)

        # returns a tuple of value(s) as output for cond fn
        cond_output = _insert_make_tuple(cond_fn, name + '_cond_output')
        for o in cond_fn.outputs:
            connect_edge(cond_fn.graph, o, cond_output.name)

        cond_return = _insert_return(cond_fn, name + '_cond_return')
        connect_edge(cond_fn.graph, cond_output.name, cond_return.name)

        # returns a tuple of value(s) as output for body branch
        body_output = _insert_make_tuple(body_fn, name + '_body_output')

        for o in body_fn.outputs:
            connect_edge(body_fn.graph, o, body_output.name)

        body_return = _insert_return(body_fn, name + '_body_return')
        connect_edge(body_fn.graph, body_output.name, body_return.name)


def flatten_while_loop_namespaces(tf_ssa):
    _flatten_while_loop_namespaces(tf_ssa, fn_name='main')


def rewrite_control_flow_functions(tf_ssa):
    for fn_name, fn in tf_ssa.functions.items():
        _rewrite_cond_functions(tf_ssa, fn)
    for fn_name, fn in tf_ssa.functions.items():
        _eliminate_loop_cond_nodes(tf_ssa, fn)
        _rewrite_while_loop_functions(tf_ssa, fn)
