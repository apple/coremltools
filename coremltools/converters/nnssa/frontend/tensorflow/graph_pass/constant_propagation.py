# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import tensorflow as tf

from ...graph_pass.delete_constant import delete_unnecessary_constant_nodes
from ....commons import builtins
from ....commons.parse import numpy_val_to_builtin_val
from ....commons.basic_graph_ops import const_determined_nodes


def constant_propagation(nnssa):
    # we are going to rely on the tensorflow graph to perform constant
    # propagation. We construct a new graph comprising of only the
    # constant nodes.

    from tensorflow.core.framework import graph_pb2
    from tensorflow.core.framework import node_def_pb2
    new_graph = graph_pb2.GraphDef()
    constant_nodes = set()
    constant_node_num_outputs = {}
    for f in nnssa.functions.values():
        generated_nodes = [k for k, v in f.graph.items() if v.original_node is None]
        const_nodes_in_this_graph = const_determined_nodes(f.graph, set(generated_nodes))
        # we can only run TF on nodes with outputs since we must evaluate
        # tensors and not ops
        const_nodes_in_this_graph = [
            i for i in const_nodes_in_this_graph if f.graph[i].op != "NoOp"
        ]
        constant_nodes = constant_nodes.union(set(const_nodes_in_this_graph))

        # topological sort const nodes
        topsort = []
        topsort_set = set()
        while len(const_nodes_in_this_graph) > 0:
            for n in const_nodes_in_this_graph:
                if len(set(f.graph[n].inputs).difference(topsort_set)) == 0:
                    topsort.append(n)
                    topsort_set.add(n)

            const_nodes_in_this_graph = set(const_nodes_in_this_graph).difference(topsort_set)

        for node in topsort:
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(f.graph[node].original_node)
            if '_class' in new_node.attr:
                del new_node.attr['_class']
            del new_node.input[:]
            new_node.input.extend(f.graph[node].inputs)
            if '_output_shapes' in f.graph[node].attr:
                constant_node_num_outputs[node] = len(f.graph[node].attr['_output_shapes'])
            else:
                constant_node_num_outputs[node] = 1
            new_graph.node.extend([new_node])
    constant_nodes = list(constant_nodes)
    try:
        if len(constant_nodes) > 0:
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(new_graph, name="")
                with tf.compat.v1.Session(graph=graph) as sess:
                    query_list = list()
                    for c in constant_nodes:
                        for j in range(constant_node_num_outputs[c]):
                            query_list.append(c + ':' + str(j))
                    control_flow_ops = list()
                    for query in list(query_list):
                        op_name = query.lower()
                        if 'switch' in op_name or 'cond' in op_name:
                            control_flow_ops.append(query)
                            query_list.remove(query)
                    result_list = sess.run(query_list)
                    result = {query_list[i]: result_list[i] for i in range(len(query_list))}
                    # propagate switch one by one
                    for op in control_flow_ops:
                        try:
                            res = sess.run([op])
                            result.update({op: res[0]})
                        except:
                            print('[Constant Propagation] Skip "dead" tensor: {}'.format(op))
                            result.update({op: None})
            for f in nnssa.functions.values():
                for k, v in f.graph.items():
                    if k in constant_node_num_outputs:
                        if constant_node_num_outputs[k] == 1:
                            result_entry = k + ':0'
                            try:
                                v.value, v.datatype = numpy_val_to_builtin_val(result[result_entry])
                            except:
                                print(result_entry)
                                print(result[result_entry])
                        else:
                            values = [
                                result[k + ':' + str(i)]
                                for i in range(constant_node_num_outputs[k])
                            ]
                            try:
                                npval = [numpy_val_to_builtin_val(i) for i in values]
                                v.value = [val[0] for val in npval]
                                v.datatype = builtins.tuple(tuple([val[1] for val in npval]))
                            except:
                                print(values)
    except Exception as e:
        raise RuntimeError("Constant propagation failed: {}".format(e))

    delete_unnecessary_constant_nodes(nnssa)
