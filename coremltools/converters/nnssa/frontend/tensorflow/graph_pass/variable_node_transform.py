# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from ....commons.basic_graph_ops import disconnect_vertex_ins, delete_node

from coremltools.converters.nnssa.commons.features import Features

# Variable nodes are not horribly complicated.
#
# There are Variable nodes which don't really do much on their own
#
# To initialize, there is an additional Assign op which is just dangling away
# on one side which assigns from "Variable/initial_value".
#
# [Variable] --> Assign <-- Const (VariableName/initial_value)
#      |
#      | ... rest of graph ...
#      v
# ... Assign <---- New Values
# ... etc
#
# Reads of the variable go through an Identity node with the name
# VariableName/read, and has attribute _class:loc:@VariableName.
#
# Writes of the variable go through an Assign nodes which take as input
# one Variable and one value, and has attribute _class:loc:@VariableName.
# Assign also returns the new value of the variable.
#
#
#
#  - We transform Variable to a function attribute
#  - We transform Assign ops to just "set_global" with attribute variable:VariableName
#  - We transform Read ops to just "get_global" with attribute variable:VariableName
def remove_variable_node_impl(fn, ssa):
    variables = [var for var in fn.graph.values() if var.op == 'VariableV2']
    assigns = [assign for assign in fn.graph.values() if assign.op == 'Assign']
    if Features.new_ssa():
        reads = [
            read for read in fn.graph.values() if read.op == 'Identity' and len(read.inputs) == 1
            and fn.graph[read.inputs[0].name].op == 'VariableV2'
        ]
    else:
        reads = [
            read for read in fn.graph.values() if read.op == 'Identity' and len(read.inputs) == 1
            and fn.graph[read.inputs[0]].op == 'VariableV2'
        ]

    # find the variable initial values
    variable_values = {}
    additional_nodes_to_delete = []
    for v in variables:
        v.parse_from_attr()
        variable_values[v.name] = v.datatype()
        for node in fn.graph.values():
            if Features.new_ssa():
                if node.op == 'Assign' and node.inputs[0] == v.name and node.inputs[
                        1].name == v.name + "/initial_value":
                    variable_values[v.name] = fn.graph[node.inputs[1].name].value
                    additional_nodes_to_delete += [node.name, node.inputs[1].name]
            else:
                if node.op == 'Assign' and node.inputs[0] == v.name and node.inputs[
                        1] == v.name + "/initial_value":
                    variable_values[v.name] = fn.graph[node.inputs[1]].value
                    additional_nodes_to_delete += [node.name, node.inputs[1]]
    for r in reads:
        r.op = 'get_global'
        if Features.new_ssa():
            r.attr['variable'] = r.inputs[0].name
        else:
            r.attr['variable'] = r.inputs[0]
        disconnect_vertex_ins(fn.graph, r.name)

    # transform writes to set_global
    for r in assigns:
        r.op = 'set_global'
        if Features.new_ssa():
            r.attr['variable'] = r.inputs[0].name
        else:
            r.attr['variable'] = r.inputs[0]

    for var in variables:
        delete_node(fn.graph, var.name)

    for node in additional_nodes_to_delete:
        delete_node(fn.graph, node)

    for k, v in variable_values.items():
        ssa.variables[k] = v


def remove_variable_nodes(ssa):
    """
    This should be performed after constant propagation pass.
    """
    for v in ssa.functions.values():
        remove_variable_node_impl(v, ssa)
