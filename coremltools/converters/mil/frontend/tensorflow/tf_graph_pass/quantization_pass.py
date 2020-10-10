# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from ..basic_graph_ops import delete_node
import logging
import sys

def delete_fakequant_node_and_repair_graph(g, node):
    inputs = node.inputs
    # Delete const inputs of the fakequant op
    for i in inputs:
        if g[i].op == 'Const':
            delete_node(g, i)
        else:
            non_const_input = i
    outputs = node.outputs
    # Append FakeQuant Op's outputs to its input node's outputs
    g[non_const_input].outputs = [i for i in g[non_const_input].outputs if i != node.name]
    g[non_const_input].outputs.extend(outputs)
    # Modify the FakeQuant op's outputs to set FakeQuant op's parent node as the new input.
    for i in outputs:
        for j in range(len(g[i].inputs)):
            if g[i].inputs[j] == node.name:
                g[i].inputs[j] = non_const_input
    delete_node(g, node)

def quantization_pass_impl(fn):
    all_quantization_ops = [i for i in fn.graph.values() if "FakeQuant" in i.op]
    for node in all_quantization_ops:
        is_const_input = True
        for input in node.inputs:
            if fn.graph[input].op != 'Const':
                is_const_input = False
        if not is_const_input and ('weights_quant' not in input):
            # If activation quantization -
            # Delete the FakeQuant op and its const inputs,
            # Append FakeQuant Op's outputs to its input node's outputs,
            # Modify the FakeQuant op's outputs to reflect the 'new' input node.
            delete_fakequant_node_and_repair_graph(fn.graph, node)
        else:
            # If weight quantization -
            # Add attributes of the FakeQuant op to its output's attr dict
            for output in node.outputs:
                output_node = fn.graph[output]
                output_node.attr['quantize'] = True
                output_node.attr['num_bits'] = node.attr['num_bits']
                output_node.attr['narrow_range'] = node.attr['narrow_range']
                output_node.attr['quantize_min'] = fn.graph[node.inputs[1]].value.val
                output_node.attr['quantize_max'] = fn.graph[node.inputs[2]].value.val

def quantization_pass(tfssa):
    """
    Delete activation quantization ops and repair TF graph:
        If the FakeQuant op is not connected to constant inputs (which means that the op performs activation
        quantization) then delete that FakeQuant op and repair the graph.
    Edit weight quantization ops:
        If the FakeQuant op is connected to constant inputs then add its attributes to its output op so that parameters
        min, max, narrow_range, num_bits are available (in addition to weights) to downstream ops for denoting and
        supporting weight quantization.
    """
    for v in tfssa.functions.values():
        quantization_pass_impl(v)