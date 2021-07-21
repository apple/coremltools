# -*- coding: utf-8 -*-

#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import defaultdict
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

def _is_sink(op):
    return sum(len(output._child_ops) for output in op.outputs) == 0

def _topological_reorder_block(block):
    sink_nodes = []
    other_ordered_operations = []
    for i, op in enumerate(block.operations):
        for b in op.blocks:
            _topological_reorder_block(b)

        if _is_sink(op):
            sink_nodes.append(op)
        else:
            other_ordered_operations.append(op)

    block.operations = other_ordered_operations + sink_nodes

@register_pass(namespace="common")
def topological_reorder(prog):
    """
    Topologically reorders the list of operations in a program by moving all sink nodes to the very end in that list

    Please checkout: test_move_sink_casts_to_the_end in test_passes.py::TestTopologicalReorder
    """
    for f_name, f in prog.functions.items():
        _topological_reorder_block(f)

