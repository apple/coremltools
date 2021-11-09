#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict
import numpy as np

from .helper import _are_ops_identical
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


def _is_op_eligible_to_be_removed(op):
    if len(op.blocks) != 0 or op.op_type.startswith("random"):
        return False
    else:
        return True


def _get_candidate_ops_list(prospective_ops_list):
    od = OrderedDict()
    enclosing_block = [op.enclosing_block for op in prospective_ops_list]
    if len(set(enclosing_block)) > 1: # all candidate ops must belong to the same block
        return []
    for op in prospective_ops_list:
        if _is_op_eligible_to_be_removed(op):
            od[op] = enclosing_block[0].operations.index(op)
    # sort the ops according to their index of appearing in block.operations, which is topologically sorted
    return [x[0] for x in sorted(od.items(), key=lambda t: t[1])]


def _get_candidate_ops_lists_from_var(var):
    '''
    Return a list of lists.
    Each element is a list of a subset of the child ops of var, which satisifies the following conditions:
    - they are of the same op_type
    - ops are not repeated in it. The .child_ops property of a var may sometimes contain an op repeated more than once
    - the ops are ordered based on the order in which they appear in the block.operations list (which is topologically sorted),
      with ops appearing earlier in that list appearing first here.
    '''
    candidate_ops_lists = []

    op_types_to_ops = OrderedDict()
    for op in var.child_ops:
        if op.op_type in op_types_to_ops:
            op_types_to_ops[op.op_type].append(op)
        else:
            op_types_to_ops[op.op_type] = [op]

    for v in op_types_to_ops.values():
        if len(v) > 1:
            candidate_ops_list = _get_candidate_ops_list(v)
            if len(candidate_ops_list) > 1:
                candidate_ops_lists.append(candidate_ops_list)

    return candidate_ops_lists


def _try_to_remove_ops(candidate_ops_list):
    # candidate_ops_list contains ops in topological order.
    # All the ops in candidate_ops_list will be compared to the first op, and removed if identical to it.
    # Removing ops later in the topological order is much easier, as their output vars
    # can simply be replaced by the output var of the first_op, this doesn't require
    # changing any op order in the block.
    if len(candidate_ops_list) < 2:
        return False
    first_op = candidate_ops_list[0]
    block = first_op.enclosing_block

    # currently, we only consider the cases when the op has 1 output.
    # The replace var logic below only handles the single output case.
    if len(first_op.outputs) > 1:
        return False

    ops_to_remove = []
    for op in candidate_ops_list[1:]:
        if op.outputs[0] not in block.outputs: # to make sure we don't remove an output op
            if _are_ops_identical(first_op, op):
                ops_to_remove.append(op)

    if len(ops_to_remove) == 0:
        return False

    # remove uses of output vars of the ops to be removed.
    # This can be safely done, since all the ops in ops_to_remove
    # appear after first_op, hence first_op.outputs[0] variable is in
    # scope before the op's output var
    for op in ops_to_remove:
        op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=op, old_var=op.outputs[0], new_var=first_op.outputs[0]
        )
    block.remove_ops(ops_to_remove)
    return True



def _try_to_transform(parent_var):
    """
    scan the children ops to parent_var, to find and remove indentical ops, if any.
    Returns True, if succesful in finding such redundant ops.
    """
    candidate_ops_lists = _get_candidate_ops_lists_from_var(parent_var)
    block_changed = False
    for ops_list in candidate_ops_lists:
        if _try_to_remove_ops(ops_list):
            block_changed = True
    return block_changed


def _remove_redundant_ops_in_block(block):

    if isinstance(block.inputs, dict):
        block_input_var_list = list(block.inputs.values())
    elif isinstance(block.inputs, (list, tuple)):
        block_input_var_list = block.inputs
    else:
        raise ValueError("Unrecognized type of block.inputs, its neither a list nor dict.")

    # iterate over the block inputs
    for input_var in block_input_var_list:
        if len(input_var.child_ops) > 1:
            with block:
                _try_to_transform(input_var)

    # iterate over the ops in the block
    graph_updated = False
    for op in block.operations:
        if op.op_type == "const":
            continue

        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = _remove_redundant_ops_in_block(b)

        if len(op.outputs) > 0 and len(op.outputs[0].child_ops) > 1:
            with block:
                # currently, we only check the first output of the op
                # this can be extended, if required, to check for other outputs.
                graph_updated = _try_to_transform(op.outputs[0])
            # has to break as the downstream iterator is affected.
            if graph_updated:
                return graph_updated
    return graph_updated


@register_pass(namespace="common")
class remove_redundant_ops(AbstractGraphPass):
    """
    If there are multiple ops with "identical" inputs, then they are redundant and all but one of them can be removed.
    This pass checks and removes such ops.

    Since all inputs to ops in MIL are named, two ops with same op_types, can be compared by comparing their
    correspondingly named inputs. Inputs are treated as identical if:
    - if the input is a constant var, then its value should have the same dtype and numerical value
    - if the input is a non constant var, then it should be the same var object

    This pass iterates over the ops, takes its first output var, and then builds a candidate op list from the child
    ops of this var.
    These candidate ops list contains ops, of the same op_type, and arranged in the topological order.
    From each of these candidate ops list, the 2nd, 3rd,....so on, ops are pairwise compared with the first op,
    and if identical to it, they are removed.
    e.g.:

    Input:
        %0 = op0(...)
        %1 = op1(...)
        %2 = const(val=4.5)
        %3 = const(val=4.5)
        %4 = op2(%1, %0, %2)
        %5 = op3(%1, %0, %3)

    op3 will be removed and all uses of %5 will be replaced by %4

    Output:
        %0 = op0(...)
        %1 = op1(...)
        %2 = const(val=4.5)
        %3 = const(val=4.5) # this will get removed later by dead code elimination pass
        %4 = op2(%1, %0, %2)

    For more examples, see "TestRemoveRedundantOpsPass"
    """
    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = _remove_redundant_ops_in_block(f)


