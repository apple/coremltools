#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import (
    _check_child_op_type, block_context_manager)
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


def _match_and_replace_pattern(block, relu_op):
    if not (relu_op.op_type == "relu" and _check_child_op_type(relu_op, "relu")):
        return False

    child_relu_op = list(relu_op.outputs[0].child_ops)[0]
    return _replace_ops(block, relu_op, child_relu_op)


def _replace_ops(block, relu_op, child_relu_op):
    if relu_op.enclosing_block.try_replace_uses_of_var_after_op(
        anchor_op=relu_op, old_var=child_relu_op.outputs[0], new_var=relu_op.outputs[0]
    ):
        block.remove_ops([child_relu_op])
        return True
    return False


@block_context_manager
def _merge_relus_in_block(block):
    def help_merge_relu_ops(block):
        for op in list(block.operations):
            if _match_and_replace_pattern(block, op):
                return True
        return False

    block_changed = True
    while block_changed:
        block_changed = help_merge_relu_ops(block)


@register_pass(namespace="common")
class merge_consecutive_relus(AbstractGraphPass):
    """
    Identify consecutive 'relu' layers which could be merged into a single 'relu' layer.

    Input graph:
    input ------> relu -----> 1 or more relu layers ---> out

    Output graph:
    input ------> relu ---> out
    """
    def apply(self, prog):
        for f in prog.functions.values():
            _merge_relus_in_block(f)
