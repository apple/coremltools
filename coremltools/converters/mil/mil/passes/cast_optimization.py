#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil import Builder as mb

@register_pass(namespace="common")
class cast_optimization(AbstractGraphPass):
    """
    This optimization pass,
    - Removes redundant cast op i.e cast where source and destination tensors have same dtypes.
    - Either Cancel or Fuses any two consecutive cast ops, repeatedly.

    After this pass, there can't be any consecutive casts present in the program.

    Please checkout: test_cast_optimization.py, for examples.

    It is a non-algebraic translation which assumes that the upcasting doesn't change the user-intent.
    For example,
    Input graph:
    input -----> cast(dtype="fp16") -----> cast(dtype="fp32") ----> square ---> out

    Output graph:
    input -----> square -----> out

    The input graph has maximum precision of fp16 while the output graph has fp32 precision.

    """
    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            cached_vars = {}
            """
            Cached vars is used when all the following conditions are met:
            1. When the output of a cast gets fed into multiple casts of same configuration
            2. And, these 2 consecutive casts can be fused into a single cast.
            When above conditions are satisfied, we create a NEW fused cast op ONLY once and
            the output of all these consecutive casts gets replaced with the ouptut of this fused cast.

            Input graph:
                                        |---->cast(dtype="fp16")---->square--->out_1
                                        |
            input---->cast(dtype="int32")---->cast(dtype="fp16")---->relu--->out_2
                                        |
                                        |---->cast(dtype="fp16")---->log--->out_3

            Output graph:

                                                 |---->square--->out_1
                                                 |
            input---->new_fused_cast(dtype="fp16")---->relu--->out_2
                                                 |
                                                 |---->log--->out_3
            """
            while block_changed:
                block_changed = _fuse_or_cancel_consecutive_casts_block(f, cached_vars)

class Node(object):
    def __init__(self, op_type, match_criterion=None):
        """

        :param op_type: Type of an operation.
        :param match_criterion: A callable function that a MIL op and returns a boolean

        Examples:
            Node("mul"),
            Node("round"),
            Node("add", lambda op: op.y.val == 0),
            Node("clip", lambda op: op.alpha.val == -128 and op.beta.val == 127),
            Node("cast", lambda op: op.dtype.val == "int8"),
            Node("cast", lambda op: op.dtype.val == "fp32"),
        """

        self.op_type = op_type
        if not match_criterion:
            match_criterion = lambda op: True

        self.match_criterion = match_criterion


def _match_linear_pattern(root, pattern):
    """
    Use Depth First Search to match the pattern

    :param root: operation
    :param pattern: List[Node]
    :return: Return List[operation] if pattern matches entirely else []
    """
    op = root
    if not pattern or len(op.outputs) != 1:
        return []

    node = pattern[0]
    if op.op_type != node.op_type:
        return []

    if not node.match_criterion(op):
        return []

    for child in op.outputs[0].child_ops:
        op_list = [op] + _match_linear_pattern(child, pattern[1:])
        if len(op_list) == len(pattern):
            return op_list

    return []


def _try_to_transform(root_op, cached_vars):
    block = root_op.enclosing_block

    # Scenario: Redundant cast when source and destination dtype are same.
    if root_op.op_type == "cast" and root_op.x.is_tensor_or_scalar_of(dtype=root_op.dtype.val):
        block.replace_uses_of_var_after_op(
            anchor_op=root_op,
            old_var=root_op.outputs[0],
            new_var=root_op.x,
        )
        block.remove_ops([root_op])
        return True

    # Scenario: Consecutive casts
    list_of_ops_in_pattern = _match_linear_pattern(
        root_op,
        [
            Node("cast"),
            Node("cast"),
        ],
    )

    if not list_of_ops_in_pattern:
        return False

    cast_1, cast_2 = list_of_ops_in_pattern

    fused_output_var_name = cast_1.x.name + "_to_{}".format(cast_2.dtype.val)

    if cast_1.x.is_tensor_or_scalar_of(dtype=cast_2.dtype.val):
        # when consecutive casts cancel each other
        # Please checkout: test_linear_consecutive_cast_ops_cancellation in test_cast_optimization.py
        new_output_var = cast_1.x
    elif fused_output_var_name in cached_vars:
        # When the output of 1 cast goes into multiple casts of same configuration
        # Please checkout: test_consecutive_fusable_casts_on_all_branches in test_cast_optimization.py
        new_output_var = cached_vars[fused_output_var_name]
    else:
        new_output_var = mb.cast(
            x=cast_1.x,
            dtype=cast_2.dtype,
            name=fused_output_var_name,
            before_op=cast_2,
        )
        cached_vars[fused_output_var_name] = new_output_var

    # It's important to use `cast_2.enclosing_block` over `block` since `cast_2` might be present in
    # a block nested under `block`
    cast_2.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=cast_2,
        old_var=cast_2.outputs[0],
        new_var=new_output_var,
    )

    # Remove just the last cast op and let dce eliminate the rest of the ops if needed,
    # The reason is that first cast op could be feeding into other non-cast ops.
    cast_2.enclosing_block.remove_ops([cast_2])
    return True


def _fuse_or_cancel_consecutive_casts_block(block, cached_vars):
    block_changed = False
    for i, op in enumerate(list(block.operations)):
        for b in op.blocks:
            nested_block_changed = True
            nested_block_cached_vars = {}
            nested_block_cached_vars.update(cached_vars)
            while nested_block_changed:
                nested_block_changed = _fuse_or_cancel_consecutive_casts_block(b, nested_block_cached_vars)

        if len(op.blocks) > 0:
            continue

        # start pattern match if cast op is encountered
        if op.op_type == "cast":
            with block:
                block_changed = _try_to_transform(op, cached_vars)
            # has to break as the downstream iterator is affected.
            if block_changed:
                return block_changed
    return block_changed
