#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil import Builder as mb

def match_pattern(op):
    if op.op_type == "sigmoid":
        # abort fusion if op output is also a block output
        if op.outputs[0] in op.enclosing_block.outputs:
            return None
        # find following op
        child_ops = op.outputs[0].child_ops
        if len(child_ops) == 1:
            mul_op_candidate = list(child_ops)[0]
            if mul_op_candidate.op_type != "mul":
                return None
            mul_inputs_actual = {mul_op_candidate.x.name, mul_op_candidate.y.name}
            mul_inputs_expect = {op.x.name, op.outputs[0].name}
            if mul_inputs_actual != mul_inputs_expect:
                return None
            return mul_op_candidate

    return None


def try_to_transform(sigmoid_op, mul_op, block):
    out_name = mul_op.outputs[0].name
    # create a new silu op
    x = mb.silu(x=sigmoid_op.x, name=out_name, before_op=sigmoid_op)
    mul_op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=mul_op, old_var=mul_op.outputs[0], new_var=x
    )
    # Remove all the ops at once
    block.remove_ops([sigmoid_op, mul_op])
    return True


def fuse_activation_silu_block(block):
    fusion_status = False
    for op in list(block.operations):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = fuse_activation_silu_block(b)
        if len(op.blocks) > 0:
            continue

        mul_op = match_pattern(op)
        if mul_op is not None:
            with block:
                fusion_status = try_to_transform(op, mul_op, block)
            # has to break as the downstream iterator is affected.
            if fusion_status:
                return fusion_status
    return fusion_status


@register_pass(namespace="mil_backend")
def fuse_activation_silu(prog):
    """
    Fold x * sigmoid(x) into silu(x)

    Given:
        %1 = sigmoid(x=%0)
        %2 = mul(x=%0, y=%1) or mul(x=%1, y=%0)
        ...

    Result:
        %3 = silu(%0)
        ...
    """
    for f_name, f in prog.functions.items():
        block_changed = True
        while block_changed:
            block_changed = fuse_activation_silu_block(f)
