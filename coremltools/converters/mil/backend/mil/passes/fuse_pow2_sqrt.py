#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


def _match_pattern(op):
    pow_op, sqrt_op = None, None

    # check the current op is pow(2) or sqrt
    if op.op_type == "pow" and op.y.val == 2:
        pow_op = op
    if op.op_type == "sqrt":
        sqrt_op = op

    # check the children of the current op
    child_ops = op.outputs[0].child_ops

    # if the op output is a block output or there is more than one child, fast fail
    if op.outputs[0] in op.enclosing_block.outputs or len(child_ops) != 1:
        return None

    # if we have pow(2), check for sqrt
    if pow_op and child_ops[0].op_type == "sqrt":
        sqrt_op = child_ops[0]
    # if we have sqrt, check for pow(2)
    elif sqrt_op and child_ops[0].op_type == "pow" and child_ops[0].y.val == 2:
        pow_op = child_ops[0]

    # if we don't have both ops, fast fail
    if not pow_op or not sqrt_op:
        return None

    # check that the two ops are connected
    if pow_op.outputs[0].name != sqrt_op.x.name and sqrt_op.outputs[0].name != pow_op.x.name:
        return None

    # return the other op
    return pow_op if pow_op != op else sqrt_op


def _try_to_transform(op1, op2, block):
    # replace the pow2(x) --> sqrt(x) with identity(x)
    x = mb.identity(x=op1.x, name= op2.outputs[0].name, before_op=op1)

    # update the graph
    op2.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=op2, old_var=op2.outputs[0], new_var=x
    )

    # remove the ops
    block.remove_ops([op1, op2])

    return True


@block_context_manager
def _fuse_pow2_sqrt(block):
    fusion_occurred = False
    for op in list(block.operations):
        if op.enclosing_block is None:
            continue
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = _fuse_pow2_sqrt(b)
        if len(op.blocks) > 0:
            continue

        op2 = _match_pattern(op)
        if op2 is not None:
            if _try_to_transform(op, op2, block):
                fusion_occurred = True
    return fusion_occurred


@register_pass(namespace="mil_backend")
class fuse_pow2_sqrt(AbstractGraphPass):
    """
    Fold pow(x, 2) --> sqrt(x) into identity(x)

    Given:
        %1 = pow(x=%0, y=2)
        %2 = sqrt(x=%1)
        ...
        %1 = sqrt(x=%0)
        %2 = pow(x=%1, y=2)
        ...

    Result:
        %3 = identity(%0)
        ...
    """
    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = _fuse_pow2_sqrt(f)
