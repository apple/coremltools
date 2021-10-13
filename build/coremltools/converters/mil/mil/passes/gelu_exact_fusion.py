#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil import Builder as mb
from .helper import _check_child_op_type, _check_var_scalar_value
import numpy as np

def _try_to_transform(op, block):
    ops_to_remove = []
    if op.x.val is None and op.y.val is None:
        return False

    # check either the op is mul(1/sqrt(2)) or real_div(sqrt(2))
    root_var = op.x if op.y.val is not None else op.y
    if op.op_type == "real_div":
        if not _check_var_scalar_value(op.y, 2 ** 0.5):
            return False
    elif op.op_type == "mul":
        if not (_check_var_scalar_value(op.x, 2 ** -0.5) or _check_var_scalar_value(op.y, 2 ** -0.5)):
            return False
    ops_to_remove.append(op)

    # check if the child op is erf
    if not _check_child_op_type(op, "erf"):
        return False
    erf_op = list(op.outputs[0].child_ops)[0]
    ops_to_remove.append(erf_op)

    # check if the child op is add
    if not _check_child_op_type(erf_op, "add"):
        return False
    add_op = list(erf_op.outputs[0].child_ops)[0]
    if not (_check_var_scalar_value(add_op.x, 1) or _check_var_scalar_value(add_op.y, 1)):
        return False
    ops_to_remove.append(add_op)

    # check if the child op is mul
    if not _check_child_op_type(add_op, "mul"):
        return False
    mul_op = list(add_op.outputs[0].child_ops)[0]

    # now we have two case:
    # (1) first mul by 0.5 and by the root var
    if _check_var_scalar_value(mul_op.x, 0.5) or _check_var_scalar_value(mul_op.y, 0.5):
        ops_to_remove.append(mul_op)
        if not _check_child_op_type(mul_op, "mul"):
            return False
        mul_op_2 = list(mul_op.outputs[0].child_ops)[0]
        if not (mul_op_2.x == root_var or mul_op_2.y == root_var):
            return False
        ops_to_remove.append(mul_op_2)

    # (2) first mul by the root var and then mul by 0.5
    elif mul_op.x == root_var or mul_op.y == root_var:
        ops_to_remove.append(mul_op)
        if not _check_child_op_type(mul_op, "mul"):
            return False
        mul_op_2 = list(mul_op.outputs[0].child_ops)[0]
        if not (_check_var_scalar_value(mul_op_2.x, 0.5) or _check_var_scalar_value(mul_op_2.y, 0.5)):
            return False
        ops_to_remove.append(mul_op_2)

    else:
        return False

    # check that none of the op in this pattern is connected to the output
    # (except the last mul op)
    for op in ops_to_remove[:-1]:
        for out in op.outputs:
            if out in block.outputs:
                return False

    # remove all the ops, and replace with a gelu op
    out_name = mul_op_2.outputs[0].name
    x = mb.gelu(x=root_var, mode="EXACT", name=out_name, before_op=op)

    mul_op_2.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=mul_op_2, old_var=mul_op_2.outputs[0], new_var=x
    )
    # Remove all the ops at once
    block.remove_ops(ops_to_remove)
    return True


def _fuse_gelu_exact_block(block):
    fusion_occurred = False
    for op in list(block.operations):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = _fuse_gelu_exact_block(b)
        if len(op.blocks) > 0:
            # This op can't be real_div or mul
            continue

        if op.op_type in ["mul", "real_div"]:
            with block:
                fusion_occurred = _try_to_transform(op, block)
                # has to break as the downstream iterator is affected.
                if fusion_occurred:
                    return fusion_occurred
    return fusion_occurred


@register_pass(namespace="common")
def fuse_gelu_exact(prog):
    """
    Identify the pattern that corresponds to the exact version of gelu, and replace it with a single
    gelu layer with mode=EXACT
    y = 0.5 * x * (1 + erf (x / srqt (2))

    which can be represented by either:
    (1)
        [...] ----> div (1.414) ---> erf ---> add (1) -----> mul (0.5) ---> mul ---> [...]
          |                                                                  ^
          |                                                                  |
          |-------------------------------------------------------------------

    (2)
        [...] ----> div (1.414) ---> erf ---> add (1) -----> mul ---> mul (0.5) ---> [...]
          |                                                   ^
          |                                                   |
          |----------------------------------------------------

    both result in :
        [...] ----> gelu (mode=EXACT) ---> [...]
    """
    for f in prog.functions.values():
        block_changed = True
        while block_changed:
            block_changed = _fuse_gelu_exact_block(f)
