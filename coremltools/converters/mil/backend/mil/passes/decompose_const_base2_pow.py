#  Copyright (c) 2025, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


def _is_const_base2_pow(op):
    # match pow(x, y) where x is a floating-point constant equal to 2
    if op.op_type != "pow":
        return False
    if not types.is_float(op.y.dtype):
        return False
    return op.x.val is not None and np.all(op.x.val == 2)


def _try_to_transform(op, block):
    # 2 ** y == exp(y * log(2))
    log2 = np.log(2.0).astype(types.nptype_from_builtin(op.y.dtype))
    scaled = mb.mul(x=op.y, y=log2, before_op=op)
    new_var = mb.exp(x=scaled, name=op.outputs[0].name, before_op=op)
    op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=op, old_var=op.outputs[0], new_var=new_var
    )
    block.remove_ops([op])


@block_context_manager
def _decompose_const_base2_pow(block):
    for op in list(block.operations):
        if op.enclosing_block is None:
            continue
        for b in op.blocks:
            _decompose_const_base2_pow(b)
        if len(op.blocks) > 0:
            continue
        if _is_const_base2_pow(op):
            _try_to_transform(op, block)


@register_pass(namespace="mil_backend")
class decompose_const_base2_pow(AbstractGraphPass):
    """
    Rewrite pow(2, y) into exp(y * log(2)).

    Works around a Core ML runtime bug where pow with a constant base of 2
    returns y ** 2 instead of 2 ** y. Floating-point pow only.

    Given:
        %out = pow(x=2, y=%y)

    Result:
        %scaled = mul(x=%y, y=0.6931471805599453)
        %out = exp(x=%scaled)
    """
    def apply(self, prog):
        for f in prog.functions.values():
            _decompose_const_base2_pow(f)
