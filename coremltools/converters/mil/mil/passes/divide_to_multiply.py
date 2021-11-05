# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types as _types

def _divide_to_multiply_block(block):
    for op in list(block.operations):
        for b in op.blocks:
            _divide_to_multiply_block(b)
        if len(op.blocks) > 0:
            # This op can't be divide.
            continue

        # If real_div has integer input, the result is an integer (following TensorFlow spec).
        # Hence this pass needs disabled if the input is not float, since it translates y
        # to a floating point number. If x or y was originally an integer, and y becomes
        # a floating point number, then the original type
        # signature (with integer output) would not be preserved.
        if op.op_type == "real_div" and op.y.val is not None and _types.is_float(op.x.dtype):
            with block:
                new_y_val = np.array(1.0, dtype=op.y.val.dtype) / op.y.val
                if not np.isfinite(new_y_val).all():
                    continue

                x = mb.mul(
                    x=op.x, y=new_y_val, name="_inversed_" + op.name, before_op=op
                )
                op.enclosing_block.replace_uses_of_var_after_op(
                    anchor_op=op, old_var=op.outputs[0], new_var=x
                )
                block.remove_ops([op])


@register_pass(namespace="common")
class divide_to_multiply(AbstractGraphPass):
    """
    Convert divide into multiply if divisor is const.
    """
    def apply(self, prog):
        for f in prog.functions.values():
            _divide_to_multiply_block(f)
