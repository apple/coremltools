#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.types.symbolic import any_symbolic


@register_pass(namespace="common")
class add_conv_transpose_output_shape(AbstractGraphPass):
    """
    ``conv_transpose`` input ``output_shape`` is an optional input.
    Since we can infer the output shape from type_inference, we add
    ``output_shape`` input whenever it is known to be constant at
    compile time. Ex:

    Given:
      %1: (1, 5, 39, fp32) = conv_transpose(...) # no output_shape input.

    Result:
      %2: (3, i32) = const(val=[1,5,39])
      %3: (1, 5, 39, fp32) = conv_transpose(..., output_shape=%2)
    """
    def apply(self, prog):
        for f in prog.functions.values():
            _handle_block(f)

def _match_pattern(op):
  return op.op_type == "conv_transpose" \
      and op.output_shape is None \
      and not any_symbolic(op.outputs[0].shape)

def _handle_block(block):
    for op in list(block.operations):
        for b in op.blocks:
            _handle_block(b)

        if not _match_pattern(op):
            continue

        # matched pattern
        with block:
            x = mb.conv_transpose(
                **op.inputs, output_shape=op.outputs[0].shape, \
                name=op.name+'_has_output_shape', before_op=op
            )
            op.enclosing_block.replace_uses_of_var_after_op(
                anchor_op=op, old_var=op.outputs[0], new_var=x
            )
            block.remove_ops([op])
