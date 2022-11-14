#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@block_context_manager
def _prelu_to_lrelu_block(block):
    for op in list(block.operations):
        for b in op.blocks:
            _prelu_to_lrelu_block(b)
        if len(op.blocks) > 0:
            # This op can't be prelu.
            continue

        if op.op_type == "prelu":
            alpha_val = op.alpha.val
            common_leakage_factor = True
            for c in range(1, op.alpha.val.shape[0]):
                if alpha_val[c] != alpha_val[0]:
                    common_leakage_factor = False
                    break
            if common_leakage_factor:
                lrelu_out = mb.leaky_relu(
                    x=op.x, alpha=alpha_val[0], name=op.outputs[0].name, before_op=op)
                op.enclosing_block.replace_uses_of_var_after_op(
                    anchor_op=op, old_var=op.outputs[0], new_var=lrelu_out)
                block.remove_ops([op])


@register_pass(namespace="common")
class prelu_to_lrelu(AbstractGraphPass):
    """
    If PRelu has the same leakage factor across all channels, it can be converted to Leaky Relu.
    This pass makes that optimization.
    """
    def apply(self, prog):
        for f in prog.functions.values():
            _prelu_to_lrelu_block(f)