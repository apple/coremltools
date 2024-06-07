#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import Block
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, Program, Var
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="common")
class expand_dynamic_linear(AbstractGraphPass):
    """
    Translate to ``linear`` when the operand is a descendant of const, since such an operand
    may be folded into const or fused into constexpr later by graph passes. In op translation,
    we prefer ``linear`` whenever possible because it requires const or constexpr ``weight`` and ``bias``.

    If such const folding or constexpr fusion did not happen, this pass would clean up
    the too-ambitious ``linear`` ops by replacing them with ``matmul`` ops.
    """

    def apply(self, prog: Program) -> None:
        for f in prog.functions.values():
            self._expand_dynamic_linear_block(f)

    @block_context_manager
    def _expand_dynamic_linear_block(self, block: Block) -> None:
        # use shallow copy to hide changes on block.operations during the loop,
        # since we do not need to deal with the newly expanded matmul + add ops
        for op in list(block.operations):
            for b in op.blocks:
                self._expand_dynamic_linear_block(b)

            if op.op_type == "linear":
                self._try_expand_dynamic_linear(op, block)

    @staticmethod
    def _is_operand_static(var: Var) -> bool:
        if var is None:
            return True

        op = var.op
        if op is None:
            return False

        op_type = op.op_type
        return op_type == "const" or op_type.startswith("constexpr_")

    def _try_expand_dynamic_linear(self, op: Operation, block: Block) -> None:
        assert op.op_type == "linear", "Should only apply to linear op"

        is_weight_static = self._is_operand_static(op.weight)
        is_bias_static = self._is_operand_static(op.bias)

        if is_weight_static:
            if is_bias_static:
                # static weight and bias, linear is good
                return
            else:
                # static weight with dynamic bias, so linear for weight matmul + add for bias add
                matmul = mb.linear(x=op.x, weight=op.weight, before_op=op)
                add = mb.add(x=matmul, y=op.bias, before_op=op, name=op.name)
                block.replace_uses_of_var_after_op(
                    anchor_op=op,
                    old_var=op.outputs[0],
                    new_var=add,
                )
                op.remove_from_block()
        else:
            # dynamic weight, have to expand to at least matmul
            result = mb.matmul(x=op.x, y=op.weight, transpose_y=True, before_op=op)
            # static bias, try skipping add if all zero
            if is_bias_static:
                force_replace = False
                # if no bias provided, default to 0, can skip
                # if bias provided, need to inspect its value
                if op.bias is not None:
                    bias_op = op.bias.op
                    bias_op_type = bias_op.op_type
                    if bias_op_type == "const":
                        is_nonzero_bias = np.any(op.bias.val != 0)
                    else:
                        if bias_op_type == "constexpr_affine_dequantize":
                            is_nonzero_bias = not bias_op.is_all_zeros()
                        # cowardly treat other types of compressed bias as if nonzero
                        else:
                            is_nonzero_bias = True
                        # For such a compressed all-zero bias, if we skip add, then
                        # the result (matmul output) would only descend from weight but not bias,
                        # i.e. need to force replacing descendant of bias
                        if not is_nonzero_bias:
                            force_replace = True
                    if is_nonzero_bias:
                        result = mb.add(x=result, y=op.bias, before_op=op, name=op.name)
                block.replace_uses_of_var_after_op(
                    anchor_op=op,
                    old_var=op.outputs[0],
                    new_var=result,
                    force_replace=force_replace,
                )
                op.remove_from_block()
            # dynamic bias, have to further expand to matmul + add
            else:
                result = mb.add(x=result, y=op.bias, before_op=op, name=op.name)
                block.replace_uses_of_var_after_op(
                    anchor_op=op,
                    old_var=op.outputs[0],
                    new_var=result,
                )
                op.remove_from_block()
