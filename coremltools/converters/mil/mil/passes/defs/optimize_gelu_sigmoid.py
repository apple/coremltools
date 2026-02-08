#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import (
    _check_var_scalar_value,
    block_context_manager,
)
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="common")
class fuse_gelu_sigmoid_approximation(AbstractGraphPass):
    """
    Detect the pattern that corresponds to the sigmoid approximation version of ``gelu``,
    and replace it with a single ``gelu`` layer with ``mode=SIGMOID_APPROXIMATION``.

    The sigmoid approximation of GELU is: ``y = x * sigmoid(1.702 * x)``

    This can be represented by one of the following patterns:

    .. code-block::

        Pattern 1: x -> mul(1.702) -> sigmoid -> mul(x) -> output
            [...] ----> mul (1.702) ---> sigmoid ---> mul ---> [...]
              |                                        ^
              |                                        |
              |----------------------------------------

        Pattern 2: x -> mul(x) -> sigmoid(1.702 * x) -> output (less common)

    All patterns are converted to:

    .. code-block::

        [...] ----> gelu (mode=SIGMOID_APPROXIMATION) ---> [...]
    """

    GELU_SIGMOID_CONST = 1.702

    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self._fuse_gelu_sigmoid_block(f)

    @block_context_manager
    def _fuse_gelu_sigmoid_block(self, block):
        fusion_occurred = False
        for op in list(block.operations):
            if op.enclosing_block is None:
                continue

            for b in op.blocks:
                nested_changed = True
                while nested_changed:
                    nested_changed = self._fuse_gelu_sigmoid_block(b)

            if len(op.blocks) > 0:
                continue

            if op.op_type == "mul":
                if self._try_match_and_transform_pattern1(op, block):
                    fusion_occurred = True
        return fusion_occurred

    def _try_match_and_transform_pattern1(self, mul_op, block):
        """
        Match pattern: x -> mul(1.702) -> sigmoid -> mul(x) -> output

        Where the final mul combines x with sigmoid(1.702 * x).
        """
        if mul_op.outputs[0] in block.outputs:
            return False

        mul_x = mul_op.x
        mul_y = mul_op.y

        sigmoid_var = None
        root_var = None

        if mul_x.op is not None and mul_x.op.op_type == "sigmoid":
            sigmoid_var = mul_x
            root_var = mul_y
        elif mul_y.op is not None and mul_y.op.op_type == "sigmoid":
            sigmoid_var = mul_y
            root_var = mul_x
        else:
            return False

        sigmoid_op = sigmoid_var.op

        if sigmoid_op.outputs[0] in block.outputs:
            return False

        sigmoid_input_op = sigmoid_op.x.op
        if sigmoid_input_op is None or sigmoid_input_op.op_type != "mul":
            return False

        scale_mul_op = sigmoid_input_op

        is_x_const = _check_var_scalar_value(scale_mul_op.x, self.GELU_SIGMOID_CONST, tol=0.01)
        is_y_const = _check_var_scalar_value(scale_mul_op.y, self.GELU_SIGMOID_CONST, tol=0.01)

        if not (is_x_const or is_y_const):
            return False

        scale_mul_input = scale_mul_op.y if is_x_const else scale_mul_op.x

        if scale_mul_input.name != root_var.name:
            return False

        if scale_mul_op.outputs[0] in block.outputs:
            return False

        return self._transform_to_gelu(
            block=block,
            root_var=root_var,
            ops_to_remove=[scale_mul_op, sigmoid_op, mul_op],
            output_op=mul_op,
        )

    def _transform_to_gelu(self, block, root_var, ops_to_remove, output_op):
        """Replace the matched pattern with a single gelu op."""
        for op in ops_to_remove[:-1]:
            for out in op.outputs:
                if out in block.outputs:
                    return False

        out_name = output_op.outputs[0].name
        gelu_out = mb.gelu(
            x=root_var,
            mode="SIGMOID_APPROXIMATION",
            name=out_name,
            before_op=ops_to_remove[0],
        )

        output_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=output_op,
            old_var=output_op.outputs[0],
            new_var=gelu_out,
        )

        block.remove_ops(ops_to_remove)
        return True

