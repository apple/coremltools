#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import numpy as np

from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="common")
class noop_elimination(AbstractGraphPass):
    """
    Remove ops that have no effect.

    .. code-block::

        Given:
            %1 (1, 96, 128, 64, fp32) = ...
            %2 (1, 96, 128, 64, fp32) = reshape(%1)
            ...
            %3 (1, 96, 128, 64, fp32) = add(%2, constant)
            ...

        Result:
            %1 (1, 96, 128, 64, fp32) = ...
            %3 (1, 96, 128, 64, fp32) = add(%1, constant)
        ...
    """

    _SUPPORTED_OPS = {
        "identity",
        "add",
        "mul",
        "floor_div",
        "pow",
        "real_div",
        "sub",
        "reshape",
        "split",
        "slice_by_index",
        "slice_by_size",
        "pad",
        "tile",
        "transpose",
        "upsample_nearest_neighbor",
        "upsample_bilinear",
        "resize_bilinear",
        "crop",
        "linear_activation",
    }

    def apply(self, prog):
        for f in prog.functions.values():
            self._noop_elimination_block_wrapper(f)

    @staticmethod
    def _match_pattern(op):
        def remove_identity(op):
            if op.enclosing_block.try_replace_uses_of_var_after_op(
                anchor_op=op,
                old_var=op.outputs[0],
                new_var=op.x,
            ):
                op.enclosing_block.remove_ops([op])
                return True
            return False

        def _remove_elementwise_binary(op, x, y):
            # We remove the ops that has op.x == x or op.y == y
            def has_all_elements_equal_to(var, value):
                if value is None:
                    return False

                if var.val is not None:
                    return np.all(var.val == value)
                elif var.op is not None and var.op.op_type == "fill":
                    fill_value = var.op.value.val
                    return fill_value is not None and (fill_value == value)
                else:
                    return False

            if has_all_elements_equal_to(op.x, x):
                input_var = op.y
            elif has_all_elements_equal_to(op.y, y):
                input_var = op.x
            else:
                return False

            input_shape = input_var.sym_type
            output_shape = op.outputs[0].sym_type

            # We might be using elementwise as broadcasting
            if input_shape != output_shape:
                return False

            if op.enclosing_block.try_replace_uses_of_var_after_op(
                anchor_op=op,
                old_var=op.outputs[0],
                new_var=input_var,
            ):
                op.enclosing_block.remove_ops([op])
                return True
            return False

        def remove_elementwise(op):
            if op.op_type in {"add"}:
                return _remove_elementwise_binary(op, 0, 0)
            elif op.op_type in {"mul"}:
                return _remove_elementwise_binary(op, 1, 1)
            elif op.op_type in {"floor_div", "pow", "real_div"}:
                return _remove_elementwise_binary(op, None, 1)
            elif op.op_type in {"sub"}:
                return _remove_elementwise_binary(op, None, 0)
            else:
                return False

        def remove_slice_by_index(op):
            input_shape = op.x.sym_type
            output_shape = op.outputs[0].sym_type

            if input_shape != output_shape:
                return False

            if op.stride is not None and op.stride.val is not None:
                stride = op.stride.val.flatten().tolist()
                if any([x < 0 for x in stride]):
                    return False

            if op.enclosing_block.try_replace_uses_of_var_after_op(
                anchor_op=op,
                old_var=op.outputs[0],
                new_var=op.x,
            ):
                op.enclosing_block.remove_ops([op])
                return True
            return False

        def remove_same_shape(op):
            input_shape = op.x.sym_type
            output_shape = op.outputs[0].sym_type

            if input_shape != output_shape:
                return False

            if op.enclosing_block.try_replace_uses_of_var_after_op(
                anchor_op=op,
                old_var=op.outputs[0],
                new_var=op.x,
            ):
                op.enclosing_block.remove_ops([op])
                return True
            return False

        def remove_linear(op):
            if op.alpha.val != 1 or op.beta.val != 0:
                return False

            if op.enclosing_block.try_replace_uses_of_var_after_op(
                anchor_op=op,
                old_var=op.outputs[0],
                new_var=op.x,
            ):
                op.enclosing_block.remove_ops([op])
                return True
            return False

        def remove_transpose(op):
            perm = np.array([p if p >= 0 else p + len(op.perm.val) for p in op.perm.val])
            sorted_perm = np.sort(perm)
            if (perm != sorted_perm).any():
                return False

            if op.enclosing_block.try_replace_uses_of_var_after_op(
                anchor_op=op,
                old_var=op.outputs[0],
                new_var=op.x,
            ):
                op.enclosing_block.remove_ops([op])
                return True
            return False

        op_to_removal_fn = {
            "identity": remove_identity,
            "add": remove_elementwise,
            "mul": remove_elementwise,
            "floor_div": remove_elementwise,
            "pow": remove_elementwise,
            "real_div": remove_elementwise,
            "sub": remove_elementwise,
            "reshape": remove_same_shape,
            "split": remove_same_shape,
            "slice_by_index": remove_slice_by_index,
            "slice_by_size": remove_same_shape,
            "pad": remove_same_shape,
            "tile": remove_same_shape,
            "transpose": remove_transpose,
            "upsample_nearest_neighbor": remove_same_shape,
            "upsample_bilinear": remove_same_shape,
            "resize_bilinear": remove_same_shape,
            "crop": remove_same_shape,
            "linear_activation": remove_linear,
        }
        # abort if op output is a block output
        if op.outputs[0] in op.enclosing_block.outputs:
            return None

        if op.op_type in noop_elimination._SUPPORTED_OPS:

            if len(op.outputs) != 1:
                return None
            return op_to_removal_fn[op.op_type]

        return None

    @block_context_manager
    def _noop_elimination_block_wrapper(self, block):
        def _noop_elimination_block(block):
            status = False
            for op in list(block.operations):
                if op.enclosing_block is None:
                    continue

                for b in op.blocks:
                    block_changed = True
                    while block_changed:
                        block_changed = _noop_elimination_block(b)
                if len(op.blocks) > 0:
                    continue

                remove_fn = noop_elimination._match_pattern(op)
                if remove_fn is not None and remove_fn(op):
                    status = True
            return status

        block_changed = True
        while block_changed:
            block_changed = _noop_elimination_block(block)
