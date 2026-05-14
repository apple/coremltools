#  Copyright (c) 2025, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="mil_backend")
class split_non_constant_pads(AbstractGraphPass):
    """
    Split ``pad`` ops that use non-constant modes (``reflect``, ``replicate``)
    and pad more than two dimensions, because the CoreML ML Program runtime
    rejects such ops with the error:
    "Padding for more than two dimensions only supports `constant` mode".

    Each split step pads at most two dimensions, which CoreML supports for all
    padding modes.

    .. code-block::

        Input:
        x(1, 3, 4, 4, 4) -> pad([0,0, 0,0, 2,2, 2,2, 2,2], mode="replicate") -> (1, 3, 8, 8, 8)

        Output:
        x(1, 3, 4, 4, 4) -> pad([0,0, 0,0, 2,2, 2,2, 0,0], mode="replicate") -> (1, 3, 8, 8, 4)
                          -> pad([0,0, 0,0, 0,0, 0,0, 2,2], mode="replicate") -> (1, 3, 8, 8, 8)
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._split_pads_block(f)

    @block_context_manager
    def _split_pads_block(self, block):
        for op in list(block.operations):
            for b in op.blocks:
                self._split_pads_block(b)

            if op.op_type != "pad":
                continue

            mode = op.inputs["mode"].val
            if mode == "constant":
                continue

            pad = op.inputs["pad"].val
            if pad is None:
                continue

            # Find dimensions with non-zero padding
            pad_pairs = pad.reshape(-1, 2)
            nonzero_dims = [
                i for i, (before, after) in enumerate(pad_pairs) if before != 0 or after != 0
            ]

            if len(nonzero_dims) <= 2:
                continue

            # Split into sequential pads, each covering at most 2 dimensions
            x = op.inputs["x"]
            constant_val = op.inputs["constant_val"].val
            num_chunks = (len(nonzero_dims) + 1) // 2
            result = x
            for chunk_idx, chunk_start in enumerate(range(0, len(nonzero_dims), 2)):
                chunk_dims = nonzero_dims[chunk_start : chunk_start + 2]
                chunk_pad = np.zeros_like(pad)
                for dim in chunk_dims:
                    chunk_pad[2 * dim] = pad_pairs[dim][0]
                    chunk_pad[2 * dim + 1] = pad_pairs[dim][1]

                is_last = chunk_idx == num_chunks - 1
                step_name = op.name if is_last else f"{op.name}_split_{chunk_idx}"
                result = mb.pad(
                    x=result,
                    pad=chunk_pad,
                    mode=mode,
                    constant_val=constant_val,
                    before_op=op,
                    name=step_name,
                )

            op.enclosing_block.replace_uses_of_var_after_op(
                anchor_op=op, old_var=op.outputs[0], new_var=result
            )
            op.enclosing_block.remove_ops([op])
