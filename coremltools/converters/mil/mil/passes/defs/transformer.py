#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import ClassVar, List, Tuple

import numpy as np

from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as target
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types

from coremltools import _logger as logger


@register_pass(namespace="common")
class scaled_dot_product_attention_sliced_q(AbstractGraphPass):
    """
    Replace the ios18.scaled_dot_product_attention operation with a memory efficient
    implementation of attention calculation based on slicing Q. The benefits are clearly
    visible for higher Q sequence lengths, though.

    Graph pass options:
      - min_seq_length: int
        Only operations working with Q of sequence length greater or equal to this value will be transformed.
      - seq_length_divider: int
        Defines the size of the chunks of Q being processed in SDPA (chunk_size = seq_length / seq_length_divider)
    """

    _DEFAULT_MIN_SEQ_LENGTH: ClassVar[int] = 1280
    _DEFAULT_SEQ_LENGTH_DIVIDER: ClassVar[int] = 16

    _min_seq_length: int
    _seq_length_divider: int

    def __init__(self):
        super().__init__()
        self._min_seq_length = self._DEFAULT_MIN_SEQ_LENGTH
        self._seq_length_divider = self._DEFAULT_SEQ_LENGTH_DIVIDER

    @property
    def min_seq_length(self) -> int:
        return self._min_seq_length

    @min_seq_length.setter
    def min_seq_length(self, length: int) -> None:
        if not isinstance(length, int):
            raise ValueError("pass option min_seq_length must be an int")
        if length < 0:
            raise ValueError("pass option min_seq_length must be >= 0")
        self._min_seq_length = length

    @property
    def seq_length_divider(self) -> int:
        return self._seq_length_divider

    @seq_length_divider.setter
    def seq_length_divider(self, divider: int) -> None:
        if not isinstance(divider, int):
            raise ValueError("pass option seq_length_divider must be an int")
        if divider < 1:
            raise ValueError("pass option seq_length_divider must be >= 1")
        self._seq_length_divider = divider

    def apply(self, prog):
        for f in prog.functions.values():
            if f.opset_version < target.iOS18:
                logger.debug(f"ignoring block '{f.name}', target {f.opset_version} (required min iOS18)")
                return

            for op in list(f.operations):
                if op.op_type == "scaled_dot_product_attention":
                    self._replace_scaled_dot_product_attention(op)

    @staticmethod
    def _get_input_vars(op):
        mandatory_params = ["query", "key", "value"]
        inputs = {}
        for param in mandatory_params:
            inputs[param] = op.inputs.get(param)
            if inputs[param] is None:
                raise ValueError(f"operation 'scaled_dot_product_attention': mandatory input '{param}' not present")
        return tuple([inputs[param] for param in mandatory_params]) + (op.inputs.get("attn_mask"),)

    @staticmethod
    def _split_to_chunks(seq_length: int, count: int) -> List[Tuple[int, int]]:
        chunk_size = max(seq_length // count, 1)
        remainder = seq_length % count

        result = []
        chunk_start = 0
        for i in range(count):
            if chunk_start >= seq_length:
                break
            chunk_end = chunk_start + chunk_size + (1 if i < remainder else 0)
            result.append((chunk_start, chunk_end))
            chunk_start = chunk_end

        return result

    def _replace_scaled_dot_product_attention(self, op):
        q, k, v, mask = self._get_input_vars(op)

        q_size = len(q.shape)
        q_seq_length = q.shape[-2]
        if q_seq_length < self._min_seq_length:
            logger.debug(
                f"skipping SDPA op, Q seq_length is {q_seq_length} (minimum seq length needed: {self._min_seq_length}"
            )
            return

        dims = q.shape[-1]
        normalize_factor = float(dims) ** -0.5

        q_dtype = types.nptype_from_builtin(type(q.dtype()))

        chunks = self._split_to_chunks(q_seq_length, self._seq_length_divider)

        concat_out = None
        with op.enclosing_block:
            if mask is not None:
                if mask.dtype == types.bool:
                    cond_out = mb.logical_not(x=mask, before_op=op)
                    mask_zeros = mb.const(val=np.zeros(mask.shape, dtype=q_dtype), before_op=op)
                    mask_float = mb.select(cond=cond_out, a=q_dtype(-np.inf), b=mask_zeros, before_op=op)
                else:
                    mask_float = mask

            for chunk_start, chunk_end in chunks:
                # Get a chunk of Q.
                slice_begin = [0] * (q_size - 2) + [chunk_start, 0]
                slice_end = list(q.shape[:-2] + (chunk_end, dims))
                slice_end_mask = tuple([True] * (q_size - 2) + [False, True])
                slice_out = mb.slice_by_index(
                    x=q,
                    begin=slice_begin,
                    end=slice_end,
                    end_mask=slice_end_mask,
                    before_op=op,
                )

                # Calculate chunk of Q x KT
                matmul_out = mb.matmul(x=slice_out, y=k, transpose_x=False, transpose_y=True, before_op=op)
                mul_out = mb.mul(x=matmul_out, y=np.array(normalize_factor, dtype=q_dtype), before_op=op)

                # Apply the attention mask.
                if mask is not None:
                    if mask.shape[-2] == 1:
                        mul_out = mb.add(x=mul_out, y=mask_float, before_op=op)
                    else:
                        mask_out = mb.slice_by_index(
                            x=mask_float,
                            begin=[chunk_start, 0],
                            end=[chunk_end, mask.shape[-1]],
                            end_mask=[False, True],
                            before_op=op,
                        )
                        mul_out = mb.add(x=mul_out, y=mask_out, before_op=op)

                # Calculate softmax of the product.
                softmax_out = mb.softmax(x=mul_out, axis=-1, before_op=op)

                # Calculate the chunk of attention.
                matmul_v_out = mb.matmul(
                    x=softmax_out,
                    y=v,
                    transpose_x=False,
                    transpose_y=False,
                    before_op=op,
                )

                # Add the chunk of attention to the result value.
                concat_values = [concat_out] if concat_out is not None else []
                concat_out = mb.concat(values=concat_values + [matmul_v_out], axis=-2, interleave=False, before_op=op)

            # Remove the original SDPA operation.
            op.enclosing_block.replace_uses_of_var_after_op(
                anchor_op=op,
                old_var=op.outputs[0],
                new_var=concat_out,
            )
            op.enclosing_block.remove_ops([op])
