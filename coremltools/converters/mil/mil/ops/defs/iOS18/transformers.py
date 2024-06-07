#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.operation import Operation
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs._utils import broadcast_shapes
from coremltools.converters.mil.mil.ops.defs.iOS18 import _IOS18_TARGET
from coremltools.converters.mil.mil.types.symbolic import any_symbolic, is_symbolic


@register_op(opset_version=_IOS18_TARGET)
class scaled_dot_product_attention(Operation):
    """
    Source: `PyTorch scaled dot product attention <https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>`_.
    Computes the scaled dot product attention on query, key, and value tensors, using an optional attention mask if passed.
    In PyTorch, this is equivalent to::

       attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
       attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))) + attn_mask, dim=-1)
       return attn_weight @ V

    Shape key:
       - ``B`` = Batch size
       - ``S`` = Source sequence length
       - ``L`` = Target sequence length
       - ``E`` = Query/Key embedding dimension
       - ``EV`` = Value embedding dimension

    Numerical values can differ due to floating point fusion/accumulation between backends.
    Note: We currently do not support the ``dropout_p`` and ``is_causal``.

    Mask can either be bool or float matching query, key, or value. For bool, it indicates
    whether the element should take part in the attention. Floats are added to the attention score.
    Mask shape must be broadcastable to ``[B, \*?, L, S]``.

    Parameters
    ----------
    query: tensor<[B, \*?, L, E], T> (Required)
    key: tensor<[B, \*?, S, E], T> (Required)
    value: tensor<[B, \*?, S, EV], T> (Required)
    attn_mask: tensor<[\*?, S], M> (Optional)

    Returns
    -------
    tensor<[B, \*?, L, EV], T>

    Attributes
    ----------
    T: fp16, fp32
    M: bool, fp16, fp32
    """

    input_spec = InputSpec(
        query=TensorInputType(type_domain="T"),
        key=TensorInputType(type_domain="T"),
        value=TensorInputType(type_domain="T"),
        attn_mask=TensorInputType(optional=True, type_domain="M"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "M": (types.bool, types.fp16, types.fp32),
    }

    def _validate_inputs(self):
        query_rank = self.query.rank
        key_rank = self.key.rank
        value_rank = self.value.rank
        if query_rank != key_rank or query_rank != value_rank:
            raise ValueError(
                f"query, key, value must have a same rank, got\n"
                f"* query rank = {query_rank}\n"
                f"* key rank = {key_rank}\n"
                f"* value rank = {value_rank}"
            )
        if query_rank < 3:
            raise ValueError(
                f"query, key, value must have at lease rank 3 "
                f"for batch, sequence length, embedding, got rank {query_rank}"
            )

        query_shape = self.query.shape
        key_shape = self.key.shape
        value_shape = self.value.shape
        B_query = query_shape[:-2]
        E_query = query_shape[-1]
        B_key = key_shape[:-2]
        S_key = key_shape[-2]
        E_key = key_shape[-1]
        B_value = value_shape[:-2]
        S_value = value_shape[-2]

        batch_dims = [B_query, B_key, B_value]
        batch_dims = [batch_dim for batch_dim in batch_dims if not any_symbolic(batch_dims)]
        if len(set(batch_dims)) > 1:
            raise ValueError(
                "query, key, value must have a same batch dimension, got\n"
                f"* query batch = {B_query}\n"
                f"* key batch = {B_key}\n"
                f"* value batch = {B_value}"
            )
        if not is_symbolic(E_query) and not is_symbolic(E_key) and E_query != E_key:
            raise ValueError(
                "query and key must have a same embedding dimension, got\n"
                f"* query embedding = {E_query}\n"
                f"* key embedding = {E_key}"
            )
        if not is_symbolic(S_key) and not is_symbolic(S_value) and S_key != S_value:
            raise ValueError(
                "key and value must have a same sequence length, got\n"
                f"* key sequence = {S_key}\n"
                f"* value sequence = {S_value}"
            )

        if self.attn_mask is not None:
            mask_shape = self.attn_mask.shape
            S_mask = mask_shape[-1]
            if not is_symbolic(S_mask) and not is_symbolic(S_key) and S_mask != S_key:
                raise ValueError(
                    "key and mask must have a same sequence length, got\n"
                    f"* key sequence = {S_key}\n"
                    f"* mask sequence = {S_mask}"
                )
            # If shapes are inconsistent, then `broadcast_shapes` would raise exception
            broadcast_shapes(query_shape[:-1], mask_shape[:-1])

    def type_inference(self):
        self._validate_inputs()

        shape = list(self.query.shape[:-1]) + [self.value.shape[-1]]
        return types.tensor(self.query.dtype, shape)

    def value_inference(self):
        query = self.query.val
        key = self.key.val
        value = self.value.val
        if query is None or key is None or value is None:
            return None

        float_mask = None
        if self.attn_mask is not None and self.attn_mask.val is not None:
            mask = self.attn_mask.val
            if mask.dtype == bool:
                float_mask = np.zeros(mask.shape)
                float_mask[np.where(np.logical_not(mask))] = -np.inf
            else:
                float_mask = mask

        similarity = np.matmul(query, key.swapaxes(-2, -1)) / np.sqrt(query.shape[-1])
        if float_mask is not None:
            similarity += float_mask
        attention_weight = self.numpy_softmax_last_dim(similarity)
        attention = np.matmul(attention_weight, value)
        return attention

    @staticmethod
    def numpy_softmax_last_dim(x: np.ndarray) -> np.ndarray:
        exps = np.exp(x - np.max(x, axis=-1)[..., None])
        softmax = exps / np.sum(exps, axis=-1)[..., None]
        return softmax
