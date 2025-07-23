#  Copyright (c) 2025, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Tuple

import numpy as np
import torch

from coremltools.converters.mil.mil.builder import Builder as mb
from coremltools.converters.mil.mil.types import builtin_to_string
from coremltools.converters.mil.mil.var import Var

from .ops import _cast_to, _get_inputs, _get_kwinputs
from .torch_op_registry import register_torch_op
from .utils import NUM_TO_DTYPE_STRING, NUM_TO_NUMPY_DTYPE, NUM_TO_TORCH_DTYPE, dtype_to_32bit


@register_torch_op(torch_alias=["dim_order_ops::_to_dim_order_copy"])
def _to_dim_order_copy(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, min_expected=1)
        nargs = len(inputs)

        x = inputs[0]

        dtype = inputs[1] if nargs > 1 else None
        layout = inputs[2] if nargs > 2 else None
        device = inputs[3] if nargs > 3 else None
        pin_memory = inputs[4] if nargs > 4 else None
        non_blocking = inputs[5] if nargs > 5 else False
        dim_order = inputs[6] if nargs > 6 else None

        return x, dtype, layout, device, pin_memory, non_blocking, dim_order

    def _parse_keyword_args(
        context,
        node,
        dtype: Var,
        layout: Var,
        device: Var,
        pin_memory: Var,
        non_blocking: Var,
        dim_order: Var,
    ) -> Tuple[Var]:
        dtype = _get_kwinputs(context, node, "dtype", default=[dtype])[0]
        layout = _get_kwinputs(context, node, "layout", default=[layout])[0]
        device = _get_kwinputs(context, node, "device", default=[device])[0]
        pin_memory = _get_kwinputs(context, node, "pin_memory", default=[pin_memory])[0]
        non_blocking = _get_kwinputs(context, node, "non_blocking", default=[non_blocking])[0]
        dim_order = _get_kwinputs(context, node, "dim_order", default=[dim_order])[0]
        return dtype, layout, device, pin_memory, non_blocking, dim_order

    x, dtype, layout, device, pin_memory, non_blocking, dim_order = _parse_positional_args(
        context, node
    )
    dtype, layout, device, pin_memory, non_blocking, dim_order = _parse_keyword_args(
        context, node, dtype, layout, device, pin_memory, non_blocking, dim_order
    )

    if dim_order is not None:
        contiguous_dim_order = np.arange(x.rank)
        # For now, we simply error out non-contiguous dim order
        # TODO (rdar://150313325): Support general dim order
        if any(dim_order.val != contiguous_dim_order):
            raise NotImplementedError("Core ML does not support non-contiguous dim order")

    x_dtype_str = builtin_to_string(x.dtype)
    dtype_str = NUM_TO_DTYPE_STRING[dtype.val]
    if x_dtype_str == dtype_str:
        context.add(mb.identity(x=x, name=node.name))
    else:
        context.add(_cast_to(x, dtype_str, node.name))


@register_torch_op(torch_alias=["dim_order_ops::_empty_dim_order"])
def _empty_dim_order(context, node):
    "_empty_dim_order(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, int[]? dim_order=None) -> Tensor"

    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, min_expected=1)
        nargs = len(inputs)

        size = inputs[0]

        dtype = inputs[1] if nargs > 1 else None
        layout = inputs[2] if nargs > 2 else None
        device = inputs[3] if nargs > 3 else None
        pin_memory = inputs[4] if nargs > 4 else None
        dim_order = inputs[5] if nargs > 5 else None

        return size, dtype, layout, device, pin_memory, dim_order

    def _parse_keyword_args(
        context, node, dtype: Var, layout: Var, device: Var, pin_memory: Var, dim_order: Var
    ) -> Tuple[Var]:
        dtype = _get_kwinputs(context, node, "dtype", default=[dtype])[0]
        layout = _get_kwinputs(context, node, "layout", default=[layout])[0]
        device = _get_kwinputs(context, node, "device", default=[device])[0]
        pin_memory = _get_kwinputs(context, node, "pin_memory", default=[pin_memory])[0]
        dim_order = _get_kwinputs(context, node, "dim_order", default=[dim_order])[0]
        return dtype, layout, device, pin_memory, dim_order

    size, dtype, layout, device, pin_memory, dim_order = _parse_positional_args(context, node)
    dtype, layout, device, pin_memory, dim_order = _parse_keyword_args(
        context, node, dtype, layout, device, pin_memory, dim_order
    )

    if dim_order is not None:
        contiguous_dim_order = np.arange(size.shape[0])
        # For now, we simply error out non-contiguous dim order
        # TODO (rdar://150313325): Support general dim order
        if any(dim_order.val != contiguous_dim_order):
            raise NotImplementedError("Core ML does not support non-contiguous dim order")

    if dtype is None:
        dtype = torch.get_default_dtype()
        assert dtype in (torch.float32, torch.float64)
        dtype = 6
    else:
        if isinstance(dtype, Var):
            dtype = dtype.val

    if isinstance(size, list) or not size.can_be_folded_to_const():
        # the size is dynamic or this zeros op cannot be folded into const.
        size = mb.concat(values=size, axis=0) if isinstance(size, list) else size
        np_type = NUM_TO_NUMPY_DTYPE[dtype]
        zeros = mb.fill(shape=size, value=np_type(0), name=node.name)
    else:
        # the size is static and this zeros op can be folded into const.
        size = size.val
        torch_dtype = dtype_to_32bit(NUM_TO_TORCH_DTYPE[dtype])
        zeros_array = torch.zeros(tuple(size)).type(torch_dtype).numpy()
        zeros = mb.const(val=zeros_array, name=node.name)

    context.add(zeros)
