#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clausefrom

import numpy as np

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import (DefaultInputs,
                                                       InputSpec,
                                                       TensorInputType)
from coremltools.converters.mil.mil.operation import (VALUE, Operation,
                                                      precondition)
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS15.tensor_operation import \
    topk as _topk_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS16 import _IOS16_TARGET


@register_op(opset_version=_IOS16_TARGET)
class fill_like(Operation):
    """
    Returns a tensor with the same shape as the input tensor filled with a constant value.

    Parameters
    ----------
    ref_tensor: tensor<\\*?, T> (Required)
        * Input tensor.
    value: const<U> (Optional)
        * Default is ``0.0``.
        * Constant value to fill in.

    Returns
    -------
    tensor<\\*?, T>
        * Tensor with shape determined by the input tensor.

    Attributes
    ----------
    T: fp16, fp32, int32, bool
    U: fp16, fp32, int32, bool
    """

    input_spec = InputSpec(
        ref_tensor=TensorInputType(type_domain="T"),
        value=TensorInputType(const=True, optional=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
        "U": (types.fp16, types.fp32, types.int32, types.bool),
    }

    def default_inputs(self):
        return DefaultInputs(
            value=0.
        )

    def type_inference(self):
        return types.tensor(self.value.dtype, self.ref_tensor.shape)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.full(shape=self.ref_tensor.shape, fill_value=self.value.val)

@register_op(opset_version=_IOS16_TARGET)
class topk(_topk_iOS15):
    """
    A version of ``topk`` for iOS 16+. This section documents the differences. The following are additional parameters for the iOS 16+ version. For the
    rest of the documentation, see `the iOS 15 version of topk <#coremltools.converters.mil.mil.ops.defs.iOS15.tensor_operation.topk>`_.

    Parameters
    ----------
    sort: const<bool> (Optional)
        * Defaults to ``True``.
        * If ``True``, ``top-k`` elements are themselves sorted.
          Otherwise, no particular ordering is guaranteed.
    return_indices: const<bool> (Optional)
        * Defaults to ``True``.
        * If ``True``, returns both values and indices. Otherwise, returns only the ``top-k`` values.

    Returns
    -------
    tensor<\\*?, T>
        * Values of top/bottom ``k`` elements.

    tensor<\\*?, int32>
        * Only returned when ``return_indices = True``
        * Indices of the top/bottom ``k`` elements along axis.

    Attributes
    ----------
    T: fp32, int32
    """

    input_spec = _topk_iOS15.input_spec + InputSpec(
        sort=TensorInputType(const=True, optional=True, type_domain=types.bool),
        return_indices=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    def default_inputs(self):
        return super().default_inputs() + DefaultInputs(sort=True, return_indices=True)

    def type_inference(self):
        value_type, indices_type = super().type_inference()
        if not self.return_indices.val:
            return value_type
        return value_type, indices_type

    @precondition(allow=VALUE)
    def value_inference(self):
        values, indices = super().value_inference()
        if not self.return_indices.val:
            return values
        return values, indices
