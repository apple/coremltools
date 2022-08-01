#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clausefrom coremltools.converters.mil.mil import types

from coremltools.converters.mil.mil.input_type import (
    InputSpec,
    TensorInputType, 
    ScalarOrTensorInputType,
    BoolInputType,
    DefaultInputs,
)
from coremltools.converters.mil.mil.operation import (
    Operation,
    precondition,
    VALUE
)
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS15.tensor_operation import topk as _topk_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS16 import _IOS16_TARGET


@register_op(opset_version=_IOS16_TARGET)
class topk(_topk_iOS15):
    """
    An iOS16 version of topk

    Additional Parameters
    ----------
    * sort: const<bool> (Optional)
        * Default to ``True``
        * If true, top-k elements are themselves sorted. 
          Otherwise, no particular ordering is guaranteed.
    * return_indices: const<bool> (Optional)
        # Default to ``True``
        # If true, returns both values and indices. Otherwise, returns only the top-k values.

    Returns
    -------
    tensor<\*?, T>
        * Values of top/bottom ``k`` elements.

    tensor<\*?, int32>
        * Only returned when ``return_indices = True``
        * Indices of the top/bottom ``k`` elements along axis.

    Attributes
    ----------
    T: fp32, int32
    """

    input_spec = _topk_iOS15.input_spec + InputSpec(
        sort=BoolInputType(const=True, optional=True),
        return_indices=BoolInputType(const=True, optional=True),
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
