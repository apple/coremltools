#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clausefrom coremltools.converters.mil.mil import types

import numpy as np

from coremltools.converters.mil.mil.input_type import(
    InputSpec,
    TensorInputType,
    ScalarOrTensorInputType,
)
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as target
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.operation import Operation
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS16 import _IOS16_TARGET

@register_op(opset_version=_IOS16_TARGET)
class pixel_unshuffle(Operation):
    """
    Rearrange elements in a tensor from spatial dimensions into depth (channel).
    It is basically the inverse operation of pixel_shuffle.
    Equivalent to PyTorch's ``PixelUnshuffle``.

    Parameters
    ----------
    x: tensor<[n, C, H / f , W / f], T> (Required)
        * Input tensor of rank ``4``.
    downscale_factor: const<i32>
        * Factor to decrease spatial resolution by.

    Returns
    -------
    tensor<[n, C * f^2, H, W], T>
        * Where ``f`` is the downscale factor.

    Attributes
    ----------
    T: fp32

    References
    ----------
    `torch.nn.PixelUnshuffle <https://pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html>`_
    """

    input_spec = InputSpec(
        x=TensorInputType(), downscale_factor=ScalarOrTensorInputType(const=True, type_domain=(np.uint32,)),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        n, c, h, w = self.x.shape
        f = self.downscale_factor.val
        ret_shape = (n, c * f * f, h / f, w / f)
        return types.tensor(x_type, ret_shape)
