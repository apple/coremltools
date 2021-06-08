#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil import Operation
from coremltools.converters.mil.mil.input_type import *
from coremltools.converters.mil.mil.ops.registry import SSAOpRegistry
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from coremltools.converters.mil.mil import get_new_symbol

register_op = SSAOpRegistry.register_op


# This file contains the Torch dialect of SSA. Briefly, these ops are only
# understandable in the Torch frontend and not acceptable in the standard op set.
# No backend would support any of the op here. These ops exist to facilitate
# frontend SSA passes, but must be replaced with standard ops during SSA
# passes.

# All torch op must start with 'torch_' prefix.

# torch_upsample_nearest_neighbor is dealing with upsample layer which has flexible input shape,
# and recompute_scale_factor is set to True in the original torch layer.
@register_op(doc_str="TODO", namespace="torch")
class torch_upsample_nearest_neighbor(Operation):
    """
    Upsample the spatial dimensions (last two dimensions) of the input by
    scale factors using nearest-neighbor interpolation.
    It corresponds to `torch.nn.functional.interpolate` function with `mode=nearest`,
    `recompute_scale_factor=True`, and input with flexible shape.
    source: https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#interpolate

    Parameters
    ----------
    x: tensor<[\*D, H1, W1],T>  (Required)
        * Must be at least rank ``3``.
    output_height: i32
        * Output height for the height dimension.
    output_width: i32
        * Output width for the width dimension.

    Returns
    -------
    tensor<[\*D, H2, W2],T>
        * Tensor with same type as the input.
        * ``H2`` = output_height
        * ``W2`` = output_width

    Attributes
    ----------
    T: fp32
    """
    input_spec = InputSpec(
        x=TensorInputType(),
        output_height=IntOrFloatInputType(),
        output_width=IntOrFloatInputType(),
    )

    def __init__(self, **kwargs):
        super(torch_upsample_nearest_neighbor, self).__init__(**kwargs)

    def type_inference(self):
        if self.x.rank < 3:
            raise ValueError(
                'input to the "torch_upsample_nearest_neighbor" op must have rank at least 3'
            )
        ret_shape = list(self.x.shape)
        ret_shape[-1] = get_new_symbol()
        ret_shape[-2] = get_new_symbol()
        return types.tensor(self.x.dtype, ret_shape)

# torch_upsample_bilinear is dealing with upsample layer which has flexible input shape,
# and recompute_scale_factor is set to True in the original torch layer.
@register_op(doc_str="TODO", namespace="torch")
class torch_upsample_bilinear(Operation):
    """
    Upsample the spatial dimensions (last two dimensions) of the input by
    scale factors using bilinear interpolation.
    It corresponds to `torch.nn.functional.interpolate` function with `mode=bilinear`,
    `recompute_scale_factor=True`, and input with flexible shape.
    source: https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#interpolate

    Parameters
    ----------
    x: tensor<[\*D, H1, W1],T>  (Required)
        * Must be rank ``3``.
    output_height: i32
        * Output height for the height dimension.
    output_width: i32
        * Output width for the width dimension.
    aligh_corners: const<bool>
        * The `aligh_corners` parameter for the original torch op.

    Returns
    -------
    tensor<[\*D, H2, W2],T>
        * Tensor with same type as the input.
        * ``H2`` = output_height
        * ``W2`` = output_width

    Attributes
    ----------
    T: fp32
    """
    input_spec = InputSpec(
        x=TensorInputType(),
        output_height=IntOrFloatInputType(),
        output_width=IntOrFloatInputType(),
        align_corners=BoolInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            align_corners=True,
            )

    def __init__(self, **kwargs):
        super(torch_upsample_bilinear, self).__init__(**kwargs)

    def type_inference(self):
        if self.x.rank < 3:
            raise ValueError(
                'input to the "torch_upsample_bilinear" op must have rank at least 3'
            )
        ret_shape = list(self.x.shape)
        ret_shape[-1] = get_new_symbol()
        ret_shape[-2] = get_new_symbol()
        return types.tensor(self.x.dtype, ret_shape)
