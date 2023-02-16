#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Operation, get_new_symbol, types
from coremltools.converters.mil.mil.input_type import (DefaultInputs,
                                                       InputSpec,
                                                       TensorInputType)
from coremltools.converters.mil.mil.ops.defs._utils import \
    solve_slice_by_index_shape
from coremltools.converters.mil.mil.ops.registry import SSAOpRegistry
from coremltools.converters.mil.mil.types.symbolic import \
    is_compatible_symbolic_vector

register_op = SSAOpRegistry.register_op


# This file contains the Paddle dialect of SSA. Briefly, these ops are only
# understandable in the Paddle frontend and not acceptable in the standard op set.
# No backend would support any of the op here. These ops exist to facilitate
# frontend SSA passes, but must be replaced with standard ops during SSA
# passes.

# All paddle op must start with 'paddle_' prefix.

# paddle_upsample_nearest_neighbor is dealing with upsample layer which has flexible input shape,
# and recompute_scale_factor is set to True in the original paddle layer.
@register_op(namespace="paddle")
class paddle_upsample_nearest_neighbor(Operation):
    """
    Upsample the spatial dimensions (last two dimensions) of the input by
    scale factors using nearest-neighbor interpolation.
    It corresponds to `paddle.nn.functional.interpolate` function with `mode=nearest`,
    `recompute_scale_factor=True`, and input with flexible shape.
    source: https://paddle.org/docs/stable/_modules/paddle/nn/functional.html#interpolate

    Parameters
    ----------
    x: tensor<[b, C, H1, W1],T>  (Required)
        * Must be rank ``4``.
    output_height: i32
        * Output height for the height dimension.
    output_width: i32
        * Output width for the width dimension.

    Returns
    -------
    tensor<[b, C, H2, W2],T>
        * Tensor with same type as the input.
        * ``H2`` = output_height
        * ``W2`` = output_width

    Attributes
    ----------
    T: fp16, fp32
    """
    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        output_height=TensorInputType(type_domain=types.int32),
        output_width=TensorInputType(type_domain=types.int32),
    )
 
    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def type_inference(self):
        if self.x.rank != 4:
            raise ValueError(
                'input to the "paddle_upsample_nearest_neighbor" op must have rank 4'
            )
        ret_shape = list(self.x.shape)
        ret_shape[2] = get_new_symbol()
        ret_shape[3] = get_new_symbol()
        return types.tensor(self.x.dtype, ret_shape)

# paddle_upsample_bilinear is dealing with upsample layer which has flexible input shape,
# and recompute_scale_factor is set to True in the original paddle layer.
@register_op(namespace="paddle")
class paddle_upsample_bilinear(Operation):
    """
    Upsample the spatial dimensions (last two dimensions) of the input by
    scale factors using bilinear interpolation.
    It corresponds to `paddle.nn.functional.interpolate` function with `mode=bilinear`,
    `recompute_scale_factor=True`, and input with flexible shape.
    source: https://paddle.org/docs/stable/_modules/paddle/nn/functional.html#interpolate

    Parameters
    ----------
    x: tensor<[b, C, H1, W1], T>  (Required)
        * Must be rank ``4``.
    output_height: i32
        * Output height for the height dimension.
    output_width: i32
        * Output width for the width dimension.
    aligh_corners: const<bool>
        * The `aligh_corners` parameter for the original paddle op.

    Returns
    -------
    tensor<[b, C, H2, W2], T>
        * Tensor with same type as the input.
        * ``H2`` = output_height
        * ``W2`` = output_width

    Attributes
    ----------
    T: fp16, fp32
    """
    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        output_height=TensorInputType(type_domain=types.int32),
        output_width=TensorInputType(type_domain=types.int32),
        align_corners=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def default_inputs(self):
        return DefaultInputs(
            align_corners=True,
        )

    def type_inference(self):
        if self.x.rank != 4:
            raise ValueError(
                'input to the "paddle_upsample_bilinear" op must have rank 4'
            )
        ret_shape = list(self.x.shape)
        ret_shape[2] = get_new_symbol()
        ret_shape[3] = get_new_symbol()
        return types.tensor(self.x.dtype, ret_shape)

# paddle_tensor_assign is dealing with the tensor assignment operation
@register_op(namespace="paddle")
class paddle_tensor_assign(Operation):
    """
    Method for tensor value assignment via indexing and slicing.
    Suppose we have a tensor ``x``, this method achieves:
    ``x[begin[0]: end[0]: stride[0], begin[1]: end[1]: stride[1], ...] = value``

    Parameters
    ----------
    data: tensor<*?, T> (Required)
        * Input tensor
    updates: tensor<\*K, T> (Required)
        * Value tensor to be inserted
        * The shape of the updates tensor must match the slicing result of the input data.
    begin: tensor<[rank<x>], i32> (Required)
        * Starting index for the dimension of slicing.
    end: tensor<[rank(x)], i32> (Required)
        * Ending index for the dimension of slicing.
    stride: tensor<[rank(x)], i32> (Optional)
        * Default as all ``1``s.
        * Stride for the dimension of slicing.
    begin_mask: tensor<[rank(x)], bool> (Optional)
        * Default to all ``False``.
        * If ``begin_mask[i]==True``, neglect ``begin[i]``, and set ``begin[i]`` to ``0``.
    end_mask: tensor<[rank(x)], bool> (Optional)
        * Default to all ``False``.
        * If ``end_mask[i]==True``, neglect ``end[i]``, and set ``end[i]`` to ``x.shape[i]``.
    squeeze_mask: tensor<[rank(x)], bool> (Optional)
        * Default to all ``False``.
        * If ``squeeze_mask[i]==true``, neglect ``end[i]``, and do the pure index at ``begin[i]``.

    Returns
    -------
    tensor<*?, T>
        - Scalar or tensor.

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    input_spec = InputSpec(
        data=TensorInputType(type_domain="T"),
        updates=TensorInputType(type_domain="T"),
        begin=TensorInputType(const=True, type_domain=types.int32),
        end=TensorInputType(const=True, type_domain=types.int32),
        stride=TensorInputType(const=True, optional=True, type_domain=types.int32),
        begin_mask=TensorInputType(const=True, optional=True, type_domain=types.bool),
        end_mask=TensorInputType(const=True, optional=True, type_domain=types.bool),
        squeeze_mask=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )
    
    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
    }

    def default_inputs(self):
        return DefaultInputs(
            stride=None,
            begin_mask=None,
            end_mask=None,
            squeeze_mask=None,
        )

    def type_inference(self):
        # Verify the updates and the data slicing have the same shape
        begin = self.begin.val
        end = self.end.val
        data_rank = self.data.rank
        stride = self.stride.val if self.stride is not None else [1] * data_rank
        begin_mask = (
            self.begin_mask.val if self.begin_mask is not None else [False] * data_rank
        )
        end_mask = self.end_mask.val if self.end_mask is not None else [False] * data_rank
        squeeze_mask = (
            self.squeeze_mask.val if self.squeeze_mask is not None else [False] * data_rank
        )
        data_shape = self.data.shape
        expected_updates_shape = tuple(solve_slice_by_index_shape(data_shape, begin, end, stride, begin_mask, end_mask, squeeze_mask))
        if not is_compatible_symbolic_vector(expected_updates_shape, self.updates.shape):
            raise ValueError("The updates tensor should have shape {}. Got {}".format(expected_updates_shape, self.updates.shape))
        return self.data.sym_type
