#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as _np
import torch as _torch

from coremltools import _logger as logger
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Var, types

from .ops import NUM_TO_TORCH_DTYPE, _create_linear_layer, _get_inputs, promote_input_dtypes
from .torch_op_registry import register_torch_op

TORCH_QTYPE_TO_NP_TYPE = {_torch.qint8: _np.int8, _torch.quint8: _np.uint8}

TORCH_QTYPE_TO_STR = {_torch.qint8: "int8", _torch.quint8: "uint8"}


def _quantize_general(
    context,
    node,
    input: Var,
    scale_var: Var,
    zero_point_var: Var,
    torch_dtype_var: Var,
    axis: int = None,
):
    scale = scale_var.val
    if scale is None:
        raise ValueError("quantization scale must be const at compile time")
    if len(scale.shape) > 0 and _np.prod(scale.shape) == 1:
        scale = scale.reshape(-1)[0]
        axis = None

    zero_point = zero_point_var.val
    if zero_point is None:
        raise ValueError("quantization zero point must be const at compile time")
    if len(zero_point.shape) > 0 and _np.prod(zero_point.shape) == 1:
        zero_point = zero_point.reshape(-1)[0]

    torch_dtype = NUM_TO_TORCH_DTYPE.get(torch_dtype_var.val)
    if torch_dtype is None:
        raise ValueError("quantization dtype must be const at compile time")
    dtype = TORCH_QTYPE_TO_STR.get(torch_dtype)
    # pytorch quantization dtype can be int32, which is not supported in MIL
    if dtype is None:
        raise ValueError("MIL quantization dtype must be int8 or uint8")

    # perf: all 0 zero point can be no zero point in MIL
    if zero_point is not None and _np.all(zero_point == 0):
        zero_point = None

    # make sure zero point dtype is consistent with quantization dtype,
    # since torch may provide int32 zero point
    if zero_point is not None:
        if dtype == "int8" and _np.all(-128 <= zero_point) and _np.all(zero_point < 128):
            zero_point = zero_point.astype(_np.int8)
        elif dtype == "uint8" and _np.all(0 <= zero_point) and _np.all(zero_point < 256):
            zero_point = zero_point.astype(_np.uint8)
        else:
            raise ValueError("cannot fit zero point into quantization dtype")

    result = mb.quantize(
        input=input,
        zero_point=zero_point,
        scale=scale,
        output_dtype=dtype,
        axis=axis,
    )
    context.add(result, node.name)
    context.quant_context.add_quantization_info(node.name, torch_dtype, scale, zero_point, axis)


@register_torch_op
def quantize_per_tensor(context, node):
    input, scale, zero_point, torch_dtype = _get_inputs(context, node, expected=[4])

    _quantize_general(context, node, input, scale, zero_point, torch_dtype)


@register_torch_op
def quantize_per_channel(context, node):
    input, scale, zero_point, axis, torch_dtype = _get_inputs(context, node, expected=[5])

    if axis.val is None:
        raise ValueError("quantization axis must be const at compile time")

    _quantize_general(context, node, input, scale, zero_point, torch_dtype, axis.val)


@register_torch_op
def dequantize(context, node):
    context.quant_context.get_dequantized_var(node.inputs[0], node.name)


def _dequantized_weight(qweight, name:str = None):
    """
    Given the first output (qweight) of torch.ops.quantized.conv2d/linear_unpack,
    this returns a dequantized version of the tensor to be added to the context.
    """
    if qweight.qscheme() == _torch.per_tensor_affine:
        quant_dtype_np = TORCH_QTYPE_TO_NP_TYPE[qweight.dtype]
        scale = _np.float32(qweight.q_scale())
        zero_point = quant_dtype_np(qweight.q_zero_point())
        quantized_weights = _torch.int_repr(qweight).numpy()
        # Axis doesn't matter for per-tensor quantization.
        axis = _np.int32(0)
        kwargs = {
            "quantized_data": quantized_weights,
            "zero_point": zero_point,
            "scale": scale,
            "axis": axis,
        }
        if name is not None:
            kwargs["name"] = name
        dequant_weights = mb.constexpr_affine_dequantize(**kwargs)
    # per_channel_affine_float_qparams is same as per_channel_affine except that it
    # expects both scale and zero point to be floating point values.
    elif qweight.qscheme() in {_torch.per_channel_affine, _torch.per_channel_affine_float_qparams}:
        quant_dtype_np = TORCH_QTYPE_TO_NP_TYPE[qweight.dtype]
        # TODO: How do we set the appropriate dtype here (fp16/fp32)?
        scale = qweight.q_per_channel_scales().numpy()
        zero_point = quant_dtype_np(qweight.q_per_channel_zero_points().numpy())
        if qweight.qscheme() == _torch.per_channel_affine:
            zero_point = quant_dtype_np(qweight.q_per_channel_zero_points().numpy())
        else:
            logger.warning(
                "Found per_channel_affine_float_qparams qscheme, which isn't directly "
                "supported by coremltools. Casting zero-points to quantized type loses some "
                "precision."
            )
            dtype_info = _np.iinfo(quant_dtype_np)
            val = _np.clip(
                _np.around(qweight.q_per_channel_zero_points().numpy()),
                dtype_info.min,
                dtype_info.max,
            )
            zero_point = quant_dtype_np(val)
        quantized_weights = _torch.int_repr(qweight).numpy()
        # Axis doesn't matter for per-tensor quantization.
        axis = _np.int32(0)
        kwargs = {
            "quantized_data": quantized_weights,
            "zero_point": zero_point,
            "scale": scale,
            "axis": axis,
        }
        if name is not None:
            kwargs["name"] = name
        dequant_weights = mb.constexpr_affine_dequantize(**kwargs)
    else:
        raise ValueError(f'Unsupported quant scheme "{qweight.qscheme()}"')
    return dequant_weights


def _process_conv(context, node, add_relu=False):
    # Node has 4 inputs:
    # 1. The input activations
    # 2. The packed weights/biases (need to get from context.torch_graph)
    # 3. output scale
    # 4. output zero-point

    # Unpack weights/bias & dequantize weights.
    packed_params = context.torch_graph.params[node.inputs[1]]
    qweight, bias = _torch.ops.quantized.conv2d_unpack(packed_params)
    dequant_weights = _dequantized_weight(qweight)
    context.add(dequant_weights)
    # Bias can be fed as-is.
    bias = bias.detach().numpy()

    # Convolution Parameters.
    x, x_dtype = context.quant_context.get_dequantized_var(node.inputs[0])
    raw_params = tuple(list(packed_params.__getstate__())[:-1])
    conv_attr_raw = raw_params[0][1][0].detach().numpy().astype(_np.int32)
    # Stride
    strides = conv_attr_raw[1:3]
    # Padding. torch.nn.quantized.Conv2d & its variants only support 'zeros' mode.
    pad = conv_attr_raw[3:5]
    assert conv_attr_raw[8] == 0
    if len(dequant_weights.shape) in (3, 4):
        # 1D and 2D: Need to explicitly state L-R, T-B pad
        pad = _np.repeat(pad, 2)
    else:
        raise ValueError("Invalid weight dimension. Must be 4 for 2D convolution.")
    # Dilation.
    dilations = conv_attr_raw[5:7]
    # Group.
    group = conv_attr_raw[9]
    kwargs = {
        "x": x,
        "weight": dequant_weights,
        "bias": bias,
        "strides": strides,
        "pad_type": "custom",
        "pad": pad,
        "dilations": dilations,
    }
    if group > 0:
        kwargs["groups"] = group

    res = mb.conv(**kwargs)
    if add_relu:
        res = mb.relu(x=res)
    context.add(res)

    out_scale = context[node.inputs[2]]
    out_zero_point = context[node.inputs[3]].val
    context.quant_context.get_quantized_per_tensor(
        res.name, x_dtype, out_scale, out_zero_point, node.name
    )


def _process_linear(context, node, add_relu=False):
    # Node has 4 inputs:
    # 1. The input activations
    # 2. The packed weights/biases (need to get from context.torch_graph)
    # 3. output scale
    # 4. output zero-point

    # Unpack PyTorch's packed params.
    if node.inputs[1] not in context:
        packed_params = context.torch_graph.params[node.inputs[1]]
        qweight, bias = _torch.ops.quantized.linear_unpack(packed_params)
        dequant_weights = _dequantized_weight(qweight)
        context.add(dequant_weights)
        bias = bias.detach().numpy()
    else:
        dequant_weights, bias = context[node.inputs[1]]

    x, x_dtype = context.quant_context.get_dequantized_var(node.inputs[0])

    x, dequant_weights = promote_input_dtypes([x, dequant_weights])
    res = _create_linear_layer(x, dequant_weights, bias)
    if add_relu:
        res = mb.relu(x=res)
    context.add(res)

    out_scale = context[node.inputs[2]]
    out_zero_point = context[node.inputs[3]].val
    if out_scale.val != 0 or out_zero_point != 0:
        context.quant_context.get_quantized_per_tensor(
            res.name, x_dtype, out_scale, out_zero_point, node.name
        )
    else:
        context.add(res, node.name)


def _process_binary(context, node, binary_op, add_relu=False):
    # Node has 4 inputs:
    # 1. LHS
    # 2. RHS
    # 3. output scale
    # 4. output zero-point

    assert len(node.inputs) == 4
    assert len(node.outputs) == 1

    lhs, lhs_dtype = context.quant_context.get_dequantized_var(node.inputs[0])
    rhs, rhs_dtype = context.quant_context.get_dequantized_var(node.inputs[1])
    assert lhs_dtype == rhs_dtype

    res = binary_op(x=lhs, y=rhs)
    if add_relu:
        res = mb.relu(x=res)
    context.add(res)

    out_scale = context[node.inputs[2]]
    out_zero_point = context[node.inputs[3]].val
    context.quant_context.get_quantized_per_tensor(
        res.name, lhs_dtype, out_scale, out_zero_point, node.name
    )


@register_torch_op(torch_alias=["quantized::matmul"])
def quantized_matmul(context, node):
    inputs = _get_inputs(context, node, expected=4)
    assert types.is_float(inputs[0].dtype)
    assert types.is_float(inputs[1].dtype)
    x, y = promote_input_dtypes([inputs[0], inputs[1]])
    assert (
        inputs[2].val == 0 and inputs[3].val == 0
    ), "non zero scale / zero-point not supported in quantized_matmul op."
    res = mb.matmul(x=x, y=y, name=node.name)
    context.add(res)


# Defines all the quantization-related nodes that are noOps
@register_torch_op(
    torch_alias=[
        "quantized::linear_prepack",
    ]
)
def quant_noop(context, node):
    logger.info("Setting pytorch op: {} to no-op.".format(node))
    inputs = _get_inputs(context, node)
    context.add(inputs, torch_name=node.name)


@register_torch_op(torch_alias=["quantized::linear"])
def quantized_linear(context, node):
    _process_linear(context, node)


@register_torch_op(torch_alias=["quantized::linear_relu"])
def quantized_linear_relu(context, node):
    _process_linear(context, node, add_relu=True)


@register_torch_op(torch_alias=["quantized::conv2d_relu"])
def quantized_conv2d_relu(context, node):
    _process_conv(context, node, add_relu=True)


@register_torch_op(torch_alias=["quantized::conv2d"])
def quantized_conv2d(context, node):
    _process_conv(context, node)


@register_torch_op(torch_alias=["quantized::add"])
def quantized_add(context, node):
    _process_binary(context, node, mb.add)


@register_torch_op(torch_alias=["quantized::add_relu"])
def quantized_add_relu(context, node):
    _process_binary(context, node, mb.add, add_relu=True)


@register_torch_op(torch_alias=["quantized::mul"])
def quantized_mul(context, node):
    _process_binary(context, node, mb.mul)


@register_torch_op(torch_alias=["quantized::embedding_byte"])
def quantized_embedding(context, node):
    packed_params = context.torch_graph.params[node.inputs[0]]
    qweight = _torch.ops.quantized.embedding_bag_unpack(packed_params)
    dequant_weights = _dequantized_weight(qweight)
    indices = context[node.inputs[1]]

    if len(node.inputs) >= 3:
        logger.warning(
            "Core ML quantized embedding (gather) layer does not support any "
            "inputs besides the weights and indices. Those given "
            "will be ignored."
        )

    if isinstance(indices, tuple):
        # Sometimes inputs will be a tuple, so handle that correctly.
        assert len(indices) == 1
        indices = indices[0]
    indices = mb.cast(x=indices, dtype="int32")

    #  Changing the axis from 0 is not an option in torch, so we don't expose it
    gather = mb.gather(x=dequant_weights, indices=indices, name=node.name)
    context.add(gather)
