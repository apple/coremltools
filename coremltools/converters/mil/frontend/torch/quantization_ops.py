#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as _np
import torch as _torch
from packaging.version import Version

from coremltools import _logger as logger
from coremltools._deps import _HAS_TORCHAO, MSG_TORCHAO_NOT_FOUND
from coremltools.converters.mil.frontend import _utils
from coremltools.converters.mil.frontend.torch.ops import NUM_TO_NUMPY_DTYPE
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Var, types

from .ops import _create_linear_layer, _get_inputs, promote_input_dtypes
from .torch_op_registry import register_torch_op
from .utils import (
    NUM_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_NUM,
    TORCH_EXPORT_BASED_FRONTENDS,
    TORCH_QTYPE_TO_NP_TYPE,
    TORCH_QTYPE_TO_STR,
    TYPE_TO_DTYPE_STRING,
    TorchFrontend,
)

if _HAS_TORCHAO:
    from torchao.quantization import quant_primitives as torchao_quant


def _quantize_general(
    context,
    node,
    input: Var,
    scale_var: Var,
    zero_point_var: Var,
    torch_dtype_var: Var,
    axis: int = None,
):
    if input.op is not None and input.op.op_type.startswith("constexpr_"):
        # Skip already quantized weight, which was done by using compression metadata.
        context.add(input, node.name)
        return

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
    if context.frontend == TorchFrontend.TORCHSCRIPT:
        context.quant_context.add_quantization_info(node.name, torch_dtype, scale, zero_point, axis)


@register_torch_op(
    torch_alias=[
        "quantized_decomposed::quantize_per_tensor",
        "quantized_decomposed.quantize_per_tensor",
    ]
)
def quantize_per_tensor(context, node):
    inputs = _get_inputs(
        context,
        node,
        expected={
            TorchFrontend.TORCHSCRIPT: 4,
            TorchFrontend.TORCHEXPORT: 6,
            TorchFrontend.EXECUTORCH: 6,
        },
    )
    if context.frontend == TorchFrontend.TORCHSCRIPT:
        input, scale, zero_point, torch_dtype = inputs
    elif context.frontend in TORCH_EXPORT_BASED_FRONTENDS:
        input, scale, zero_point, qmin, qmax, torch_dtype = inputs
        if qmax.val - qmin.val <= 16:
            logger.warning(
                f"Core ML does not support 4-bit activation, so {torch_dtype.val} is used instead"
            )
    else:
        raise ValueError(f"Invalid PyTorch frontend {context.frontend}")

    _quantize_general(context, node, input, scale, zero_point, torch_dtype)


@register_torch_op
def quantize_per_channel(context, node):
    input, scale, zero_point, axis, torch_dtype = _get_inputs(context, node, expected=[5])

    if axis.val is None:
        raise ValueError("quantization axis must be const at compile time")

    _quantize_general(context, node, input, scale, zero_point, torch_dtype, axis.val)


@register_torch_op(
    torch_alias=[
        "quantized_decomposed::choose_qparams_per_token_asymmetric",
        "quantized_decomposed.choose_qparams_per_token_asymmetric",
    ]
)
def choose_qparams_per_token_asymmetric(context, node):
    """PyTorch uses this op to calculate scale and zero_point on-the-fly for input data."""
    raise NotImplementedError("Dynamic activation quantization is not supported in Core ML.")


def _dequantize_general(
    context,
    node,
    input: Var,
    scale: Var,
    zero_point: Var,
    axis: Var,
    qmin: Var,
    qmax: Var,
) -> None:
    # torch may use different dtype for input and zero_point,
    # but Core ML requires input and zero_point to have a same dtype,
    # so cast zero_point dtype to input dtype
    if input.dtype != zero_point.dtype:
        zero_point = mb.cast(x=zero_point, dtype=TYPE_TO_DTYPE_STRING[input.dtype])
    # Not sure why torch may quantize a scalar... does not make sense,
    # since the floating point scale is as big as the original floating point input data scalar
    if input.rank == 0:
        # For const input, translate to the const floating point scalar output
        if input.val is not None:
            output_value = scale.val * (input.val - zero_point.val)
            output = mb.const(val=output_value, name=node.name)
        # For variable input, we have no choice but to expand and squeeze,
        # since Core ML dequantize op requires tensor input
        else:
            expanded_input = mb.expand_dims(x=input, axes=(0,))
            dequantize_output = mb.dequantize(
                input=expanded_input,
                zero_point=zero_point,
                scale=scale,
                axis=axis,
            )
            output = mb.squeeze(x=dequantize_output, name=node.name)
    else:
        # activation quantization
        if input.val is None:
            if qmax.val - qmin.val <= 16:
                logger.warning(
                    f"Core ML does not support 4-bit activation, so {input.dtype} is used instead"
                )
            output = mb.dequantize(
                input=input,
                zero_point=zero_point,
                scale=scale,
                axis=axis,
                name=node.name,
            )
        # weight compression
        else:
            if qmax.val - qmin.val <= 8:
                logger.warning(
                    "Core ML does not support less than 4-bit compression, so 4 bit is used instead"
                )
            input_val = input.val
            zero_point_val = zero_point.val
            if zero_point_val.dtype != input_val.dtype:
                zero_point_val = zero_point_val.astype(input_val.dtype)
            axis_val = None if axis is None else axis.val
            output = _utils._construct_constexpr_dequant_op(
                input_val, zero_point_val, scale.val, axis=axis_val, name=node.name
            )
    context.add(output, node.name)


@register_torch_op(
    torch_alias=[
        "quantized_decomposed::dequantize_per_tensor",
        "quantized_decomposed.dequantize_per_tensor",
        "quantized_decomposed::dequantize_per_channel",
        "quantized_decomposed.dequantize_per_channel",
    ]
)
def dequantize(context, node):
    if context.frontend == TorchFrontend.TORCHSCRIPT:
        context.quant_context.get_dequantized_var(node.inputs[0], node.name)
    elif context.frontend in TORCH_EXPORT_BASED_FRONTENDS:
        inputs = _get_inputs(
            context, node, min_expected={TorchFrontend.TORCHEXPORT: 6, TorchFrontend.EXECUTORCH: 6}
        )
        num_inputs = len(inputs)
        if num_inputs == 6:
            input, scale, zero_point, qmin, qmax, _ = inputs
            axis = None
        elif num_inputs == 7:
            input, scale, zero_point, axis, qmin, qmax, _ = inputs
        else:
            raise ValueError(f"dequantize should have 6 or 7 inputs, but got {num_inputs}")
        _dequantize_general(context, node, input, scale, zero_point, axis, qmin, qmax)
    else:
        raise ValueError(f"Invalid PyTorch frontend {context.frontend}")


def _dequantized_weight(qweight, name: str = None):
    """
    Given the first output (qweight) of torch.ops.quantized.conv2d/linear_unpack,
    this returns a dequantized version of the tensor to be added to the context.
    """
    if qweight.qscheme() == _torch.per_tensor_affine:
        quant_dtype_np = TORCH_QTYPE_TO_NP_TYPE[qweight.dtype]
        scale = _np.float32(qweight.q_scale())
        zero_point = quant_dtype_np(qweight.q_zero_point())
        quantized_weights = _torch.int_repr(qweight).numpy()
        dequant_weights = _utils._construct_constexpr_dequant_op(
            quantized_weights, zero_point, scale, axis=None, name=name
        )
    # per_channel_affine_float_qparams is same as per_channel_affine except that it
    # expects both scale and zero point to be floating point values.
    elif qweight.qscheme() in {_torch.per_channel_affine, _torch.per_channel_affine_float_qparams}:
        quant_dtype_np = TORCH_QTYPE_TO_NP_TYPE[qweight.dtype]
        # TODO: How do we set the appropriate dtype here (fp16/fp32)?
        scale = qweight.q_per_channel_scales().numpy()
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
        axis = _np.int32(qweight.q_per_channel_axis())
        dequant_weights = _utils._construct_constexpr_dequant_op(
            quantized_weights, zero_point, scale, axis=axis, name=name
        )
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


@register_torch_op(
    torch_alias=[
        "quantized_decomposed::embedding_4bit",
        "quantized_decomposed::embedding_4bit.dtype",
        "quantized_decomposed.embedding_4bit",
        "quantized_decomposed.embedding_4bit.dtype",
    ]
)
def quantized_embedding_4bit(context, node):
    """Lower the 4-bit quantized embedding op used in executorch."""
    inputs = _get_inputs(context, node, expected=[6, 7])
    weight = inputs[0].val
    weight_scales = inputs[1].val
    weight_zero_points = None
    if inputs[2] is not None and inputs[2].val is not None:
        weight_zero_points = inputs[2].val
    weight_quant_min = inputs[3].val
    weight_quant_max = inputs[4].val
    indices = inputs[5]

    out_np_dtype = None
    if len(inputs) > 6:
        if isinstance(inputs[6].val, _torch.dtype):
            out_np_dtype = NUM_TO_NUMPY_DTYPE[TORCH_DTYPE_TO_NUM[inputs[6].val]]
        elif isinstance(inputs[6].val, (int, _np.generic)):
            out_np_dtype = NUM_TO_NUMPY_DTYPE[inputs[6].val]
    if out_np_dtype is not None:
        weight_scales = weight_scales.astype(out_np_dtype)

    if weight_quant_min == 0 and weight_quant_max == 0:
        # Executorch wrongly passes both weight_quant_min and weight_quant_max. We should set it to correct numbers.
        signed = True
        weight_quant_min = -8
        weight_quant_max = 7
    else:
        signed = weight_quant_min < 0

    quant_low = -8 if signed else 0
    quant_high = 7 if signed else 15
    quant_torch_dtype = _torch.int8 if signed else _torch.uint8
    if weight_quant_min != quant_low:
        raise ValueError(
            f"The weight_quant_min should be {quant_low} for 4-bit embedding, but got {weight_quant_min}."
        )
    if weight_quant_max != quant_high:
        raise ValueError(
            f"The weight_quant_max should be {quant_high} for 4-bit embedding, but got {weight_quant_max}."
        )

    # Unpack the weight to the normal layout.
    with _torch.no_grad():
        weight = _torch.from_numpy(weight)
        # The original weight was packed by using 8-bit to represent two numbers, so we need to separate them.
        help_move_bits = 2**4
        weight_even = weight.div(help_move_bits, rounding_mode="trunc")
        weight_odd = weight.remainder(help_move_bits)
        weight_unpacked = _torch.stack((weight_even, weight_odd), dim=-1)
        weight = weight_unpacked.view(weight.shape[0], -1)
        weight = weight.view(quant_torch_dtype).add(weight_quant_min).numpy()

    if not _np.logical_and(weight >= quant_low, weight <= quant_high).all():
        raise ValueError(
            f"All elements in weight should be within 4-bit range ({quant_low} to {quant_high})."
        )

    quantized_np_dtype = types.nptype_from_builtin(
        types.string_to_builtin("int4" if signed else "uint4")
    )
    dequant_weight = _utils._construct_constexpr_dequant_op(
        weight.astype(quantized_np_dtype),
        weight_zero_points,
        weight_scales,
        axis=-1,
        name=inputs[0].name,
    )

    gather = mb.gather(x=dequant_weight, indices=indices, name=node.name)
    context.add(gather)


@register_torch_op
def _convert_weight_to_int4pack(context, node):
    """Pack weight to int4pack format which will be fed into `_weight_int4pack_mm` op."""
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0].val
    inner_k_tiles = inputs[1].val

    if x is None or inner_k_tiles is None:
        raise NotImplementedError(
            "For `_convert_weight_to_int4pack` op, we only support static case, where x, "
            "and inner_k_tiles are all known during compilation time."
        )

    with _torch.no_grad():
        x_int4packed = _torch._convert_weight_to_int4pack(
            _torch.from_numpy(x), inner_k_tiles
        ).numpy()

    res = mb.const(val=x_int4packed, name=node.name)
    context.add(res)


@register_torch_op
def _weight_int4pack_mm(context, node):
    """
    The first argument is the same as torch.mm, but the second argument (weight) is packed.
    The packed weight has rank=4, because the meta registration in dynamo requires operator has the same output shape
    for each device. So it creates a fake shape {N / 8, K / (16 * innerKTiles), 32, innerKTiles / 2} for CPU.

    More specifically:

        # Original torch.mm
        torch.mm(a, b)

        # The int4 packed version mm
        b_uint8, b_scales_and_zeros = _group_quantize_tensor(
            b, n_bit=4, q_group_size=q_group
        )
        b_int4pack = torch._convert_weight_to_int4pack(
            b_uint8, inner_k_tiles
        )
        weight_int4pack_mm(a, b_int4pack, b_scales_and_zeros)
    """
    if Version(_torch.__version__) < Version("2.4.0"):
        raise AssertionError("To lower _weight_int4pack_mm, requires torch >= 2.4.0")

    logger.warning(
        "The current conversion of `_weight_int4pack_mm` op only works with model produced by torchao. "
        "If the op is produced by other libs, you may observe large numerical discrepancy."
    )

    if not _HAS_TORCHAO:
        raise AssertionError(
            f"{MSG_TORCHAO_NOT_FOUND}\n torchao is needed to convert torch blockwise quantized model."
        )

    inputs = _get_inputs(context, node, expected=4)
    x = inputs[0]
    y_int4pack = inputs[1].val
    group_size = inputs[2].val
    y_scales_and_zeros = inputs[3].val

    if y_int4pack is None or group_size is None or y_scales_and_zeros is None:
        raise NotImplementedError(
            "For `_weight_int4pack_mm` op, we only support static case, where y_int4pack, "
            "group_size, y_scales_and_zeros are all known during compilation time."
        )

    if not (len(y_scales_and_zeros.shape) == 3 and y_scales_and_zeros.shape[2] == 2):
        raise ValueError(
            "The scales_and_zeros from torch should have 3 dims and last dim has size 2."
        )
    scales = _np.transpose(y_scales_and_zeros[:, :, 0])
    zero_points = _np.transpose(y_scales_and_zeros[:, :, 1])

    if _np.allclose(zero_points, zero_points.astype("int32")):
        zero_points = zero_points.astype("int32")
    else:
        zero_points = zero_points.astype(_np.float32)

    # Unpack the result of `torch._convert_weight_to_int4pack` back to plain layout.
    # TODO: Use `torchao.ops.unpack_tensor_core_tiled_layout` to unpack after it has CPU implementation.
    # The current way to unpack by using _weight_int4pack_mm with eye matrix is a workaround on CPU.
    if len(y_int4pack.shape) != 4:
        raise ValueError(
            f"The packed y from torch should have 4 dims, but got {len(y_int4pack.shape)}."
        )
    inner_k_tiles = y_int4pack.shape[-1] * 2
    y_unpacked_shape = (y_int4pack.shape[0] * 8, y_int4pack.shape[1] * (inner_k_tiles * 16))
    eye_shape = y_unpacked_shape[1]
    quant_min = 0
    quant_max = 2**4 - 1
    with _torch.no_grad():
        y_dequantized = (
            _torch._weight_int4pack_mm(
                _torch.eye(eye_shape, device=_torch.device("cpu"), dtype=_torch.float32),
                _torch.from_numpy(y_int4pack),
                group_size,
                _torch.from_numpy(y_scales_and_zeros).float(),
            )
            .t()
            .contiguous()
        )
        zero_point_domain = (
            torchao_quant.ZeroPointDomain.INT
            if _np.issubdtype(zero_points.dtype, _np.integer)
            else torchao_quant.ZeroPointDomain.FLOAT
        )
        y_quantized = torchao_quant.quantize_affine(
            y_dequantized,
            (1, group_size),
            _torch.from_numpy(scales),
            _torch.from_numpy(zero_points),
            _torch.int32,
            quant_min=quant_min,
            quant_max=quant_max,
            zero_point_domain=zero_point_domain,
        )
    y_quantized = y_quantized.numpy().astype(_np.uint8)
    if len(y_quantized.shape) != 2:
        raise ValueError(
            f"The unpacked quantized y should have 2 dims, but got {len(y_quantized.shape)}."
        )
    if not _np.logical_and(y_quantized >= 0, y_quantized <= 15).all():
        raise ValueError("All elements should be within 4-bit range (0 to 15).")

    # If zero_point_domain in `quantize_affine` is set to `ZeroPointDomain.INT`, it matches with CoreML implementation:
    #         quant = torch.clamp(torch.round(input * (1.0 / scale)) + zero_point, quant_min, quant_max)
    # However, for `ZeroPointDomain.FLOAT`, torchao did following transformations to make it compatible with `tinygemm`:
    #         mid_point = (quant_max + quant_min + 1) / 2
    #         min_val = zero_point - scale * mid_point
    #         quant = torch.clamp(torch.round((input - min_val) / scale), quant_min, quant_max))
    # As we want to make sure the quantize matches CoreML dequant op, we have to do following transformations:
    #         dequant = (quant - mid_point) * scale + zp
    # so we can re-write the expression as
    #         dequant = (quant - (mid_point - zp / scale)) * scale
    # which means the zero_point in CoreML is actually `mid_point - zp / scale`.
    if not _np.issubdtype(zero_points.dtype, _np.integer):
        mid_point = (quant_max + quant_min + 1) / 2
        zero_points = mid_point - zero_points / scales

    # Use MIL constexpr op to represent the quantization.
    quantized_np_dtype = types.nptype_from_builtin(types.string_to_builtin("uint4"))
    dequant_weights = _utils._construct_constexpr_dequant_op(
        y_quantized.astype(quantized_np_dtype), zero_points, scales, axis=-1, name=inputs[1].name
    )

    res = mb.linear(x=x, weight=dequant_weights, name=node.name)
    context.add(res)
