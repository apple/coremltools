from __future__ import print_function
from __future__ import division
from tensorflow.python.util import compat
import numpy as np
import tensorflow as tf

from ._layers_common import (
    add_const,
    identity,
    make_tensor,
    skip,
    _get_const_tensor_value,
)
from . import _shape_sensitive_layers as ss_layers

_SKIP_OP_TYPES = ["NoOp", "ExpandDims", "Cast", "Squeeze"]


def _compare(a, b, encoding="utf8"):  # type: (Text, Text, Text) -> bool
    if isinstance(a, bytes):
        a = a.decode(encoding)
    if isinstance(b, bytes):
        b = b.decode(encoding)
    return a == b


def _is_skip_type(op):
    return op.type in _SKIP_OP_TYPES


def _backtrace_skip_ops(start_op):
    # if start_op is skippable, trace down its path to an unskippable op.
    op = start_op
    if not _is_skip_type(op):
        return op
    pred = None if len(op.inputs) == 0 else op.inputs[0].op
    while pred is not None and (_is_skip_type(pred)):
        op = pred
        pred = None if len(op.inputs) == 0 else op.inputs[0].op
    return pred


def add_tensor_sub(builder, name, x_name, y_name, output_name):
    y_out_name = "negated_" + y_name + "_" + output_name
    builder.add_activation(y_out_name, "LINEAR", y_name, y_out_name, [-1.0, 0])
    builder.add_elementwise(name, [x_name, y_out_name], output_name, "ADD")


def add_tensor_div(builder, name, x_name, y_name, output_name):
    y_out_name = "inversed_" + y_name + "_" + output_name
    builder.add_unary(y_out_name, y_name, y_out_name, "inverse")
    builder.add_elementwise(name, [x_name, y_out_name], output_name, "MULTIPLY")


def _update_padding_and_crop_values_2D(pad_values, crop_values, params):
    def _new_pad_crop_1D(p1, p2, c1, c2, k, s, n1):
        n2 = np.floor((n1 + p1 + p2 - k) / s) + 1
        if 1 + c1 * s <= p1:
            p1 -= c1 * s
            c1 = 0
        if k + (n2 - c2 - 1) * s > p1 + n1:
            p2 = k + (n2 - c2 - 1) - (p1 + n1)
            c2 = 0
        return p1, p2, c1, c2

    p1, p2, c1, c2 = _new_pad_crop_1D(
        pad_values[2],
        pad_values[3],
        crop_values[2],
        crop_values[3],
        params["kh"],
        params["sh"],
        params["Hin"],
    )
    pad_values[2:] = np.array([p1, p2], dtype=np.int)
    crop_values[2:] = np.array([c1, c2], dtype=np.int)

    p1, p2, c1, c2 = _new_pad_crop_1D(
        pad_values[0],
        pad_values[1],
        crop_values[0],
        crop_values[1],
        params["kw"],
        params["sw"],
        params["Win"],
    )
    pad_values[:2] = np.array([p1, p2], dtype=np.int)
    crop_values[:2] = np.array([c1, c2], dtype=np.int)


def placeholder(op, context):
    context.translated[compat.as_str_any(op.outputs[0].name)] = True
    try:
        inname = op.inputs[0].name
        # chain together no-ops here
        if inname in context.out_name_to_in_name:
            context.out_name_to_in_name[
                op.outputs[0].name
            ] = context.out_name_to_in_name[op.inputs[0].name]
        else:
            context.out_name_to_in_name[op.outputs[0].name] = op.inputs[0].name
    except:
        print("Skipping name of placeholder")


def batchnorm(op, context):

    input_name = compat.as_str_any(op.inputs[0].name)
    output_name = compat.as_str_any(op.outputs[0].name)
    num_channels = int(op.inputs[0].shape[-1])

    instance_normalization = False

    if op.type == "BatchNormWithGlobalNormalization":
        mean = context.consts[compat.as_str_any(op.inputs[1].name)]
        variance = context.consts[compat.as_str_any(op.inputs[2].name)]
        beta = context.consts[compat.as_str_any(op.inputs[3].name)]
        gamma = context.consts[compat.as_str_any(op.inputs[4].name)]
        epsilon = op.get_attr("variance_epsilon")
    elif op.type == "FusedBatchNorm":
        param_list = []
        for idx in range(1, 5):
            t = _get_const_tensor_value(context, op.inputs[idx].name, op.inputs[idx].op)
            if t is None:
                raise ValueError("Value not found for {}".format(op.inputs[idx].name))
            param_list.append(t)
        gamma, beta, mean, variance = param_list
        is_training = op.get_attr("is_training")
        if mean.shape == (0,) and variance.shape == (0,) and is_training:
            instance_normalization = True
        if mean.shape == (0,):
            mean = np.zeros((num_channels,))
        if variance.shape == (0,):
            variance = np.ones((num_channels,))
        epsilon = op.get_attr("epsilon")

    context.translated[output_name] = True
    context.builder.add_batchnorm(
        output_name,
        num_channels,
        gamma,
        beta,
        mean,
        variance,
        input_name,
        output_name,
        epsilon=epsilon,
        compute_mean_var=instance_normalization,
        instance_normalization=instance_normalization,
    )


def concat(op, context):
    ss_layers._add_concat(op, context)


def split(op, context):
    ss_layers._add_split(op, context)


def reshape(op, context):
    ss_layers._add_reshape(op, context)


def conv2d(op, context):
    x_name = compat.as_str_any(op.inputs[0].name)
    W_name = compat.as_str_any(op.inputs[1].name)
    output_name = compat.as_str_any(op.outputs[0].name)

    if x_name in context.consts:
        add_const(context, x_name, context.consts[x_name], x_name)

    # get input's height and width
    Hin = context.shape_dict[x_name][1]
    Win = context.shape_dict[x_name][2]

    weight_through_dequantized_op = False
    # check if weights are getting fed via dequantized op
    if op.inputs[1].op.type == "Identity":
        identity_op = op.inputs[1].op
        if identity_op.inputs[0].op.type == "Dequantize":
            back_op = identity_op.inputs[0].op
            if not (
                back_op.get_attr("T") == tf.quint8
                and _compare(back_op.get_attr("mode"), "MIN_COMBINED")
            ):
                raise NotImplementedError(
                    "Dequantize mode not supported for convolution weights"
                )
            min_W = context.session.run(back_op.inputs[1])
            max_W = context.session.run(back_op.inputs[2])
            W = context.session.run(back_op.inputs[0])
            quant_scale = np.array([((max_W - min_W) / 255.0)])
            quant_bias = np.array([min_W])
            nbits = 8
            quantization_type = "linear"
            assert len(W.shape) <= 4
            W_shape = [1] * (4 - len(W.shape)) + list(W.shape)
            W = W.reshape(W_shape)
            W = W.flatten().tobytes()
            weight_through_dequantized_op = True

    # Variables are sometimes 'read' via an Identity
    # Try to get the source of the Identity op if W is not already a constant
    if not weight_through_dequantized_op:
        if W_name in context.consts:
            W = context.consts[W_name]
        else:
            if _is_skip_type(op.inputs[1].op):
                identity_op = _backtrace_skip_ops(op.inputs[1].op)
            else:
                identity_op = op.inputs[1].op
            assert (
                identity_op.type == "Identity"
            ), "Weight input has to be an identity op"
            W_name = compat.as_str_any(identity_op.inputs[0].name)
            W = _get_const_tensor_value(context, W_name, identity_op)
            if W is None:
                raise ValueError("Value not found for {}".format(W_name))

    if op.type == "DepthwiseConv2dNative":
        W = np.transpose(W, (0, 1, 3, 2))

    # Force W to be rank 4
    if not weight_through_dequantized_op:
        assert len(W.shape) <= 4
        W_shape = [1] * (4 - len(W.shape)) + list(W.shape)
        W = W.reshape(W_shape)

    if op.type == "QuantizedConv2D":
        assert (
            op.inputs[4].name in context.consts
        ), "minimum value of quantized weights not available"
        assert (
            op.inputs[5].name in context.consts
        ), "maximum value of quantized weights not available"
        min_W = context.consts[op.inputs[4].name]
        max_W = context.consts[op.inputs[5].name]
        if op.get_attr("Tfilter") == tf.quint8:
            # W = ((max_W - min_W)/255.0) * W + min_W
            quant_scale = np.array([((max_W - min_W) / 255.0)])
            quant_bias = np.array([min_W])
            nbits = 8
            quantization_type = "linear"
            W = W.flatten().tobytes()
        else:
            assert False, "Only uint8 weights handled currently by the converter"

        context.translated[compat.as_str_any(op.outputs[1].name)] = True
        context.translated[compat.as_str_any(op.outputs[2].name)] = True

    inp_shape = context.shape_dict[x_name]
    out_shape = context.shape_dict[output_name]

    kernelChannels = inp_shape[-1]
    if op.type == "DepthwiseConv2dNative":
        kernelChannels = 1
    outputChannels = out_shape[-1]
    height = W_shape[0]
    width = W_shape[1]
    strides = op.get_attr("strides")
    stride_height = strides[1]
    stride_width = strides[2]
    borderMode = compat.as_str_any(op.get_attr("padding").lower())
    groups = 1
    if op.type == "DepthwiseConv2dNative":
        groups = inp_shape[-1]
    b = None
    has_bias = False
    is_deconv = False
    output_shape = None
    if op.type == "DepthwiseConv2dNative":
        output_shape = out_shape
    input_name = x_name

    # dilated conv uses SpatialToBatchND as input; grab dilation rate there
    dilation_factors = [1, 1]

    is_pad_before = False  # is there padding before Conv
    is_crop_after = False  # is there cropping after Conv
    pad_values = [0, 0, 0, 0]  # in order left, right (W), top, bottom (H)
    crop_values = [0, 0, 0, 0]  # in order left, right (W), top, bottom (H)

    ######## 2-D Conv case ######################
    # check for SpaceToBatch and padding
    if op.inputs[0].op.type == "SpaceToBatchND":
        op1 = op.inputs[0].op
        Hin = context.shape_dict[op1.inputs[0].name][1]
        Win = context.shape_dict[op1.inputs[0].name][2]
        dilation_factors = context.consts[op1.inputs[1].name]
        dilation_factors = list(dilation_factors.astype("int"))
        if op1.inputs[2].name in context.consts:
            padding = context.consts[op1.inputs[2].name]
        else:
            padding = context.session.run(
                op1.inputs[2].name, feed_dict=context.input_feed_dict
            )
        pad_values[2] = padding[0, 0]  # top
        pad_values[3] = padding[0, 1]  # bottom
        pad_values[0] = padding[1, 0]  # left
        pad_values[1] = padding[1, 1]  # right
        # check for BatchToSpace and cropping
        if len(context.blob_graph[output_name]) > 0:
            op2 = context.blob_graph[output_name][0]
            if op2.type == "BatchToSpaceND":
                if op2.inputs[2].name in context.consts:
                    crops = context.consts[op2.inputs[2].name]
                else:
                    crops = context.session.run(
                        op2.inputs[2].name, feed_dict=context.input_feed_dict
                    )
                crop_values[2] = crops[0, 0]  # top
                crop_values[3] = crops[0, 1]  # bottom
                crop_values[0] = crops[1, 0]  # left
                crop_values[1] = crops[1, 1]  # right
    ######## 1-D Conv case ######################
    # check for SpaceToBatch and padding
    elif (
        op.inputs[0].op.type == "ExpandDims"
        and op.inputs[0].op.inputs[0].op.type == "SpaceToBatchND"
    ):
        op1 = op.inputs[0].op.inputs[0].op
        Hin = 1
        Win = context.shape_dict[op.inputs[0].op.inputs[0].op.inputs[0].name][-2]
        df = context.consts[op1.inputs[1].name][0]
        dilation_factors[-1] = df
        padding = context.consts[op1.inputs[2].name]
        pad_values[0:2] = padding[0][0:2]
        # check for BatchToSpace and cropping
        if len(context.blob_graph[output_name]) > 0:
            if context.blob_graph[output_name][0].type == "Squeeze":
                squeeze_op = context.blob_graph[output_name][0]
                squeeze_op_output_name = squeeze_op.outputs[0].name
                if len(context.blob_graph[squeeze_op_output_name]) > 0:
                    op2 = context.blob_graph[squeeze_op_output_name][0]
                    if op2.type == "BatchToSpaceND":
                        crops = context.consts[op2.inputs[2].name]
                        crop_values[0:2] = crops[0][0:2]

    if sum(crop_values) != 0:
        if borderMode != "valid":
            is_crop_after = True
        else:
            # check whether padding values can be changed to avoid having a crop
            # layer later
            params = dict()
            params["kh"] = (height - 1) * dilation_factors[0] + 1
            params["kw"] = (width - 1) * dilation_factors[1] + 1
            params["sh"] = stride_height
            params["sw"] = stride_width
            params["Hin"] = Hin
            params["Win"] = Win
            _update_padding_and_crop_values_2D(
                pad_values=pad_values, crop_values=crop_values, params=params
            )
            if sum(crop_values) != 0:
                is_crop_after = True
    is_pad_before = True if sum(pad_values) != 0 else False

    conv_output_name = output_name
    conv_input_name = input_name

    if is_pad_before:
        output_name_pad = conv_input_name + "_padded"
        context.builder.add_padding(
            name=output_name_pad,
            left=pad_values[0],
            right=pad_values[1],
            top=pad_values[2],
            bottom=pad_values[3],
            value=0,
            input_name=input_name,
            output_name=output_name_pad,
        )
        conv_input_name = output_name_pad

    if is_crop_after:
        conv_output_name = conv_output_name + "_precrop"

    if op.type == "QuantizedConv2D" or weight_through_dequantized_op:
        context.builder.add_convolution(
            name=conv_output_name,
            kernel_channels=kernelChannels,
            output_channels=outputChannels,
            height=height,
            width=width,
            stride_height=stride_height,
            stride_width=stride_width,
            border_mode=borderMode,
            groups=groups,
            W=W,
            b=b,
            has_bias=has_bias,
            is_deconv=is_deconv,
            output_shape=output_shape,
            input_name=conv_input_name,
            output_name=conv_output_name,
            dilation_factors=dilation_factors,
            quantization_type=quantization_type,
            quant_scale=quant_scale,
            quant_bias=quant_bias,
            nbits=nbits,
        )
        context.builder.spec.specificationVersion = 3
    else:
        context.builder.add_convolution(
            name=conv_output_name,
            kernel_channels=kernelChannels,
            output_channels=outputChannels,
            height=height,
            width=width,
            stride_height=stride_height,
            stride_width=stride_width,
            border_mode=borderMode,
            groups=groups,
            W=W,
            b=b,
            has_bias=has_bias,
            is_deconv=is_deconv,
            output_shape=output_shape,
            input_name=conv_input_name,
            output_name=conv_output_name,
            dilation_factors=dilation_factors,
        )

    context.translated[compat.as_str_any(op.outputs[0].name)] = True

    if is_crop_after:
        context.builder.add_crop(
            name=output_name,
            left=crop_values[0],
            right=crop_values[1],
            top=crop_values[2],
            bottom=crop_values[3],
            offset=0,
            input_names=[conv_output_name],
            output_name=output_name,
        )


def deconv2d(op, context):
    x_name = compat.as_str_any(op.inputs[2].name)
    W_name = compat.as_str_any(op.inputs[1].name)
    output_name = compat.as_str_any(op.outputs[0].name)
    if W_name in context.consts:
        W = context.consts[W_name]
    else:
        identity_op = op.inputs[1].op
        assert identity_op.type == "Identity", "Weight input has to be an identity op"
        W_name = compat.as_str_any(identity_op.inputs[0].name)
        assert W_name in context.consts, "Value not found for {}".format(W_name)
        W = context.consts[W_name]

    inp_shape = context.shape_dict[x_name]
    out_shape = context.shape_dict[output_name]

    W_shape = W.shape
    kernelChannels = inp_shape[-1]
    outputChannels = out_shape[-1]
    height = W_shape[0]
    width = W_shape[1]
    strides = op.get_attr("strides")
    stride_height = strides[1]
    stride_width = strides[2]
    borderMode = compat.as_str_any(op.get_attr("padding").lower())
    groups = 1
    b = None
    has_bias = False
    is_deconv = True
    output_shape = None
    input_name = x_name
    context.builder.add_convolution(
        name=output_name,
        kernel_channels=kernelChannels,
        output_channels=outputChannels,
        height=height,
        width=width,
        stride_height=stride_height,
        stride_width=stride_width,
        border_mode=borderMode,
        groups=groups,
        W=np.transpose(W, (0, 1, 3, 2)),
        b=b,
        has_bias=has_bias,
        is_deconv=is_deconv,
        output_shape=output_shape,
        input_name=input_name,
        output_name=output_name,
    )
    context.translated[compat.as_str_any(op.outputs[0].name)] = True


def _pool(op, context, mode):
    x_name = compat.as_str_any(op.inputs[0].name)
    output_name = compat.as_str_any(op.outputs[0].name)

    inp_shape = context.shape_dict[x_name]

    # Unlike conv that uses width axis for 1D computation,
    # Tensorflow uses height axis for 1D pooling. For 1D case we need to swap
    # height and width.
    # is_1d = (inp_shape[1] > 1 and inp_shape[2] == 1)
    is_1d = op.inputs[0].op.type == "ExpandDims"
    if is_1d:
        dim_input = op.inputs[0].op.inputs[1]
        dim = context.session.run(dim_input.name, feed_dict=context.input_feed_dict)

    W_shape = op.get_attr("ksize")
    height = W_shape[2] if (is_1d and dim == 2) else W_shape[1]
    width = W_shape[1] if (is_1d and dim == 2) else W_shape[2]
    strides = op.get_attr("strides")
    stride_height = strides[2] if (is_1d and dim == 2) else strides[1]
    stride_width = strides[1] if (is_1d and dim == 2) else strides[2]
    borderMode = compat.as_str_any(op.get_attr("padding"))
    context.builder.add_pooling(
        name=output_name,
        height=height,
        width=width,
        stride_height=stride_height,
        stride_width=stride_width,
        layer_type=mode,
        padding_type=borderMode,
        exclude_pad_area=True,
        is_global=False,
        input_name=x_name,
        output_name=output_name,
    )

    context.translated[compat.as_str_any(op.outputs[0].name)] = True


def avgpool(op, context):
    _pool(op, context, "AVERAGE")


def maxpool(op, context):
    _pool(op, context, "MAX")


def inner_product(op, context):
    x_name = compat.as_str_any(op.inputs[0].name)
    W_name = compat.as_str_any(op.inputs[1].name)
    output_name = compat.as_str_any(op.outputs[0].name)

    if W_name in context.consts:
        W = context.consts[W_name]
    else:
        identity_op = op.inputs[1].op
        assert identity_op.type == "Identity", "Weight input has to be an identity op"
        W_name = compat.as_str_any(identity_op.inputs[0].name)
        assert W_name in context.consts, "Value not found for {}".format(W_name)
        W = context.consts[W_name]
    assert not op.get_attr("transpose_a") and not op.get_attr(
        "transpose_b"
    ), "Transpose on inputs not supported"

    # Force W to be rank 2
    assert len(W.shape) <= 2
    W_shape = [1] * (2 - len(W.shape)) + list(W.shape)
    W = W.reshape(W_shape)
    W = np.transpose(W, (1, 0))

    if op.type == "QuantizedMatMul":
        assert (
            op.inputs[4].name in context.consts
        ), "minimum value of quantized weights not available"
        assert (
            op.inputs[5].name in context.consts
        ), "maximum value of quantized weights not available"
        min_W = context.consts[op.inputs[4].name]
        max_W = context.consts[op.inputs[5].name]
        if op.get_attr("T2") == tf.quint8:
            # W = ((max_W - min_W)/255.0) * W + min_W
            quant_scale = np.array([((max_W - min_W) / 255.0)])
            quant_bias = np.array([min_W])
            nbits = 8
            quantization_type = "linear"
            W = W.flatten().tobytes()
        else:
            assert False, "Only uint8 weights handled currently by the converter"

        context.translated[compat.as_str_any(op.outputs[1].name)] = True
        context.translated[compat.as_str_any(op.outputs[2].name)] = True

    inp_shape = context.shape_dict[x_name]
    out_shape = context.shape_dict[output_name]

    nB = inp_shape[-1]
    nC = out_shape[-1]

    bias = None
    has_bias = False

    # See if BiasAdd or Add can be fused
    for ops in context.all_ops:
        if ops.type == "BiasAdd" or ops.type == "Add":
            if compat.as_str_any(ops.inputs[0].name) == output_name:
                if compat.as_str_any(ops.inputs[1].name) in context.consts:
                    bias = context.consts[compat.as_str_any(ops.inputs[1].name)]
                    has_bias = True
                if (
                    ops.inputs[1].op.type == "Identity"
                    and compat.as_str_any(ops.inputs[1].op.inputs[0].name)
                    in context.consts
                ):
                    bias = context.consts[
                        compat.as_str_any(ops.inputs[1].op.inputs[0].name)
                    ]
                    has_bias = True
                if has_bias:
                    BiasAdd_out_name = compat.as_str_any(ops.outputs[0].name)
                    context.translated[BiasAdd_out_name] = True
                    context.translated[output_name] = True
                    output_name = BiasAdd_out_name
                    break

    if op.type == "QuantizedMatMul":
        context.builder.add_inner_product(
            op.name,  # name
            W,  # W
            bias,  # Wb
            nB,  # nB
            nC,  # nC
            has_bias,  # has_bias
            x_name,  # input_name
            output_name,  # output_name
            quantization_type=quantization_type,
            quant_scale=quant_scale,
            quant_bias=quant_bias,
            nbits=nbits,
        )
        context.builder.spec.specificationVersion = 3
    else:
        context.builder.add_inner_product(
            op.name,  # name
            W,  # W
            bias,  # Wb
            nB,  # nB
            nC,  # nC
            has_bias,  # has_bias
            x_name,  # input_name
            output_name,  # output_name
        )

    context.translated[output_name] = True


def _get_broadcasted_shape4(shapes):
    broadcasted_shape = [1, 1, 1, 1]
    for shape in shapes:
        rank = len(shape)
        shape4 = [1] * (4 - rank) + shape
        broadcasted_shape = [
            max(shape4[i], broadcasted_shape[i]) for i in range(len(broadcasted_shape))
        ]
    return broadcasted_shape


def _broadcast_axis(ref_shape4, shape):
    if None in shape:  # when shape is not fully determined, just skip
        return None
    ref_shape = ref_shape4[-3:]
    rank = len(shape)
    shape = shape[-3:] if rank >= 3 else [1] * (3 - rank) + shape
    # shape and ref_shape are [H,W,C] now
    ratios = np.array(ref_shape) / np.array(shape)
    if ratios[0] != 1 or ratios[1] != 1:
        if ratios[0] != 1 and ratios[1] != 1:
            return None
        return 1 if ratios[0] != 1 else 2
    return None


def add(op, context):
    output_name = compat.as_str_any(op.outputs[0].name)
    if op.type == "QuantizedBiasAdd":
        input_names = [make_tensor(ts, context) for ts in op.inputs[:2]]
        input_shapes = [context.shape_dict[ts.name] for ts in op.inputs[:2]]
    else:
        # input_names: names of input tensors
        input_names = [make_tensor(ts, context) for ts in op.inputs]
        # input_shapes: shapes of input tensors
        input_shapes = [context.shape_dict[ts.name] for ts in op.inputs]

    mult_input_names = input_names

    # For rank-4 inputs, CoreML only allows [1], [C], [1,H,W] blobs to be
    # broadcasted in elementwise operations. To handle other broadcasting cases,
    # (e.g. [1,1,W] --> [C,H,W]), we insert up-sampling layers
    input_ranks = [len(shape) for shape in input_shapes]
    if 4 in input_ranks:
        broadcasted_shape4 = _get_broadcasted_shape4(input_shapes)
        for idx, in_name in enumerate(input_names):
            input_shape = input_shapes[idx]
            if len(input_shape) == 1 and input_shape[0] == broadcasted_shape4[-1]:
                continue
            axis = _broadcast_axis(broadcasted_shape4, input_shape)
            if axis is not None:
                # add upsample layer
                upsampled_in_name = in_name + "__upsampled"
                mult_input_names[idx] = upsampled_in_name
                input_axis_dim = 1 if axis >= len(input_shape) else input_shape[axis]
                scale = broadcasted_shape4[axis] // input_axis_dim
                if axis == 1:
                    context.builder.add_upsample(
                        upsampled_in_name, scale, 1, in_name, upsampled_in_name
                    )
                else:
                    context.builder.add_upsample(
                        upsampled_in_name, 1, scale, in_name, upsampled_in_name
                    )

    context.builder.add_elementwise(output_name, mult_input_names, output_name, "ADD")
    context.translated[output_name] = True
    if op.type == "QuantizedBiasAdd":
        context.translated[op.outputs[1].name] = True
        context.translated[op.outputs[2].name] = True


def mul(op, context):
    output_name = compat.as_str_any(op.outputs[0].name)

    # input_names: names of input tensors
    input_names = [make_tensor(ts, context) for ts in op.inputs]
    # input_shapes: shapes of input tensors
    input_shapes = [context.shape_dict[ts.name] for ts in op.inputs]
    mult_input_names = input_names

    # For rank-4 inputs, CoreML only allows [1], [C], [1,H,W] blobs to be
    # broadcasted in elementwise operations. To handle other broadcasting cases,
    # (e.g. [1,1,W] --> [C,H,W]), we insert up-sampling layers
    input_ranks = [len(shape) for shape in input_shapes]
    if 4 in input_ranks:
        broadcasted_shape4 = _get_broadcasted_shape4(input_shapes)
        for idx, in_name in enumerate(input_names):
            input_shape = input_shapes[idx]
            if len(input_shape) == 1 and input_shape[0] == broadcasted_shape4[-1]:
                continue
            axis = _broadcast_axis(broadcasted_shape4, input_shape)
            if axis is not None:
                # add upsample layer
                upsampled_in_name = in_name + "__upsampled"
                mult_input_names[idx] = upsampled_in_name
                input_axis_dim = 1 if axis >= len(input_shape) else input_shape[axis]
                scale = broadcasted_shape4[axis] // input_axis_dim
                if axis == 1:
                    context.builder.add_upsample(
                        upsampled_in_name, scale, 1, in_name, upsampled_in_name
                    )
                else:
                    context.builder.add_upsample(
                        upsampled_in_name, 1, scale, in_name, upsampled_in_name
                    )

    context.builder.add_elementwise(
        output_name, mult_input_names, output_name, "MULTIPLY"
    )
    context.translated[output_name] = True


def abs(op, context):
    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)
    context.builder.add_unary(output_name, input_name, output_name, "abs")
    context.translated[output_name] = True


def neg(op, context):
    input_name = compat.as_str_any(op.inputs[0].name)
    output_name = compat.as_str_any(op.outputs[0].name)
    make_tensor(op.inputs[0], context)
    context.builder.add_activation(
        output_name, "LINEAR", input_name, output_name, [-1.0, 0]
    )
    context.translated[output_name] = True


def sub(op, context):
    assert len(op.inputs) == 2, "Sub op currently supports only two inputs"
    output_name = compat.as_str_any(op.outputs[0].name)
    input_1_name = make_tensor(op.inputs[0], context)
    input_2_name = make_tensor(op.inputs[1], context)
    add_tensor_sub(
        context.builder, output_name, input_1_name, input_2_name, output_name
    )
    context.translated[output_name] = True


def rsqrt(op, context):
    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)
    context.builder.add_unary(output_name, input_name, output_name, "rsqrt")
    context.translated[output_name] = True


def relu(op, context):
    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)
    context.builder.add_activation(output_name, "RELU", input_name, output_name)
    context.translated[output_name] = True
    if op.type == "QuantizedRelu":
        context.translated[op.outputs[1].name] = True
        context.translated[op.outputs[2].name] = True


def leaky_relu(op, context):
    input_name = make_tensor(op.inputs[0], context)
    alpha = op.get_attr("alpha")
    output_name = compat.as_str_any(op.outputs[0].name)
    context.builder.add_activation(
        output_name, "LEAKYRELU", input_name, output_name, [alpha]
    )
    context.translated[output_name] = True


def exp(op, context):
    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)
    context.builder.add_unary(output_name, input_name, output_name, "exp")
    context.translated[output_name] = True


def elu(op, context):
    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)
    context.builder.add_activation(output_name, "ELU", input_name, output_name, 1.0)
    context.translated[output_name] = True


def tanh(op, context):
    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)
    context.builder.add_activation(output_name, "TANH", input_name, output_name)
    context.translated[output_name] = True


def relu6(op, context):
    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)

    relu_output_name = "relu_" + output_name
    context.builder.add_activation(
        relu_output_name, "RELU", input_name, relu_output_name
    )
    neg_output_name = relu_output_name + "_neg"
    # negate it
    context.builder.add_activation(
        neg_output_name, "LINEAR", relu_output_name, neg_output_name, [-1.0, 0]
    )
    # apply threshold
    clip_output_name = relu_output_name + "_clip"
    context.builder.add_unary(
        clip_output_name, neg_output_name, clip_output_name, "threshold", alpha=-6.0
    )
    # negate it back
    context.builder.add_activation(
        output_name, "LINEAR", clip_output_name, output_name, [-1.0, 0]
    )
    context.translated[output_name] = True


def softmax(op, context):
    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)
    context.builder.add_softmax(output_name, input_name, output_name)
    context.translated[output_name] = True


def constant(op, context):
    assert (
        compat.as_str_any(op.outputs[0].name) in context.consts
    ), "Value for {} not found".format(op.outputs[0].name)


def greater(op, context):
    output_name = compat.as_str_any(op.outputs[0].name)
    assert len(op.inputs) == 2, "Op Greater sees more than 2 inputs"
    assert (
        len(op.inputs[1].shape) == 0
    ), "Op Greater conversion can't handle non-constant"
    input_name = compat.as_str_any(op.inputs[0].name)
    const_name = compat.as_str_any(op.inputs[1].name)
    if const_name not in context.consts:
        raise NotImplementedError(
            "GreaterEqual op not fully supported by CoreML currently. \
                              Please try to use the graph transform tool to simplify the frozen graph."
        )
    const_val = context.consts[const_name]
    alpha = 1000.0
    beta = 0.5 - alpha * const_val
    context.builder.add_activation(
        output_name, "SIGMOID_HARD", input_name, output_name, params=[alpha, beta]
    )
    context.translated[output_name] = True


def reduce_sum(op, context):
    ss_layers._add_reduce(op, context, "sum")


def reduce_max(op, context):
    ss_layers._add_reduce(op, context, "max")


def reduce_min(op, context):
    ss_layers._add_reduce(op, context, "min")


def mean(op, context):
    ss_layers._add_reduce(op, context, "avg")


def product(op, context):

    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)
    start_ind = context.consts[op.inputs[1].name]

    assert start_ind == 0, "Prod: only start index = 0 case supported"

    input_shape = context.shape_dict[input_name]

    if len(input_shape) == 1:
        axis = "C"
    else:
        assert False, "Reduce Sum axis case not handled currently"

    mode = "prod"
    context.translated[output_name] = True
    context.builder.add_reduce(output_name, input_name, output_name, axis, mode)


def mirror_pad(op, context):
    input_name = compat.as_str_any(op.inputs[0].name)
    output_name = compat.as_str_any(op.outputs[0].name)

    paddings = context.consts[op.inputs[1].name]
    top = paddings[1][0]
    bottom = paddings[1][1]
    left = paddings[2][0]
    right = paddings[2][1]

    assert (
        compat.as_str_any(op.get_attr("mode")) != "SYMMETRIC"
    ), "symmetric mode is not supported by Core ML"

    context.translated[output_name] = True
    context.builder.add_padding(
        output_name,
        left,
        right,
        top,
        bottom,
        input_name=input_name,
        output_name=output_name,
        padding_type="reflection",
    )


def pad(op, context):

    input_name = compat.as_str_any(op.inputs[0].name)
    output_name = compat.as_str_any(op.outputs[0].name)
    padding_input_name = compat.as_str_any(op.inputs[1].name)

    if padding_input_name in context.consts:
        paddings = context.consts[op.inputs[1].name]
    else:
        paddings = context.session.run(
            padding_input_name, feed_dict=context.input_feed_dict
        )

    if paddings.shape[0] == 4:
        offset = 1
    elif paddings.shape[0] == 3:
        offset = 0
    else:
        raise NotImplementedError("Padding case not supported")

    top = paddings[offset][0]
    bottom = paddings[offset][1]
    left = paddings[offset + 1][0]
    right = paddings[offset + 1][1]
    channel_begin = paddings[offset + 2][0]
    channel_end = paddings[offset + 2][1]

    if channel_begin + channel_end == 0:
        context.builder.add_padding(
            output_name,
            left,
            right,
            top,
            bottom,
            input_name=input_name,
            output_name=output_name,
            padding_type="constant",
        )
    elif top + bottom + left + right == 0:
        top = channel_begin
        bottom = channel_end
        context.builder.add_permute(
            output_name + "swap_H_C", (0, 2, 1, 3), input_name, output_name + "swap_H_C"
        )
        context.builder.add_padding(
            output_name + "padded_channel",
            left,
            right,
            top,
            bottom,
            input_name=output_name + "swap_H_C",
            output_name=output_name + "padded_channel",
            padding_type="constant",
        )
        context.builder.add_permute(
            output_name, (0, 2, 1, 3), output_name + "padded_channel", output_name
        )
    else:
        assert False, "Padding case not supported"

    context.translated[output_name] = True


def squared_difference(op, context):

    input_name = compat.as_str_any(op.inputs[0].name)
    input2 = compat.as_str_any(op.inputs[1].name)
    output_name = compat.as_str_any(op.outputs[0].name)

    context.translated[output_name] = True
    add_tensor_sub(
        context.builder,
        output_name + "_difference",
        input_name,
        input2,
        output_name + "_difference",
    )
    context.builder.add_elementwise(
        output_name,
        [output_name + "_difference", output_name + "_difference"],
        output_name,
        "MULTIPLY",
    )


def square(op, context):

    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)

    context.translated[output_name] = True
    context.builder.add_elementwise(
        output_name, [input_name, input_name], output_name, "MULTIPLY"
    )


def resize_nearest_neighbor(op, context):

    input_name = compat.as_str_any(op.inputs[0].name)
    output_name = compat.as_str_any(op.outputs[0].name)

    if op.inputs[1].name in context.consts:
        output_spatial_sizes = context.consts[op.inputs[1].name]
    else:
        output_spatial_sizes = context.session.run(
            op.inputs[1].name, feed_dict=context.input_feed_dict
        )

    shape = context.shape_dict[input_name]

    assert (
        len(shape) == 4
    ), "Resize Nearest Neighbour: unrecognized 4-D shape. Input shape = {}".format(
        str(shape)
    )
    assert output_spatial_sizes[0] % shape[1] == 0, (
        "Resize Nearest Neighbour: height upsampling factor must be an integer. "
        "Input height = {}, output height = {}, ratio = {}".format(
            shape[1], output_spatial_sizes[0], output_spatial_sizes[0] / shape[1]
        )
    )
    assert output_spatial_sizes[1] % shape[2] == 0, (
        "Resize Nearest Neighbour: width upsampling factor must be an integer. "
        "Input width = {}, output width = {}, ratio = {}".format(
            shape[2], output_spatial_sizes[1], output_spatial_sizes[1] / shape[2]
        )
    )

    upsample_factor_height = output_spatial_sizes[0] // shape[1]
    upsample_factor_width = output_spatial_sizes[1] // shape[2]

    context.builder.add_upsample(
        output_name,
        upsample_factor_height,
        upsample_factor_width,
        input_name,
        output_name,
        mode="NN",
    )
    context.translated[output_name] = True


def resize_bilinear(op, context):

    input_name = compat.as_str_any(op.inputs[0].name)
    output_name = compat.as_str_any(op.outputs[0].name)

    if op.inputs[1].name in context.consts:
        output_spatial_sizes = context.consts[op.inputs[1].name]
    else:
        output_spatial_sizes = context.session.run(
            op.inputs[1].name, feed_dict=context.input_feed_dict
        )

    shape = context.shape_dict[input_name]

    assert (
        len(shape) == 4
    ), "Resize Bilinear: input must be 4-D shape. Input shape = {}".format(str(shape))

    if op.get_attr("align_corners"):
        mode = "STRICT_ALIGN_ENDPOINTS_MODE"
    else:
        mode = "UPSAMPLE_MODE"

    if (
        mode == "UPSAMPLE_MODE"
        and (output_spatial_sizes[0] % shape[1] == 0)
        and (output_spatial_sizes[1] % shape[2] == 0)
    ):
        upsample_factor_height = output_spatial_sizes[0] // shape[1]
        upsample_factor_width = output_spatial_sizes[1] // shape[2]
        context.builder.add_upsample(
            output_name,
            upsample_factor_height,
            upsample_factor_width,
            input_name,
            output_name,
            mode="BILINEAR",
        )
    else:
        context.builder.add_resize_bilinear(
            output_name,
            input_name,
            output_name,
            target_height=output_spatial_sizes[0],
            target_width=output_spatial_sizes[1],
            mode=mode,
        )
        context.builder.spec.specificationVersion = 3

    context.translated[output_name] = True


def crop_and_resize(op, context):
    input_name = compat.as_str_any(op.inputs[0].name)
    boxes_name = compat.as_str_any(op.inputs[1].name)
    box_ind_name = compat.as_str_any(op.inputs[2].name)
    output_name = compat.as_str_any(op.outputs[0].name)
    shape = context.shape_dict[input_name]
    assert (
        len(shape) == 4
    ), "Crop and Resize: input must be 4-D shape. Input shape = {}".format(str(shape))
    output_spatial_sizes = context.consts[op.inputs[3].name]

    if boxes_name in context.consts and box_ind_name in context.consts:
        boxes = context.consts[boxes_name]  # (N,4)
        box_ind = context.consts[box_ind_name]  # (N)
        b = np.concatenate((np.expand_dims(box_ind, axis=1), boxes), axis=1)
        context.builder.add_load_constant(
            boxes_name + "0", boxes_name + "0", b.flatten(), [b.shape[0], b.shape[1], 1]
        )
        context.builder.add_permute(
            boxes_name, [1, 2, 0, 3], boxes_name + "0", boxes_name
        )
    elif boxes_name in context.consts and box_ind_name not in context.consts:
        boxes = context.consts[boxes_name]  # (N,4)
        context.builder.add_load_constant(
            boxes_name + "0",
            boxes_name + "0",
            boxes.flatten(),
            [boxes.shape[0], boxes.shape[1], 1],
        )
        context.builder.add_permute(
            boxes_name + "1", [1, 2, 0, 3], boxes_name + "0", boxes_name + "1"
        )
        context.builder.add_elementwise(
            boxes_name + "2", [box_ind_name, boxes_name + "1"], boxes_name, "CONCAT"
        )
    elif boxes_name not in context.consts and box_ind_name in context.consts:
        box_ind = context.consts[box_ind_name]  # (N)
        context.builder.add_load_constant(
            box_ind_name + "0",
            box_ind_name + "0",
            box_ind.flatten(),
            [box_ind.shape[0], 1, 1],
        )
        context.builder.add_permute(
            box_ind_name + "1", [1, 0, 2, 3], box_ind_name + "0", box_ind_name + "1"
        )
        context.builder.add_elementwise(
            boxes_name + "2",
            [box_ind_name + "1", boxes_name],
            boxes_name + "_concat",
            "CONCAT",
        )
        boxes_name += "_concat"
    else:
        context.builder.add_elementwise(
            boxes_name, [box_ind_name, boxes_name], boxes_name + "_concat", "CONCAT"
        )
        boxes_name += "_concat"

    context.builder.add_crop_resize(
        output_name,
        [input_name, boxes_name],
        output_name,
        target_height=output_spatial_sizes[0],
        target_width=output_spatial_sizes[1],
        mode="ALIGN_ENDPOINTS_MODE",
        normalized_roi=True,
        box_indices_mode="CORNERS_HEIGHT_FIRST",
        spatial_scale=1.0,
    )
    context.builder.spec.specificationVersion = 3
    context.translated[output_name] = True


def sigmoid(op, context):
    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)

    context.translated[output_name] = True
    context.builder.add_activation(output_name, "SIGMOID", input_name, output_name)


def transpose(op, context):
    assert len(op.inputs) == 2, "Op Greater sees more than 2 inputs"
    output_name = compat.as_str_any(op.outputs[0].name)
    input_name = compat.as_str_any(op.inputs[0].name)
    param_name = compat.as_str_any(op.inputs[1].name)
    axes = list(context.consts[param_name])
    assert len(axes) == 4, "Op Transpose conversion only works with 4D tensors"

    # TODO - only works for 4D tensor without batch axis
    assert axes[0] == 0, "only works for 4D tensor without batch axis"

    # Permutation without swapping appears to give wrong results, which may be
    # a bug in coreml itself
    if axes[1] != 1 and axes[2] != 2 and axes[3] != 3:
        assert False, "Only swapping permutes work"

    # First, work out where the indicies should move in TF
    target_idx = (axes.index(0), axes.index(1), axes.index(2), axes.index(3))

    # Translate from NHWC to NCHW order
    target_idx = (target_idx[0], target_idx[3], target_idx[1], target_idx[2])

    def translate_transpose(idx):
        if idx == 0:
            return 0
        if idx == 1:
            return 2
        if idx == 2:
            return 3
        if idx == 3:
            return 1
        assert False

    coreml_axes = map(translate_transpose, target_idx)

    context.builder.add_permute(output_name, list(coreml_axes), input_name, output_name)
    context.translated[output_name] = True


def real_div(op, context):
    output_name = compat.as_str_any(op.outputs[0].name)
    input_names = []
    for inp in op.inputs:
        input_names.append(make_tensor(inp, context))
    add_tensor_div(
        context.builder, output_name, input_names[0], input_names[1], output_name
    )
    context.translated[output_name] = True


def maximum(op, context):
    input_names = [compat.as_str_any(x.name) for x in op.inputs]
    output_name = compat.as_str_any(op.outputs[0].name)
    output_shape = context.shape_dict[output_name]
    for inp in input_names:
        if inp in context.consts:
            x = context.consts[inp]
            x = np.broadcast_to(x, output_shape)
            add_const(context, inp, x, inp)
    context.builder.add_elementwise(output_name, input_names, output_name, "MAX")
    context.translated[output_name] = True


def minimum(op, context):
    input_names = [compat.as_str_any(x.name) for x in op.inputs]
    output_name = compat.as_str_any(op.outputs[0].name)
    output_shape = context.shape_dict[output_name]
    for inp in input_names:
        if inp in context.consts:
            x = context.consts[inp]
            x = np.broadcast_to(x, output_shape)
            add_const(context, inp, x, inp)
    context.builder.add_elementwise(output_name, input_names, output_name, "MIN")
    context.translated[output_name] = True


def shape(op, context):
    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)
    input_shape = context.shape_dict[input_name]
    if isinstance(input_shape, list):
        x = np.asarray(input_shape)
    else:
        x = np.asarray(list(input_shape))
    add_const(context, output_name, x, output_name, [len(input_shape), 1, 1])
    context.translated[output_name] = True


def random(op, context):
    # TODO - CoreML does not have random
    output_name = compat.as_str_any(op.outputs[0].name)
    output_shape = context.shape_dict[output_name]
    add_const(context, output_name, np.zeros((output_shape)), output_name)
    context.translated[output_name] = True


def argmax(op, context):
    input_name = compat.as_str_any(op.inputs[0].name)
    output_name = compat.as_str_any(op.outputs[0].name)

    input_shape = context.shape_dict[input_name]
    output_shape = context.shape_dict[output_name]

    axis_tensor = compat.as_str_any(op.inputs[1].name)
    if axis_tensor in context.consts:
        axis_tf = context.consts[axis_tensor]
    else:
        axis_tf = context.session(axis_tensor, feed_dict=context.input_feed_dict)

    assert (
        isinstance(axis_tf, int)
        or isinstance(axis_tf, np.int32)
        or isinstance(axis_tf, np.int)
    ), (
        "Argmax: Only case that is convertible is when axis is an integer. "
        "Input shape = {}, output shape = {}, axis = {}".format(
            str(input_shape), str(output_shape), str(axis_tf)
        )
    )

    if len(input_shape) == 1:
        axis = "C"
    elif len(input_shape) == 2 and axis_tf == 1:
        axis = "C"
    elif len(input_shape) == 3:
        axis = ["H", "W", "C"][axis_tf]
    elif len(input_shape) == 4 and axis_tf > 0:
        axis = ["B", "H", "W", "C"][axis_tf]
    else:
        assert False, (
            "ArgMax: Axis translation case not handled currently. "
            "Input shape = {}, output shape = {}, axis = {}".format(
                str(input_shape), str(output_shape), str(axis_tf)
            )
        )

    context.builder.add_reduce(output_name, input_name, output_name, axis, "argmax")
    context.translated[output_name] = True


def extract_image_patches(op, context):
    # use a big convolution layer (that has weights!) for this op
    input_name = compat.as_str_any(op.inputs[0].name)
    output_name = compat.as_str_any(op.outputs[0].name)
    ksizes = op.get_attr("ksizes")
    padding_type = compat.as_str_any(op.get_attr("padding"))
    if padding_type == "VALID":
        padding_type = "valid"
    elif padding_type == "SAME":
        padding_type = "same"
    else:
        raise NotImplementedError("%s not implemented" % (padding_type))
    strides = op.get_attr("strides")
    rates = op.get_attr("rates")
    assert rates == [1] * len(rates), "Only supports when rates are all 1s"
    kh, kw = ksizes[1], ksizes[2]
    sh, sw = strides[1], strides[2]

    c_in = context.shape_dict[input_name][-1]
    n_filters = kh * kw * c_in
    W = np.zeros((kh, kw, c_in, n_filters))
    for i_h in range(kh):
        for i_w in range(kw):
            for i_c in range(c_in):
                idx = i_c + (i_w * c_in) + (i_h * c_in * kw)
                W[i_h, i_w, i_c, idx] = 1

    context.builder.add_convolution(
        name=output_name,
        kernel_channels=c_in,
        output_channels=n_filters,
        height=kh,
        width=kw,
        stride_height=sh,
        stride_width=sw,
        border_mode=padding_type,
        groups=1,
        W=W,
        b=None,
        has_bias=False,
        is_deconv=False,
        output_shape=None,
        input_name=input_name,
        output_name=output_name,
    )
    context.translated[output_name] = True


def one_hot(op, context):
    input_name = compat.as_str_any(op.inputs[0].name)
    output_name = compat.as_str_any(op.outputs[0].name)

    depth = context.consts[compat.as_str_any(op.inputs[1].name)]
    on_value = context.consts[compat.as_str_any(op.inputs[2].name)]
    off_value = context.consts[compat.as_str_any(op.inputs[3].name)]

    n_dims = depth
    W = np.ones((depth, depth)) * off_value
    for i in range(depth):
        W[i, i] = on_value
    context.builder.add_embedding(
        name=output_name,
        W=W,
        b=None,
        input_dim=n_dims,
        output_channels=n_dims,
        has_bias=False,
        input_name=input_name,
        output_name=output_name,
    )
    context.translated[output_name] = True


def fill(op, context):
    output_name = op.outputs[0].name

    assert (
        op.inputs[1].name in context.consts
    ), "Second input to the Fill op must be a constant"
    assert (
        output_name in context.shape_dict
    ), "Shape of the output of Fill must be known"
    shape = context.shape_dict[output_name]
    x = np.zeros(shape)
    add_const(context, output_name, x, output_name)
    context.translated[output_name] = True


def strided_slice(op, context):

    input_name = compat.as_str_any(op.inputs[0].name)
    output_name = compat.as_str_any(op.outputs[0].name)
    make_tensor(op.inputs[0], context)

    [x, y] = context.session.run(
        [input_name, output_name], feed_dict=context.input_feed_dict
    )

    if op.inputs[1].name in context.consts:
        begin = context.consts[compat.as_str_any(op.inputs[1].name)]
    else:
        begin = context.session.run(
            op.inputs[1].name, feed_dict=context.input_feed_dict
        )

    if op.inputs[2].name in context.consts:
        end = context.consts[compat.as_str_any(op.inputs[2].name)]
    else:
        end = context.session.run(op.inputs[2].name, feed_dict=context.input_feed_dict)

    if op.inputs[3].name in context.consts:
        strides = context.consts[compat.as_str_any(op.inputs[3].name)]
    else:
        strides = context.session.run(
            op.inputs[3].name, feed_dict=context.input_feed_dict
        )

    begin_mask = op.get_attr("begin_mask")
    end_mask = op.get_attr("end_mask")
    ellipsis_mask = op.get_attr("ellipsis_mask")
    new_axis_mask = op.get_attr("new_axis_mask")
    shrink_axis_mask = op.get_attr("shrink_axis_mask")

    input_shape = context.shape_dict[input_name]
    output_shape = context.shape_dict[output_name]

    if (
        len(input_shape) == 1
        and len(begin) == 1
        and len(end) == 1
        and len(strides) == 1
    ):
        if begin_mask:
            begin[0] = 0
        if end_mask:
            end[0] = input_shape[0]
        context.builder.add_slice(
            output_name,
            input_name,
            output_name,
            "channel",
            int(begin[0]),
            int(end[0]),
            int(strides[0]),
        )
    elif len(x.shape) == 4 and len(y.shape) == 3 and x.shape[:3] == y.shape:
        context.builder.add_slice(
            output_name,
            input_name,
            output_name,
            "channel",
            int(begin[-1]),
            int(end[-1]),
            1,
        )
    elif input_name in context.consts:
        # this means all the inputs to the strided slice layer are constant
        add_const(context, output_name, y, output_name)
    elif np.array_equal(np.squeeze(x), np.squeeze(y)):
        skip(op, context)
    # check for slice along the height and width axis
    elif (
        len(input_shape) == 4
        and len(begin) == 4
        and len(strides) == 4
        and len(end) == 4
        and (begin_mask == 9 or (begin[0] == 0 and begin[-1] == 0))
    ):

        end_masks = [int(mask) for mask in list(bin(end_mask))[2:][::-1]]
        for dim in range(4):
            end[dim] = end[dim] if end[dim] >= 0 else end[dim] + input_shape[dim]
            if end_masks[dim] == 1:
                end[dim] = input_shape[dim]

        size = [end[i] - begin[i] for i in range(4)]

        if input_shape[1] > size[1] and input_shape[2] > size[2]:
            tmp_output_name = output_name + "_height_sliced"
            tmp_input_name = tmp_output_name
        elif input_shape[1] > size[1] or input_shape[2] > size[2]:
            tmp_output_name = output_name
            tmp_input_name = input_name
        else:
            raise NotImplementedError(
                "Strided Slice case not handled. Input shape = {}, output shape = {}".format(
                    str(input_shape), str(output_shape)
                )
            )

        if input_shape[1] > size[1]:
            context.builder.add_slice(
                tmp_output_name,
                input_name,
                tmp_output_name,
                "height",
                int(begin[1]),
                int(end[1]),
                int(strides[1]),
            )
        if input_shape[2] > size[2]:
            context.builder.add_slice(
                output_name,
                tmp_input_name,
                output_name,
                "width",
                int(begin[2]),
                int(end[2]),
                int(strides[2]),
            )

    elif (
        len(input_shape) == 4
        and len(begin) == 4
        and len(size) == 4
        and all([input_shape[i] == size[i] for i in range(3)])
    ):
        context.builder.add_slice(
            output_name,
            input_name,
            output_name,
            "channel",
            int(begin[3]),
            int(begin[3]) + int(size[3]),
            1,
        )
    else:
        assert (
            False
        ), "Strided Slice case not handled. Input shape = {}, output shape = {}".format(
            str(input_shape), str(output_shape)
        )
    context.translated[output_name] = True


def slice(op, context):

    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)
    input_shape = context.shape_dict[input_name]
    output_shape = context.shape_dict[output_name]

    # skip if output shape is same as input shape: in that case its a dummy operation
    if input_shape == output_shape:
        skip(op, context)
        return

    [x, y] = context.session.run(
        [input_name, output_name], feed_dict=context.input_feed_dict
    )

    if op.inputs[1].name in context.consts:
        begin = context.consts[compat.as_str_any(op.inputs[1].name)]
    else:
        begin = context.session.run(
            op.inputs[1].name, feed_dict=context.input_feed_dict
        )

    if op.inputs[2].name in context.consts:
        size = context.consts[compat.as_str_any(op.inputs[2].name)]
    else:
        size = context.session.run(op.inputs[2].name, feed_dict=context.input_feed_dict)

    size = [
        input_shape[i] - begin[i] if dim_size == -1 else dim_size
        for i, dim_size in enumerate(size)
    ]
    # check for slice along the channel axis
    if len(input_shape) == 1 and len(begin) == 1 and len(size) == 1:
        context.builder.add_slice(
            output_name,
            input_name,
            output_name,
            "channel",
            int(begin[0]),
            int(begin[0]) + int(size[0]),
            1,
        )
    # cases requiring simultaneous slice across height, width and channel not handled yet.
    elif (
        len(input_shape) == 4
        and len(begin) == 4
        and len(size) == 4
        and input_shape[0] == size[0]
        and input_shape[-1] == size[-1]
    ):
        if input_shape[1] > size[1] and input_shape[2] > size[2]:
            tmp_output_name = output_name + "_height_sliced"
            tmp_input_name = tmp_output_name
        elif input_shape[1] > size[1] or input_shape[2] > size[2]:
            tmp_output_name = output_name
            tmp_input_name = input_name
        else:
            raise NotImplementedError(
                "Slice case not handled "
                "(input shape: %s, output shape: %s)"
                % (str(input_shape), str(output_shape))
            )
        if input_shape[1] > size[1]:
            context.builder.add_slice(
                tmp_output_name,
                input_name,
                tmp_output_name,
                "height",
                int(begin[1]),
                int(begin[1]) + int(size[1]),
                1,
            )
        if input_shape[2] > size[2]:
            context.builder.add_slice(
                output_name,
                tmp_input_name,
                output_name,
                "width",
                int(begin[2]),
                int(begin[2]) + int(size[2]),
                1,
            )
    elif (
        len(input_shape) == 4
        and len(begin) == 4
        and len(size) == 4
        and all([input_shape[i] == size[i] for i in range(3)])
    ):
        context.builder.add_slice(
            output_name,
            input_name,
            output_name,
            "channel",
            int(begin[3]),
            int(begin[3]) + int(size[3]),
            1,
        )
    elif input_name in context.consts:
        # this means all the inputs to the slice layer are constant
        add_const(context, output_name, y, output_name)
    elif np.array_equal(np.squeeze(x), np.squeeze(y)):
        skip(op, context)
    else:
        raise NotImplementedError(
            "Slice case not handled "
            "(input shape: %s, output shape: %s)"
            % (str(input_shape), str(output_shape))
        )

    context.translated[output_name] = True


# connect i-th output to the i-th input
def skip_one_to_one(op, context):
    for out in op.outputs:
        if out.name in context.output_names:
            identity(op, context)
            return

    assert len(op.inputs) == len(
        op.outputs
    ), "must have same number of outputs as inputs"

    for i, out in enumerate(op.outputs):
        inp_name = op.inputs[i].name
        if inp_name not in context.skip_map_names:
            context.skip_map_names[out.name] = inp_name
        else:
            context.skip_map_names[out.name] = context.skip_map_names[inp_name]
        context.translated[out.name] = True


# Only a very specific case of the gather op is handled
# Currently, CoreML cannot implement the general functionality of a gather op
def gather(op, context):

    output_name = op.outputs[0].name
    input_name = op.inputs[0].name

    input_shape = context.shape_dict[input_name]
    output_shape = context.shape_dict[output_name]

    assert (
        len(context.shape_dict[op.inputs[0].name]) == 1
    ), "first input to 'gather' must be a 1-D tensor. Input shape = {}, output shape = {}".format(
        str(input_shape), str(output_shape)
    )
    assert (
        op.inputs[1].name in context.consts
    ), "second input to 'gather' must be a constant"

    indices = context.consts[op.inputs[1].name]
    # check that indices are contiguous
    for i in range(len(indices) - 1):
        if indices[i + 1] - indices[i] != 1:
            raise ValueError("indices of the gather op must be contiguous")

    context.builder.add_slice(
        output_name, input_name, output_name, "channel", indices[0], indices[-1] + 1, 1
    )

    context.translated[output_name] = True


def reciprocal(op, context):
    input_name = make_tensor(op.inputs[0], context)
    output_name = op.outputs[0].name
    context.builder.add_unary(output_name, input_name, output_name, "inverse")
    context.translated[output_name] = True


def lrn(op, context):
    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)

    input_shape = context.shape_dict[input_name]
    C = input_shape[-1]
    alpha = op.get_attr("alpha")
    beta = op.get_attr("beta")
    bias = op.get_attr("bias")
    depth_radius = op.get_attr("depth_radius")
    context.builder.add_lrn(
        output_name,
        input_name,
        output_name,
        alpha=alpha * C,
        beta=beta,
        local_size=depth_radius,
        k=bias,
    )
    context.translated[output_name] = True


def log(op, context):
    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)
    context.builder.add_unary(output_name, input_name, output_name, "log")
    context.translated[output_name] = True


def space_to_batch(op, context):
    # check for a particular order:
    # 1. 'SpaceToBatchND' --> 'Conv2D' --> 'BatchToSpaceND' OR
    # 2. 'SpaceToBatchND' --> 'ExpandDims' ---> 'Conv2D' --> 'Squeeze' ----> 'BatchToSpaceND'
    # If this is the pattern then skip this op otherwise raise an error
    op_type_list = []
    next_op = op
    for i in range(4):
        if len(context.blob_graph[next_op.outputs[0].name]) == 0:
            break
        else:
            next_op = context.blob_graph[next_op.outputs[0].name][0]
            op_type_list.append(next_op.type)

    if (
        len(op_type_list) > 1
        and (op_type_list[0] == "Conv2D" or op_type_list[0] == "DepthwiseConv2dNative")
        and op_type_list[1] == "BatchToSpaceND"
    ):
        skip(op, context)
    elif (
        len(op_type_list) > 3
        and op_type_list[0] == "ExpandDims"
        and op_type_list[1] == "Conv2D"
        and op_type_list[2] == "Squeeze"
        and op_type_list[3] == "BatchToSpaceND"
    ):
        skip(op, context)
    else:
        raise NotImplementedError(
            "SpaceToBatch op as used in this network is not supported by CoreML currently."
        )


def batch_to_space(op, context):
    # check for a particular order:
    # 1. 'SpaceToBatchND' --> 'Conv2D' --> 'BatchToSpaceND' OR
    # 2. 'SpaceToBatchND' --> 'ExpandDims' ---> 'Conv2D' --> 'Squeeze' ----> 'BatchToSpaceND'
    # If this is the pattern then skip this op otherwise raise an error
    op_type_list = []
    prev_op = op
    for i in range(4):
        if len(prev_op.inputs) > 0 and prev_op.inputs[0].op:
            prev_op = prev_op.inputs[0].op
            op_type_list.append(prev_op.type)
        else:
            break

    if (
        len(op_type_list) > 1
        and (op_type_list[0] == "Conv2D" or op_type_list[0] == "DepthwiseConv2dNative")
        and op_type_list[1] == "SpaceToBatchND"
    ):
        skip(op, context)
    elif (
        len(op_type_list) > 3
        and op_type_list[0] == "Squeeze"
        and op_type_list[1] == "Conv2D"
        and op_type_list[2] == "ExpandDims"
        and op_type_list[3] == "SpaceToBatchND"
    ):
        skip(op, context)
    else:
        raise NotImplementedError(
            "BatchToSpace op as used in this network is not supported by CoreML currently."
        )


# TODO: this might fail in some circumstances
# this op is generally used to set other parameters
# Hence we simply add the output as a load constant in the Core ML graph
def floormod(op, context):
    output_name = compat.as_str_any(op.outputs[0].name)
    x = context.session.run(output_name, feed_dict=context.input_feed_dict)
    add_const(context, output_name, x, output_name)
    context.translated[output_name] = True


def sqrt(op, context):
    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)
    context.builder.add_unary(output_name, input_name, output_name, "sqrt")
    context.translated[output_name] = True


def pow(op, context):
    input_name = compat.as_str_any(op.inputs[0].name)
    power_name = compat.as_str_any(op.inputs[1].name)

    if power_name in context.consts and context.consts[power_name].dtype == np.float32:
        alpha = context.consts[power_name]
        output_name = compat.as_str_any(op.outputs[0].name)
        context.builder.add_unary(
            output_name, input_name, output_name, mode="power", alpha=alpha
        )
        context.translated[output_name] = True
    else:
        raise ValueError("Pow op only supported when power is a fixed constant")


def _add_reorganize_data(op, context, mode):
    input_name = make_tensor(op.inputs[0], context)
    output_name = compat.as_str_any(op.outputs[0].name)
    block_size = op.get_attr("block_size")
    context.builder.add_reorganize_data(
        output_name, input_name, output_name, mode=mode, block_size=block_size
    )
    context.translated[output_name] = True


def depth_to_space(op, context):
    _add_reorganize_data(op, context, "DEPTH_TO_SPACE")


def space_to_depth(op, context):
    _add_reorganize_data(op, context, "SPACE_TO_DEPTH")
