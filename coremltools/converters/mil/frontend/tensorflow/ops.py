#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as _np
import numpy as np

from coremltools import _logger as logger
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as target
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.block import is_current_opset_version_compatible_with
from coremltools.converters.mil.mil.ops.defs._utils import broadcast_shapes, promote_input_dtypes
from coremltools.converters.mil.mil.types import builtin_to_string
from coremltools.converters.mil.mil.types.symbolic import is_symbolic

from .._utils import build_einsum_mil
from .convert_utils import convert_graph
from .tf_op_registry import register_tf_op


def _adjust_min_max(min, max, num_bits=8):
    if (min <= max) and (max <= 0):
        min = (min - max) * 1.0
        max = 0.0
    elif (min >= 0) and (max >= min):
        max = (max - min) * 1.0
        min = 0.0
    else:
        scale = (max - min) / (2 ** num_bits - 1)
        min_adj = scale * round(min / scale)
        max_adj = max + min_adj - min
        min = min_adj
        max = max_adj
    return min, max


def _is_scalar(type_):
    if type_ is None:
        return False
    result = types.is_int(type_) or types.is_float(type_) or types.is_bool(type_)
    if types.is_tensor(type_) and (len(type_.get_shape()) == 0):
        result = True
    return result


def _transpose_NHWC_to_NCHW(x):
    return mb.transpose(x=x, perm=[0, 3, 1, 2])


def _transpose_NCHW_to_NHWC(x, node_name):
    return mb.transpose(x=x, perm=[0, 2, 3, 1], name=node_name)


def _transpose_NDHWC_to_NCDHW(x):
    return mb.transpose(x=x, perm=[0, 4, 1, 2, 3])


def _transpose_NCDHW_to_NDHWC(x, node_name):
    return mb.transpose(x=x, perm=[0, 2, 3, 4, 1], name=node_name)


def _check_axes_type(x):
    if x is None or x.val is None:
        return None
    if isinstance(x.val, _np.int32):
        return _np.array([x.val])
    return x.val


def _value_at(x, idx):
    """
    input x: 1D tensor (vector).
    return value at index idx. x[idx].
    """
    assert x.rank == 1
    return mb.slice_by_index(x=x, begin=[idx], end=[0], squeeze_mask=[True])


def _freq_to_mel(freq):
    return 1127.0 * _np.log(1 + freq / 700.0)


def _get_MFCC_constants(spectrogram_N,
                        sample_rate,
                        upper_frequency_limit,
                        lower_frequency_limit,
                        filterbank_channel_count,
                        dct_coefficient_count):

    """
    params:
    spectrogram_N : int
    sample_rate: int
    upper_frequency_limit : int
    filterbank_channel_count : int
    dct_coefficient_count : int

    returns:
    array(shape: (spectrogram_N,))
    array(shape: (spectrogram_N, filterbank_channel_count))
    array(shape: (spectrogram_N, filterbank_channel_count))
    array(shape: (filterbank_channel_count, dct_coefficient_count))

    reference:
    https://github.com/tensorflow/tensorflow/blob/dec8e0b11f4f87693b67e125e67dfbc68d26c205/tensorflow/core/kernels/mfcc_mel_filterbank.cc
    """

    center_frequencies = _np.zeros((filterbank_channel_count + 1))
    mel_low = _freq_to_mel(lower_frequency_limit)
    mel_hi = _freq_to_mel(upper_frequency_limit)
    mel_span = mel_hi - mel_low
    mel_spacing = mel_span / (filterbank_channel_count + 1)
    for i in range(filterbank_channel_count + 1):
        center_frequencies[i] = mel_low + (mel_spacing * (i + 1))

    hz_per_sbin = 0.5 * sample_rate / (spectrogram_N - 1)
    start_index = int(1.5 + (lower_frequency_limit / hz_per_sbin))
    end_index = int(upper_frequency_limit / hz_per_sbin)

    band_mapper = _np.zeros((spectrogram_N))
    channel = 0
    for i in range(spectrogram_N):
        melf = _freq_to_mel(i * hz_per_sbin)
        if (i < start_index) or (i > end_index):
            band_mapper[i] = -2
        else:
            while channel < filterbank_channel_count and center_frequencies[channel] < melf:
                channel += 1
            band_mapper[i] = channel - 1  # Can be == -1

    weights = _np.zeros((spectrogram_N))
    for i in range(spectrogram_N):
        channel = int(band_mapper[i])
        if (i < start_index) or (i > end_index):
            weights[i] = 0
        else:
            if channel >= 0:
                weights[i] = (center_frequencies[channel + 1] - _freq_to_mel(i * hz_per_sbin)) / (
                    center_frequencies[channel + 1] - center_frequencies[channel])
            else:
                weights[i] = (center_frequencies[0] - _freq_to_mel(i * hz_per_sbin)) / (center_frequencies[0] - mel_low)

    mat_spec_val = _np.zeros((spectrogram_N, filterbank_channel_count))
    mat_weighted = _np.zeros((spectrogram_N, filterbank_channel_count))
    for i in range(start_index, end_index + 1): # For each FFT bin
        channel = int(band_mapper[i])
        if channel >= 0:
            mat_weighted[i, channel] = 1 # Right side of triangle, downward slope
        channel += 1
        if channel < filterbank_channel_count:
            mat_weighted[i, channel] = -1 # Left side of triangle
            mat_spec_val[i, channel] = 1 # Left side of triangle

    # compute the dct matrix
    cosines = _np.zeros((filterbank_channel_count, dct_coefficient_count))
    fnorm = _np.sqrt(2.0 / filterbank_channel_count)
    arg = _np.pi / filterbank_channel_count
    for i in range(filterbank_channel_count):
        for j in range(dct_coefficient_count):
            cosines[i, j] = fnorm * _np.cos(j * arg * (i + 0.5))

    return weights, mat_weighted, mat_spec_val, cosines


def _reshape_remaining_dimensions_to_canonical_shape(x, remaining_rank):
    # An utility function that reshape a tensor with shape [batch, spatial_dims, remaining_dim_1, ..., remaining_dim_N]
    # to [batch, spatial_dims, remaining_dim_1 * ... * remaining_dim_N]
    # For the special case where there is no remaining dimensions, we expand the last axis
    assert remaining_rank != 1
    if remaining_rank == 0:
        return mb.expand_dims(x=x, axes=[-1])
    else:
        x_shape = mb.shape(x=x)
        batch_and_spatial_shape = mb.slice_by_size(x=x_shape, begin=[0], size=[x.rank-remaining_rank])
        reshape_shape = mb.concat(values=[batch_and_spatial_shape, [-1]], axis=0)
        return mb.reshape(x=x, shape=reshape_shape)


def _reshape_remaining_dimension_to_original_shape(x, original_shape, remaining_rank):
    # An utility function that reshape the tensor with shape [batch_new, spatial_dims_new, remaining_dims] to the original
    # form, which is [batch_new, spatial_dims_new, remaining_dim_1, ..., remaining_dim_N]
    assert remaining_rank != 1
    if remaining_rank == 0:
        return mb.squeeze(x=x, axes=[-1])
    else:
        x_shape = mb.shape(x=x)
        spatial_rank = original_shape.shape[0] - remaining_rank - 1
        batch_and_spatial_shape = mb.slice_by_size(x=x_shape, begin=[0], size=[1+spatial_rank])
        remaining_shape = mb.slice_by_size(x=original_shape, begin=[1+spatial_rank], size=[-1])
        reshape_shape = mb.concat(values=[batch_and_spatial_shape, remaining_shape], axis=0)
        return mb.reshape(x=x, shape=reshape_shape)


@register_tf_op(tf_alias=["BiasAdd", "AddV2"])
def Add(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x, y = promote_input_dtypes([x, y])

    if "data_format" in node.attr and node.attr["data_format"] == "NCHW":
        if x.rank != 1 and y.rank != 1:
            raise AssertionError("Bias needs to have its rank equals to 1")

        bias, data = (y, x) if y.rank == 1 else (x, y)

        if not data.rank >= 3:
            raise AssertionError("Data needs to be of at least ranke 3")

        axes = [-(i + 1) for i in range(data.rank - 2)]

        x = data
        y = mb.expand_dims(x=bias, axes=axes, name=node.name + "_expanded_bias")

    x = mb.add(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def AddN(context, node):
    values = [context[name] for name in node.inputs]
    if len(values) == 1:
        Identity(context, node)
        return
    prev_var = values[0]
    for idx, var in enumerate(values[1:]):
        if var == values[-1]:
            x = mb.add(x=prev_var, y=var, name=node.name)
        else:
            prev_var = mb.add(x=prev_var, y=var, name=node.name + "_tmpAddN_" + str(idx))
    context.add(node.name, x)


@register_tf_op
def Abs(context, node):
    x = context[node.inputs[0]]
    x = mb.abs(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Acos(context, node):
    x = context[node.inputs[0]]
    x = mb.acos(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def All(context, node):
    x = context[node.inputs[0]]
    axes = _check_axes_type(context[node.inputs[1]])
    keep_dims = node.attr.get("keep_dims", False)
    x = mb.cast(x=x, dtype="int32")
    x = mb.reduce_prod(x=x, axes=axes, keep_dims=keep_dims)
    x = mb.cast(x=x, dtype="bool", name=node.name)
    context.add(node.name, x)


@register_tf_op
def Any(context, node):
    x = context[node.inputs[0]]
    axes = _check_axes_type(context[node.inputs[1]])
    keep_dims = node.attr.get("keep_dims", False)
    x = mb.cast(x=x, dtype="int32")
    x = mb.reduce_sum(x=x, axes=axes, keep_dims=keep_dims)
    x = mb.cast(x=x, dtype="bool", name=node.name)
    context.add(node.name, x)


@register_tf_op
def ArgMax(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
    x = mb.reduce_argmax(x=x, axis=axis, name=node.name)
    context.add(node.name, x)


@register_tf_op
def ArgMin(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
    x = mb.reduce_argmin(x=x, axis=axis, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Asin(context, node):
    x = context[node.inputs[0]]
    x = mb.asin(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Atan(context, node):
    x = context[node.inputs[0]]
    x = mb.atan(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Atanh(context, node):
    x = context[node.inputs[0]]
    x = mb.atanh(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def AvgPool(context, node):
    x = context[node.inputs[0]]
    in_shape = x.sym_type.get_shape()
    d_rank = len(in_shape) - 2
    data_format = node.attr.get("data_format", "NHWC")
    ksize = node.attr.get("ksize", None)
    kernel_sizes = _pool_pads_or_strides(ksize, data_format, d_rank)
    strides = node.attr.get("strides", None)
    if strides is not None:
        strides = _pool_pads_or_strides(strides, data_format, d_rank)
    pad_type = node.attr["padding"].lower()
    if data_format == "NHWC":
        x = _transpose_NHWC_to_NCHW(x)
        x = mb.avg_pool(
            x=x,
            kernel_sizes=kernel_sizes,
            strides=strides,
            pad_type=pad_type,
            exclude_padding_from_average=True,
        )
        x = _transpose_NCHW_to_NHWC(x, node.name)
    else:
        x = mb.avg_pool(
            x=x,
            kernel_sizes=kernel_sizes,
            strides=strides,
            pad_type=pad_type,
            exclude_padding_from_average=True,
            name=node.name,
        )
    context.add(node.name, x)


@register_tf_op
def AvgPool3D(context, node):
    x = context[node.inputs[0]]
    d_rank = x.rank - 2
    data_format = node.attr.get("data_format", "NDHWC")
    ksize = node.attr.get("ksize", None)
    kernel_sizes = _pool_pads_or_strides(ksize, data_format, d_rank)
    strides = node.attr.get("strides", None)
    if strides is not None:
        strides = _pool_pads_or_strides(strides, data_format, d_rank)
    pad_type = node.attr["padding"].lower()
    if data_format == "NDHWC":
        x = _transpose_NDHWC_to_NCDHW(x)
        x = mb.avg_pool(
            x=x,
            kernel_sizes=kernel_sizes,
            strides=strides,
            pad_type=pad_type,
            exclude_padding_from_average=True,
        )
        x = _transpose_NCDHW_to_NDHWC(x, node.name)
    else:
        x = mb.avg_pool(
            x=x,
            kernel_sizes=kernel_sizes,
            strides=strides,
            pad_type=pad_type,
            exclude_padding_from_average=True,
            name=node.name,
        )

    context.add(node.name, x)


@register_tf_op
def BatchToSpaceND(context, node):
    # In tensorflow, the input tensor has the shape of (batch,) + spatial_shape + remaining_shape.
    # The shape is treated as a combination of 3 components:
    # 1. A single batch dimension
    # 2. Spatial dimensions, with a length spatial_rank, which could be neither 1 or 2. Also, spatial_rank
    #    is equal to the length of block_shape
    # 3. Remaining dimensions, with a length remaining_rank

    # The logic of translating this op is as followed:
    # 1. We first reshape the input to a canonical shape (rolling the remaining shape dimensions into a
    #    single dimension): (batch,) + spatial_shape + (R), where R = remaining_dim_1 * ... * remaining_dim_n
    # 2. We support rank 1 and rank 2 spatial shape:
    #    (i) rank 1: We decompose the BatchToSpace into small basic ops.
    #    (ii) rank 2: We directly use the built in batch_to_space op.
    #    The output would have shape (batch_new,) + spatial_shape_new + (R)
    # 3. We transform the tensor back, by unrolling the remaining shape: (B_new,) + spatial_shape_new + remaining_shape

    x = context[node.inputs[0]]
    block_shape = context[node.inputs[1]].val
    crops = context[node.inputs[2]]
    original_shape = mb.shape(x=x)

    input_rank = x.rank
    spatial_rank = len(block_shape)
    remaining_rank = x.rank - 1 - spatial_rank
    has_non_unity_remaining_dims = remaining_rank != 1

    if block_shape is None:
        raise NotImplementedError("Not support dynamic block_shape for BatchToSpaceND!")

    if crops.val is not None:
        is_static_crops = True
        crops = crops.val
    else:
        is_static_crops = False

    if has_non_unity_remaining_dims:
        # Reshape the input tensor to shape [batch, spatial_shape, remaining_dim_1 * ... * remaining_dim_N]
        x = _reshape_remaining_dimensions_to_canonical_shape(x, remaining_rank)

    if spatial_rank >= 3:
        raise NotImplementedError("Rank of spatial shape > 2 is not supported.")

    if spatial_rank == 2:
        # Tensor has shape [B, H, W, C], we can directly use the batch_to_space op by doing
        # [B, H, W, C] -> transpose -> [B, C, H, W] -> batch_to_space -> [B_new, C, H_new, W_new] ->
        # transpose -> [B_new, H_new, W_new, C]
        x = mb.transpose(x=x, perm=[0, 3, 1, 2])
        x = mb.batch_to_space(
            x=x, block_shape=block_shape, crops=_np.zeros((2, 2), _np.int32), name=node.name
        )
        need_crop = not is_static_crops or (tuple(crops[0]) != (0, 0) or tuple(crops[1]) != (0, 0))
        if need_crop:
            # crop_height, crop_width = crops[0, :], crops[1, :]
            crop_height = mb.slice_by_index(
                x=crops,
                begin=[0, 0],
                end=[0, 0],
                begin_mask=[False, True],
                end_mask=[False, True],
                squeeze_mask=[True, False],
            )
            crop_width = mb.slice_by_index(
                x=crops,
                begin=[1, 0],
                end=[0, 0],
                begin_mask=[False, True],
                end_mask=[False, True],
                squeeze_mask=[True, False],
            )

            if is_static_crops:
                # If crops is known at compile time, we can directly use mb.crop
                x = mb.crop(x=x, crop_height=crop_height, crop_width=crop_width)
            else:
                # Otherwise, we need to use slice_by_index to implement the crop
                a, b = _value_at(crop_height, 0), _value_at(crop_height, 1)
                c, d = _value_at(crop_width, 0), _value_at(crop_width, 1)

                shape = mb.shape(x=x)
                height, width = _value_at(shape, 2), _value_at(shape, 3)
                begin_idx_height, end_idx_height = a, mb.sub(x=height, y=b)
                begin_idx_width, end_idx_width = c, mb.sub(x=width, y=d)

                begin = mb.concat(values=[0, 0, begin_idx_height, begin_idx_width], axis=0)
                end = mb.concat(values=[0, 0, end_idx_height, end_idx_width], axis=0)
                begin_mask = [True, True, False, False]
                end_mask = [True, True, False, False]
                x = mb.slice_by_index(
                    x=x, begin=begin, end=end, begin_mask=begin_mask, end_mask=end_mask
                )

        x = mb.transpose(x=x, perm=[0, 2, 3, 1])

    if spatial_rank == 1:
        # In this case, we decompose space_to_batch into small basic ops
        # [B, H, C] -> decomposite ops -> [B_new, H_new, C]

        # reshape input to [block_shape, B/block_shape, H, C]
        input_shape = mb.shape(x=x)
        block_shape = block_shape[0]
        batch_size = _value_at(input_shape, 0)
        spatial_size = _value_at(input_shape, 1)
        channel_size = _value_at(input_shape, 2)
        new_batch_size = mb.cast(x=mb.real_div(x=batch_size, y=block_shape), dtype="int32")
        reshape_values = [block_shape, new_batch_size, spatial_size, channel_size]
        reshape_shape = mb.concat(values=reshape_values, axis=0)
        x = mb.reshape(x=x, shape=reshape_shape, name=node.name)

        # permute the tensor to [B/block_shape, H, block_shape, C]
        x = mb.transpose(x=x, perm=[1, 2, 0, 3])

        # reshape the tensor to [B/block_shape, H*block_shape, C]
        new_spatial_size = mb.cast(x=mb.mul(x=spatial_size, y=block_shape), dtype="int32")
        reshape_values = [new_batch_size, new_spatial_size, channel_size]
        reshape_shape = mb.concat(values=reshape_values, axis=0)
        x = mb.reshape(x=x, shape=reshape_shape)

        # crop the tensor to [B/block_shape, H*block_shape - crops[0][0] - crops[0][1], C]
        if is_static_crops:
            # If crops is known at compile time, we can directly call mb.crop
            x = mb.crop(x=x, crop_height=crops[0], crop_width=[0, 0])
        else:
            # For the dynamic crops, we implement it with slice_by_index
            flatten_crops = mb.reshape(x=crops, shape=[-1])
            a, b = _value_at(flatten_crops, 0), _value_at(flatten_crops, 1)

            shape = mb.shape(x=x)
            height = _value_at(shape, 1)
            begin_idx, end_idx = a, mb.sub(x=height, y=b)

            begin = mb.concat(values=[0, begin_idx, 0], axis=0)
            end = mb.concat(values=[0, end_idx, 0], axis=0)
            begin_mask = [True, False, True]
            end_mask = [True, False, True]
            x = mb.slice_by_index(
                x=x, begin=begin, end=end, begin_mask=begin_mask, end_mask=end_mask
            )

    if has_non_unity_remaining_dims:
        # Reshape the tensor from shape [batch_new, spatial_shape_new, remaining_dim_1 * ... * remaining_dim_N] back to
        # shape [batch_new, spatial_shape_new, remaining_shape]
        x = _reshape_remaining_dimension_to_original_shape(x, original_shape, remaining_rank)

    context.add(node.name, mb.identity(x=x, name=node.name))


@register_tf_op
def Ceil(context, node):
    x = context[node.inputs[0]]
    x = mb.ceil(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Const(context, node):
    if node.value is None:
        raise ValueError("Const node '{}' cannot have no value".format(node.name))
    x = mb.const(val=node.value.val, name=node.name)
    context.add(node.name, x)


def _conv2d3d_strides_or_dilations(name, value, data_format, default_value=1):
    """Compute strides or dilation values for 2D and 3D convolutions."""
    if value is None:
        value = default_value
    if not isinstance(value, (int, list)):
        raise ValueError("{} must be an int or list".format(name))

    # Parse number of spatial dimensions from `data_format`, assuming N (batch) and C
    # (input channels) are present
    n_dims = len(data_format) - 2

    if isinstance(value, int):
        return [value] * n_dims

    if len(value) == 1:
        return value * n_dims
    if len(value) == n_dims:
        return value
    if len(value) != n_dims + 2:
        raise ValueError(
            "{} must have length 1, {}, or {}".format(name, n_dims, n_dims + 2)
        )

    if data_format == "NHWC":
        # Only support stride/dilation along N, C == 1
        if not (value[0] == value[3] == 1):
            raise ValueError(
                "{} along N and C other than 1 not implemented".format(name)
            )
        return value[1:3]
    elif data_format == "NCHW" or data_format == "NCDHW":
        if not (value[0] == value[1] == 1):
            raise ValueError(
                "{} along N and C other than 1 not implemented".format(name)
            )
        return value[2:]
    # "NDHWC"
    if not (value[0] == value[4] == 1):
        raise ValueError("{} along N and C other than 1 not implemented".format(name))
    return value[1:4]


@register_tf_op
def Cos(context, node):
    x = context[node.inputs[0]]
    x = mb.cos(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Cosh(context, node):
    x = context[node.inputs[0]]
    x = mb.cosh(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Cross(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    # last dim must be 3; other dims must match
    assert x.shape[1:] == y.shape[1:]
    assert x.shape[-1] == 3
    x1 = mb.gather(x=x, indices=[1, 2, 0], axis=-1)
    x2 = mb.gather(x=x, indices=[2, 0, 1], axis=-1)
    y1 = mb.gather(x=y, indices=[1, 2, 0], axis=-1)
    y2 = mb.gather(x=y, indices=[2, 0, 1], axis=-1)
    z = mb.sub(x=mb.mul(x=x1, y=y2), y=mb.mul(x=x2, y=y1), name=node.name)
    context.add(node.name, z)


@register_tf_op
def Einsum(context, node):
    equation = node.attr["equation"]
    a = context[node.inputs[0]]
    b = context[node.inputs[1]]
    x = build_einsum_mil([a, b], equation, node.name)
    context.add(node.name, x)


@register_tf_op
def Equal(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = mb.equal(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def ExtractImagePatches(context, node):
    x = context[node.inputs[0]]
    sizes = node.attr.get("ksizes")
    strides = node.attr.get("strides")
    rates = node.attr.get("rates")
    padding = node.attr.get("padding")
    if x.rank != 4:
        raise ValueError("input for ExtractImagePatches should be a 4D tensor.")
    if not all([rate == 1 for rate in rates]):
        raise NotImplementedError(
            "only rates with all 1s is implemented for ExtractImagePatches."
        )
    if len(sizes) != 4 or sizes[0] != 1 or sizes[3] != 1:
        raise ValueError(
            "ExtractImagePatches only supports sizes (4D tensor) with 1s for batch and channel dimensions."
        )
    if len(sizes) != 4 or strides[0] != 1 or strides[3] != 1:
        raise ValueError(
            "ExtractImagePatches only supports strides (4D tensor) with 1s for batch and channel dimensions."
        )
    if padding not in ["VALID", "SAME"]:
        raise ValueError("non-supported padding for ExtractImagePatches.")
    h, w = x.shape[1], x.shape[2]

    # padding for SAME mode
    if padding == "SAME":
        delta_h = h % strides[1] if h % strides[1] != 0 else strides[1]
        delta_w = w % strides[2] if w % strides[2] != 0 else strides[2]
        last_h = h - delta_h + 1
        last_w = w - delta_w + 1
        pad_h = max(0, last_h + sizes[1] - 1 - h)
        pad_w = max(0, last_w + sizes[2] - 1 - w)
        pad_h = [pad_h // 2, pad_h // 2 if pad_h % 2 == 0 else pad_h // 2 + 1]
        pad_w = [pad_w // 2, pad_w // 2 if pad_w % 2 == 0 else pad_w // 2 + 1]
        pad = _np.array([[0, 0], pad_h, pad_w, [0, 0]]).astype(_np.int32)
        pad = pad.reshape(-1)
        if not all(pad == 0):
            x = mb.pad(x=x, pad=pad, mode="constant", constant_val=0.0)
            h, w = x.shape[1], x.shape[2]

    # compute boxes
    batch = x.shape[0]
    boxes = []
    h_index = list(range(0, h - sizes[1] + 1, strides[1]))
    w_index = list(range(0, w - sizes[2] + 1, strides[2]))
    for hi in h_index:
        for wi in w_index:
            boxes.append((hi, wi, hi + sizes[1] - 1, wi + sizes[2] - 1))

    boxes = _np.array(boxes, dtype=_np.float32)
    box_indices = _np.arange(batch)
    box_indices = _np.tile(box_indices, (len(boxes), 1))
    box_indices = _np.transpose(box_indices)
    box_indices = box_indices.reshape(-1, 1)
    boxes = _np.tile(boxes, (batch, 1))

    x = _transpose_NHWC_to_NCHW(x)
    crop_resize_args = {
        "x": x,
        "target_height": sizes[1],
        "target_width": sizes[2],
        "normalized_coordinates": False,
        "spatial_scale": 1.0,
        "box_coordinate_mode": "CORNERS_HEIGHT_FIRST",
        "sampling_mode": "ALIGN_CORNERS",
    }
    if not is_current_opset_version_compatible_with(target.iOS17):
        # Before IOS17, boxes need to be shape [N,1,4,1,1] or [N,1,5,1,1].
        boxes = _np.concatenate([box_indices, boxes], axis=1)
        boxes = boxes.reshape(boxes.shape[0], 1, boxes.shape[1], 1, 1)
        # Before IOS17, the input param is `roi` instead of `boxes`.
        crop_resize_args["roi"] = boxes
        x = mb.crop_resize(**crop_resize_args)
        # Before IOS17, the output has an extra dim at axis 1.
        x = mb.squeeze(x=x, axes=[1])
    else:
        # At this point `boxes` has shape [N, 4], which is good enough for IOS17+.
        crop_resize_args["boxes"] = boxes
        box_indices = np.squeeze(box_indices, axis=-1)
        crop_resize_args["box_indices"] = box_indices
        x = mb.crop_resize(**crop_resize_args)
    x = _transpose_NCHW_to_NHWC(x, node_name=node.name + "_transpose_to_nhwc")
    x = mb.reshape(x=x, shape=(batch, len(h_index), len(w_index), -1), name=node.name)
    context.add(node.name, x)


@register_tf_op
def Exp(context, node):
    x = context[node.inputs[0]]
    x = mb.exp(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Floor(context, node):
    x = context[node.inputs[0]]
    x = mb.floor(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def FloorDiv(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = mb.floor_div(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Greater(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = mb.greater(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def GreaterEqual(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = mb.greater_equal(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Less(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = mb.less(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def LessEqual(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = mb.less_equal(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Log(context, node):
    x = context[node.inputs[0]]
    x = mb.log(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Log1p(context, node):
    x = context[node.inputs[0]]
    x = mb.log(x=x, epsilon=1., name=node.name)
    context.add(node.name, x)

@register_tf_op
def LogicalAnd(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = mb.logical_and(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def LogicalNot(context, node):
    x = context[node.inputs[0]]
    x = mb.logical_not(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def LogicalOr(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = mb.logical_or(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def LogicalXor(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = mb.logical_xor(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def LRN(context, node):
    x = context[node.inputs[0]]
    depth_radius = node.attr.get("depth_radius")
    size = (depth_radius * 2) + 1
    alpha = node.attr.get("alpha") * size
    beta = node.attr.get("beta")
    bias = node.attr.get("bias")
    x = _transpose_NHWC_to_NCHW(x)
    x = mb.local_response_norm(x=x, size=size, alpha=alpha, beta=beta, k=bias)
    x = _transpose_NCHW_to_NHWC(x, node.name)
    context.add(node.name, x)


@register_tf_op
def Maximum(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = mb.maximum(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Minimum(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = mb.minimum(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def FloorMod(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    floor = mb.floor_div(x=x, y=y, name=node.name + "_floor_div")
    floor_mutiply = mb.mul(x=floor, y=y, name=node.name + "_multiply")
    x = mb.sub(x=x, y=floor_mutiply, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Mul(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = mb.mul(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Neg(context, node):
    x = context[node.inputs[0]]
    x, y = promote_input_dtypes([x, -1])
    x = mb.mul(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def NotEqual(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = mb.not_equal(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Pow(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = mb.pow(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def DepthwiseConv2dNative(context, node):
    # [kH, kW, C_in, multiplier]
    W_hwim = context[node.inputs[1]]  # m = multiplier
    # [kH, kW, 1, C_in * multipler]
    shape_hw1o = list(W_hwim.shape[:2]) + [1, W_hwim.shape[2] * W_hwim.shape[3]]
    W_hw1o = mb.reshape(x=W_hwim, shape=shape_hw1o)
    # [C_in * multipler, 1, kH, kW]. Note that C_in * multiplier = C_out in
    # MIL. C_in / groups = 1 in depthwise conv.
    W_o1hw = mb.transpose(x=W_hw1o, perm=[3, 2, 0, 1])
    data_format = node.attr.get("data_format", "NHWC")
    HW_dilations = _conv2d3d_strides_or_dilations(
        "dilations", node.attr.get("dilations"), data_format
    )
    HW_strides = _conv2d3d_strides_or_dilations(
        "strides", node.attr.get("strides"), data_format
    )

    pad_type = node.attr.get("padding")
    if pad_type not in ["VALID", "SAME"]:
        raise ValueError("Invalid padding type for tf.nn.depthwise_conv2d")

    pad_type = pad_type.lower()
    x = context[node.inputs[0]]
    C_in = x.shape[-1]
    if data_format == "NHWC":
        x = _transpose_NHWC_to_NCHW(x)
    # Only the last op should have the same name as node.name
    conv_name = node.name + "x" if data_format == "NHWC" else node.name
    x = mb.conv(
        x=x,
        weight=W_o1hw,
        pad_type=pad_type,
        strides=HW_strides,
        dilations=HW_dilations,
        groups=C_in,
        name=conv_name,
    )
    if data_format == "NHWC":
        x = _transpose_NCHW_to_NHWC(x, node.name)
    context.add(node.name, x)


@register_tf_op
def FakeQuantWithMinMaxVars(context, node):
    w = context[node.inputs[0]]
    min = context[node.inputs[1]].val
    max = context[node.inputs[2]].val
    num_bits = node.attr['num_bits']
    narrow_range = node.attr['narrow_range']

    min, max = _adjust_min_max(min, max, num_bits)

    if narrow_range:
        scale = (max-min) / (2 ** (num_bits) - 2)
        bias = min - scale
    else:
        scale = (max-min) / (2 ** (num_bits) - 1)
        bias = min

    w = mb.clip(x=w, alpha=min, beta=max)
    w = mb.sub(x=w, y=bias)
    x = mb.real_div(x=w, y=scale)
    x = mb.round(x=x)
    x = mb.mul(x=x, y=scale)
    x = mb.add(x=x, y=bias, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Conv2D(context, node):
    if "quantize" in node.attr:
        quantization_type = "linear"
        min = node.attr['quantize_min']
        max = node.attr['quantize_max']
        nbits = node.attr['num_bits']
        narrow_range = node.attr['narrow_range']

        w = context[node.inputs[1]].sym_val

        min, max = _adjust_min_max(min, max, nbits)

        if narrow_range:
            quant_scale = (max - min) / (2 ** (nbits) - 2)
            quant_bias = (min-quant_scale)
        else:
            quant_scale = (max - min) / (2 ** (nbits) - 1)
            quant_bias = (min)

        w_clip = _np.clip(w, min, max)
        w_round = _np.round((w_clip-quant_bias)/quant_scale)
        W_hwio = w_round.astype(_np.uint8)

        if not isinstance(quant_scale, list) and not isinstance(quant_scale, tuple):
            quant_bias = [quant_bias]
            quant_scale = [quant_scale]
    else:
        quantization_type = None
        nbits = None
        quant_scale = None
        quant_bias = None
        W_hwio = context[node.inputs[1]]

    if quantization_type is not None:
        W_oihw = _np.transpose(W_hwio, axes=[3, 2, 0, 1])
    else:
        W_oihw = mb.transpose(x=W_hwio, perm=[3, 2, 0, 1])

    data_format = node.attr.get("data_format", "NHWC")
    HW_dilations = _conv2d3d_strides_or_dilations(
        "dilations", node.attr.get("dilations"), data_format
    )
    HW_strides = _conv2d3d_strides_or_dilations(
        "strides", node.attr.get("strides"), data_format
    )

    pad_type = node.attr.get("padding")
    pad_type = pad_type.lower()
    pad_type = "custom" if pad_type == "explicit" else pad_type
    assert pad_type in {"same", "valid", "custom"}
    x = context[node.inputs[0]]
    if data_format == "NHWC":
        x = _transpose_NHWC_to_NCHW(x)
        if pad_type == "custom":
            pad_val = node.attr["explicit_paddings"]
            pad_val = pad_val[2:-2]
    elif data_format == "NCHW" and pad_type == "custom":
        pad_val = node.attr["explicit_paddings"]
        pad_val = pad_val[4:]
    # Only the last op should have the same name as node.name
    conv_name = node.name + "x" if data_format == "NHWC" else node.name

    # get the groups from the weighs shape and the input shape
    _, in_channel, _, _ = x.shape
    _, weight_in_channel, _, _ = W_oihw.shape
    if in_channel % weight_in_channel != 0:
        raise ValueError("input channel should be divided by the weight channel.")
    groups = int(in_channel / weight_in_channel)

    if quantization_type is not None:
        x = mb.conv_quantized(
            x=x,
            weight=W_oihw,
            pad_type=pad_type,
            strides=HW_strides,
            dilations=HW_dilations,
            name=conv_name,
            quantization_type=quantization_type,
            nbits=nbits,
            quant_scale=quant_scale,
            quant_bias=quant_bias,
            groups=groups,
        )
    elif pad_type == "custom":
        x = mb.conv(
            x=x,
            weight=W_oihw,
            pad_type=pad_type,
            strides=HW_strides,
            dilations=HW_dilations,
            pad=pad_val,
            groups=groups,
            name=conv_name,
        )
    else:
        x = mb.conv(
            x=x,
            weight=W_oihw,
            pad_type=pad_type,
            strides=HW_strides,
            dilations=HW_dilations,
            groups=groups,
            name=conv_name,
        )
    if data_format == "NHWC":
        x = _transpose_NCHW_to_NHWC(x, node.name)
    context.add(node.name, x)


@register_tf_op
def Conv3D(context, node):
    W_dhwio = context[node.inputs[1]]
    W_oidhw = mb.transpose(x=W_dhwio, perm=[4, 3, 0, 1, 2])
    data_format = node.attr.get("data_format", "NDHWC")
    DHW_dilations = _conv2d3d_strides_or_dilations(
        "dilations", node.attr.get("dilations"), data_format
    )
    DHW_strides = _conv2d3d_strides_or_dilations(
        "strides", node.attr.get("strides"), data_format
    )

    pad_type = node.attr.get("padding")
    if not isinstance(pad_type, str):
        pad_type = "custom"
        raise NotImplementedError("Custom padding not implemented for TF")
    pad_type = pad_type.lower()
    x = context[node.inputs[0]]
    if data_format == "NDHWC":
        # Convert input to NCDHW
        x = _transpose_NDHWC_to_NCDHW(x)
    # Only the last op should have the same name as node.name
    conv_name = node.name + "x" if data_format == "NDHWC" else node.name
    _, in_channel, _, _, _ = x.shape
    _, weight_in_channel, _, _, _ = W_oidhw.shape
    if in_channel % weight_in_channel != 0:
        raise ValueError("input channel should be divided by the weight channel.")
    groups = int(in_channel / weight_in_channel)

    x = mb.conv(
        x=x,
        weight=W_oidhw,
        pad_type=pad_type,
        strides=DHW_strides,
        dilations=DHW_dilations,
        groups=groups,
        name=conv_name,
    )
    if data_format == "NDHWC":
        # Convert input back to NDHWC (from NCDHW)
        x = _transpose_NCDHW_to_NDHWC(x, node.name)
    context.add(node.name, x)


@register_tf_op
def Conv3DBackpropInputV2(context, node):
    # Output shape: [N, D_out, H_out, W_out, C_out]
    output_shape = context[node.inputs[0]].val
    # Weight shape: [D, H, W, C_out, C_in]
    W_dhwoi = context[node.inputs[1]]
    W_iodhw = mb.transpose(x=W_dhwoi, perm=[4, 3, 0, 1, 2])
    # Input shape: [N, D_in, H_in, W_in, C_in]
    x = context[node.inputs[2]]

    data_format = node.attr.get("data_format", "NDHWC")
    DHW_dilations = _conv2d3d_strides_or_dilations(
        "dilations", node.attr.get("dilations"), data_format
    )
    DHW_strides = _conv2d3d_strides_or_dilations(
        "strides", node.attr.get("strides"), data_format
    )
    pad_type = node.attr.get("padding", None)

    if pad_type is None:
        raise ValueError("Padding type not specified for op: {}".format(node.name))

    if not isinstance(pad_type, str):
        pad_type = "custom"
        raise NotImplementedError("Custom padding not implemented for TF")
    pad_type = pad_type.lower()

    if data_format == "NDHWC":
        # Convert input to NCDHW
        x = _transpose_NDHWC_to_NCDHW(x)
        if output_shape is not None:
            output_shape = [output_shape[0], output_shape[4],
                            output_shape[1], output_shape[2], output_shape[3]]

    # Only the last op should have the same name as node.name
    conv_name = node.name + "_x" if data_format == "NDHWC" else node.name
    # Pass output shape provided above
    x = mb.conv_transpose(
        x=x,
        weight=W_iodhw,
        pad_type=pad_type,
        strides=DHW_strides,
        output_shape=output_shape,
        dilations=DHW_dilations,
        name=conv_name,
    )
    if data_format == "NDHWC":
        # Convert input back to NDHWC (from NCDHW)
        x = _transpose_NCDHW_to_NDHWC(x, node.name)
    context.add(node.name, x)


@register_tf_op
def DepthToSpace(context, node):
    x = context[node.inputs[0]]
    block_size = node.attr.get("block_size")
    data_format = node.attr.get("data_format", "NHWC")
    if data_format == "NHWC":
        x = _transpose_NHWC_to_NCHW(x)
        x = mb.depth_to_space(x=x, block_size=block_size)
        x = _transpose_NCHW_to_NHWC(x, node.name)
    else:
        x = mb.depth_to_space(x=x, block_size=block_size, name=node.name)
    context.add(node.name, x)


@register_tf_op
def EuclideanNorm(context, node):
    x = context[node.inputs[0]]
    axes = _check_axes_type(context[node.inputs[1]])
    keep_dims = node.attr.get("keep_dims", False)
    x = mb.reduce_l2_norm(x=x, axes=axes, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)

@register_tf_op
def IdentityN(context, node):
    res = [mb.identity(x=context[x]) for x in node.inputs]
    context.add(node.name, res)


@register_tf_op
def ExpandDims(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
    if axis.op.op_type == "const" and (axis.val is not None and axis.val.size == 1):
        axis = axis.val[0] if axis.shape == (1,) else axis.val
    else:
        raise ValueError("Expand Dims: Invalid value for parameter axis")
    x = mb.expand_dims(x=x, axes=[axis], name=node.name)
    context.add(node.name, x)


@register_tf_op(tf_alias=["FusedBatchNormV2", "FusedBatchNormV3"])
def FusedBatchNorm(context, node):
    # Get attributes
    data_format = node.attr.get("data_format", "NHWC")
    epsilon = node.attr.get("epsilon", None)

    # Get inputs
    x = context[node.inputs[0]]
    scale = context[node.inputs[1]]
    offset = context[node.inputs[2]]
    mean = context[node.inputs[3]]
    variance = context[node.inputs[4]]
    if data_format == "NHWC":
        # TF's FusedBatchNorm is only for 4D inputs
        x = _transpose_NHWC_to_NCHW(x)
        x = mb.batch_norm(
            x=x, mean=mean, variance=variance, gamma=scale, beta=offset, epsilon=epsilon
        )
        x = _transpose_NCHW_to_NHWC(x, node.name + ":0")
    else:
        x = mb.batch_norm(
            x=x,
            mean=mean,
            variance=variance,
            gamma=scale,
            beta=offset,
            epsilon=epsilon,
            name=node.name + ":0",
        )
    # Inference only batch norm does not have meaningful outputs for
    # batch_mean, batch_variance etc.
    context.add(node.name, [x, mean, variance])


@register_tf_op
def Fill(context, node):
    shape = context[node.inputs[0]]
    value = context[node.inputs[1]]
    x = mb.fill(shape=shape, value=value, name=node.name)
    context.add(node.name, x)


@register_tf_op(tf_alias=["ImageProjectiveTransformV3"])
def ImageProjectiveTransformV2(context, node):
    # Data shape format: [batch, height, width, channels]
    x = context[node.inputs[0]]
    # Transforms shape format: [batch, 8] or [1, 8] matrix, [a0, a1, a2, b0, b1, b2, c0, c1]
    transforms = context[node.inputs[1]]
    # 1-D Tensor [new_height, new_width]
    output_shape = context[node.inputs[2]]

    # For V3, there is an additional fill_value input
    if len(node.inputs) == 4:
        fill_value = context[node.inputs[3]].val
        if fill_value != 0.0:
            msg = ("fill_value {} not supported for tf ImageProjectiveTransformV2/V3 op {}. "
                   "Only fill_value = 0.0 is supported.").format(fill_value, node.name)
            raise ValueError(msg)

    interpolation = node.attr.get("interpolation")
    if interpolation != "BILINEAR":
        msg = ("interpolation {} not supported for tf ImageProjectiveTransformV2/V3 op {}. "
               "Only interpolation = BILINEAR is supported.").format(interpolation, node.name)
        raise ValueError(msg)

    fill_mode = node.attr.get("fill_mode")
    if fill_mode != "CONSTANT":
        msg = ("fill_mode {} not supported for tf ImageProjectiveTransformV2/V3 op {}. "
               "Only fill_mode = CONSTANT is supported.").format(fill_mode, node.name)
        raise ValueError(msg)

    h_out = output_shape.val[0]
    w_out = output_shape.val[1]
    h_in = x.shape[1]
    w_in = x.shape[2]

    # Don't allow non-zero c0 or c1, check for each batch
    n_batch = transforms.val.shape[0]
    transform_matrix = []
    for batch in range(n_batch):
        c0 = transforms.val[batch][6]
        c1 = transforms.val[batch][7]
        if not (c0 == c1 == 0.0):
            raise NotImplementedError(
                "'affine' op with 'transforms' contains non-zero " +
                "c0 or c1 is not supported, Got: {}".format(
                    transforms
                )
            )
        # In the tensorflow affine transform function, the coordinate is in the original image size range,
        # i.e., for the input image, x is in range [0, W_in), and y is in range [0, H_in)
        # For the output image, x is in range [0, W_out), and y is in range [0, H_out)
        # However, the MIL affine op is in the normalized coordinate, in which x and y are both in range [-1, 1]
        # So we need to update the affine transformation matrix.
        # We have the following four equations:
        # (1) x_original_in = (2 * x_normalized_in + 1) * (W_in - 1)
        # (2) y_original_in = (2 * y_normalized_in + 1) * (H_in - 1)
        # (3) x_original_out = (2 * x_normalized_out + 1) * (W_out - 1)
        # (4) y_original_out = (2 * y_normalized_out + 1) * (H_out - 1)
        # The original transforms matrix is in the original coordinate:
        # (i)  x_original_in = a * x_original_out + b * y_original_out + c
        # (ii) y_original_in = d * x_original_out + e * y_original_out + f
        # After plugging (1) - (4) into (i) (ii), we could have the new transformation matrix in the normalized coordinate
        a, b, c, d, e, f = transforms.val[batch].tolist()[:6]
        new_a = a * (w_out - 1) / (w_in - 1)
        new_b = b * (h_out - 1) / (w_in - 1)
        new_c = (2 * c + a * (w_out - 1) + b * (h_out - 1)) / (w_in - 1) - 1
        new_d = d * (w_out - 1) / (h_in - 1)
        new_e = e * (h_out - 1) / (h_in - 1)
        new_f = (2 * f + d * (w_out - 1) + e * (h_out - 1)) / (h_in - 1) - 1
        transform_matrix.append([new_a, new_b, new_c, new_d, new_e, new_f])

    transform_matrix = _np.array(transform_matrix)

    x = _transpose_NHWC_to_NCHW(x)
    x = mb.affine(
        x=x,
        transform_matrix=transform_matrix,
        output_height=output_shape.val[0],
        output_width=output_shape.val[1],
        sampling_mode="bilinear",
        padding_mode="constant",
        padding_value=0.0,
        coordinates_mode="normalized_minus_one_to_one",
        align_corners=True,
        name=node.name + "_affine",
    )
    x = _transpose_NCHW_to_NHWC(x, node.name)
    context.add(node.name, x)


@register_tf_op(tf_alias=["DivNoNan"])
def RealDiv(context, node):
    x = mb.cast(x=context[node.inputs[0]], dtype="fp32")
    y = mb.cast(x=context[node.inputs[1]], dtype="fp32")
    x = mb.real_div(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op(tf_alias=["Addons>Resampler"])
def Resampler(context, node):
    # Data shape format: (Batch, Hin, Win, C)
    x = context[node.inputs[0]]
    # Warp shape format: (Batch, Hout, Wout, 2)
    warp = context[node.inputs[1]]

    # Handle rank-3 warp tensor
    is_rank3_warp = warp.rank == 3
    if is_rank3_warp:  # expand spatial dimension
        warp = mb.expand_dims(x=warp, axes=[1], name=warp.name + "_expand_dims")

    x = _transpose_NHWC_to_NCHW(x)
    x = mb.resample(
        x=x,
        coordinates=warp,
        sampling_mode="bilinear",
        padding_mode="constant",
        padding_value=0.0,
        coordinates_mode="unnormalized",
        align_corners=True,
        name=node.name + "_resample",
    )
    x = _transpose_NCHW_to_NHWC(
        x, node.name + "_transpose" if is_rank3_warp else node.name
    )
    if is_rank3_warp:  # squeeze spatial dimension
        x = mb.squeeze(x=x, axes=[1], name=node.name)

    context.add(node.name, x)


@register_tf_op
def Rsqrt(context, node):
    x = context[node.inputs[0]]
    x = mb.rsqrt(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Sub(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = mb.sub(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def StopGradient(context, node):
    Identity(context, node)


@register_tf_op
def Identity(context, node):
    x = context[node.inputs[0]]
    # In many cases we can skip and just make downstream ops reference the
    # pre-identity op. However, when identity is an output or pre-identity
    # is a placeholder, an identity op, or mb.mul(x, 1.0) is required.
    if len(node.outputs) != 0 or x.op is not None:
        context.add(node.name, x, is_new_var=False)
    else:
        x = mb.mul(x=x, y=1.0, name=node.name)
        context.add(node.name, x)


@register_tf_op
def Print(context, node):
    Identity(context, node)


@register_tf_op
def Placeholder(context, node):
    # no-op as we add Placeholder separately.
    pass


def _pool_pads_or_strides(tf_spec, data_format, d_rank):
    if tf_spec is None:
        d_spec = [1] * d_rank
    elif not isinstance(tf_spec, list):
        d_spec = [tf_spec] * d_rank
    elif len(tf_spec) == 2:
        d_spec = tf_spec
    elif len(tf_spec) == 4:
        if data_format == "NHWC":
            d_spec = tf_spec[1:3]
        else:
            d_spec = tf_spec[2:]
    elif len(tf_spec) == 5:
        if data_format == "NDHWC":
            d_spec = tf_spec[1:4]
        else:
            # NCDHW
            d_spec = tf_spec[2:]
    else:
        raise ValueError("Unsupported tf_spec: %s" % tf_spec)
    return d_spec


@register_tf_op(tf_alias=["BatchMatMul", "BatchMatMulV2"])
def MatMul(context, node):
    a = context[node.inputs[0]]
    b = context[node.inputs[1]]
    transpose_a = node.attr.get("adj_x", False) or node.attr.get("transpose_a", False)
    transpose_b = node.attr.get("adj_y", False) or node.attr.get("transpose_b", False)
    a, b = promote_input_dtypes([a, b])
    x = mb.matmul(
        x=a, y=b, transpose_x=transpose_a, transpose_y=transpose_b, name=node.name
    )
    context.add(node.name, x)


@register_tf_op
def MaxPool(context, node):
    x = context[node.inputs[0]]
    in_shape = x.sym_type.get_shape()
    d_rank = len(in_shape) - 2
    data_format = node.attr.get("data_format", "NHWC")
    ksize = node.attr.get("ksize", None)
    kernel_sizes = _pool_pads_or_strides(ksize, data_format, d_rank)
    strides = node.attr.get("strides", None)
    if strides is not None:
        strides = _pool_pads_or_strides(strides, data_format, d_rank)
    pad_type = node.attr["padding"].lower()
    if data_format == "NHWC":
        x = _transpose_NHWC_to_NCHW(x)
        x = mb.max_pool(
            x=x, kernel_sizes=kernel_sizes, strides=strides, pad_type=pad_type
        )
        x = _transpose_NCHW_to_NHWC(x, node.name)
    else:
        x = mb.max_pool(
            x=x,
            kernel_sizes=kernel_sizes,
            strides=strides,
            pad_type=pad_type,
            name=node.name,
        )
    context.add(node.name, x)


@register_tf_op
def MaxPool3D(context, node):
    x = context[node.inputs[0]]
    d_rank = x.rank - 2
    data_format = node.attr.get("data_format", "NDHWC")
    ksize = node.attr.get("ksize", None)
    kernel_sizes = _pool_pads_or_strides(ksize, data_format, d_rank)
    strides = node.attr.get("strides", None)
    if strides is not None:
        strides = _pool_pads_or_strides(strides, data_format, d_rank)
    pad_type = node.attr["padding"].lower()
    if data_format == "NDHWC":
        x = _transpose_NDHWC_to_NCDHW(x)
        x = mb.max_pool(
            x=x, kernel_sizes=kernel_sizes, strides=strides, pad_type=pad_type
        )
        x = _transpose_NCDHW_to_NDHWC(x, node.name)
    else:
        x = mb.max_pool(
            x=x,
            kernel_sizes=kernel_sizes,
            strides=strides,
            pad_type=pad_type,
            name=node.name,
        )

    context.add(node.name, x)


@register_tf_op
def MatrixBandPart(context, node):
    x = context[node.inputs[0]]
    lower = context[node.inputs[1]]
    upper = context[node.inputs[2]]
    x = mb.band_part(x=x, lower=lower, upper=upper, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Max(context, node):
    x = context[node.inputs[0]]
    axes = _check_axes_type(context[node.inputs[1]])
    keep_dims = node.attr.get("keep_dims", False)
    x = mb.reduce_max(x=x, axes=axes, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Min(context, node):
    x = context[node.inputs[0]]
    axes = _check_axes_type(context[node.inputs[1]])
    keep_dims = node.attr.get("keep_dims", False)
    x = mb.reduce_min(x=x, axes=axes, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Prod(context, node):
    x = context[node.inputs[0]]
    axes = _check_axes_type(context[node.inputs[1]])
    keep_dims = node.attr.get("keep_dims", False)
    x = mb.reduce_prod(x=x, axes=axes, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Cast(context, node):
    type_map = {
        types.fp16: "fp16",
        types.float: "fp32",
        types.double: "fp32",
        types.int32: "int32",
        types.int64: "int32",
    }
    if node.attr["DstT"] not in type_map.keys():
        raise NotImplementedError(
            "Cast: Provided destination type {} not "
            "supported.".format(types.get_type_info(node.attr["DstT"]))
        )
    x = context[node.inputs[0]]
    dtype = type_map[node.attr["DstT"]]
    x = mb.cast(x=x, dtype=dtype, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Round(context, node):
    x = context[node.inputs[0]]
    x = mb.round(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Sign(context, node):
    x = context[node.inputs[0]]
    x = mb.sign(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Sin(context, node):
    x = context[node.inputs[0]]
    x = mb.sin(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Sinh(context, node):
    x = context[node.inputs[0]]
    x = mb.sinh(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Slice(context, node):
    x = context[node.inputs[0]]
    begin = context[node.inputs[1]]
    size = context[node.inputs[2]]
    res = mb.slice_by_size(x=x, begin=begin, size=size, name=node.name)
    context.add(node.name, res)


@register_tf_op
def Sqrt(context, node):
    x = context[node.inputs[0]]
    x = mb.sqrt(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Square(context, node):
    x = context[node.inputs[0]]
    x = mb.mul(x=x, y=x, name=node.name)
    context.add(node.name, x)


def _softmax_cross_entropy_with_logits(feats, labels, name):
    # compute the log softmax
    y = mb.reduce_log_sum_exp(x=feats, axes=[-1], keep_dims=True)
    log_softmax = mb.sub(x=feats, y=y)
    loss = mb.mul(x=labels, y=log_softmax)
    loss = mb.mul(x=loss, y=-1.)
    loss = mb.reduce_sum(x=loss, axes=[-1], name=name)
    return loss


@register_tf_op
def SparseSoftmaxCrossEntropyWithLogits(context, node):
    feats = context[node.inputs[0]]
    labels = context[node.inputs[1]]
    class_nums = feats.shape[1]
    labels = mb.one_hot(
        indices=labels,
        one_hot_vector_size=class_nums,
    )
    labels = mb.cast(x=labels, dtype="fp32")
    loss = _softmax_cross_entropy_with_logits(feats, labels, node.name)
    context.add(node.name, loss)


@register_tf_op
def SoftmaxCrossEntropyWithLogits(context, node):
    feats = context[node.inputs[0]]
    labels = context[node.inputs[1]]
    loss = _softmax_cross_entropy_with_logits(feats, labels, node.name)
    context.add(node.name, loss)


@register_tf_op
def StridedSlice(context, node):
    x = context[node.inputs[0]]
    begin = context[node.inputs[1]]
    end = context[node.inputs[2]]
    stride = context[node.inputs[3]]

    def bitmask_to_array(bit):
        if bit < 0:
            arr = _np.binary_repr(bit, width=8)[::-1]
            arr = [bool(int(x)) for x in list(arr)]
            if node.attr.get("ellipsis_mask", 0) != 0:
                # In case of non-zero ellipsis_mask, we compute the output rank to be the
                # max rank of all the masks. This doesn't work if we computed a mask of constant
                # width 8 here (since the max rank is then taken to be 8 wrongly).
                raise ValueError("Cannot figure out slice rank with negative mask values and " \
                    "non-zero ellipsis_mask")
        else:
            # This method prevents unnecessary padding of the bitmask when it is not negative.
            # It can be padded with any extra False values later, based on output rank.
            arr = []
            while bit > 0:
                if bit & 1:
                    arr.append(True)
                else:
                    arr.append(False)
                bit >>= 1

        return arr

    begin_mask = bitmask_to_array(node.attr.get("begin_mask", 0))
    end_mask = bitmask_to_array(node.attr.get("end_mask", 0))
    squeeze_mask = bitmask_to_array(node.attr.get("shrink_axis_mask", 0))
    ellipsis_mask = bitmask_to_array(node.attr.get("ellipsis_mask", 0))
    new_axis_mask = bitmask_to_array(node.attr.get("new_axis_mask", 0))

    def _pad_mask(
        x,
        begin,
        end,
        stride,
        begin_mask,
        end_mask,
        squeeze_mask,
        ellipsis_mask,
        new_axis_mask,
    ):
        # This function pad the masks, stride, begin and end to the same rank as the input tensor.
        if begin.rank != 1:
            raise ValueError(
                "begin should be 1-D tensor, got {}-D tensor instead".format(begin.rank)
            )
        if end.rank != 1:
            raise ValueError(
                "end should be 1-D tensor, got {}-D tensor instead".format(end.rank)
            )

        # check if inputs can be determined
        begin_cache = begin
        end_cache = end
        begin = [] if begin.val is None else begin.val.tolist()
        end = [] if end.val is None else end.val.tolist()
        stride = [] if stride is None else stride.val.tolist()

        # pad masks function
        new_dims = sum(i is True for i in new_axis_mask)
        if new_dims > 0:
            x_rank = x.rank + new_dims
        else:
            x_rank = x.rank

        def pad_array(arr, max_rank, idx, default_value):
            """
            This function pads the arr to x_rank with default_value.
            idx is the index where ellipis_mask = True.
            max_rank is the maximum rank of the masks, stride, begin and end.
            """
            mask = arr[:]
            mask += [default_value] * (x_rank - len(mask))
            new_mask = []

            for i in range(max_rank):
                num = 1 if i != idx else x_rank - max_rank + 1
                new_mask += [mask[i]] * num
            return new_mask

        mask_list = [
            begin_mask,
            end_mask,
            squeeze_mask,
            ellipsis_mask,
            new_axis_mask,
            stride,
            begin,
            end,
        ]
        max_rank = max([len(arr) for arr in mask_list])

        # If ellipsis_mask is given, the last element of it would be True
        # Otherwise, we simply pad each mask by appending default value
        if ellipsis_mask != []:
            rank = max_rank
            idx = len(ellipsis_mask) - 1
        else:
            rank = x_rank
            idx = -1

        begin_mask = pad_array(begin_mask, rank, idx, False)
        end_mask = pad_array(end_mask, rank, idx, False)
        squeeze_mask = pad_array(squeeze_mask, rank, idx, False)
        ellipsis_mask = pad_array(ellipsis_mask, rank, idx, False)
        new_axis_mask = pad_array(new_axis_mask, rank, idx, False)
        stride = pad_array(stride, rank, idx, 1)

        # pad begin and end if they are determined during compile time
        if begin != []:
            begin = pad_array(begin, rank, idx, 0)
        if end != []:
            end = pad_array(end, rank, idx, 0)

        # make sure begin_mask, end_mask, and stride are consistent with ellipsis mask
        # begin_mask and end_mask should be True, and stride should be 1.
        for i, mask in enumerate(ellipsis_mask):
            if mask:
                begin_mask[i] = True
                end_mask[i] = True
                stride[i] = 1

        # make sure begin_mask, end_mask, and stride are consistent with new axis mask
        # begin_mask and end_mask should be True, and stride should be 1.
        for i, mask in enumerate(new_axis_mask):
            if mask:
                begin_mask[i] = True
                end_mask[i] = True
                stride[i] = 1

        # convert begin and end back to cache value if they are run-time determined
        if begin == []:
            begin = begin_cache

        if end == []:
            end = end_cache

        # check which mask is adding by our default value
        # This happens when the given index is less than the tensor rank,
        # for instance, indexing a 3D tensor A with A[:1, :1] is equivalent to
        # A[:1, :1, :]. In this case we should append True to begin_mask and end_mask
        if ellipsis_mask == [False] * x_rank:
            for i in range(max_rank, x_rank):
                begin_mask[i] = True
                end_mask[i] = True

        return begin, end, stride, begin_mask, end_mask, squeeze_mask, new_axis_mask

    begin, end, stride, begin_mask, end_mask, squeeze_mask, new_axis_mask = _pad_mask(
        x,
        begin,
        end,
        stride,
        begin_mask,
        end_mask,
        squeeze_mask,
        ellipsis_mask,
        new_axis_mask,
    )

    if sum(i is True for i in new_axis_mask) > 0:
        axes = [i for i, val in enumerate(new_axis_mask) if val is True]
        x = mb.expand_dims(x=x, axes=axes, name=node.name + "_new_axes")

    x = mb.slice_by_index(
        x=x,
        name=node.name,
        begin=begin,
        end=end,
        stride=stride,
        begin_mask=begin_mask,
        end_mask=end_mask,
        squeeze_mask=squeeze_mask,
    )

    context.add(node.name, x)


@register_tf_op
def Sum(context, node):
    x = context[node.inputs[0]]
    axes = _check_axes_type(context[node.inputs[1]])
    keep_dims = node.attr.get("keep_dims", False)
    input_type = x.sym_type
    if _is_scalar(input_type):
        context.add(node.name, x, is_new_var=False)
    else:
        x = mb.reduce_sum(x=x, axes=axes, keep_dims=keep_dims, name=node.name)
        context.add(node.name, x)


@register_tf_op
def Tan(context, node):
    x = context[node.inputs[0]]
    x = mb.tan(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def get_tuple(context, node):
    x = context[node.inputs[0]]
    if not isinstance(x, (list, tuple)):
        # In some rare cases, the upstream op produces a single output
        x = [x]
    idx = node.attr["index"]
    if idx >= len(x):
        msg = "Index {} out of range, op '{}' only has {} outputs: {}"
        raise IndexError(msg.format(idx, node.inputs[0], len(x), [v.name for v in x]))
    context.add(node.name, x[idx], is_new_var=False)


@register_tf_op
def Mean(context, node):
    x = context[node.inputs[0]]
    axes = _check_axes_type(context[node.inputs[1]])
    keep_dims = node.attr.get("keep_dims", False)
    x = mb.reduce_mean(x=x, axes=axes, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)


@register_tf_op
def MatrixDiag(context, node):
    x = context[node.inputs[0]]
    if x.rank != 1:
        raise NotImplementedError('Only support MatrixDiag op with input rank = 1.')
    length = mb.shape(x=x)
    x = mb.expand_dims(x=x, axes=[0])
    reps = mb.concat(values=[length, [1]], axis=0)
    x = mb.tile(x=x, reps=reps)
    x = mb.band_part(x=x, lower=0, upper=0, name=node.name)
    context.add(node.name, x)


@register_tf_op
def MirrorPad(context, node):
    x = context[node.inputs[0]]
    pad = context[node.inputs[1]]
    constant_val = node.attr.get("constant_val", 0.0)

    if pad is None:
        raise ValueError("TF `paddings` in Pad op must be const.")

    mode = node.attr.get("mode", "reflect").lower()
    if mode == "symmetric":
        mode = "reflect"
    in_rank = len(x.sym_type.get_shape())

    if in_rank > 5 or in_rank < 2:
        raise ValueError(
            "Unsupported Pad configuration with input rank {}!".format(str(in_rank))
        )

    if pad.val.shape != (in_rank, 2):
        raise ValueError("Padding must have length as input tensor rank.")

    pad = pad.val

    # get axix which is non zero
    non_zero_axis = []
    for i in range(len(pad)):
        if not all(pad[i] == 0):
            non_zero_axis.append(i)

    if len(non_zero_axis) > 2:
        raise ValueError("Unsupported configuration for Pad layer!")

    # make padding a 2 x 2 tensor if len(non_zero_axis) < 2
    if len(non_zero_axis) == 0:
        non_zero_axis = [0, 1]

    if len(non_zero_axis) == 1:
        if non_zero_axis[0] != len(pad) - 1:
            non_zero_axis.append(len(pad) - 1)
        else:
            non_zero_axis = [0, non_zero_axis[0]]

    # transpose the input such that the padding dim is the last two
    perm = [i for i in range(in_rank) if i not in non_zero_axis] + non_zero_axis
    x = mb.transpose(x=x, perm=perm, name=node.name + "_transpose_1")
    pad = pad[non_zero_axis, :]
    pad = pad.reshape(-1)
    x = mb.pad(
        x=x, pad=pad, name=node.name + "_pad", constant_val=constant_val, mode=mode
    )
    inverse_perm = [-1] * len(perm)
    for i, index in enumerate(perm):
        inverse_perm[index] = i
    x = mb.transpose(x=x, perm=inverse_perm, name=node.name)

    context.add(node.name, x)


@register_tf_op
def Pad(context, node):
    x = context[node.inputs[0]]
    pad = context[node.inputs[1]]
    input_dtype = x.dtype

    mode = node.attr.get("mode", "constant").lower()
    if mode == "symmetric":
        mode = "reflect"
    constant_val = node.attr.get("constant_val", 0.0)
    constant_val = mb.const(val=constant_val)
    in_rank = len(x.sym_type.get_shape())

    if in_rank > 5:
        raise ValueError("Unsupported Pad configuration!")

    if pad.val is None:
        pad = mb.reshape(x=pad, shape=[-1])
    else:
        pad = pad.val.reshape(-1)

    x = mb.cast(x=x, dtype=builtin_to_string(constant_val.dtype))
    x = mb.pad(x=x, pad=pad, mode=mode, constant_val=constant_val)
    x = mb.cast(x=x, dtype=builtin_to_string(input_dtype), name=node.name)

    context.add(node.name, x)


@register_tf_op
def PadV2(context, node):
    # compared to tf.raw_ops.Pad, tf.raw_ops.PadV2 allow constant values rather than 0.
    x = context[node.inputs[0]]
    pad = context[node.inputs[1]]
    constant_val = context[node.inputs[2]]

    if constant_val.shape != ():
        raise NotImplementedError(
            "TF `constant_values` in PadV2 op must be const scalar."
        )
    in_rank = x.rank
    if in_rank > 5:
        raise ValueError("Unsupported Pad configuration!")

    if pad.val is None:
        pad = mb.reshape(x=pad, shape=[-1])
    else:
        pad = pad.val.reshape(-1)

    constant_val = constant_val.val
    if constant_val == -_np.inf:
        INT_MIN = -_np.iinfo(_np.int64).max - 1
        constant_val = float(INT_MIN)

    if constant_val == _np.inf:
        INT_MAX = _np.iinfo(_np.int64).max
        constant_val = float(INT_MAX)

    x = mb.pad(x=x, pad=pad, name=node.name, mode="constant", constant_val=constant_val)
    context.add(node.name, x)


@register_tf_op
def Relu(context, node):
    x = context[node.inputs[0]]
    x = mb.relu(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Reciprocal(context, node):
    x = context[node.inputs[0]]
    x = mb.inverse(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Relu6(context, node):
    x = context[node.inputs[0]]
    x = mb.relu6(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Reshape(context, node):
    x = context[node.inputs[0]]
    new_shape = context[node.inputs[1]]
    x = mb.reshape(x=x, shape=new_shape, name=node.name)
    context.add(node.name, x)


@register_tf_op(tf_alias=["ReverseV2"])
def Reverse(context, node):
    x = context[node.inputs[0]]
    axes = context[node.inputs[1]]
    x = mb.reverse(x=x, axes=axes, name=node.name)
    context.add(node.name, x)


@register_tf_op
def ReverseSequence(context, node):
    x = context[node.inputs[0]]
    lengths = context[node.inputs[1]]
    seq_axis = node.attr.get("seq_dim")
    batch_axis = node.attr.get("batch_dim")
    x = mb.reverse_sequence(
        x=x, lengths=lengths, seq_axis=seq_axis, batch_axis=batch_axis, name=node.name
    )
    context.add(node.name, x)


@register_tf_op
def Transpose(context, node):
    x = context[node.inputs[0]]
    perm = context[node.inputs[1]]
    x = mb.transpose(x=x, perm=perm, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Squeeze(context, node):
    x = context[node.inputs[0]]
    axes = node.attr.get("squeeze_dims", [])
    if axes == []:
        axes = None
    x = mb.squeeze(x=x, axes=axes, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Multinomial(context, node):
    x = context[node.inputs[0]]
    size = context[node.inputs[1]]
    x = mb.random_categorical(x=x, size=size, name=node.name)
    context.add(node.name, x)


@register_tf_op(tf_alias=["Elu"])
def ELU(context, node):
    x = context[node.inputs[0]]
    x = mb.elu(x=x, alpha=1.0, name=node.name)
    context.add(node.name, x)


@register_tf_op(tf_alias=["Erf"])
def ERF(context, node):
    x = context[node.inputs[0]]
    x = mb.erf(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op(tf_alias=["LeakyRelu"])
def LeakyReLU(context, node):
    x = context[node.inputs[0]]
    alpha = node.attr["alpha"]
    x = mb.leaky_relu(x=x, alpha=alpha, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Selu(context, node):
    x = context[node.inputs[0]]
    x = mb.elu(x=x, alpha=1.6732632423543772)
    x = mb.mul(x=x, y=1.0507009873554805, name=node.name)
    context.add(node.name, x)


@register_tf_op(tf_alias=["SelectV2"])
def Select(context, node):
    cond = context[node.inputs[0]]
    a = context[node.inputs[1]]
    b = context[node.inputs[2]]

    # broadcast vector type cond
    rank_cond = cond.rank
    rank_a = a.rank
    if rank_cond == 1 and rank_a > 1:
        axes = [-i - 1 for i in range(rank_a - rank_cond)]
        cond = mb.expand_dims(x=cond, axes=axes)

    if not types.is_bool(cond.dtype):
        # cond must be bool type
        cond = mb.cast(x=cond, dtype="bool")

    x = mb.select(cond=cond, a=a, b=b, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Sigmoid(context, node):
    x = context[node.inputs[0]]
    x = mb.sigmoid(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Softplus(context, node):
    x = context[node.inputs[0]]
    x = mb.softplus(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Softsign(context, node):
    x = context[node.inputs[0]]
    x = mb.softsign(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Softmax(context, node):
    logit = context[node.inputs[0]]
    axis = node.attr.get("axis")
    x = mb.softmax(x=logit, axis=axis, name=node.name)
    context.add(node.name, x)


@register_tf_op
def SpaceToBatchND(context, node):
    # In tensorflow, the input tensor has the shape of (batch,) + spatial_shape + remaining_shape.
    # The shape is treated as a combination of 3 components:
    # 1. A single batch dimension
    # 2. Spatial dimensions, with a length spatial_rank, which could be neither 1 or 2. Also, spatial_rank
    #    is equal to the length of block_shape
    # 3. Remaining dimensions, with a length remaining_rank

    # The logic of translating this op is as followed:
    # 1. We first reshape the input to a canonical shape (rolling the remaining shape dimensions into a
    #    single dimension): (batch,) + spatial_shape + (R), where R = remaining_dim_1 * ... * remaining_dim_n
    # 2. We support rank 1 and rank 2 spatial shape:
    #    (i) rank 1: We decompose the SpaceToBatch into small basic ops.
    #    (ii) rank 2: We directly use the built in space_to_batch op.
    #    The output would have shape (batch_new,) + spatial_shape_new + (R)
    # 3. We transform the tensor back, by unrolling the remaining shape: (B_new,) + spatial_shape_new + remaining_shape

    x = context[node.inputs[0]]
    block_shape = context[node.inputs[1]].val
    paddings = context[node.inputs[2]]
    original_shape = mb.shape(x=x)

    input_rank = x.rank
    spatial_rank = len(block_shape)
    remaining_rank = x.rank - 1 - spatial_rank
    has_non_unity_remaining_dims = remaining_rank != 1

    if block_shape is None:
        raise NotImplementedError("Not support dynamic block_shape for SpaceToBatchND!")

    if paddings.val is not None:
        is_static_paddings = True
        paddings = paddings.val
    else:
        is_static_paddings = False

    if has_non_unity_remaining_dims:
        # Reshape the input tensor to shape [batch, spatial_shape, remaining_dim_1 * ... * remaining_dim_N]
        x = _reshape_remaining_dimensions_to_canonical_shape(x, remaining_rank)

    if spatial_rank >= 3:
        raise NotImplementedError("Rank of spatial shape > 2 is not supported.")

    if spatial_rank == 2:
        # Tensor has shape [B, H, W, C], we can directly use the space_to_batch op by doing
        # [B, H, W, C] -> transpose -> [B, C, H, W] -> space_to_batch -> [B_new, C, H_new, W_new] ->
        # transpose -> [B_new, H_new, W_new, C]
        x = mb.transpose(x=x, perm=[0, 3, 1, 2])
        needs_paddings = not is_static_paddings or (
            tuple(paddings[0]) != (0, 0) or tuple(paddings[1]) != (0, 0)
        )
        if needs_paddings:
            flatten_paddings = mb.reshape(
                x=paddings,
                shape=[
                    4,
                ],
            )
            flatten_paddings = mb.cast(x=flatten_paddings, dtype="int32")
            flatten_paddings = mb.concat(values=[[0, 0, 0, 0], flatten_paddings], axis=0)
            x = mb.pad(x=x, pad=flatten_paddings, mode="constant")

        x = mb.space_to_batch(x=x, block_shape=block_shape, paddings=_np.zeros((2, 2), _np.int32))
        x = mb.transpose(x=x, perm=[0, 2, 3, 1])

    if spatial_rank == 1:
        # In this case, we decompose space_to_batch into small basic ops
        # [B, H, C] -> decomposite ops -> [B_new, H_new, C]

        # expand padding to shape [3, 2]
        paddings = mb.cast(x=paddings, dtype="int32")
        values = [[[0, 0]], paddings, [[0, 0]]]
        paddings = mb.concat(values=values, axis=0)
        needs_paddings = not is_static_paddings or any(paddings.val.flatten())

        if needs_paddings:
            flatten_paddings = mb.reshape(x=paddings, shape=[-1])
            padded = mb.pad(x=x, pad=flatten_paddings, mode="constant")
            x = padded
        else:
            padded = x

        # padded_shape = [B, H_padded, C]
        padded_shape = mb.shape(x=padded)

        # reshape to [B, H_padded/block_shape, block_shape, C]
        block_shape = block_shape[0]
        batch_size = _value_at(padded_shape, 0)
        spatial_dim = mb.real_div(x=_value_at(padded_shape, 1), y=block_shape)
        spatial_dim = mb.cast(x=spatial_dim, dtype="int32")
        remain_dim = _value_at(padded_shape, 2)
        reshape_shape = mb.concat(values=[batch_size, spatial_dim, block_shape, remain_dim], axis=0)
        reshaped_padded = mb.reshape(x=padded, shape=reshape_shape)

        # permute the shape to: [block_shape, B, H_padded/block_shape, C]
        permuted_reshaped_padded = mb.transpose(x=reshaped_padded, perm=[2, 0, 1, 3])

        # reshape the tensor to [block_shape * B, H_padded/block_shape, C]
        final_reshape_values = [mb.mul(x=batch_size, y=block_shape), spatial_dim, remain_dim]
        final_shape = mb.concat(values=final_reshape_values, axis=0)
        x = mb.reshape(x=permuted_reshaped_padded, shape=final_shape)

    if has_non_unity_remaining_dims:
        # Reshape the tensor from shape [batch_new, spatial_shape_new, remaining_dim_1 * ... * remaining_dim_N] back to
        # shape [batch_new, spatial_shape_new, remaining_shape]
        x = _reshape_remaining_dimension_to_original_shape(x, original_shape, remaining_rank)

    context.add(node.name, mb.identity(x=x, name=node.name))


@register_tf_op
def SpaceToDepth(context, node):
    x = context[node.inputs[0]]
    block_size = node.attr.get("block_size")
    data_format = node.attr.get("data_format", "NHWC")
    if data_format == "NHWC":
        x = _transpose_NHWC_to_NCHW(x)
        x = mb.space_to_depth(x=x, block_size=block_size)
        x = _transpose_NCHW_to_NHWC(x, node.name)
    else:
        x = mb.space_to_depth(x=x, block_size=block_size, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Tanh(context, node):
    x = context[node.inputs[0]]
    x = mb.tanh(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op(tf_alias=["TopKV2"])
def TopK(context, node):
    x = context[node.inputs[0]]
    k = context[node.inputs[1]].val
    sort = node.attr["sorted"]

    kwargs = {
        "x": x,
        "k": k,
        "axis": -1,
        "name": node.name
    }

    if is_current_opset_version_compatible_with(target.iOS16):
        kwargs["sort"] = sort
    elif not sort:
        raise ValueError("For opset <= iOS16, only sorted=True supported for the topk")

    context.add(node.name, mb.topk(**kwargs))

@register_tf_op(tf_alias=["InTopKV2"])
def InTopK(context, node):
    x = context[node.inputs[0]]
    target = context[node.inputs[1]]
    k = context[node.inputs[2]].val

    _, class_num = x.shape
    if not is_symbolic(class_num):
        k = min(k, class_num)

    _, indices = mb.topk(x=x, k=k, axis=-1)
    target = mb.expand_dims(x=target, axes=[-1])
    x = mb.equal(x=target, y=indices)
    x = mb.cast(x=x, dtype="fp32")
    x = mb.reduce_sum(x=x, axes=[-1], keep_dims=False)
    x = mb.cast(x=x, dtype="bool", name=node.name)
    context.add(node.name, x)


@register_tf_op
def Cumsum(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
    exclusive = node.attr.get("exclusive", False)
    reverse = node.attr.get("reverse", False)
    x = mb.cumsum(x=x, axis=axis, exclusive=exclusive, reverse=reverse, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Gather(context, node):
    x = context[node.inputs[0]]
    indices = context[node.inputs[1]]
    axis = 0
    x = mb.gather(x=x, indices=indices, axis=axis, name=node.name)
    context.add(node.name, x)


def _perform_gather_with_batch_dims(x, indices, batch_dims, gather_func, func_args, name):
    """
    An utility function to compute gather and gather_nd with batch_dims
    """
    # (Step 1)
    # Reshape x, indices with shape
    # x: [batch_1, ..., batch_n, *remaining_x_shape]
    # indices: [batch_1, ..., batch_n, *remaing_indices_shape]
    # into shape
    # x_reshape: [prod(batch_1, ..., batch_n), *remaning_x_shape]
    # indices_reshape: [prod(batch_1, ..., batch_n), *remaning_indices_shape]
    msg = ("The implementation of gather/gather_nd for iOS15 and older is not efficient. Highly recommend "
           " set minimum_deployment_target=coremltools.target.iOS16 in the coremltools.convert() function."
    )
    logger.warning(msg)
    x_shape = mb.shape(x=x)
    indices_shape = mb.shape(x=indices)
    batch_shape = mb.gather(x=x_shape, indices=_np.array(range(batch_dims)), axis=0)
    batch_prod = mb.reduce_prod(x=batch_shape, axes=[0], keep_dims=True)
    x_remaining_shape = mb.gather(x=x_shape, indices=_np.array(range(batch_dims, x.rank)), axis=0)
    indices_remaining_shape = mb.gather(x=indices_shape, indices=_np.array(range(batch_dims, indices.rank)), axis=0)
    new_x_shape = mb.concat(values=[batch_prod, x_remaining_shape], axis=0)
    new_indices_shape = mb.concat(values=[batch_prod, indices_remaining_shape], axis=0)
    x_reshape = mb.reshape(x=x, shape=new_x_shape)
    indices_reshape = mb.reshape(x=indices, shape=new_indices_shape)

    # (Step 2)
    # We iterate through the batch dimension, and compute the gather individually for each batch
    # All results are stacked into a tensor with shape [prod(batch_1, ..., batch_n), *remaning_result_shape]
    res = []
    if batch_prod.val is None:
        raise ValueError("batch dimenstion must be known at compile time")
    for i in range(batch_prod.val[0]):
        temp_x = mb.gather(x=x_reshape, indices=[i], axis=0)
        temp_indices = mb.gather(x=indices_reshape, indices=[i], axis=0)
        temp_x = mb.squeeze(x=temp_x, axes=[0])
        temp_indices = mb.squeeze(x=temp_indices, axes=[0])
        func_args.update({"x": temp_x, "indices": temp_indices})
        temp = gather_func(**func_args)
        res.append(temp)
    res = mb.stack(values=res, axis=0)

    # (Step 3)
    # Finally, we reshape the result to shape [batch_1, ..., batch_n, *remaining_result_shape]
    res_shape = mb.shape(x=res)
    res_remaning_shape = mb.gather(x=res_shape, indices=_np.array(range(1, res_shape.shape[0])), axis=0)
    res_new_shape = mb.concat(values=[batch_shape, res_remaning_shape], axis=0)
    return mb.reshape(x=res, shape=res_new_shape, name=name)


@register_tf_op
def GatherV2(context, node):
    x = context[node.inputs[0]]
    indices = context[node.inputs[1]]
    axis = context[node.inputs[2]].val
    batch_dims = node.attr.get("batch_dims", 0)
    if is_current_opset_version_compatible_with(target.iOS16):
        # For iOS16 and above, we can directly use the batch_dims argument
        x = mb.gather(x=x, indices=indices, axis=axis, batch_dims=batch_dims, name=node.name)
    else:
        # For iOS15 or below, we have to manually compute it
        if batch_dims == 0:
            x = mb.gather(x=x, indices=indices, axis=axis, name=node.name)
        else:
            func_args = {"axis": axis - batch_dims}
            x = _perform_gather_with_batch_dims(x, indices, batch_dims, mb.gather, func_args, node.name)

    context.add(node.name, x)


@register_tf_op
def GatherNd(context, node):
    x = context[node.inputs[0]]
    indices = context[node.inputs[1]]
    batch_dims = node.attr.get("batch_dims", 0)
    if is_current_opset_version_compatible_with(target.iOS16):
        # For iOS16 and above, we can directly use the batch_dims argument
        x = mb.gather_nd(x=x, indices=indices, batch_dims=batch_dims, name=node.name)
    else:
        if batch_dims == 0:
            x = mb.gather_nd(x=x, indices=indices, name=node.name)
        else:
            x = _perform_gather_with_batch_dims(x, indices, batch_dims, mb.gather_nd, {}, node.name)

    context.add(node.name, x)


@register_tf_op
def Tile(context, node):
    x = context[node.inputs[0]]
    reps = context[node.inputs[1]]
    x = mb.tile(x=x, reps=reps, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Where(context, node):
    if len(node.inputs) > 1:
        raise NotImplementedError('tf.where with x,y will be supported by '
                                  'MIL::select in the future')
    x = context[node.inputs[0]]
    x = mb.non_zero(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def SquaredDifference(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = mb.sub(x=x, y=y, name=node.name + '_sub')
    x = mb.square(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Conv2DBackpropInput(context, node):
    # Output shape: [N, H_out, W_out, C_out]
    output_shape = context[node.inputs[0]].val
    # Weight shape: [H, W, C_out, C_in]
    W_hwoi = context[node.inputs[1]]
    W_iohw = mb.transpose(x=W_hwoi, perm=[3, 2, 0, 1])
    # Input shape: [N, H_in, W_in, C_in]
    x = context[node.inputs[2]]

    data_format = node.attr.get("data_format", "NHWC")
    HW_dilations = _conv2d3d_strides_or_dilations(
        "dilations", node.attr.get("dilations"), data_format
    )
    HW_strides = _conv2d3d_strides_or_dilations(
        "strides", node.attr.get("strides"), data_format
    )
    pad_type = node.attr.get("padding")

    if not isinstance(pad_type, str):
        pad_type = "custom"
        raise NotImplementedError("Custom padding not implemented for TF")

    pad_type = pad_type.lower()
    # CoreML expects input to be in NCHW format
    # Transpose input to NCHW format
    if data_format == "NHWC":
        x = _transpose_NHWC_to_NCHW(x)
        if output_shape is not None:
            output_shape = [output_shape[0], output_shape[3],
                            output_shape[1], output_shape[2]]

    # Only the last op should have the same name as node.name
    conv_name = node.name + "x" if data_format == "NHWC" else node.name
    # Pass output shape provided above
    x = mb.conv_transpose(
        x=x,
        weight=W_iohw,
        pad_type=pad_type,
        output_shape=output_shape,
        strides=HW_strides,
        dilations=HW_dilations,
        name=conv_name,
    )

    # Convert NCHW output back to NHWC format
    if data_format == "NHWC":
        x = _transpose_NCHW_to_NHWC(x, node.name)
    context.add(node.name, x)


@register_tf_op
def Range(context, node):
    start = context[node.inputs[0]]
    end = context[node.inputs[1]]
    step = context[node.inputs[2]]
    x = mb.range_1d(start=start, end=end, step=step, name=node.name)
    context.add(node.name, x)


@register_tf_op
def RandomUniform(context, node):
    shape = context[node.inputs[0]]
    seed = node.attr["seed"]
    x = mb.random_uniform(shape=shape, seed=seed, name=node.name)
    context.add(node.name, x)


@register_tf_op
def RandomStandardNormal(context, node):
    shape = context[node.inputs[0]]
    seed = node.attr["seed"]
    x = mb.random_normal(shape=shape, seed=seed, name=node.name)
    context.add(node.name, x)


@register_tf_op
def OneHot(context, node):
    indices = context[node.inputs[0]]
    depth = context[node.inputs[1]]
    on_value = context[node.inputs[2]]
    off_value = context[node.inputs[3]]
    axis = node.attr.get("axis", -1)
    x = mb.one_hot(
        indices=indices,
        one_hot_vector_size=depth,
        axis=axis,
        on_value=on_value,
        off_value=off_value,
        name=node.name,
    )
    context.add(node.name, x)


def _get_non_maximum_supression(context, node, iou_threshold_override=None, score_threshold_override=None):
    """
    The helper function returns the outputs from mb.non_maximum_suppression,
    along with the number of boxes and the maximum number of boxes.
    """
    boxes = context[node.inputs[0]]
    scores = context[node.inputs[1]]
    max_boxes = context[node.inputs[2]]
    iou_threshold = iou_threshold_override or context[node.inputs[3]]
    score_threshold = score_threshold_override or context[node.inputs[4]]

    # The boxes' coordinates in Tensorflow is (y1, x1, y2, x2) where (y1, x1) and (y2, x2) are the
    # coordinates of diagonal pair of box corners. However, MIL NMS expects CENTER_SIZE_WIDTH_FIRST
    # format, which is (x, y, width, height) where (x, y) is the center coordinate.
    y1, x1, y2, x2 = mb.split(x=boxes, num_splits=4, axis=-1)
    # As the input coordinates could be any diagonal pair of box corners, it's not guaranteed that
    # x2 > x1 nor y2 > y1. So we need to use abs to get width/height, and (x1+x2)/2 to get center.
    width = mb.abs(x=mb.sub(x=x2, y=x1))
    height = mb.abs(x=mb.sub(x=y2, y=y1))
    center_x = mb.real_div(x=mb.add(x=x1, y=x2), y=2.0)
    center_y = mb.real_div(x=mb.add(x=y1, y=y2), y=2.0)
    boxes = mb.concat(values=[center_x, center_y, width, height], axis=-1)

    if score_threshold.val == float("-inf"):
        # TensorFlow's default value for score_threshold, Core ML does not
        # have float('-inf') support, converted to minimum float32 instead
        score_threshold = -3.4e38

    boxes = mb.expand_dims(x=boxes, axes=[0])
    scores = mb.expand_dims(x=scores, axes=[0, -1])
    coordinates, scores, indices, valid_outputs = mb.non_maximum_suppression(
        boxes=boxes,
        scores=scores,
        max_boxes=max_boxes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )

    # The results from MIL NMS op are padded to max_boxes. We need to extract the valid part for TF.
    # Notice that the batch dim and class num dim also need to be squeezed.
    valid_outputs = mb.squeeze(x=valid_outputs, axes=[0])
    range = mb.range_1d(end=valid_outputs, start=0, step=1)
    coordinates = mb.squeeze(x=coordinates, axes=[0])
    valid_coordinates = mb.gather(x=coordinates, indices=range, axis=0)
    scores = mb.squeeze(x=scores, axes=[0, -1])
    valid_scores = mb.gather(x=scores, indices=range, axis=0)
    indices = mb.squeeze(x=indices, axes=[0])
    valid_indices = mb.cast(
        x=mb.gather(x=mb.cast(x=indices, dtype="fp32"), indices=range, axis=0),
        dtype="int32",
        name=node.name,
    )

    return valid_coordinates, valid_scores, valid_indices, valid_outputs


@register_tf_op(tf_alias=["NonMaxSuppressionV3"])
def NonMaxSuppression(context, node):
    _, _, valid_indices, valid_outputs = _get_non_maximum_supression(context, node)
    context.add(node.name, valid_indices)


@register_tf_op
def NonMaxSuppressionV5(context, node):
    """
    Different from NonMaxSuppression/NonMaxSuppressionV3, which only returns the indices of the selected boxes,
    NonMaxSuppressionV5 returns all indices, scores and number of the selected boxes.
    """
    soft_nms_sigma = context[node.inputs[5]].val
    iou_threshold_override = None
    score_threshold_override = None
    if soft_nms_sigma != 0:
        # fallback to "hard" NMS with sensible defaults
        iou_threshold_override = types.fp32(0.5)
        score_threshold_override = types.fp32(float("-inf"))
        logger.warning("NonMaxSuppressionV5 with soft_nms_sigma != 0 not supported. "
                       "Setting soft_nms_sigma to zero.")

    _, valid_scores, valid_indices, valid_outputs = _get_non_maximum_supression(
        context, node, iou_threshold_override=iou_threshold_override, score_threshold_override=score_threshold_override
    )
    res = [valid_indices, valid_scores, valid_outputs]
    context.add(node.name, res)


@register_tf_op
def Shape(context, node):
    x = context[node.inputs[0]]
    if types.is_complex(x.dtype):
        x = mb.complex_shape(x=x, name=node.name)
    else:
        x = mb.shape(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def ResizeNearestNeighbor(context, node):
    # "ResizeNearestNeighbor" op in TF is always in the channel last mode
    # instead of upsample factor, it uses output size, which is the second input
    x = context[node.inputs[0]]

    input_shape = x.shape  # (N,Hin,Win,C)
    if len(input_shape) != 4:
        raise ValueError('"ResizeNearestNeighbor" op: input rank is not 4')

    if len(context[node.inputs[1]].shape) != 1:
        raise ValueError('"ResizeNearestNeighbor" op: the second input, must have rank 1')

    if context[node.inputs[1]].shape[0] != 2:
        raise ValueError(
            '"ResizeNearestNeighbor" op: the second input, which is the output size, must have 2 elements'
        )
    Hout, Wout = None, None
    scaling_factor_h, scaling_factor_w = None, None
    target_shape = context[node.inputs[1]]
    if target_shape.val is None:
        if target_shape.op is not None and target_shape.op.op_type == "mul":
            scaling_factor_h = target_shape.op.y.val[0]
            scaling_factor_w = target_shape.op.y.val[1]
        elif not is_current_opset_version_compatible_with(target.iOS17):
            # For the dynamic input shape case before iOS17,
            # context[node.inputs[1]] need to be a mul(x=input_shape, y=scaling_factor) op.
            raise ValueError(
                "Cannot determine the scale factor for the resize layer. "
                "Please make sure the target size is known statically, or "
                "use mul op to get the target size. If the target size has to be dynamic, please"
                "set minimum_deployment_target to iOS17 during conversion."
            )
    else:
        Hin, Win = input_shape[1], input_shape[2]
        Hout, Wout = target_shape.val
        scaling_factor_h = Hout / Hin if Hout % Hin == 0 else (Hout + 1e-4) / Hin
        scaling_factor_w = Wout / Win if Wout % Win == 0 else (Wout + 1e-4) / Win

    if (
        scaling_factor_h is not None
        and scaling_factor_w is not None
        and scaling_factor_h < 1
        and scaling_factor_w < 1
    ):
        ResizeBilinear(context, node)
        return

    # first transpose to from channel last to channel first format for coreml
    x = _transpose_NHWC_to_NCHW(x)

    align_corners = node.attr.get("align_corners", False)
    half_pixel_centers = node.attr.get("half_pixel_centers", False)

    # add either the resize or the upsample layer
    if align_corners is False and half_pixel_centers is False:
        x = mb.upsample_nearest_neighbor(
            x=x,
            scale_factor_height=scaling_factor_h,
            scale_factor_width=scaling_factor_w,
            name=node.name + "_channel_first_upsample",
        )
    elif align_corners is False and half_pixel_centers is True:
        # if output size can be determined at compile time,
        # we call the core op resize_nearest_neighbor,
        # otherwise we use upsample_nearest_neighbor for approximation.
        # rdar://75204549 (resize_nearest_neighbor need to support dynamic input shape)
        if Hout is not None and Wout is not None:
            x = mb.resize_nearest_neighbor(
                x=x,
                target_size_height=Hout,
                target_size_width=Wout,
                name=node.name + "_channel_first_resize",
            )
        elif is_current_opset_version_compatible_with(target.iOS17):
            x = mb.resize(
                x=x,
                shape=target_shape,
                resized_dims=np.uint32(2),
                interpolation_mode="NEAREST_NEIGHBOR",
                name=node.name + "_channel_first_resize",
            )
        else:
            logger.warning('Using upsample_nearest_neighbor to approximate resize_nearest_neighbor.')
            x = mb.upsample_nearest_neighbor(
                x=x,
                scale_factor_height=scaling_factor_h,
                scale_factor_width=scaling_factor_w,
                name=node.name + "_channel_first_upsample",
            )

    else:
        raise NotImplementedError(
            "ResizeNearestNeighbor op with align_corners={}and half_pixel_centers={} not supported".format(
                align_corners, half_pixel_centers
            )
        )

    # transpose again
    x = _transpose_NCHW_to_NHWC(x, node.name)
    context.add(node.name, x)


@register_tf_op
def ResizeBilinear(context, node):
    # "ResizeBilinear" op in TF is always in the channel last mode
    # second input is the output size

    x = context[node.inputs[0]]
    input_shape = x.shape  # (N,Hin,Win,C)
    if len(input_shape) != 4:
        raise ValueError('"ResizeBilinear" op: input rank is not 4')

    if len(context[node.inputs[1]].shape) != 1:
        raise ValueError('"ResizeBilinear" op: the second input, must have rank 1')

    if context[node.inputs[1]].shape[0] != 2:
        raise ValueError(
            '"ResizeBilinear" op: the second input, which is the output size, must have 2 elements'
        )

    align_corners = node.attr.get("align_corners", False)
    half_pixel_centers = node.attr.get("half_pixel_centers", False)

    if align_corners and half_pixel_centers:
        # we should not come here since TF does not support align_corners=True and half_pixel_centers=True
        raise ValueError(
            '"ResizeBilinear" op: "align_corners" and "half_pixel_centers" are both True and this mode is not supported'
        )

    # In iOS16, we can support dynamic shape + any combination of aligh_corners and half_pixel_centers,
    # if the output_shape comes from a pattern of input_shape * (h_scale, w_scale)
    if is_current_opset_version_compatible_with(target.iOS16) and context[node.inputs[1]].val is None:
        output_shape = context[node.inputs[1]]
        if output_shape.op is not None and output_shape.op.op_type == "mul":
            scale_factor_height = context[node.inputs[1]].op.y.val[0]
            scale_factor_width = context[node.inputs[1]].op.y.val[1]
            x = _transpose_NHWC_to_NCHW(x)
            x = mb.upsample_bilinear(
                x=x,
                scale_factor_height=scale_factor_height,
                scale_factor_width=scale_factor_width,
                align_corners=align_corners,
                half_pixel_centers=half_pixel_centers,
            )
            x = _transpose_NCHW_to_NHWC(x, node.name)
            context.add(node.name, x)
            return

    # first transpose to from channel last to channel first format for coreml
    x = _transpose_NHWC_to_NCHW(x)

    # [half_pixel_centers = False]
    if not half_pixel_centers:
        sampling_mode = "STRICT_ALIGN_CORNERS" if align_corners else "DEFAULT"
        node_name = node.name + "_channel_first_resize_bilinear"
        target_size = context[node.inputs[1]]

        if target_size.val is not None:
            Hout, Wout = target_size.val
            if not (
                isinstance(Hout, (_np.int32, _np.int64))
                and isinstance(Wout, (_np.int32, _np.int64))
            ):
                raise ValueError(
                    '"ResizeBilinear" op: the second input, which is the output size, must have elements of type int32 or int64'
                )
            x = mb.resize_bilinear(
                x=x,
                target_size_height=Hout,
                target_size_width=Wout,
                sampling_mode=sampling_mode,
                name=node_name,
            )
        elif is_current_opset_version_compatible_with(target.iOS17):
            x = mb.resize(
                x=x,
                shape=target_size,
                resized_dims=np.uint32(2),
                sampling_mode=sampling_mode,
                name=node_name,
            )
        else:
            raise ValueError(
                '"ResizeBilinear" op: the second input, which is the output size, must be known '
                "statically. Consider setting minimum_deployment_target to iOS17 during conversion."
            )

    # [align_corners = False, half_pixel_centers = True]
    elif not align_corners and half_pixel_centers:
        if context[node.inputs[1]].val is None:
            # for the dynamic input shape case,
            # context[node.inputs[1]] is a mul(x=input_shape, y=scaling_factor) op.
            if context[node.inputs[1]].op.op_type != "mul":
                raise NotImplementedError("Cannot determine the scale factor for the bilinear resize layer.")
            scale_factor_height = context[node.inputs[1]].op.y.val[0]
            scale_factor_width = context[node.inputs[1]].op.y.val[1]
        else:
            Hin, Win = input_shape[1], input_shape[2]
            Hout, Wout = context[node.inputs[1]].val
            # check if the output size divide the input size,
            # if not, then cast the scale factor to float type.
            scale_factor_height = Hout / Hin if Hout % Hin == 0 else (Hout + 1e-4) / Hin
            scale_factor_width = Wout / Win if Wout % Win == 0 else (Wout + 1e-4) / Win

        x = mb.upsample_bilinear(
            x=x,
            scale_factor_height=scale_factor_height,
            scale_factor_width=scale_factor_width,
            align_corners=False,
            name=node.name + "_channel_first_upsample_bilinear",
        )

    # transpose again
    x = _transpose_NCHW_to_NHWC(x, node.name)
    context.add(node.name, x)


@register_tf_op
def make_tuple(context, node):
    res = tuple([context[in_name] for in_name in node.inputs])
    context.add(node.name, res)


@register_tf_op
def function_entry(context, node):
    if context.get_func_inputs() is None:
        msg = (
            "function_entry requires function inputs stored in "
            + "context.curr_func_inputs"
        )
        raise ValueError(msg)
    context.add(node.name, context.get_func_inputs())


@register_tf_op(tf_alias=["while"])
def While(context, node):
    # TF while will never have break statement, because break can always be
    # transformed into while and condition. Example:
    #
    #   while pred:
    #    a = op1(...)
    #    if a == 0:
    #      break
    #    b = op2(...)
    #
    # is equivalent to
    #
    #   while pred and not break_a:
    #    a = op1(...)
    #    break_a = a == 0
    #    if not break_a:
    #      b = op2(...)

    # node.inputs[0] == 'make_tuple_X' (always a make_tuple)
    loop_vars = context[node.inputs[0]]  # python tuple of Vars
    cond_graph = context.get_graph(node.attr["cond_function"])
    body_graph = context.get_graph(node.attr["body_function"])

    def cond(*loop_vars):
        context.stack_func_inputs(loop_vars)

        # convert_graph uses context to convert cond_graph. During conversion
        # it constructs operations (mb.some_op). Note that cond(*loop_vars) is
        # only evaluated inside while_loop's type_inference(), not here. In
        # other words, we use python's deferred function evaluation to defer
        # the SSA block construction until inside while_loop Operation.
        res = convert_graph(context, cond_graph)
        # Done with translating the function
        context.unstack_func_inputs()
        return res

    def body(*loop_vars):
        context.stack_func_inputs(loop_vars)
        res = convert_graph(context, body_graph)
        # Done with translating the function
        context.unstack_func_inputs()
        return res

    x = mb.while_loop(_cond=cond, _body=body, loop_vars=loop_vars, name=node.name)
    # wraps x as tuple for get_tuple that always follow the while node.
    if not isinstance(x, (tuple, list)):
        x = (x,)
    context.add(node.name, x)


@register_tf_op
def iff(context, node):
    pred = context[node.inputs[0]]

    # this is always a tensor, as TF uses one iff op for each returned value.
    #
    # Example TF program:
    #
    #  x = tf.placeholder(tf.float32, shape=(1,))
    #  y = tf.placeholder(tf.float32, shape=(1,))
    #  z = tf.multiply(x, y)
    #  pred = tf.less(tf.math.reduce_mean(x), tf.math.reduce_mean(y))
    #  def true_fn(): return tf.add(x, z), x
    #  def false_fn(): return tf.square(y), z
    #  res = tf.cond(pred, true_fn, false_fn)
    #
    # There will be 2 iffs:
    #
    # iff('cond/pred_id', 'cond/Add', 'cond/Square')
    # iff('cond/pred_id', 'cond/Add/Switch', 'cond/Switch_1')
    #
    # where
    #   'cond/pred_id': pred
    #   'cond/Add': tf.add(x, z)
    #   'cond/Square': tf.square(y)
    #   'cond/Add/Switch': x
    #   'cond/Switch_1': z
    #
    # And both branches are executed, and one of the results will be
    # discarded at iff nodes.
    #
    # Note that the above program would translate to two cond ops, each with
    # two blocks.
    true_output_var = context[node.inputs[1]]
    false_output_var = context[node.inputs[2]]

    def true_fn():
        return mb.identity(x=true_output_var)

    def false_fn():
        return mb.identity(x=false_output_var)

    x = mb.cond(pred=pred, _true_fn=true_fn, _false_fn=false_fn, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Concat(context, node):
    values = [context[input] for input in node.inputs[1:]]
    axis = context[node.inputs[0]]
    x = mb.concat(values=values, axis=axis, name=node.name)
    context.add(node.name, x)


@register_tf_op
def ConcatV2(context, node):
    values = [context[input] for input in node.inputs[:-1]]
    axis = context[node.inputs[-1]]
    x = mb.concat(values=values, axis=axis, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Pack(context, node):
    values = [context[name] for name in node.inputs]
    axis = node.attr["axis"]
    if axis < 0:
        # TF axis = -1 creates new dim at the end
        axis += values[0].rank + 1
    if len(values) == 1:
        # for example:
        # y = tf.raw_ops.Pack(values=[2], axis=0).
        # or y = tf.raw_ops.Pack(values=[tf.constant([1,2])], axis=0)
        input_type = values[0].sym_type
        if _is_scalar(input_type):
            x = mb.mul(x=_np.array([1], dtype=_np.int32), y=values[0], name=node.name)
        else:
            x = mb.expand_dims(x=values[0], axes=[axis], name=node.name)
    else:
        if all([_is_scalar(input.sym_type) for input in values]):
            x = mb.concat(values=values, axis=axis, name=node.name)
        else:
            x = mb.stack(values=values, axis=axis, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Unpack(context, node):
    x = context[node.inputs[0]]
    axis = int(node.attr["axis"])
    num_splits = node.attr.get("num", None)
    if num_splits is None:
        num_splits = x.shape[axis]
    if num_splits == 1:
        y = [x]
    else:
        y = mb.split(x=x, num_splits=num_splits, axis=axis, name=node.name + "_unsqueezed")
    output_vars = []
    for i in range(num_splits):
        output_vars.append(
            mb.squeeze(x=y[i], axes=[axis], name=node.name + ":{}".format(i))
        )

    context.add(node.name, output_vars)


@register_tf_op
def Split(context, node):
    axis = context[node.inputs[0]]
    x = context[node.inputs[1]]
    if "num_split" not in node.attr:
        raise ValueError("num_splits not found in TF op {}".format(node.name))
    num_splits = node.attr["num_split"]
    if num_splits == 1:
        if len(node.outputs) == 0:
            x = mb.mul(x=x, y=1.0, name=node.name)
            context.add(node.name, x)
        else:
            # Don't change tfssa. Just make downstream ops reference the pre-identity op.
            context.add(node.name, [x], is_new_var=False)
    else:
        x = mb.split(x=x, num_splits=num_splits, axis=axis, name=node.name)
        context.add(node.name, x)
        # TODO : If tf.split output is returned, there's no
        # get_tuple nodes. Some graph pass is needed. Example:
        #
        #    x = tf.placeholder(tf.float32, shape=input_shape1)
        #    res = tf.split(x, 3, axis=0)
        #
        # res are ['split:0', 'split:1', 'split']
        #
        # but node.outputs == ['gto_1', 'gto_2', 'gto_3']


@register_tf_op
def SplitV(context, node):
    x = context[node.inputs[0]]
    split_sizes = context[node.inputs[1]]
    axis = context[node.inputs[2]]
    if "num_split" not in node.attr:
        raise ValueError("num_splits not found in TF op {}".format(node.name))
    num_splits = node.attr["num_split"]
    if num_splits == 1:
        Identity(context, node)
    else:
        x = mb.split(
            x=x,
            num_splits=num_splits,
            split_sizes=split_sizes,
            axis=axis,
            name=node.name,
        )
        context.add(node.name, x)


@register_tf_op
def ScatterNd(context, node):
    indices = context[node.inputs[0]]
    updates = context[node.inputs[1]]
    shape = context[node.inputs[2]]
    x = mb.fill(shape=shape, value=types.nptype_from_builtin(updates.dtype)(0))
    x = mb.scatter_nd(data=x, indices=indices, updates=updates, name=node.name)
    context.add(node.name, x)


@register_tf_op
def TensorScatterAdd(context, node):
    tensor, indices, updates, = [context[name] for name in node.inputs]
    output = mb.scatter_nd(data=tensor, indices=indices, updates=updates, mode="add", name=node.name)
    context.add(node.name, output)


@register_tf_op
def ZerosLike(context, node):
    x = context[node.inputs[0]]
    if x.rank == 0:
        np_type = types.nptype_from_builtin(x.sym_type)
        x = mb.const(val=np_type(0), name=node.name)
    else:
        np_type = types.nptype_from_builtin(x.sym_type.get_primitive())
        x = mb.fill(shape=mb.shape(x=x), value=np_type(0), name=node.name)
    context.add(node.name, x)


@register_tf_op
def IsFinite(context, node):
    x = context[node.inputs[0]]

    # In floating-point arithmetic, symbolically, inf + anything = inf,
    # so we can detect if x is finite by x + y != x
    #
    # To avoid false alarm, i.e. x + y = x due to rounding error for small y,
    # here we use the fp16 max as y
    dtype = types.nptype_from_builtin(x.sym_type.get_primitive())
    y_add = dtype(_np.finfo(_np.float16).max)
    x_plus = mb.add(x=x, y=y_add)
    result = mb.not_equal(x=x, y=x_plus, name=node.name)

    context.add(node.name, result)


@register_tf_op
def CropAndResize(context, node):
    x = context[node.inputs[0]]
    input_shape = x.shape  # (B, h_in, w_in, C)
    if len(input_shape) != 4:
        raise ValueError(
            '"CropResize" op: expected input rank 4, got {}'.format(x.rank)
        )
    Hin, Win = input_shape[1:3]

    const_box_info = True
    if context[node.inputs[1]].val is None or context[node.inputs[2]].val is None:
        const_box_info = False

    crop_size = context[node.inputs[3]].val
    method = node.attr.get("method", "bilinear")
    pad_value = node.attr.get("extrapolation_value", 0.0)

    # CoreML index information along with boxes
    if const_box_info:
        boxes = context[node.inputs[1]].val
        box_indices = context[node.inputs[2]].val
        if not is_current_opset_version_compatible_with(target.iOS17):
            # Before IOS17, CoreML expects boxes/ROI in [N, 1, 5, 1, 1] shape.
            box_indices = _np.expand_dims(box_indices, axis=1)
            boxes = _np.concatenate([box_indices, boxes], axis=1)
            boxes = boxes.reshape(boxes.shape[0], 1, boxes.shape[1], 1, 1)
    else:
        box_indices = context[node.inputs[2]]
        boxes = context[node.inputs[1]]
        if not is_current_opset_version_compatible_with(target.iOS17):
            # Before IOS17, CoreML expects ROI in [N, 1, 5, 1, 1] shape.
            if box_indices.dtype != boxes.dtype:
                box_indices = mb.cast(x=box_indices, dtype=types.builtin_to_string(boxes.dtype))
            box_indices = mb.expand_dims(x=box_indices, axes=[1])
            boxes = mb.concat(values=(box_indices, boxes), axis=1)
            # TODO: Dynamic rank: Use GetShape and select indices dynamically
            boxes = mb.reshape(x=boxes, shape=[boxes.shape[0], 1, boxes.shape[1], 1, 1])

    # Get Height and Width of crop
    h_out, w_out = crop_size[0], crop_size[1]

    # TF `nearest` mode not supported
    method_map = {"bilinear": "ALIGN_CORNERS"}
    if method not in method_map:
        raise ValueError(
            "CropResize op: Unsupported method {}. Supports {}".format(
                method, method_map.keys()
            )
        )
    method = method_map[method]

    # TF input format: [B, h_in, w_in, C]
    # CoreML input format: [B, C, h_in, w_in]
    x = _transpose_NHWC_to_NCHW(x)

    # Crop Resize
    crop_resize_args = {
        "x": x,
        "target_height": h_out,
        "target_width": w_out,
        "normalized_coordinates": True,
        "spatial_scale": 1.0,
        "box_coordinate_mode": "CORNERS_HEIGHT_FIRST",
        "sampling_mode": method,
    }
    if is_current_opset_version_compatible_with(target.iOS16):
        crop_resize_args["pad_value"] = pad_value
    else:
        if pad_value != 0.0:
            raise ValueError(
                f"For iOS15 or older, only extrapolation_value=0.0 is supported or the tf CropAndResize op. Got {pad_value}"
            )
    if not is_current_opset_version_compatible_with(target.iOS17):
        # Before IOS17, the input param is `roi` instead of `boxes`.
        crop_resize_args["roi"] = boxes
    else:
        crop_resize_args["boxes"] = boxes
        crop_resize_args["box_indices"] = box_indices

    x = mb.crop_resize(**crop_resize_args)

    if not is_current_opset_version_compatible_with(target.iOS17):
        # Before IOS17, the output has an extra dim at axis 1.
        # CoreML output format: [N, 1, C, h_out, w_out]
        # TF output format: [N, h_out, w_out, C]
        x = mb.squeeze(x=x, axes=[1])
    x = _transpose_NCHW_to_NHWC(x, node.name)
    context.add(node.name, x)


@register_tf_op
def TensorArrayV3(context, node):
    if "infer_shape" in node.attr:
        if not node.attr["infer_shape"]:
            raise ValueError("Only fixed size TensorArray is supported")

    dynamic_length = node.attr.get("dynamic_size", True)
    elem_shape = node.attr.get("element_shape", None)
    size = node.attr.get("size", None)
    if size is None:
        size = context[node.inputs[0]]

    if size.val is None:
        init_length = size
    else:
        init_length = size.val
        if init_length == 0:
            # Dynamic list. Use 1 as init_length
            init_length = 1

    builtin_dtype = node.attr["dtype"]
    dtype_str = types.builtin_to_string(builtin_dtype)
    if elem_shape is not None and -1 not in elem_shape:
        ls = mb.make_list(
            init_length=init_length,
            dtype=dtype_str,
            elem_shape=elem_shape,
            dynamic_length=dynamic_length,
            name=node.name,
        )
    else:
        ls = mb.tf_make_list(
            init_length=init_length,
            dtype=dtype_str,
            dynamic_length=dynamic_length,
            name=node.name,
        )
    context.add(node.name, ls)


@register_tf_op
def TensorArrayWriteV3(context, node):
    index = context[node.inputs[0]]
    new_val = context[node.inputs[1]]
    ls = context[node.inputs[2]]
    new_list = mb.list_write(ls=ls, index=index, value=new_val, name=node.name)
    context.add(node.name, new_list)


@register_tf_op
def TensorArraySizeV3(context, node):
    ls = context[node.inputs[0]]
    length = mb.list_length(ls=ls, name=node.name)
    context.add(node.name, length)


@register_tf_op
def TensorArrayGatherV3(context, node):
    indices = context[node.inputs[0]]
    ls = context[node.inputs[1]]
    tensor = mb.list_gather(ls=ls, indices=indices, name=node.name)
    context.add(node.name, tensor)


@register_tf_op
def TensorArrayReadV3(context, node):
    idx = context[node.inputs[0]]
    ls = context[node.inputs[1]]
    ls = mb.list_read(ls=ls, index=idx, name=node.name)
    context.add(node.name, ls)


@register_tf_op
def TensorArrayScatterV3(context, node):
    indices = context[node.inputs[0]]
    value = context[node.inputs[1]]
    ls = context[node.inputs[2]]
    ls = mb.list_scatter(ls=ls, indices=indices, value=value, name=node.name)
    context.add(node.name, ls)


@register_tf_op
def BroadcastTo(context, node):
    x = context[node.inputs[0]]
    shape = context[node.inputs[1]]
    if shape.val is None:  # dynamic shape
        raise NotImplementedError("dynamic shape not yet supported")
    else:  # static shape
        target_shape = tuple(shape.val)
        broadcast_shape = broadcast_shapes(x.shape, target_shape)
        if target_shape != broadcast_shape:
            msg = "shapes are not broadcastable: {} vs. {}"
            raise ValueError(msg.format(x.shape, target_shape))
        target_rank = len(target_shape)
        if x.rank != target_rank:
            axes = [i for i in range(target_rank - x.rank)]
            x = mb.expand_dims(x=x, axes=axes)
        reps = [1] * target_rank
        for i in range(target_rank):
            reps[i] = target_shape[i] // x.shape[i]

    x = mb.tile(x=x, reps=reps, name=node.name)
    context.add(node.name, x)


@register_tf_op
def get_global(context, node):
    # Design comment: This is only works if variable doesn't cross block
    # boundary (e.g. while_loop, cond, function)
    variable_name = node.attr["variable"]
    x = context[variable_name]  # This must've been set by set_global
    context.add(node.name, x, is_new_var=False)


@register_tf_op
def set_global(context, node):
    x = context[node.inputs[0]]
    variable_name = node.attr["variable"]
    context.add(variable_name, x, is_new_var=False)


def _get_const_or_raise(variable):
    if variable.val is None:
        raise ValueError("Var {} must be const".format(variable.name))
    return variable.val


@register_tf_op
def LSTMBlockCell(context, node):
    x = context[node.inputs[0]]  # [batch, input_dim]
    c_prev = context[node.inputs[1]]  # [b, hidden_dim]
    h_prev = context[node.inputs[2]]  # [b, hidden_dim]
    # W layout is ifco
    W = context[node.inputs[3]]  # [input_dim + hidden_dim, 4*hidden_dim]

    kwargs = {}
    use_peephole = node.attr["use_peephole"]
    if use_peephole:
        peep_i = context[node.inputs[4]]  # [hidden_dim,]
        peep_f = context[node.inputs[5]]  # [hidden_dim,]
        peep_o = context[node.inputs[6]]  # [hidden_dim,]
        kwargs["weight_peep_i"] = peep_i
        kwargs["weight_peep_f"] = peep_f
        kwargs["weight_peep_o"] = peep_o

    bias = context[node.inputs[7]]  # [4*hidden_dim,]

    forget_bias = node.attr["forget_bias"]
    cell_clip = None
    if node.attr["cell_clip"] is not None and node.attr["cell_clip"] > 0:
        cell_clip = node.attr["cell_clip"]

    res = mb.tf_lstm_block_cell(
        x=x,
        c_prev=c_prev,
        h_prev=h_prev,
        weight=W,
        bias=bias,
        forget_bias=forget_bias,
        cell_clip=cell_clip,
        use_peephole=use_peephole,
        name=node.name,
        **kwargs
    )
    context.add(node.name, res)

@register_tf_op(tf_alias=["BlockLSTMV2"])
def BlockLSTM(context, node):
    # BlockLSTM: https://www.tensorflow.org/api_docs/python/tf/raw_ops/BlockLSTM
    # BlockLSTMV2: https://www.tensorflow.org/api_docs/python/tf/raw_ops/BlockLSTMV2
    seq_len = context[node.inputs[0]]  # int
    x = context[node.inputs[1]]  # [padded_len, batch, input_dim]
    init_c = context[node.inputs[2]]  # [1, hidden_dim]
    init_h = context[node.inputs[3]]  # [1, hidden_dim]
    # BlockLSTM: icfo format, BlockLSTMV2: ifco format
    weight = context[node.inputs[4]]  # [input_dim + hidden_dim, 4*hidden_dim]

    kwargs = {}
    use_peephole = node.attr["use_peephole"]
    if use_peephole:
        peep_i = context[node.inputs[5]]  # [hidden_dim,]
        peep_f = context[node.inputs[6]]  # [hidden_dim,]
        peep_o = context[node.inputs[7]]  # [hidden_dim,]
        kwargs["weight_peep_i"] = peep_i
        kwargs["weight_peep_f"] = peep_f
        kwargs["weight_peep_o"] = peep_o

    # BlockLSTM: icfo format, BlockLSTMV2: ifco format
    bias = context[node.inputs[8]]  # [4*hidden_dim,]

    # forget bias is always 0 for BlockLSTMV2
    forget_bias = 0.0 if node.op == "BlockLSTMV2" else node.attr["forget_bias"]
    cell_clip = None
    if node.attr["cell_clip"] is not None and node.attr["cell_clip"] > 0:
        cell_clip = node.attr["cell_clip"]

    if node.op == "BlockLSTMV2":
        # mb.tf_lstm_block takes weights and bias in icfo format
        # BlockLSTMV2's weights and bias are in ifco format
        # convert from ifco to icfo format
        w_i, w_f, w_c, w_o = mb.split(x=weight, num_splits=4, axis=-1)
        weight = mb.concat(values=(w_i, w_c, w_f, w_o), axis=1, name=weight.name)
        b_i, b_f, b_c, b_o = mb.split(x=bias, num_splits=4, axis=-1)
        bias = mb.concat(values=(b_i, b_c, b_f, b_o), axis=0, name=bias.name)

    res = mb.tf_lstm_block(
        seq_len=seq_len,
        x=x,
        c_prev=init_c,
        h_prev=init_h,
        weight=weight,
        bias=bias,
        forget_bias=forget_bias,
        cell_clip=cell_clip,
        use_peephole=use_peephole,
        name=node.name,
        **kwargs
    )
    context.add(node.name, res)

@register_tf_op
def ClipByValue(context, node):
    x = context[node.inputs[0]]
    min_value = context[node.inputs[1]]
    max_value = context[node.inputs[2]]
    if min_value.val < max_value.val:
        x = mb.clip(x=x, alpha=min_value, beta=max_value, name=node.name)
    else:
        # When min >= max, TensorFlow sets all values to min.
        x = mb.fill(shape=mb.shape(x=x), value=min_value, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Size(context, node):
    x = context[node.inputs[0]]
    x = mb.shape(x=x)
    x = mb.reduce_prod(x=x, axes=[0], name=node.name)
    context.add(node.name, x)

@register_tf_op
def LogSoftmax(context, node):
    x = context[node.inputs[0]]
    axis = node.attr.get('axis', -1)
    x_max = mb.reduce_max(x=x, axes=[axis], keep_dims=True)
    x_off = mb.sub(x=x, y=x_max)
    y = mb.reduce_log_sum_exp(x=x_off, axes=[axis], keep_dims=True)
    res = mb.sub(x=x_off, y=y, name=node.name)
    context.add(node.name, res)

@register_tf_op
def AudioSpectrogram(context, node):
    """
    input shape: (Tin, channels)
    attributes: stride (int), window_size (int), magnitude_squared (bool)

    output shape : (channels, Tout, fout)
    where,
    Tout = floor((Tin - window_size)/stride + 1)
    fout = N / 2 + 1
    where N = next_power_of_2(window_size) = 2 ^ ceil(log2(window_size))

    reference:
    https://github.com/tensorflow/tensorflow/blob/dec8e0b11f4f87693b67e125e67dfbc68d26c205/tensorflow/core/kernels/spectrogram_op.cc
    """

    x = context[node.inputs[0]] # (Tin, channels)
    if x.rank != 2:
        raise NotImplementedError("AudioSpectrogram op: rank of the input must be 2")

    if "magnitude_squared" not in node.attr:
        raise ValueError("AudioSpectrogram op: missing attribute: 'magnitude_squared'")
    if "stride" not in node.attr:
        raise ValueError("AudioSpectrogram op: missing attribute: 'stride'")
    if "window_size" not in node.attr:
        raise ValueError("AudioSpectrogram op: missing attribute: 'window_size'")

    magnitude_squared = node.attr["magnitude_squared"]
    stride = node.attr["stride"]
    window_size = node.attr["window_size"]

    N = 2 ** _np.ceil(_np.log2(window_size))
    N = N.astype(_np.int32)
    fout = N / 2 + 1
    fout = fout.astype(_np.int32)

    # construct constant for hann window tensor, of shape (window_size,)
    h = _np.arange(window_size) * ((2 * _np.pi) / window_size)
    h = 0.5 - 0.5 * _np.cos(h)

    # construct the constant DFT matrices
    k = _np.arange(fout).reshape(1, fout) # (1, fout)
    n = _np.arange(N).reshape(N, 1)  # (N, 1)
    kn = _np.matmul(n, k) * (2 * _np.pi / N) # (N, fout)
    Re_DFT_matrix_const = _np.cos(kn)  # (N, fout)
    Im_DFT_matrix_const = -_np.sin(kn) # (N, fout)

    # transpose input
    x = mb.transpose(x=x, perm=[1,0]) # (channels, Tin)
    # extract slices from the input
    x = mb.sliding_windows(x=x, axis=1, size=window_size, stride=stride) # (channels, Tout, window_size)
    # multiply with hann window
    x = mb.mul(x=x, y=h)
    # pad the last dimension to size N
    x = mb.pad(x=x, pad=[0,0,0,0,0,N - window_size], mode="constant", constant_val=0.0) # (channels, Tout, N)
    # multiply by DFT matrices
    re = mb.matmul(x=x, y=Re_DFT_matrix_const) # (channels, Tout, fout)
    im = mb.matmul(x=x, y=Im_DFT_matrix_const)  # (channels, Tout, fout)

    # compute spectrogram
    re = mb.mul(x=re, y=re)
    im = mb.mul(x=im, y=im)
    if not magnitude_squared:
        y = mb.add(x=re, y=im)
        y = mb.sqrt(x=y, name=node.name)
    else:
        y = mb.add(x=re, y=im, name=node.name)
    context.add(node.name, y)

@register_tf_op
def Mfcc(context, node):
    """
    inputs:
    - x : (channels, T, N)
    - sampling rate: int

    attributes:
    - upper_frequency_limit : int
    - lower_frequency_limit : int
    - filterbank_channel_count : int
    - dct_coefficient_count : int

    output shape: (channels, T, dct_coefficient_count)
    """
    x = context[node.inputs[0]]  # (channels, T, F)
    if x.rank != 3:
        raise NotImplementedError("Mfcc op: rank of the input must be 3")
    sampling_rate_var = context[node.inputs[1]]
    if sampling_rate_var.val is None:
        raise NotImplementedError("Mfcc op: dynamic sampling rate not supported")
    sample_rate = sampling_rate_var.val
    if is_symbolic(x.shape[2]):
        raise NotImplementedError("Mfcc op: the last dimension, i.e. spectrogram size of the input must be known")

    spectrogram_N = x.shape[2]
    upper_frequency_limit = node.attr.get("upper_frequency_limit", 4000)
    lower_frequency_limit = node.attr.get("lower_frequency_limit", 20)
    filterbank_channel_count = node.attr.get("filterbank_channel_count", 40)
    dct_coefficient_count = node.attr.get("dct_coefficient_count", 13)

    # get the constant weights, matrices for MFCC filterbank and for DCT
    # weights: (N,)
    # mat_weighted, mat_spec_val : (N, filterbank_channel_count)
    # cosines : (filterbank_channel_count, dct_coefficient_count)
    weights, mat_weighted, mat_spec_val, cosines = _get_MFCC_constants(spectrogram_N,
                                                                       sample_rate,
                                                                       upper_frequency_limit,
                                                                       lower_frequency_limit,
                                                                       filterbank_channel_count,
                                                                       dct_coefficient_count)

    spectogram_value = mb.sqrt(x=x) # (channels, T, N)
    weighted_spectogram_value = mb.mul(x=spectogram_value, y=weights)  # (channels, T, N)
    x1 = mb.matmul(x=weighted_spectogram_value, y=mat_weighted) # (channels, T, filterbank_channel_count)
    x2 = mb.matmul(x=spectogram_value, y=mat_spec_val) # (channels, T, filterbank_channel_count)
    y = mb.add(x=x1, y=x2) # (channels, T, filterbank_channel_count)
    y = mb.log(x=y, epsilon=1e-12)
    y = mb.matmul(x=y, y=cosines, name=node.name) # (channels, T, dct_coefficient_count)
    context.add(node.name, y)


@register_tf_op
def Complex(context, node):
    real_part = context[node.inputs[0]]
    imag_part = context[node.inputs[1]]
    result = mb.complex(real_data=real_part, imag_data=imag_part, name=node.name)
    context.add(node.name, result)


@register_tf_op
def Real(context, node):
    input_data = context[node.inputs[0]]
    if types.is_complex(input_data.dtype):
        real_part = mb.complex_real(data=input_data, name=node.name)
    else:
        real_part = input_data
    context.add(node.name, real_part)


@register_tf_op
def Imag(context, node):
    input_data = context[node.inputs[0]]
    if types.is_complex(input_data.dtype):
        imag_part = mb.complex_imag(data=input_data, name=node.name)
    else:
        # According to the doc of tf.math.imag, it returns a tensor of all zeros if input is real.
        np_type = types.nptype_from_builtin(input_data.sym_type.get_primitive())
        imag_part = mb.fill(
            shape=mb.shape(x=input_data), value=np_type(0), name=node.name
        )
    context.add(node.name, imag_part)


@register_tf_op
def FFT(context, node):
    input_data = context[node.inputs[0]]
    fft_res = mb.complex_fft(data=input_data, name=node.name)
    context.add(node.name, fft_res)


@register_tf_op
def RFFT(context, node):
    input_data = context[node.inputs[0]]
    fft_length = context[node.inputs[1]]
    # The fft_length is an int32 tensor of shape [1] instead of an integer. To make it compatible
    # to complex_rfft (which use PyTorch's params as reference), we extract the value from tensor.
    rfft_res = mb.complex_rfft(
        data=input_data, n=mb.const(val=fft_length.val[0]), name=node.name
    )
    context.add(node.name, rfft_res)


@register_tf_op
def IFFT(context, node):
    input_data = context[node.inputs[0]]
    ifft_res = mb.complex_ifft(data=input_data, name=node.name)
    context.add(node.name, ifft_res)


@register_tf_op
def IRFFT(context, node):
    input_data = context[node.inputs[0]]
    fft_length = context[node.inputs[1]]
    # The fft_length is an int32 tensor of shape [1] instead of an integer. To make it compatible
    # to complex_rfft (which use PyTorch's params as reference), we extract the value from tensor.
    irfft_res = mb.complex_irfft(
        data=input_data, n=mb.const(val=fft_length.val[0]), name=node.name
    )
    context.add(node.name, irfft_res)
