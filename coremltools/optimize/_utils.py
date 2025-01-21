# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import math
from collections import namedtuple
from typing import Collection, Dict, List, Optional, Tuple, Union

import numpy as np

from coremltools import _getLogger
from coremltools.converters.mil.mil import Operation, types
from coremltools.optimize.coreml._config import (
    CompressionGranularity,
    OpLinearQuantizerConfig,
    OpPalettizerConfig,
)

_logger = _getLogger()

SparseParamsIos16 = namedtuple("SparseParamsIos16", "nonzero_data mask shape")
LutParamsIos16 = namedtuple("LutParamsIos16", "lut indices shape")
QuantParamsIos16 = namedtuple("QuantParamsIos16", "quantized_data zero_point scale axis")

SparseParams = namedtuple("SparseParams", "nonzero_data mask")
LutParams = namedtuple("LutParams", "indices lut vector_axis")
QuantParams = namedtuple("QuantParams", "data scale offset nbits")


def get_quant_range_by_dtype(
    dtype: types, mode: str
) -> Tuple[Union[int, float], Union[int, float]]:
    if types.is_int(dtype):
        nbits = dtype.get_bitwidth()
        signed = not dtype.is_unsigned()
        return get_quant_range(nbits, signed, mode)
    else:
        raise NotImplementedError(
            "Only support getting quant range for int dtype, "
            f"but got {types.builtin_to_string(dtype)}"
        )


def get_quant_range(n_bits: int, signed: bool, mode: str) -> Tuple[int, int]:
    """
    Utility to get the quantization range for a given quantization config
    Adapted from phoenix/quatization/_utils.py
    """
    max_q = 2**n_bits
    if not signed:
        quant_min = 0
        quant_max = max_q - 1
        if mode == "LINEAR_SYMMETRIC":
            quant_max -= 1
    else:
        quant_min = -max_q / 2
        quant_max = max_q / 2 - 1
        if mode == "LINEAR_SYMMETRIC":
            quant_min += 1
    return int(quant_min), int(quant_max)


def quantize_weight(
    weight: np.ndarray,
    axes: Tuple[int, ...],
    nbits: int,
    signed: bool,
    quantization_mode: str,
    dtype: np.dtype,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Get quantized data along with metadata (scale, zero_point)."""
    builtin_dtype = types.get_nbits_int_builtin_type(nbits, signed)
    quantized_data, scale, zero_point = quantize_weight_by_dtype(
        weight, axes, builtin_dtype, quantization_mode
    )
    quantized_data = quantized_data.astype(dtype)
    if zero_point is not None:
        zero_point = zero_point.astype(dtype)
    return quantized_data, scale, zero_point


def quantize_weight_by_dtype(
    weight: np.ndarray,
    axes: Union[int, Tuple[int, ...]],
    dtype: types,
    quantization_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Similar to `quantize_weight, but use `dtype` to specify quant dtype."""
    if not np.issubdtype(weight.dtype, np.floating):
        # In principle, all dtypes are quantizable, e.g. int can be cast to float then quantize
        # In practise, Core ML constexpr dequantization ops return float,
        # so such constexpr can only replace float const
        raise ValueError("Only floating numpy arrays are supported.")

    val_min = np.amin(weight, axis=axes, keepdims=True)
    val_max = np.amax(weight, axis=axes, keepdims=True)
    q_val_min, q_val_max = get_quant_range_by_dtype(dtype, quantization_mode)
    zero_point = None

    if quantization_mode == "LINEAR_SYMMETRIC":
        # For the linear_symmetric quantization_mode, the range is symmetrical to 0
        max_abs = np.maximum(np.abs(val_min), np.abs(val_max))
        val_min = -max_abs
        val_max = max_abs
    else:
        assert quantization_mode == "LINEAR"
        # For the linear quantization_mode, we need to make sure the data range contains `0`
        val_min = np.minimum(0.0, val_min)
        val_max = np.maximum(0.0, val_max)

    scale = (val_max - val_min) / (q_val_max - q_val_min)
    quantized_data = np.round(weight / scale)

    if types.is_int(dtype):
        if quantization_mode == "LINEAR_SYMMETRIC" and dtype.is_unsigned():
            zero_point_shift = q_val_max // 2
            zero_point = zero_point_shift * np.ones(val_min.shape)
        elif quantization_mode == "LINEAR":
            zero_point = (q_val_min * val_max - q_val_max * val_min) / (val_max - val_min)
            zero_point = np.round(zero_point)
            zero_point = np.clip(zero_point, q_val_min, q_val_max)

    if zero_point is not None:
        quantized_data += zero_point
        zero_point = zero_point.squeeze()
    quantized_data = np.clip(quantized_data, q_val_min, q_val_max)
    scale = scale.astype(weight.dtype).squeeze()

    return quantized_data, scale, zero_point


def promote_rank_to_same_as_data(
    param: np.ndarray, data: np.ndarray, axis: Optional[int]
) -> np.ndarray:
    """
    Promote param (i.e. zero point or scale) rank to same as quantized data,
    so subtraction or multiplication can happen properly on the specified axis.

    For example, when `param` has shape [2] while `data` has shape [2, 2, 2],
    - if `axis` is 0, `param` will be promoted to shape [2, 1, 1].
    - if `axis` is 1, `param` will be promoted to shape [1, 2, 1].
    - if `axis` is 2, `param` will be promoted to shape [1, 1, 2].
    """
    if len(param.shape) == len(data.shape):
        return param  # already have the same rank

    if axis is not None:
        axis = axis if axis >= 0 else axis + len(data.shape)

    if len(param.shape) == 0:
        return np.reshape(param, np.ones(len(data.shape), np.int32))
    else:
        axes = [i for i in range(len(data.shape)) if i != axis]
        return np.expand_dims(param, axis=tuple(axes))


def _cast_to_quantized_dtype(input_data: np.ndarray, output_dtype: types) -> np.ndarray:
    """Cast the input from dequantized dtype to quantized dtype."""
    return input_data.astype(types.nptype_from_builtin(output_dtype))


def _cast_from_quantized_dtype(input_data: np.ndarray) -> np.ndarray:
    """Cast the input from quantized dtype to dequantized dtype."""
    return input_data.astype(np.float32)


def _need_manual_broadcast(input1: np.ndarray, input2: np.ndarray) -> bool:
    """Check if the np auto broadcast can handle. If not, need manual broadcasting."""
    for axis, (dim1, dim2) in enumerate(zip(input1.shape, input2.shape)):
        if dim1 != dim2 and not (dim1 == 1 or dim2 == 1):
            return True
    return False


def quantize_by_scale_and_zp(
    input_data: np.ndarray,
    scale: np.ndarray,
    zero_point: Optional[np.ndarray],
    output_dtype: Union["types", str],
) -> np.ndarray:
    """Given scale and zero_point (if any), calculate quantized data."""
    if isinstance(output_dtype, str):
        output_dtype = types.string_to_builtin(output_dtype)
    output_range = types.type_mapping.builtin_to_range(output_dtype)

    if _need_manual_broadcast(scale, input_data):
        scale = repeat_data_as(scale, input_data.shape)
    quantized_data = input_data / scale
    if types.is_int(output_dtype):
        quantized_data = np.around(quantized_data)
    if zero_point is not None:
        if _need_manual_broadcast(zero_point, input_data):
            zero_point = repeat_data_as(zero_point, input_data.shape)
        quantized_data += zero_point.astype(np.float32)
    quantized_data = np.clip(quantized_data, output_range.low, output_range.high)
    return _cast_to_quantized_dtype(quantized_data, output_dtype)


def dequantize_by_scale_and_zp(
    quantized_data: np.ndarray,
    scale: np.ndarray,
    zero_point: Optional[np.ndarray],
    axis: Optional[int] = None,
) -> np.ndarray:
    """
    Given scale and zero_point (if any), calculate dequantized data.

    The axis is for promoting scale and zero_point to the same rank as quantized_data.
    """
    scale = promote_rank_to_same_as_data(scale, quantized_data, axis)
    if zero_point is not None:
        zero_point = promote_rank_to_same_as_data(zero_point, quantized_data, axis)

    dequantized_data = _cast_from_quantized_dtype(quantized_data)
    if zero_point is not None:
        if _need_manual_broadcast(zero_point, dequantized_data):
            zero_point = repeat_data_as(zero_point, dequantized_data.shape)
        dequantized_data -= _cast_from_quantized_dtype(zero_point)
    if _need_manual_broadcast(scale, dequantized_data):
        scale = repeat_data_as(scale, dequantized_data.shape)
    dequantized_data *= scale
    return dequantized_data.astype(scale.dtype)


def compute_qparams(
    weight: np.ndarray,
    nbits: int,
    signed: bool,
    quantization_mode: str,
    dtype: np.dtype,  # Unused parameter for backward compatibility.
    block_sizes: List[int],
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Compress the given weight matrix by quantizing the weights.
    Provide different configurations of quantization by specifying a ``block_sizes`` which
    is a list containing the block size for each dimension of the weight or 0 otherwise.

    Note that per-tensor, per-channel, channelwise-grouped and per-block are
    just variants of specifying the block sizes for each dimension.
    """
    dtype = types.get_nbits_int_builtin_type(nbits, signed)
    return compute_qparams_by_dtype(weight, dtype, quantization_mode, block_sizes)


def compute_qparams_by_dtype(
    weight: np.ndarray,
    dtype: types,
    quantization_mode: str,
    block_sizes: List[int],
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Similar to `compute_qparams`, ut use `dtype` to specify quant dtype.
    """
    if not isinstance(weight, np.ndarray):
        raise ValueError(f"Only numpy arrays are supported, but got weight type {type(weight)}")

    if len(block_sizes) != len(weight.shape):
        raise AssertionError(
            "Each axis should have a block size, which means len(block_sizes) must be "
            f"equal to weight's rank, but got {len(block_sizes)} vs {len(weight.shape)}"
        )

    new_shape, scale_shape, axes_to_skip = [], [], []
    for axis, (dim_size, block_size) in enumerate(zip(weight.shape, block_sizes)):
        if block_size > 0:
            if dim_size % block_size != 0:
                _logger.warning(
                    f"Invalid block_sizes; On {axis}th axis, the dim size {dim_size} is "
                    f"not divisible by block size {block_size}. Unable to perform "
                    "structured quantization."
                )
                return None

            # Skip this axis while computing min & max
            axes_to_skip.append(len(new_shape))

            # channel dim now will be (num_blocks, block_size)
            num_blocks = dim_size // block_size
            new_shape.extend([num_blocks, block_size])
            scale_shape.append(num_blocks)
        else:
            new_shape.append(dim_size)
            scale_shape.append(1)

    # Axes to reduce while compute min & max values
    axes = tuple(filter(lambda x: x not in axes_to_skip, range(len(new_shape))))

    quantized_data, scale, zero_point = quantize_weight_by_dtype(
        weight.reshape(new_shape), axes, dtype, quantization_mode
    )

    np_dtype = types.nptype_from_builtin(dtype)
    quantized_data = quantized_data.reshape(weight.shape).astype(np_dtype)
    scale = scale.reshape(scale_shape)
    if zero_point is not None:
        zero_point = zero_point.reshape(scale_shape).astype(np_dtype)

    return quantized_data, scale, zero_point


def repeat_data_as(input_data: np.ndarray, target_shape: Collection[int]) -> np.ndarray:
    """
    Repeat the first K dimensions of input_data where len(target_shape) == K.
    It should be guaranteed that each dimension of the K dimensions is divisible by the
    corresponding dimension given by the target_shape.

    For example, lut has shape [block_num0, block_num1, ..., 2**nbits, vector_size], so
    it needs to be interleaved repeated to make each block match the target shape.
    """
    if len(input_data.shape) < len(target_shape):
        raise ValueError(
            "The provided input_data's rank should be >= target shape's "
            f"length. But got {len(input_data.shape)} vs {len(target_shape)}"
        )
    repeated_data = input_data
    for axis, target_dim_size in enumerate(target_shape):
        input_dim_size = input_data.shape[axis]
        if target_dim_size % input_dim_size != 0:
            raise ValueError(
                "The dim size in each axis must be divisible by the tensor dimension. "
                f"Got invalid input_data shape {input_data.shape} for target shape "
                f"{target_shape[axis]} at axis {axis}"
            )
        block_size = target_dim_size // input_dim_size
        # Can use kron for higher efficiency, but repeat is easier to understand.
        if block_size > 1:
            repeated_data = np.repeat(repeated_data, block_size, axis=axis)
    return repeated_data


def lut_to_dense(
    indices: np.ndarray,
    lut: np.ndarray,
    vector_axis: Optional[int] = None,
) -> np.ndarray:
    """Decompress the lut with indices to dense (decompressed) output."""
    palette_num = lut.shape[-2]
    vector_size = lut.shape[-1]

    output_shape = list(indices.shape)
    if vector_size > 1:
        if vector_axis < 0:
            vector_axis += len(output_shape)
        # Swap the vector axis to the last dim to vectorize computations.
        indices = np.swapaxes(indices, -1, vector_axis)
        lut = np.swapaxes(lut, -3, vector_axis)
        output_shape = list(indices.shape)
        output_shape[-1] *= vector_size

    flattened_indices = indices.astype(np.uint8).flatten()
    repeated_lut = repeat_data_as(lut, indices.shape).reshape(
        len(flattened_indices), palette_num, vector_size
    )

    output = repeated_lut[np.arange(repeated_lut.shape[0]), flattened_indices.astype(np.int32)]
    output = output.reshape(output_shape)
    if vector_size > 1:
        output = np.swapaxes(output, -1, vector_axis)

    return output


def sparse_to_dense(nonzero_data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Decompress the sparse data to dense (decompressed) output."""
    decompressed_val = np.zeros_like(mask, dtype=nonzero_data.dtype)
    decompressed_val[mask != 0] = nonzero_data
    return decompressed_val


def reshape_weight_for_vector_lut(
    weight: np.ndarray, vector_size: int, vector_axis: int
) -> np.ndarray:
    """
    For vector palettization, we need to extract vectors and move them to the last dim.
    If the input weight has shape [s0, s1, s2, ... , sn], the output shape should have shape
        [s0, ..., si // vector_size, ..., sn, vector_size] where i == vector_axis.

    For example, starting from weight `a` which has shape `[4, 4]` and `vector_size=2` and
    `vector_axis=0`, we want to reshape the matrix into `c` with shape `[2, 4, 2]`, where `c[0, 0]`
    contains `a[0,0], a[1, 0]`, and `c[1, 0]` contains `a[2, 0], a[3, 0]`, etc.

    To achieve this, we need to first swap the vector_axis to last dim, and split out the vector_size,
    and finally swap it back. Here is a concrete exmaple:
        a = np.array([[ 0,  1,  2,  3],
                      [ 4,  5,  6,  7],
                      [ 8,  9, 10, 11],
                      [12, 13, 14, 15]])
        b = np.swapaxes(a, 0, -1).reshape((4, 2, 2))
        c = np.swapaxes(b, 0, 1)
        # c[0, 0] is array([0, 4])
        # c[1, 0] is array([8, 12])
    """
    weight = np.swapaxes(weight, -1, vector_axis)
    weight = weight.reshape((*weight.shape[:-1], weight.shape[-1] // vector_size, vector_size))
    return np.swapaxes(weight, -2, vector_axis)


def find_indices_for_lut(
    data: np.ndarray, lut: np.ndarray, vector_axis: Optional[int] = None
) -> np.ndarray:
    """
    Given a data and a look-up-table (LUT), find the closest indices in LUT that each element in
    data correspond to. It's the reverse process of "Given a LUT and indices, produce data using
    indices to fetch elements in LUT".

    Note the elements in data may not exactly match the elements in lut due to numerical instability.
    So we use fuzzy match to find the closest one instead of doing exact match.

    Parameters
    - data: Arbitrary numpy array.
    - lut: [block_num1, ..., 2**nbits, vector_size]. LUT's rank is K + 2, where K is the rank of data.
        Each dimension of data should be divisible by each corresponding dimension of the LUT.
        e.g., when data's shape is [2, 3, 4], the first three elements in lut's shape is [1, 1, 2],
        it means that there are two lookup tables over the last axis, and each of them have their
        own LUT values. See details in the iOS18 `constexpr_lut_to_dense` op.
    - vector_axis: Only effective when lut's last dim (vector_size) > 1. It denotes which axis the
        vector is along.
    """
    if len(lut.shape) != len(data.shape) + 2:
        raise ValueError(
            "The lut's rank should be data's rank + 2. See constexpr_lut_to_dense op definition."
        )
    if lut.shape[-1] > 1:
        if vector_axis is None:
            raise ValueError("The vector_axis must be provided for vector palettization.")
        if not len(data.shape) > vector_axis >= -len(data.shape):
            raise ValueError(f"Invalid vector_axis ({vector_axis})")
        if vector_axis < 0:
            vector_axis += len(data.shape)
        vector_size = lut.shape[-1]
        if data.shape[vector_axis] % vector_size != 0:
            raise ValueError(
                f"The data dim on {vector_axis}th axis ({data.shape[vector_axis]}) "
                f"must be divisible by vector_size ({vector_size})"
            )
        data = reshape_weight_for_vector_lut(data, vector_size, vector_axis)

    repeated_lut_shape = data.shape
    if lut.shape[-1] > 1:
        repeated_lut_shape = repeated_lut_shape[:-1]
    repeated_lut = repeat_data_as(lut, repeated_lut_shape)

    if lut.shape[-1] == 1:
        # For scalar palettization, we can simply find the closest value for each element.
        indices = np.argmin(
            np.abs(np.expand_dims(data, axis=-1) - np.squeeze(repeated_lut, axis=-1)), axis=-1
        )
    else:
        # For vector palettization, find the closest vector by Euclidean distance.
        dist = np.linalg.norm(np.expand_dims(data, axis=-2) - repeated_lut, axis=-1)
        indices = np.argmin(dist, axis=-1)

    nbits = int(math.log2(lut.shape[-2]))
    indices = indices.astype(types.nptype_from_builtin(types.string_to_builtin(f"uint{nbits}")))
    return indices


def infer_block_sizes(
    op: "Operation",
    op_config: Union[OpLinearQuantizerConfig, OpPalettizerConfig],
    weight_to_compress: np.ndarray,
    return_channel_axis: bool = False,
) -> Union[Optional[List[int]], Tuple[Optional[List[int]], Optional[int]]]:
    """
    Infer block size on each axis based on the op and compression config.

    For per-channel, the channel axis is auto-picked.
    For per-block, the input/output axis is auto-picked if block_size is int.
    See the docstring of OpLinearQuantizerConfig for more details.
    """
    input_channel_axis, output_channel_axis = select_input_output_channel_axis(op)

    if op_config.granularity == CompressionGranularity.PER_BLOCK and not isinstance(
        op_config.block_size, int
    ):
        if len(op_config.block_size) != len(weight_to_compress.shape):
            raise ValueError(
                "The block_size in config must has one element for each axis. However, for op "
                f"{op.name}, there are {len(op_config.block_size)} elements in block_size, "
                f"but there are {len(weight_to_compress.shape)} axes in the weight."
            )
        channel_axis_candidates = []
        for axis, (b_size, dim_size) in enumerate(
            zip(op_config.block_size, weight_to_compress.shape)
        ):
            if b_size != 0 and b_size != dim_size:
                channel_axis_candidates.append(axis)
        if len(channel_axis_candidates) == 1:
            # Set channel axis if we can infer it from block sizes; else just use the default one
            # inferred by op type.
            output_channel_axis = channel_axis_candidates[0]

        if return_channel_axis:
            return list(op_config.block_size), output_channel_axis
        return list(op_config.block_size)

    if input_channel_axis is None or output_channel_axis is None:
        if return_channel_axis:
            return None, output_channel_axis
        return None

    if (
        op_config.granularity == CompressionGranularity.PER_GROUPED_CHANNEL
        and op_config.channel_axis is not None
    ):
        output_channel_axis = op_config.channel_axis

    block_sizes = [0] * len(weight_to_compress.shape)
    if op_config.granularity == CompressionGranularity.PER_TENSOR:
        input_channel_block_size = 0
        output_channel_block_size = 0
    elif op_config.granularity == CompressionGranularity.PER_CHANNEL:
        input_channel_block_size = 0
        output_channel_block_size = 1
    elif op_config.granularity == CompressionGranularity.PER_GROUPED_CHANNEL:
        input_channel_block_size = 0
        output_channel_block_size = op_config.group_size
    else:
        assert op_config.granularity == CompressionGranularity.PER_BLOCK and isinstance(
            op_config.block_size, int
        )
        input_channel_block_size = op_config.block_size
        output_channel_block_size = 1

    if input_channel_axis < len(block_sizes):
        block_sizes[input_channel_axis] = input_channel_block_size
    if output_channel_axis < len(block_sizes):
        block_sizes[output_channel_axis] = output_channel_block_size

    if return_channel_axis:
        return block_sizes, output_channel_axis
    return block_sizes


def select_input_output_channel_axis(op: "Operation") -> Tuple[Optional[int], Optional[int]]:
    """
    Here are some representative ops:
    - linear: [D_out, D_in]
    - matmul's y: [..., D_in, D_out] if transpose_y is False, else [..., D_out, D_in]
    - conv: [C_out, C_in_div_group, KH, KW]
    - conv_transpose: [C_in, C_out_div_group, KH, KW]

    The input output channel axis selection criteria is:
    - For conv_transpose the output channel is 1 and input channel is 0.
    - For matmul's y:
        - When transpose_y=False, output channel is -1 and input channel is -2
        - When transpose_y=True, output channel is -2 and input channel is -1
    - For matmul's x:
        - When transpose_x=False, output channel is -2 and input channel is -1
        - When transpose_y=True, output channel is -1 and input channel is -2
    - For all other ops, output channel is 0 and input channel is 1.

    If cannot determine the input/output axis, return None to denote unknown.
    """
    var = op.outputs[0]

    # The op could be fed into multiple ops, so we traverse all children ops to see if they
    # have consistent input/output axis, otherwise set the axis to None.
    output_channel_axis_set = set()
    input_channel_axis_set = set()
    for child_op in var.child_ops:
        # By default, output channel axis is 0 and input channel axis is 1.
        output_channel_axis, input_channel_axis = 0, 1

        if child_op.op_type == "conv_transpose":
            output_channel_axis = 1
            input_channel_axis = 0
        elif child_op.op_type == "matmul":
            if child_op.y == var:
                if child_op.transpose_y.val:
                    output_channel_axis = -2
                    input_channel_axis = -1
                else:
                    output_channel_axis = -1
                    input_channel_axis = -2
            else:  # var is used as matmul's x.
                if child_op.transpose_x.val:
                    output_channel_axis = -1
                    input_channel_axis = -2
                else:
                    output_channel_axis = -2
                    input_channel_axis = -1
        elif child_op.op_type.startswith("constexpr_"):
            # In joint compression constexpr op could be chained together.
            input_channel_axis, output_channel_axis = select_input_output_channel_axis(child_op)

        if output_channel_axis < 0:
            output_channel_axis += var.rank
        if input_channel_axis < 0:
            input_channel_axis += var.rank
        output_channel_axis_set.add(output_channel_axis)
        input_channel_axis_set.add(input_channel_axis)

    output_channel_axis, input_channel_axis = 0, 1
    if len(output_channel_axis_set) > 1:
        _logger.warning(
            f"Can't decide output axis for op {op.name}, because it's fed "
            f"into multiple downstream ops which require different output axes."
        )
        output_channel_axis = None
    elif len(output_channel_axis_set) == 1:
        output_channel_axis = output_channel_axis_set.pop()

    if len(input_channel_axis_set) > 1:
        _logger.warning(
            f"Can't decide input axis for op {op.name}, because it's fed "
            f"into multiple downstream ops which require different input axes."
        )
        input_channel_axis = None
    elif len(input_channel_axis_set) == 1:
        input_channel_axis = input_channel_axis_set.pop()

    return input_channel_axis, output_channel_axis


def is_cluster_dim_valid(op: "Operation", cluster_dim: int, channel_axis: int) -> bool:
    """
    Check op-dependent restrictions for cluster_dim.

    For example, the conv's weight has shape [C_out, C_in/groups], but the effective shape in each
    group is actually [C_out/groups, C_in/groups], so we need to make sure the effective dim on
    channel_axis is divisible by `cluster_dim`. Similarly, for conv_transpose the weight has shape
    [C_in, C_out/groups], but the effective shape in each group is [C_in/groups, C_out/groups].

    Returns True if the cluster_dim is valid, False otherwise.
    """
    var = op.outputs[0]
    if channel_axis < 0:
        channel_axis += var.rank

    for child_op in var.child_ops:
        if child_op.op_type in {"conv", "conv_transpose"}:
            effective_shape = list(var.shape)
            if child_op.groups.val is not None and child_op.groups.val > 1:
                effective_shape[0] //= child_op.groups.val
            if effective_shape[channel_axis] % cluster_dim != 0:
                return False
    return True


def ios16_sparse_params_to_ios18(sparse_params: SparseParamsIos16) -> SparseParams:
    """
    The iOS18 constexpr_sparse_to_dense no longer accepts `shape` param. Instead, the `mask` param
    has shape info. So we need to convert the old bit-packed `mask` to new uint1 `mask`.
    """
    if not isinstance(sparse_params, SparseParamsIos16):
        raise ValueError("Invalid type of params")

    mask = (
        np.unpackbits(sparse_params.mask, count=np.prod(sparse_params.shape), bitorder="little")
        .reshape(sparse_params.shape)
        .astype(types.np_uint1_dtype)
    )

    return SparseParams(nonzero_data=sparse_params.nonzero_data, mask=mask)


def ios18_sparse_params_to_ios16(sparse_params: SparseParams) -> SparseParamsIos16:
    """The iOS16 sparse params pack mask into bytes, and need a `shape` parameter."""
    return SparseParamsIos16(
        nonzero_data=sparse_params.nonzero_data,
        mask=np.packbits(sparse_params.mask, bitorder="little"),
        shape=sparse_params.mask.shape,
    )


def ios16_lut_params_to_ios18(lut_params: LutParamsIos16) -> LutParams:
    """
    The iOS18 constexpr_lut_to_dense no longer accepts `shape` param. We need to convert the iOS16
    params to the format acceptable by the iOS18 op.
    """
    num_palettes = lut_params.lut.shape[0]
    nbits = int(math.log2(num_palettes))
    if 2**nbits != num_palettes:
        raise AssertionError(
            f"Invalid number of palettes in lut_params. It should be 2**nbits, but got {num_palettes}"
        )
    # Notice that the indices in iOS16 is packed, so we need to unpack first.
    unpacked_indices = restore_elements_from_packed_bits(
        lut_params.indices, nbits, np.prod(lut_params.shape)
    )
    indices = unpacked_indices.reshape(lut_params.shape).astype(
        types.type_mapping.string_to_nptype(f"uint{nbits}")
    )
    lut_shape = [1] * len(lut_params.shape) + [num_palettes, 1]
    lut = lut_params.lut.reshape(lut_shape)
    return LutParams(indices=indices, lut=lut, vector_axis=None)


def ios18_lut_params_to_ios16(lut_params: LutParams) -> LutParamsIos16:
    """The iOS16 lut params pack indices into bytes, and need a `shape` parameter."""
    if len(lut_params.lut.shape) < 3:
        raise ValueError(
            "lut should have at least three dimensions, with shape [..., num_palettes, vector_size]."
        )
    for idx, dim_size in enumerate(lut_params.lut.shape[:-2]):
        if dim_size > 1:
            raise AssertionError(
                "The pre-iOS18 palettization only supports per-tensor lut, but got more than one "
                f"lut on {idx}th axis. LUT shape: {lut_params.lut.shape}"
                "\nPlease set the minimum_deployment_target to iOS18 or later."
            )
    if lut_params.lut.shape[-1] > 1:
        raise ValueError(
            "Vector palettization (lut last dim > 1) is only supported in iOS18+. "
            "Please set the minimum_deployment_target to iOS18 or later."
        )

    num_palettes = lut_params.lut.shape[-2]
    nbits = int(math.log2(num_palettes))
    return LutParamsIos16(
        lut=lut_params.lut.reshape((num_palettes,)),
        indices=pack_elements_into_bits(lut_params.indices, nbits),
        shape=np.array(
            lut_params.indices.shape, dtype=np.uint32
        ),  # The op requires shape parameter to be uint32.
    )


def ios18_quant_params_to_ios16(quant_params: QuantParams) -> QuantParamsIos16:
    """
    Transform iOS18 quant params to iOS16 version.

    The iOS16 constexpr_affine_dequantize op requires axis, and it requires scale and zero_point to
    have rank 0 or 1.
    """
    # Infer the axis based on scale's shape.
    non_single_dim = [dim for dim, dim_size in enumerate(quant_params.scale.shape) if dim_size > 1]
    if len(non_single_dim) > 2:
        raise AssertionError(
            "The constexpr_affine_dequantize op doesn't support scale which "
            "have more than one non-single dimensions. Got scale with shape "
            f"{quant_params.scale.shape}"
        )
    # If non_single_dim is empty, it means it's per-tensor quantization, just use a dummy axis.
    axis = 0 if len(non_single_dim) == 0 else non_single_dim[0]

    scale = quant_params.scale
    zero_point = quant_params.offset
    if zero_point is None:
        # The constexpr_affine_dequantize op requires zero_point.
        zero_point = np.zeros_like(scale).astype(quant_params.data.dtype)

    # The constexpr_affine_dequantize op requires scale and zero_point to have rank 0 or 1.
    if isinstance(scale, (np.ndarray, np.generic)):
        scale = np.squeeze(scale)
    if isinstance(zero_point, (np.ndarray, np.generic)):
        zero_point = np.squeeze(zero_point)

    return QuantParamsIos16(
        quantized_data=quant_params.data, zero_point=zero_point, scale=scale, axis=np.int32(axis)
    )


def pack_elements_into_bits(elements: np.ndarray, nbits: int) -> np.ndarray:
    """
    Pack elements into nbits representation, by starting with the least significant bit (LSB) and
    moving upward to the most significant bit (MSB).

    Returns packed elements as np.uint8.
    """
    if not np.issubdtype(elements.dtype, np.integer):
        raise ValueError(f"Only support packing integers elements, but got {elements.dtype}")

    # Adjust allowed value range based on if the input is signed or unsigned.
    if np.issubdtype(elements.dtype, np.signedinteger):
        max_val = 2 ** (nbits - 1) - 1
        min_val = -max_val - 1
    else:
        max_val = 2**nbits - 1
        min_val = 0
    if np.max(elements) > max_val:
        raise ValueError(
            f"To pack elements into {nbits}-bit, the max value is {max_val}, but got {np.max(elements)}"
        )
    if np.min(elements) < min_val:
        raise ValueError(
            f"To pack elements into {nbits}-bit, the min value is {min_val}, but got {np.min(elements)}"
        )

    # As np.unpackbits only supports uint8, convert to uint8 first.
    # Notice that it will not lose information, because the bits are unchanged when converting int8
    # to uint8. For example, the signed int -6 has bit representation '11111010', and when we unpackbits
    # we get [0, 1, 0, 1, 1, 1, 1, 1], where only first 4 elements are needed for 4-bit representation.
    elements = elements.astype(np.uint8)
    bitarray = np.unpackbits(elements.reshape(-1, 1), bitorder="little", axis=-1)[:, :nbits]
    return np.packbits(bitarray.flatten(), bitorder="little")


def restore_elements_from_packed_bits(
    packed_values: np.ndarray, nbits: int, element_num: int, are_packed_values_signed: bool = False
) -> np.ndarray:
    """
    Restore elements from packed bits. Requires values that are packed by starting with the
    least significant bit (LSB) and moving upward to the most significant bit (MSB), which is the
    method used in `pack_elements_into_bits`.

    are_packed_values_signed: Indicates if the packed_values were packed from signed integers. If
        True, the n-bit number unpacked from packed_values will be interpreted as signed integers,
        and the returned ndarray will have dtype np.int8. Otherwise, np.uint8 will be used.
    """
    if len(packed_values.shape) != 1:
        raise NotImplementedError(
            f"Only support 1-rank packed_values. But got {len(packed_values.shape)}"
        )

    if packed_values.dtype == np.int8:
        # As np.unpackbits only supports uint8, need to convert first.
        packed_values = packed_values.astype(np.uint8)
    elif packed_values.dtype != np.uint8:
        raise NotImplementedError(
            f"Only support int8 or uint8 packed_values, but got {packed_values.dtype}"
        )

    bitarray = np.unpackbits(packed_values, bitorder="little")
    pad_required = bitarray.size % nbits != 0
    if pad_required:
        bitarray = np.concatenate([bitarray, np.zeros(nbits - bitarray.size % nbits)]).astype(
            bitarray.dtype
        )
        if bitarray.size % nbits != 0:
            raise ValueError(
                f"The length of bitarray ({bitarray.size}) should be divisible by "
                f"nbits ({nbits})."
            )
    bitarray = bitarray.reshape(-1, nbits)[:element_num, :]
    # The np.packbits doesn't work well for signed int if we feed `bitarray` to it directly.
    # For example, the original signed int is -6, which is packed as 1010 for 4-bit representation,
    # and here `bitarray` is [[0, 1, 0, 1]], where the value will be interpreted as 10 (b'1010')
    # by np.packbits.
    # To make np.packbits work correctly, we need to repeat the sign bit. For example, 1010 will
    # become 11111010, where np.packbits can correctly handle and after converting to int8 it's -6.
    if are_packed_values_signed:
        # Repeat the sign bit to make uint8 to int8 works.
        bitarray = np.repeat(bitarray, [1] * (nbits - 1) + [8 - nbits + 1], axis=1)
    restored_elements = np.packbits(bitarray, bitorder="little", axis=-1).reshape(-1)
    if are_packed_values_signed:
        restored_elements = restored_elements.astype(np.int8)
    return restored_elements


def get_min_and_max_values(
    activation_stats: Dict[str, float],
    var_name: str,
) -> np.ndarray:
    """
    Utility to get the "rmin" and "rmax" values for a given var name in activation quantization.
    """
    if activation_stats is None:
        raise AssertionError(
            "'activation_stats' is required for calibration in activation quantization."
        )

    return np.array(
        [activation_stats[var_name]["rmin"], activation_stats[var_name]["rmax"]], dtype=np.float16
    )


def _update_tensor_range(
    tensor_name: str,
    tensor_value: Union[int, float],
    activation_stats_dict: Dict[str, Dict[str, float]],
) -> None:
    """
    Utility to update the "rmin" and "rmax" values in activation stats dictionary.
    """
    tensor_min = np.min(np.array(tensor_value).flatten())
    tensor_max = np.max(np.array(tensor_value).flatten())
    activation_stats_dict[tensor_name]["rmin"] = tensor_min
    activation_stats_dict[tensor_name]["rmax"] = tensor_max
    if tensor_name in activation_stats_dict:
        activation_stats_dict[tensor_name]["rmin"] = min(
            tensor_min, activation_stats_dict[tensor_name]["rmin"]
        )
        activation_stats_dict[tensor_name]["rmax"] = max(
            tensor_max, activation_stats_dict[tensor_name]["rmax"]
        )
    else:
        activation_stats_dict[tensor_name]["rmin"] = tensor_min
        activation_stats_dict[tensor_name]["rmax"] = tensor_max


def _combine_lists_with_common_elements(data: List[List[str]]) -> List[List[str]]:
    """
    Utility for merging lists with common elements.

    Parameters
    ----------
    data: list[list[]]
        data is a list of lists with strings.

    Returns
    -------
    merged: combined lists with common elements.

    Example
    -------
    input: [["conv0", "conv1", "conv2"], ["conv0", "conv3"], ["relu0"]]
    output: [["conv0", "conv1", "conv2", "conv3"], ["relu0"]]
    """

    merged = []
    for item in data:
        item_set = set(item)
        not_exsit = True
        for result in merged:
            if result & item_set:
                result.update(item_set)
                not_exsit = False
                break
        if not_exsit:
            merged.append(item_set)
    return merged


def _adjust_concat_surrounding_activation_stats(
    concat_op_info_list: List[List[str]], activation_stats_dict: Dict[str, Dict[str, float]]
) -> None:
    """
    Adjust the activation calibration stats of inputs/outputs to the same concat ops to maximize hardware efficiency.
    Tensor values of inputs/outputs to the same concat op should share same range (same min/max), so the quantized
    concat could be surrounded by quantize/dequantize pairs with same scale and zero point values.

    Example
    -------
    - concat 1 -
    inputs:  "input_1", "input_2", "input_3"
    output:  "output_1"

    - concat 2 -
    inputs:  "input_1", "input_4"
    output:  "output_2"

    Input/output tensors range of concat 1 should be identical.
    Input/output tensors range of concat 2 should be identical.
    "input_1" is in both, which means activation calibration stats of all 6 tensors above should be identical.
    """

    if concat_op_info_list is None:
        return

    # Merge tensor names which should have identical values, to the same list.
    concat_list_adjusted = _combine_lists_with_common_elements(concat_op_info_list)

    for concat_group in concat_list_adjusted:
        group_rmin_list, group_rmax_list = [], []

        for tensor_name in concat_group:
            # Some tensor_name may not have rmin/rmax if the calibration failed before.
            if tensor_name in activation_stats_dict:
                group_rmin_list.append(activation_stats_dict[tensor_name]["rmin"])
                group_rmax_list.append(activation_stats_dict[tensor_name]["rmax"])

        if len(group_rmin_list) == 0:
            raise ValueError(
                "None of the calibration run succeeded. Please check logs about calibrating sample failures."
            )
        group_rmin, group_rmax = min(group_rmin_list), max(group_rmax_list)

        for tensor_name in concat_group:
            if tensor_name in activation_stats_dict:
                activation_stats_dict[tensor_name]["rmin"] = group_rmin
                activation_stats_dict[tensor_name]["rmax"] = group_rmax
