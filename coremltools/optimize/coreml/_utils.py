# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import math
from collections import namedtuple
from typing import List, Optional, Tuple, Union

import numpy as np

from coremltools import _getLogger
from coremltools.converters.mil.mil import Operation, types
from coremltools.optimize.coreml import _utils as optimize_utils
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
LutParams = namedtuple("LutParams", "indices lut")
QuantParams = namedtuple("QuantParams", "data scale offset nbits")


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
    if not np.issubdtype(weight.dtype, np.floating):
        raise ValueError("Only floating numpy arrays are supported.")

    val_min = np.amin(weight, axis=axes, keepdims=True)
    val_max = np.amax(weight, axis=axes, keepdims=True)

    q_val_min, q_val_max = get_quant_range(nbits, signed, quantization_mode)

    zero_point = None
    if quantization_mode == "LINEAR_SYMMETRIC":
        # For the linear_symmetric quantization_mode, the range is symmetrical to 0
        max_abs = np.maximum(np.abs(val_min), np.abs(val_max))
        val_min = -max_abs
        val_max = max_abs

        if not signed:
            zero_point_shift = q_val_max // 2
            zero_point = zero_point_shift * np.ones(val_min.shape)
    else:
        assert quantization_mode == "LINEAR"
        # For the linear quantization_mode, we need to make sure the data range contains `0`
        val_min = np.minimum(0.0, val_min)
        val_max = np.maximum(0.0, val_max)
        zero_point = (q_val_min * val_max - q_val_max * val_min) / (val_max - val_min)
        zero_point = np.round(zero_point)
        zero_point = np.clip(zero_point, q_val_min, q_val_max)

    scale = (val_max - val_min) / (q_val_max - q_val_min)
    quantized_data = np.round(weight / scale)
    if zero_point is not None:
        quantized_data += zero_point
        zero_point = zero_point.squeeze().astype(dtype)
    quantized_data = np.clip(quantized_data, q_val_min, q_val_max).astype(dtype)
    scale = scale.astype(weight.dtype).squeeze()

    return quantized_data, scale, zero_point


def compute_qparams(
    weight: np.ndarray,
    nbits: int,
    signed: bool,
    quantization_mode: str,
    dtype: np.dtype,
    block_sizes: List[int],
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Compress the given weight matrix by quantizing the weights.
    Provide different configurations of quantization by specifying a ``block_sizes`` which
    is a list containing the block size for each dimension of the weight or 0 otherwise.

    Note that per-tensor, per-channel, channelwise-grouped and per-block are
    just variants of specifying the block sizes for each dimension.
    """
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

    quantized_data, scale, zero_point = quantize_weight(
        weight.reshape(new_shape), axes, nbits, signed, quantization_mode, dtype
    )

    quantized_data = quantized_data.reshape(weight.shape)
    scale = scale.reshape(scale_shape)
    if zero_point is not None:
        zero_point = zero_point.reshape(scale_shape)

    return quantized_data, scale, zero_point


def find_indices_for_lut(data: np.ndarray, lut: np.ndarray) -> np.ndarray:
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
    """
    if len(lut.shape) != len(data.shape) + 2:
        raise ValueError("The lut's rank should be data's rank + 2. See constexpr_lut_to_dense.")

    # TODO (rdar://124474258): Handle vector palettization.
    if lut.shape[-1] > 1:
        raise NotImplementedError(
            "Not support vector palettization. Progress tracked in rdar://124474258."
        )

    # lut has shape [block_num0, block_num1, ..., 2**nbits, vector_size], so need to interleaved
    # repeat it to make each block match the weight.
    repeated_lut = lut
    for axis, block_num in enumerate(lut.shape[:-2]):
        weight_dim_size = data.shape[axis]
        if weight_dim_size % block_num != 0:
            raise ValueError(
                "The weight dim size in each axis must be divisible by the number "
                f"of luts. Got invalid lut {lut.shape} for weight shape "
                f"{data.shape[axis]} at axis {axis}"
            )
        block_size = weight_dim_size // block_num
        # Can use np.kron for higher efficiency, but repeat is easier to understand.
        if block_size > 1:
            repeated_lut = np.repeat(repeated_lut, block_size, axis=axis)

    # Find the closest value for each element.
    indices = np.argmin(
        np.abs(np.expand_dims(data, axis=-1) - np.squeeze(repeated_lut, axis=-1)), axis=-1
    )
    nbits = int(math.log2(lut.shape[-2]))
    indices = indices.astype(types.nptype_from_builtin(types.string_to_builtin(f"uint{nbits}")))
    return indices


def infer_block_sizes(
    op: "Operation",
    op_config: Union[OpLinearQuantizerConfig, OpPalettizerConfig],
    weight_to_compress: np.ndarray,
) -> List[int]:
    """
    Infer block size on each axis based on the op and compression config.

    For per-channel, the channel axis is auto-picked.
    For per-block, the input/output axis is auto-picked if block_size is int.
    See the docstring of OpLinearQuantizerConfig for more details.
    """
    if op_config.granularity == CompressionGranularity.PER_BLOCK and not isinstance(
        op_config.block_size, int
    ):
        if len(op_config.block_size) != len(weight_to_compress.shape):
            raise ValueError(
                "The block_size in config must has one element for each axis. However, for op "
                f"{op.name}, there are {len(op_config.block_size)} elements in block_size, "
                f"but there are {len(weight_to_compress.shape)} axes in the weight."
            )
        return list(op_config.block_size)

    input_channel_axis, output_channel_axis = optimize_utils.select_input_output_channel_axis(op)
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
    return block_sizes


def select_input_output_channel_axis(op: "Operation") -> Tuple[int, int]:
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
    """
    output_channel_axis, input_channel_axis = 0, 1
    var = op.outputs[0]
    if len(var.child_ops) == 1:
        child_op = var.child_ops[0]
        if child_op.op_type == "conv_transpose":
            output_channel_axis = 1
            input_channel_axis = 0
        if child_op.op_type == "matmul":
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
        if child_op.op_type.startswith("constexpr_"):
            return select_input_output_channel_axis(child_op)
    return input_channel_axis, output_channel_axis


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
    return LutParams(indices=indices, lut=lut)


def ios18_lut_params_to_ios16(lut_params: LutParams) -> LutParamsIos16:
    """The iOS16 lut params pack indices into bytes, and need a `shape` parameter."""
    for idx, dim_size in enumerate(lut_params.lut.shape[:-2]):
        if dim_size > 1:
            raise AssertionError(
                "The iOS16 only supports per-tensor lut, but got more than one "
                f"lut on {idx}th axis. LUT shape: {lut_params.lut.shape}"
            )

    num_palettes = lut_params.lut.shape[-2]
    nbits = int(math.log2(num_palettes))
    return LutParamsIos16(
        lut=lut_params.lut.reshape((num_palettes,)),
        indices=pack_elements_into_bits(lut_params.indices, nbits),
        shape=lut_params.indices.shape,
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
