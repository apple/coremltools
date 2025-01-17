# Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import math
from typing import List, Optional

import numpy as np

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.operation import Operation
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS16.constexpr_ops import (
    constexpr_cast as _constexpr_cast_iOS16,
)
from coremltools.converters.mil.mil.ops.defs.iOS18 import _IOS18_TARGET
from coremltools.converters.mil.mil.var import Var


@register_op(opset_version=_IOS18_TARGET)
class constexpr_blockwise_shift_scale(Operation):
    """
    A compile-time operation that returns a constant output value upon dequantizing its constant inputs.

    It's similar to iOS 16 :py:class:`~.iOS16.constexpr_ops.constexpr_affine_dequantize`, but supports
    block-wise quantization for int4 and int8.

    Although all parameters of this op are constants, this op is not constant-folded to a single
    const op at the time of model serialization. The unquantized output will be decompressed later,
    based on the implementation detail (either at model load time or runtime).

    Generic expression: output = scale * (data - offset)

    Algorithm:
        Assuming Rank 3 scenario:
            output_data[i, j, k] = scale[i0, j0, k0] * (data[i, j, k] - offset[i0, j0, k0])
            where
                i0 = floor(i/block_size[0]),
                j0 = floor(j/block_size[1]),
                k0 = floor(k/block_size[2])
        The block size is implied by block_size[m] = data.shape[m] / scale.shape[m]

    Constraints:
    - All tensors: scale, data, offset and output have same rank.
    - Inputs: scale and offset (if provided) have same shape.
    - Output shape is same as the shape of input argument: `data`.
    - Number of scales along each dimension should be a factor of corresponding dimension size of
      `data`.  That is, block_size[i] should be an integer where block_size[i] = data.shape[i] / scale.shape[i]

    Parameters
    ----------
    data: const tensor<SrcT, [1..]> (Required)

    scale: const tensor<DstT, [1..]> (Required)

    offset: const tensor<OffsetT, [1..]> (Optional)
        * If provided, must have the same shape as the ``scale``.
        * If dtype is not fp16 or fp32, it must be the same as SrcT.

    Returns
    -------
    const tensor<DstT, [1..]>

    Attributes
    ----------
    SrcT: int4, uint4, int8, uint8, fp16, fp32
    DstT: fp16, fp32
    OffsetT: int4, uint4, int8, uint8, fp16, fp32
    """

    input_spec = InputSpec(
        data=TensorInputType(const=True, type_domain="SrcT"),
        scale=TensorInputType(const=True, type_domain="DstT"),
        offset=TensorInputType(const=True, optional=True, type_domain="OffsetT"),
    )

    type_domains = {
        "SrcT": (types.int4, types.uint4, types.int8, types.uint8, types.fp16, types.fp32),
        "DstT": (types.fp16, types.fp32),
        "OffsetT": (types.int4, types.uint4, types.int8, types.uint8, types.fp16, types.fp32),
    }

    @staticmethod
    def _validate_shift_scale_inputs(
        data_shape: List[int], data_dtype: types, scale: Var, offset: Var
    ):
        data_rank = len(data_shape)
        if data_rank != scale.rank:
            raise ValueError(
                f"Parameter 'data' and 'scale' need to have the same rank, but got {data_rank} vs {scale.rank}."
            )
        if data_rank < 1:
            raise ValueError("Parameter 'data' needs to have at least rank 1, but got scalar.")
        for rank_idx in range(data_rank):
            data_dim = data_shape[rank_idx]
            scale_dim = scale.shape[rank_idx]
            if data_dim % scale_dim != 0:
                raise ValueError(
                    "Number of scales along each dimension should be a factor of "
                    "corresponding dimension size of 'data'. However, at dim "
                    f"{rank_idx}, the 'data' has {data_dim} while 'scale' has {scale_dim}."
                )

        if offset is not None:
            if offset.shape != scale.shape:
                raise ValueError(
                    "Invalid parameter 'offset'; the shape of 'offset' should match the shape of "
                    f"'scale', but got ({offset.shape}) vs ({scale.shape})."
                )
            if not types.is_float(offset.dtype) and offset.dtype != data_dtype:
                raise ValueError(
                    "Invalid parameter 'offset'; the dtype of 'offset' should match the dtype of "
                    f"'data', but got ({types.builtin_to_string(offset.dtype)}) vs "
                    f"({types.builtin_to_string(data_dtype)})."
                )

    def _validate_inputs(self):
        self._validate_shift_scale_inputs(self.data.shape, self.data.dtype, self.scale, self.offset)

    def type_inference(self):
        self._validate_inputs()
        return types.tensor(self.scale.dtype, self.data.shape)

    def materialized_val_inference(self):
        data = self.data.val
        scale = self.scale.val
        if data is None and self.data.op.op_type.startswith("constexpr_"):
            data = self.data.op.materialized_val_inference()
        if scale is None and self.scale.op.op_type.startswith("constexpr_"):
            scale = self.scale.op.materialized_val_inference()

        return self.decompress(
            data,
            scale,
            None if self.offset is None else self.offset.val,
        )

    @staticmethod
    def decompress(
        data: np.ndarray,
        scale: np.ndarray,
        offset: Optional[np.ndarray],
    ):
        # Adjust dtype to avoid overflow in the quantized dtype.
        data = data.astype(scale.dtype)

        # Interleaved repeat scale and offset to make it match the shape of data.
        block_sizes = [
            data_shape // scale_shape for (data_shape, scale_shape) in zip(data.shape, scale.shape)
        ]
        for axis, block_size in enumerate(block_sizes):
            if block_size > 1:
                scale = np.repeat(scale, block_size, axis)
                if offset is not None:
                    offset = np.repeat(offset, block_size, axis)

        if offset is not None:
            data = data - offset
        data = scale * data

        return data


@register_op(opset_version=_IOS18_TARGET)
class constexpr_lut_to_dense(Operation):
    """
    A compile-time operation that returns a constant output value upon dequantizing its constant inputs.

    This operator is used to store constant weights in lookup tables format (aka palettized weights).
    It's similar to iOS 16 :py:class:`~.iOS16.constexpr_ops.constexpr_lut_to_dense`, but supports
    block-wise / vector palettization.

    LUT's rank is K + 2, where K is the rank of indices.
    Each dimension of LUT's first K dimensions should be divisible by each corresponding dimension
    of the decompressed tensor.
    e.g., when indices_shape = [2, 3, 4], lut_shape[:3] = [1, 1, 2], it means that there are two
    lookup tables over the last axis. And each of them have their own LUT values.
    See Case 1 below for details.

    VECTOR_SIZE is added to support vector palettization.
    - When VECTOR_SIZE is 1, it is scalar palettization.
    - When VECTOR_SIZE is larger than 1, it retrieves a vector instead of a single value from the
      lookup table, and fill the result continuously.
    The vector_axis is used to define which axis the vectored elements in the lookup table be filled
    across the output tensor. vector_axis is only optional if VECTOR_SIZE is 1.
    As a result:
        output_shape[i] = indices_shape[i] , i != vector_axis
        output_shape[i] = indices_shape[i] * VECTOR_SIZE, i == vector_axis
    See Case 2 below for details.

    Examples:

      Case 1: per-group scalar palettization:
        e.g.:
        - indices = tensor<uint2, [6, 2]>>([2, 3, 3, 0, 1, 0, 3, 0, 2, 1, 0, 3])
        - lut = tensor<fp16, [2, 1, 4, 1]>([1.0, 5.0, 9.0, 13.0, 2.0, 10.0, 18.0, 26.0])

        It is effectively a 2-group 2-bit scalar palettization.
        The output shape would be [6, 2], which is the same as the indices shape.
        The output tensor values are:
        [[lut0[2]->9.0,  lut0[3]->13.0],
          [lut0[3]->13.0, lut0[0]->1.0],
          [lut0[1]->5.0,  lut0[0]->1.0],
          [lut1[3]->26.0, lut1[0]->2.0],
          [lut1[2]->18.0, lut1[1]->10.0],
          [lut1[0]->2.0,  lut1[3]->26.0]]
        where lut0 is the first lookup table (lut[0, :, :, :]) and lut1 is the second lookup table.

      Case 2: per-tensor vector palettization:
        e.g.:
        - indices = tensor<uint1, [2, 2, 2]>>.
        The indices values are:
               [
                 [
                  [0, 0],
                  [1, 0]
                 ],
                 [
                  [1, 1],
                  [0, 0]
                 ]
               ]
        - lut = tensor<int8, [1, 1, 1, 2, 3]>([a0, a1, a2,
                    b0, b1, b2])
           which means the two centroids are [a1, a2, a3] and [b1, b2, b3].

      Case 2.1: vector_axis = 1
        It is effectively a 1-bit vector palettization.
        The output shape would be [2, 2*3, 2], where each index in the indices would be effectively replaced with
        the 3 elements in the vector over the 1st dimension to construct the output tensor.
        The output values are:
        [
         [
          [a0, a0],
          [a1, a1],
          [a2, a2],
          [b0, a0],
          [b1, a1],
          [b2, a2],
         ],
         [
          [b0, b0],
          [b1, b1],
          [b2, b2],
          [a0, a0],
          [a1, a1],
          [a2, a2],
         ]
        ]

      Case 2.2: vector_axis = 2
        The output shape would be [2, 2, 2*3], where each index in the indices would be effectively replaced with
        the 3 elements in the vector over the last dimension to construct the output tensor.
        The output values are:
        [
         [
          [a0, a1, a2, a0, a1, a2],
          [b0, b1, b2, a0, a1, a2],
         ],
         [
          [b0, b1, b2, b0, b1, b2],
          [a0, a1, a2, a0, a1, a2],
         ]
        ]

    Parameters
    ----------
    indices: const tensor<IndicesT, [1..]> (Required)

    lut: const tensor<T, [1.., NUM_PALETTES, VECTOR_SIZE]> (Required)
        * NUM_PALETTES needs to be 2^nbits where nbits is indicated by IndicesT.

    vector_axis: const tensor<int32, []> (Optional)
        * vector_axis can be optional if VECTOR_SIZE is 1.

    Returns
    -------
    const tensor<T, [1..]>
        * output_shape = indices_shape * [1..1, VECTOR_SIZE, 1..1] (all 1 but VECTOR_SIZE at vector_axis dimension).

    Attributes
    ----------
    IndicesT: uint1, uint2, uint3, uint4, uint6, uint8
    T: uint8, int8, fp16, fp32
    """

    input_spec = InputSpec(
        indices=TensorInputType(const=True, type_domain="IndicesT"),
        lut=TensorInputType(const=True, type_domain="T"),
        vector_axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "IndicesT": (types.uint1, types.uint2, types.uint3, types.uint4, types.uint6, types.uint8),
        "T": (types.int8, types.uint8, types.fp16, types.fp32),
    }

    @staticmethod
    def _validate_lut_inputs(
        indices_shape: List[int], indices_dtype: types, lut_shape: List[int], vector_axis: Var
    ):
        indices_rank = len(indices_shape)
        lut_rank = len(lut_shape)

        if indices_rank < 1:
            raise ValueError("Parameter 'indices' needs to have at least rank 1, but got scalar.")

        if lut_rank != indices_rank + 2:
            raise ValueError(
                f"Parameter 'lut' need to have 2 more dim than 'indices', but got "
                f"{lut_rank}-rank 'lut' and {indices_rank}-rank 'indices'."
            )

        for rank_idx in range(indices_rank):
            indices_dim = indices_shape[rank_idx]
            lut_dim = lut_shape[rank_idx]
            if indices_dim % lut_dim != 0:
                raise ValueError(
                    f"Each dimension of 'indices' should be divisible by each corresponding "
                    f"dimension of the 'lut'. However, at dim {rank_idx}, the 'indices' has "
                    f"{indices_dim} while 'lut' has {lut_dim}."
                )

        num_palettes = lut_shape[-2]
        nbits = int(math.log2(num_palettes))
        if num_palettes != 2**nbits:
            raise ValueError(
                f"Invalid parameter 'lut'; the second last dim should have size 2^nbits, but got {lut_shape[-2]}."
            )
        if nbits != indices_dtype.get_bitwidth():
            raise ValueError(
                f"Invalid parameter 'indices'; the second last dim indicate number of palettes ({num_palettes}), "
                f"which means nbits is {nbits}, so the dtype of indices should be uint{nbits}, but got "
                f"{types.builtin_to_string(indices_dtype)}."
            )

        if vector_axis is not None:
            if vector_axis.rank > 0:
                raise ValueError(
                    "Invalid parameter 'vector_axis'; It should be a scalar, but got " "a tensor."
                )
            if not -indices_rank <= vector_axis.val < indices_rank:
                raise ValueError(
                    f"Invalid parameter 'vector_axis'; The valid range is between "
                    f"{-indices_rank} and {indices_rank}, but got {vector_axis.val}."
                )
        else:
            if lut_shape[-1] > 1:
                raise ValueError(
                    "When lut's last dim (VECTOR_SIZE) > 1, the parameter "
                    "'vector_axis' need to be provided."
                )

    def _validate_inputs(self):
        self._validate_lut_inputs(
            self.indices.shape, self.indices.dtype, self.lut.shape, self.vector_axis
        )

    def type_inference(self):
        self._validate_inputs()
        output_shape = self.indices.shape
        vector_size = self.lut.shape[-1]
        if vector_size > 1:
            output_shape = list(output_shape)
            output_shape[self.vector_axis.val] *= vector_size
            output_shape = tuple(output_shape)
        return types.tensor(self.lut.dtype, output_shape)

    def materialized_val_inference(self):
        return self.decompress(
            self.indices.val,
            self.lut.val,
            None if self.vector_axis is None else self.vector_axis.val,
        )

    @staticmethod
    def decompress(
        indices: np.ndarray,
        lut: np.ndarray,
        vector_axis: Optional[np.generic],
    ):
        num_palettes = lut.shape[-2]
        vector_size = lut.shape[-1]
        original_lut_shape = lut.shape
        block_size = [indices.shape[idx] // lut.shape[idx] for idx in range(len(indices.shape))]

        if vector_axis is not None and vector_axis < 0:
            vector_axis += len(indices.shape)

        lut = lut.reshape(-1, num_palettes, vector_size)
        decompressed_res = indices.astype(lut.dtype)
        if vector_size > 1:
            # Tile the vector_axis to make room for the vector retrieved from lut.
            decompressed_res = np.repeat(decompressed_res, vector_size, axis=vector_axis)
        else:
            lut = np.squeeze(lut, axis=-1)

        # TODO (rdar://115061946): Vectorize the computation.
        for table_idx in range(lut.shape[0]):
            # Get the corresponding idx in indices for the current table.
            # For example, if table coord is (1, 3), the corresponding indices should be
            # [1*block_size[0] : 2*block_size[0], 3*block_size[1], 4*block_size[1]].
            original_table_coord = np.unravel_index(table_idx, original_lut_shape[:-2])
            slice_idxes = tuple(
                slice(coord * block_size[idx], (coord + 1) * block_size[idx])
                for idx, coord in enumerate(original_table_coord)
            )
            unquantized_values = lut[table_idx][indices[slice_idxes]]
            if vector_size > 1:
                if vector_axis is None:
                    raise ValueError("vector_axis must be provided for vector lut.")
                # Merge the vector dim into the decompressed values (flatten the vector).
                unquantized_values = np.swapaxes(unquantized_values, vector_axis, -2)
                unquantized_values = unquantized_values.reshape(
                    unquantized_values.shape[:-2] + (-1,)
                )
                unquantized_values = np.swapaxes(unquantized_values, vector_axis, -1)
                # Resize the slice to make room for the merged vector dequantized values.
                slice_idxes = list(slice_idxes)
                resized_slice = slice(
                    slice_idxes[vector_axis].start * vector_size,
                    slice_idxes[vector_axis].stop * vector_size,
                    slice_idxes[vector_axis].step,
                )
                slice_idxes[vector_axis] = resized_slice
            decompressed_res[tuple(slice_idxes)] = unquantized_values

        return decompressed_res


@register_op(opset_version=_IOS18_TARGET)
class constexpr_sparse_to_dense(Operation):
    """
    A compile-time operation that returns a constant output value upon de-sparsification of its constant inputs.

    The differences from iOS16 :py:class:`~.iOS16.constexpr_ops.constexpr_sparse_to_dense` are:
    - In iOS16, the mask parameter is 'const tensor<uint8, [M]>', which is a flat tensor with length
      M, so it requires a parameter `shape` to determine the output shape.
      In iOS18, we use uint1 (0 or 1) to represent bitmask, which packs the bitmask data and costs
      the same memory as the uint8 mask in iOS16, but can explicitly tell the tensor shape. We use
      uint1 instead of bool because bool in MIL uses uint8 as the storage dtype, which costs 8x
      memory compared to uint1.
    - Support more dtypes (int4 and uint4) for the input/output data.

    Parameters
    ----------
    nonzero_data: const tensor<T, [D]> (Required)

    mask: const tensor<uint1, [1..]> (Required)

    Returns
    -------
    const tensor<T, [1..]>

    Attributes
    ----------
    T: int4, uint4, int8, uint8, fp16, fp32
    """

    input_spec = InputSpec(
        nonzero_data=TensorInputType(const=True, type_domain="T"),
        mask=TensorInputType(const=True, type_domain=types.uint1),
    )

    type_domains = {"T": (types.int4, types.uint4, types.int8, types.uint8, types.fp16, types.fp32)}

    @staticmethod
    def decompress(nonzero_data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        decompressed_val = np.zeros_like(mask, dtype=nonzero_data.dtype)
        decompressed_val[mask != 0] = nonzero_data
        return decompressed_val

    @staticmethod
    def _validate_sparse_inputs(nonzero_data: Var, mask: Var):
        if nonzero_data.rank != 1:
            raise ValueError(
                f"Parameter nonzero_data needs to have rank 1, but got {nonzero_data.rank}"
            )
        if mask.val is not None and np.count_nonzero(mask.val) != nonzero_data.shape[0]:
            raise AssertionError(
                "Number of 1s in mask not match number of elements in parameter nonzero_data"
            )

    def type_inference(self):
        self._validate_sparse_inputs(self.nonzero_data, self.mask)
        return types.tensor(self.nonzero_data.dtype, self.mask.shape)

    def materialized_val_inference(self):
        nonzero_data = self.nonzero_data.val
        mask = self.mask.val
        if nonzero_data is None and self.nonzero_data.op.op_type.startswith("constexpr_"):
            nonzero_data = self.nonzero_data.op.materialized_val_inference()
            if isinstance(nonzero_data, tuple) and len(nonzero_data) > 0:
                # For sparse constexpr ops they have two outputs, one for mask and one for val.
                nonzero_data = nonzero_data[1]
        if mask is None and self.mask.op.op_type.startswith("constexpr_"):
            mask = self.mask.op.materialized_val_inference()
            if isinstance(mask, tuple) and len(mask) > 0:
                mask = mask[0]
        return self.decompress(nonzero_data, mask)


@register_op(opset_version=_IOS18_TARGET)
class constexpr_lut_to_sparse(Operation):
    """
    A compile-time operation that returns a constant output value upon de-palettizing its constant inputs.

    This op is a sparse-to-sparse op to support `constexpr_lut_to_dense` on sparse data, where the
    de-palettization is only applied on the nonzero data. Usually it would be followed by a
    `constexpr_sparse_to_dense` op to get the dense tensor. So, parameters of this op are similar to
    `constexpr_sparse_to_dense` and `constexpr_lut_to_dense`. For detailed descriptions
    about its parameters, please refer to iOS 18 :py:class:`~.iOS18.constexpr_ops.constexpr_sparse_to_dense`
    and :py:class:`~.iOS18.constexpr_ops.constexpr_lut_to_dense`.

    This op has two outputs:
        1. the mask of the de-palettized nonzero_data.
        2. the de-palettized nonzero_data.

    Parameters
    ----------
    indices_mask: const tensor<uint1, [1..]> (Required)

    indices_nonzero_data: const tensor<IndicesT, [D]> (Required)

    lut: const tensor<T, [1.., NUM_PALETTES, VECTOR_SIZE]> (Required)
        * NUM_PALETTES needs to be 2^nbits where nbits is indicated by IndicesT.

    vector_axis: const tensor<int32, []> (Optional)
        * vector_axis can be optional if VECTOR_SIZE is 1.

    Returns
    -------
    const tensor<uint1, [1..]>
        * the mask of the de-palettized nonzero_data.
          For scalar palettization, it's the same as the input indices_mask.
          For vector palettization, it's expanded of the indices_mask over axis=vector_axis.
    const tensor<T, [VD]>
        * the de-palettized nonzero_data.
          For scalar palettization, VD=D (same size as indices_nonzero_data).
          For vector palettization, VD=VECTOR_SIZE * D (each entry is expanded by a vector).

    Attributes
    ----------
    IndicesT: uint1, uint2, uint3, uint4, uint6, uint8
    T: uint8, int8, fp16, fp32

    Examples
    ----------
    Assume we have the following inputs:
        indices_mask<uint1, [4, 6]> = [[1, 1, 0, 0, 0, 0],
                                       [1, 1, 0, 0, 0, 1],
                                       [0, 1, 1, 0, 1, 0],
                                       [0, 0, 0, 1, 0, 0]]
        indices_nonzero_data<uint1, [9]> = [0, 1, 1, 0, 1, 1, 0, 0, 1]

        Notice that:
        - The uint1 in `indices_mask` and `indices_nonzero_data` has different meanings. For
          `indices_mask` the dtype is always uint1 to represent bit mask. For `indices_nonzero_data`
          the uint1 means the LUT only has two entries, so only 1 bit is needed to represent indices.
        - The 0 in `indices_mask` and `indices_nonzero_data` has different meanings. For
          `indices_mask` the 0 means empty entry in sparse representation. For `indices_nonzero_data`
          the 0 means index 0 in LUT.

    With the given indices_mask and indices_nonzero_data, an example for "Scalar Palettization":
         lut<fp16, [1, 1, 2, 1]> = [2.0, 3.0] (indices-to-values mapping is {0: 2.0, 1: 3.0})

         The sparse indices in the dense layout would look like:
         0   1   .   .   .   .
         1   0   .   .   .   1
         .   1   0   .   0   .
         .   .   .   1   .   .
         (here "." means spare elements in sparse representation)

         When we apply per-tensor de-palettization with this sparse indices, the `indices_nonzero_data`
         is used to read the values from the LUT as in the dense layout. The output sparse tensor in
         the dense layout would be:
         2.0  3.0   .    .   .    .
         3.0  2.0   .    .   .   3.0
          .   3.0  2.0   .  2.0   .
          .    .    .   3.0  .    .
         The first output would be the same as the indices_mask.
         The second output would be [2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0]

    With the given indices_mask and indices_nonzero_data, an example for "Vector Palettization":
         lut<fp16, [1, 1, 2, 2] = [[2.0, 2.0], [3.0, 3.0]]
         vector_axis = 0

         The first output would be the expanded mask of the indices_mask over axis=0, which is:
         output<uint1, [8, 6]> = [
             [1, 1, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 1],
             [1, 1, 0, 0, 0, 1],
             [0, 1, 1, 0, 1, 0],
             [0, 1, 1, 0, 1, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0, 0],
         ]
         The second output in the dense layout would be:
         2.0  3.0   .    .   .    .
         2.0  3.0   .    .   .    .
         3.0  2.0   .    .   .   3.0
         3.0  2.0   .    .   .   3.0
          .   3.0  2.0   .  2.0   .
          .   3.0  2.0      2.0   .
          .    .    .   3.0  .    .
          .    .    .   3.0  .    .
         It is created by fetching the vector entry from the lut for every bit 1 in the data_mask,
         and filling the vector over axis=0.

    Those two outputs of this op could be passed as inputs to a following `sparse_to_dense` op
    in order to recover the dense weights.
    """

    input_spec = InputSpec(
        indices_mask=TensorInputType(const=True, type_domain=types.uint1),
        indices_nonzero_data=TensorInputType(const=True, type_domain="IndicesT"),
        lut=TensorInputType(const=True, type_domain="T"),
        vector_axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "IndicesT": (types.uint1, types.uint2, types.uint3, types.uint4, types.uint6, types.uint8),
        "T": (types.int8, types.uint8, types.fp16, types.fp32),
    }

    def _validate_inputs(self):
        constexpr_sparse_to_dense._validate_sparse_inputs(
            self.indices_nonzero_data, self.indices_mask
        )
        constexpr_lut_to_dense._validate_lut_inputs(
            self.indices_mask.shape,
            self.indices_nonzero_data.dtype,
            self.lut.shape,
            self.vector_axis,
        )

    def type_inference(self):
        self._validate_inputs()
        output_mask_shape = self.indices_mask.shape
        output_nonzero_data_shape = self.indices_nonzero_data.shape
        vector_size = self.lut.shape[-1]
        if vector_size > 1:
            output_mask_shape = list(output_mask_shape)
            output_mask_shape[self.vector_axis.val] *= vector_size
            output_mask_shape = tuple(output_mask_shape)
            output_nonzero_data_shape = tuple(
                [dim * vector_size for dim in output_nonzero_data_shape]
            )

        output_mask_type = types.tensor(self.indices_mask.dtype, output_mask_shape)
        output_nonzero_data_type = types.tensor(self.lut.dtype, output_nonzero_data_shape)
        return output_mask_type, output_nonzero_data_type

    @staticmethod
    def decompress(
        indices_mask: np.ndarray,
        indices_nonzero_data: np.ndarray,
        lut: np.ndarray,
        vector_axis: Optional[np.generic],
    ):
        indices = constexpr_sparse_to_dense.decompress(indices_nonzero_data, indices_mask)
        output_nonzero_data = constexpr_lut_to_dense.decompress(indices, lut, vector_axis)
        output_mask = indices_mask
        if vector_axis is not None:
            vector_size = lut.shape[-1]
            output_mask = np.repeat(output_mask, vector_size, axis=vector_axis)
        output_nonzero_data = output_nonzero_data[output_mask != 0].flatten()

        return output_mask, output_nonzero_data

    def materialized_val_inference(self):
        vector_axis = self.vector_axis.val if self.vector_axis is not None else None
        return self.decompress(
            self.indices_mask.val, self.indices_nonzero_data.val, self.lut.val, vector_axis
        )


@register_op(opset_version=_IOS18_TARGET)
class constexpr_sparse_blockwise_shift_scale(Operation):
    """
    A compile-time operation that returns a constant output value upon de-quantize (shift-scale) its
    constant inputs.
    This op is a sparse-to-sparse op to support `constexpr_blockwise_shift_scale` on sparse data,
    where the de-quantization is only applied on the nonzero data. Usually it would be followed by a
    `constexpr_sparse_to_dense` op to get the dense tensor. So, parameters of this op are similar to
    `constexpr_sparse_to_dense` and `constexpr_blockwise_shift_scale`. For detailed descriptions
    about its parameters, please refer to iOS 18 :py:class:`~.iOS18.constexpr_ops.constexpr_sparse_to_dense`
    and :py:class:`~.iOS18.constexpr_ops.constexpr_blockwise_shift_scale`.

    This op has two outputs:
         1. the mask of the de-quantized nonzero_data.
         2. the de-quantized nonzero_data.

    Parameters
    -------
    data_mask: const tensor<uint1, [1..]> (Required)

    nonzero_data: const tensor<SrcT, [D]> (Required)

    scale: const tensor<DstT, [1..]> (Required)

    offset: const tensor<OffsetT, [1..]> (Optional)
        * If provided, must have the same shape as the ``scale``.

    Returns
    -------
    const tensor<uint1, [1..]>
         * the mask of the shift-scaled nonzero_data.
    const tensor<DstT, [D]>
         * the shift-scaled nonzero_data.

    Attributes
    -------
    SrcT: int4, uint4, int8, uint8, fp16, fp32
    DstT: fp16, fp32
    OffsetT: int4, uint4, int8, uint8, fp16, fp32

    Examples
    -------
    For example:
        data_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]]
        nonzero_data = [10, 11, 3, 4, 5, 6, 7, 8, 9]
        The sparse tensor in the dense layout would look like:
         10   11    .    .
          3    4    5    .
          .    .    6    7
          8    9    .    .

        When we apply per-channel de-quantization on this sparse tensor, where:
        scale = [[0.1, 0.2, 0.3, 0.4]]
        offset = [[1, 2, 3, 4]]
        The input `nonzero_data` would be dequantized per-column as in the dense layout, and the
        output sparse tensor in the dense layout would be:
         (10-1)*0.1   (11-2)*0.2        .            .
         (10-1)*0.1   (11-2)*0.2        .            .
          (3-1)*0.1    (4-2)*0.2    (5-3)*0.3        .
              .            .        (6-3)*0.3    (7-4)*0.4
          (8-1)*0.1    (9-2)*0.2        .            .

        The first output would be the same as the `data_mask`,
        The second output would be [0.9, 1.8, 0.2, 0.4, 0.6, 0.9, 1.2, 0.7, 1.4].
        The two outputs could be passed as inputs to the following `sparse_to_dense` op in order to
        get the dense weights.
    """

    input_spec = InputSpec(
        data_mask=TensorInputType(const=True, type_domain=types.uint1),
        nonzero_data=TensorInputType(const=True, type_domain="SrcT"),
        scale=TensorInputType(const=True, type_domain="DstT"),
        offset=TensorInputType(const=True, optional=True, type_domain="OffsetT"),
    )

    type_domains = {
        "SrcT": (types.int4, types.uint4, types.int8, types.uint8, types.fp16, types.fp32),
        "DstT": (types.fp16, types.fp32),
        "OffsetT": (types.int4, types.uint4, types.int8, types.uint8, types.fp16, types.fp32),
    }

    def _validate_inputs(self):
        constexpr_sparse_to_dense._validate_sparse_inputs(self.nonzero_data, self.data_mask)
        constexpr_blockwise_shift_scale._validate_shift_scale_inputs(
            self.data_mask.shape, self.nonzero_data.dtype, self.scale, self.offset
        )

    def type_inference(self):
        self._validate_inputs()
        output_mask_shape = self.data_mask.shape
        output_nonzero_data_shape = self.nonzero_data.shape
        output_mask_type = types.tensor(self.data_mask.dtype, output_mask_shape)
        output_nonzero_data_type = types.tensor(self.scale.dtype, output_nonzero_data_shape)
        return output_mask_type, output_nonzero_data_type

    @staticmethod
    def decompress(
        data_mask: np.ndarray,
        nonzero_data: np.ndarray,
        scale: np.ndarray,
        offset: Optional[np.ndarray],
    ):
        data = constexpr_sparse_to_dense.decompress(nonzero_data, data_mask)
        dequantized_data = constexpr_blockwise_shift_scale.decompress(data, scale, offset)
        output_nonzero_data = dequantized_data[data_mask != 0].flatten()
        return data_mask, output_nonzero_data

    def materialized_val_inference(self):
        offset = self.offset.val if self.offset is not None else None
        return self.decompress(self.data_mask.val, self.nonzero_data.val, self.scale.val, offset)


@register_op(opset_version=_IOS18_TARGET)
class constexpr_cast(_constexpr_cast_iOS16):
    """
    A compile-time operation that returns a constant output value upon casting its constant input.

    The only difference between this version and the iOS 16 :py:class:`~.iOS16.constexpr_ops.constexpr_cast` is
    the parameters are treated as inputs, instead of attributes in the MIL backend framework.
    """

    input_spec = InputSpec(
        source_val=TensorInputType(const=True, type_domain=types.fp16),
        output_dtype=TensorInputType(const=True, type_domain=types.str),
    )
