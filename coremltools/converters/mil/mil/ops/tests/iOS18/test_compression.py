#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import math
import re
from typing import List, Tuple

import numpy as np
import pytest

from coremltools import utils
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.defs._utils import promote_input_dtypes
from coremltools.converters.mil.mil.ops.defs.iOS18 import (
    _IOS18_TARGET,
    constexpr_blockwise_shift_scale,
    constexpr_lut_to_dense,
    constexpr_lut_to_sparse,
    constexpr_sparse_blockwise_shift_scale,
    constexpr_sparse_to_dense,
)
from coremltools.converters.mil.mil.ops.tests.iOS18 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.testing_reqs import compute_units


def _convert_to_sub_byte_dtype(data: np.ndarray, sub_byte_dtype: type) -> np.ndarray:
    """Convert data to a specific sub-byte dtype, including shift between signed and unsigned range."""
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError("Input data must be integer.")
    if not types.is_sub_byte(sub_byte_dtype):
        raise ValueError("Target dtype must be a sub-byte dtype.")

    original_signed = np.issubdtype(data.dtype, np.signedinteger)
    target_signed = not sub_byte_dtype.is_unsigned()
    if original_signed != target_signed:
        shift = 2 ** (sub_byte_dtype.get_bitwidth() - 1)
        if original_signed:
            data += shift
        else:
            data -= shift

    dtype_range = types.type_mapping.builtin_to_range(sub_byte_dtype)
    if np.max(data) > dtype_range.high:
        raise ValueError(
            f"Data has element {np.max(data)}, which is larger than the lower-bound {dtype_range.high}"
        )
    if np.min(data) < dtype_range.low:
        raise ValueError(
            f"Data has element {np.min(data)}, which is smaller than the lower-bound {dtype_range.low}"
        )

    return data.astype(types.nptype_from_builtin(sub_byte_dtype))


def _infer_lut_shape(
    indices_shape: Tuple[int, ...], block_sizes: Tuple[int, ...], nbits: int, vector_size: int
):
    """Infer the shape of look-up-table (LUT)."""
    lut_shape = []
    for axis, dim_size in enumerate(indices_shape):
        lut_dim_size = 1 if block_sizes[axis] == 0 else dim_size // block_sizes[axis]
        lut_shape.append(lut_dim_size)
    lut_shape.extend([2**nbits, vector_size])
    return lut_shape


class TestConstexprBlockwiseDequantize:
    def test_builder_eval_basic_8bit(self):
        @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
        def prog():
            return mb.constexpr_blockwise_shift_scale(
                data=np.array([4, 8, 10, 13, 24, 5, 6, 9]).reshape((1, 2, 4)).astype(np.int8),
                scale=np.array([4, 8]).reshape((1, 1, 2)).astype(np.float16),
                offset=np.array([4, 0]).reshape((1, 1, 2)).astype(np.int8),
            )

        main_func = prog.functions["main"]
        constexpr_blockwise_shift_scale_op = main_func.find_ops(
            op_type="constexpr_blockwise_shift_scale"
        )[0]
        decompressed_res = (
            np.array([0, 16, 80, 104, 80, 4, 48, 72]).reshape((1, 2, 4)).astype(np.float16)
        )
        np.testing.assert_allclose(
            decompressed_res,
            constexpr_blockwise_shift_scale_op.outputs[0].op.materialized_val_inference(),
        )

    @pytest.mark.parametrize(
        "scale_shape_output, quantized_dtype",
        itertools.product(
            [
                ((1, 1, 2), [0, -16, -64, 0, -40, -16, -24, 0]),
                ((1, 2, 1), [0, -16, -48, -16, -48, 0, -24, 0]),
            ],
            ["int4", "uint4"],
        ),
    )
    def test_builder_eval_basic_4bit(
        self, scale_shape_output: Tuple[Tuple[int], List[int]], quantized_dtype: str
    ):
        quantized_dtype = types.string_to_builtin(quantized_dtype)
        scale_shape, expected_output = scale_shape_output

        @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
        def prog():
            quantized_data = _convert_to_sub_byte_dtype(
                np.array([4, 0, -8, 0, -6, 0, -3, 0]).reshape((1, 2, 4)), quantized_dtype
            )
            offset = _convert_to_sub_byte_dtype(
                np.array([4, 0]).reshape(scale_shape), quantized_dtype
            )
            quantized_data = mb.const(val=quantized_data, name="quantized_data")
            offset = mb.const(val=offset, name="offset")
            return mb.constexpr_blockwise_shift_scale(
                data=quantized_data,
                scale=np.array([4, 8]).reshape(scale_shape).astype(np.float32),
                offset=offset,
            )

        constexpr_blockwise_shift_scale_op = prog.functions["main"].find_ops(
            op_type="constexpr_blockwise_shift_scale"
        )[0]
        np.testing.assert_allclose(
            np.array(expected_output).reshape((1, 2, 4)).astype(np.float32),
            constexpr_blockwise_shift_scale_op.outputs[0].op.materialized_val_inference(),
        )

    def test_builder_eval_basic_no_offset(self):
        @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
        def prog():
            quantized_data = mb.const(
                val=np.array([4, 0, -8, 0, -6, 0, -3, 0])
                .reshape((1, 2, 4))
                .astype(types.np_int4_dtype),
                name="quantized_data",
            )
            return mb.constexpr_blockwise_shift_scale(
                data=quantized_data,
                scale=np.array([4, 8]).reshape((1, 1, 2)).astype(np.float32),
            )

        constexpr_blockwise_shift_scale_op = prog.functions["main"].find_ops(
            op_type="constexpr_blockwise_shift_scale"
        )[0]
        np.testing.assert_allclose(
            np.array([16, 0, -64, 0, -24, 0, -24, 0]).reshape((1, 2, 4)).astype(np.float32),
            constexpr_blockwise_shift_scale_op.outputs[0].op.materialized_val_inference(),
        )

    @pytest.mark.parametrize(
        "nbits, block_size, mode",
        itertools.product(
            (4, 8),
            (1, 2, 4),
            ("linear_symmetric", "linear"),
        ),
    )
    def test_builder_eval_numerical_stress(self, nbits, block_size, mode):
        nbits_range_max = 2 ** (nbits - 1) - 1
        nbits_range_min = -nbits_range_max
        if mode == "linear":
            nbits_range_min -= 1

        nbits_range = nbits_range_max - nbits_range_min
        # As small-bit quantization has a lot of information loss, we use int input to make the
        # information loss less critical when comparing the dequantized data with original data.
        original_data = (
            np.random.randn(2, 3, 8)
            if block_size == 1
            else np.random.randint(nbits_range_min, nbits_range_max, (2, 3, 8))
        )

        scaled_data = original_data.flatten()
        scales = []
        zero_points = []
        for i in range(0, scaled_data.size, block_size):
            block_data = scaled_data[i : i + block_size]
            offset = 0

            if mode == "linear_symmetric":
                block_range = np.max(np.abs(block_data)) * 2
            else:
                assert mode == "linear"
                # For the linear mode, we need to make sure the data range contains `0`.
                block_max = np.maximum(0.0, np.max(block_data))
                block_min = np.minimum(0.0, np.min(block_data))
                block_range = block_max - block_min
                offset = (
                    (nbits_range_min * block_max - nbits_range_max * block_min) / block_range
                    if block_range != 0.0
                    else 0.0
                )
                zero_points.append(offset)

            block_scale = block_range / nbits_range
            scales.append(block_scale)
            scaled_data[i : i + block_size] = np.round(block_data / block_scale + offset)
        scaled_data = np.minimum(scaled_data, nbits_range_max)
        scaled_data = np.maximum(scaled_data, nbits_range_min)
        scaled_data = scaled_data.reshape(original_data.shape).astype(np.int8)
        scales_shape = original_data.shape[:-1] + (original_data.shape[-1] // block_size,)

        @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
        def prog():
            quantized_data = scaled_data
            if nbits == 4:
                quantized_data = mb.const(val=quantized_data.astype(types.np_int4_dtype))
            return mb.constexpr_blockwise_shift_scale(
                data=quantized_data,
                scale=np.array(scales).reshape(scales_shape).astype(np.float32),
                offset=None
                if mode == "linear_symmetric"
                else np.array(zero_points).reshape(scales_shape).astype(np.float32),
            )

        constexpr_blockwise_shift_scale_op = prog.functions["main"].find_ops(
            op_type="constexpr_blockwise_shift_scale"
        )[0]

        if block_size == 1:
            # With block_size==1, the quantization will not have information loss.
            atol, rtol = 1e-06, 1e-06
        elif nbits > 4 and block_size < 3:
            # When block size is small and nbits is large, the information loss is limited.
            atol, rtol = 1e-04, 1e-04
        else:
            atol, rtol = 1e-02, 1e-02

        dequantized_data = constexpr_blockwise_shift_scale_op.outputs[
            0
        ].op.materialized_val_inference()
        if np.issubdtype(original_data.dtype, np.integer):
            dequantized_data = np.round(dequantized_data)
        np.testing.assert_allclose(
            original_data,
            dequantized_data,
            atol=atol,
            rtol=rtol,
        )

    def test_builder_eval_invalid_parameter(self):
        with pytest.raises(
            ValueError, match=r"Parameter 'data' needs to have at least rank 1, but got scalar."
        ):

            @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
            def prog():
                return mb.constexpr_blockwise_shift_scale(
                    data=np.int8(10),
                    scale=np.float32(2.0),
                )

        with pytest.raises(
            ValueError, match=r"Parameter 'data' and 'scale' need to have the same rank"
        ):

            @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
            def prog():
                return mb.constexpr_blockwise_shift_scale(
                    data=np.int8(10),
                    scale=np.array([1, 2]).astype(np.float32),
                )

        with pytest.raises(
            ValueError,
            match=r"Number of scales along each dimension should be a "
            r"factor of corresponding dimension size of 'data'.",
        ):

            @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
            def prog():
                return mb.constexpr_blockwise_shift_scale(
                    data=np.array([1, 2]).reshape((1, 2)).astype(np.int8),
                    scale=np.array([1, 2]).reshape((2, 1)).astype(np.float16),
                )

        with pytest.raises(
            ValueError,
            match=r"Invalid parameter 'offset'; the shape of 'offset' "
            r"should match the shape of 'scale'",
        ):

            @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
            def prog():
                return mb.constexpr_blockwise_shift_scale(
                    data=np.array([1, 2]).astype(np.int8),
                    scale=np.array([1, 2]).astype(np.float16),
                    offset=np.array([1, 2]).reshape((1, 2)).astype(np.int8),
                )

        with pytest.raises(
            ValueError,
            match=r"Invalid parameter 'offset'; the dtype of 'offset' "
            r"should match the dtype of 'data'",
        ):

            @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
            def prog():
                return mb.constexpr_blockwise_shift_scale(
                    data=np.array([1, 2]).astype(types.nptype_from_builtin(types.int4)),
                    scale=np.array([1, 2]).astype(np.float16),
                    offset=np.array([1, 2]).astype(np.int8),
                )

        # When the offset is float, it doesn't need to have the same dtype as data.
        @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
        def prog():
            return mb.constexpr_blockwise_shift_scale(
                data=np.array([1, 2]).astype(types.nptype_from_builtin(types.int4)),
                scale=np.array([1, 2]).astype(np.float16),
                offset=np.array([1, 2]).astype(np.float32),
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, nbits, has_offset",
        itertools.product(compute_units, backends, [4, 8], [True, False]),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, nbits, has_offset):
        x_val = np.ones(1).astype(np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        np_dtype = types.nptype_from_builtin(types.string_to_builtin(f"int{nbits}"))

        if nbits == 8:
            data_val = [4, 8, 10, 13, 24, 5, 6, 9]
        elif nbits == 4:
            data_val = [2, 3, 5, 7, 6, 5, 3, 1]
        data = np.array(data_val).reshape((1, 2, 4)).astype(np_dtype)

        if has_offset is True:
            if nbits == 8:
                offset_val = [4, 0]
            elif nbits == 4:
                offset_val = [1, 0]
        else:
            offset_val = [0, 0]
        offset = np.array(offset_val).reshape((1, 1, 2)).astype(np_dtype)

        scale = np.array([1, 2]).reshape((1, 1, 2)).astype(np.float32)

        # Calculate expected output based on op definition.
        expected_output = np.zeros(data.shape)
        for n in range(0, 1):
            for i in range(0, data.shape[0]):
                for j in range(0, data.shape[1]):
                    for k in range(0, data.shape[2]):
                        i0 = math.floor(i / (data.shape[0] / scale.shape[0]))
                        j0 = math.floor(j / (data.shape[1] / scale.shape[1]))
                        k0 = math.floor(k / (data.shape[2] / scale.shape[2]))
                        expected_output[i][j][k] = (
                            scale[i0][j0][k0] * (data[i][j][k] - offset[i0][j0][k0]) + 1
                        )

        def build(x):
            output = mb.constexpr_blockwise_shift_scale(
                data=data,
                scale=scale,
                offset=offset,
            )
            return mb.add(x=x, y=output)

        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (types.fp32,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, dtype, block_sizes, has_offset",
        itertools.product(
            compute_units,
            backends,
            ["int4", "uint4", "int8", "uint8", "fp16"],
            [(0, 1, 1, 1), (0, 0, 0, 2), (0, 0, 0, 0), (1, 1, 1, 1), (0, 4, 2, 0), (4, 8, 16, 8)],
            [True, False],
        ),
    )
    def test_builder_to_backend_stress(self, compute_unit, backend, dtype, block_sizes, has_offset):
        """
        Use constexpr_blockwise_shift_scale op's value inference to check backends outputs.

        Following combinations will fail if enable BNNS (rdar://125854036).
        - dtype = 'uint4'/'int4', block_sizes = (1, 1, 1, 1)
        - dtype = 'uint4'/'int4', block_sizes = (0, 1, 1, 1)
        """
        quantized_data_shape = (4, 8, 16, 8)
        builtin_dtype = types.string_to_builtin(dtype)
        np_dtype = types.nptype_from_builtin(builtin_dtype)

        if types.is_int(builtin_dtype):
            data_range = types.type_mapping.builtin_to_range(builtin_dtype)
            quantized_data = np.random.randint(
                low=data_range.low, high=data_range.high + 1, size=quantized_data_shape
            ).astype(np_dtype)
        else:
            quantized_data = np.random.rand(*quantized_data_shape).astype(np_dtype)

        scale_shape = [
            1 if block_sizes[axis] == 0 else dim_size // block_sizes[axis]
            for axis, dim_size in enumerate(quantized_data.shape)
        ]
        scale = np.random.rand(*scale_shape)
        offset = None
        if has_offset:
            if types.is_int(builtin_dtype):
                offset = np.random.randint(
                    low=data_range.low, high=data_range.high + 1, size=scale.shape
                ).astype(np_dtype)
            else:
                offset = np.random.rand(*scale.shape).astype(np_dtype)

        def build(x):
            output = mb.constexpr_blockwise_shift_scale(
                data=quantized_data,
                scale=scale,
                offset=offset,
            )
            return mb.add(x=x, y=output)

        x_val = np.ones_like(quantized_data).astype(np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        expected_output = (
            constexpr_blockwise_shift_scale.decompress(quantized_data, scale, offset) + 1
        )

        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (types.fp32,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestConstexprLut:
    @staticmethod
    def _pad_lut_for_nbits_requirements(lut: np.ndarray, nbits: int):
        """
        Make the number of palettes in lut size (second last dim) meet the 2^nbits requirement.

        This util function is needed before we add all uint sub-byte dtypes.
        """
        pad_shape = lut.shape[:-2] + (2**nbits - lut.shape[-2], lut.shape[-1])
        return np.concatenate((lut, np.zeros(pad_shape)), axis=-2)

    @staticmethod
    def _generate_lut(shape: Tuple[int, ...]):
        """It follows the MIL test cases."""
        total_num = np.prod(shape)
        lut = np.arange(min(total_num, 128))
        if total_num > lut.size:
            lut = np.concatenate((lut, np.ones(total_num - lut.size) * 127))
        return lut.reshape(shape)

    @pytest.mark.parametrize("nbits", [1, 2, 3, 4, 6, 8])
    def test_builder_eval_channelwise_lut(self, nbits):
        """
        Test channel-wise lut with first axis as channel axis (the first dim of lut has size > 1).

        indices = tensor<uint2, [6, 2]>>([2, 3, 3, 0, 1, 0, 3, 0, 2, 1, 0, 3])
        lut = tensor<int8, [2, 1, 4, 1]>([1, 5, 9, 13, 2, 10, 18, 26])

        It is effectively a 2-group 2-bit scalar palettization.
        The output shape would be [6, 2], which is the same as the indices shape.
        The output tensor values are:
        [[lut0[2]->9,  lut0[3]->13],
          [lut0[3]->13, lut0[0]->1],
          [lut0[1]->5,  lut0[0]->1],
          [lut1[3]->26, lut1[0]->2],
          [lut1[2]->18, lut1[1]->10],
          [lut1[0]->2,  lut1[3]->26]]
        """

        @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
        def prog():
            if nbits == 1:
                indices = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1]).reshape((6, 2))
                lut = np.array([1, 5, 9, 13]).reshape((2, 1, 2, 1)).astype(np.int8)
            else:
                indices = np.array([2, 3, 3, 0, 1, 0, 3, 0, 2, 1, 0, 3]).reshape((6, 2))
                lut = self._pad_lut_for_nbits_requirements(
                    np.array([1, 5, 9, 13, 2, 10, 18, 26]).reshape((2, 1, 4, 1)).astype(np.int8),
                    nbits=nbits,
                )
            indices_np_dtype = types.nptype_from_builtin(types.string_to_builtin(f"uint{nbits}"))
            indices = indices.astype(indices_np_dtype)
            return mb.constexpr_lut_to_dense(indices=indices, lut=lut)

        constexpr_lut_to_dense_op = prog.functions["main"].find_ops(
            op_type="constexpr_lut_to_dense"
        )[0]
        if nbits == 1:
            decompressed_res = np.array([1, 5, 5, 1, 5, 1, 13, 9, 13, 13, 9, 13])
        else:
            decompressed_res = np.array([9, 13, 13, 1, 5, 1, 26, 2, 18, 10, 2, 26])
        decompressed_res = decompressed_res.reshape((6, 2)).astype(np.int8)
        np.testing.assert_allclose(
            decompressed_res, constexpr_lut_to_dense_op.outputs[0].op.materialized_val_inference()
        )

    @pytest.mark.parametrize("compute_unit, backend, vector_axis", itertools.product(compute_units, backends, (0, 1, 2, -1)))
    def test_builder_eval_vector_lut(self, compute_unit, backend, vector_axis):
        """
        Test vector lut on different axis.

        indices = [
                    [
                       [4, 8], -> group 0
                       [10, 13], -> group 0
                       [24, 5], -> group 1
                       [6, 9] -> group 1
                    ],
                    [
                       [13, 31], -> group 0
                       [17, 7], -> group 0
                       [2, 8], -> group 1
                       [3, 1] -> group 1
                    ]
                  ]
        """
        def build():
            return mb.constexpr_lut_to_dense(
                indices=np.array([4, 8, 10, 13, 24, 5, 6, 9, 13, 31, 17, 7, 2, 8, 3, 1])
                .reshape((2, 4, 2))
                .astype(np.uint8),
                lut=self._generate_lut(shape=(1, 2, 1, 256, 3)),
                vector_axis=vector_axis,
            )

        prog = mb.program(input_specs=[], opset_version=_IOS18_TARGET)(build)
        constexpr_lut_to_dense_op = prog.functions["main"].find_ops(
            op_type="constexpr_lut_to_dense"
        )[0]
        if vector_axis == 0:
            decompressed_res = (
                np.array(
                    [
                        12,
                        24,
                        30,
                        39,
                        127,
                        127,
                        127,
                        127,
                        13,
                        25,
                        31,
                        40,
                        127,
                        127,
                        127,
                        127,
                        14,
                        26,
                        32,
                        41,
                        127,
                        127,
                        127,
                        127,
                        39,
                        93,
                        51,
                        21,
                        127,
                        127,
                        127,
                        127,
                        40,
                        94,
                        52,
                        22,
                        127,
                        127,
                        127,
                        127,
                        41,
                        95,
                        53,
                        23,
                        127,
                        127,
                        127,
                        127,
                    ]
                )
                .reshape((2 * 3, 4, 2))
                .astype(np.int8)
            )
        elif vector_axis == 1:
            decompressed_res = (
                np.array(
                    [
                        12,
                        24,
                        13,
                        25,
                        14,
                        26,
                        30,
                        39,
                        31,
                        40,
                        32,
                        41,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        39,
                        93,
                        40,
                        94,
                        41,
                        95,
                        51,
                        21,
                        52,
                        22,
                        53,
                        23,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                    ]
                )
                .reshape((2, 4 * 3, 2))
                .astype(np.int8)
            )
        else:
            decompressed_res = (
                np.array(
                    [
                        12,
                        13,
                        14,
                        24,
                        25,
                        26,
                        30,
                        31,
                        32,
                        39,
                        40,
                        41,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        39,
                        40,
                        41,
                        93,
                        94,
                        95,
                        51,
                        52,
                        53,
                        21,
                        22,
                        23,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                        127,
                    ]
                )
                .reshape((2, 4, 2 * 3))
                .astype(np.int8)
            )
        np.testing.assert_allclose(
            decompressed_res, constexpr_lut_to_dense_op.outputs[0].op.materialized_val_inference()
        )

        run_compare_builder(
            build,
            {},
            input_values={},
            expected_output_types=decompressed_res.shape + (types.fp32,),
            expected_outputs=decompressed_res,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, nbits", itertools.product(compute_units, backends, [2, 3, 4, 6, 8])
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, nbits):
        x_val = np.ones(12).astype(np.float32).reshape(6, 2)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}

        def build(x):
            indices = np.array([2, 3, 3, 0, 1, 0, 3, 0, 2, 1, 0, 3]).reshape((6, 2))
            lut = self._pad_lut_for_nbits_requirements(
                np.array([1, 5, 9, 13, 2, 10, 18, 26]).reshape((2, 1, 4, 1)).astype(np.int8),
                nbits=nbits,
            )
            indices_np_dtype = types.nptype_from_builtin(types.string_to_builtin(f"uint{nbits}"))
            indices = indices.astype(indices_np_dtype)

            output = mb.constexpr_lut_to_dense(
                indices=indices,
                lut=lut,
            )
            return mb.add(x=x, y=output)

        expected_output = np.array([9, 13, 13, 1, 5, 1, 26, 2, 18, 10, 2, 26]).reshape(6, 2) + 1
        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (types.fp32,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, nbits, block_sizes, vector_size, vector_axis, lut_dtype",
        itertools.product(
            compute_units,
            backends,
            [2, 3, 4, 6, 8],
            [(0, 2, 0, 0), (2, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1), (4, 2, 0, 0), (4, 8, 16, 8)],
            [1, 4],
            [0, 1, -1],
            ["fp16", "fp32"],  # TODO (rdar://125859751): Add "int8" and "uint8".
        ),
    )
    def test_builder_to_backend_stress(
        self, compute_unit, backend, nbits, block_sizes, vector_size, vector_axis, lut_dtype
    ):
        """Use constexpr_lut_to_dense op's value inference to check backends outputs."""
        indices_shape = (4, 8, 16, 8)
        builtin_dtype = types.string_to_builtin(f"uint{nbits}")
        np_dtype = types.nptype_from_builtin(builtin_dtype)
        indices = np.random.randint(low=0, high=2**nbits, size=indices_shape).astype(np_dtype)

        lut_np_dtype = types.nptype_from_builtin(types.string_to_builtin(lut_dtype))
        lut_shape = _infer_lut_shape(indices_shape, block_sizes, nbits, vector_size)
        lut = np.random.rand(*lut_shape).astype(lut_np_dtype)

        def build(x):
            return mb.constexpr_lut_to_dense(
                indices=indices,
                lut=lut,
                vector_axis=vector_axis,
            )

        output_shape = list(indices.shape)
        if vector_size > 1:
            output_shape[vector_axis] *= vector_size
        x_val = np.ones(output_shape).astype(np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        expected_output = constexpr_lut_to_dense.decompress(indices, lut, vector_axis=vector_axis)

        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (types.fp32,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestConstexprSparseToDense:
    def test_builder_eval_basic(self):
        @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
        def prog():
            return mb.constexpr_sparse_to_dense(
                nonzero_data=np.array([3.0, 5.0, 4.0]),
                mask=np.array([1, 0, 1, 0, 1, 0]).reshape((2, 3)).astype(types.np_uint1_dtype),
            )

        constexpr_sparse_to_dense_op = prog.functions["main"].find_ops(
            op_type="constexpr_sparse_to_dense"
        )[0]
        decompressed_res = np.array([[3.0, 0.0, 5.0], [0.0, 4.0, 0.0]])
        np.testing.assert_allclose(
            decompressed_res,
            constexpr_sparse_to_dense_op.outputs[0].op.materialized_val_inference(),
        )

    @pytest.mark.parametrize(
        "shape, data_dtype",
        itertools.product(
            ((2, 3, 4), (3, 8), (24,)),
            (types.int4, types.uint4, types.int8, types.uint8, types.fp16, types.fp32),
        ),
    )
    def test_builder_eval_numerical_stress(self, shape, data_dtype):
        np_dtype = types.nptype_from_builtin(data_dtype)

        @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
        def prog():
            return mb.constexpr_sparse_to_dense(
                nonzero_data=np.array([3.0, 5.0, 4.0]).astype(np_dtype),
                mask=np.array([1, 0, 1, 0, 1, 0] + [0] * 18)
                .reshape(shape)
                .astype(types.np_uint1_dtype),
            )

        constexpr_sparse_to_dense_op = prog.functions["main"].find_ops(
            op_type="constexpr_sparse_to_dense"
        )[0]
        decompressed_res = np.array([3, 0, 5, 0, 4, 0] + [0] * 18).reshape(shape).astype(np_dtype)
        np.testing.assert_allclose(
            decompressed_res,
            constexpr_sparse_to_dense_op.outputs[0].op.materialized_val_inference(),
        )

    def test_builder_eval_invalid_parameter(self):
        with pytest.raises(
            ValueError, match="Parameter nonzero_data needs to have rank 1, but got 2"
        ):

            @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
            def prog():
                return mb.constexpr_sparse_to_dense(
                    nonzero_data=np.array([1.0, 5.0, 4.0]).reshape((3, 1)),
                    mask=np.array([1, 1, 1, 0, 0, 0]).reshape((2, 3)).astype(types.np_uint1_dtype),
                )

        with pytest.raises(
            AssertionError,
            match="Number of 1s in mask not match number of elements in parameter nonzero_data",
        ):

            @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
            def prog():
                return mb.constexpr_sparse_to_dense(
                    nonzero_data=np.array([1.0, 5.0, 4.0]),
                    mask=np.array([1, 1, 1, 0, 1, 0]).reshape((2, 3)).astype(types.np_uint1_dtype),
                )

    @pytest.mark.parametrize(
        "compute_unit, backend, data_dtype",
        itertools.product(
            compute_units,
            backends,
            ("fp16", "fp32"),  # TODO (rdar://125859751): Add "int8" and "uint8".
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, data_dtype):
        builtin_dtype = types.string_to_builtin(data_dtype)
        np_dtype = types.nptype_from_builtin(builtin_dtype)
        x_val = np.array([1, 1, 1, 1, 1, 1], dtype=np_dtype).reshape((2, 3))
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape, dtype=builtin_dtype)}

        def build(x):
            nonzero_data = np.array([3.0, 5.0, 4.0]).astype(np_dtype)
            mask = np.array([1, 0, 1, 0, 1, 0]).reshape((2, 3)).astype(types.np_uint1_dtype)

            output = mb.constexpr_sparse_to_dense(
                nonzero_data=nonzero_data,
                mask=mask,
            )
            return mb.add(x=x, y=output)

        expected_output = np.array([[3.0, 0.0, 5.0], [0.0, 4.0, 0.0]]).astype(np_dtype) + 1
        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (builtin_dtype,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, sparse_ratio, data_dtype",
        itertools.product(
            compute_units,
            backends,
            [0.01, 0.5, 0.99],
            ["fp16", "fp32"],  # TODO (rdar://125859751): Add "int8" and "uint8".
        ),
    )
    def test_builder_to_backend_stress(self, compute_unit, backend, sparse_ratio, data_dtype):
        """Use constexpr_sparse_to_dense op's value inference to check backends outputs."""
        dense_data_shape = (4, 8, 16, 8)
        mask = np.random.choice(
            [0, 1], size=dense_data_shape, p=[sparse_ratio, 1.0 - sparse_ratio]
        ).astype(types.np_uint1_dtype)
        non_zero_element_num = np.sum(mask)
        data_np_dtype = types.nptype_from_builtin(types.string_to_builtin(data_dtype))
        nonzero_data = np.random.rand(non_zero_element_num).astype(data_np_dtype)

        def build(x):
            output = mb.constexpr_sparse_to_dense(
                nonzero_data=nonzero_data,
                mask=mask,
            )
            x, output = promote_input_dtypes([x, output])
            return mb.add(x=x, y=output)

        x_val = np.ones_like(mask).astype(np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        expected_output = constexpr_sparse_to_dense.decompress(nonzero_data, mask) + 1

        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (types.fp32,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestConstexprLutToSparse:
    def test_builder_eval_scalar_lut(self):
        """
        indices_mask<uint1, [4, 6]> =
            [[1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0]]
        indices_nonzero_data<uint1, [9]> = [0, 1, 1, 0, 1, 1, 0, 0, 1]
        lut<fp16, [1, 1, 2, 1]> = [2.0, 3.0]

        The output mask is the same as input indices_mask.
        The output sparse tensor in the dense layout is:
             2.0  3.0
             3.0  2.0                3.0
                  3.0  2.0      2.0
                            3.0
        So the output nonzero_data is [2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0].
        """

        @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
        def prog():
            return mb.constexpr_lut_to_sparse(
                indices_mask=np.array(
                    [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 0, 0]]
                ).astype(types.np_uint1_dtype),
                indices_nonzero_data=np.array([0, 1, 1, 0, 1, 1, 0, 0, 1]).astype(
                    types.np_uint1_dtype
                ),
                lut=np.array([2.0, 3.0]).reshape((1, 1, 2, 1)),
            )

        constexpr_lut_to_sparse_op = prog.functions["main"].find_ops(
            op_type="constexpr_lut_to_sparse"
        )[0]
        expected_output_mask = np.array(
            [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 0, 0]]
        )
        expected_output_nonzero_data = np.array([2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0])
        output_mask, output_nonzero_data = constexpr_lut_to_sparse_op.outputs[
            0
        ].op.materialized_val_inference()
        np.testing.assert_allclose(output_mask, expected_output_mask)
        np.testing.assert_allclose(output_nonzero_data, expected_output_nonzero_data)

    def test_builder_eval_vector_lut(self):
        """
        indices_mask<uint1, [4, 6]> =
            [[1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0]]
        indices_nonzero_data<uint1, [9]> = [0, 1, 1, 0, 1, 1, 0, 0, 1]
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
         2.0  3.0
         2.0  3.0
         3.0  2.0                3.0
         3.0  2.0                3.0
              3.0  2.0      2.0
              3.0  2.0      2.0
                        3.0
                        3.0
         It is created by fetching the vector entry from the lut for every bit 1 in the data_mask,
         and filling the vector over axis=0.
        """

        @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
        def prog():
            return mb.constexpr_lut_to_sparse(
                indices_mask=np.array(
                    [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 0, 0]]
                ).astype(types.np_uint1_dtype),
                indices_nonzero_data=np.array([0, 1, 1, 0, 1, 1, 0, 0, 1]).astype(
                    types.np_uint1_dtype
                ),
                lut=np.array([[2.0, 2.0], [3.0, 3.0]]).reshape((1, 1, 2, 2)),
                vector_axis=0,
            )

        constexpr_lut_to_sparse_op = prog.functions["main"].find_ops(
            op_type="constexpr_lut_to_sparse"
        )[0]
        expected_output_mask = np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [0, 1, 1, 0, 1, 0],
                [0, 1, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]
        )
        expected_output_nonzero_data = np.array(
            [
                2.0,
                3.0,
                2.0,
                3.0,
                3.0,
                2.0,
                3.0,
                3.0,
                2.0,
                3.0,
                3.0,
                2.0,
                2.0,
                3.0,
                2.0,
                2.0,
                3.0,
                3.0,
            ]
        )
        output_mask, output_nonzero_data = constexpr_lut_to_sparse_op.outputs[
            0
        ].op.materialized_val_inference()
        np.testing.assert_allclose(output_mask, expected_output_mask)
        np.testing.assert_allclose(output_nonzero_data, expected_output_nonzero_data)

    def test_builder_eval_invalid_parameter(self):
        with pytest.raises(
            AssertionError,
            match="Number of 1s in mask not match number of elements in parameter nonzero_data",
        ):

            @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
            def prog():
                return mb.constexpr_lut_to_sparse(
                    indices_mask=np.array([1, 1, 1, 0, 1, 0])
                    .reshape((2, 3))
                    .astype(types.np_uint1_dtype),
                    indices_nonzero_data=np.array([0, 1, 0]).astype(types.np_uint1_dtype),
                    lut=np.array([2.0, 3.0]).reshape((1, 1, 2, 1)),
                )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "When lut's last dim (VECTOR_SIZE) > 1, the parameter "
                "'vector_axis' need to be provided."
            ),
        ):

            @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
            def prog():
                return mb.constexpr_lut_to_sparse(
                    indices_mask=np.array([1, 1, 1, 0, 1, 0])
                    .reshape((2, 3))
                    .astype(types.np_uint1_dtype),
                    indices_nonzero_data=np.array([0, 1, 0, 1]).astype(types.np_uint1_dtype),
                    lut=np.array([2.0, 3.0, 2.0, 3.0]).reshape((1, 1, 2, 2)),
                )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x_val = np.ones(18).astype(np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}

        def build(x):
            indices_mask = np.array(
                [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 0, 0]]
            ).astype(types.np_uint1_dtype)
            indices_nonzero_data = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1]).astype(
                types.np_uint1_dtype
            )
            lut = np.array([[2.0, 2.0], [3.0, 3.0]]).reshape((1, 1, 2, 2))
            vector_axis = 0

            output_mask, output_nonzero_data = mb.constexpr_lut_to_sparse(
                indices_mask=indices_mask,
                indices_nonzero_data=indices_nonzero_data,
                lut=lut,
                vector_axis=vector_axis,
            )
            return mb.add(x=x, y=output_nonzero_data)

        expected_output = 1 + np.array(
            [
                2.0,
                3.0,
                2.0,
                3.0,
                3.0,
                2.0,
                3.0,
                3.0,
                2.0,
                3.0,
                3.0,
                2.0,
                2.0,
                3.0,
                2.0,
                2.0,
                3.0,
                3.0,
            ]
        )

        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (types.fp32,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, nbits, block_sizes, vector_size, sparse_ratio, lut_dtype",
        itertools.product(
            compute_units,
            backends,
            [2, 3, 4, 6, 8],
            [(0, 1, 1, 1), (0, 0, 0, 2), (0, 0, 0, 0), (1, 1, 1, 1), (0, 4, 2, 0), (4, 8, 16, 8)],
            [1, 4],
            [0.01, 0.5, 0.99],
            ["fp16", "fp32"],  # TODO (rdar://125859751): Add "int8" and "uint8".
        ),
    )
    def test_builder_to_backend_stress(
        self, compute_unit, backend, nbits, block_sizes, vector_size, sparse_ratio, lut_dtype
    ):
        """Use constexpr_lut_to_sparse op's value inference to check backends outputs."""
        indices_shape = (4, 8, 16, 8)
        indices_mask = np.random.choice(
            [0, 1], size=indices_shape, p=[sparse_ratio, 1.0 - sparse_ratio]
        ).astype(types.np_uint1_dtype)
        indices_nonzero_element_num = np.sum(indices_mask)
        indices_np_dtype = types.nptype_from_builtin(types.string_to_builtin(f"uint{nbits}"))
        indices_nonzero_data = np.random.randint(
            low=0, high=2**nbits, size=indices_nonzero_element_num
        ).astype(indices_np_dtype)

        lut_np_dtype = types.nptype_from_builtin(types.string_to_builtin(lut_dtype))
        lut_shape = _infer_lut_shape(indices_shape, block_sizes, nbits, vector_size)
        lut = np.random.rand(*lut_shape).astype(lut_np_dtype)
        vector_axis = 0 if vector_size > 1 else None

        def build(x):
            output_mask, output_nonzero_data = mb.constexpr_lut_to_sparse(
                indices_mask=indices_mask,
                indices_nonzero_data=indices_nonzero_data,
                lut=lut,
                vector_axis=vector_axis,
            )
            x, output_nonzero_data = promote_input_dtypes([x, output_nonzero_data])
            return mb.add(x=x, y=output_nonzero_data)

        output_shape = int(indices_nonzero_element_num * vector_size)
        x_val = np.ones(output_shape).astype(np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        expected_output = (
            constexpr_lut_to_sparse.decompress(
                indices_mask, indices_nonzero_data, lut, vector_axis
            )[1]
            + 1
        )

        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (types.fp32,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestConstexprSparseBlockwiseShiftScale:
    def test_builder_eval_sparse_per_channel(self):
        """
        Test per-channel de-quantization on sparse tensor.

        data_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]]
        nonzero_data = [10, 11, 3, 4, 5, 6, 7, 8, 9]
        scale = [[0.1, 0.2, 0.3, 0.4]]
        offset = [[1, 2, 3, 4]]
        The sparse tensor in the dense layout would look like:
         10   11
          3    4    5
                    6    7
          8    9

        The input `nonzero_data` would be dequantized per-column as in the dense layout, and the
        output sparse tensor in the dense layout would be:
         (10-1)*0.1   (11-2)*0.2
          (3-1)*0.1    (4-2)*0.2    (5-3)*0.3
                                    (6-3)*0.3    (7-4)*0.4
          (8-1)*0.1    (9-2)*0.2

        The first output would be the same as the `data_mask`,
        The second output would be [0.9, 1.8, 0.2, 0.4, 0.6, 0.9, 1.2, 0.7, 1.4]
        """

        @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
        def prog():
            return mb.constexpr_sparse_blockwise_shift_scale(
                data_mask=np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]]).astype(
                    types.np_uint1_dtype
                ),
                nonzero_data=np.array([10, 11, 3, 4, 5, 6, 7, 8, 9]).astype(np.int8),
                scale=np.array([[0.1, 0.2, 0.3, 0.4]]),
                offset=np.array([[1, 2, 3, 4]]).astype(np.int8),
            )

        constexpr_sparse_blockwise_shift_scale_op = prog.functions["main"].find_ops(
            op_type="constexpr_sparse_blockwise_shift_scale"
        )[0]
        expected_output_mask = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]])
        expected_output_nonzero_data = np.array([0.9, 1.8, 0.2, 0.4, 0.6, 0.9, 1.2, 0.7, 1.4])
        output_mask, output_nonzero_data = constexpr_sparse_blockwise_shift_scale_op.outputs[
            0
        ].op.materialized_val_inference()
        np.testing.assert_allclose(output_mask, expected_output_mask)
        np.testing.assert_allclose(output_nonzero_data, expected_output_nonzero_data)

    def test_builder_eval_sparse_per_block(self):
        """
        Test per-block de-quantization on sparse tensor with block size 2.

        data_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]]  # shape [4, 4]
        nonzero_data = [10, 11, 3, 4, 5, 6, 7, 8, 9, 2]
        scale = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]  # shape [4, 2] because block size is [1, 2]
        offset = [[1, 2], [3, 4], [5, 6], [7, 8]]
        The sparse tensor in the dense layout would look like:
         10   11
          3    4    5
                    6    7
          8    9         2

        The input `nonzero_data` would be dequantized per-column as in the dense layout, and the
        output sparse tensor in the dense layout would be:
         (10-1)*0.1   (11-1)*0.1
          (3-3)*0.3    (4-3)*0.3    (5-4)*0.4
                                    (6-6)*0.6    (7-6)*0.6
          (8-7)*0.7    (9-7)*0.7                 (2-8)*0.8

        The first output would be the same as the `data_mask`,
        The second output would be [0.9, 1.0, 0.0, 0.3, 0.4, 0.0, 0.6, 0.7, 1.4, -4.8]
        """

        @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
        def prog():
            return mb.constexpr_sparse_blockwise_shift_scale(
                data_mask=np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]]).astype(
                    types.np_uint1_dtype
                ),
                nonzero_data=np.array([10, 11, 3, 4, 5, 6, 7, 8, 9, 2]).astype(np.int8),
                scale=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]),
                offset=np.array([[1, 2], [3, 4], [5, 6], [7, 8]]).astype(np.int8),
            )

        constexpr_sparse_blockwise_shift_scale_op = prog.functions["main"].find_ops(
            op_type="constexpr_sparse_blockwise_shift_scale"
        )[0]
        expected_output_mask = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]])
        expected_output_nonzero_data = np.array([0.9, 1.0, 0.0, 0.3, 0.4, 0.0, 0.6, 0.7, 1.4, -4.8])
        output_mask, output_nonzero_data = constexpr_sparse_blockwise_shift_scale_op.outputs[
            0
        ].op.materialized_val_inference()
        np.testing.assert_allclose(output_mask, expected_output_mask)
        np.testing.assert_allclose(output_nonzero_data, expected_output_nonzero_data)

    def test_builder_eval_invalid_parameter(self):
        with pytest.raises(
            AssertionError,
            match="Number of 1s in mask not match number of elements in parameter nonzero_data",
        ):

            @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
            def prog():
                return mb.constexpr_sparse_blockwise_shift_scale(
                    data_mask=np.array([1, 1, 1, 0, 1, 0])
                    .reshape((2, 3))
                    .astype(types.np_uint1_dtype),
                    nonzero_data=np.array([0, 1, 0]).astype(np.int8),
                    scale=np.array([[0.1, 0.2, 0.3]]),
                )

        with pytest.raises(
            ValueError,
            match=re.escape("the shape of 'offset' should match the shape of 'scale'"),
        ):

            @mb.program(input_specs=[], opset_version=_IOS18_TARGET)
            def prog():
                return mb.constexpr_sparse_blockwise_shift_scale(
                    data_mask=np.array([1, 1, 1, 0, 1, 0])
                    .reshape((2, 3))
                    .astype(types.np_uint1_dtype),
                    nonzero_data=np.array([0, 1, 0, 1]).astype(np.int8),
                    scale=np.array([[0.1, 0.2, 0.3]]),
                    offset=np.array([[1, 2, 3, 4]]).astype(np.int8),
                )

    @pytest.mark.parametrize(
        "compute_unit, backend, per_block, data_dtype",
        itertools.product(
            compute_units,
            backends,
            (True, False),
            (types.uint4, types.int8, types.uint8, types.fp32),
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, per_block, data_dtype):
        x_val = np.ones(10).astype(np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        np_dtype = types.nptype_from_builtin(data_dtype)

        def build(x):
            data_mask_val = np.array(
                [[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]]
            ).astype(types.np_uint1_dtype)
            nonzero_data_val = np.array([10, 11, 3, 4, 5, 6, 7, 8, 9, 2]).astype(np_dtype)

            if per_block:
                scale_val = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
            else:
                scale_val = np.array([[0.1, 0.2, 0.3, 0.4]])

            if per_block:
                offset_val = np.array([[1, 2], [3, 4], [5, 6], [7, 8]]).astype(np_dtype)
            else:
                offset_val = np.array([[1, 2, 3, 4]]).astype(np_dtype)

            output_mask, output_nonzero_data = mb.constexpr_sparse_blockwise_shift_scale(
                data_mask=data_mask_val,
                nonzero_data=nonzero_data_val,
                scale=scale_val,
                offset=offset_val,
            )
            return mb.add(x=x, y=output_nonzero_data)

        if per_block:
            expected_output = np.array([0.9, 1.0, 0.0, 0.3, 0.4, 0.0, 0.6, 0.7, 1.4, -4.8]) + 1
        else:
            expected_output = np.array([0.9, 1.8, 0.2, 0.4, 0.6, 0.9, 1.2, 0.7, 1.4, -0.8]) + 1
        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (types.fp32,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
            atol=1e-3,
            rtol=1e-3,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_builder_to_backend_corner_case(self, compute_unit, backend):
        """
        This test case uses the real data from a conv model.

        It's for testing the scale/offset is correctly repeated and the joint ops
        materialized_val_inference work as expected.
        """

        def build_weight():
            data_mask = np.array(
                [
                    [[[0, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 0], [1, 1]]],
                    [[[1, 1], [1, 1]], [[1, 1], [0, 0]], [[1, 1], [1, 1]], [[1, 0], [0, 0]]],
                ]
            ).astype(types.np_uint1_dtype)
            data_mask, nonzero_data = mb.constexpr_sparse_blockwise_shift_scale(
                data_mask=data_mask,
                nonzero_data=np.array(
                    [
                        -8,
                        -2,
                        7,
                        -4,
                        -7,
                        -6,
                        -5,
                        2,
                        -6,
                        7,
                        -5,
                        2,
                        -8,
                        -6,
                        -7,
                        -8,
                        -5,
                        -8,
                        6,
                        7,
                        6,
                        -7,
                        7,
                        2,
                        -8,
                    ]
                ).astype(np.int8),
                scale=np.array([[[[0.01955]], [[0.02809]]], [[[0.02898]], [[0.02487]]]]),
                offset=np.array([[[[3]], [[-1]]], [[[-2]], [[-3]]]]).astype(np.int8),
            )
            return mb.constexpr_sparse_to_dense(nonzero_data=nonzero_data, mask=data_mask)

        def build(x):
            return mb.add(x=x, y=build_weight())

        # Get the const expected weight by decompressing val inference from the joint constexpr ops.
        weight_prog = mb.program(input_specs=[], opset_version=_IOS18_TARGET)(build_weight)
        result_op = weight_prog.functions["main"].find_ops(op_type="constexpr_sparse_to_dense")[0]
        expected_weight = result_op.outputs[0].op.materialized_val_inference()

        x_val = np.ones(2 * 4 * 2 * 2).reshape((2, 4, 2, 2)).astype(np.float32)
        expected_output = expected_weight + 1
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        # With joint quant + sparse ops, the backend prediction should match the expected_weight.
        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (types.fp32,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

        # Test conv using joint constexpr ops weight matches using the decompressed const weight.
        def build_conv_with_joint_constexpr_weight(x):
            return mb.conv(x=x, weight=build_weight())

        def build_conv_with_const_weight(x):
            return mb.conv(x=x, weight=expected_weight)

        x_val = np.random.rand(1, 4, 10, 10).astype(np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        mlmodel_conv_with_joint_constexpr_weight = run_compare_builder(
            build_conv_with_joint_constexpr_weight,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=(1, 2, 9, 9) + (types.fp32,),
            frontend_only=True,
            compute_unit=compute_unit,
            backend=backend,
        )
        mlmodel_conv_with_const_weight = run_compare_builder(
            build_conv_with_const_weight,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=(1, 2, 9, 9) + (types.fp32,),
            frontend_only=True,
            compute_unit=compute_unit,
            backend=backend,
        )
        result_1 = mlmodel_conv_with_joint_constexpr_weight.predict({"x": x_val})
        result_2 = mlmodel_conv_with_const_weight.predict({"x": x_val})

        np.testing.assert_allclose(result_1["conv_0"], result_2["conv_0"], rtol=3e-3, atol=3e-4)

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_builder_to_backend_no_offset(self, compute_unit, backend):
        """
        Test per-channel de-quantization on sparse tensor without offset.

        data_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]]
        nonzero_data = [10, 11, 3, 4, 5, 6, 7, 8, 9]
        scale = [[0.1, 0.2, 0.3, 0.4]]
        The sparse tensor in the dense layout would look like:
         10   11
          3    4    5
                    6    7
          8    9

        The input `nonzero_data` would be dequantized per-column as in the dense layout, and the
        output sparse tensor in the dense layout would be:
         (10)*0.1   (11)*0.2
          (3)*0.1    (4)*0.2    (5)*0.3
                                (6)*0.3    (7)*0.4
          (8)*0.1    (9)*0.2

        The first output would be the same as the `data_mask`,
        The second output would be [1.0, 1.1, 0.3, 0.8, 1.5, 1.8, 2.8, 0.8, 1.8]
        """
        data_dtype = types.int8
        x_val = np.ones(9).astype(np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        np_dtype = types.nptype_from_builtin(data_dtype)

        def build(x):
            data_mask_val = np.array(
                [[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]]
            ).astype(types.np_uint1_dtype)
            nonzero_data_val = np.array([10, 11, 3, 4, 5, 6, 7, 8, 9]).astype(np_dtype)
            scale_val = np.array([[0.1, 0.2, 0.3, 0.4]])

            output_mask, output_nonzero_data = mb.constexpr_sparse_blockwise_shift_scale(
                data_mask=data_mask_val,
                nonzero_data=nonzero_data_val,
                scale=scale_val,
            )
            return mb.add(x=x, y=output_nonzero_data)

        expected_output = np.array([1.0, 2.2, 0.3, 0.8, 1.5, 1.8, 2.8, 0.8, 1.8]) + 1
        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (types.fp32,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, dtype, block_sizes, has_offset, sparse_ratio",
        itertools.product(
            compute_units,
            backends,
            ["int4", "uint4", "int8", "uint8", "fp16"],
            [(0, 1, 1, 1), (0, 0, 0, 2), (0, 0, 0, 0), (1, 1, 1, 1), (0, 4, 2, 0), (4, 8, 16, 8)],
            [True, False],
            [0.01, 0.5, 0.99],
        ),
    )
    def test_builder_to_backend_stress(
        self, compute_unit, backend, dtype, block_sizes, has_offset, sparse_ratio
    ):
        """
        Use constexpr_sparse_blockwise_shift_scale op's value inference to check backends outputs.
        """
        quantized_data_shape = (4, 8, 16, 8)
        builtin_dtype = types.string_to_builtin(dtype)
        np_dtype = types.nptype_from_builtin(builtin_dtype)

        data_mask = np.random.choice(
            [0, 1], size=quantized_data_shape, p=[sparse_ratio, 1.0 - sparse_ratio]
        ).astype(types.np_uint1_dtype)
        data_nonzero_element_num = int(np.sum(data_mask))

        if types.is_int(builtin_dtype):
            data_range = types.type_mapping.builtin_to_range(builtin_dtype)
            quantized_data = np.random.randint(
                low=data_range.low, high=data_range.high + 1, size=data_nonzero_element_num
            ).astype(np_dtype)
        else:
            quantized_data = np.random.rand(data_nonzero_element_num).astype(np_dtype)

        scale_shape = [
            1 if block_sizes[axis] == 0 else dim_size // block_sizes[axis]
            for axis, dim_size in enumerate(quantized_data_shape)
        ]
        scale = np.random.rand(*scale_shape)
        offset = None
        if has_offset:
            if types.is_int(builtin_dtype):
                offset = np.random.randint(
                    low=data_range.low, high=data_range.high + 1, size=scale.shape
                ).astype(np_dtype)
            else:
                offset = np.random.rand(*scale.shape).astype(np_dtype)

        def build(x):
            output_mask, output_nonzero_data = mb.constexpr_sparse_blockwise_shift_scale(
                data_mask=data_mask,
                nonzero_data=quantized_data,
                scale=scale,
                offset=offset,
            )
            return mb.add(x=x, y=output_nonzero_data)

        x_val = np.ones_like(quantized_data).astype(np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        expected_output = (
            constexpr_sparse_blockwise_shift_scale.decompress(
                data_mask, quantized_data, scale, offset
            )[1]
            + 1
        )

        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (types.fp32,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestJointCompressionOps:
    @pytest.mark.parametrize(
        "compute_unit, backend, nbits, block_sizes, vector_size, lut_dtype, quant_dtype",
        itertools.product(
            compute_units,
            backends,
            [2, 3, 4, 8],
            [(0, 2, 0, 0), (2, 0, 0, 0), (4, 2, 0, 0)],
            [1, 4],
            ["fp16", "fp32"],
            ["int4", "uint4", "int8", "uint8"],
        ),
    )
    @pytest.mark.skipif(utils._macos_version() <= (15, 1), reason="Bug fixed in macOS 15.2")
    def test_quant_lut(
        self, compute_unit, backend, nbits, block_sizes, vector_size, lut_dtype, quant_dtype
    ):
        """
        Test lut with quantized (int8) entries, which is represented as
            lut(int8) -> constexpr_blockwise_shift_scale -> lut(fp) \
                                                                constexpr_lut_to_dense -> dense(fp)
                                                         indices /
        """

        indices_shape = (4, 8, 16, 8)
        builtin_dtype = types.string_to_builtin(f"uint{nbits}")
        np_dtype = types.nptype_from_builtin(builtin_dtype)
        indices = np.random.randint(low=0, high=2**nbits, size=indices_shape).astype(np_dtype)

        lut_np_dtype = types.nptype_from_builtin(types.string_to_builtin(lut_dtype))
        lut_shape = _infer_lut_shape(indices_shape, block_sizes, nbits, vector_size)
        vector_axis = 0 if vector_size > 1 else None

        quant_builtin_dtype = types.string_to_builtin(quant_dtype)
        quant_np_dtype = types.nptype_from_builtin(quant_builtin_dtype)
        quant_data_range = types.type_mapping.builtin_to_range(quant_builtin_dtype)
        quantized_data = np.random.randint(
            low=quant_data_range.low, high=quant_data_range.high + 1, size=lut_shape
        ).astype(quant_np_dtype)
        scale_shape = tuple([1] * len(lut_shape))
        scale = np.array([2.0]).reshape(scale_shape).astype(lut_np_dtype)
        offset = np.array([3]).reshape(scale_shape).astype(quant_np_dtype)

        def build(x):
            lut = mb.constexpr_blockwise_shift_scale(
                data=quantized_data,
                scale=scale,
                offset=offset,
            )
            output = mb.constexpr_lut_to_dense(
                indices=indices,
                lut=lut,
                vector_axis=vector_axis,
            )
            x, output = promote_input_dtypes([x, output])
            return mb.add(x=x, y=output)

        output_shape = list(indices.shape)
        if vector_size > 1:
            output_shape[vector_axis] *= vector_size
        x_val = np.ones(output_shape).astype(np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        lut = constexpr_blockwise_shift_scale.decompress(quantized_data, scale, offset)
        expected_output = (
            constexpr_lut_to_dense.decompress(indices, lut, vector_axis=vector_axis) + 1
        )

        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (types.fp32,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, nbits, block_sizes, vector_size, sparse_ratio, lut_dtype",
        itertools.product(
            compute_units,
            backends,
            [2, 3, 4, 8],
            [(0, 2, 0, 0), (2, 0, 0, 0), (1, 1, 1, 1), (4, 2, 0, 0)],
            [1, 4],
            [0.01, 0.5, 0.99],
            ["fp16", "fp32"],  # TODO (rdar://125859751): Add "int8" and "uint8".
        ),
    )
    def test_sparse_lut(
        self, compute_unit, backend, nbits, block_sizes, vector_size, sparse_ratio, lut_dtype
    ):
        """Joint constexpr_lut_to_sparse + constexpr_sparse_to_dense."""
        indices_shape = (4, 8, 16, 8)
        indices_mask = np.random.choice(
            [0, 1], size=indices_shape, p=[sparse_ratio, 1.0 - sparse_ratio]
        ).astype(types.np_uint1_dtype)
        indices_nonzero_element_num = np.sum(indices_mask)
        indices_np_dtype = types.nptype_from_builtin(types.string_to_builtin(f"uint{nbits}"))
        indices_nonzero_data = np.random.randint(
            low=0, high=2**nbits, size=indices_nonzero_element_num
        ).astype(indices_np_dtype)

        lut_np_dtype = types.nptype_from_builtin(types.string_to_builtin(lut_dtype))
        lut_shape = _infer_lut_shape(indices_shape, block_sizes, nbits, vector_size)
        lut = np.random.rand(*lut_shape).astype(lut_np_dtype)
        vector_axis = 0 if vector_size > 1 else None

        def build(x):
            output_mask, output_nonzero_data = mb.constexpr_lut_to_sparse(
                indices_mask=indices_mask,
                indices_nonzero_data=indices_nonzero_data,
                lut=lut,
                vector_axis=vector_axis,
            )
            output = mb.constexpr_sparse_to_dense(
                nonzero_data=output_nonzero_data,
                mask=output_mask,
            )
            x, output = promote_input_dtypes([x, output])
            return mb.add(x=x, y=output)

        output_mask, output_nonzero_data = constexpr_lut_to_sparse.decompress(
            indices_mask, indices_nonzero_data, lut, vector_axis
        )
        expected_output = constexpr_sparse_to_dense.decompress(output_nonzero_data, output_mask) + 1

        x_val = np.ones(expected_output.shape).astype(np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (types.fp32,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, dtype, block_sizes, has_offset, sparse_ratio",
        itertools.product(
            compute_units,
            backends,
            ["int4", "uint4", "int8", "uint8", "fp16"],
            [(0, 2, 0, 0), (2, 0, 0, 0), (1, 1, 1, 1), (4, 2, 0, 0)],
            [True, False],
            [0.01, 0.5, 0.99],
        ),
    )
    def test_sparse_quant(
        self, compute_unit, backend, dtype, block_sizes, has_offset, sparse_ratio
    ):
        """Joint constexpr_sparse_blockwise_shift_scale + constexpr_sparse_to_dense."""
        quantized_data_shape = (4, 8, 16, 8)
        builtin_dtype = types.string_to_builtin(dtype)
        np_dtype = types.nptype_from_builtin(builtin_dtype)

        data_mask = np.random.choice(
            [0, 1], size=quantized_data_shape, p=[sparse_ratio, 1.0 - sparse_ratio]
        ).astype(types.np_uint1_dtype)
        data_nonzero_element_num = int(np.sum(data_mask))

        if types.is_int(builtin_dtype):
            data_range = types.type_mapping.builtin_to_range(builtin_dtype)
            quantized_data = np.random.randint(
                low=data_range.low, high=data_range.high + 1, size=data_nonzero_element_num
            ).astype(np_dtype)
        else:
            quantized_data = np.random.rand(data_nonzero_element_num).astype(np_dtype)

        scale_shape = [
            1 if block_sizes[axis] == 0 else dim_size // block_sizes[axis]
            for axis, dim_size in enumerate(quantized_data_shape)
        ]
        scale = np.random.rand(*scale_shape)
        offset = None
        if has_offset:
            if types.is_int(builtin_dtype):
                offset = np.random.randint(
                    low=data_range.low, high=data_range.high + 1, size=scale.shape
                ).astype(np_dtype)
            else:
                offset = np.random.rand(*scale.shape).astype(np_dtype)

        def build(x):
            output_mask, output_nonzero_data = mb.constexpr_sparse_blockwise_shift_scale(
                data_mask=data_mask,
                nonzero_data=quantized_data,
                scale=scale,
                offset=offset,
            )
            output = mb.constexpr_sparse_to_dense(
                nonzero_data=output_nonzero_data,
                mask=output_mask,
            )
            return mb.add(x=x, y=output)

        output_mask, output_nonzero_data = constexpr_sparse_blockwise_shift_scale.decompress(
            data_mask, quantized_data, scale, offset
        )
        expected_output = constexpr_sparse_to_dense.decompress(output_nonzero_data, output_mask) + 1

        x_val = np.ones(expected_output.shape).astype(np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (types.fp32,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, nbits, block_sizes, vector_size, sparse_ratio, lut_dtype, quant_dtype",
        itertools.product(
            compute_units,
            backends,
            [2, 3, 4, 8],
            [(0, 2, 0, 0), (2, 0, 0, 0), (4, 2, 0, 0)],
            [1, 4],
            [0.01, 0.5, 0.99],
            ["fp16", "fp32"],
            ["int4", "uint4", "int8", "uint8"],
        ),
    )
    def test_quant_sparse_lut(
        self,
        compute_unit,
        backend,
        nbits,
        block_sizes,
        vector_size,
        sparse_ratio,
        lut_dtype,
        quant_dtype,
    ):
        """
        Test sparse lut with quantized (int8) entries, which is represented as
            constexpr_blockwise_shift_scale + constexpr_lut_to_sparse + constexpr_sparse_to_dense
        """
        indices_shape = (4, 8, 16, 8)
        indices_mask = np.random.choice(
            [0, 1], size=indices_shape, p=[sparse_ratio, 1.0 - sparse_ratio]
        ).astype(types.np_uint1_dtype)
        indices_nonzero_element_num = np.sum(indices_mask)
        indices_np_dtype = types.nptype_from_builtin(types.string_to_builtin(f"uint{nbits}"))
        indices_nonzero_data = np.random.randint(
            low=0, high=2**nbits, size=indices_nonzero_element_num
        ).astype(indices_np_dtype)

        lut_np_dtype = types.nptype_from_builtin(types.string_to_builtin(lut_dtype))
        lut_shape = _infer_lut_shape(indices_shape, block_sizes, nbits, vector_size)
        vector_axis = 0 if vector_size > 1 else None

        quant_builtin_dtype = types.string_to_builtin(quant_dtype)
        quant_np_dtype = types.nptype_from_builtin(quant_builtin_dtype)
        quant_data_range = types.type_mapping.builtin_to_range(quant_builtin_dtype)
        quantized_data = np.random.randint(
            low=quant_data_range.low, high=quant_data_range.high + 1, size=lut_shape
        ).astype(quant_np_dtype)
        scale_shape = tuple([1] * len(lut_shape))
        scale = np.array([2.0]).reshape(scale_shape).astype(lut_np_dtype)
        offset = np.array([3]).reshape(scale_shape).astype(quant_np_dtype)

        def build(x):
            lut = mb.constexpr_blockwise_shift_scale(
                data=quantized_data,
                scale=scale,
                offset=offset,
            )
            output_mask, output_nonzero_data = mb.constexpr_lut_to_sparse(
                indices_mask=indices_mask,
                indices_nonzero_data=indices_nonzero_data,
                lut=lut,
                vector_axis=vector_axis,
            )
            output = mb.constexpr_sparse_to_dense(
                nonzero_data=output_nonzero_data,
                mask=output_mask,
            )
            x, output = promote_input_dtypes([x, output])
            return mb.add(x=x, y=output)

        lut = constexpr_blockwise_shift_scale.decompress(quantized_data, scale, offset)
        output_mask, output_nonzero_data = constexpr_lut_to_sparse.decompress(
            indices_mask, indices_nonzero_data, lut, vector_axis
        )
        expected_output = constexpr_sparse_to_dense.decompress(output_nonzero_data, output_mask) + 1

        x_val = np.ones(expected_output.shape).astype(np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        run_compare_builder(
            build,
            input_placeholders,
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (types.fp32,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )
