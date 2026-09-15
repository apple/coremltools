# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import warnings

import numpy as np
import pytest

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.defs.iOS18.compression import constexpr_lut_to_dense
from coremltools.optimize import _utils as optimize_utils


class TestUpdateTensorRange:
    @pytest.mark.parametrize(
        "stats",
        [
            {},
            {"my_tensor": {"rmin": float("inf"), "rmax": float("-inf")}},
        ],
    )
    def test_accumulates_running_min_max(self, stats):
        optimize_utils._update_tensor_range("my_tensor", np.array([0.0, 5.0, 10.0]), stats)
        assert stats["my_tensor"]["rmin"] == 0.0
        assert stats["my_tensor"]["rmax"] == 10.0

        optimize_utils._update_tensor_range("my_tensor", np.array([2.0, 3.0, 5.0]), stats)
        assert stats["my_tensor"]["rmin"] == 0.0
        assert stats["my_tensor"]["rmax"] == 10.0

        optimize_utils._update_tensor_range("my_tensor", np.array([-1.0, 7.0, 15.0]), stats)

        assert stats["my_tensor"]["rmin"] == -1.0
        assert stats["my_tensor"]["rmax"] == 15.0


class TestComputeQuantizationParams:
    @pytest.mark.parametrize(
        "quant_mode, rank, block_size",
        itertools.product(
            ["LINEAR", "LINEAR_SYMMETRIC"],
            [1, 2, 3],
            [0, 1, 2],
        ),
    )
    def test_compute_qparams(self, quant_mode, rank, block_size):
        weight_shape = [10] * rank
        weight = np.random.randn(*weight_shape)
        ret = optimize_utils.compute_qparams(
            weight,
            nbits=8,
            signed=True,
            quantization_mode=quant_mode,
            dtype=np.int8,
            block_sizes=[block_size] * rank,
        )
        if quant_mode == "LINEAR_SYMMETRIC":
            assert ret[-1] is None
        else:
            assert ret[-1] is not None

        assert ret[0].shape == weight.shape

    @pytest.mark.parametrize(
        "quant_mode, block_sizes",
        itertools.product(
            ["LINEAR", "LINEAR_SYMMETRIC"],
            [[1, 0], [0, 0], [0, 4]],  # per-channel, per-tensor, blockwise
        ),
    )
    def test_compute_qparams_all_zero_block(self, quant_mode, block_sizes):
        """An all-zero channel/block (e.g. a pruned channel) makes
        ``val_max == val_min``, so the quantization range is 0. The scale must
        not come out as 0: previously ``weight / scale`` and the zero_point
        computation divided by zero and silently produced NaNs. The block must
        instead get a valid nonzero scale and dequantize back to 0. The
        palettization path guards this the same way (see
        ``coreml/_quantization_passes.py``).
        """
        weight = np.random.randn(4, 8).astype(np.float32)
        # Zero out exactly one full quantization block for each granularity so
        # that block's val_max == val_min (the case that used to give scale 0).
        if block_sizes == [1, 0]:
            weight[0, :] = 0.0  # one pruned output channel (per-channel)
        elif block_sizes == [0, 0]:
            weight[:] = 0.0  # whole tensor zero (per-tensor)
        else:  # [0, 4]
            weight[:, 0:4] = 0.0  # one zero column block (blockwise)

        with warnings.catch_warnings():
            # any divide-by-zero RuntimeWarning becomes a test failure
            warnings.simplefilter("error", RuntimeWarning)
            quantized_data, scale, zero_point = optimize_utils.compute_qparams(
                weight,
                nbits=8,
                signed=True,
                quantization_mode=quant_mode,
                dtype=np.int8,
                block_sizes=block_sizes,
            )

        scale = np.asarray(scale)
        quantized_data = np.asarray(quantized_data)
        # a zero scale is not a valid quantization scale
        assert np.all(scale != 0)
        assert np.isfinite(quantized_data.astype(np.float64)).all()

        # the all-zero block must round-trip back to zero
        dequantized = optimize_utils.dequantize_by_scale_and_zp(
            quantized_data, scale, zero_point
        )
        np.testing.assert_allclose(dequantized[weight == 0], 0.0, atol=1e-6)


    @pytest.mark.parametrize("quant_mode", ["LINEAR", "LINEAR_SYMMETRIC"])
    def test_compute_qparams_scale_underflow(self, quant_mode):
        """The scale is computed in the weight dtype, and Core ML weights are
        fp16 by default, so a channel whose range is very small underflows to a
        scale of 0 even though the channel is not all zeros. ``weight / scale``
        then divided by zero and the resulting inf/NaN was cast to the integer
        weight dtype.
        """
        weight = np.random.randn(4, 8).astype(np.float16)
        # a channel small enough that range / (q_max - q_min) underflows in fp16
        weight[0, :] = (np.random.randn(8) * 1e-6).astype(np.float16)

        with warnings.catch_warnings():
            # any divide-by-zero RuntimeWarning becomes a test failure
            warnings.simplefilter("error", RuntimeWarning)
            quantized_data, scale, _ = optimize_utils.compute_qparams(
                weight,
                nbits=8,
                signed=True,
                quantization_mode=quant_mode,
                dtype=np.int8,
                block_sizes=[1, 0],
            )

        # a zero scale is not a valid quantization scale
        assert np.all(np.asarray(scale) != 0)
        assert np.isfinite(np.asarray(quantized_data).astype(np.float64)).all()

    @pytest.mark.parametrize(
        "quant_mode, block_sizes",
        itertools.product(
            ["LINEAR", "LINEAR_SYMMETRIC"],
            [
                [0],
                [4, 5],
                [3, 9],
                [4, 5, 6],
            ],
        ),
    )
    def test_compute_qparams_failure(self, block_sizes, quant_mode):
        weight = np.random.randn(10, 10)
        with pytest.raises(AssertionError):
            ret = optimize_utils.compute_qparams(
                weight,
                nbits=8,
                signed=True,
                quantization_mode=quant_mode,
                dtype=np.int8,
                block_sizes=block_sizes,
            )

            assert ret is not None


class TestFindIndicesForLut:
    def test_basic(self):
        """
        data: [3.01, -7.99, -8.01, 3.02, 3.89, -1.88, -2.02, -6.98]
        lut: [-8, -7, 3, 4, -2]
        expected indices: [2, 0, 0, 2, 3, 4, 4, 1]
        """
        data = np.array([3.01, -7.99, -8.01, 3.02, 3.89, 0.98, 1.98, -6.98], dtype=np.float16)
        lut = np.array([-8, -7, 3, 4], dtype=np.int8).reshape((1, 4, 1))
        expected_indices = np.array([2, 0, 0, 2, 3, 2, 2, 1], dtype=np.uint8)
        indices = optimize_utils.find_indices_for_lut(data, lut)
        np.testing.assert_array_equal(indices, expected_indices)
        assert types.builtin_to_string(types.numpy_type_to_builtin_type(indices.dtype)) == "uint2"

    @pytest.mark.parametrize(
        "nbits, block_sizes",
        itertools.product(
            (2, 3, 4, 8),
            (
                [0],
                [1],
                [2],
                [2, 2],
                [1, 2, 1],
                [0, 2, 2],
                [4, 0, 0, 1],
                [8, 4, 2, 3],
            ),
        ),
    )
    def test_stress(self, nbits, block_sizes):
        """
        As finding indices is the reverse progress of generating data from lut, we first manually
        construct indices and lut, and then generate data from lut and salt it, and finally check
        if the restored indices are identical to the original indices.
        """
        data_shape = [8, 4, 2, 3]
        lut_shape = data_shape + [2**nbits, 1]
        for idx, dim_size in enumerate(data_shape):
            if idx < len(block_sizes):
                lut_shape[idx] = 1 if block_sizes[idx] == 0 else data_shape[idx] // block_sizes[idx]

        nbits_range = types.type_mapping.builtin_to_range(types.string_to_builtin(f"uint{nbits}"))
        lut = np.arange(np.prod(lut_shape)).reshape(lut_shape).astype(np.float32)
        expected_indices = np.random.randint(
            low=nbits_range.low, high=nbits_range.high + 1, size=data_shape, dtype=np.uint8
        )

        data = constexpr_lut_to_dense.decompress(expected_indices, lut, vector_axis=None)
        # Salting the data to manually introduce numerical instability.
        data += np.random.randint(low=0, high=2, size=data.shape) * 0.01
        data -= np.random.randint(low=0, high=2, size=data.shape) * 0.01

        indices = optimize_utils.find_indices_for_lut(data, lut)

        np.testing.assert_array_equal(indices, expected_indices)
        assert (
            types.builtin_to_string(types.numpy_type_to_builtin_type(indices.dtype))
            == f"uint{nbits}"
        )

    def test_vector_basic(self):
        """
        data: [[3.01, -7.99, 2.02, -7.05], [3.02, -8.01, 1.89, -6.88]]
        lut: [[2, -7], [3, -8]]
        expected indices: [[1, 0], [0, 1]]
        """
        data = np.array([[3.01, -7.99, 2.02, -7.05], [1.89, -6.88, 3.02, -8.01]], dtype=np.float16)
        lut = np.array([[2, -7], [3, -8]], dtype=np.int8).reshape((1, 1, 2, 2))
        expected_indices = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        indices = optimize_utils.find_indices_for_lut(data, lut, vector_axis=-1)
        np.testing.assert_array_equal(indices, expected_indices)
        assert types.builtin_to_string(types.numpy_type_to_builtin_type(indices.dtype)) == "uint1"

    @pytest.mark.parametrize(
        "nbits, vector_size, vector_axis, group_size",
        itertools.product(
            (2, 3, 4, 8),
            (1, 2, 4),
            (0, 1, -1),
            (0, 4),
        ),
    )
    def test_vector_stress(self, nbits, vector_size, vector_axis, group_size):
        data_shape = [8, 16, 32]
        lut_shape = [1] * len(data_shape)
        if group_size > 0:
            lut_shape[vector_axis] = data_shape[vector_axis] // group_size
        lut_shape += [2**nbits, vector_size]

        nbits_range = types.type_mapping.builtin_to_range(types.string_to_builtin(f"uint{nbits}"))
        lut = np.arange(np.prod(lut_shape)).reshape(lut_shape).astype(np.float16)

        indices_shape = list(data_shape)
        indices_shape[vector_axis] //= vector_size
        expected_indices = np.random.randint(
            low=nbits_range.low, high=nbits_range.high + 1, size=indices_shape, dtype=np.uint8
        )

        data = constexpr_lut_to_dense.decompress(expected_indices, lut, vector_axis=vector_axis)
        # Salting the data to manually introduce numerical instability.
        data += np.random.randint(low=0, high=2, size=data.shape) * 0.01
        data -= np.random.randint(low=0, high=2, size=data.shape) * 0.01

        indices = optimize_utils.find_indices_for_lut(data, lut, vector_axis=vector_axis)

        np.testing.assert_array_equal(indices, expected_indices)
        assert (
            types.builtin_to_string(types.numpy_type_to_builtin_type(indices.dtype))
            == f"uint{nbits}"
        )


class TestPackUnpackBits:
    def test_pack_basic(self):
        """
        Original data: [-8, 7, 3, 4, -2].
        The 4-bit binary representation for those elements are:
            -8: 1000;
             7: 0111;
             3: 0011
             4: 0100
            -2: 1110
        Hence the packed quantized_data will be 3 bytes long, i.e., 24 bits long, which is:
            0111 1000  0100 0011  0000 1110
        So the packed data is represented by 3 uint8 values: [120, 67, 14].
        """
        original_data = np.array([-8, 7, 3, 4, -2], dtype=np.int8)
        expected_packed_data = np.array([120, 67, 14], dtype=np.uint8)
        packed_data = optimize_utils.pack_elements_into_bits(original_data, nbits=4)
        np.testing.assert_array_equal(packed_data, expected_packed_data)

    def test_pack_basic_2(self):
        original_data = np.array([1, 2, 3, 4, 5], dtype=np.int8)
        expected_packed_data = np.array([33, 67, 5], dtype=np.uint8)
        packed_data = optimize_utils.pack_elements_into_bits(original_data, nbits=4)
        np.testing.assert_array_equal(packed_data, expected_packed_data)

    @pytest.mark.parametrize(
        "nbits, data_dtype, element_num",
        itertools.product(list(range(1, 9)), [np.int8, np.uint8], [1, 3, 20]),
    )
    def test_round_trip_pack_unpack(self, nbits, data_dtype, element_num):
        is_data_signed = np.issubdtype(data_dtype, np.signedinteger)
        low, high = 0, 2**nbits
        if is_data_signed:
            low, high = -(2 ** (nbits - 1)), 2 ** (nbits - 1)
        original_data = np.random.randint(low=low, high=high, size=(element_num,)).astype(
            data_dtype
        )
        packed_data = optimize_utils.pack_elements_into_bits(original_data, nbits)
        restored_data = optimize_utils.restore_elements_from_packed_bits(
            packed_data, nbits, element_num, are_packed_values_signed=is_data_signed
        )
        np.testing.assert_array_equal(restored_data, original_data)
