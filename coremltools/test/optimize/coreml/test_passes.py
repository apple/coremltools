# Copyright (c) 2023, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import os
import re
import tempfile

import cattrs
import numpy as np
import pytest
import torch
import yaml

import coremltools as ct
import coremltools.optimize as cto
import coremltools.optimize.coreml._quantization_passes as quantization
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.graph_pass import PassOption
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.converters.mil.mil.passes.tests.test_passes import CONSTEXPR_FUNCS, CONSTEXPR_OPS
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    compute_snr_and_psnr,
    gen_activation_stats_for_program,
    get_op_types_in_program,
)
from coremltools.optimize.coreml.experimental._post_training_quantization import (
    _get_activation_calibration_stats,
)
from coremltools.optimize.coreml.experimental._quantization_passes import (
    insert_prefix_quantize_dequantize_pair as _insert_prefix_quantize_dequantize_pair,
)


class TestCompressionNumerical:
    """
    This unit test is checking the numerical correctness for the compress/decompress methods
    in the compression graph paths.
    """
    @pytest.mark.parametrize(
        "axis, mode, source_dtype, target_dtype, data_range",
        itertools.product(
            [0, 1, 2, 3, -1],
            ["LINEAR", "LINEAR_SYMMETRIC"],
            [np.float16, np.float32],
            [types.uint8, types.int8],
            [
                [-1., 1.],
                [-3., -1.],
                [1., 3.],
                # Test corner case of same values
                [0., 0.],
                [1., 1.],
                [-1., -1.],
            ]
        ),
    )
    def test_linear_quantizer_compression(self, axis, mode, source_dtype, target_dtype, data_range):
        input_shape = (10, 20, 30, 40)
        low, high = data_range
        val = np.random.uniform(low, high, input_shape).astype(source_dtype)
        params = quantization.linear_quantize_weights.compress(val, axis, mode, target_dtype)
        decompressed_val = quantization.linear_quantize_weights.decompress(params)
        np.testing.assert_allclose(val, decompressed_val, rtol=1e-02, atol=1e-02)

    @pytest.mark.parametrize(
        "nbits, signed, block_size, mode, source_dtype, data_range",
        itertools.product(
            [4, 8],
            [True, False],
            [0, 1, 2, 8, 32],
            ["LINEAR", "LINEAR_SYMMETRIC"],
            [np.float16, np.float32],
            [
                [-1.0, 1.0],
                [-3.0, -1.0],
                [1.0, 3.0],
                [1.0, 1.0],  # Test corner case of same values.
            ],
        ),
    )
    def test_linear_quantizer_compression_blockwise(
        self,
        nbits,
        signed,
        block_size,
        mode,
        source_dtype,
        data_range,
    ):
        """
        This test mainly follows the weights pattern in real life's ML models. However, when compressing
        weights to a small number of bits (such as 4-bit), the information loss is critical, which
        makes the numerical test hard. That's why we adjust the atol and rtol based on nbits and
        block_size values.
        For more comprehensive numerical tests, see `test_linear_quantizer_compression_blockwise_integer`.
        """
        original_data = np.random.uniform(data_range[0], data_range[1], (32, 64)).astype(
            source_dtype
        )

        compressed_params = quantization.linear_quantize_weights.blockwise_compress(
            original_data, nbits, mode, signed, block_sizes=[1, block_size]
        )
        decompressed_val = quantization.linear_quantize_weights.decompress(compressed_params)

        if nbits > 4 and block_size < 3:
            # When block size is small and nbits is large, the information loss is limited.
            atol, rtol = 1e-02, 1e-02
        elif nbits <= 2 and block_size >= 2:
            atol, rtol = 0.5, 0.5
        else:
            atol, rtol = 0.2, 0.2
        np.testing.assert_allclose(original_data, decompressed_val, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "nbits, signed, block_size, mode",
        itertools.product(
            [4, 8],
            [True, False],
            [1, 2, 8, 32],
            ["LINEAR", "LINEAR_SYMMETRIC"],
        ),
    )
    def test_linear_quantizer_compression_blockwise_integer(self, nbits, signed, block_size, mode):
        """
        We use int input because after rounding the dequantized data the numerical loss is less
        critical when comparing it to the original data.
        """
        input_shape = (32, 64)
        nbits_range_max = 2 ** (nbits - 1) - 1
        nbits_range_min = -nbits_range_max
        original_data = np.random.randint(nbits_range_min, nbits_range_max, input_shape).astype(
            np.float32
        )
        compressed_params = quantization.linear_quantize_weights.blockwise_compress(
            original_data, nbits, mode, signed, block_sizes=[1, block_size]
        )
        decompressed_val = quantization.linear_quantize_weights.decompress(compressed_params)
        decompressed_val = np.round(decompressed_val).astype(original_data.dtype)

        assert np.sum(original_data != decompressed_val) / original_data.size < 0.03
        assert np.all(np.abs(original_data - decompressed_val) <= 1)

    def test_linear_quantizer_compression_blockwise_corner_case(self):
        """
        When the input data is [-2, -10, 6, -3], the
            np.round(quantized_data / scale) + np.round(zero_point)
        AND
            np.round(quantized_data / scale + zero_point)
        is different ([-1, -8, 7, -2] vs [0, -8, 7, -1]), while we follow PyTorch to use the former.
        """
        original_data = np.array([-2, -10, 6, -3]).astype(np.float32)
        params = quantization.linear_quantize_weights.blockwise_compress(
            original_data,
            nbits=4,
            block_sizes=[4],
            mode="LINEAR",
            signed=True,
        )
        expected_quantized_data = np.array([-1, -8, 7, -2], dtype=np.int8)
        np.testing.assert_equal(params.data, expected_quantized_data)

    def test_linear_quantizer_compression_blockwise_invalid_original_data(self):
        original_data_not_np_array = [1.0, 2.0]
        with pytest.raises(ValueError, match="Only numpy arrays are supported"):
            quantization.linear_quantize_weights.blockwise_compress(
                original_data_not_np_array,
                nbits=8,
                block_sizes=[2],
                mode="LINEAR",
                signed=True,
            )

        original_data_integer = np.random.randint(0, 10, size=(3, 2))
        with pytest.raises(ValueError, match="Only floating numpy arrays are supported."):
            quantization.linear_quantize_weights.blockwise_compress(
                original_data_integer,
                nbits=8,
                block_sizes=[0, 2],
                mode="LINEAR",
                signed=True,
            )

    def test_linear_quantizer_compression_blockwise_invalid_block_size(self, caplog):
        original_data = np.random.uniform(-1.0, 1.0, (4, 6))

        params = quantization.linear_quantize_weights.blockwise_compress(
            original_data,
            nbits=8,
            block_sizes=[1, 2],
            mode="LINEAR",
            signed=True,
        )
        assert params.scale.shape == (4, 3)

        params = quantization.linear_quantize_weights.blockwise_compress(
            original_data,
            nbits=8,
            block_sizes=[1, 6],
            mode="LINEAR",
            signed=True,
        )
        assert params.scale.shape == (4, 1)

        params = quantization.linear_quantize_weights.blockwise_compress(
            original_data,
            nbits=8,
            block_sizes=[2, 6],
            mode="LINEAR",
            signed=True,
        )
        assert params.scale.shape == (2, 1)

        result = quantization.linear_quantize_weights.blockwise_compress(
            original_data,
            nbits=8,
            block_sizes=[1, 8],
            mode="LINEAR",
            signed=True,
        )
        assert result is None
        expected_warning_msg = "Invalid block_sizes"
        assert any([expected_warning_msg in rec.message for rec in caplog.records])

    @pytest.mark.parametrize(
        "mode, nbits, shape",
        itertools.product(
            ["KMEANS", "UNIFORM", "UNIQUE"],
            [1, 2, 4, 6, 8],
            [
                (1,),
                (1, 1),
                (1, 10),
                (2, 20),
                (3, 7, 9),
                (17, 17, 17),
            ]
        ),
    )
    def test_palettizer_compression(self, mode, nbits, shape):
        val_size = np.prod(shape)
        max_val = 2 ** nbits
        val = np.arange(max_val).tolist()
        val = np.array(val * (val_size // max_val + 1))[:val_size].astype(np.float32)
        params = quantization.palettize_weights.compress(val, mode=mode, nbits=nbits)
        decompressed_val = quantization.palettize_weights.decompress(params)

        # For
        # 1. UNIQUE / KMEANS mode
        # 2. UNIFORM mode with the data range <= tensor size
        # We can perfecting re-construct the original value
        if (mode in ["UNIQUE", "KMEANS"]) or (mode == "UNIFORM" and max_val <= val_size):
            np.testing.assert_allclose(val, decompressed_val, rtol=1e-02, atol=1e-02)

    def test_palettizer_compression_channelwise_basic(self):
        original_data = np.arange(16, dtype=np.float32).reshape((4, 4))

        # Group on axis=0.
        result = quantization.palettize_weights.blockwise_compress(
            original_data, "UNIQUE", nbits=3, block_sizes=[2, 0]
        )
        expected_lut = np.array(
            [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.float32
        ).reshape((2, 1, 8, 1))
        np.testing.assert_array_equal(result.lut, expected_lut)
        expected_indices = np.array(
            [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7]]
        ).astype(np.int8)
        np.testing.assert_array_equal(result.indices, expected_indices)

        # Group on axis=1.
        result = quantization.palettize_weights.blockwise_compress(
            original_data, "UNIQUE", nbits=3, block_sizes=[0, 2]
        )
        expected_lut = np.array(
            [[0, 1, 4, 5, 8, 9, 12, 13], [2, 3, 6, 7, 10, 11, 14, 15]], dtype=np.float32
        ).reshape((1, 2, 8, 1))
        np.testing.assert_array_equal(result.lut, expected_lut)
        expected_indices = np.array(
            [[0, 1, 0, 1], [2, 3, 2, 3], [4, 5, 4, 5], [6, 7, 6, 7]]
        ).astype(np.int8)
        np.testing.assert_array_equal(result.indices, expected_indices)

    @pytest.mark.parametrize(
        "nbits, channel_axis, mode, source_dtype, data_range, channel_group_size",
        itertools.product(
            [1, 2, 3, 4, 6, 8],
            [0, 1, 2, -1],
            ["KMEANS", "UNIFORM"],
            [np.float16, np.float32],
            [
                [-1.0, 1.0],
                [-3.0, -1.0],
                [1.0, 3.0],
                [1.0, 1.0],
            ],
            [0, 1, 2],
        ),
    )
    def test_palettizer_compression_channelwise_stress(
        self, nbits, channel_axis, mode, source_dtype, data_range, channel_group_size
    ):
        if nbits < 8:
            # As sub-byte numerical accuracy loss is significant, we construct palettize-friendly data.
            upper_bound = 2**nbits
            original_data = np.stack(
                [np.arange(upper_bound).reshape((1, upper_bound)) for _ in range(4)],
                axis=channel_axis,
            )
        else:
            original_data = np.random.uniform(data_range[0], data_range[1], (2, 4, 16)).astype(
                source_dtype
            )
        block_sizes = [0] * len(original_data.shape)
        block_sizes[channel_axis] = channel_group_size
        params = quantization.palettize_weights.blockwise_compress(
            original_data,
            mode,
            nbits,
            block_sizes,
        )
        decompressed_val = quantization.palettize_weights.decompress(params)
        if nbits < 8 or mode == "KMEANS":
            np.testing.assert_allclose(original_data, decompressed_val, rtol=3e-4, atol=3e-4)
        else:
            np.testing.assert_array_almost_equal(original_data, decompressed_val, decimal=2)

    @pytest.mark.parametrize(
        "nbits, channel_axis, channel_group_size",
        itertools.product(
            [2, 3, 4, 6],
            [0, 1, -1],
            [0, 1, 2],
        ),
    )
    def test_grouped_channelwise_equivalent_to_blockwise(
        self, nbits, channel_axis, channel_group_size
    ):
        """The grouped channelwise palettization could be expressed as general blockwise."""
        original_data = np.random.randint(low=-256, high=256, size=(16, 16, 2, 2)).astype(
            np.float32
        )

        params_grouped_channelwise = quantization.palettize_weights.grouped_channelwise_compress(
            original_data, "UNIFORM", nbits, channel_axis, channel_group_size
        )
        decompressed_grouped_channelwise = quantization.palettize_weights.decompress(
            params_grouped_channelwise
        )

        block_sizes = [0] * len(original_data.shape)
        block_sizes[channel_axis] = channel_group_size
        params_blockwise = quantization.palettize_weights.blockwise_compress(
            original_data, "UNIFORM", nbits, block_sizes=block_sizes
        )
        decompressed_blockwise = quantization.palettize_weights.decompress(params_blockwise)

        np.testing.assert_allclose(
            np.sort(params_grouped_channelwise.lut, axis=None),
            np.sort(params_blockwise.lut, axis=None),
        )
        np.testing.assert_allclose(decompressed_grouped_channelwise, decompressed_blockwise)

    @pytest.mark.parametrize(
        "nbits, mode",
        itertools.product(
            [2, 3, 4, 6],
            ["KMEANS", "UNIFORM"],
        ),
    )
    def test_tensorwise_equivalent_to_blockwise_zero(self, nbits, mode):
        """The block_size=0 in palettization is equivalent to legacy tensorwise compression."""
        original_data = np.random.randint(low=-256, high=256, size=(16, 16, 2, 2)).astype(
            np.float32
        )
        params_old = quantization.palettize_weights.compress(original_data, mode, nbits)
        decompressed_old = quantization.palettize_weights.decompress(params_old)
        params_new = quantization.palettize_weights.blockwise_compress(
            original_data, mode, nbits, block_sizes=[0] * len(original_data.shape)
        )
        decompressed_new = quantization.palettize_weights.decompress(params_new)
        np.testing.assert_allclose(
            np.sort(params_old.lut, axis=None),
            np.sort(params_new.lut, axis=None),
            atol=5e-5,
            rtol=1e-6,
        )
        np.testing.assert_allclose(decompressed_old, decompressed_new, atol=5e-5, rtol=1e-6)

    @pytest.mark.parametrize(
        "nbits, channel_axis, channel_group_size",
        itertools.product(
            [2, 3, 4],
            [0, 1],
            [1, 2],
        ),
    )
    def test_grouped_channelwise_better_than_tensorwise(
        self, nbits, channel_axis, channel_group_size
    ):
        """The noise introduced by per-tensor lut should be more than grouped-channel-wise lut."""
        original_data = np.random.randint(low=-512, high=512, size=(32, 32, 2, 2)).astype(
            np.float32
        )
        block_sizes_channelwise = [0] * len(original_data.shape)
        block_sizes_channelwise[channel_axis] = channel_group_size
        params_grouped_channelwise = quantization.palettize_weights.blockwise_compress(
            original_data,
            "UNIFORM",
            nbits,
            block_sizes_channelwise,
        )

        block_sizes_per_tensor = [0] * len(original_data.shape)
        params_per_tensor = quantization.palettize_weights.blockwise_compress(
            original_data,
            "UNIFORM",
            nbits,
            block_sizes_per_tensor,
        )
        decompressed_grouped_channelwise = quantization.palettize_weights.decompress(
            params_grouped_channelwise
        )
        decompressed_per_tensor = quantization.palettize_weights.decompress(params_per_tensor)
        snr_grouped_channelwise = compute_snr_and_psnr(
            original_data, decompressed_grouped_channelwise
        )[0]
        snr_per_tensor = compute_snr_and_psnr(original_data, decompressed_per_tensor)[0]
        assert snr_grouped_channelwise > snr_per_tensor

    def test_palettizer_compression_blockwise_invalid(self):
        with pytest.raises(ValueError, match="Only numpy arrays are supported"):
            quantization.palettize_weights.blockwise_compress(10, "KMEANS", 6, [0])
        with pytest.raises(ValueError, match="Invalid nbits."):
            quantization.palettize_weights.blockwise_compress(
                np.random.uniform(-1.0, 1.0, (2, 3, 4)), "KMEANS", nbits=5, block_sizes=[0, 0, 1]
            )

        assert (
            quantization.palettize_weights.blockwise_compress(
                np.random.uniform(-1.0, 1.0, (2, 3, 4)), "KMEANS", nbits=3, block_sizes=[3, 0, 0]
            )
            is None
        )

    def test_block_sparsity_pruning_smoke(self):
        # dim = 0
        val = np.array(
            [
                [1, 3, 4],
                [-6, -7, 2],
                [0, 3, 4],
                [-9, 2, -1],
            ]
        ).astype(np.float32)

        expected_val = np.array(
            [
                [1, 3, 0],
                [-6, -7, 0],
                [0, 0, 0],
                [-9, 0, 0],
            ]
        ).astype(np.float32)

        params = quantization.prune_weights.compress_by_magnitude(
            val,
            target_sparsity=0.5,
            block_size=2,
            dim=0,
        )
        decompressed_val = quantization.prune_weights.decompress(params)
        np.testing.assert_array_equal(decompressed_val, expected_val)

        # dim = 1, with padding
        val = np.array(
            [
                [1, 3, 4, 18, 1],
                [-6, -7, 2, 2, 9],
                [0, 3, 4, 8, 9],
            ]
        ).astype(np.float32)

        expected_val = np.array(
            [
                [0, 0, 4, 18, 0],
                [-6, -7, 0, 0, 9],
                [0, 0, 0, 0, 9],
            ]
        ).astype(np.float32)

        params = quantization.prune_weights.compress_by_magnitude(
            val,
            target_sparsity=0.5,
            block_size=2,
            dim=1,
        )
        decompressed_val = quantization.prune_weights.decompress(params)
        np.testing.assert_array_equal(decompressed_val, expected_val)

    @pytest.mark.parametrize(
        "block_size, target_sparsity, shape, dim",
        itertools.product(
            [2, 5, 10, 17],
            [0.0, 0.1, 0.5, 0.75, 1.0],
            [
                (10, 25),
                (
                    10,
                    5,
                    8,
                ),
                (40, 100, 6, 7),
                (20, 60, 4, 5, 6),
            ],
            [0, 1],
        ),
    )
    def test_block_sparsity_pruning_stress(self, block_size, target_sparsity, shape, dim):
        def _is_int(val):
            return int(val) == val

        val = np.random.rand(*shape)
        rank = len(shape)

        params = quantization.prune_weights.compress_by_magnitude(
            val,
            target_sparsity=target_sparsity,
            block_size=block_size,
            dim=dim,
        )

        if block_size > shape[dim] / 2:
            assert params is None
            return

        decompressed_val = quantization.prune_weights.decompress(params)
        assert decompressed_val.shape == val.shape

        sparsity_percentile = np.sum(decompressed_val == 0) / np.prod(shape)
        if (shape[dim]) % block_size == 0 and _is_int(
            np.prod(shape) // block_size * target_sparsity
        ):
            assert sparsity_percentile == target_sparsity

        val_compress = np.copy(val)
        val_compress[np.where(decompressed_val == 0)] = 0
        np.testing.assert_array_equal(decompressed_val, val_compress)

    def test_n_m_pruning_smoke(self):
        # dim = 1
        val = np.array(
            [
                [1, 3, 4, -3],
                [-6, -7, 2, 4],
                [0, 3, 4, 1],
                [-9, 2, -1, 8],
            ]
        ).astype(np.float32)

        expected_val = np.array(
            [
                [0, 3, 4, 0],
                [0, -7, 0, 4],
                [0, 3, 4, 0],
                [-9, 0, 0, 8],
            ]
        ).astype(np.float32)

        params = quantization.prune_weights.compress_by_nm_sparsity(
            val,
            n_m_ratio=(1, 2),
            dim=1,
        )
        decompressed_val = quantization.prune_weights.decompress(params)
        np.testing.assert_array_equal(decompressed_val, expected_val)

        # dim = 0, with padding
        val = np.array(
            [
                [1, 3, 4, -3, 2, 4],
                [-6, -7, 2, 4, 6, 8],
                [0, 4, 4, 1, -9, -4],
                [-9, 2, -1, 8, 3, 9],
                [-1, 5, 0, 8, 9, -3],
                [-3, 3, 6, 3, 6, -1],
                [2, 1, -2, 8, 2, -6],
            ]
        ).astype(np.float32)

        expected_val = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [-6, -7, 0, 4, 0, 8],
                [0, 0, 4, 0, -9, 0],
                [-9, 0, 0, 0, 0, 9],
                [0, 5, 0, 8, 9, 0],
                [0, 0, 6, 0, 0, 0],
                [2, 1, -2, 8, 2, -6],
            ]
        ).astype(np.float32)

        params = quantization.prune_weights.compress_by_nm_sparsity(
            val,
            n_m_ratio=(2, 3),
            dim=0,
        )
        decompressed_val = quantization.prune_weights.decompress(params)
        print(decompressed_val)
        np.testing.assert_array_equal(decompressed_val, expected_val)

    @pytest.mark.parametrize(
        "n_m_ratio, shape",
        itertools.product(
            [
                (1, 1),
                (0, 2),
                (1, 2),
                (3, 5),
                (5, 10),
                (12, 17),
            ],
            [
                (1, 2),
                (3, 3),
                (
                    10,
                    5,
                    8,
                ),
                (80, 50, 6, 7),
                (40, 30, 4, 5, 6),
            ],
        ),
    )
    def test_n_m_pruning_stress(self, n_m_ratio, shape):
        n, m = n_m_ratio
        val = np.random.rand(*shape)
        rank = len(shape)

        for dim in [0, 1]:
            params = quantization.prune_weights.compress_by_nm_sparsity(
                val,
                n_m_ratio=n_m_ratio,
                dim=dim,
            )

            # We skip the compression if m > channel / 2
            if m > shape[dim] / 2:
                assert params is None
                return

            decompressed_val = quantization.prune_weights.decompress(params)
            assert decompressed_val.shape == val.shape

            sparsity_percentile = np.sum(decompressed_val == 0) / np.prod(shape)
            if (shape[dim]) % m == 0:
                assert sparsity_percentile == n / m

            val_compress = np.copy(val)
            val_compress[np.where(decompressed_val == 0)] = 0
            np.testing.assert_array_equal(decompressed_val, val_compress)

class TestCompressionGraphBackwardCompatibility:
    """
    Most of the numerical tests are already convered in coremltools.tests.ml_program.test_compression_utils.
    This test is checking the basic behavior of the graph pass classes using only global config.
    This test also converts the backward compatibility test for the deprecated ct.compression_utils.
    """
    @staticmethod
    def _get_conv_program():
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 30, 10, 10))], opset_version=ct.target.iOS16
        )
        def prog(x):
            conv_weight = np.random.rand(90, 30, 2, 2).astype(np.float32)
            x = mb.conv(x=x, weight=conv_weight)
            return x

        return prog

    @pytest.mark.parametrize(
        "fake_compression, is_deprecated",
        itertools.product(
            [True, False],
            [True, False],
        )
    )
    def test_affine_quantizer(self, fake_compression, is_deprecated):
        weight_threshold = None if is_deprecated else 0
        op_selector=(lambda const: True) if is_deprecated else None
        op_config = cto.coreml.OpLinearQuantizerConfig(weight_threshold=weight_threshold)
        config = cto.coreml.OptimizationConfig(global_config=op_config, is_deprecated=is_deprecated, op_selector=op_selector)
        quantizer = quantization.linear_quantize_weights(
            config=config, fake_compression=fake_compression
        )
        prog = self._get_conv_program()
        quantizer.apply(prog)
        expected_ops = ["constexpr_affine_dequantize", "conv"] if not fake_compression else ["conv"]
        assert get_op_types_in_program(prog) == expected_ops

    @pytest.mark.parametrize(
        "fake_compression, is_deprecated",
        itertools.product(
            [True, False],
            [True, False],
        )
    )
    def test_weight_pruner(self, fake_compression, is_deprecated):
        weight_threshold = None if is_deprecated else 0
        op_selector=(lambda const: True) if is_deprecated else None
        op_config = cto.coreml.OpMagnitudePrunerConfig(
                weight_threshold=weight_threshold,
                target_sparsity=0.75,
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config, is_deprecated=is_deprecated, op_selector=op_selector)
        quantizer = quantization.prune_weights(
            config=config, fake_compression=fake_compression
        )
        prog = self._get_conv_program()
        quantizer.apply(prog)
        expected_ops = ["constexpr_sparse_to_dense", "conv"] if not fake_compression else ["conv"]
        assert get_op_types_in_program(prog) == expected_ops

    @pytest.mark.parametrize(
        "fake_compression, is_deprecated",
        itertools.product(
            [True, False],
            [True, False],
        )
    )
    def test_weight_palettization(self, fake_compression, is_deprecated):
        weight_threshold = None if is_deprecated else 0
        op_selector=(lambda const: True) if is_deprecated else None
        op_config = cto.coreml.OpPalettizerConfig(
                weight_threshold=weight_threshold,
                mode="uniform",
                nbits=4,
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config, is_deprecated=is_deprecated, op_selector=op_selector)
        quantizer = quantization.palettize_weights(
            config=config, fake_compression=fake_compression
        )
        prog = self._get_conv_program()
        quantizer.apply(prog)
        expected_ops = ["constexpr_lut_to_dense", "conv"] if not fake_compression else ["conv"]
        assert get_op_types_in_program(prog) == expected_ops

class TestCompressionPasses:
    @staticmethod
    def _get_test_program():
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 30, 10, 10))], opset_version=ct.target.iOS16
        )
        def prog(x):
            # weight
            conv_weight = np.random.rand(90, 30, 2, 2).astype(np.float32)
            linear_weight = np.random.rand(70, 81).astype(np.float32)
            conv_transpose_weight = np.random.rand(30, 4, 21, 10).astype(np.float32)

            # graph
            x = mb.conv(x=x, weight=conv_weight, name="conv")
            x = mb.reshape(x=x, shape=(1, 90, 81), name="reshape_1")
            x = mb.linear(x=x, weight=linear_weight, name="linear")
            x = mb.reshape(x=x, shape=(1, 30, 21, 10), name="reshape_2")
            x = mb.conv_transpose(x=x, weight=conv_transpose_weight, name="conv_transpose")
            return x
        return prog

    @staticmethod
    def _get_test_program_2():
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 30, 10, 10))], opset_version=ct.target.iOS16
        )
        def prog(x):
            # weight
            conv1_weight = np.random.rand(40, 30, 2, 2).astype(np.float32)
            conv2_weight = np.random.rand(50, 40, 3, 3).astype(np.float32)
            conv3_weight = np.random.rand(60, 50, 2, 4).astype(np.float32)

            linear1_weight = np.random.rand(80, 60).astype(np.float32)
            linear2_weight = np.random.rand(90, 80).astype(np.float32)

            conv_transpose_weight = np.random.rand(60, 30, 6, 10).astype(np.float32)

            # graph
            x = mb.conv(x=x, weight=conv1_weight, name="conv1")
            x = mb.conv(x=x, weight=conv2_weight, name="conv2")
            x = mb.conv(x=x, weight=conv3_weight, name="conv3")
            x = mb.reshape(x=x, shape=(6, 4, 60), name="reshape1")
            x = mb.linear(x=x, weight=linear1_weight, name="linear1")
            x = mb.linear(x=x, weight=linear2_weight, name="linear2")
            x = mb.reshape(x=x, shape=(1, 30, 6, 12), name="reshape2")
            x = mb.conv_transpose(x=x, weight=conv_transpose_weight, name="conv_transpose")
            return x
        return prog

    @staticmethod
    def _get_test_program_3():
        """An iOS18 program with conv, linear, matmul, and conv_transpose."""

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 30, 10, 10))],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            # weight
            conv_weight = np.random.rand(90, 30, 2, 2).astype(np.float32)
            linear_weight = np.random.rand(70, 81).astype(np.float32)
            matmul_weight = np.random.rand(2, 1, 70, 35).astype(np.float32)
            conv_transpose_weight = np.random.rand(30, 4, 21, 10).astype(np.float32)

            # graph
            x = mb.conv(x=x, weight=conv_weight, name="conv")
            x = mb.reshape(x=x, shape=(1, 90, 81), name="reshape_1")
            x = mb.linear(x=x, weight=linear_weight, name="linear")
            x = mb.matmul(x=x, y=matmul_weight, transpose_y=False, name="matmul")
            x = mb.reshape(x=x, shape=(1, 30, 21, 10), name="reshape_2")
            x = mb.conv_transpose(x=x, weight=conv_transpose_weight, name="conv_transpose")
            return x

        return prog

    @staticmethod
    def _get_test_program_conv():
        """An iOS17 program with conv."""
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 30, 10, 10))], opset_version=ct.target.iOS17
        )
        def prog(x):
            conv_weight = np.random.rand(90, 30, 2, 2).astype(np.float32)
            x = mb.cast(x=x, dtype="fp16")
            x = mb.conv(x=x, weight=conv_weight)
            x = mb.cast(x=x, dtype="fp32")
            return x

        return prog

    @staticmethod
    def _get_test_program_conv_relu():
        """An iOS17 program with conv and relu."""
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 30, 10, 10))], opset_version=ct.target.iOS17
        )
        def prog(x):
            conv_weight = np.random.rand(90, 30, 2, 2).astype(np.float32)
            x = mb.cast(x=x, dtype="fp16")
            x = mb.conv(x=x, weight=conv_weight)
            x = mb.relu(x=x)
            x = mb.cast(x=x, dtype="fp32")
            return x

        return prog

    @staticmethod
    def _get_test_program_add():
        """An iOS17 program with add."""

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 2, 4, 4)), mb.TensorSpec(shape=(1, 2, 4, 4))],
            opset_version=ct.target.iOS17,
        )
        def prog(x1, x2):
            y1 = mb.cast(x=x1, dtype="fp16")
            y2 = mb.cast(x=x2, dtype="fp16")
            y = mb.add(x=y1, y=y2)
            z = mb.cast(x=y, dtype="fp32")
            return z

        return prog

    @staticmethod
    def _get_test_program_avgpool():
        """An iOS17 program with avg_pool"""

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 4, 4))], opset_version=ct.target.iOS17)
        def prog(x):
            # graph
            x = mb.cast(x=x, dtype="fp16")
            x = mb.avg_pool(x=x, kernel_sizes=[1, 1], strides=[1, 1], pad_type="valid")
            x = mb.cast(x=x, dtype="fp32")
            return x

        return prog

    @staticmethod
    def _get_test_program_maxpool():
        """An iOS17 program with max_pool"""

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 4, 4))], opset_version=ct.target.iOS17)
        def prog(x):
            # graph
            x = mb.cast(x=x, dtype="fp16")
            x = mb.max_pool(x=x, kernel_sizes=[1, 1], strides=[1, 1], pad_type="valid")
            x = mb.cast(x=x, dtype="fp32")
            return x

        return prog

    @staticmethod
    def _get_test_mlmodel_conv_relu():
        """A mlmodel with conv, relu"""

        # Prepare torch model.
        inputs = [ct.TensorType(name="data", shape=(5, 10, 4, 4))]
        input_data = [torch.rand(*i.shape.to_list()) for i in inputs]
        m = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=4),
            torch.nn.ReLU(),
        )
        torchmodel = torch.jit.trace(m, input_data)

        # Convert to mlmodel.
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_ONLY,
            compute_precision=ct.precision.FLOAT16,
        )

        return mlmodel

    @staticmethod
    def _get_test_mlmodel_boolean_type():
        """A mlmodel with boolean type intermediate tensor"""

        # Prepare torch model.
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear1 = torch.nn.Linear(28 * 28, 100)
                self.linear2 = torch.nn.Linear(28 * 28, 100)

            def forward(self, img):  # convert + flatten
                y1 = self.linear1(img)
                y2 = self.linear2(img)
                y = torch.logical_and(y1, y2)
                return y

        model = Net()
        inputs = [ct.TensorType(name="data", shape=(1, 28 * 28))]
        input_data = [torch.rand(*i.shape.to_list()) for i in inputs]
        torchmodel = torch.jit.trace(model, input_data)

        # Convert to mlmodel.
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_ONLY,
            compute_precision=ct.precision.FLOAT16,
        )

        return mlmodel

    @staticmethod
    def _get_test_mlmodel_conv_concat():
        """A mlmodel has a concat with 2 inputs and 1 output all surrounded by conv."""

        # Prepare torch model.
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=4)
                self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=4)
                self.conv3 = torch.nn.Conv2d(in_channels=20, out_channels=20, kernel_size=1)

            def forward(self, img):  # convert + flatten
                y1 = self.conv1(img)
                y2 = self.conv2(img)
                y = torch.concat((y1, y2), 0)
                y3 = self.conv3(y)
                return y3

        model = Net()
        inputs = [ct.TensorType(name="data_0", shape=(5, 10, 4, 4))]
        input_data = [torch.rand(*i.shape.to_list()) for i in inputs]
        torchmodel = torch.jit.trace(model, input_data)

        # Convert to mlmodel.
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_ONLY,
            compute_precision=ct.precision.FLOAT16,
        )

        return mlmodel

class TestOptimizationConfig(TestCompressionPasses):
    """
    Test some basic functionality of the OptimizationConfig.
    """
    @pytest.mark.parametrize(
        "compressor_class, fake_compression",
        itertools.product(
            [
                quantization.palettize_weights,
                quantization.prune_weights,
                quantization.linear_quantize_weights,
            ],
            [True, False],
        )
    )
    def test_empty_config(self, compressor_class, fake_compression):
        """
        For an empty config, the compression graph passes should do nothing
        """
        config = cto.coreml.OptimizationConfig()
        compressor = compressor_class(
            config=config, fake_compression=fake_compression
        )
        prog = self._get_test_program()
        compressor.apply(prog)
        expected_ops = ["conv", "reshape", "linear", "reshape", "conv_transpose"]
        assert get_op_types_in_program(prog) == expected_ops

    def test_empty_op_type(self):
        """
        If an op_type config is set to None. The entire class will not be compressed.
        """
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=2),
            op_type_configs={
                "conv": None,
            },
        )
        compressor = quantization.palettize_weights(config=config)
        prog = self._get_test_program()
        compressor.apply(prog)
        expected_ops = [
            "conv",
            "reshape",
            "constexpr_lut_to_dense",
            "linear",
            "reshape",
            "constexpr_lut_to_dense",
            "conv_transpose",
        ]
        assert get_op_types_in_program(prog) == expected_ops
        conv_op = prog.find_ops(op_type="conv")[0]
        assert conv_op.weight.op.op_type == "const"

    def test_empty_op_name(self):
        """
        If an op_name config is set to None. The op instance will not be compressed.
        """
        config = cto.coreml.OptimizationConfig(
            op_type_configs={
                "conv": cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=2),
            },
            op_name_configs={
                "conv1": None,
            },
        )
        compressor = quantization.palettize_weights(config=config)
        prog = self._get_test_program_2()
        compressor.apply(prog)
        expected_ops = [
            "conv",
            "constexpr_lut_to_dense",
            "conv",
            "constexpr_lut_to_dense",
            "conv",
            "reshape",
            "linear",
            "linear",
            "reshape",
            "conv_transpose",
        ]
        assert get_op_types_in_program(prog) == expected_ops
        conv_op = prog.find_ops(op_type="conv")[0]
        assert conv_op.weight.op.op_type == "const"

    def test_config_hierarchy(self):
        """
        This test is checking the graph pass compresses the program correctly according to the following hierarchical order (high -> low):
        1. op name
        2. op type
        3. global
        """
        prog = self._get_test_program_2()

        # global config
        global_config = cto.coreml.OpPalettizerConfig(
                nbits=8,
                mode="KMEANS",
                weight_threshold=100,
        )

        # op type config
        conv_config = cto.coreml.OpPalettizerConfig(
                nbits=6,
                mode="KMEANS",
                weight_threshold=100,
        )
        linear_config = cto.coreml.OpPalettizerConfig(
                nbits=4,
                mode="KMEANS",
                weight_threshold=100,
        )

        # op name config
        conv1_config = cto.coreml.OpPalettizerConfig(
                nbits=2,
                mode="KMEANS",
                weight_threshold=100,
        )
        linear2_config = cto.coreml.OpPalettizerConfig(
                nbits=1,
                mode="KMEANS",
                weight_threshold=100,
        )

        config = cto.coreml.OptimizationConfig()
        config.set_global(global_config)

        config.set_op_type("conv", conv_config)
        config.set_op_type("linear", linear_config)

        config.set_op_name("conv1", conv1_config)
        config.set_op_name("linear2", linear2_config)

        compressor = quantization.palettize_weights(config=config)
        compressor.apply(prog)

        expected_ops = [
            "constexpr_lut_to_dense",
            "conv",
            "constexpr_lut_to_dense",
            "conv",
            "constexpr_lut_to_dense",
            "conv",
            "reshape",
            "constexpr_lut_to_dense",
            "linear",
            "constexpr_lut_to_dense",
            "linear",
            "reshape",
            "constexpr_lut_to_dense",
            "conv_transpose",
        ]
        assert get_op_types_in_program(prog) == expected_ops

        expected_nbits = [2, 6, 6, 4, 1, 8, 8]
        lut_ops = prog.find_ops(op_type="constexpr_lut_to_dense")

        for nbits, op in zip(expected_nbits, lut_ops):
            assert op.lut.val.shape == (2**nbits,)

    def test_mixed_compression_algorithms(self):
        """
        This test is checking a program can be ran under different compression method
        """
        prog = self._get_test_program_2()

        # Run palettization for conv ops
        conv_config = cto.coreml.OpPalettizerConfig(
                nbits=1,
                mode="KMEANS",
                weight_threshold=100,
        )
        config = cto.coreml.OptimizationConfig()
        config.set_op_type("conv", conv_config)

        compressor = quantization.palettize_weights(config=config)
        compressor.apply(prog)

        expected_ops = [
            "constexpr_lut_to_dense",
            "conv",
            "constexpr_lut_to_dense",
            "conv",
            "constexpr_lut_to_dense",
            "conv",
            "reshape",
            "linear",
            "linear",
            "reshape",
            "conv_transpose",
        ]
        assert get_op_types_in_program(prog) == expected_ops

        # Run affine quanitzation for conv1 / linear1. Note that since conv1 is already compressed
        # the quantization makes no affect on it
        op_name_config = cto.coreml.OpLinearQuantizerConfig(
                mode="LINEAR_SYMMETRIC",
                dtype=np.int8,
                weight_threshold=100,
        )
        config = cto.coreml.OptimizationConfig()
        config.set_op_name("conv1", op_name_config)
        config.set_op_name("linear1", op_name_config)

        compressor = quantization.linear_quantize_weights(config=config)
        compressor.apply(prog)

        expected_ops = [
            "constexpr_lut_to_dense",
            "conv",
            "constexpr_lut_to_dense",
            "conv",
            "constexpr_lut_to_dense",
            "conv",
            "reshape",
            "constexpr_affine_dequantize",
            "linear",
            "linear",
            "reshape",
            "conv_transpose",
        ]
        assert get_op_types_in_program(prog) == expected_ops

        # Run sparsification for the whoel program
        global_config = cto.coreml.OpMagnitudePrunerConfig(
                target_sparsity=0.85,
                weight_threshold=100,
        )
        config = cto.coreml.OptimizationConfig(global_config=global_config)

        compressor = quantization.prune_weights(config=config)
        compressor.apply(prog)

        expected_ops = [
            "constexpr_lut_to_dense",
            "conv",
            "constexpr_lut_to_dense",
            "conv",
            "constexpr_lut_to_dense",
            "conv",
            "reshape",
            "constexpr_affine_dequantize",
            "linear",
            "constexpr_sparse_to_dense",
            "linear",
            "reshape",
            "constexpr_sparse_to_dense",
            "conv_transpose",
        ]
        assert get_op_types_in_program(prog) == expected_ops

    @staticmethod
    def test_const_only_used_as_output_skip_compress():
        """
        If the const is only fed to the block output, we skip the compression,
        due to the bug rdar://108274019 ([Bug] constexpr ops cannot be directly fed to block output)
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20, 30))], opset_version=ct.target.iOS16)
        def prog(x):
            val = np.random.rand(10, 20, 30).astype(np.float32)
            const = mb.const(val=val)
            output = mb.add(x=x, y=1.0)
            return output, const

        op_config = cto.coreml.OpPalettizerConfig(
            nbits=2,
            mode="kmeans",
            weight_threshold=0,
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config)
        compressor = quantization.palettize_weights(config=config)
        compressor.apply(prog)
        assert get_op_types_in_program(prog) == ["add"]

    @staticmethod
    def test_const_as_output():
        """
        If the const is fed to the block output and at least one another op, it can still be compressed
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20, 30))], opset_version=ct.target.iOS16)
        def prog(x):
            val = np.random.rand(10, 20, 30).astype(np.float32)
            const = mb.const(val=val)
            output = mb.add(x=x, y=const)
            return output, const

        op_config = cto.coreml.OpPalettizerConfig(
            nbits=2,
            mode="kmeans",
            weight_threshold=0,
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config)
        compressor = quantization.palettize_weights(config=config)
        compressor.apply(prog)
        assert get_op_types_in_program(prog) == ["constexpr_lut_to_dense", "add"]

    @staticmethod
    def test_set_op_name_for_const():
        """
        We can set_op_name for const ops
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 10, 30))], opset_version=ct.target.iOS16)
        def prog(x):
            add_const_1 = np.random.rand(10, 30).astype(np.float32)
            add_const_2 = np.random.rand(10, 30).astype(np.float32)
            const_1 = mb.const(val=add_const_1, name="const_1")
            const_2 = mb.const(val=add_const_2, name="const_2")
            x = mb.add(x=x, y=const_1)
            return mb.add(x=x, y=const_2)

        compressor = quantization.palettize_weights(
            config=cto.coreml.OptimizationConfig(
                global_config=cto.coreml.OpPalettizerConfig(nbits=2, mode="KMEANS", weight_threshold=50),
                op_name_configs={"const_2": cto.coreml.OpPalettizerConfig(nbits=4, mode="KMEANS", weight_threshold=50)}
            )
        )

        compressor.apply(prog)

        expected_ops = [
            "constexpr_lut_to_dense",
            "constexpr_lut_to_dense",
            "add",
            "add",
        ]
        assert get_op_types_in_program(prog) == expected_ops

        expected_nbits = [2, 4]
        lut_ops = prog.find_ops(op_type="constexpr_lut_to_dense")

        for nbits, op in zip(expected_nbits, lut_ops):
            assert op.lut.val.shape == (2**nbits,)

    @staticmethod
    @pytest.mark.parametrize(
        "constexpr_op",
        CONSTEXPR_OPS,
    )
    def test_constexpr_const_not_compressed(constexpr_op):
        """
        The const op which is fed into constexpr ops cannot be compressed.
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 4, 5))])
        def prog(x):
            constexpr = CONSTEXPR_FUNCS[constexpr_op]((2, 3, 4, 5))
            return mb.add(x=x, y=constexpr)

        compressor = quantization.palettize_weights(
            config=cto.coreml.OptimizationConfig(
                global_config=cto.coreml.OpPalettizerConfig(nbits=2, mode="KMEANS", weight_threshold=0),
            )
        )
        compressor.apply(prog)
        expected_ops = [constexpr_op, "add"]
        assert get_op_types_in_program(prog) == expected_ops

    @staticmethod
    def test_shared_weights():
        """
        If a const is shared with different downstream ops, we do a further conflict detection.
        """

        def _get_program():
            @mb.program(
                input_specs=[mb.TensorSpec(shape=(1, 10, 30))], opset_version=ct.target.iOS16
            )
            def prog(x):
                add_const = np.random.rand(10, 30).astype(np.float32)
                add_const = mb.const(val=add_const, name="add_const")
                x = mb.add(x=x, y=add_const, name="add_1")
                return mb.add(x=x, y=add_const, name="add_2")
            return prog

        # [Case 1] No conflict. Global and op_name level config are the same
        prog = _get_program()

        compressor = quantization.palettize_weights(
            config=cto.coreml.OptimizationConfig(
                global_config=cto.coreml.OpPalettizerConfig(nbits=2, mode="KMEANS", weight_threshold=50),
                op_name_configs={"add_2": cto.coreml.OpPalettizerConfig(nbits=2, mode="KMEANS", weight_threshold=50)}
            )
        )

        compressor.apply(prog)

        expected_ops = [
            "constexpr_lut_to_dense",
            "add",
            "add",
        ]
        assert get_op_types_in_program(prog) == expected_ops

        # [Case 2] No conflict. op_name level configs are the same
        prog = _get_program()

        compressor = quantization.palettize_weights(
            config=cto.coreml.OptimizationConfig(
                global_config=cto.coreml.OpPalettizerConfig(nbits=4, mode="KMEANS", weight_threshold=50),
                op_name_configs={
                    "add_1": cto.coreml.OpPalettizerConfig(nbits=2, mode="KMEANS", weight_threshold=50),
                    "add_2": cto.coreml.OpPalettizerConfig(nbits=2, mode="KMEANS", weight_threshold=50),
                }
            )
        )

        compressor.apply(prog)

        expected_ops = [
            "constexpr_lut_to_dense",
            "add",
            "add",
        ]
        assert get_op_types_in_program(prog) == expected_ops

        # [Case 3] Conflict. Global and op_name level config are different
        prog = _get_program()

        compressor = quantization.palettize_weights(
            config=cto.coreml.OptimizationConfig(
                global_config=cto.coreml.OpPalettizerConfig(nbits=2, mode="KMEANS", weight_threshold=50),
                op_name_configs={"add_2": cto.coreml.OpPalettizerConfig(nbits=4, mode="KMEANS", weight_threshold=50)}
            )
        )

        with pytest.raises(ValueError, match="compression config conflict detected between ops"):
            compressor.apply(prog)

        # [Case 4] Conflict. op_name level configs are different
        prog = _get_program()

        compressor = quantization.palettize_weights(
            config=cto.coreml.OptimizationConfig(
                global_config=cto.coreml.OpPalettizerConfig(nbits=2, mode="KMEANS", weight_threshold=50),
                op_name_configs={
                    "add_1": cto.coreml.OpPalettizerConfig(nbits=4, mode="KMEANS", weight_threshold=50),
                    "add_2": cto.coreml.OpPalettizerConfig(nbits=4, mode="KMEANS", weight_threshold=30),
                },
            )
        )

        with pytest.raises(ValueError, match="compression config conflict detected between ops"):
            compressor.apply(prog)


class TestLinearQuantizer(TestCompressionPasses):
    @pytest.mark.parametrize(
        "mode, dtype, weight_threshold, fake_compression",
        itertools.product(
            ["LINEAR", "LINEAR_SYMMETRIC"],
            [np.int8, np.uint8, types.int8, types.uint8],
            [1000, 7000],
            [True, False],
        ),
    )
    def test_global_config_affine_quantizer(self, mode, dtype, weight_threshold, fake_compression):
        """
        Global config would compress all operations with the same config
        """
        op_config = cto.coreml.OpLinearQuantizerConfig(
            mode=mode, dtype=dtype, weight_threshold=weight_threshold
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config)
        compressor = quantization.linear_quantize_weights(
            config=config, fake_compression=fake_compression
        )
        prog = self._get_test_program()
        compressor.apply(prog)

        if fake_compression:
            expected_ops = ["conv", "reshape", "linear", "reshape", "conv_transpose"]
        elif weight_threshold == 1000:
            expected_ops = [
                "constexpr_affine_dequantize",
                "conv",
                "reshape",
                "constexpr_affine_dequantize",
                "linear",
                "reshape",
                "constexpr_affine_dequantize",
                "conv_transpose",
            ]
        else:
            assert weight_threshold == 7000
            # linear weight size < 7000
            expected_ops = [
                "constexpr_affine_dequantize",
                "conv",
                "reshape",
                "linear",
                "reshape",
                "constexpr_affine_dequantize",
                "conv_transpose",
            ]
        assert get_op_types_in_program(prog) == expected_ops

    @pytest.mark.parametrize(
        "mode, dtype, block_size, weight_threshold, fake_compression",
        itertools.product(
            ["LINEAR", "LINEAR_SYMMETRIC"],
            ["int4", "uint4", "int8", "uint8", np.int8, np.uint8],
            [1],
            [1000, 7000],
            [True, False],
        ),
    )
    def test_global_config_affine_quantizer_blockwise(
        self, mode, dtype, block_size, weight_threshold, fake_compression
    ):
        """
        Global config would compress all operations with the same config for blockwise.
        """
        op_config = cto.coreml.OpLinearQuantizerConfig(
            mode=mode,
            dtype=dtype,
            granularity="per_block",
            block_size=block_size,
            weight_threshold=weight_threshold,
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config)
        compressor = quantization.linear_quantize_weights(
            config=config, fake_compression=fake_compression
        )
        prog = self._get_test_program_3()
        compressor.apply(prog)

        if fake_compression:
            expected_ops = ["conv", "reshape", "linear", "matmul", "reshape", "conv_transpose"]
        elif weight_threshold == 1000:
            expected_ops = [
                "constexpr_blockwise_shift_scale",
                "conv",
                "reshape",
                "constexpr_blockwise_shift_scale",
                "linear",
                "constexpr_blockwise_shift_scale",
                "matmul",
                "reshape",
                "constexpr_blockwise_shift_scale",
                "conv_transpose",
            ]
        else:
            assert weight_threshold == 7000
            # linear and matmul weight size < 7000
            expected_ops = [
                "constexpr_blockwise_shift_scale",
                "conv",
                "reshape",
                "linear",
                "matmul",
                "reshape",
                "constexpr_blockwise_shift_scale",
                "conv_transpose",
            ]
        assert get_op_types_in_program(prog) == expected_ops

    def test_op_type_config_linear_quantizer(self):
        """
        set_op_type allow the user to set different config for each op type.
        Also checking that the config can be overwritten
        """
        conv_config_1 = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR_SYMMETRIC",
            dtype=np.uint8,
            weight_threshold=2000,
        )
        # conv_config_2 overwrite conv_config_1
        conv_config_2 = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR_SYMMETRIC",
            dtype=np.int8,
            weight_threshold=2000,
        )
        # The weight_threshold is super large so linear is not going to be compressed
        linear_config = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR_SYMMETRIC",
            dtype=np.int8,
            weight_threshold=1000000,
        )
        conv_transpose_config = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR",
            dtype=np.uint8,
            weight_threshold=2000,
        )

        config = cto.coreml.OptimizationConfig()
        config.set_op_type("conv", conv_config_1)
        config.set_op_type("conv", conv_config_2)
        config.set_op_type("linear", linear_config)
        config.set_op_type("conv_transpose", conv_transpose_config)

        compressor = quantization.linear_quantize_weights(config=config)

        prog = self._get_test_program()
        compressor.apply(prog)

        expected_ops = [
            "constexpr_affine_dequantize",
            "conv",
            "reshape",
            "linear",
            "reshape",
            "constexpr_affine_dequantize",
            "conv_transpose",
        ]
        assert get_op_types_in_program(prog) == expected_ops

        # Test different dtype are applied
        assert (
            prog.find_ops(op_type="constexpr_affine_dequantize")[0].quantized_data.val.dtype
            == np.int8
        )
        assert (
            prog.find_ops(op_type="constexpr_affine_dequantize")[1].quantized_data.val.dtype
            == np.uint8
        )

    def test_op_type_config_linear_quantizer_blockwise(self):
        """
        set_op_type allow the user to set different config for each op type for blockwise.
        Also checking that the config can be overwritten.
        """
        conv_config_1 = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR_SYMMETRIC",
            dtype="int8",
            granularity="per_block",
            block_size=10,
            weight_threshold=5000,
        )
        # conv_config_2 overwrite conv_config_1
        conv_config_2 = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR_SYMMETRIC",
            dtype="int4",
            granularity="per_block",
            block_size=3,
            weight_threshold=2000,
        )
        # The weight_threshold is super large so linear is not going to be compressed
        linear_config = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR_SYMMETRIC",
            dtype="int4",
            granularity="per_block",
            weight_threshold=1000000,
        )
        conv_transpose_config = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR",
            dtype="int8",
            granularity="per_block",
            block_size=10,
            weight_threshold=2000,
        )

        config = cto.coreml.OptimizationConfig()
        config.set_op_type("conv", conv_config_1)
        config.set_op_type("conv", conv_config_2)
        config.set_op_type("linear", linear_config)
        config.set_op_type("conv_transpose", conv_transpose_config)

        compressor = quantization.linear_quantize_weights(config=config)

        prog = self._get_test_program_3()
        compressor.apply(prog)

        expected_ops = [
            "constexpr_blockwise_shift_scale",
            "conv",
            "reshape",
            "linear",
            "matmul",
            "reshape",
            "constexpr_blockwise_shift_scale",
            "conv_transpose",
        ]
        assert get_op_types_in_program(prog) == expected_ops
        assert prog.find_ops(op_type="constexpr_blockwise_shift_scale")[0].offset is None
        assert prog.find_ops(op_type="constexpr_blockwise_shift_scale")[1].offset is not None

    def test_op_name_config_linear_quantizer(self):
        """
        set_op_name allow the user to set different config for each op specified by name.
        Also checking that the config can be overwritten
        """
        conv_config_1 = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR_SYMMETRIC",
            dtype=np.uint8,
            weight_threshold=2000,
        )
        # conv_config_2 overwrite conv_config_1
        conv_config_2 = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR_SYMMETRIC",
            dtype=np.int8,
            weight_threshold=2000,
        )
        # The weight_threshold is super large so linear is not going to be compressed
        linear_config = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR_SYMMETRIC",
            dtype=np.int8,
            weight_threshold=1000000,
        )
        conv_transpose_config = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR",
            dtype=np.uint8,
            weight_threshold=2000,
        )

        config = cto.coreml.OptimizationConfig()
        config.set_op_name("conv", conv_config_1)
        config.set_op_name("conv", conv_config_2)
        config.set_op_name("linear", linear_config)
        config.set_op_name("conv_transpose", conv_transpose_config)

        compressor = quantization.linear_quantize_weights(config=config)

        prog = self._get_test_program()
        compressor.apply(prog)

        expected_ops = [
            "constexpr_affine_dequantize",
            "conv",
            "reshape",
            "linear",
            "reshape",
            "constexpr_affine_dequantize",
            "conv_transpose",
        ]
        assert get_op_types_in_program(prog) == expected_ops

        # Test different dtype are applied
        assert (
            prog.find_ops(op_type="constexpr_affine_dequantize")[0].quantized_data.val.dtype
            == np.int8
        )
        assert (
            prog.find_ops(op_type="constexpr_affine_dequantize")[1].quantized_data.val.dtype
            == np.uint8
        )

    def test_op_name_config_linear_quantizer_blockwise(self):
        """
        set_op_name allow the user to set different config for each op specified by name.
        Also checking that the config can be overwritten
        """
        conv_config_1 = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR_SYMMETRIC",
            dtype="int8",
            granularity="per_block",
            block_size=4,
            weight_threshold=2000,
        )
        # conv_config_2 overwrite conv_config_1
        conv_config_2 = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR_SYMMETRIC",
            dtype="int8",
            granularity="per_block",
            block_size=2,
            weight_threshold=2000,
        )
        # The weight_threshold is super large so linear is not going to be compressed
        linear_config = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR_SYMMETRIC",
            dtype="int4",
            weight_threshold=1000000,
        )
        conv_transpose_config = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR",
            dtype="int8",
            granularity="per_block",
            block_size=6,
            weight_threshold=2000,
        )

        config = cto.coreml.OptimizationConfig()
        config.set_op_name("conv", conv_config_1)
        config.set_op_name("conv", conv_config_2)
        config.set_op_name("linear", linear_config)
        config.set_op_name("conv_transpose", conv_transpose_config)

        compressor = quantization.linear_quantize_weights(config=config)

        prog = self._get_test_program_3()
        compressor.apply(prog)

        expected_ops = [
            "constexpr_blockwise_shift_scale",
            "conv",
            "reshape",
            "linear",
            "matmul",
            "reshape",
            "constexpr_blockwise_shift_scale",
            "conv_transpose",
        ]
        assert get_op_types_in_program(prog) == expected_ops
        blockwise_ops = prog.find_ops(op_type="constexpr_blockwise_shift_scale")
        assert blockwise_ops[0].offset is None
        assert blockwise_ops[1].offset is not None
        # Conv transpose original weight shape is (30, 4, 21, 10). The output channel axis is 1 and
        # input channel axis is 0, so the scale's first axis dim is 30 / 6 = 5.
        assert blockwise_ops[1].scale.shape == (5, 4, 1, 1)

    def test_auto_pick_channel_axis_quantizer(self):
        """
        Check the right output channel axis is picked for block-wise quantization.
        """
        global_config = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR",
            dtype="int4",
            granularity="per_block",
            block_size=2,
            weight_threshold=2000,
        )
        linear_config = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR_SYMMETRIC",
            dtype="int4",
            granularity="per_block",
            block_size=9,
            weight_threshold=100,
        )
        config = cto.coreml.OptimizationConfig()
        config.set_global(global_config)
        config.set_op_name("linear", linear_config)
        compressor = quantization.linear_quantize_weights(config=config)

        prog = self._get_test_program_3()
        compressor.apply(prog)

        blockwise_ops = prog.find_ops(op_type="constexpr_blockwise_shift_scale")
        # For conv, input channel axis is 1, output channel axis is 0.
        # The original weight shape is [90, 30, 2, 2], the scale's second dim is 30 / 2 = 15.
        assert blockwise_ops[0].scale.shape == (90, 15, 1, 1)
        # For linear, input channel axis is 1, output channel axis is 0.
        # The original weight shape is [70, 81], the scale's second dim is 81 / 9 = 9.
        assert blockwise_ops[1].scale.shape == (70, 9)
        # For matmul (transpose_y=False), input channel axis is -2, output channel axis is -1.
        # The original weight shape is [2, 1, 70, 35], the scale's third dim is 70 / 2 = 35.
        assert blockwise_ops[2].scale.shape == (1, 1, 35, 35)
        # For conv_transpose, input channel axis is 0, output channel axis is 1.
        # The original weight shape is [30, 4, 21, 10], the scale's first dim is 30 / 2 = 15.
        assert blockwise_ops[3].scale.shape == (15, 4, 1, 1)

    def test_invalid_config(self):
        with pytest.raises(
            ValueError,
            match="Invalid dtype int2. Only support int8/uint8/int4/uint4",
        ):
            cto.coreml.OpLinearQuantizerConfig(
                mode="LINEAR_SYMMETRIC",
                dtype="int2",
                block_size=2,
                weight_threshold=2000,
            )

        with pytest.raises(
            ValueError,
            match="Only mode \('LINEAR_SYMMETRIC', 'LINEAR'\) supported for weight affine quantization. Got mode: \"DUMMY\".",
        ):
            cto.coreml.OpLinearQuantizerConfig(
                mode="DUMMY",
                dtype="int4",
                block_size=32,
                weight_threshold=5000,
            )

    def test_not_divisible_block_size(self, caplog):
        global_config = cto.coreml.OpLinearQuantizerConfig(
            mode="LINEAR_SYMMETRIC",
            granularity="per_block",
            dtype="int4",
            block_size=13,
            weight_threshold=100,
        )
        config = cto.coreml.OptimizationConfig()
        config.set_global(global_config)
        compressor = quantization.linear_quantize_weights(config=config)

        prog = self._get_test_program_3()
        compressor.apply(prog)
        warning_msg = "Invalid block_sizes; On 1th axis, the dim size 30 is not divisible by block size 13. Unable to perform structured quantization."
        assert any([re.match(warning_msg, rec.message) for rec in caplog.records])


class TestPruner(TestCompressionPasses):
    @pytest.mark.parametrize(
        "mode, threshold, target_sparsity, weight_threshold, fake_compression",
        itertools.product(
            ["THRESHOLD_BASED", "PERCENTILE_BASED"],
            [1e-3, 1.0],
            [0.2, 0.98],
            [1000, 7000],
            [True, False],
        ),
    )
    def test_global_config_pruner(
        self, mode, threshold, target_sparsity, weight_threshold, fake_compression
    ):
        """
        Global config would compress all operations with the same config
        """
        if mode == "THRESHOLD_BASED":
            op_config = cto.coreml.OpThresholdPrunerConfig(
                threshold=threshold,
                weight_threshold=weight_threshold,
                minimum_sparsity_percentile=0.0,
            )
        else:
            assert mode == "PERCENTILE_BASED"
            op_config = cto.coreml.OpMagnitudePrunerConfig(
                target_sparsity=target_sparsity,
                weight_threshold=weight_threshold,
            )

        config = cto.coreml.OptimizationConfig(global_config=op_config)
        compressor = quantization.prune_weights(config=config, fake_compression=fake_compression)
        prog = self._get_test_program()
        compressor.apply(prog)

        if fake_compression:
            expected_ops = ["conv", "reshape", "linear", "reshape", "conv_transpose"]
        elif weight_threshold == 1000:
            expected_ops = [
                "constexpr_sparse_to_dense",
                "conv",
                "reshape",
                "constexpr_sparse_to_dense",
                "linear",
                "reshape",
                "constexpr_sparse_to_dense",
                "conv_transpose",
            ]
        else:
            assert weight_threshold == 7000
            # linear weight size < 7000
            expected_ops = [
                "constexpr_sparse_to_dense",
                "conv",
                "reshape",
                "linear",
                "reshape",
                "constexpr_sparse_to_dense",
                "conv_transpose",
            ]
        assert get_op_types_in_program(prog) == expected_ops

    def test_op_type_config_pruner(self):
        """
        set_op_type allow the user to set different config for each op type.
        Also checking that the config can be overwritten
        """
        conv_config_1 = cto.coreml.OpMagnitudePrunerConfig(
            target_sparsity=0.5,
            weight_threshold=2000,
        )
        # conv_config_2 overwrite conv_config_1
        conv_config_2 = cto.coreml.OpMagnitudePrunerConfig(
            target_sparsity=0.9,
            weight_threshold=2000,
        )
        linear_config = cto.coreml.OpMagnitudePrunerConfig(
            target_sparsity=0.2,
            weight_threshold=2000,
        )
        # The weight_threshold is super large so conv_transpose is not going to be compressed
        conv_transpose_config = cto.coreml.OpThresholdPrunerConfig(
            threshold=1.0,
            weight_threshold=1000000,
        )

        config = cto.coreml.OptimizationConfig()
        config.set_op_type("conv", conv_config_1)
        config.set_op_type("conv", conv_config_2)
        config.set_op_type("linear", linear_config)
        config.set_op_type("conv_transpose", conv_transpose_config)

        compressor = quantization.prune_weights(config=config)

        prog = self._get_test_program()
        compressor.apply(prog)

        expected_ops = [
            "constexpr_sparse_to_dense",
            "conv",
            "reshape",
            "constexpr_sparse_to_dense",
            "linear",
            "reshape",
            "conv_transpose",
        ]
        assert get_op_types_in_program(prog) == expected_ops

        # Test different sparcsity percentile are applied
        assert (
            prog.find_ops(op_type="constexpr_sparse_to_dense")[0].nonzero_data.val.size == 1080
        )  # 1080 * 0.1
        assert (
            prog.find_ops(op_type="constexpr_sparse_to_dense")[1].nonzero_data.val.size == 4536
        )  # 5670 * 0.8

    def test_op_name_config_pruner(self):
        """
        set_op_name allow the user to set different config for each op specified by name.
        Also checking that the config can be overwritten
        """
        conv_config_1 = cto.coreml.OpMagnitudePrunerConfig(
            target_sparsity=0.5,
            weight_threshold=2000,
        )
        # conv_config_2 overwrite conv_config_1
        conv_config_2 = cto.coreml.OpMagnitudePrunerConfig(
            target_sparsity=0.9,
            weight_threshold=2000,
        )
        linear_config = cto.coreml.OpMagnitudePrunerConfig(
            target_sparsity=0.2,
            weight_threshold=2000,
        )
        # The weight_threshold is super large so conv_transpose is not going to be compressed
        conv_transpose_config = cto.coreml.OpThresholdPrunerConfig(
            threshold=1.0,
            weight_threshold=1000000,
        )

        config = cto.coreml.OptimizationConfig()
        config.set_op_name("conv", conv_config_1)
        config.set_op_name("conv", conv_config_2)
        config.set_op_name("linear", linear_config)
        config.set_op_name("conv_transpose", conv_transpose_config)

        compressor = quantization.prune_weights(config=config)

        prog = self._get_test_program()
        compressor.apply(prog)

        expected_ops = [
            "constexpr_sparse_to_dense",
            "conv",
            "reshape",
            "constexpr_sparse_to_dense",
            "linear",
            "reshape",
            "conv_transpose",
        ]
        assert get_op_types_in_program(prog) == expected_ops

        # Test different sparcsity percentile are applied
        assert (
            prog.find_ops(op_type="constexpr_sparse_to_dense")[0].nonzero_data.val.size == 1080
        )  # 1080 * 0.1
        assert (
            prog.find_ops(op_type="constexpr_sparse_to_dense")[1].nonzero_data.val.size == 4536
        )  # 5670 * 0.8

    @pytest.mark.parametrize(
        "target_sparsity, minimum_sparsity_percentile",
        itertools.product(
            [0.1, 0.5, 0.9],
            [0.0, 0.3, 0.7],
        ),
    )
    def test_pruner_minimum_sparsity_percentile(self, target_sparsity, minimum_sparsity_percentile):
        def _get_sparse_weight(shape, target_sparsity):
            size = np.prod(shape)
            weight = 3 * np.ones(size)
            num_of_zeros = int(size * target_sparsity)
            weight[:num_of_zeros] = 0
            return np.reshape(weight, shape).astype(np.float32)

        def _get_simple_program():
            @mb.program(
                input_specs=[mb.TensorSpec(shape=(1, 30, 10, 10))], opset_version=ct.target.iOS16
            )
            def prog(x):
                conv_weight = _get_sparse_weight((90, 30, 3, 3), target_sparsity)
                x = mb.conv(x=x, weight=conv_weight, name="conv1")
                return x

            return prog

        op_config = cto.coreml.OpThresholdPrunerConfig(
            threshold=1e-3,
            minimum_sparsity_percentile=minimum_sparsity_percentile,
            weight_threshold=200,
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config)
        compressor = quantization.prune_weights(config=config)
        prog = _get_simple_program()
        compressor.apply(prog)

        if minimum_sparsity_percentile < target_sparsity:
            expected_ops = ["constexpr_sparse_to_dense", "conv"]
        else:
            expected_ops = ["conv"]
        assert get_op_types_in_program(prog) == expected_ops

    def test_structural_pruning(self):
        def _get_test_prog():
            @mb.program(
                input_specs=[mb.TensorSpec(shape=(1, 30, 10, 10))], opset_version=ct.target.iOS16
            )
            def prog(x):
                conv_weight_1 = mb.const(
                    val=np.random.rand(90, 30, 2, 2).astype(np.float32), name="w_1"
                )
                conv_bias_1 = mb.const(
                    val=np.random.rand(
                        90,
                    ).astype(np.float32),
                    name="b_1",
                )
                conv_weight_2 = mb.const(
                    val=np.random.rand(10, 90, 2, 2).astype(np.float32), name="w_2"
                )
                linear_weight = mb.const(val=np.random.rand(128, 64).astype(np.float32), name="l_w")
                linear_bias = mb.const(
                    val=np.random.rand(
                        128,
                    ).astype(np.float32),
                    name="l_b",
                )
                add_const = mb.const(
                    val=np.random.rand(10, 128).astype(np.float32), name="add_const"
                )

                x = mb.conv(x=x, weight=conv_weight_1, bias=conv_bias_1, name="conv_1")
                x = mb.conv(x=x, weight=conv_weight_2, name="conv_2")
                x = mb.reshape(x=x, shape=(10, 64))
                x = mb.linear(x=x, weight=linear_weight, bias=linear_bias, name="linear_1")
                x = mb.add(x=x, y=add_const, name="add_1")
                return x

            return prog

        # (1) Global structural pruning config will only applied to conv / linear weight
        prog = _get_test_prog()
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpMagnitudePrunerConfig(
                n_m_ratio=(2, 3),
                weight_threshold=0,
            )
        )
        compressor = quantization.prune_weights(config=config)
        compressor.apply(prog)
        expected_ops = [
            "constexpr_sparse_to_dense",
            "constexpr_sparse_to_dense",
            "constexpr_sparse_to_dense",
            "conv",
            "conv",
            "reshape",
            "linear",
            "add",
        ]
        assert get_op_types_in_program(prog) == expected_ops
        conv_ops = prog.find_ops(op_type="conv")
        assert conv_ops[0].weight.op.op_type == "constexpr_sparse_to_dense"
        assert conv_ops[1].weight.op.op_type == "constexpr_sparse_to_dense"
        assert prog.find_ops(op_type="linear")[0].weight.op.op_type == "constexpr_sparse_to_dense"

        # (2) Even by setting the ops with structural pruning, make sure only weight is sparsified, not bias
        prog = _get_test_prog()
        config = cto.coreml.OptimizationConfig(
            op_type_configs={
                "conv": cto.coreml.OpMagnitudePrunerConfig(
                    n_m_ratio=(2, 3),
                    weight_threshold=0,
                )
            },
            op_name_configs={
                "linear_1": cto.coreml.OpMagnitudePrunerConfig(
                    n_m_ratio=(1, 4),
                    weight_threshold=0,
                )
            },
        )
        compressor = quantization.prune_weights(config=config)
        compressor.apply(prog)
        expected_ops = [
            "constexpr_sparse_to_dense",
            "constexpr_sparse_to_dense",
            "constexpr_sparse_to_dense",
            "conv",
            "conv",
            "reshape",
            "linear",
            "add",
        ]
        assert get_op_types_in_program(prog) == expected_ops
        conv_ops = prog.find_ops(op_type="conv")
        assert conv_ops[0].weight.op.op_type == "constexpr_sparse_to_dense"
        assert conv_ops[1].weight.op.op_type == "constexpr_sparse_to_dense"
        assert prog.find_ops(op_type="linear")[0].weight.op.op_type == "constexpr_sparse_to_dense"

        # (3) Early error out when setting a non applicable op to structural pruning with set_op_type
        with pytest.raises(
            ValueError, match="block sparsity or n:m pruning does not support op type add"
        ):
            config = cto.coreml.OptimizationConfig(
                op_type_configs={
                    "add": cto.coreml.OpMagnitudePrunerConfig(
                        n_m_ratio=(2, 3),
                        weight_threshold=0,
                    )
                },
            )

        with pytest.raises(
            ValueError, match="block sparsity or n:m pruning does not support op type add"
        ):
            config = cto.coreml.OptimizationConfig()
            config.set_op_type(
                "add",
                cto.coreml.OpMagnitudePrunerConfig(
                    n_m_ratio=(2, 3),
                    weight_threshold=0,
                ),
            )

        # (4) By using set_op_name, we can still force a const op to use structural pruning
        prog = _get_test_prog()
        config = cto.coreml.OptimizationConfig(
            op_name_configs={
                "add_const": cto.coreml.OpMagnitudePrunerConfig(
                    n_m_ratio=(1, 4),
                    weight_threshold=0,
                )
            }
        )
        compressor = quantization.prune_weights(config=config)
        compressor.apply(prog)
        expected_ops = [
            "constexpr_sparse_to_dense",
            "conv",
            "conv",
            "reshape",
            "linear",
            "add",
        ]
        assert get_op_types_in_program(prog) == expected_ops
        assert prog.find_ops(op_type="add")[0].y.op.op_type == "constexpr_sparse_to_dense"


class TestPalettizer(TestCompressionPasses):
    @pytest.mark.parametrize(
        "nbits, mode, weight_threshold, fake_compression",
        itertools.product(
            [2, 6],
            ["KMEANS", "UNIFORM"],
            [1000, 7000],
            [True, False],
        ),
    )
    def test_global_config_palettizer(self, nbits, mode, weight_threshold, fake_compression):
        """
        Global config would compress all operations with the same config
        """
        op_config = cto.coreml.OpPalettizerConfig(
            nbits=nbits, mode=mode, weight_threshold=weight_threshold
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config)
        compressor = quantization.palettize_weights(
            config=config, fake_compression=fake_compression
        )
        prog = self._get_test_program()
        compressor.apply(prog)

        if fake_compression:
            expected_ops = ["conv", "reshape", "linear", "reshape", "conv_transpose"]
        elif weight_threshold == 1000:
            expected_ops = [
                "constexpr_lut_to_dense",
                "conv",
                "reshape",
                "constexpr_lut_to_dense",
                "linear",
                "reshape",
                "constexpr_lut_to_dense",
                "conv_transpose",
            ]
        else:
            assert weight_threshold == 7000
            # linear weight size < 7000
            expected_ops = [
                "constexpr_lut_to_dense",
                "conv",
                "reshape",
                "linear",
                "reshape",
                "constexpr_lut_to_dense",
                "conv_transpose",
            ]
        assert get_op_types_in_program(prog) == expected_ops

    def test_op_type_config_palettizer(self):
        """
        set_op_type allow the user to set different config for each op type.
        Also checking that the config can be overwritten
        """
        conv_config_1 = cto.coreml.OpPalettizerConfig(
            nbits=8,
            mode="KMEANS",
            weight_threshold=2000,
        )
        # conv_config_2 overwrite conv_config_1
        conv_config_2 = cto.coreml.OpPalettizerConfig(
            nbits=2,
            mode="KMEANS",
            weight_threshold=2000,
        )
        linear_config = cto.coreml.OpPalettizerConfig(
            nbits=4,
            mode="UNIFORM",
            weight_threshold=2000,
        )
        # The weight_threshold is super large so conv_transpose is not going to be compressed
        conv_transpose_config = cto.coreml.OpPalettizerConfig(
            nbits=4,
            mode="UNIFORM",
            weight_threshold=1000000,
        )

        config = cto.coreml.OptimizationConfig()
        config.set_op_type("conv", conv_config_1)
        config.set_op_type("conv", conv_config_2)
        config.set_op_type("linear", linear_config)
        config.set_op_type("conv_transpose", conv_transpose_config)

        compressor = quantization.palettize_weights(config=config)

        prog = self._get_test_program()
        compressor.apply(prog)

        expected_ops = [
            "constexpr_lut_to_dense",
            "conv",
            "reshape",
            "constexpr_lut_to_dense",
            "linear",
            "reshape",
            "conv_transpose",
        ]
        assert get_op_types_in_program(prog) == expected_ops

        # Test different nbits are applied
        assert prog.find_ops(op_type="constexpr_lut_to_dense")[0].lut.val.shape == (4,)
        assert prog.find_ops(op_type="constexpr_lut_to_dense")[1].lut.val.shape == (16,)

    def test_op_name_config_palettizer(self):
        """
        set_op_name allow the user to set different config for each op specified by name.
        Also checking that the config can be overwritten
        """
        conv_config_1 = cto.coreml.OpPalettizerConfig(
            nbits=8,
            mode="KMEANS",
            weight_threshold=2000,
        )
        # conv_config_2 overwrite conv_config_1
        conv_config_2 = cto.coreml.OpPalettizerConfig(
            nbits=2,
            mode="KMEANS",
            weight_threshold=2000,
        )
        linear_config = cto.coreml.OpPalettizerConfig(
            nbits=4,
            mode="UNIFORM",
            weight_threshold=2000,
        )
        # The weight_threshold is super large so conv_transpose is not going to be compressed
        conv_transpose_config = cto.coreml.OpPalettizerConfig(
            nbits=4,
            mode="UNIFORM",
            weight_threshold=1000000,
        )

        config = cto.coreml.OptimizationConfig()
        config.set_op_name("conv", conv_config_1)
        config.set_op_name("conv", conv_config_2)
        config.set_op_name("linear", linear_config)
        config.set_op_name("conv_transpose", conv_transpose_config)

        compressor = quantization.palettize_weights(config=config)

        prog = self._get_test_program()
        compressor.apply(prog)

        expected_ops = [
            "constexpr_lut_to_dense",
            "conv",
            "reshape",
            "constexpr_lut_to_dense",
            "linear",
            "reshape",
            "conv_transpose",
        ]
        assert get_op_types_in_program(prog) == expected_ops

        # Test different nbits are applied
        assert prog.find_ops(op_type="constexpr_lut_to_dense")[0].lut.val.shape == (4,)
        assert prog.find_ops(op_type="constexpr_lut_to_dense")[1].lut.val.shape == (16,)

    def test_op_name_config_palettizer_blockwise(self):
        """
        set_op_name allow the user to set different config for each op specified by name.
        Also checking that the config can be overwritten.
        """
        conv_config_1 = cto.coreml.OpPalettizerConfig(
            mode="uniform",
            nbits=4,
            granularity="per_tensor",
            weight_threshold=500000,
        )
        # The conv_config_2 overwrites conv_config_1.
        conv_config_2 = cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=8,
            granularity="per_grouped_channel",
            group_size=1,
            channel_axis=1,
            weight_threshold=2000,
        )
        # The weight_threshold is super large so linear is not going to be compressed.
        linear_config = cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=4,
            weight_threshold=1000000,
        )
        conv_transpose_config = cto.coreml.OpPalettizerConfig(
            mode="uniform",
            nbits=4,
            granularity="per_grouped_channel",
            group_size=1,
            weight_threshold=2000,
        )

        config = cto.coreml.OptimizationConfig()
        config.set_op_name("conv", conv_config_1)
        config.set_op_name("conv", conv_config_2)
        config.set_op_name("linear", linear_config)
        config.set_op_name("conv_transpose", conv_transpose_config)

        prog = self._get_test_program_3()
        compressor = quantization.palettize_weights(config=config)
        compressor.apply(prog)

        expected_ops = [
            "constexpr_lut_to_dense",
            "conv",
            "reshape",
            "linear",
            "matmul",
            "reshape",
            "constexpr_lut_to_dense",
            "conv_transpose",
        ]
        assert get_op_types_in_program(prog) == expected_ops
        assert prog.find_ops(op_type="constexpr_lut_to_dense")[0].vector_axis is None
        # Makes sure the channel_axis in conv_config_2 is effective.
        conv_lut = prog.find_ops(op_type="constexpr_lut_to_dense")[0].lut
        assert conv_lut.shape[0] == 1
        assert conv_lut.shape[1] == 30

    def test_invalid_granularity(self):
        with pytest.raises(
            ValueError,
            match='"granularity" must be one of .*, but got CompressionGranularity.PER_CHANNEL',
        ):
            cto.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=4,
                granularity="per_channel",
                weight_threshold=2000,
            )

        with pytest.raises(TypeError, match="got an unexpected keyword argument 'block_size'"):
            cto.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=4,
                granularity="per_tensor",
                block_size=2,
                weight_threshold=2000,
            )

    def test_auto_pick_channel_axis_palettizer(self):
        """
        Check the right output channel axis is picked for granularity='per_grouped_channel'.
        """
        global_config = cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=4,
            granularity="per_grouped_channel",
            group_size=1,
            weight_threshold=2000,
        )
        config = cto.coreml.OptimizationConfig()
        config.set_global(global_config)
        compressor = quantization.palettize_weights(config=config)

        prog = self._get_test_program_3()
        compressor.apply(prog)

        # For conv, the output channel-axis is 0.
        conv_lut = prog.find_ops(op_type="constexpr_lut_to_dense")[0].lut
        assert conv_lut.shape[0] == 90
        assert conv_lut.shape[1] == 1
        # For linear, the output channel-axis is 0.
        linear_lut = prog.find_ops(op_type="constexpr_lut_to_dense")[1].lut
        assert linear_lut.shape[0] == 70
        assert linear_lut.shape[1] == 1
        # For matmul with transpose_y=False, the output channel-axis is -1.
        matmul_lut = prog.find_ops(op_type="constexpr_lut_to_dense")[2].lut
        assert matmul_lut.shape == (1, 1, 1, 35, 16, 1)
        # For conv_transpose, the output channel-axis is -2.
        conv_transpose_lut = prog.find_ops(op_type="constexpr_lut_to_dense")[3].lut
        assert conv_transpose_lut.shape[0] == 1
        assert conv_transpose_lut.shape[1] == 4

    def test_group_channel_wise(self):
        global_config = cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=3,
            granularity="per_grouped_channel",
            group_size=2,
            weight_threshold=2000,
        )
        config = cto.coreml.OptimizationConfig()
        config.set_global(global_config)
        compressor = quantization.palettize_weights(config=config)

        prog = self._get_test_program_3()
        compressor.apply(prog)
        lut_ops = prog.find_ops(op_type="constexpr_lut_to_dense")
        # The conv weight dense shape is (90, 30, 2, 2).  Auto-picked axis=0.
        assert lut_ops[0].lut.shape == (45, 1, 1, 1, 8, 1)
        # The linear weight dense shape is (70, 81). Auto-picked axis=0.
        assert lut_ops[1].lut.shape == (35, 1, 8, 1)
        # The matmul y dense shape is (2, 1, 70, 35). Auto-picked axis=-1.
        # However, the 35 is not divisible by 2, so it will get skipped.
        assert prog.find_ops(op_type="matmul")[0].y.op.op_type == "const"
        # The conv_transpose weight dense shape is (30, 4, 21, 10).  Auto-picked axis=-2.
        assert lut_ops[2].lut.shape == (1, 2, 1, 1, 8, 1)

    def test_tensor_wise(self):
        """Test granularity='per_block' with block_size=0 equivalent to granularity='per_tensor'."""
        global_config_1 = cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=3,
            granularity="per_tensor",
            weight_threshold=2000,
        )
        global_config_2 = cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=3,
            granularity="per_grouped_channel",
            group_size=0,
            weight_threshold=2000,
        )

        for global_config in (global_config_1, global_config_2):
            config = cto.coreml.OptimizationConfig(global_config=global_config)
            compressor = quantization.palettize_weights(config=config)

            prog = self._get_test_program_3()
            compressor.apply(prog)
            lut_ops = prog.find_ops(op_type="constexpr_lut_to_dense")
            # The conv weight dense shape is (90, 30, 2, 2).
            assert lut_ops[0].lut.shape == (1, 1, 1, 1, 8, 1)
            # The linear weight dense shape is (70, 81).
            assert lut_ops[1].lut.shape == (1, 1, 8, 1)
            # The matmul y dense shape is (2, 1, 70, 35).
            assert lut_ops[2].lut.shape == (1, 1, 1, 1, 8, 1)
            # The conv_transpose weight dense shape is (30, 4, 21, 10).
            assert lut_ops[3].lut.shape == (1, 1, 1, 1, 8, 1)

    def test_not_divisible_channel_group_size(self, caplog):
        global_config = cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=4,
            granularity="per_grouped_channel",
            group_size=3,
            weight_threshold=2000,
        )
        config = cto.coreml.OptimizationConfig()
        config.set_global(global_config)
        compressor = quantization.palettize_weights(config=config)

        prog = self._get_test_program_3()
        compressor.apply(prog)

        # The axis-0 in linear (70), axis-3 in matmul (35), and axis-1 in conv_transpose (4) are not divisible by 3.
        for axis in (0, 3, 1):
            warning_msg = (
                f"Can't perform palettization: The number of channels at {axis}th axis .* is not "
                "divisible by channel_group_size"
            )
            assert any([re.match(warning_msg, rec.message) for rec in caplog.records])
        # Only the conv get compressed.
        lut_ops = prog.find_ops(op_type="constexpr_lut_to_dense")
        assert len(lut_ops) == 1
        assert lut_ops[0].outputs[0].child_ops[0].op_type == "conv"

    def test_ios16_program_not_support_channel_wise_lut(self):
        global_config = cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=4,
            granularity="per_grouped_channel",
            group_size=3,
            weight_threshold=2000,
        )
        config = cto.coreml.OptimizationConfig()
        config.set_global(global_config)
        compressor = quantization.palettize_weights(config=config)

        prog = self._get_test_program()
        with pytest.raises(
            AssertionError,
            match=re.escape(
                "The pre-iOS18 palettization only supports per-tensor lut, but got more than one lut "
                "on 0th axis. LUT shape: (30, 1, 1, 1, 16, 1)\nPlease set the minimum_deployment_target to iOS18"
            ),
        ):
            compressor.apply(prog)


class TestCompressionOperations(TestCompressionPasses):
    """
    This test is checking compression for some common operations.
    """

    COMPRESSORS = [
        quantization.palettize_weights(
            config=cto.coreml.OptimizationConfig(
                global_config=cto.coreml.OpPalettizerConfig(
                    nbits=2, mode="KMEANS", weight_threshold=50
                )
            )
        ),
        quantization.linear_quantize_weights(
            config=cto.coreml.OptimizationConfig(
                global_config=cto.coreml.OpLinearQuantizerConfig(
                    mode="LINEAR_SYMMETRIC", dtype=np.int8, weight_threshold=50
                )
            )
        ),
        quantization.prune_weights(
            config=cto.coreml.OptimizationConfig(
                global_config=cto.coreml.OpMagnitudePrunerConfig(
                    target_sparsity=0.9, weight_threshold=50
                )
            )
        ),
    ]

    COMPRESSOR_TO_OP_TYPE = {
        "palettize_weights": "constexpr_lut_to_dense",
        "linear_quantize_weights": "constexpr_affine_dequantize",
        "prune_weights": "constexpr_sparse_to_dense",
    }

    @staticmethod
    @pytest.mark.parametrize(
        "compressor",
        COMPRESSORS,
    )
    def test_conv_compress(compressor):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 30, 10, 10))], opset_version=ct.target.iOS16
        )
        def prog(x):
            conv_weight = np.random.rand(90, 30, 2, 2).astype(np.float32)
            return mb.conv(x=x, weight=conv_weight)

        compressor.apply(prog)
        op_type = TestCompressionOperations.COMPRESSOR_TO_OP_TYPE[compressor.__class__.__name__]
        assert get_op_types_in_program(prog) == [op_type, "conv"]

    @staticmethod
    @pytest.mark.parametrize(
        "compressor",
        COMPRESSORS,
    )
    def test_conv_transpose_compress(compressor):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 30, 10, 10))], opset_version=ct.target.iOS16
        )
        def prog(x):
            conv_weight = np.random.rand(90, 30, 2, 2).astype(np.float32)
            return mb.conv_transpose(x=x, weight=conv_weight)

        compressor.apply(prog)
        op_type = TestCompressionOperations.COMPRESSOR_TO_OP_TYPE[compressor.__class__.__name__]
        assert get_op_types_in_program(prog) == [op_type, "conv_transpose"]

    @staticmethod
    @pytest.mark.parametrize(
        "compressor",
        COMPRESSORS,
    )
    def test_liear_compress(compressor):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 30, 10))], opset_version=ct.target.iOS16)
        def prog(x):
            linear_weight = np.random.rand(40, 10).astype(np.float32)
            return mb.linear(x=x, weight=linear_weight)

        compressor.apply(prog)
        op_type = TestCompressionOperations.COMPRESSOR_TO_OP_TYPE[compressor.__class__.__name__]
        assert get_op_types_in_program(prog) == [op_type, "linear"]

    @staticmethod
    @pytest.mark.parametrize(
        "compressor",
        COMPRESSORS,
    )
    def test_matmul_compress(compressor):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 30, 10))], opset_version=ct.target.iOS16)
        def prog(x):
            weight1 = np.random.rand(10, 40).astype(np.float32)
            weight2 = np.random.rand(20, 30).astype(np.float32)

            x = mb.matmul(x=x, y=weight1)
            return mb.matmul(x=weight2, y=x)

        compressor.apply(prog)
        op_type = TestCompressionOperations.COMPRESSOR_TO_OP_TYPE[compressor.__class__.__name__]
        assert get_op_types_in_program(prog) == [op_type, "matmul", op_type, "matmul"]

    @staticmethod
    @pytest.mark.parametrize(
        "compressor",
        COMPRESSORS,
    )
    def test_gru_compress(compressor):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 10, 30)), mb.TensorSpec(shape=(10, 40))],
            opset_version=ct.target.iOS16,
        )
        def prog(x, initial_h):
            weight_ih = np.random.rand(120, 30).astype(np.float32)
            weight_hh = np.random.rand(120, 40).astype(np.float32)
            return mb.gru(x=x, initial_h=initial_h, weight_ih=weight_ih, weight_hh=weight_hh)

        compressor.apply(prog)
        op_type = TestCompressionOperations.COMPRESSOR_TO_OP_TYPE[compressor.__class__.__name__]
        assert get_op_types_in_program(prog) == [op_type, op_type, "gru"]

    @staticmethod
    @pytest.mark.parametrize(
        "compressor",
        COMPRESSORS,
    )
    def test_lstm_compress(compressor):
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 10, 30)),
                mb.TensorSpec(shape=(10, 40)),
                mb.TensorSpec(shape=(10, 40)),
            ],
            opset_version=ct.target.iOS16,
        )
        def prog(x, initial_h, initial_c):
            weight_ih = np.random.rand(160, 30).astype(np.float32)
            weight_hh = np.random.rand(160, 40).astype(np.float32)
            return mb.lstm(
                x=x,
                initial_h=initial_h,
                initial_c=initial_c,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
            )

        compressor.apply(prog)
        op_type = TestCompressionOperations.COMPRESSOR_TO_OP_TYPE[compressor.__class__.__name__]
        assert get_op_types_in_program(prog) == [op_type, op_type, "lstm"]

    @staticmethod
    @pytest.mark.parametrize(
        "compressor",
        COMPRESSORS,
    )
    def test_rnn_compress(compressor):
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 10, 30)),
                mb.TensorSpec(shape=(10, 40)),
            ],
            opset_version=ct.target.iOS16,
        )
        def prog(x, initial_h):
            weight_ih = np.random.rand(40, 30).astype(np.float32)
            weight_hh = np.random.rand(40, 40).astype(np.float32)
            return mb.rnn(x=x, initial_h=initial_h, weight_ih=weight_ih, weight_hh=weight_hh)

        compressor.apply(prog)
        op_type = TestCompressionOperations.COMPRESSOR_TO_OP_TYPE[compressor.__class__.__name__]
        assert get_op_types_in_program(prog) == [op_type, op_type, "rnn"]

    @staticmethod
    @pytest.mark.parametrize(
        "compressor",
        COMPRESSORS,
    )
    def test_add_compress(compressor):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 10, 30))], opset_version=ct.target.iOS16)
        def prog(x):
            add_const = np.random.rand(10, 30).astype(np.float32)
            return mb.add(x=x, y=add_const)

        compressor.apply(prog)
        op_type = TestCompressionOperations.COMPRESSOR_TO_OP_TYPE[compressor.__class__.__name__]
        assert get_op_types_in_program(prog) == [op_type, "add"]

    @staticmethod
    def test_add_compress_set_op_type():
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 10, 30))], opset_version=ct.target.iOS16)
        def prog(x):
            add_const = np.random.rand(10, 30).astype(np.float32)
            return mb.add(x=x, y=add_const)

        compressor = quantization.palettize_weights(
            config=cto.coreml.OptimizationConfig(
                global_config=cto.coreml.OpPalettizerConfig(
                    nbits=2, mode="KMEANS", weight_threshold=50
                ),
                op_type_configs={
                    "add": cto.coreml.OpPalettizerConfig(
                        nbits=4, mode="KMEANS", weight_threshold=50
                    )
                },
            )
        )
        compressor.apply(prog)
        assert get_op_types_in_program(prog) == ["constexpr_lut_to_dense", "add"]
        # also check the compression config comes from set_op_type
        assert prog.find_ops(op_type="constexpr_lut_to_dense")[0].lut.val.shape == (16,)


class TestInvalidConfig:
    """
    This test is checking error handling for invalid configuration.
    """

    @staticmethod
    def test_invalid_config_type():
        err_msg = "config must be of type OptimizationConfig"
        with pytest.raises(ValueError, match=err_msg):
            compressor = quantization.palettize_weights(
                config=1,
            )

        with pytest.raises(ValueError, match=err_msg):
            compressor = quantization.linear_quantize_weights(
                config="12",
            )

        with pytest.raises(ValueError, match=err_msg):
            compressor = quantization.prune_weights(
                config=[12, 3],
            )

        msg = "palettize_weights only accept OpPalettizerConfig type config"
        with pytest.raises(ValueError, match=msg):
            compressor = quantization.palettize_weights(
                config=cto.coreml.OptimizationConfig(
                    global_config=cto.coreml.OpLinearQuantizerConfig(),
                )
            )

        with pytest.raises(ValueError, match=msg):
            compressor = quantization.palettize_weights(
                config=cto.coreml.OptimizationConfig(
                    op_type_configs={"op": cto.coreml.OpLinearQuantizerConfig()},
                )
            )

        with pytest.raises(ValueError, match=msg):
            compressor = quantization.palettize_weights(
                config=cto.coreml.OptimizationConfig(
                    op_name_configs={"name": cto.coreml.OpLinearQuantizerConfig()},
                )
            )

        msg = "linear_quantize_weights only accept OpLinearQuantizerConfig type config"
        with pytest.raises(ValueError, match=msg):
            compressor = quantization.linear_quantize_weights(
                config=cto.coreml.OptimizationConfig(
                    global_config=cto.coreml.OpPalettizerConfig(nbits=2),
                )
            )

        with pytest.raises(ValueError, match=msg):
            compressor = quantization.linear_quantize_weights(
                config=cto.coreml.OptimizationConfig(
                    op_type_configs={"op": cto.coreml.OpPalettizerConfig(nbits=2)},
                )
            )

        with pytest.raises(ValueError, match=msg):
            compressor = quantization.linear_quantize_weights(
                config=cto.coreml.OptimizationConfig(
                    op_name_configs={"op": cto.coreml.OpPalettizerConfig(nbits=2)},
                )
            )

        msg = "prune_weights only accept (OpMagnitudePrunerConfig, OpThresholdPrunerConfig) type config"
        with pytest.raises(ValueError, match=msg):
            compressor = quantization.prune_weights(
                config=cto.coreml.OptimizationConfig(
                    global_config=cto.coreml.OpPalettizerConfig(nbits=2),
                )
            )

        with pytest.raises(ValueError, match=msg):
            compressor = quantization.prune_weights(
                config=cto.coreml.OptimizationConfig(
                    op_type_configs={"op": cto.coreml.OpPalettizerConfig(nbits=2)},
                )
            )

        with pytest.raises(ValueError, match=msg):
            compressor = quantization.prune_weights(
                config=cto.coreml.OptimizationConfig(
                    op_name_configs={"name": cto.coreml.OpPalettizerConfig(nbits=2)},
                )
            )

        msg = "config must be type of OpCompressorConfig."
        with pytest.raises(ValueError, match=msg):
            cto.coreml.OptimizationConfig(
                global_config="str",
            )

        with pytest.raises(ValueError, match=msg):
            cto.coreml.OptimizationConfig(
                op_type_configs={"op": 123},
            )

        with pytest.raises(ValueError, match=msg):
            cto.coreml.OptimizationConfig(
                op_name_configs={"name": []},
            )

        msg = 'Invalid value of "minimum_sparsity_percentile":'
        with pytest.raises(ValueError, match=msg):
            config = cto.coreml.OpThresholdPrunerConfig(
                threshold=0.8,
                minimum_sparsity_percentile=1.2,
            )

        with pytest.raises(ValueError, match=msg):
            config = cto.coreml.OpThresholdPrunerConfig(
                threshold=0.8,
                minimum_sparsity_percentile=-9.0,
            )

        msg = '"weight_threshold" must be a non-negative integer.'
        with pytest.raises(ValueError, match=msg):
            config = cto.coreml.OpThresholdPrunerConfig(
                threshold=0.8,
                weight_threshold=-9,
            )

        with pytest.raises(ValueError, match=msg):
            config = cto.coreml.OpMagnitudePrunerConfig(
                target_sparsity=1.0,
                weight_threshold=-8,
            )

        with pytest.raises(ValueError, match=msg):
            config = cto.coreml.OpLinearQuantizerConfig(
                weight_threshold=-9,
            )

        with pytest.raises(ValueError, match=msg):
            config = cto.coreml.OpPalettizerConfig(
                nbits=2,
                weight_threshold=-10,
            )

        msg = 'Either "target_sparsity" or "n_m_ratio" need to be set. They cannot be set at the same time.'
        with pytest.raises(ValueError, match=msg):
            config = cto.coreml.OpMagnitudePrunerConfig()

        with pytest.raises(ValueError, match=msg):
            config = cto.coreml.OpMagnitudePrunerConfig(
                target_sparsity=0.0,
                n_m_ratio=(2, 10),
            )

        msg = 'Invalid value of "target_sparsity":'
        with pytest.raises(ValueError, match=msg):
            config = cto.coreml.OpMagnitudePrunerConfig(
                target_sparsity=-0.9,
            )

        with pytest.raises(ValueError, match=msg):
            config = cto.coreml.OpMagnitudePrunerConfig(
                target_sparsity=1.1,
            )

        with pytest.raises(
            ValueError, match='"block_size" and "n_m_ratio" cannot be set at the same time.'
        ):
            config = cto.coreml.OpMagnitudePrunerConfig(
                n_m_ratio=(2, 2),
                block_size=9,
            )

        msg = '"block_size" must be an integer \> 1'
        with pytest.raises(ValueError, match=msg):
            config = cto.coreml.OpMagnitudePrunerConfig(
                target_sparsity=0.9,
                block_size=1,
            )

        with pytest.raises(ValueError, match=msg):
            config = cto.coreml.OpMagnitudePrunerConfig(
                target_sparsity=0.9,
                block_size=-9,
            )

        msg = '"n_m_ratio" must be a tuple of two integers \(n, m\). n \<\= m. Got'
        with pytest.raises(ValueError, match=msg):
            config = cto.coreml.OpMagnitudePrunerConfig(
                n_m_ratio=(2, 2, 2),
            )

        with pytest.raises(ValueError, match=msg):
            config = cto.coreml.OpMagnitudePrunerConfig(
                n_m_ratio=(6, 1),
            )

        msg = '"dim" must be 1 or 0'
        with pytest.raises(ValueError, match=msg):
            config = cto.coreml.OpMagnitudePrunerConfig(
                n_m_ratio=(1, 1),
                dim=-1,
            )

        with pytest.raises(ValueError, match=msg):
            config = cto.coreml.OpMagnitudePrunerConfig(
                target_sparsity=1.0,
                block_size=2,
                dim=2,
            )

        with pytest.raises(
            ValueError, match='"dim" can only be set along with "block_size" or "n_m_ratio".'
        ):
            config = cto.coreml.OpMagnitudePrunerConfig(
                target_sparsity=1.0,
                dim=1,
            )

    @staticmethod
    def test_set_op_type_error_out_for_const():
        """
        We cannot use set_op_type for const op
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 10, 30))], opset_version=ct.target.iOS16)
        def prog(x):
            add_const = np.random.rand(10, 30).astype(np.float32)
            return mb.add(x=x, y=add_const, name="add1")

        compressor = quantization.palettize_weights(
            config=cto.coreml.OptimizationConfig(
                global_config=cto.coreml.OpPalettizerConfig(
                    nbits=2, mode="KMEANS", weight_threshold=50
                ),
                op_type_configs={
                    "const": cto.coreml.OpPalettizerConfig(
                        nbits=4, mode="KMEANS", weight_threshold=50
                    )
                },
            )
        )

        with pytest.raises(
            ValueError,
            match="const ops cannot be set by the `set_op_type` function. Please use `set_global`",
        ):
            compressor.apply(prog)


class TestConfigurationFromDictFromYaml:
    """
    Test the from_dict and from_yaml functionality.
    """

    @staticmethod
    def load_to_yaml(config_dict):
        with tempfile.NamedTemporaryFile("w") as file:
            yaml.dump(config_dict, file)
            yaml_dict = yaml.safe_load(open(file.name))
            file.close()
        return yaml_dict

    @staticmethod
    def get_yaml(config_dict):
        with tempfile.NamedTemporaryFile("w", delete=False) as file:
            yaml.dump(config_dict, file)
            return file.name

    def get_opt_config(self, config_dict, from_yaml, yaml_as_string):
        if from_yaml:
            yaml_file_name = self.get_yaml(config_dict)
            if not yaml_as_string:
                yaml = open(yaml_file_name)
            else:
                yaml = yaml_file_name
            config = quantization.OptimizationConfig.from_yaml(yaml)
            os.remove(yaml_file_name)
        else:
            config = quantization.OptimizationConfig.from_dict(config_dict)
        return config

    @staticmethod
    @pytest.mark.parametrize(
        "config_cls",
        [
            quantization.OpLinearQuantizerConfig,
            quantization.OpThresholdPrunerConfig,
            quantization.OpMagnitudePrunerConfig,
            quantization.OpPalettizerConfig,
        ],
    )
    def test_config_load_invalid_key(config_cls):
        # Invalid key
        config_dict = {"invalid": 2}
        with pytest.raises(cattrs.errors.ClassValidationError):
            config_cls._from_dict(config_dict)

    @pytest.mark.parametrize(
        "mode, dtype, granularity, block_size, weight_threshold, use_yaml",
        itertools.product(
            ["linear", "linear_symmetric"],
            ["int4", "uint4", "int8", "uint8", np.int8, np.uint8, types.int8, types.uint8],
            ["per_tensor", "per_channel", "per_block"],
            [0, 1, 2, [0, 1]],
            [1024, None],
            [True, False],
        ),
    )
    def test_linear_quantizer_config_load_stress(
        self, mode, dtype, granularity, block_size, weight_threshold, use_yaml
    ):
        config_dict = {
            "mode": mode,
            "dtype": dtype,
            "granularity": granularity,
            "block_size": block_size,
            "weight_threshold": weight_threshold,
        }

        if use_yaml and isinstance(dtype, str):
            config_dict = self.load_to_yaml(config_dict)

        config = quantization.OpLinearQuantizerConfig._from_dict(config_dict)

        expected_config = quantization.OpLinearQuantizerConfig(
            mode=mode,
            dtype=dtype,
            granularity=granularity,
            block_size=block_size,
            weight_threshold=weight_threshold,
        )
        assert config == expected_config

    @pytest.mark.parametrize(
        "threshold, minimum_sparsity_percentile, weight_threshold, use_yaml",
        itertools.product(
            [0.0, 1.0],
            [0.0, 1.0],
            [1024, None],
            [True, False],
        ),
    )
    def test_threshold_pruner_config_load_stress(
        self, threshold, minimum_sparsity_percentile, weight_threshold, use_yaml
    ):
        config_dict = {
            "threshold": threshold,
            "minimum_sparsity_percentile": minimum_sparsity_percentile,
            "weight_threshold": weight_threshold,
        }

        if use_yaml:
            config_dict = self.load_to_yaml(config_dict)

        config = quantization.OpThresholdPrunerConfig._from_dict(config_dict)

        expected_config = quantization.OpThresholdPrunerConfig(
            threshold=threshold,
            minimum_sparsity_percentile=minimum_sparsity_percentile,
            weight_threshold=weight_threshold,
        )
        assert config == expected_config

    @pytest.mark.parametrize(
        "n_m_ratio, dim, weight_threshold, use_yaml",
        itertools.product(
            [[1, 1], (2, 3)],
            [0, 1],
            [1024, None],
            [True, False],
        ),
    )
    def test_magnitude_nm_pruner_config_load_stress(
        self, n_m_ratio, dim, weight_threshold, use_yaml
    ):
        config_dict = {
            "n_m_ratio": n_m_ratio,
            "dim": dim,
            "weight_threshold": weight_threshold,
        }

        if use_yaml and not isinstance(n_m_ratio, tuple):
            config_dict = self.load_to_yaml(config_dict)

        config = quantization.OpMagnitudePrunerConfig._from_dict(config_dict)

        expected_config = quantization.OpMagnitudePrunerConfig(
            n_m_ratio=tuple(n_m_ratio),
            dim=dim,
            weight_threshold=weight_threshold,
        )
        assert config == expected_config

    @pytest.mark.parametrize(
        "target_sparsity, block_size, dim, weight_threshold, use_yaml",
        itertools.product(
            [0.0, 1.0],
            [None, 2],
            [None, 0, 1],
            [None, 1024],
            [True, False],
        ),
    )
    def test_magnitude_block_sparsity_pruner_config_load_stress(
        self, target_sparsity, block_size, dim, weight_threshold, use_yaml
    ):
        if block_size is None and dim is not None:
            return

        config_dict = {
            "target_sparsity": target_sparsity,
            "block_size": block_size,
            "dim": dim,
            "weight_threshold": weight_threshold,
        }

        if use_yaml:
            config_dict = self.load_to_yaml(config_dict)

        config = quantization.OpMagnitudePrunerConfig._from_dict(config_dict)

        expected_config = quantization.OpMagnitudePrunerConfig(
            target_sparsity=target_sparsity,
            block_size=block_size,
            dim=dim,
            weight_threshold=weight_threshold,
        )
        assert config == expected_config

    @pytest.mark.parametrize(
        "mode, nbits, granularity, group_size, channel_axis, weight_threshold, num_kmeans_workers, use_yaml",
        itertools.product(
            ["kmeans", "uniform"],
            [1, 2, 3, 4, 6, 8],
            ["per_tensor", "per_grouped_channel"],
            [0, 1, 32],
            [None, 0, 1],
            [1024, None],
            [1, 4],
            [True, False],
        ),
    )
    def test_palettizer_config_load_stress(
        self,
        mode,
        nbits,
        granularity,
        group_size,
        channel_axis,
        weight_threshold,
        num_kmeans_workers,
        use_yaml,
    ):
        config_dict = {
            "mode": mode,
            "nbits": nbits,
            "granularity": granularity,
            "group_size": group_size,
            "channel_axis": channel_axis,
            "weight_threshold": weight_threshold,
            "num_kmeans_workers": num_kmeans_workers,
        }

        if use_yaml:
            config_dict = self.load_to_yaml(config_dict)

        config = quantization.OpPalettizerConfig._from_dict(config_dict)

        expected_config = quantization.OpPalettizerConfig(
            mode=mode,
            nbits=nbits,
            granularity=granularity,
            group_size=group_size,
            channel_axis=channel_axis,
            weight_threshold=weight_threshold,
            num_kmeans_workers=num_kmeans_workers,
        )
        assert config == expected_config

    @pytest.mark.parametrize(
        "from_yaml, yaml_as_string",
        itertools.product(
            [True, False],
            [True, False],
        ),
    )
    def test_optimization_config_load_corner_cases(self, from_yaml, yaml_as_string):
        config_dict = {
            "bobby_joe": 56,
        }
        with pytest.raises(
            ValueError, match="Invalid key bobby_joe to construct an OptimizationConfig object."
        ):
            self.get_opt_config(config_dict, from_yaml, yaml_as_string)

        config_dict = {
            "global_config": None,
        }
        with pytest.raises(ValueError, match="config_type must be provided with type of string."):
            self.get_opt_config(config_dict, from_yaml, yaml_as_string)

        config_dict = {
            "config_type": "OpLinearQuantizerConfig",
            "op_type_configs": 123,
        }
        with pytest.raises(ValueError, match="op_type_configs must be type of dict. Got"):
            self.get_opt_config(config_dict, from_yaml, yaml_as_string)

        config_dict = {
            "config_type": "OpLinearQuantizerConfig",
            "op_name_configs": "eric",
        }
        with pytest.raises(ValueError, match="op_name_configs must be type of dict. Got"):
            self.get_opt_config(config_dict, from_yaml, yaml_as_string)

        # check that the value of the dictionary can be None or not provided
        config_dict = {
            "config_type": "OpLinearQuantizerConfig",
        }
        config = self.get_opt_config(config_dict, from_yaml, yaml_as_string)

        assert config.global_config is None
        assert config.op_type_configs == {}
        assert config.op_name_configs == {}

        config_dict = {
            "config_type": "OpLinearQuantizerConfig",
            "global_config": None,
            "op_type_configs": {
                "conv": None,
            },
            "op_name_configs": {
                "op_1": None,
            },
        }
        config = self.get_opt_config(config_dict, from_yaml, yaml_as_string)
        assert config.global_config is None
        assert config.op_type_configs["conv"] is None
        assert config.op_name_configs["op_1"] is None

    @pytest.mark.parametrize(
        "from_yaml, yaml_as_string",
        itertools.product(
            [True, False],
            [True, False],
        ),
    )
    def test_optimization_config_load_linear_quantizer(self, from_yaml, yaml_as_string):
        config_dict = {
            "config_type": "OpLinearQuantizerConfig",
            "global_config": {
                "mode": "linear",
                "dtype": "int8",
                "weight_threshold": None,
            },
            "op_type_configs": {
                "linear": {
                    "mode": "linear_symmetric",
                    "dtype": "uint8",
                    "weight_threshold": None,
                },
            },
            "op_name_configs": {
                "op_1": {
                    "mode": "linear_symmetric",
                    "dtype": "int8",
                    "weight_threshold": 2047,
                },
                "op_2": {
                    "mode": "linear",
                    "dtype": "uint8",
                    "weight_threshold": 1,
                },
            },
        }
        config = self.get_opt_config(config_dict, from_yaml, yaml_as_string)

        expected_global_config = quantization.OpLinearQuantizerConfig(
            mode="linear",
            dtype=np.int8,
            weight_threshold=None,
        )
        assert config.global_config == expected_global_config

        expected_config = quantization.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=np.uint8,
            weight_threshold=None,
        )
        assert config.op_type_configs["linear"] == expected_config

        expected_config = quantization.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=np.int8,
            weight_threshold=2047,
        )
        assert config.op_name_configs["op_1"] == expected_config

        expected_config = quantization.OpLinearQuantizerConfig(
            mode="linear",
            dtype=np.uint8,
            weight_threshold=1,
        )
        assert config.op_name_configs["op_2"] == expected_config

    @pytest.mark.parametrize(
        "from_yaml, yaml_as_string",
        itertools.product(
            [True, False],
            [True, False],
        ),
    )
    def test_optimization_config_load_pruner(self, from_yaml, yaml_as_string):
        """
        This test also checking the override of the config_type
        """
        config_dict = {
            "config_type": "OpThresholdPrunerConfig",
            "global_config": {
                "config_type": "OpMagnitudePrunerConfig",
                "target_sparsity": 0.3,
            },
            "op_type_configs": {
                "linear": {
                    "config_type": "OpMagnitudePrunerConfig",
                    "n_m_ratio": [4, 5],
                    "dim": 0,
                    "weight_threshold": 2,
                },
                "conv": {
                    "threshold": 0.01,
                    "minimum_sparsity_percentile": 0.01,
                    "weight_threshold": 45,
                },
            },
            "op_name_configs": {
                "op_1": {
                    "threshold": 0.1,
                    "minimum_sparsity_percentile": 0.1,
                    "weight_threshold": 1,
                },
                "op_2": {
                    "config_type": "OpMagnitudePrunerConfig",
                    "target_sparsity": 0.5,
                    "block_size": 100,
                },
            },
        }
        config = self.get_opt_config(config_dict, from_yaml, yaml_as_string)

        expected_global_config = quantization.OpMagnitudePrunerConfig(
            target_sparsity=0.3,
        )
        assert config.global_config == expected_global_config

        expected_config = quantization.OpMagnitudePrunerConfig(
            n_m_ratio=(4, 5),
            dim=0,
            weight_threshold=2,
        )
        assert config.op_type_configs["linear"] == expected_config

        expected_config = quantization.OpThresholdPrunerConfig(
            threshold=0.01,
            minimum_sparsity_percentile=0.01,
            weight_threshold=45,
        )
        assert config.op_type_configs["conv"] == expected_config

        expected_config = quantization.OpThresholdPrunerConfig(
            threshold=0.1,
            minimum_sparsity_percentile=0.1,
            weight_threshold=1,
        )
        assert config.op_name_configs["op_1"] == expected_config

        expected_config = quantization.OpMagnitudePrunerConfig(
            target_sparsity=0.5,
            block_size=100,
        )
        assert config.op_name_configs["op_2"] == expected_config

    @pytest.mark.parametrize(
        "from_yaml, yaml_as_string",
        itertools.product(
            [True, False],
            [True, False],
        ),
    )
    def test_optimization_config_load_palettizer(self, from_yaml, yaml_as_string):
        config_dict = {
            "config_type": "OpPalettizerConfig",
            "global_config": {
                "mode": "kmeans",
                "nbits": 1,
                "weight_threshold": 2,
            },
            "op_type_configs": {
                "linear": {
                    "mode": "uniform",
                    "nbits": 6,
                    "weight_threshold": None,
                },
            },
            "op_name_configs": {
                "op_1": {
                    "config_type": "OpPalettizerConfig",
                    "mode": "unique",
                },
            },
        }
        config = self.get_opt_config(config_dict, from_yaml, yaml_as_string)

        expected_global_config = quantization.OpPalettizerConfig(
            mode="kmeans",
            nbits=1,
            weight_threshold=2,
        )
        assert config.global_config == expected_global_config

        expected_config = quantization.OpPalettizerConfig(
            mode="uniform",
            nbits=6,
            weight_threshold=None,
        )
        assert config.op_type_configs["linear"] == expected_config

        expected_config = quantization.OpPalettizerConfig(
            mode="unique",
        )
        assert config.op_name_configs["op_1"] == expected_config


class TestLinearActivationQuantizer(TestCompressionPasses):
    @pytest.mark.parametrize(
        "mode, dtype, weight_threshold",
        itertools.product(
            ["LINEAR_SYMMETRIC"],
            [np.int8, types.int8],
            [1000],
        ),
    )
    def test_global_config_activation_quantizer_on_pattern_1(self, mode, dtype, weight_threshold):
        """
        Global config would compress all operations with the same config
        Valid patterns:
        - conv
        - conv + relu
        """

        # Insert prefix quantize/dequantize pairs
        op_config = cto.coreml.experimental.OpActivationLinearQuantizerConfig(
            mode=mode, dtype=dtype, weight_threshold=weight_threshold
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config)

        # Test case: conv
        prog = self._get_test_program_conv()

        # Create activation_stats to all intermediate tensors
        activation_stats = gen_activation_stats_for_program(prog)

        # Insert prefix quantize/dequantize pairs
        graph_pass_1 = _insert_prefix_quantize_dequantize_pair(config)
        graph_pass_1.set_options([PassOption("activation_stats", activation_stats)])

        # Insert suffix quantize/dequantize pairs
        graph_pass_2 = PASS_REGISTRY["compression::insert_suffix_quantize_dequantize_pair"]
        graph_pass_2.set_options(
            [PassOption("config", config), PassOption("activation_stats", activation_stats)]
        )

        apply_pass_and_basic_check(prog, graph_pass_1)
        apply_pass_and_basic_check(prog, graph_pass_2)

        assert get_op_types_in_program(prog) == [
            "cast",
            "quantize",
            "dequantize",
            "conv",
            "quantize",
            "dequantize",
            "cast",
        ]

        # Test case: conv + relu
        prog = self._get_test_program_conv_relu()

        # Create activation_stats to all intermediate tensors
        activation_stats = gen_activation_stats_for_program(prog)

        # Insert prefix quantize/dequantize pairs
        graph_pass_1 = _insert_prefix_quantize_dequantize_pair(config)
        graph_pass_1.set_options([PassOption("activation_stats", activation_stats)])

        # Insert suffix quantize/dequantize pairs
        graph_pass_2 = PASS_REGISTRY["compression::insert_suffix_quantize_dequantize_pair"]
        graph_pass_2.set_options(
            [PassOption("config", config), PassOption("activation_stats", activation_stats)]
        )

        apply_pass_and_basic_check(prog, graph_pass_1)
        apply_pass_and_basic_check(prog, graph_pass_2)

        assert get_op_types_in_program(prog) == [
            "cast",
            "quantize",
            "dequantize",
            "conv",
            "relu",
            "quantize",
            "dequantize",
            "cast",
        ]

    @pytest.mark.parametrize(
        "mode, dtype, weight_threshold",
        itertools.product(
            ["LINEAR_SYMMETRIC"],
            [np.int8, types.int8],
            [1000],
        ),
    )
    def test_global_config_activation_quantizer_on_pattern_2(self, mode, dtype, weight_threshold):
        """
        Global config would compress all operations with the same config
        Valid patterns: add
        """

        op_config = cto.coreml.experimental.OpActivationLinearQuantizerConfig(
            mode=mode, dtype=dtype, weight_threshold=weight_threshold
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config)

        # Create activation_stats to all intermediate tensors
        prog = self._get_test_program_add()
        activation_stats = gen_activation_stats_for_program(prog)

        # Insert prefix quantize/dequantize pairs
        graph_pass_1 = _insert_prefix_quantize_dequantize_pair(config)
        graph_pass_1.set_options([PassOption("activation_stats", activation_stats)])

        # Insert suffix quantize/dequantize pairs
        graph_pass_2 = PASS_REGISTRY["compression::insert_suffix_quantize_dequantize_pair"]
        graph_pass_2.set_options(
            [PassOption("config", config), PassOption("activation_stats", activation_stats)]
        )

        # Test case: add
        apply_pass_and_basic_check(prog, graph_pass_1)
        apply_pass_and_basic_check(prog, graph_pass_2)

        assert get_op_types_in_program(prog) == [
            "cast",
            "cast",
            "quantize",
            "dequantize",
            "quantize",
            "dequantize",
            "add",
            "quantize",
            "dequantize",
            "cast",
        ]

    @pytest.mark.parametrize(
        "mode, dtype, weight_threshold",
        itertools.product(
            ["LINEAR_SYMMETRIC"],
            [np.int8, types.int8],
            [1000],
        ),
    )
    def test_global_config_activation_quantizer_on_pattern_3(self, mode, dtype, weight_threshold):
        """
        Global config would compress all operations with the same config
        Valid pattern: pooling (avg_pool, max_pool)
        """

        op_config = cto.coreml.experimental.OpActivationLinearQuantizerConfig(
            mode=mode, dtype=dtype, weight_threshold=weight_threshold
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config)

        # Test case: avg_pool
        # Create activation_stats to all intermediate tensors
        prog = self._get_test_program_avgpool()
        activation_stats = gen_activation_stats_for_program(prog)

        # Insert prefix quantize/dequantize pairs
        graph_pass_1 = _insert_prefix_quantize_dequantize_pair(config)
        graph_pass_1.set_options([PassOption("activation_stats", activation_stats)])

        # Insert suffix quantize/dequantize pairs
        graph_pass_2 = PASS_REGISTRY["compression::insert_suffix_quantize_dequantize_pair"]
        graph_pass_2.set_options(
            [PassOption("config", config), PassOption("activation_stats", activation_stats)]
        )

        apply_pass_and_basic_check(prog, graph_pass_1)
        apply_pass_and_basic_check(prog, graph_pass_2)

        assert get_op_types_in_program(prog) == [
            "cast",
            "quantize",
            "dequantize",
            "avg_pool",
            "quantize",
            "dequantize",
            "cast",
        ]

        # Test case: max_pool
        prog = self._get_test_program_maxpool()

        # Create activation_stats to all intermediate tensors.
        activation_stats = gen_activation_stats_for_program(prog)

        # Insert prefix quantize/dequantize pairs
        graph_pass_1 = _insert_prefix_quantize_dequantize_pair(config)
        graph_pass_1.set_options([PassOption("activation_stats", activation_stats)])

        # Insert suffix quantize/dequantize pairs
        graph_pass_2 = PASS_REGISTRY["compression::insert_suffix_quantize_dequantize_pair"]
        graph_pass_2.set_options(
            [PassOption("config", config), PassOption("activation_stats", activation_stats)]
        )

        apply_pass_and_basic_check(prog, graph_pass_1)
        apply_pass_and_basic_check(prog, graph_pass_2)

        assert get_op_types_in_program(prog) == [
            "cast",
            "quantize",
            "dequantize",
            "max_pool",
            "quantize",
            "dequantize",
            "cast",
        ]


class TestGetActivationStats(TestCompressionPasses):
    def test_get_activation_calibration_stats_basic(self):
        """
        Calibration a floating point model with sample data.
        """

        # Prepare sample data
        sample_data = []
        for _ in range(3):
            input_data = np.random.rand(5, 10, 4, 4)
            sample_data.append({"data": input_data})

        # Loading a floating point mlmodel
        mlmodel = self._get_test_mlmodel_conv_relu()

        activation_stats = _get_activation_calibration_stats(mlmodel, sample_data)

    def test_get_activation_calibration_stats_skip_invalid_ops(self):
        """
        Calibration a floating point model with sample data.
        rdar://130623705 A unit test for model with boolean type intermediate tensor.
        """

        # Prepare sample data
        sample_data = []
        for _ in range(3):
            input_data = np.random.rand(1, 28 * 28)
            sample_data.append({"data": input_data})

        # Loading a floating point mlmodel
        mlmodel = self._get_test_mlmodel_boolean_type()

        activation_stats = _get_activation_calibration_stats(mlmodel, sample_data)

    def test_get_activation_calibration_stats_concat_surrounding_ops(self):
        """
        Calibration a floating point model with sample data.
        rdar://132017374 A unit test for model with concat would be surrounded by quantize/dequantize pairs after activation quantization.
        The activation_stats of concat surrounding nodes should be the same, so quantize/dequantize pairs could share same scale/zp.
        """

        # Prepare sample data
        sample_data = []
        for _ in range(3):
            input_data = np.random.rand(5, 10, 4, 4)
            sample_data.append({"data_0": input_data})

        # Loading a floating point mlmodel
        mlmodel = self._get_test_mlmodel_conv_concat()

        activation_stats = _get_activation_calibration_stats(mlmodel, sample_data)

        activation_stats_unique = set()
        for value in activation_stats.values():
            activation_stats_unique.add((value["rmin"], value["rmax"]))

        # Since mlmodel has a concat with 2 inputs and 1 output, we should see at least 3 rmin/rmax pairs are identical in activation_stats.
        # If we dedup rmin/rmax pairs with identical values, the length of unique values should at least reduced by 2 compared with original one.
        assert len(activation_stats) - len(activation_stats_unique) >= 2
