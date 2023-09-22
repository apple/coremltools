# Copyright (c) 2023, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import os
import tempfile

import cattrs
import numpy as np
import pytest
import yaml

import coremltools as ct
import coremltools.optimize as cto
import coremltools.optimize.coreml._quantization_passes as quantization
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.tests.test_passes import CONSTEXPR_FUNCS, CONSTEXPR_OPS
from coremltools.converters.mil.testing_utils import get_op_types_in_program


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
    This test also convers the backward compatibility test for the deprecated ct.compression_utils.
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

class TestOptimizationConfig(TestCompressionPasses):
    """
    Test some basic funtionality of the OptimizationConfig.
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
        This test is checking the graph pass compresses the program correctly according to the following heirarchical order (high -> low):
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
    This test is checking error handling for invalid configuraion.
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
        "mode, dtype, weight_threshold, use_yaml",
        itertools.product(
            ["linear", "linear_symmetric"],
            ["int8", "uint8", np.int8, np.uint8, types.int8, types.uint8],
            [1024, None],
            [True, False],
        ),
    )
    def test_linear_quantizer_config_load_stress(self, mode, dtype, weight_threshold, use_yaml):
        config_dict = {
            "mode": mode,
            "dtype": dtype,
            "weight_threshold": weight_threshold,
        }

        if use_yaml and dtype in ("int8", "uint8"):
            config_dict = self.load_to_yaml(config_dict)

        config = quantization.OpLinearQuantizerConfig._from_dict(config_dict)

        if dtype in ["int8", np.int8, types.int8]:
            expected_dtype = np.int8
        elif dtype in ["uint8", np.uint8, types.uint8]:
            expected_dtype = np.uint8

        expected_config = quantization.OpLinearQuantizerConfig(
            mode=mode,
            dtype=expected_dtype,
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
        "mode_nbits, weight_threshold, use_yaml",
        itertools.product(
            [
                ("kmeans", 2),
                ("uniform", 1),
                ("unique", None),
            ],
            [None, 1024],
            [True, False],
        ),
    )
    def test_palettizer_config_load_stress(self, mode_nbits, weight_threshold, use_yaml):
        mode, nbits = mode_nbits

        config_dict = {
            "mode": mode,
            "nbits": nbits,
            "weight_threshold": weight_threshold,
        }

        if use_yaml:
            config_dict = self.load_to_yaml(config_dict)

        config = quantization.OpPalettizerConfig._from_dict(config_dict)

        expected_config = quantization.OpPalettizerConfig(
            mode=mode,
            nbits=nbits,
            weight_threshold=weight_threshold,
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
