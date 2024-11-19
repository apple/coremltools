# Copyright (c) 2023, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import logging
import platform
import re
import shutil
import tempfile
from typing import Tuple

import numpy as np
import pytest
import torch

import coremltools as ct
import coremltools.optimize as cto
from coremltools._deps import _HAS_SKLEARN
from coremltools.converters.mil.frontend.torch.test.test_torch_conversion_api import (
    TestPyTorchConverterExamples,
)
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.tests.iOS18 import backends
from coremltools.converters.mil.testing_reqs import compute_units
from coremltools.converters.mil.testing_utils import compute_snr_and_psnr, get_op_types_in_program
from coremltools.models.utils import MultiFunctionDescriptor, _macos_version, save_multifunction
from coremltools.optimize.coreml import _utils as optimize_utils
from coremltools.optimize.coreml._post_training_quantization import CoreMLWeightMetaData
from coremltools.test.ml_program.test_compression import get_test_model_and_data


# Wrapper functions that create the optimization config and call ct.optimize.coreml APIs
def linear_quantize_weights(mlmodel, mode="linear", dtype=np.int8):
    op_config = cto.coreml.OpLinearQuantizerConfig(mode=mode, dtype=dtype)
    config = cto.coreml.OptimizationConfig(global_config=op_config)
    return cto.coreml.linear_quantize_weights(mlmodel, config)

def palettize_weights(mlmodel, nbits=None, mode="kmeans", lut_function=None):
    op_config = cto.coreml.OpPalettizerConfig(mode=mode, nbits=nbits, lut_function=lut_function)
    config = cto.coreml.OptimizationConfig(global_config=op_config)
    return cto.coreml.palettize_weights(mlmodel, config)

def prune_weights(
        mlmodel,
        mode="threshold_based",
        threshold=1e-3,
        target_sparsity=1.0,
        block_size=-1,
        n_m_ratio=(),
    ):
    if mode == "threshold_based":
        op_config = cto.coreml.OpThresholdPrunerConfig(
            threshold=threshold,
            minimum_sparsity_percentile=0.0,
        )
    elif mode == "percentile_based":
        op_config = cto.coreml.OpMagnitudePrunerConfig(
            target_sparsity=target_sparsity,
        )
    elif mode == "block_sparsity":
        op_config = cto.coreml.OpMagnitudePrunerConfig(
            target_sparsity=target_sparsity,
            block_size=block_size,
        )
    else:
        assert mode == "n_m_pruning"
        op_config = cto.coreml.OpMagnitudePrunerConfig(
            n_m_ratio=n_m_ratio,
        )

    config = cto.coreml.OptimizationConfig(global_config=op_config)
    return cto.coreml.prune_weights(mlmodel, config)

def decompress_weights(mlmodel):
    return cto.coreml.decompress_weights(mlmodel)


# Utility functions for testing
def get_test_model_and_data_complex():
    inputs = [ct.TensorType(name="data", shape=(1, 64, 10, 10))]
    torch_input_values = [torch.rand(*i.shape.to_list()) for i in inputs]
    coreml_input_values = {
        i.name: val.detach().numpy() for i, val in zip(inputs, torch_input_values)
    }
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv_1 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2)
            self.conv_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
            self.linear_1 = torch.nn.Linear(64, 128)
            self.linear_2 = torch.nn.Linear(128, 256)
            self.lstm = torch.nn.LSTM(256, 80)

        def forward(self, x):
            conv_1 = self.conv_1(x)
            conv_2 = self.conv_2(conv_1)
            reshape = torch.reshape(conv_2, (1, 64, 64))
            linear_1 = self.linear_1(reshape)
            linear_2 = self.linear_2(linear_1)
            lstm = self.lstm(linear_2)
            return lstm[0]

    return Model().eval(), inputs, torch_input_values, coreml_input_values


def get_test_model_and_data_conv_transpose():
    """Two conv transpose layer which share the same weight."""
    inputs = [ct.TensorType(name="data", shape=(1, 64, 5, 5))]
    torch_input_values = [torch.rand(*i.shape.to_list()) for i in inputs]
    coreml_input_values = {
        i.name: val.detach().numpy() for i, val in zip(inputs, torch_input_values)
    }

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv_transpose1 = torch.nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=2
            )
            self.conv_transpose2 = torch.nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=2
            )
            self.conv_transpose1.weight = self.conv_transpose2.weight

        def forward(self, x):
            return self.conv_transpose1(x) + self.conv_transpose2(x)

    return Model().eval(), inputs, torch_input_values, coreml_input_values


def create_unique_weight(weight, nbits, vector_size=1, vector_axis=None):
    shape = list(weight.detach().numpy().shape)
    unique_number = 1 << nbits

    if vector_size == 1:
        weight = np.random.randint(low=0, high=unique_number, size=shape)
    else:
        if shape[vector_axis] % vector_size != 0:
            raise ValueError(
                f"weight's dim at {vector_axis}th axis must be divisible by "
                f"vector_size {vector_size}"
            )
        # Swap the dim size of vector_axis with last dim.
        shape[vector_axis], shape[-1] = shape[-1], shape[vector_axis]
        shape[-1] //= vector_size
        weight = np.random.randint(low=0, high=unique_number, size=shape)
        weight = np.repeat(weight, vector_size, axis=-1)
        weight = np.swapaxes(weight, -1, vector_axis)

    return weight.astype(np.float32)


def create_sparse_weight(weight, target_sparsity):
    shape = list(weight.shape)
    size = np.prod(shape)
    weight = 100 * np.random.rand(size)
    num_of_zeros = int(size * target_sparsity)
    weight[:num_of_zeros] = 0
    return np.reshape(weight, shape).astype(np.float32)


def create_quantize_friendly_weight(
    weight: np.ndarray, nbits: int, signed: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create quantize friendly weight by first quantize and then de-quantize the weight."""
    axes = tuple(axis for axis in range(len(weight.shape)) if axis != 0)
    quantized_weight, scale, zero_point = optimize_utils.quantize_weight(
        weight,
        axes,
        nbits,
        signed,
        quantization_mode="LINEAR",
        dtype=np.int8 if signed else np.uint8,
    )
    scale_shape = scale.shape + tuple([1] * len(axes))
    scale = scale.reshape(scale_shape)
    zero_point = zero_point.reshape(scale_shape)
    dequantized_weight = scale * (
        quantized_weight.astype(np.float32) - zero_point.astype(np.float32)
    )
    return dequantized_weight, scale, zero_point


def create_weight_with_inf(weight, inf_ratio, inf_val=np.inf):
    """Create weight that has the same shape as input weight, but with inf_ratio elements set as inf_val."""
    shape = list(weight.shape)
    size = np.prod(shape)
    weight = 100 * np.random.rand(size)
    num_of_inf = int(size * inf_ratio)
    weight[:num_of_inf] = inf_val
    np.random.shuffle(weight)
    return np.reshape(weight, shape).astype(np.float32)


def verify_model_outputs(model, compressed_model, input_values, rtol=1e-7, atol=0.0):
    """
    This utility functions does the following checks:

    (1) Verify the output of the compressed model has the same shape / type of the original model
    (2) The decompressed and compressed model have the same numerical outputs
    """

    # Make sure the model can be decompressed
    decompressed_model = decompress_weights(compressed_model)

    # Validate the output shape / type
    ref_outputs = model._mil_program.functions["main"].outputs
    outputs = compressed_model._mil_program.functions["main"].outputs

    assert len(ref_outputs) == len(outputs)

    for a, b in zip(ref_outputs, outputs):
        assert a.name == b.name
        assert a.shape == a.shape
        assert a.dtype == b.dtype

    if ct.utils._macos_version() < (13, 0):
        return

    # Validate that the compressed model could be decompressed, and produces correct outputs
    output_dict = compressed_model.predict(input_values)
    de_output_dict = decompressed_model.predict(input_values)
    for k, v in de_output_dict.items():
        assert k in output_dict
        np.testing.assert_allclose(v, output_dict[k], rtol=rtol, atol=atol)


class TestLinearQuantizeWeights:
    @staticmethod
    def test_linear_quantization_with_classifier():
        traced_model, example_input = TestPyTorchConverterExamples._get_classifier_model()
        for class_type in ("str", "int"):
            mlmodel = TestPyTorchConverterExamples._convert_classifier_model(
                traced_model, example_input, class_type
            )
            config = cto.coreml.OptimizationConfig()
            global_config = cto.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype=np.int8, weight_threshold=0
            )
            config.set_global(global_config)
            mlmodel = cto.coreml.linear_quantize_weights(mlmodel, config)
            expected_ops = [
                "cast",
                "reshape",
                "constexpr_affine_dequantize",
                "linear",
                "relu",
                "constexpr_affine_dequantize",
                "linear",
                "relu",
                "constexpr_affine_dequantize",
                "linear",
                "cast",
                "classify",
            ]
            assert get_op_types_in_program(mlmodel._mil_program) == expected_ops

    @staticmethod
    def test_linear_quantization():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32)

        config = cto.coreml.OptimizationConfig()
        conv_config = cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype=np.int8, weight_threshold=500)
        lstm_config = cto.coreml.OpLinearQuantizerConfig(mode="linear", dtype=np.uint8, weight_threshold=4800)

        config.set_op_type("conv", conv_config)
        config.set_op_type("lstm", lstm_config)
        config.set_op_name("conv_2_1", None)

        mlmodel = cto.coreml.linear_quantize_weights(mlmodel, config)
        expected_ops = [
            "constexpr_affine_dequantize",
            "conv",
            "conv",
            "reshape",
            "linear",
            "linear",
            "constexpr_affine_dequantize",
            "constexpr_affine_dequantize",
            "constexpr_affine_dequantize",
            "lstm",
        ]
        prog = mlmodel._mil_program
        assert get_op_types_in_program(prog) == expected_ops
        assert prog.find_ops(op_type="conv")[1].weight.op.op_type == "const"

        expected_dtype = [np.int8, np.uint8, np.uint8, np.uint8, np.uint8]
        affine_ops = prog.find_ops(op_type="constexpr_affine_dequantize")
        for dtype, op in zip(expected_dtype, affine_ops):
            assert op.quantized_data.val.dtype == dtype

    @staticmethod
    @pytest.mark.parametrize(
        "mode, dtype",
        itertools.product(
            ("linear", "linear_symmetric"),
            (np.int8, np.uint8, types.int8, types.uint8),
        ),
    )
    def test_linear_quanitzation_stress(mode, dtype):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")

        mlmodel_quantized = linear_quantize_weights(mlmodel, mode=mode, dtype=dtype)

        # validate parameters
        expected_ops = ['constexpr_affine_dequantize', 'cast', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_quantized._mil_program) == expected_ops

        quanitze_op = mlmodel_quantized._mil_program.functions["main"].find_ops(op_type="constexpr_affine_dequantize")[0]
        assert model.weight.detach().numpy().shape == quanitze_op.quantized_data.shape

        verify_model_outputs(mlmodel, mlmodel_quantized, coreml_input_values)

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_blockwise_quantization(self, compute_unit, backend):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        config = cto.coreml.OptimizationConfig()
        conv_config = cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int4",
            granularity="per_block",
            block_size=2,
            weight_threshold=500,
        )
        lstm_config = cto.coreml.OpLinearQuantizerConfig(
            mode="linear",
            dtype="int4",
            granularity="per_block",
            block_size=2,
            weight_threshold=4800,
        )

        config.set_op_type("conv", conv_config)
        config.set_op_type("lstm", lstm_config)
        # Set a specific conv's config to None to prevent it being compressed.
        conv_not_to_compress_name = "conv_2_1"
        if backend.precision == "fp16":
            conv_not_to_compress_name += "_cast_fp16"
        config.set_op_name(conv_not_to_compress_name, None)

        mlmodel_quantized = cto.coreml.linear_quantize_weights(mlmodel, config)
        expected_ops = [
            "constexpr_blockwise_shift_scale",
            "conv",
            "conv",
            "reshape",
            "linear",
            "linear",
            "constexpr_blockwise_shift_scale",
            "constexpr_blockwise_shift_scale",
            "constexpr_blockwise_shift_scale",
            "lstm",
        ]
        prog = mlmodel_quantized._mil_program
        assert get_op_types_in_program(prog) == expected_ops
        assert prog.find_ops(op_type="conv")[1].weight.op.op_type == "const"

        quantize_ops = prog.find_ops(op_type="constexpr_blockwise_shift_scale")
        for quantize_op in quantize_ops:
            assert quantize_op.data.dtype == types.int4
            assert types.builtin_to_string(quantize_op.scale.dtype) == backend.precision

        if _macos_version() >= (15, 0):
            verify_model_outputs(
                mlmodel, mlmodel_quantized, coreml_input_values, rtol=1e-2, atol=4e-2
            )

    @staticmethod
    @pytest.mark.parametrize(
        "compute_unit, backend, mode, nbits, signed, block_size",
        itertools.product(
            compute_units,
            backends,
            ("linear", "linear_symmetric"),
            (4, 8),
            (True, False),
            (0, 1, 2, 4),
        ),
    )
    def test_blockwise_quanitzation_stress(compute_unit, backend, mode, nbits, signed, block_size):
        if platform.machine() == "x86_64":
            pytest.xfail("rdar://137153993 ([CI] Quantization Tests Failing only on *native* x86_64 (not with Rosetta))")

        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        dtype_str = types.builtin_to_string(types.get_nbits_int_builtin_type(nbits, signed))
        op_config = cto.coreml.OpLinearQuantizerConfig(
            mode=mode, dtype=dtype_str, granularity="per_block", block_size=block_size
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config)
        mlmodel_quantized = cto.coreml.linear_quantize_weights(mlmodel, config)

        # Verify ops.
        if backend.precision == "fp16":
            # For fp16 precision there is no extra cast op inserted.
            expected_ops = ["constexpr_blockwise_shift_scale", "conv"]
        else:
            expected_ops = ["constexpr_blockwise_shift_scale", "cast", "conv", "cast"]
        assert get_op_types_in_program(mlmodel_quantized._mil_program) == expected_ops
        quantize_op = mlmodel_quantized._mil_program.functions["main"].find_ops(
            op_type="constexpr_blockwise_shift_scale"
        )[0]
        assert types.builtin_to_string(quantize_op.data.dtype) == dtype_str
        # For sub-byte dtype, we still use np.int8/uint8 to store the data.
        assert quantize_op.data.val.dtype == np.int8 if signed else np.uint8
        assert model.weight.detach().numpy().size == quantize_op.data.val.size
        # Weight shape is [32, 64, 2, 2]. The scale's shape reflects number of blocks on each axis.
        assert quantize_op.scale.shape == (32, 64 // block_size if block_size > 0 else 1, 1, 1)

        if _macos_version() >= (15, 0):
            verify_model_outputs(mlmodel, mlmodel_quantized, coreml_input_values)

            # The verify_model_outputs only check compressed and decompressed consistency.
            # Also need to compare original and compressed model.
            original_output = mlmodel.predict(coreml_input_values)
            quantized_output = mlmodel_quantized.predict(coreml_input_values)

            for k, v in quantized_output.items():

                if nbits <= 4 and block_size != 1:
                    # Low-bit has too much info lost when block size is not 1.
                    continue

                # When nbits is larger and block_size is smaller, the info lost is less.
                atol, rtol = 0.4, 0.4
                if block_size == 1 and nbits > 4:
                    atol, rtol = 1e-2, 1e-2

                np.testing.assert_allclose(v, original_output[k], atol=atol, rtol=rtol)

    @staticmethod
    @pytest.mark.parametrize(
        "compute_unit, backend, mode, nbits",
        itertools.product(
            compute_units,
            backends,
            ("linear", "linear_symmetric"),
            (4, 8),
        ),
    )
    def test_per_tensor_quantization_with_blockwise_op(compute_unit, backend, mode, nbits):
        if platform.machine() == "x86_64":
            pytest.xfail("rdar://137153993 ([CI] Quantization Tests Failing only on *native* x86_64 (not with Rosetta))")

        op_config = cto.coreml.OpLinearQuantizerConfig(
            mode=mode, dtype=f"int{nbits}", granularity="per_tensor"
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config)

        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data(
            quantize_config=op_config
        )
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        mlmodel_quantized = cto.coreml.linear_quantize_weights(mlmodel, config)

        # Verify ops.
        if backend.precision == "fp16":
            # For fp16 precision there is no extra cast op inserted.
            expected_ops = ["constexpr_blockwise_shift_scale", "conv"]
        else:
            expected_ops = ["constexpr_blockwise_shift_scale", "cast", "conv", "cast"]
        assert get_op_types_in_program(mlmodel_quantized._mil_program) == expected_ops
        quantize_op = mlmodel_quantized._mil_program.functions["main"].find_ops(
            op_type="constexpr_blockwise_shift_scale"
        )[0]
        assert types.builtin_to_string(quantize_op.data.dtype) == f"int{nbits}"
        if mode == "linear":
            assert types.builtin_to_string(quantize_op.offset.dtype) == f"int{nbits}"
        # For int4, we still use np.int8 to store the data.
        assert quantize_op.data.val.dtype == np.int8
        assert model.weight.detach().numpy().size == quantize_op.data.val.size

        if _macos_version() >= (15, 0):
            verify_model_outputs(mlmodel, mlmodel_quantized, coreml_input_values)

    @staticmethod
    @pytest.mark.parametrize(
        "compute_unit, backend, mode, nbits, granularity",
        itertools.product(
            compute_units,
            backends,
            ("linear", "linear_symmetric"),
            (4, 8),
            ("per_tensor", "per_channel", "per_block"),
        ),
    )
    def test_quantization_conv_transpose_axis(compute_unit, backend, mode, nbits, granularity):
        """The conv_transpose has [Cin, Cout, ...], which is different from conv."""
        (
            model,
            inputs,
            torch_input_values,
            coreml_input_values,
        ) = get_test_model_and_data_conv_transpose()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        dtype_str = f"int{nbits}"
        op_config = cto.coreml.OpLinearQuantizerConfig(
            mode=mode, dtype=dtype_str, granularity=granularity
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config)
        mlmodel_quantized = cto.coreml.linear_quantize_weights(mlmodel, config)

        # Verify ops.
        if backend.precision == "fp16":
            # For fp16 precision there is no extra cast op inserted.
            expected_ops = [
                "constexpr_blockwise_shift_scale",
                "conv_transpose",
                "conv_transpose",
                "add",
            ]
        else:
            expected_ops = [
                "constexpr_blockwise_shift_scale",
                "cast",
                "conv_transpose",
                "conv_transpose",
                "add",
                "cast",
            ]
        assert get_op_types_in_program(mlmodel_quantized._mil_program) == expected_ops

        # Verify quantization ops are on the expected axis.
        quantize_op = mlmodel_quantized._mil_program.functions["main"].find_ops(
            op_type="constexpr_blockwise_shift_scale"
        )[0]
        assert types.builtin_to_string(quantize_op.data.dtype) == dtype_str
        if granularity == "per_tensor":
            expected_scale_shape = (1, 1, 1, 1)
        elif granularity == "per_channel":
            # The weight has shape [64, 32, 2, 2], and the second axis is output channel.
            expected_scale_shape = (1, 32, 1, 1)
        else:
            # The per_block has default block_size 32.
            expected_scale_shape = (64 // 32, 32, 1, 1)
        assert quantize_op.scale.shape == expected_scale_shape

        if _macos_version() >= (15, 0):
            verify_model_outputs(mlmodel, mlmodel_quantized, coreml_input_values, atol=2e-2)

    @staticmethod
    @pytest.mark.parametrize(
        "backend, skip_model_load",
        itertools.product(backends, (True, False)),
    )
    def test_skip_model_load_in_compression_pass(backend, skip_model_load):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16,
            skip_model_load=skip_model_load,
        )

        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int4",
                granularity="per_block",
                block_size=2,
                weight_threshold=500,
            )
        )
        mlmodel_quantized = cto.coreml.linear_quantize_weights(mlmodel, config)

        if skip_model_load:
            # If the mlmodel before compression is not compiled and loaded, the compression pass
            # should keep the model skip_model_load.
            with pytest.raises(Exception, match="Cannot make predictions"):
                mlmodel_quantized.predict(coreml_input_values)
        else:
            mlmodel_quantized.predict(coreml_input_values)

    @staticmethod
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_skip_inf(compute_unit, backend):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()

        # Make conv_1's weight with inf and linear_2's weight with -inf.
        weight_1_inf = create_weight_with_inf(model.conv_1.weight, 0.1, np.inf)
        weight_2_neg_inf = create_weight_with_inf(model.linear_2.weight, 0.1, -np.inf)
        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(torch.Tensor(weight_1_inf))
            model.linear_2.weight = torch.nn.Parameter(torch.Tensor(weight_2_neg_inf))

        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                weight_threshold=500,
            )
        )
        mlmodel_quantized = cto.coreml.linear_quantize_weights(mlmodel, config)

        prog = mlmodel_quantized._mil_program
        quant_op_name = (
            "constexpr_blockwise_shift_scale"
            if backend.opset_version >= ct.target.iOS18
            else "constexpr_affine_dequantize"
        )
        # The conv_1 and linear_2 quantization is skipped due to inf in weights.
        conv_ops = prog.find_ops(op_type="conv")
        assert conv_ops[0].weight.op.op_type == "const"
        assert conv_ops[1].weight.op.op_type == quant_op_name
        linear_ops = prog.find_ops(op_type="linear")
        assert linear_ops[0].weight.op.op_type == quant_op_name
        assert linear_ops[1].weight.op.op_type == "const"

        if _macos_version() >= (13, 0):
            verify_model_outputs(mlmodel, mlmodel_quantized, coreml_input_values)

    @staticmethod
    @pytest.mark.parametrize(
        "compute_unit, backend, use_int01_as_bool",
        itertools.product(compute_units, backends, (True, False)),
    )
    def test_skip_bool(compute_unit, backend, use_int01_as_bool):
        batch_dim, seq_len = 128, 512
        embedding_dim, projected_embedding_dim = 32, 64

        if use_int01_as_bool:

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                    self.wq = torch.nn.Linear(embedding_dim, projected_embedding_dim)
                    self.wk = torch.nn.Linear(embedding_dim, projected_embedding_dim)
                    self.wv = torch.nn.Linear(embedding_dim, projected_embedding_dim)
                    self.wo = torch.nn.Linear(projected_embedding_dim, embedding_dim)

                    all_trues = torch.full((seq_len, seq_len), True)
                    self.causal_mask = torch.tril(all_trues)

                    self.k_cache = torch.rand((1, seq_len, projected_embedding_dim))
                    self.v_cache = torch.rand((1, seq_len, projected_embedding_dim))

                @torch.no_grad()
                def forward(self, embedding, input_pos):
                    input_pos = input_pos[0]

                    q = self.wq(embedding)
                    current_k = self.wk(embedding)
                    current_v = self.wv(embedding)

                    k = torch.ops.aten.index_put_(
                        self.k_cache, [None, input_pos], current_k[:, 0, :]
                    )
                    v = torch.ops.aten.index_put_(
                        self.v_cache, [None, input_pos], current_v[:, 0, :]
                    )
                    # This torch index op will be translated to Core ML gather op
                    # Since Core ML gather op does not support bool,
                    # Core ML will cast bool mask to int8, gather, then cast back to bool.
                    # The int8 cast is then const eliminated, resulting in int8 0/1 const mask
                    mask = torch.ops.aten.index(self.causal_mask, [input_pos])

                    attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
                    o = self.wo(attention)
                    return o

            coreml_input_types = [
                ct.TensorType(name="embedding", shape=(1, 1, embedding_dim)),
                ct.TensorType(name="input_pos", shape=(1,), dtype=np.int32),
            ]
            torch_input_values = [
                torch.rand((1, 1, embedding_dim)),
                torch.tensor([4], dtype=torch.int32),
            ]

        else:

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                    self.wq = torch.nn.Linear(embedding_dim, projected_embedding_dim)
                    self.wk = torch.nn.Linear(embedding_dim, projected_embedding_dim)
                    self.wv = torch.nn.Linear(embedding_dim, projected_embedding_dim)
                    self.wo = torch.nn.Linear(projected_embedding_dim, embedding_dim)

                    all_trues = torch.full((seq_len, seq_len), True)
                    self.causal_mask = torch.tril(all_trues)

                def forward(self, embedding):
                    q = self.wq(embedding)
                    k = self.wk(embedding)
                    v = self.wv(embedding)
                    attention = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, self.causal_mask
                    )
                    o = self.wo(attention)
                    return o

            coreml_input_types = [
                ct.TensorType(name="embedding", shape=(batch_dim, seq_len, embedding_dim))
            ]
            torch_input_values = [torch.rand((batch_dim, seq_len, embedding_dim))]

        model = Model()
        model.eval()

        coreml_input_values = {
            i.name: val.detach().numpy() for i, val in zip(coreml_input_types, torch_input_values)
        }

        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=coreml_input_types,
            convert_to="mlprogram",
            # must deploy to iOS 18+ for scaled_dot_product_attention op
            minimum_deployment_target=max(backend.opset_version, ct.target.iOS18),
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                weight_threshold=500,
            )
        )
        mlmodel_quantized = cto.coreml.linear_quantize_weights(mlmodel, config)

        prog = mlmodel_quantized._mil_program
        quant_op_name = (
            "constexpr_blockwise_shift_scale"
            if backend.opset_version >= ct.target.iOS18
            else "constexpr_affine_dequantize"
        )
        # The sdpa mask quantization is skipped due to bool
        sdpa_op = prog.find_ops(op_type="scaled_dot_product_attention")[0]
        if use_int01_as_bool:
            cast_int8_to_bool_op = sdpa_op.attn_mask.op
            assert cast_int8_to_bool_op.op_type == "cast"
            gather_op = cast_int8_to_bool_op.x.op
            assert gather_op.op_type == "gather"
            assert gather_op.x.op.op_type == "const"
        else:
            assert sdpa_op.attn_mask.op.op_type == "const"
        # The linear ops, however, should all have weight quantized
        linear_ops = prog.find_ops(op_type="linear")
        for linear_op in linear_ops:
            assert linear_op.weight.op.op_type == quant_op_name

        if _macos_version() >= (13, 0):
            if use_int01_as_bool:
                pytest.xfail(
                    "Cannot run model, because: "
                    "1. BNNS does not support sdpa; "
                    "2. Classic CPU does not support int8 weight; "
                    "3. GPU supports everything but segmentor just won't choose it, "
                    "even if set compute_units=ct.ComputeUnit.CPU_AND_GPU. "
                    "So in the end the model gets dispatched to classic CPU "
                    "and we observe the int8 weight error"
                )
            else:
                verify_model_outputs(mlmodel, mlmodel_quantized, coreml_input_values)


class TestPalettizeWeights:
    @staticmethod
    def test_palettization_with_classifier():
        traced_model, example_input = TestPyTorchConverterExamples._get_classifier_model()
        for class_type in ("str", "int"):
            mlmodel = TestPyTorchConverterExamples._convert_classifier_model(
                traced_model, example_input, class_type
            )
            config = cto.coreml.OptimizationConfig()
            global_config = cto.coreml.OpPalettizerConfig(
                nbits=8, mode="kmeans", weight_threshold=2
            )
            config.set_global(global_config)
            mlmodel = cto.coreml.palettize_weights(mlmodel, config)
            expected_ops = [
                "cast",
                "reshape",
                "constexpr_lut_to_dense",
                "linear",
                "relu",
                "constexpr_lut_to_dense",
                "linear",
                "relu",
                "constexpr_lut_to_dense",
                "linear",
                "cast",
                "classify",
            ]
            assert get_op_types_in_program(mlmodel._mil_program) == expected_ops

    @staticmethod
    def test_palettization():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32)

        config = cto.coreml.OptimizationConfig()
        global_config = cto.coreml.OpPalettizerConfig(nbits=8, mode="kmeans", weight_threshold=500)
        conv_config = cto.coreml.OpPalettizerConfig(nbits=6, mode="kmeans", weight_threshold=500)
        conv_2_config = cto.coreml.OpPalettizerConfig(nbits=4, mode="kmeans", weight_threshold=500)
        linear_1_config = cto.coreml.OpPalettizerConfig(nbits=2, mode="kmeans", weight_threshold=500)

        config.set_global(global_config)
        config.set_op_type("conv", conv_config)
        config.set_op_name("conv_2_1", conv_2_config)
        config.set_op_name("linear_0", linear_1_config)

        mlmodel = cto.coreml.palettize_weights(mlmodel, config)
        expected_ops = [
            "constexpr_lut_to_dense",
            "constexpr_lut_to_dense",
            "constexpr_lut_to_dense",
            "constexpr_lut_to_dense",
            "conv",
            "conv",
            "reshape",
            "linear",
            "linear",
            "constexpr_lut_to_dense",
            "constexpr_lut_to_dense",
            "constexpr_lut_to_dense",
            "lstm",
        ]
        prog = mlmodel._mil_program
        assert get_op_types_in_program(prog) == expected_ops

        expected_nbits = [6, 4, 2, 8, 8, 8, 8, 8]
        lut_ops = prog.find_ops(op_type="constexpr_lut_to_dense")

        for nbits, op in zip(expected_nbits, lut_ops):
            assert op.lut.val.shape == (2**nbits,)

    @staticmethod
    @pytest.mark.parametrize(
        "mode",
        ("uniform", "kmeans") if _HAS_SKLEARN else ("uniform",)
    )
    def test_weight_palettization_stress(mode):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")
        mlmodel_palettized = palettize_weights(mlmodel, nbits=4, mode=mode)

        # validate parameters
        expected_ops = ['constexpr_lut_to_dense', 'cast', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_palettized._mil_program) == expected_ops

        main_func = mlmodel_palettized._mil_program.functions["main"]
        lut_to_dense_op = main_func.find_ops(op_type="constexpr_lut_to_dense")[0]

        assert lut_to_dense_op.shape.val.tolist() == list(model.weight.detach().numpy().shape)

        # validate the model
        verify_model_outputs(mlmodel, mlmodel_palettized, coreml_input_values)

    @staticmethod
    def test_weight_palettization_unique_case_1():
        # In this model, both conv weights can be palettized
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data(multi_layer=True)

        weight_1_unique = create_unique_weight(model.conv_1.weight, nbits=2)
        weight_2_unique = create_unique_weight(model.conv_2.weight, nbits=6)

        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(torch.Tensor(weight_1_unique))
            model.conv_2.weight = torch.nn.Parameter(torch.Tensor(weight_2_unique))

        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")

        # validate parameters
        mlmodel_palettized = palettize_weights(mlmodel, mode="unique")
        expected_ops = ['constexpr_lut_to_dense', 'cast', 'conv', 'constexpr_lut_to_dense', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_palettized._mil_program) == expected_ops

        main_func = mlmodel_palettized._mil_program.functions["main"]
        lut_to_dense_op_1 = main_func.find_ops(op_type="constexpr_lut_to_dense")[0]
        lut_to_dense_op_2 = main_func.find_ops(op_type="constexpr_lut_to_dense")[1]

        assert lut_to_dense_op_1.shape.val.tolist() == list(model.conv_1.weight.detach().numpy().shape)
        assert lut_to_dense_op_2.shape.val.tolist() == list(model.conv_2.weight.detach().numpy().shape)

        # validate the model
        verify_model_outputs(mlmodel, mlmodel_palettized, coreml_input_values)

    def test_weight_palettization_unique_case_2(self, caplog):
        # In this model, only one conv weights can be palettized, the converter should warn the users that one weight is skipped
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data(multi_layer=True)

        weight_1_unique = create_unique_weight(model.conv_1.weight, nbits=2)

        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(torch.Tensor(weight_1_unique))

        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")

        # validate parameters
        # converter should warn the user that one weight is not compressed
        mlmodel_palettized = palettize_weights(mlmodel, mode="unique")
        warning_msg = "Unique values in weight cannot be represented by 8 bits palettization."
        assert any([warning_msg in rec.message for rec in caplog.records])

        expected_ops = ['constexpr_lut_to_dense', 'cast', 'conv', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_palettized._mil_program) == expected_ops

        main_func = mlmodel_palettized._mil_program.functions["main"]
        lut_to_dense_op_1 = main_func.find_ops(op_type="constexpr_lut_to_dense")[0]
        assert lut_to_dense_op_1.shape.val.tolist() == list(model.conv_1.weight.detach().numpy().shape)

        # validate the model
        verify_model_outputs(mlmodel, mlmodel_palettized, coreml_input_values)

    @staticmethod
    def test_weight_palettization_custom():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")

        def lut_function(weight):
            nbits = 4
            weight = weight.flatten()
            unique_elements = np.unique(weight)
            k = (1 << nbits) - 1
            top_k = np.partition(weight, -k)[-k:]
            np.sort(top_k)
            lut = np.array([0.] + top_k.tolist()).astype(weight.dtype)
            mapping = {v: idx for idx, v in enumerate(lut)}
            indices = np.array([mapping[v] if v in mapping else 0 for v in weight]).astype(np.uint8)
            return lut, indices

        mlmodel_palettized = palettize_weights(mlmodel, mode="custom", lut_function=lut_function)

        # validate parameters
        expected_ops = ['constexpr_lut_to_dense', 'cast', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_palettized._mil_program) == expected_ops

        main_func = mlmodel_palettized._mil_program.functions["main"]
        lut_to_dense_op = main_func.find_ops(op_type="constexpr_lut_to_dense")[0]

        assert lut_to_dense_op.shape.val.tolist() == list(model.weight.detach().numpy().shape)

        # validate the model
        verify_model_outputs(mlmodel, mlmodel_palettized, coreml_input_values)

    @staticmethod
    def test_convert_palettized_source_model_default():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()

        weight_1_unique = create_unique_weight(model.conv_1.weight, nbits=2)
        weight_2_unique = create_unique_weight(model.conv_2.weight, nbits=6)
        linear_1_unique = create_unique_weight(model.linear_1.weight, nbits=4)

        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(torch.Tensor(weight_1_unique))
            model.conv_2.weight = torch.nn.Parameter(torch.Tensor(weight_2_unique))
            model.linear_1.weight = torch.nn.Parameter(torch.Tensor(linear_1_unique))

        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            pass_pipeline=ct.PassPipeline.DEFAULT_PALETTIZATION,
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )

        expected_ops = [
            "constexpr_lut_to_dense",
            "constexpr_lut_to_dense",
            "constexpr_lut_to_dense",
            "conv",
            "conv",
            "reshape",
            "linear",
            "linear",
            "constexpr_lut_to_dense",
            "squeeze",
            "lstm",
        ]
        prog = mlmodel._mil_program
        assert get_op_types_in_program(prog) == expected_ops

        expected_nbits = [2, 6, 4, 1, 1]
        lut_ops = prog.find_ops(op_type="constexpr_lut_to_dense")

        for nbits, op in zip(expected_nbits, lut_ops):
            assert op.lut.val.shape == (2**nbits,)

    @staticmethod
    def test_convert_palettized_source_model_custom():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()

        weight_1_unique = create_unique_weight(model.conv_1.weight, nbits=2)
        weight_2_unique = create_unique_weight(model.conv_2.weight, nbits=6)
        linear_1_unique = create_unique_weight(model.linear_1.weight, nbits=4)

        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(torch.Tensor(weight_1_unique))
            model.conv_2.weight = torch.nn.Parameter(torch.Tensor(weight_2_unique))
            model.linear_1.weight = torch.nn.Parameter(torch.Tensor(linear_1_unique))

        pipeline = ct.PassPipeline.DEFAULT_PALETTIZATION
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(mode="unique"),
            op_type_configs={
                "conv": None,
                "linear": cto.coreml.OpPalettizerConfig(nbits=1, mode="kmeans"),
            }
        )
        pipeline.set_options("compression::palettize_weights", {"config": config})

        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            pass_pipeline=pipeline,
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )

        expected_ops = [
            "constexpr_lut_to_dense",
            "constexpr_lut_to_dense",
            "conv",
            "conv",
            "reshape",
            "linear",
            "linear",
            "constexpr_lut_to_dense",
            "squeeze",
            "lstm",
        ]
        prog = mlmodel._mil_program
        assert get_op_types_in_program(prog) == expected_ops

        expected_nbits = [1, 1, 1, 1]
        lut_ops = prog.find_ops(op_type="constexpr_lut_to_dense")

        for nbits, op in zip(expected_nbits, lut_ops):
            assert op.lut.val.shape == (2**nbits,)

        conv_ops = prog.find_ops(op_type="conv")
        assert conv_ops[0].weight.op.op_type == "const"
        assert conv_ops[1].weight.op.op_type == "const"

        linear_ops = prog.find_ops(op_type="linear")
        assert linear_ops[0].weight.op.op_type == "constexpr_lut_to_dense"
        assert linear_ops[1].weight.op.op_type == "constexpr_lut_to_dense"

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_channelwise_palettization(self, compute_unit, backend):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        config = cto.coreml.OptimizationConfig()
        conv_config = cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=8,
            granularity="per_grouped_channel",
            group_size=1,
            weight_threshold=500,
        )
        lstm_config = cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=4,
            granularity="per_grouped_channel",
            group_size=1,
            weight_threshold=4800,
        )

        config.set_op_type("conv", conv_config)
        config.set_op_type("lstm", lstm_config)
        # Set a specific conv's config to None to prevent it being compressed.
        conv_not_to_compress_name = "conv_2_1"
        if backend.precision == "fp16":
            conv_not_to_compress_name += "_cast_fp16"
        config.set_op_name(conv_not_to_compress_name, None)

        mlmodel_palettized = cto.coreml.palettize_weights(mlmodel, config)
        expected_ops = [
            "constexpr_lut_to_dense",
            "conv",
            "conv",
            "reshape",
            "linear",
            "linear",
            "constexpr_lut_to_dense",
            "constexpr_lut_to_dense",
            "constexpr_lut_to_dense",
            "lstm",
        ]
        prog = mlmodel_palettized._mil_program
        assert get_op_types_in_program(prog) == expected_ops
        assert prog.find_ops(op_type="conv")[1].weight.op.op_type == "const"

        palettize_ops = prog.find_ops(op_type="constexpr_lut_to_dense")
        for quantize_op in palettize_ops:
            assert types.builtin_to_string(quantize_op.lut.dtype) == backend.precision
        assert types.builtin_to_string(palettize_ops[0].indices.dtype) == "uint8"
        assert types.builtin_to_string(palettize_ops[3].indices.dtype) == "uint4"

        if _macos_version() >= (15, 0):
            verify_model_outputs(mlmodel, mlmodel_palettized, coreml_input_values)

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_channelwise_palettization_unique_skip_op(self, compute_unit, backend, caplog):
        """Test where mode is unique and can't use nbits to represent the weight"""
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()
        traced_model = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        config = cto.coreml.OptimizationConfig()
        global_config = cto.coreml.OpPalettizerConfig(
            mode="unique",
            granularity="per_grouped_channel",
            group_size=1,
            weight_threshold=100,
        )
        # For conv weight in the whole tensor cannot be represented by 2**8 unique values.
        conv_config = cto.coreml.OpPalettizerConfig(
            mode="unique",
            granularity="per_tensor",
            weight_threshold=100,
        )
        config.set_global(global_config)
        config.set_op_type("conv", conv_config)
        mlmodel_palettized = cto.coreml.palettize_weights(mlmodel, config)
        assert any(
            [
                "Unique values in weight cannot be represented by 8 bits palettization."
                in rec.message
                for rec in caplog.records
            ]
        )
        # There is no constexpr for the conv weight.
        for conv_op in mlmodel_palettized._mil_program.find_ops(op_type="conv"):
            assert conv_op.weight.op.op_type == "const"
        # There are still constexpr ops for linear and lstm weights.
        assert len(mlmodel_palettized._mil_program.find_ops(op_type="constexpr_lut_to_dense")) == 5

        if _macos_version() >= (15, 0):
            verify_model_outputs(mlmodel, mlmodel_palettized, coreml_input_values)

    @staticmethod
    @pytest.mark.parametrize(
        "compute_unit, backend, mode, nbits, channel_axis, channel_group_size",
        itertools.product(
            compute_units,
            backends,
            ("kmeans", "uniform"),
            (1, 2, 3, 4, 6, 8),
            (0, 1),
            (0, 1, 2),
        ),
    )
    def test_channelwise_palettization_stress(
        compute_unit, backend, mode, nbits, channel_axis, channel_group_size
    ):
        if platform.machine() == "x86_64":
            pytest.xfail("rdar://137153993 ([CI] Quantization Tests Failing only on *native* x86_64 (not with Rosetta))")

        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        op_config = cto.coreml.OpPalettizerConfig(
            mode=mode,
            nbits=nbits,
            granularity="per_grouped_channel",
            group_size=channel_group_size,
            channel_axis=channel_axis,
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config)
        mlmodel_palettized = cto.coreml.palettize_weights(mlmodel, config)

        # Verify ops.
        if backend.precision == "fp16":
            # For fp16 precision there is no extra cast op inserted.
            expected_ops = ["constexpr_lut_to_dense", "conv"]
        else:
            expected_ops = ["constexpr_lut_to_dense", "cast", "conv", "cast"]
        assert get_op_types_in_program(mlmodel_palettized._mil_program) == expected_ops
        palettize_op = mlmodel_palettized._mil_program.functions["main"].find_ops(
            op_type="constexpr_lut_to_dense"
        )[0]
        assert types.builtin_to_string(palettize_op.indices.dtype) == f"uint{nbits}"
        # For uint4, we still use np.uint8 to store the data.
        assert palettize_op.indices.val.dtype == np.uint8
        assert model.weight.detach().numpy().shape == palettize_op.indices.val.shape

        if _macos_version() >= (15, 0):
            verify_model_outputs(mlmodel, mlmodel_palettized, coreml_input_values)

            # The verify_model_outputs compares the decompressed model with compressed model.
            # We further compare the compressed model with original model.
            ref_output_dict = mlmodel.predict(coreml_input_values)
            output_dict = mlmodel_palettized.predict(coreml_input_values)
            for k, v in output_dict.items():
                assert k in ref_output_dict
                if nbits == 1:
                    continue  # nbits=1 numerical loss is too significant.
                elif nbits <= 3:
                    large_diff_count = np.sum((v - ref_output_dict[k]) > 0.2)
                    threshold = 0.15 if channel_group_size != 0 else 0.5
                    assert large_diff_count / v.size < threshold
                elif nbits < 8:
                    np.testing.assert_almost_equal(v, ref_output_dict[k], decimal=1)
                else:
                    err_tol = 1e-5 if mode == "kmeans" and channel_group_size == 1 else 1e-2
                    np.testing.assert_allclose(v, ref_output_dict[k], atol=err_tol, rtol=err_tol)

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_grouped_channelwise_palettization_better_than_per_tensor(self, compute_unit, backend):
        """The grouped channelwise lut should be better than per-tensor lut."""
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        per_tensor_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=4,
                granularity="per_tensor",
            )
        )
        grouped_channelwise_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=4,
                granularity="per_grouped_channel",
                group_size=1,
            )
        )

        if _macos_version() < (15, 0):
            pytest.skip("Channelwise palettization prediction only support in iOS18+")

        mlmodel_per_tensor_palettized = cto.coreml.palettize_weights(mlmodel, per_tensor_config)
        mlmodel_grouped_channelwise_palettized = cto.coreml.palettize_weights(
            mlmodel, grouped_channelwise_config
        )
        output_ref = mlmodel.predict(coreml_input_values)
        output_per_tensor = mlmodel_per_tensor_palettized.predict(coreml_input_values)
        output_grouped_channelwise = mlmodel_grouped_channelwise_palettized.predict(
            coreml_input_values
        )
        for k_ref, v_ref in output_ref.items():
            snr_per_tensor = compute_snr_and_psnr(v_ref, output_per_tensor[k_ref])[0]
            snr_grouped_channelwise = compute_snr_and_psnr(
                v_ref, output_grouped_channelwise[k_ref]
            )[0]
            assert snr_grouped_channelwise > snr_per_tensor

    def test_channelwise_palettization_invalid_config(self):
        with pytest.raises(ValueError, match='Invalid value of "nbits" \(7\) for palettization'):
            cto.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=7,
                granularity="per_tensor",
                weight_threshold=500,
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, group_size",
        itertools.product(compute_units, backends, [1, 16]),
    )
    def test_convert_palettized_model_with_pipeline(self, compute_unit, backend, group_size):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data(
            multi_layer=True
        )
        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(
                torch.Tensor(create_unique_weight(model.conv_1.weight, nbits=2))
            )
            model.conv_2.weight = torch.nn.Parameter(
                torch.Tensor(create_unique_weight(model.conv_2.weight, nbits=6))
            )

        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        pass_pipeline = ct.PassPipeline.DEFAULT_PALETTIZATION
        pass_pipeline.set_options(
            "compression::palettize_weights",
            {
                "config": cto.coreml.OptimizationConfig(
                    global_config=cto.coreml.OpPalettizerConfig(
                        mode="unique", granularity="per_grouped_channel", group_size=group_size
                    )
                )
            },
        )
        mlmodel_palettized = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
            pass_pipeline=pass_pipeline,
        )

        expected_ops = ["constexpr_lut_to_dense", "constexpr_lut_to_dense", "conv", "conv"]
        assert get_op_types_in_program(mlmodel_palettized._mil_program) == expected_ops
        palettize_ops = mlmodel_palettized._mil_program.functions["main"].find_ops(
            op_type="constexpr_lut_to_dense"
        )
        assert types.builtin_to_string(palettize_ops[0].indices.dtype) == "uint2"
        assert palettize_ops[0].lut.shape == (32 // group_size, 1, 1, 1, 4, 1)
        assert types.builtin_to_string(palettize_ops[1].indices.dtype) == "uint6"
        assert palettize_ops[1].lut.shape == (64 // group_size, 1, 1, 1, 64, 1)

        if _macos_version() >= (15, 0):
            verify_model_outputs(mlmodel, mlmodel_palettized, coreml_input_values)

    @pytest.mark.parametrize(
        "compute_unit, backend, mode, cluster_dim",
        itertools.product(compute_units, backends, ("kmeans", "unique"), (2, 4)),
    )
    def test_vector_palettization(self, compute_unit, backend, mode, cluster_dim):
        """Test the vector palettization (cluster_dim > 1)."""
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        if mode == "unique":
            weight_unique = create_unique_weight(
                model.weight, nbits=4, vector_size=cluster_dim, vector_axis=0
            )
            with torch.no_grad():
                model.weight = torch.nn.Parameter(torch.Tensor(weight_unique))

        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        vector_lut_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(
                mode=mode,
                nbits=4 if mode == "kmeans" else None,
                granularity="per_grouped_channel",
                group_size=0,
                cluster_dim=cluster_dim,
                weight_threshold=500,
            )
        )
        mlmodel_palettized = cto.coreml.palettize_weights(mlmodel, vector_lut_config)

        # Verify ops.
        palettize_op = mlmodel_palettized._mil_program.functions["main"].find_ops(
            op_type="constexpr_lut_to_dense"
        )[0]
        produced_nbits = 4
        assert types.builtin_to_string(palettize_op.indices.dtype) == f"uint{produced_nbits}"
        # The shape on the Cout (0th for conv) should match after multiplying cluster_dim.
        assert model.weight.shape[0] == palettize_op.indices.val.shape[0] * cluster_dim
        # The last dim of lut should match cluster_dim.
        assert palettize_op.lut.shape[-2:] == (2**produced_nbits, cluster_dim)

        if _macos_version() >= (15, 4):
            verify_model_outputs(mlmodel, mlmodel_palettized, coreml_input_values)

    @pytest.mark.parametrize(
        "compute_unit, backend, mode, cluster_dim, op_type",
        itertools.product(
            compute_units, backends, ("kmeans", "unique"), (2, 4), ("conv", "conv_transpose")
        ),
    )
    def test_vector_palettization_skip_conv(
        self, compute_unit, backend, mode, cluster_dim, op_type, caplog
    ):
        """Test grouped conv/conv_transpose where effective dim size is not divisible by cluster_dim."""
        inputs = [ct.TensorType(name="data", shape=(1, 32, 10, 10))]
        torch_input_values = [torch.rand(*i.shape.to_list()) for i in inputs]
        if op_type == "conv":
            model = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, groups=32)
        else:
            model = torch.nn.ConvTranspose2d(
                in_channels=32, out_channels=32, kernel_size=1, groups=32
            )
        model.eval()

        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        vector_lut_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(
                mode=mode,
                nbits=4 if mode == "kmeans" else None,
                granularity="per_grouped_channel",
                group_size=0,
                cluster_dim=cluster_dim,
                weight_threshold=30,  # The weight shape is [32, 1, 1, 1].
            )
        )
        with caplog.at_level(logging.WARNING):
            mlmodel_palettized = cto.coreml.palettize_weights(mlmodel, vector_lut_config)

        # As the effective dim size (1) is not divisible by cluster_dim, the op won't be palettized.
        warning_msg = "The `cluster_dim` is invalid for .* Skipped this op."
        assert any([re.match(warning_msg, rec.message) for rec in caplog.records])
        assert get_op_types_in_program(mlmodel._mil_program) == get_op_types_in_program(
            mlmodel_palettized._mil_program
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_palettization_pcs(self, compute_unit, backend):
        """Test the palettization with per-channel-scale."""
        if platform.machine() == "x86_64":
            pytest.xfail("rdar://137153993 ([CI] Quantization Tests Failing only on *native* x86_64 (not with Rosetta))")

        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()

        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        vector_lut_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=4,
                granularity="per_grouped_channel",
                group_size=0,
                enable_per_channel_scale=True,
                weight_threshold=500,
            )
        )
        mlmodel_palettized = cto.coreml.palettize_weights(mlmodel, vector_lut_config)

        # Verify ops.
        palettize_op = mlmodel_palettized._mil_program.functions["main"].find_ops(
            op_type="constexpr_lut_to_dense"
        )[0]
        assert types.builtin_to_string(palettize_op.indices.dtype) == "uint4"
        # The per-channel-scale is represented by a quant op to do scaling.
        quantize_ops = mlmodel_palettized._mil_program.functions["main"].find_ops(
            op_type="constexpr_blockwise_shift_scale"
        )
        assert len(quantize_ops) > 0
        # Order of quant and lut op is determined by canonicalize_quantized_lut_pattern graph pass.
        assert quantize_ops[0].outputs[0].child_ops[0].op_type == "constexpr_lut_to_dense"

        if _macos_version() >= (15, 0):
            verify_model_outputs(mlmodel, mlmodel_palettized, coreml_input_values)


class TestPruneWeights:
    @staticmethod
    def test_pruning_with_classifier():
        traced_model, example_input = TestPyTorchConverterExamples._get_classifier_model()
        for class_type in ("str", "int"):
            mlmodel = TestPyTorchConverterExamples._convert_classifier_model(
                traced_model, example_input, class_type
            )
            config = cto.coreml.OptimizationConfig()
            global_config = cto.coreml.OpMagnitudePrunerConfig(
                target_sparsity=0.9, weight_threshold=2
            )
            config.set_global(global_config)
            mlmodel = cto.coreml.prune_weights(mlmodel, config)
            expected_ops = [
                "cast",
                "reshape",
                "constexpr_sparse_to_dense",
                "linear",
                "relu",
                "constexpr_sparse_to_dense",
                "linear",
                "relu",
                "constexpr_sparse_to_dense",
                "linear",
                "cast",
                "classify",
            ]
            assert get_op_types_in_program(mlmodel._mil_program) == expected_ops

    @staticmethod
    def test_pruning():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32)

        config = cto.coreml.OptimizationConfig()
        global_config = cto.coreml.OpMagnitudePrunerConfig(target_sparsity=0.9, weight_threshold=500)

        config.set_global(global_config)
        config.set_op_type("lstm", None)
        config.set_op_name("linear_0", None)

        mlmodel = cto.coreml.prune_weights(mlmodel, config)
        expected_ops = [
            "constexpr_sparse_to_dense",
            "constexpr_sparse_to_dense",
            "constexpr_sparse_to_dense",
            "conv",
            "conv",
            "reshape",
            "linear",
            "linear",
            "lstm",
        ]
        prog = mlmodel._mil_program
        assert get_op_types_in_program(prog) == expected_ops
        assert prog.find_ops(op_type="linear")[0].weight.op.op_type == "const"

    @staticmethod
    @pytest.mark.parametrize(
        "threshold",
        (0.0, 0.001, 1e2),
    )
    def test_weight_pruning_threshold_based(threshold):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        with torch.no_grad():
            model.weight[0][0][0][0] = 101
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")
        mlmodel_sparsified = prune_weights(mlmodel, mode="threshold_based", threshold=threshold)

        # validate parameters
        expected_ops = ['constexpr_sparse_to_dense', 'cast', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_sparsified._mil_program) == expected_ops

        main_func = mlmodel_sparsified._mil_program.functions["main"]
        sparse_to_dense_op = main_func.find_ops(op_type="constexpr_sparse_to_dense")[0]
        non_sparse_data = sparse_to_dense_op.nonzero_data

        if threshold != 1e2:
            assert np.min(np.absolute(non_sparse_data.val)) >= threshold
        else:
            assert non_sparse_data.val.size == 1

        assert sparse_to_dense_op.shape.val.tolist() == list(model.weight.detach().numpy().shape)

        # validate the model
        verify_model_outputs(mlmodel, mlmodel_sparsified, coreml_input_values)

    @staticmethod
    @pytest.mark.parametrize(
        "percentile",
        (0., 0.5, 1.0),
    )
    def test_weight_pruning_percentile_based(percentile):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        # Make sure no weight element is randomed to 0, to eliminate testing noise
        # e.g. in percentile 0 test case, we would expect no element gets pruned
        # if there is no 0 in initial weight
        with torch.no_grad():
            non0_weight = torch.where(torch.abs(model.weight) > 1e-6, model.weight, 1e-6)
            model.weight.copy_(non0_weight)
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")
        mlmodel_sparsified = prune_weights(mlmodel, mode="percentile_based", target_sparsity=percentile)

        # validate parameters
        expected_ops = ['constexpr_sparse_to_dense', 'cast', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_sparsified._mil_program) == expected_ops

        main_func = mlmodel_sparsified._mil_program.functions["main"]
        sparse_to_dense_op = main_func.find_ops(op_type="constexpr_sparse_to_dense")[0]
        non_sparse_data = sparse_to_dense_op.nonzero_data
        weight = model.weight.detach().numpy()

        if percentile == 0.:
            assert non_sparse_data.val.size == weight.size
        elif percentile == 0.5:
            lower = 0.49 * weight.size
            upper = 0.51 * weight.size
            actual = non_sparse_data.val.size
            assert lower <= actual and actual <= upper
        else:
            assert non_sparse_data.val.size == 0

        assert sparse_to_dense_op.shape.val.tolist() == list(model.weight.detach().numpy().shape)

        # validate the model
        verify_model_outputs(mlmodel, mlmodel_sparsified, coreml_input_values)

    def test_weight_pruning_block_sparsity(self):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")
        mlmodel_sparsified = prune_weights(mlmodel, mode="block_sparsity", target_sparsity=0.3, block_size=5)

        # validate parameters
        expected_ops = ['constexpr_sparse_to_dense', 'cast', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_sparsified._mil_program) == expected_ops

    def test_weight_pruning_n_m(self):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")
        mlmodel_sparsified = prune_weights(mlmodel, mode="n_m_pruning", n_m_ratio=(2, 3))

        # validate parameters
        expected_ops = ['constexpr_sparse_to_dense', 'cast', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_sparsified._mil_program) == expected_ops

    def test_convert_sparse_source_model_default(self):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()

        weight_1_sparse = create_sparse_weight(model.conv_1.weight, 0.5)
        weight_2_sparse = create_sparse_weight(model.conv_2.weight, 0.1)
        linear_1_sparse = create_sparse_weight(model.linear_1.weight, 0.9)

        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(torch.Tensor(weight_1_sparse))
            model.conv_2.weight = torch.nn.Parameter(torch.Tensor(weight_2_sparse))
            model.linear_1.weight = torch.nn.Parameter(torch.Tensor(linear_1_sparse))

        torchmodel = torch.jit.trace(model, torch_input_values)

        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            pass_pipeline=ct.PassPipeline.DEFAULT_PRUNING,
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )

        prog = mlmodel._mil_program

        # The default minimum_sparsity_percentile is 0.3, so only conv1, linear1, and two initialize states of lstm
        # are compressed

        expected_ops = [
            "constexpr_sparse_to_dense",
            "constexpr_sparse_to_dense",
            "conv",
            "conv",
            "reshape",
            "linear",
            "linear",
            "constexpr_sparse_to_dense",
            "squeeze",
            "lstm",
        ]
        assert get_op_types_in_program(prog) == expected_ops

        conv_ops = prog.find_ops(op_type="conv")
        assert conv_ops[0].weight.op.op_type == "constexpr_sparse_to_dense"
        assert conv_ops[1].weight.op.op_type == "const"

        linear_ops = prog.find_ops(op_type="linear")
        assert linear_ops[0].weight.op.op_type == "constexpr_sparse_to_dense"
        assert linear_ops[1].weight.op.op_type == "const"

    def test_convert_sparse_source_model_custom(self):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()

        weight_1_sparse = create_sparse_weight(model.conv_1.weight, 0.5)
        weight_2_sparse = create_sparse_weight(model.conv_2.weight, 0.1)
        linear_1_sparse = create_sparse_weight(model.linear_1.weight, 0.9)

        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(torch.Tensor(weight_1_sparse))
            model.conv_2.weight = torch.nn.Parameter(torch.Tensor(weight_2_sparse))
            model.linear_1.weight = torch.nn.Parameter(torch.Tensor(linear_1_sparse))

        torchmodel = torch.jit.trace(model, torch_input_values)

        pipeline = ct.PassPipeline.DEFAULT_PRUNING
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpThresholdPrunerConfig(
                threshold=1e-12, minimum_sparsity_percentile=0.05
            ),
            op_type_configs={"conv": None},
        )
        pipeline.set_options("compression::prune_weights", {"config": config})
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            pass_pipeline=pipeline,
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )

        prog = mlmodel._mil_program
        expected_ops = [
            "constexpr_sparse_to_dense",
            "conv",
            "conv",
            "reshape",
            "linear",
            "linear",
            "constexpr_sparse_to_dense",
            "squeeze",
            "lstm",
        ]
        assert get_op_types_in_program(prog) == expected_ops

        conv_ops = prog.find_ops(op_type="conv")
        assert conv_ops[0].weight.op.op_type == "const"
        assert conv_ops[1].weight.op.op_type == "const"

        linear_ops = prog.find_ops(op_type="linear")
        assert linear_ops[0].weight.op.op_type == "constexpr_sparse_to_dense"
        assert linear_ops[1].weight.op.op_type == "const"

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_default_prune_pipeline_ios18(self, compute_unit, backend):
        """Make sure the new iOS18 op is used for DEFAULT_PRUNING pass pipeline."""
        # Make the weight size not divisible by 8, to make sure the internal conversion to ios18
        # sparse_to_dense op handles sub-byte masks correctly.
        model = torch.nn.Linear(21, 121)
        model.eval()
        weight_sparse = create_sparse_weight(model.weight, 0.7)
        with torch.no_grad():
            model.weight = torch.nn.Parameter(torch.Tensor(weight_sparse))

        inputs = [ct.TensorType(name="data", shape=(4, 21))]
        torch_input_values = [torch.rand(*i.shape.to_list()) for i in inputs]
        coreml_input_values = {
            i.name: val.detach().numpy() for i, val in zip(inputs, torch_input_values)
        }
        torchmodel = torch.jit.trace(model, torch_input_values)

        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )
        mlmodel_pruned = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
            pass_pipeline=ct.PassPipeline.DEFAULT_PRUNING,
        )
        sparse_ops = mlmodel_pruned._mil_program.find_ops(op_type="constexpr_sparse_to_dense")
        assert len(sparse_ops) > 0
        for sparse_op in sparse_ops:
            assert types.builtin_to_string(sparse_op.nonzero_data.dtype) == backend.precision
            if backend.opset_version >= ct.target.iOS18:
                assert types.builtin_to_string(sparse_op.mask.dtype) == "uint1"
            else:
                assert types.builtin_to_string(sparse_op.mask.dtype) == "uint8"
                assert types.builtin_to_string(sparse_op.shape.dtype) == "uint32"

        if _macos_version() >= (15, 0):
            verify_model_outputs(mlmodel, mlmodel_pruned, coreml_input_values, rtol=3e-3, atol=2e-3)


class TestJointCompressWeights:
    """Test using coremltools PTQ to do joint compression."""

    @pytest.mark.parametrize(
        "compute_unit, backend, dtype, block_size, output_channel_block_size, prune_first",
        itertools.product(
            compute_units,
            backends,
            ("int4", "int8", "uint4", "uint8"),
            (0, 1, 2),
            (0, 1),
            (True, False),
        ),
    )
    def test_joint_prune_quantize_weights(
        self, compute_unit, backend, dtype, block_size, output_channel_block_size, prune_first
    ):
        """Jointly prune and quantize the model, where non-sparse entries are quantized."""
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        prune_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpMagnitudePrunerConfig(
                target_sparsity=0.5, weight_threshold=500
            )
        )

        quant_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                mode="linear",
                dtype=dtype,
                granularity="per_block",
                block_size=[0, block_size] if output_channel_block_size == 0 else block_size,
                weight_threshold=500,
            ),
            op_type_configs={
                "conv": cto.coreml.OpLinearQuantizerConfig(
                    mode="linear",
                    dtype=dtype,
                    granularity="per_block",
                    block_size=[0, block_size, 0, 0]
                    if output_channel_block_size == 0
                    else block_size,
                    weight_threshold=500,
                ),
            },
        )

        if prune_first:
            mlmodel_pruned = cto.coreml.prune_weights(mlmodel, prune_config)
            mlmodel_joint_pruned_quantized = cto.coreml.linear_quantize_weights(
                mlmodel_pruned, quant_config, joint_compression=True
            )
        else:
            mlmodel_quantized = cto.coreml.linear_quantize_weights(mlmodel, quant_config)
            mlmodel_joint_pruned_quantized = cto.coreml.prune_weights(
                mlmodel_quantized, prune_config, joint_compression=True
            )

        # If run prune first, the all-zero const for lstm won't have nonzero-data, so it won't be
        # further quantized.
        lstm_weight_compression_ops = (
            ["constexpr_sparse_to_dense"]
            if prune_first
            else ["constexpr_sparse_blockwise_shift_scale", "constexpr_sparse_to_dense"]
        )
        expected_ops = (
            ["constexpr_sparse_blockwise_shift_scale", "constexpr_sparse_to_dense", "conv"] * 2
            + ["reshape"]
            + ["constexpr_sparse_blockwise_shift_scale", "constexpr_sparse_to_dense", "linear"] * 2
            + lstm_weight_compression_ops
            + ["constexpr_sparse_blockwise_shift_scale", "constexpr_sparse_to_dense"] * 2
            + ["lstm"]
        )
        prog = mlmodel_joint_pruned_quantized._mil_program
        assert get_op_types_in_program(prog) == expected_ops

        for linear_op in prog.find_ops(op_type="linear"):
            assert linear_op.weight.op.op_type == "constexpr_sparse_to_dense"
        for conv_op in prog.find_ops(op_type="conv"):
            assert conv_op.weight.op.op_type == "constexpr_sparse_to_dense"

        sparse_quantize_ops = prog.find_ops(op_type="constexpr_sparse_blockwise_shift_scale")
        assert len(sparse_quantize_ops) > 0
        for sparse_quantize_op in sparse_quantize_ops:
            assert types.builtin_to_string(sparse_quantize_op.nonzero_data.dtype) == dtype
            assert sparse_quantize_op.data_mask.dtype == types.uint1
            assert sparse_quantize_op.scale.dtype == types.fp16
            assert types.builtin_to_string(sparse_quantize_op.offset.dtype) == dtype
            assert sparse_quantize_op.outputs[1].child_ops[0].op_type == "constexpr_sparse_to_dense"
            # As both quantization and pruning is on the original weight, the shape of scale should
            # match the original weight's shape except on the input/output channel.
            weight_shape = sparse_quantize_op.outputs[1].child_ops[0].outputs[0].shape
            expected_scale_shape = [1] * len(weight_shape)
            if block_size > 0:
                expected_scale_shape[1] = weight_shape[1] // block_size
            if output_channel_block_size > 0:
                expected_scale_shape[0] = weight_shape[0] // output_channel_block_size
            assert sparse_quantize_op.scale.shape == tuple(expected_scale_shape)

        sparse_ops = prog.find_ops(op_type="constexpr_sparse_to_dense")
        assert len(sparse_ops) > 0
        for sparse_op in sparse_ops:
            assert sparse_op.mask.dtype == types.uint1
            assert sparse_op.nonzero_data.dtype == types.fp16

        if _macos_version() >= (15, 0):
            verify_model_outputs(
                mlmodel, mlmodel_joint_pruned_quantized, coreml_input_values, atol=5e-4
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, nbits, channel_group_size, prune_first",
        itertools.product(
            compute_units,
            backends,
            (3, 4, 8),
            (0, 1, 2),
            (True, False),
        ),
    )
    def test_joint_prune_palettize_weights(
        self, compute_unit, backend, nbits, channel_group_size, prune_first
    ):
        """Jointly prune and palettize the model, where non-sparse entries are palettized."""
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        prune_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpMagnitudePrunerConfig(
                target_sparsity=0.2,
                weight_threshold=500,
            )
        )
        palettize_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(
                mode="uniform",
                nbits=nbits,
                granularity="per_grouped_channel",
                group_size=channel_group_size,
                weight_threshold=500,
            )
        )

        if prune_first:
            mlmodel_pruned = cto.coreml.prune_weights(mlmodel, prune_config)
            mlmodel_joint_pruned_palettized = cto.coreml.palettize_weights(
                mlmodel_pruned, palettize_config, joint_compression=True
            )
        else:
            mlmodel_palettized = cto.coreml.palettize_weights(mlmodel, palettize_config)
            mlmodel_joint_pruned_palettized = cto.coreml.prune_weights(
                mlmodel_palettized, prune_config, joint_compression=True
            )

        # If run prune first, the all-zero const for lstm won't have nonzero-data, so it won't be
        # further quantized.
        lstm_weight_compression_ops = (
            ["constexpr_sparse_to_dense"]
            if prune_first
            else ["constexpr_lut_to_sparse", "constexpr_sparse_to_dense"]
        )
        expected_ops = (
            ["constexpr_lut_to_sparse", "constexpr_sparse_to_dense", "conv"] * 2
            + ["reshape"]
            + ["constexpr_lut_to_sparse", "constexpr_sparse_to_dense", "linear"] * 2
            + lstm_weight_compression_ops
            + ["constexpr_lut_to_sparse", "constexpr_sparse_to_dense"] * 2
            + ["lstm"]
        )
        prog = mlmodel_joint_pruned_palettized._mil_program
        assert get_op_types_in_program(prog) == expected_ops

        for linear_op in prog.find_ops(op_type="linear"):
            assert linear_op.weight.op.op_type == "constexpr_sparse_to_dense"
        for conv_op in prog.find_ops(op_type="conv"):
            assert conv_op.weight.op.op_type == "constexpr_sparse_to_dense"

        sparse_palettize_ops = prog.find_ops(op_type="constexpr_lut_to_sparse")
        assert len(sparse_palettize_ops) > 0
        for sparse_palettize_op in sparse_palettize_ops:
            assert sparse_palettize_op.indices_nonzero_data.dtype == types.string_to_builtin(
                f"uint{nbits}"
            )
            assert sparse_palettize_op.indices_mask.dtype == types.uint1
            assert sparse_palettize_op.lut.dtype == types.fp16
            # Both outputs of the sparse_palettize_op should be used by the child constexpr_sparse_to_dense op.
            assert (
                sparse_palettize_op.outputs[0].child_ops[0].op_type == "constexpr_sparse_to_dense"
            )
            assert (
                sparse_palettize_op.outputs[1].child_ops[0].op_type == "constexpr_sparse_to_dense"
            )
            # As both palettization and pruning is on the original weight, the shape of lut should
            # match the original weight's shape except on the output channel.
            weight_shape = sparse_palettize_op.outputs[1].child_ops[0].outputs[0].shape
            expected_lut_shape = [1] * len(weight_shape) + [2**nbits] + [1]
            if channel_group_size > 0:
                expected_lut_shape[0] = weight_shape[0] // channel_group_size
            assert sparse_palettize_op.lut.shape == tuple(expected_lut_shape)

        sparse_ops = prog.find_ops(op_type="constexpr_sparse_to_dense")
        assert len(sparse_ops) > 0
        for sparse_op in sparse_ops:
            assert sparse_op.mask.dtype == types.uint1
            assert sparse_op.nonzero_data.dtype == types.fp16

        if _macos_version() >= (15, 0):
            verify_model_outputs(
                mlmodel, mlmodel_joint_pruned_palettized, coreml_input_values, atol=5e-4
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, nbits, channel_group_size, quantize_first",
        itertools.product(
            compute_units,
            backends,
            (3, 4, 8),
            (0, 1, 2),
            (True, False),
        ),
    )
    def test_joint_palettize_quantize_weights(
        self, compute_unit, backend, nbits, channel_group_size, quantize_first
    ):
        """
        If quantize_first is True:
            First quantize to get int8 weight, and then palettize to n-bit lut with int8 entries.
        If quantize_first is False:
            First palettize to get fp16 lut, and then quantize the lut to make int8 lut.

        Notice no matter applies which one first, the final output model's op order is guaranteed to be consistent
        by the common::canonicalize_quantized_lut_pattern graph pass.
        """
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        palettize_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(
                mode="uniform",
                nbits=nbits,
                granularity="per_grouped_channel",
                group_size=channel_group_size,
                weight_threshold=500,
            )
        )
        quant_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                # Quantize the whole lut tensor as the lut usually is not huge.
                mode="linear",
                dtype="int8",
                granularity="per_tensor",
                weight_threshold=500,
            )
        )

        if quantize_first:
            mlmodel_quantized = cto.coreml.linear_quantize_weights(mlmodel, quant_config)
            mlmodel_joint_palettized_quantized = cto.coreml.palettize_weights(
                mlmodel_quantized, palettize_config, joint_compression=True
            )
        else:
            mlmodel_palettized = cto.coreml.palettize_weights(mlmodel, palettize_config)
            mlmodel_joint_palettized_quantized = cto.coreml.linear_quantize_weights(
                mlmodel_palettized, quant_config, joint_compression=True
            )

        expected_ops = (
            ["constexpr_blockwise_shift_scale", "constexpr_lut_to_dense", "conv"] * 2
            + ["reshape"]
            + ["constexpr_blockwise_shift_scale", "constexpr_lut_to_dense", "linear"] * 2
            + ["constexpr_blockwise_shift_scale", "constexpr_lut_to_dense"] * 3
            + ["lstm"]
        )
        prog = mlmodel_joint_palettized_quantized._mil_program
        if channel_group_size == 0:
            # When doing lut first with per-tensor lut, the lut size is too small, so it's stored as ImmediateValue
            # which won't be quantized.
            ops_in_prog = get_op_types_in_program(prog)
            if nbits < 4 and not quantize_first:
                assert ops_in_prog.count("constexpr_blockwise_shift_scale") == 0
            else:
                assert ops_in_prog.count("constexpr_blockwise_shift_scale") >= 6
        else:
            assert get_op_types_in_program(prog) == expected_ops

        for linear_op in prog.find_ops(op_type="linear"):
            assert linear_op.weight.op.op_type == "constexpr_lut_to_dense"
        for conv_op in prog.find_ops(op_type="conv"):
            assert conv_op.weight.op.op_type == "constexpr_lut_to_dense"

        for quantize_op in prog.find_ops(op_type="constexpr_blockwise_shift_scale"):
            assert quantize_op.data.dtype == types.int8
            assert quantize_op.scale.dtype == types.fp16
            assert quantize_op.offset.dtype == types.int8
            assert quantize_op.outputs[0].child_ops[0].op_type == "constexpr_lut_to_dense"

        for palettize_op in prog.find_ops(op_type="constexpr_lut_to_dense"):
            assert palettize_op.lut.dtype == types.fp16
            assert palettize_op.indices.dtype == types.string_to_builtin(f"uint{nbits}")

        if _macos_version() >= (15, 0):
            verify_model_outputs(mlmodel, mlmodel_joint_palettized_quantized, coreml_input_values)

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_joint_palettize_quantize_weights_invalid(self, compute_unit, backend):
        """Only support per-tensor quantization for this case."""
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        palettize_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(
                mode="uniform",
                nbits=4,
                granularity="per_grouped_channel",
                group_size=1,
                weight_threshold=500,
            )
        )
        quant_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                mode="linear",
                block_size=1,
                weight_threshold=500,
            )
        )

        mlmodel_palettized = cto.coreml.palettize_weights(mlmodel, palettize_config)
        with pytest.raises(
            NotImplementedError,
            match="When use joint compression for palettization-quantization, "
            "please make sure to use per-tensor quantization",
        ):
            cto.coreml.linear_quantize_weights(
                mlmodel_palettized, quant_config, joint_compression=True
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, nbits, channel_group_size, target_sparsity",
        itertools.product(
            compute_units,
            backends,
            (3, 4, 8),
            (0, 1, 2),
            (0.2, 0.8),
        ),
    )
    def test_joint_prune_palettize_quantize_weights(
        self, compute_unit, backend, nbits, channel_group_size, target_sparsity
    ):
        """
        First prune to get sparse weight, and then palettize the non-sparse entries to get fp16
        lut, and then quantize the lut to make int8 lut.
        """
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        prune_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpMagnitudePrunerConfig(
                target_sparsity=target_sparsity, weight_threshold=500
            )
        )
        palettize_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=nbits,
                granularity="per_grouped_channel",
                group_size=channel_group_size,
                weight_threshold=500,
            )
        )
        quant_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                mode="linear",
                dtype="int8",
                granularity="per_tensor",
                weight_threshold=200,  # Need to be smaller than entries in lut (2**8=256).
            )
        )

        mlmodel_pruned = cto.coreml.prune_weights(mlmodel, prune_config)
        mlmodel_joint_pruned_palettized = cto.coreml.palettize_weights(
            mlmodel_pruned, palettize_config, joint_compression=True
        )
        mlmodel_joint_pruned_palettized_quantized = cto.coreml.linear_quantize_weights(
            mlmodel_joint_pruned_palettized, quant_config, joint_compression=True
        )
        expected_ops = (
            [
                "constexpr_blockwise_shift_scale",
                "constexpr_lut_to_sparse",
                "constexpr_sparse_to_dense",
                "conv",
            ]
            * 2
            + ["reshape"]
            + [
                "constexpr_blockwise_shift_scale",
                "constexpr_lut_to_sparse",
                "constexpr_sparse_to_dense",
                "linear",
            ]
            * 2
            + ["constexpr_sparse_to_dense"]
            + [
                "constexpr_blockwise_shift_scale",
                "constexpr_lut_to_sparse",
                "constexpr_sparse_to_dense",
            ]
            * 2
            + ["lstm"]
        )
        if nbits < 4 and channel_group_size == 0:
            # The lut tensor is too small, which is stored as immediate values.
            expected_ops = [
                expected_op
                for expected_op in expected_ops
                if expected_op != "constexpr_blockwise_shift_scale"
            ]
        prog = mlmodel_joint_pruned_palettized_quantized._mil_program
        assert get_op_types_in_program(prog) == expected_ops

        for linear_op in prog.find_ops(op_type="linear"):
            assert linear_op.weight.op.op_type == "constexpr_sparse_to_dense"
        for conv_op in prog.find_ops(op_type="conv"):
            assert conv_op.weight.op.op_type == "constexpr_sparse_to_dense"

        for quantize_op in prog.find_ops(op_type="constexpr_blockwise_shift_scale"):
            assert types.builtin_to_string(quantize_op.data.dtype) == "int8"
            assert types.builtin_to_string(quantize_op.scale.dtype) == backend.precision
            assert types.builtin_to_string(quantize_op.offset.dtype) == "int8"
            assert quantize_op.outputs[0].child_ops[0].op_type == "constexpr_lut_to_sparse"

        for sparse_palettize_op in prog.find_ops(op_type="constexpr_lut_to_sparse"):
            assert (
                types.builtin_to_string(sparse_palettize_op.indices_nonzero_data.dtype)
                == f"uint{nbits}"
            )
            assert sparse_palettize_op.indices_mask.dtype == types.uint1
            assert (
                sparse_palettize_op.outputs[1].child_ops[0].op_type == "constexpr_sparse_to_dense"
            )

        for sparse_op in prog.find_ops(op_type="constexpr_sparse_to_dense"):
            assert sparse_op.mask.dtype == types.uint1
            assert types.builtin_to_string(sparse_op.nonzero_data.dtype) == backend.precision

        if _macos_version() >= (15, 0):
            verify_model_outputs(
                mlmodel,
                mlmodel_joint_pruned_palettized_quantized,
                coreml_input_values,
                atol=5e-4,
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, hidden_size",
        itertools.product(
            compute_units,
            backends,
            (128, 80),
        ),
    )
    def test_lstm_state(self, compute_unit, backend, hidden_size):
        inputs = [ct.TensorType(name="data", shape=(1, 16, 64))]
        torch_input_values = [torch.ones(*i.shape.to_list()) for i in inputs]
        coreml_input_values = {
            i.name: val.detach().numpy() for i, val in zip(inputs, torch_input_values)
        }

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.lstm = torch.nn.LSTM(64, hidden_size)

            def forward(self, x):
                return self.lstm(x)

        torchmodel = torch.jit.trace(Model().eval(), torch_input_values)
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=backend.opset_version,
            compute_precision=ct.precision.FLOAT16
            if backend.precision == "fp16"
            else ct.precision.FLOAT32,
            compute_units=compute_unit,
        )

        prune_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpMagnitudePrunerConfig(
                target_sparsity=0.5, weight_threshold=500
            )
        )
        quant_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                mode="linear",
                granularity="per_block",
                weight_threshold=500,
            )
        )
        mlmodel_pruned = cto.coreml.prune_weights(mlmodel, prune_config)
        mlmodel_joint_pruned_quantized = cto.coreml.linear_quantize_weights(
            mlmodel_pruned, quant_config, joint_compression=True
        )

        if _macos_version() >= (15, 0):
            if hidden_size == 80:
                pytest.xfail("rdar://137003019 (Joint Pruning + Quantization for LSTM Has Incorrect Results on BNNS)")
            verify_model_outputs(
                mlmodel, mlmodel_joint_pruned_quantized, coreml_input_values, atol=1e-3
            )


class TestDecompressWeights:
    @staticmethod
    def test_weight_decopmression_coreml_optimize():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()

        weight_1_sparse = create_sparse_weight(model.conv_1.weight, 0.5)
        weight_2_sparse = create_sparse_weight(model.conv_2.weight, 0.1)
        linear_1_unique = create_unique_weight(model.linear_1.weight, nbits=4)

        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(torch.Tensor(weight_1_sparse))
            model.conv_2.weight = torch.nn.Parameter(torch.Tensor(weight_2_sparse))
            model.linear_1.weight = torch.nn.Parameter(torch.Tensor(linear_1_unique))

        torchmodel = torch.jit.trace(model, torch_input_values)

        pipeline = ct.PassPipeline.DEFAULT_PRUNING

        # Add a palettization pass after the pruning pass.
        prune_pass_idx = pipeline.passes.index("compression::prune_weights")
        pipeline.insert_pass(prune_pass_idx + 1, "compression::palettize_weights")
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(mode="unique"),
        )
        pipeline.set_options("compression::palettize_weights", {"config": config})

        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            pass_pipeline=pipeline,
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )

        decompressed_model = cto.coreml.decompress_weights(mlmodel)
        prog = decompressed_model._mil_program
        op_types =  get_op_types_in_program(prog)
        for val in op_types:
            assert "constexpr" not in val

        if ct.utils._macos_version() < (13, 0):
            return

        if platform.machine() == "x86_64":
            pytest.xfail(
                "rdar://140156685 ([CI] test_weight_decopmression_coreml_optimize fails on x86_64)"
            )

        # compared the numerical outputs
        output_dict = mlmodel.predict(coreml_input_values)
        de_output_dict = decompressed_model.predict(coreml_input_values)

        for k, v in output_dict.items():
            assert k in de_output_dict
            np.testing.assert_allclose(v, de_output_dict[k])


class TestConvertMixedCompression:
    @staticmethod
    def test_convert_sparse_and_palettized_source_model_custom():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()

        weight_1_sparse = create_sparse_weight(model.conv_1.weight, 0.5)
        weight_2_sparse = create_sparse_weight(
            model.conv_2.weight, 0.1
        )  # the sparsity of 0.1 is filtered out by the minimum_sparsity_percentile
        linear_1_unique = create_unique_weight(model.linear_1.weight, nbits=4)

        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(torch.Tensor(weight_1_sparse))
            model.conv_2.weight = torch.nn.Parameter(torch.Tensor(weight_2_sparse))
            model.linear_1.weight = torch.nn.Parameter(torch.Tensor(linear_1_unique))

        torchmodel = torch.jit.trace(model, torch_input_values)

        pipeline = ct.PassPipeline.DEFAULT_PRUNING

        # Add a palettization pass after the pruning pass.
        prune_pass_idx = pipeline.passes.index("compression::prune_weights")
        pipeline.insert_pass(prune_pass_idx + 1, "compression::palettize_weights")
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(mode="unique"),
        )
        pipeline.set_options("compression::palettize_weights", {"config": config})

        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            pass_pipeline=pipeline,
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )

        prog = mlmodel._mil_program
        expected_ops = [
            "constexpr_sparse_to_dense",
            "constexpr_lut_to_dense",
            "conv",
            "conv",
            "reshape",
            "linear",
            "linear",
            "constexpr_sparse_to_dense",
            "squeeze",
            "lstm",
        ]
        assert get_op_types_in_program(prog) == expected_ops

        conv_ops = prog.find_ops(op_type="conv")
        assert conv_ops[0].weight.op.op_type == "constexpr_sparse_to_dense"
        assert conv_ops[1].weight.op.op_type == "const"

        linear_ops = prog.find_ops(op_type="linear")
        assert linear_ops[0].weight.op.op_type == "constexpr_lut_to_dense"
        assert linear_ops[1].weight.op.op_type == "const"

class TestErrorHandling:
    @staticmethod
    def test_error_handling():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")

        # Test invalid mode for affine quantization
        expected_err_str = "supported for weight affine quantization. Got mode"
        with pytest.raises(ValueError, match=expected_err_str):
            linear_quantize_weights(mlmodel, mode="invalid_mode")

        # Test invalid dtype for affine quantization
        expected_err_str = "Should be int4/8 or uint4/8, but got int32"
        with pytest.raises(ValueError, match=expected_err_str):
            linear_quantize_weights(mlmodel, dtype=np.int32)

        expected_err_str = "Should be int4/8 or uint4/8, but got int32"
        with pytest.raises(ValueError, match=expected_err_str):
            linear_quantize_weights(mlmodel, dtype="int32")

        # Test invalid threshold for weight sparsification
        expected_err_str = 'Invalid value of "threshold": \-1.0. Needs to be in \[0, inf\)'
        with pytest.raises(ValueError, match=expected_err_str):
            prune_weights(mlmodel, mode="threshold_based", threshold=-1.0)

        # Test invalid percentile for weight sparsification
        expected_err_str = "Invalid value of \"target_sparsity\": 1.2. Needs to be in \[0, 1\]"
        with pytest.raises(ValueError, match=expected_err_str):
           prune_weights(mlmodel, mode="percentile_based", target_sparsity=1.2)

        # Test invalid mode for weight palettization
        expected_err_str = "supported for weight palettization. Got \"mode\""
        with pytest.raises(ValueError, match=expected_err_str):
            palettize_weights(mlmodel, mode="invalid_mode")

        # Test nbits must be provided for kmeans, uniform mode for weight palettization
        expected_err_str = "\"nbits\" must be provided for"
        with pytest.raises(ValueError, match=expected_err_str):
            palettize_weights(mlmodel, mode="kmeans")

        with pytest.raises(ValueError, match=expected_err_str):
            palettize_weights(mlmodel, mode="uniform")

        # Test nbits must not be provided for unique, custom mode for weight palettization
        expected_err_str = "\"nbits\" must NOT be provided for"
        with pytest.raises(ValueError, match=expected_err_str):
            palettize_weights(mlmodel, mode="unique", nbits=2)

        with pytest.raises(ValueError, match=expected_err_str):
            palettize_weights(mlmodel, mode="custom", nbits=2)

        # Test lut_function must be provided for custom mode, and must not be provided otherwise
        with pytest.raises(ValueError, match="\"lut_function\" can not be None, if \"mode\" is \"custom\"."):
            palettize_weights(mlmodel, mode="custom")
        with pytest.raises(ValueError, match="\"lut_function\" must be None, if \"mode\" is not \"custom\"."):
            palettize_weights(mlmodel, mode="unique", lut_function=lambda op: True)

        # Test lut_function must be a function object
        expected_err_str = "A function object must be provided as \"lut_function\""
        with pytest.raises(ValueError, match=expected_err_str):
            palettize_weights(mlmodel, mode="custom", lut_function=1)

    @staticmethod
    def test_error_out_multifunction():
        # prepare a mlmodel from a torch model
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")

        # make a multifunction model
        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        mlmodel.save(package_path)
        desc = MultiFunctionDescriptor(package_path)
        desc.default_function_name = "main"
        multifunction_path = tempfile.mkdtemp(suffix=".mlpackage")
        save_multifunction(desc, multifunction_path)
        multifunction_mlmodel = ct.models.MLModel(multifunction_path)

        # all PTQ API should error out, until the radar is fixed:
        # rdar://126084385 ([Infra] Figure out the story of PTQ or other passes operate on loaded Mutli-function model)
        def run_palettization(mlmodel):
            return palettize_weights(mlmodel, nbits=2)

        for func in [
            linear_quantize_weights,
            prune_weights,
            run_palettization,
            decompress_weights,
            ct.optimize.coreml.get_weights_metadata,
        ]:
            with pytest.raises(ValueError, match="is not supported for a multifunction model"):
                func(multifunction_mlmodel)

        # cleanup
        shutil.rmtree(package_path)
        shutil.rmtree(multifunction_path)


class TestCoreMLWeightMetaData:
    """
    This test includes unit tests for:
    1. CoreMLWeightMetaData
    2. coremltools.optimize.coreml.get_weights_metadata
    """
    @staticmethod
    def test_coreml_weight_metadata_api():
        """
        Test the example in the CoreMLWeightMetaData api doc string.
        """
        data = np.array([[1.0, 0.0], [0.0, 6.0]], dtype=np.float32)
        meta_data = CoreMLWeightMetaData(data)
        assert meta_data.val is data
        assert meta_data.sparsity == 0.5
        assert meta_data.unique_values == 3

    @staticmethod
    def test_get_weights_metadata():
        """
        Test the example in the get_weights_metadata functionality with op_type is None.
        """
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data_complex()

        weight_1_sparse = create_sparse_weight(model.conv_1.weight, 0.5)
        weight_2_sparse = create_sparse_weight(model.conv_2.weight, 0.8)
        linear_1_palettized = create_unique_weight(model.linear_1.weight, 2)
        linear_2_palettized = create_unique_weight(model.linear_2.weight, 4)

        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(torch.Tensor(weight_1_sparse))
            model.conv_2.weight = torch.nn.Parameter(torch.Tensor(weight_2_sparse))
            model.linear_1.weight = torch.nn.Parameter(torch.Tensor(linear_1_palettized))
            model.linear_2.weight = torch.nn.Parameter(torch.Tensor(linear_2_palettized))

        torchmodel = torch.jit.trace(model, torch_input_values)

        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )

        # test the weight_threshold can filter out weights with size
        weight_threshold = 10
        weight_metadata_dict = ct.optimize.coreml.get_weights_metadata(
            mlmodel, weight_threshold=weight_threshold
        )
        for v in weight_metadata_dict.values():
            assert v.val.size >= weight_threshold

        # test the functionality of using the returned meta data
        weight_metadata_dict = ct.optimize.coreml.get_weights_metadata(mlmodel)

        # get the weight names with size > 25600
        large_weights = []
        for k, v in weight_metadata_dict.items():
            if v.val.size >= 25600:
                large_weights.append(k)

        # get the weight names with sparsity >= 50%
        sparse_weights = []
        for k, v in weight_metadata_dict.items():
            if v.sparsity >= 0.5:
                sparse_weights.append(k)

        # get the weight names with unique elements <= 16
        palettized_weights = []
        for k, v in weight_metadata_dict.items():
            if v.unique_values <= 16:
                palettized_weights.append(k)

        meta_data_1 = weight_metadata_dict["conv_1_weight"]

        # testing
        expected_large_weights = [
            "linear_2_weight",
            "concat_1",
            "concat_2",
        ]
        assert large_weights == expected_large_weights

        expected_sparse_weights = [
            "conv_1_weight",
            "conv_2_weight",
            "op_59_lstm_h0_squeeze",
        ]
        assert sparse_weights == expected_sparse_weights

        expected_palettized_weights = [
            "linear_1_weight",
            "linear_2_weight",
            "op_59_lstm_h0_squeeze",
        ]
        assert palettized_weights == expected_palettized_weights

    @staticmethod
    def test_get_weights_metadata_shared_weight():
        """
        Test the get_weights_metadata functionality for models with weight-sharing layers.
        """
        def _test_child_ops(child_ops):
            assert len(child_ops) == 2

            assert child_ops[0].name == "add_1"
            assert child_ops[0].op_type == "add"
            assert child_ops[0].params_name_mapping["y"] == "w_1"

            assert child_ops[1].name == "add_2"
            assert child_ops[1].op_type == "add"
            assert child_ops[1].params_name_mapping["y"] == "w_1"

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 30, 10, 10)),
                mb.TensorSpec(shape=(1, 30, 10, 10)),
            ],
        )
        def prog(x, y):
            shared_weight = mb.const(
                val=np.random.rand(1, 30, 10, 10).astype(np.float32), name="w_1"
            )
            x = mb.add(x=x, y=shared_weight, name="add_1")
            y = mb.add(x=y, y=shared_weight, name="add_2")
            return x, y

        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT32,
        )

        ops_metadata_dict = ct.optimize.coreml.get_weights_metadata(
            mlmodel,
            weight_threshold=100,
        )
        assert len(ops_metadata_dict) == 1
        child_ops = ops_metadata_dict["w_1"].child_ops
        _test_child_ops(child_ops)

    @staticmethod
    def test_get_weights_metadata_op_var_different_name():
        """
        For several rare corner cases, the const var and op have different names.
        Test that the API is correctly using the op's name.
        """
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 30, 10, 10)),
            ],
        )
        def prog(x):
            shared_weight = mb.const(
                val=np.random.rand(1, 30, 10, 10).astype(np.float32), name="w_1"
            )
            shared_weight.name = "w_1_new"
            x = mb.add(x=x, y=shared_weight, name="add_1")
            return x

        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT32,
        )

        ops_metadata_dict = ct.optimize.coreml.get_weights_metadata(
            mlmodel,
            weight_threshold=100,
        )
        assert "w_1" in ops_metadata_dict
        assert ops_metadata_dict["w_1"].child_ops[0].params_name_mapping["y"] == "w_1"
