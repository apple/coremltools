# Copyright (c) 2023, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest
import torch

import coremltools as ct
import coremltools.optimize as cto
from coremltools._deps import _HAS_SKLEARN
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.testing_utils import get_op_types_in_program
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
            return lstm

    return Model().eval(), inputs, torch_input_values, coreml_input_values


def create_unique_weight(weight, nbits):
    shape = weight.detach().numpy().shape
    size = weight.detach().numpy().size

    unique_number = 1 << nbits
    weight = list(range(unique_number))
    if size > unique_number:
        weight.extend([unique_number - 1] * (size - unique_number))
    weight = np.reshape(np.array(weight[:size]).astype(np.float32), shape)
    return weight


def create_sparse_weight(weight, target_sparsity):
    shape = list(weight.shape)
    size = np.prod(shape)
    weight = 100 * np.random.rand(size)
    num_of_zeros = int(size * target_sparsity)
    weight[:num_of_zeros] = 0
    return np.reshape(weight, shape).astype(np.float32)


def verify_model_outputs(model, compressed_model, input_values, rtol=1e-7, atol=0):
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
            "expand_dims",
            "expand_dims"
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


class TestPalettizeWeights:
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
            "expand_dims",
            "expand_dims"
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
            "expand_dims",
            "expand_dims",
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
            "expand_dims",
            "expand_dims",
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


class TestPruneWeights:
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
            "expand_dims",
            "expand_dims"
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
            assert non_sparse_data.val.size <= 0.51 * (weight.size) and non_sparse_data.val.size >= 0.49 * (weight.size)
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
            "expand_dims",
            "expand_dims"
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
            "expand_dims",
            "expand_dims"
        ]
        assert get_op_types_in_program(prog) == expected_ops

        conv_ops = prog.find_ops(op_type="conv")
        assert conv_ops[0].weight.op.op_type == "const"
        assert conv_ops[1].weight.op.op_type == "const"

        linear_ops = prog.find_ops(op_type="linear")
        assert linear_ops[0].weight.op.op_type == "constexpr_sparse_to_dense"
        assert linear_ops[1].weight.op.op_type == "const"

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
            "expand_dims",
            "expand_dims"
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
        expected_err_str = "is unsupported for affine_quantize_weight"
        with pytest.raises(ValueError, match=expected_err_str):
            linear_quantize_weights(mlmodel, dtype=np.int32)

        expected_err_str = "\'dtype\' must be \<class \'type\'\> \(got \'int32\'"
        with pytest.raises(TypeError, match=expected_err_str):
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
