# Copyright (c) 2022, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import unittest

import numpy as np
import pytest
import torch

import coremltools as ct
from coremltools._deps import _HAS_SKLEARN
from coremltools.converters.mil.testing_utils import get_op_types_in_program


def create_unique_weight(weight, nbits):
    shape = weight.detach().numpy().shape
    size = weight.detach().numpy().size

    unique_number = 1 << 4
    weight = []
    partition_len = size // unique_number + 1
    for i in range(unique_number):
        weight += [i] * (partition_len)
    weight = np.reshape(np.array(weight[:size]).astype(np.float32), shape)
    return weight

def get_test_model_and_data(multi_layer=False):
    inputs = [ct.TensorType(name="data", shape=(1, 64, 10, 10))]
    torch_input_values = [torch.rand(*i.shape.to_list()) for i in inputs]
    coreml_input_values = {
        i.name: val.detach().numpy() for i, val in zip(inputs, torch_input_values)
    }
    if multi_layer:
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv_1 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2)
                self.conv_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)

            def forward(self, x):
                conv_1 = self.conv_1(x)
                conv_2 = self.conv_2(conv_1)
                return conv_2

        model = Model().eval()
    else:
        model = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2)

    return model, inputs, torch_input_values, coreml_input_values

def verify_model_outputs(model, compressed_model, input_values):
    """
    This utility functions does the following checks:

    (1) Verify the output of the compressed model has the same shape / type of the original model
    (2) The decompressed and compressed model have the same numerical outputs
    """

    # Make sure the model can be decompressed
    decompressed_model = ct.compression_utils.decompress_weights(compressed_model)

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
        np.testing.assert_allclose(v, output_dict[k])

class TestCompressionUtils:

    @staticmethod
    def test_op_selector():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")
        mlmodel_no_quantized = ct.compression_utils.affine_quantize_weights(mlmodel, mode="linear", op_selector=lambda const_op: const_op.val.val.size > 1e7)
        expected_ops = ['cast', 'conv', 'cast']        
        assert get_op_types_in_program(mlmodel_no_quantized._mil_program) == expected_ops

    @staticmethod
    @unittest.skipIf(not _HAS_SKLEARN, "Missing scikit-learn. Skipping tests.")
    def test_weight_decompression():
        """
        This test is doing the following steps

        (1) compress a model with two conv layers into a compressed model with two different constexpr ops
            
            [Original model]:

                     weight_1      weight_2
                       |             |
                       v             v
            input -> conv_1 -----> conv_2 ---> output


            [Compressed model]:

                   weight_1_lut   weight_2_affine
                       |               |
                       v               v
            input -> conv_1 ------>  conv_2 ---> output

            , where weight_1_lut is a constexpr_lut_to_dense op and weight_2_affine is a constexpr_affine_dequantize op
        
        (2) decompress the compressed model

            [Decompressed model]:

                   weight_1_new   weight_2_new
                       |               |
                       v               v
            input -> conv_1 ------>  conv_2 ---> output

            , note that, weight_1_new is equivalent to weight_1_lut, and weight_2_new is equivalent to weight_2_affine
        """
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data(multi_layer=True)
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")
        
        # we first compress the model
        mlmodel = ct.compression_utils.palettize_weights(mlmodel, mode="kmeans", nbits=4, op_selector=lambda const_op: const_op.name == "conv_1_weight_to_fp16")
        mlmodel = ct.compression_utils.affine_quantize_weights(mlmodel, mode="linear", op_selector=lambda const_op: const_op.name == "conv_2_weight_to_fp16")
        expected_ops = ['constexpr_lut_to_dense', 'cast', 'conv', 'constexpr_affine_dequantize', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel._mil_program) == expected_ops

        # decompress the model
        decompressed_model = ct.compression_utils.decompress_weights(mlmodel)
        assert get_op_types_in_program(decompressed_model._mil_program) == ['cast', 'conv', 'conv', 'cast']

        if ct.utils._macos_version() < (13, 0):
            return

        # compared the numerical outputs
        output_dict = mlmodel.predict(coreml_input_values)
        de_output_dict = decompressed_model.predict(coreml_input_values)

        for k, v in output_dict.items():
            assert k in de_output_dict
            np.testing.assert_allclose(v, de_output_dict[k])

    @staticmethod
    def test_compression_utils_error_handling():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")

        # Test invalid mode for affine quantization
        expected_err_str = "supported for weight affine quantization. Got mode"
        with pytest.raises(ValueError, match=expected_err_str):
            ct.compression_utils.affine_quantize_weights(mlmodel, mode="invalid_mode")

        # Test invalid mode for weight sparsification
        expected_err_str = "supported for weight sparsification. Got mode"
        with pytest.raises(ValueError, match=expected_err_str):
            ct.compression_utils.sparsify_weights(mlmodel, mode="invalid_mode")

        # Test invalid threshold for weight sparsification
        expected_err_str = "Invalid value of threshold: \-1. Needs to be in \[0, inf\)"
        with pytest.raises(ValueError, match=expected_err_str):
            ct.compression_utils.sparsify_weights(mlmodel, mode="threshold_based", threshold=-1)

        # Test invalid percentile for weight sparsification
        expected_err_str = "Invalid value of target_percentile: 1.2. Needs to be in \[0, 1\]"
        with pytest.raises(ValueError, match=expected_err_str):
           ct.compression_utils.sparsify_weights(mlmodel, mode="percentile_based", target_percentile=1.2)

        # Test invalid mode for weight palettization
        expected_err_str = "supported for weight palettization. Got mode"
        with pytest.raises(ValueError, match=expected_err_str):
            ct.compression_utils.palettize_weights(mlmodel, mode="invalid_mode")

        # Test nbits must be provided for kmeans, uniform mode for weight palettization
        expected_err_str = "nbits must be provided for mode"
        with pytest.raises(ValueError, match=expected_err_str):
            ct.compression_utils.palettize_weights(mlmodel, mode="kmeans")

        with pytest.raises(ValueError, match=expected_err_str):
            ct.compression_utils.palettize_weights(mlmodel, mode="uniform")

        # Test nbits must not be provided for unique, custom mode for weight palettization
        expected_err_str = "nbits must NOT be provided for mode"
        with pytest.raises(ValueError, match=expected_err_str):
            ct.compression_utils.palettize_weights(mlmodel, mode="unique", nbits=2)

        with pytest.raises(ValueError, match=expected_err_str):
            ct.compression_utils.palettize_weights(mlmodel, mode="custom", nbits=2)

        # Test lut_function must be provided for custom mode, and must not be provided otherwise
        expected_err_str = "lut_function must be None if mode is not custom, and that it cannot be None when the mode is custom."
        with pytest.raises(ValueError, match=expected_err_str):
            ct.compression_utils.palettize_weights(mlmodel, mode="custom")
        with pytest.raises(ValueError, match=expected_err_str):
            ct.compression_utils.palettize_weights(mlmodel, mode="unique", lut_function=lambda op: True)

        # Test lut_function must be a function obejct
        expected_err_str = "A function object must be provided as lut_function"
        with pytest.raises(ValueError, match=expected_err_str):
            ct.compression_utils.palettize_weights(mlmodel, mode="custom", lut_function=1)


    @staticmethod
    @pytest.mark.parametrize(
        "mode",
        ("linear", "linear_symmetric"),
    )
    def test_linear_quanitzation(mode):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")

        mlmodel_quantized = ct.compression_utils.affine_quantize_weights(mlmodel, mode=mode)

        # validate parameters
        expected_ops = ['constexpr_affine_dequantize', 'cast', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_quantized._mil_program) == expected_ops

        quanitze_op = mlmodel_quantized._mil_program.functions["main"].find_ops(op_type="constexpr_affine_dequantize")[0]
        assert model.weight.detach().numpy().shape == quanitze_op.quantized_data.shape

        verify_model_outputs(mlmodel, mlmodel_quantized, coreml_input_values)

    @staticmethod
    @pytest.mark.parametrize(
        "threshold",
        (0.0, 0.001, 1e2),
    )
    def test_weight_sparsify_threshold_based(threshold):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        with torch.no_grad():
            model.weight[0][0][0][0] = 101
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")
        mlmodel_sparsified = ct.compression_utils.sparsify_weights(mlmodel, mode="threshold_based", threshold=threshold)

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
    def test_weight_sparsify_percentile_based(percentile):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")
        mlmodel_sparsified = ct.compression_utils.sparsify_weights(mlmodel, mode="percentile_based", target_percentile=percentile)

        # validate parameters        
        expected_ops = ['constexpr_sparse_to_dense', 'cast', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_sparsified._mil_program) == expected_ops

        main_func = mlmodel_sparsified._mil_program.functions["main"]
        sparse_to_dense_op = main_func.find_ops(op_type="constexpr_sparse_to_dense")[0]
        non_sparse_data = sparse_to_dense_op.nonzero_data
        weight = model.weight.detach().numpy()

        if percentile == 0.:
            assert non_sparse_data.val.size == weight.size - 1
        elif percentile == 0.5:
            assert non_sparse_data.val.size <= 0.51 * (weight.size) and non_sparse_data.val.size >= 0.49 * (weight.size)
        else:
            assert non_sparse_data.val.size == 0

        assert sparse_to_dense_op.shape.val.tolist() == list(model.weight.detach().numpy().shape)

        # validate the model
        verify_model_outputs(mlmodel, mlmodel_sparsified, coreml_input_values)

    @staticmethod
    @pytest.mark.parametrize(
        "mode",
        ("uniform", "kmeans") if _HAS_SKLEARN else ("uniform",)
    )
    def test_weight_palettization(mode):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")
        mlmodel_palettized = ct.compression_utils.palettize_weights(mlmodel, nbits=4, mode=mode)

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
        mlmodel_palettized = ct.compression_utils.palettize_weights(mlmodel, mode="unique")
        expected_ops = ['constexpr_lut_to_dense', 'cast', 'conv', 'constexpr_lut_to_dense', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_palettized._mil_program) == expected_ops

        main_func = mlmodel_palettized._mil_program.functions["main"]
        lut_to_dense_op_1 = main_func.find_ops(op_type="constexpr_lut_to_dense")[0]
        lut_to_dense_op_2 = main_func.find_ops(op_type="constexpr_lut_to_dense")[1]

        assert lut_to_dense_op_1.shape.val.tolist() == list(model.conv_1.weight.detach().numpy().shape)
        assert lut_to_dense_op_2.shape.val.tolist() == list(model.conv_2.weight.detach().numpy().shape)

        # validate the model 
        verify_model_outputs(mlmodel, mlmodel_palettized, coreml_input_values)

    @staticmethod
    def test_weight_palettization_unique_case_2(caplog):
        # In this model, only one conv weights can be palettized, the converter should warn the users that one weight is skipped
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data(multi_layer=True)

        weight_1_unique = create_unique_weight(model.conv_1.weight, nbits=2)
        
        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(torch.Tensor(weight_1_unique))

        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")

        # validate parameters
        # converter should warn the user that one weight is not compressed
        mlmodel_palettized = ct.compression_utils.palettize_weights(mlmodel, mode="unique")
        warning_msg = "weight value cannot be represented in an 8 bits palettization. Skipped."
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

        mlmodel_palettized = ct.compression_utils.palettize_weights(mlmodel, mode="custom", lut_function=lut_function)

        # validate parameters
        expected_ops = ['constexpr_lut_to_dense', 'cast', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_palettized._mil_program) == expected_ops

        main_func = mlmodel_palettized._mil_program.functions["main"]
        lut_to_dense_op = main_func.find_ops(op_type="constexpr_lut_to_dense")[0]

        assert lut_to_dense_op.shape.val.tolist() == list(model.weight.detach().numpy().shape)
    
        # validate the model 
        verify_model_outputs(mlmodel, mlmodel_palettized, coreml_input_values)
