# Copyright (c) 2022, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import torch

import coremltools as ct
from coremltools.models.ml_program.compression_utils import (
    affine_quantize_weights,
    decompress_weights,
    palettize_weights,
    sparsify_weights,
)
from coremltools.converters.mil.testing_utils import get_op_types_in_program


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


class TestCompressionUtils:
    """
    Since ct.compression_utils is deprecated, this test is only checking the API is still working.
    """
    @staticmethod
    def test_op_selector():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")
        mlmodel_no_quantized = affine_quantize_weights(mlmodel, mode="linear", op_selector=lambda const_op: const_op.val.val.size > 1e7)
        expected_ops = ['cast', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_no_quantized._mil_program) == expected_ops

    @staticmethod
    def test_affine_quantize_weights_smoke():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")
        mlmodel_quantized = affine_quantize_weights(mlmodel, mode="linear_symmetric", dtype=np.int8)

        # validate parameters
        expected_ops = ['constexpr_affine_dequantize', 'cast', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_quantized._mil_program) == expected_ops

    @staticmethod
    def test_palettize_weights_smoke():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")
        mlmodel_palettized = palettize_weights(mlmodel, nbits=4, mode="uniform")

        # validate parameters
        expected_ops = ['constexpr_lut_to_dense', 'cast', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_palettized._mil_program) == expected_ops

    @staticmethod
    def test_sparsify_weights_threshold_smoke():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        with torch.no_grad():
            model.weight[0][0][0][0] = 101
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")
        mlmodel_sparsified = sparsify_weights(mlmodel, mode="threshold_based", threshold=0.01)

        # validate parameters
        expected_ops = ['constexpr_sparse_to_dense', 'cast', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_sparsified._mil_program) == expected_ops

    @staticmethod
    def test_sparsify_weights_percentile_smoke():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        with torch.no_grad():
            model.weight[0][0][0][0] = 101
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")
        mlmodel_sparsified = sparsify_weights(mlmodel, mode="percentile_based", target_percentile=0.8)

        # validate parameters
        expected_ops = ['constexpr_sparse_to_dense', 'cast', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel_sparsified._mil_program) == expected_ops

    @staticmethod
    def test_weight_decompression_smoke():
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data(multi_layer=True)
        torchmodel = torch.jit.trace(model, torch_input_values)
        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram")

        # we first compress the model
        mlmodel = palettize_weights(mlmodel, mode="kmeans", nbits=4, op_selector=lambda const_op: const_op.name == "conv_1_weight_to_fp16")
        mlmodel = affine_quantize_weights(mlmodel, mode="linear", op_selector=lambda const_op: const_op.name == "conv_2_weight_to_fp16")
        expected_ops = ['constexpr_lut_to_dense', 'cast', 'conv', 'constexpr_affine_dequantize', 'conv', 'cast']
        assert get_op_types_in_program(mlmodel._mil_program) == expected_ops

        # decompress the model
        decompressed_model = decompress_weights(mlmodel)
        assert get_op_types_in_program(decompressed_model._mil_program) == ['cast', 'conv', 'conv', 'cast']
