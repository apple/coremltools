# Copyright (c) 2022, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Optional

import numpy as np
import torch

import coremltools as ct
from coremltools.converters.mil.testing_utils import get_op_types_in_program
from coremltools.models.ml_program.compression_utils import (
    affine_quantize_weights,
    decompress_weights,
    palettize_weights,
    sparsify_weights,
)
from coremltools.optimize.coreml._config import OpCompressorConfig


def get_test_model_and_data(
    multi_layer: bool = False,
    quantize_config: Optional[OpCompressorConfig] = None,
    use_linear: bool = False,
):
    """
    Prepare test model and data.

    :param multi_layer: If set, the test model will have multiple `nn.Conv2d` layers.
    :param quantize_config: If set, the weights in the test model will be nbits quantization-friendly,
        which means it will be first quantized according to the config, and then dequantized, so the
        numerical error introduced during the quantization test will be minimum.
    :param use_linear: If set, use linear instead of conv in the model.
    """
    if quantize_config is not None and multi_layer:
        raise AssertionError("Multi-layer model doesn't support pre_quantize_nbits.")

    inputs = [ct.TensorType(name="data", shape=(1, 64, 10, 10))]
    if use_linear:
        inputs = [ct.TensorType(name="data", shape=(1, 64))]

    torch_input_values = [torch.rand(*i.shape.to_list()) for i in inputs]
    coreml_input_values = {
        i.name: val.detach().numpy() for i, val in zip(inputs, torch_input_values)
    }
    if multi_layer:

        class ConvModel(torch.nn.Module):
            def __init__(self):
                super(ConvModel, self).__init__()
                self.conv_1 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2)
                self.conv_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)

            def forward(self, x):
                conv_1 = self.conv_1(x)
                conv_2 = self.conv_2(conv_1)
                return conv_2

        class LinearModel(torch.nn.Module):
            def __init__(self):
                super(LinearModel, self).__init__()
                self.linear_1 = torch.nn.Linear(in_features=64, out_features=32, bias=False)
                self.linear_2 = torch.nn.Linear(in_features=32, out_features=16, bias=False)

            def forward(self, x):
                linear_1 = self.linear_1(x)
                return self.linear_2(linear_1)

        model = LinearModel().eval() if use_linear else ConvModel().eval()
    else:
        model = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2)
        if use_linear:
            model = torch.nn.Linear(in_features=64, out_features=32, bias=False)

        if quantize_config is not None:
            # Manually change weight to make it quantization friendly.
            nbits_range_max = 2 ** (quantize_config.nbits - 1) - 1
            mode_to_range = {
                "LINEAR": (-nbits_range_max - 1, nbits_range_max),
                "LINEAR_SYMMETRIC": (-nbits_range_max, nbits_range_max),
            }
            q_val_min, q_val_max = mode_to_range[quantize_config.mode]
            original_shape = model.weight.detach().numpy().shape
            fake_scale = 2.0
            quantize_friendly_weight = (
                np.random.randint(low=q_val_min, high=q_val_max + 1, size=original_shape)
                * fake_scale
            )
            with torch.no_grad():
                model.weight = torch.nn.Parameter(
                    torch.from_numpy(quantize_friendly_weight).float()
                )
        model = model.eval()

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
