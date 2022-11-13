# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Module containing unit tests for verifying various quantizations.
"""

import unittest

import numpy as np
import pytest

import coremltools
import coremltools.models.datatypes as datatypes
from coremltools import ComputeUnit
from coremltools.models import (_QUANTIZATION_MODE_LINEAR_QUANTIZATION,
                                neural_network)
from coremltools.models.neural_network import quantization_utils
from coremltools.models.neural_network.quantization_utils import (
    MatrixMultiplyLayerSelector, _quantize_spec_weights,
    activate_int8_int8_matrix_multiplications)


@unittest.skipIf(
    not coremltools.utils._is_macos() or coremltools.utils._macos_version() < (10, 16),
    "Missing macOS 10.16+. Skipping tests.",
)
class DynamicQuantizedInt8Int8MatMul(unittest.TestCase):
    """
    Quantization tests for dynamic Int8 - Int8 matrix multiplications
    """

    def initialize(self):
        np.random.seed(1988)
        self.Cout, self.Cin = 16, 32
        self.W = np.random.rand(self.Cout, self.Cin) * 20.0 - 10.0
        self.b = np.random.rand(self.Cout) * 20.0 - 10.0
        self.input_shape = (5, self.Cin)
        input_features = [("data", datatypes.Array(*self.input_shape))]
        output_features = [("output", None)]
        self.builder = neural_network.NeuralNetworkBuilder(
            input_features, output_features, disable_rank5_shape_mapping=True
        )
        self.selector = MatrixMultiplyLayerSelector()

    def _test_predictions(
        self, np_preds, coreml_preds, SNR=30, PSNR=40,
    ):

        np_preds = np_preds.flatten()
        coreml_preds = coreml_preds.flatten()

        noise = np_preds - coreml_preds
        noise_var = np.sum(noise ** 2) / len(noise) + 1e-7
        signal_energy = np.sum(np_preds ** 2) / len(np_preds)
        max_signal_energy = np.amax(np_preds ** 2)
        snr = 10 * np.log10(signal_energy / noise_var)
        psnr = 10 * np.log10(max_signal_energy / noise_var)
        self.assertGreaterEqual(snr, SNR)
        self.assertGreaterEqual(psnr, PSNR)

    def compare(self, specification_modified=True):
        x = np.random.rand(*self.input_shape)

        def _get_preds(spec):
            mlmodel = coremltools.models.MLModel(spec, compute_units=ComputeUnit.CPU_ONLY)
            return mlmodel.predict({"data": x})["output"]

        preds = _get_preds(self.builder.spec)
        self.assertEqual(self.builder.spec.specificationVersion, 4)

        quantized_spec = activate_int8_int8_matrix_multiplications(
            self.builder.spec, self.selector
        )

        layer = self.builder.spec.neuralNetwork.layers[0]
        layer_type = layer.WhichOneof("layer")
        if layer_type == "innerProduct":
            matmul_layer = layer.innerProduct

        elif layer_type == "batchedMatmul":
            matmul_layer = layer.batchedMatmul
        wp = matmul_layer.weights

        if specification_modified:
            self.assertEqual(self.builder.spec.specificationVersion, 5)
            quant_preds = _get_preds(quantized_spec)
            self._test_predictions(preds, quant_preds, SNR=40)
            self.assertEqual(len(wp.floatValue), 0)
        else:
            self.assertEqual(self.builder.spec.specificationVersion, 4)
            quant_preds = _get_preds(quantized_spec)
            np.testing.assert_array_almost_equal(preds, quant_preds)
            self.assertGreater(len(wp.floatValue), 0)

    def test_single_batched_matmul_no_bias(self):

        self.initialize()
        self.builder.add_batched_mat_mul(
            name="batched_matmul",
            input_names=["data"],
            output_name="output",
            weight_matrix_rows=self.Cin,
            weight_matrix_columns=self.Cout,
            W=self.W,
        )
        self.compare()

    def test_single_batched_matmul_with_bias(self):

        self.initialize()
        self.builder.add_batched_mat_mul(
            name="batched_matmul",
            input_names=["data"],
            output_name="output",
            weight_matrix_rows=self.Cin,
            weight_matrix_columns=self.Cout,
            W=self.W,
            bias=self.b,
        )
        self.compare()

    def test_single_inner_product_no_bias(self):

        self.initialize()
        self.builder.add_inner_product(
            name="ip",
            input_name="data",
            output_name="output",
            input_channels=self.Cin,
            output_channels=self.Cout,
            W=self.W,
            b=None,
            has_bias=False,
        )
        self.compare()

    def test_single_inner_product_with_bias(self):

        self.initialize()
        self.builder.add_inner_product(
            name="ip",
            input_name="data",
            output_name="output",
            input_channels=self.Cin,
            output_channels=self.Cout,
            W=self.W,
            b=self.b,
            has_bias=True,
        )
        self.compare()

    def test_inner_product_min_input_channels_valid(self):
        self.initialize()
        self.builder.add_inner_product(
            name="ip",
            input_name="data",
            output_name="output",
            input_channels=self.Cin,
            output_channels=self.Cout,
            W=self.W,
            b=self.b,
            has_bias=True,
        )
        self.selector.minimum_input_channels = 31
        self.compare()

    def test_batched_matmul_min_input_channels_valid(self):

        self.initialize()
        self.builder.add_batched_mat_mul(
            name="batched_matmul",
            input_names=["data"],
            output_name="output",
            weight_matrix_rows=self.Cin,
            weight_matrix_columns=self.Cout,
            W=self.W,
        )
        self.selector.minimum_input_channels = 32
        self.compare()

    def test_inner_product_min_input_channels_invalid(self):
        self.initialize()
        self.builder.add_inner_product(
            name="ip",
            input_name="data",
            output_name="output",
            input_channels=self.Cin,
            output_channels=self.Cout,
            W=self.W,
            b=self.b,
            has_bias=True,
        )
        self.selector.minimum_input_channels = 33
        self.compare(specification_modified=False)

    def test_batched_matmul_min_input_channels_invalid(self):

        self.initialize()
        self.builder.add_batched_mat_mul(
            name="batched_matmul",
            input_names=["data"],
            output_name="output",
            weight_matrix_rows=self.Cin,
            weight_matrix_columns=self.Cout,
            W=self.W,
        )
        self.selector.minimum_input_channels = 33
        self.compare(specification_modified=False)

    def test_batched_matmul_max_input_channels_valid(self):

        self.initialize()
        self.builder.add_batched_mat_mul(
            name="batched_matmul",
            input_names=["data"],
            output_name="output",
            weight_matrix_rows=self.Cin,
            weight_matrix_columns=self.Cout,
            W=self.W,
        )
        self.selector.maximum_input_channels = 32
        self.compare()

    def test_inner_product_max_input_channels_valid(self):
        self.initialize()
        self.builder.add_inner_product(
            name="ip",
            input_name="data",
            output_name="output",
            input_channels=self.Cin,
            output_channels=self.Cout,
            W=self.W,
            b=self.b,
            has_bias=True,
        )
        self.selector.maximum_input_channels = 33
        self.compare()

    def test_batched_matmul_max_input_channels_invalid(self):

        self.initialize()
        self.builder.add_batched_mat_mul(
            name="batched_matmul",
            input_names=["data"],
            output_name="output",
            weight_matrix_rows=self.Cin,
            weight_matrix_columns=self.Cout,
            W=self.W,
        )
        self.selector.maximum_input_channels = 31
        self.compare(specification_modified=False)

    def test_inner_product_max_input_channels_invalid(self):
        self.initialize()
        self.builder.add_inner_product(
            name="ip",
            input_name="data",
            output_name="output",
            input_channels=self.Cin,
            output_channels=self.Cout,
            W=self.W,
            b=self.b,
            has_bias=True,
        )
        self.selector.maximum_input_channels = 30
        self.compare(specification_modified=False)

    def test_inner_product_min_output_channels_valid(self):
        self.initialize()
        self.builder.add_inner_product(
            name="ip",
            input_name="data",
            output_name="output",
            input_channels=self.Cin,
            output_channels=self.Cout,
            W=self.W,
            b=self.b,
            has_bias=True,
        )
        self.selector.minimum_output_channels = 16
        self.compare()

    def test_batched_matmul_min_output_channels_valid(self):

        self.initialize()
        self.builder.add_batched_mat_mul(
            name="batched_matmul",
            input_names=["data"],
            output_name="output",
            weight_matrix_rows=self.Cin,
            weight_matrix_columns=self.Cout,
            W=self.W,
        )
        self.selector.minimum_output_channels = 16
        self.compare()

    def test_inner_product_min_output_channels_invalid(self):
        self.initialize()
        self.builder.add_inner_product(
            name="ip",
            input_name="data",
            output_name="output",
            input_channels=self.Cin,
            output_channels=self.Cout,
            W=self.W,
            b=self.b,
            has_bias=True,
        )
        self.selector.minimum_output_channels = 17
        self.compare(specification_modified=False)

    def test_batched_matmul_min_output_channels_invalid(self):

        self.initialize()
        self.builder.add_batched_mat_mul(
            name="batched_matmul",
            input_names=["data"],
            output_name="output",
            weight_matrix_rows=self.Cin,
            weight_matrix_columns=self.Cout,
            W=self.W,
        )
        self.selector.minimum_output_channels = 17
        self.compare(specification_modified=False)

    def test_batched_matmul_max_output_channels_valid(self):

        self.initialize()
        self.builder.add_batched_mat_mul(
            name="batched_matmul",
            input_names=["data"],
            output_name="output",
            weight_matrix_rows=self.Cin,
            weight_matrix_columns=self.Cout,
            W=self.W,
        )
        self.selector.maximum_output_channels = 17
        self.compare()

    def test_inner_product_max_output_channels_valid(self):
        self.initialize()
        self.builder.add_inner_product(
            name="ip",
            input_name="data",
            output_name="output",
            input_channels=self.Cin,
            output_channels=self.Cout,
            W=self.W,
            b=self.b,
            has_bias=True,
        )
        self.selector.maximum_output_channels = 16
        self.compare()

    def test_batched_matmul_max_output_channels_invalid(self):

        self.initialize()
        self.builder.add_batched_mat_mul(
            name="batched_matmul",
            input_names=["data"],
            output_name="output",
            weight_matrix_rows=self.Cin,
            weight_matrix_columns=self.Cout,
            W=self.W,
        )
        self.selector.maximum_output_channels = 14
        self.compare(specification_modified=False)

    def test_inner_product_max_output_channels_invalid(self):
        self.initialize()
        self.builder.add_inner_product(
            name="ip",
            input_name="data",
            output_name="output",
            input_channels=self.Cin,
            output_channels=self.Cout,
            W=self.W,
            b=self.b,
            has_bias=True,
        )
        self.selector.maximum_output_channels = 15
        self.compare(specification_modified=False)

    def test_inner_product_min_weight_count_valid(self):
        self.initialize()
        self.builder.add_inner_product(
            name="ip",
            input_name="data",
            output_name="output",
            input_channels=self.Cin,
            output_channels=self.Cout,
            W=self.W,
            b=self.b,
            has_bias=True,
        )
        self.selector.minimum_weight_count = 512
        self.compare()

    def test_batched_matmul_min_weight_count_invalid(self):

        self.initialize()
        self.builder.add_batched_mat_mul(
            name="batched_matmul",
            input_names=["data"],
            output_name="output",
            weight_matrix_rows=self.Cin,
            weight_matrix_columns=self.Cout,
            W=self.W,
        )
        self.selector.minimum_weight_count = 513
        self.compare(specification_modified=False)

    def test_inner_product_layer_names_invalid(self):
        self.initialize()
        self.builder.add_inner_product(
            name="ip",
            input_name="data",
            output_name="output",
            input_channels=self.Cin,
            output_channels=self.Cout,
            W=self.W,
            b=self.b,
            has_bias=True,
        )
        self.selector.include_layers_with_names = ["ip1", "ip2"]
        self.compare(specification_modified=False)

    def test_batched_matmul_layer_names_valid(self):

        self.initialize()
        self.builder.add_batched_mat_mul(
            name="batched_matmul",
            input_names=["data"],
            output_name="output",
            weight_matrix_rows=self.Cin,
            weight_matrix_columns=self.Cout,
            W=self.W,
        )
        self.selector.include_layers_with_names = ["bm1", "batched_matmul"]
        self.compare()

    def test_batched_matmul_8bit_weight_quantized(self):

        self.initialize()
        self.builder.add_batched_mat_mul(
            name="batched_matmul",
            input_names=["data"],
            output_name="output",
            weight_matrix_rows=self.Cin,
            weight_matrix_columns=self.Cout,
            W=self.W,
        )
        _quantize_spec_weights(
            self.builder.spec, 8, _QUANTIZATION_MODE_LINEAR_QUANTIZATION
        )
        self.compare()

    def test_batched_matmul_4bit_weight_quantized(self):

        self.initialize()
        self.builder.add_batched_mat_mul(
            name="batched_matmul",
            input_names=["data"],
            output_name="output",
            weight_matrix_rows=self.Cin,
            weight_matrix_columns=self.Cout,
            W=self.W,
        )
        _quantize_spec_weights(
            self.builder.spec, 4, _QUANTIZATION_MODE_LINEAR_QUANTIZATION
        )
        self.compare()

    def test_batched_matmul_2bit_weight_quantized(self):

        self.initialize()
        self.builder.add_batched_mat_mul(
            name="batched_matmul",
            input_names=["data"],
            output_name="output",
            weight_matrix_rows=self.Cin,
            weight_matrix_columns=self.Cout,
            W=self.W,
        )
        _quantize_spec_weights(
            self.builder.spec, 2, _QUANTIZATION_MODE_LINEAR_QUANTIZATION
        )
        self.compare()

    def test_batched_matmul_1bit_weight_quantized(self):

        self.initialize()
        self.builder.add_batched_mat_mul(
            name="batched_matmul",
            input_names=["data"],
            output_name="output",
            weight_matrix_rows=self.Cin,
            weight_matrix_columns=self.Cout,
            W=self.W,
        )
        _quantize_spec_weights(
            self.builder.spec, 1, _QUANTIZATION_MODE_LINEAR_QUANTIZATION
        )
        self.compare()


class TestQuantizeWeightsAPI:
    @staticmethod
    @pytest.mark.parametrize(
        "compute_units", [ComputeUnit.ALL, ComputeUnit.CPU_AND_GPU, ComputeUnit.CPU_ONLY]
    )
    def test_embeddingND_quantize(compute_units):
        input_features = [("data", datatypes.Array(10, 1))]
        output_features = [("output", None)]
        builder = neural_network.NeuralNetworkBuilder(
            input_features, output_features, disable_rank5_shape_mapping=True
        )

        builder.add_embedding_nd(
            name="embedding_nd",
            input_name="data",
            output_name="output",
            vocab_size=300,
            embedding_size=20,
            W=np.random.rand(20, 300),
        )

        spec = builder.spec
        model_fp32 = coremltools.models.MLModel(spec, compute_units=compute_units)
        assert len(spec.neuralNetwork.layers[0].embeddingND.weights.floatValue) == 6000

        # quantize to FP16
        model_fp16 = quantization_utils.quantize_weights(model_fp32, nbits=16)
        assert model_fp16.compute_unit == compute_units
        spec_fp16 = model_fp16.get_spec()
        assert len(spec_fp16.neuralNetwork.layers[0].embeddingND.weights.floatValue) == 0
        assert len(spec_fp16.neuralNetwork.layers[0].embeddingND.weights.float16Value) == 2 * 6000

        # quantize to uint8
        model_uint8 = quantization_utils.quantize_weights(model_fp32, nbits=8)
        assert model_uint8.compute_unit == compute_units
        spec_uint8 = model_uint8.get_spec()
        assert len(spec_uint8.neuralNetwork.layers[0].embeddingND.weights.floatValue) == 0
        assert len(spec_uint8.neuralNetwork.layers[0].embeddingND.weights.float16Value) == 0
        assert len(spec_uint8.neuralNetwork.layers[0].embeddingND.weights.rawValue) == 6000

        # quantize to uint5
        model_uint5 = quantization_utils.quantize_weights(model_fp32, nbits=5)
        assert model_uint5.compute_unit == compute_units
        spec_uint5 = model_uint5.get_spec()
        assert len(spec_uint5.neuralNetwork.layers[0].embeddingND.weights.floatValue) == 0
        assert len(spec_uint5.neuralNetwork.layers[0].embeddingND.weights.float16Value) == 0
        assert len(spec_uint5.neuralNetwork.layers[0].embeddingND.weights.rawValue) == 3750  # 3750 = 5*6000/8

    @unittest.skipIf(coremltools.utils._macos_version() < (13, 0),
                     'ComputeUnit.CPU_AND_NE is only available on macOS >= 13.0'
    )
    def test_embeddingND_quantize_CPU_and_NE(self):
        self.test_embeddingND_quantize(ComputeUnit.CPU_AND_NE)
