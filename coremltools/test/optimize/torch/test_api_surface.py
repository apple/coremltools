#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import List

import coremltools.optimize.torch


def _get_visible_items(d):
    return [x for x in dir(d) if not x.startswith("_")]


def _check_visible_modules(actual: List[str], expected: List[str]):
    assert set(actual) == set(expected), "API mis-matched. Got %s, expected %s" % (
        actual,
        expected,
    )


class TestApiVisibilities:
    """Test APIs visible to users"""

    def test_top_level(self):
        # coremltools.optimize.torch.*
        expected = [
            "base_model_optimizer",
            "optimization_config",
            "palettization",
            "pruning",
            "quantization",
            "layerwise_compression",
        ]
        visible_modules = _get_visible_items(coremltools.optimize.torch)
        _check_visible_modules(visible_modules, expected)

    def test_base_model_optimizer_module(self):
        # coremltools.optimize.torch.base_model_optimizer.*
        expected = [
            "BaseModelOptimizer",
            "BaseTrainingTimeModelOptimizer",
            "BasePostTrainingModelOptimizer",
            "BaseDataCalibratedModelOptimizer",
        ]
        visible_modules = _get_visible_items(coremltools.optimize.torch.base_model_optimizer)
        _check_visible_modules(visible_modules, expected)

    def test_optimization_config_module(self):
        # coremltools.optimize.torch.optimization_config.*
        expected = [
            "PalettizationGranularity",
            "QuantizationGranularity",
            "ModuleOptimizationConfig",
            "OptimizationConfig",
        ]
        visible_modules = _get_visible_items(coremltools.optimize.torch.optimization_config)
        _check_visible_modules(visible_modules, expected)

    def test_palettization_module(self):
        # coremltools.optimize.torch.palettization.*
        expected = [
            "FakePalettize",
            "DKMPalettizer",
            "DKMPalettizerConfig",
            "ModuleDKMPalettizerConfig",
            "palettization_config",
            "fake_palettize",
            "palettizer",
            "post_training_palettization",
            "ModulePostTrainingPalettizerConfig",
            "PostTrainingPalettizer",
            "PostTrainingPalettizerConfig",
            "sensitive_k_means",
            "ModuleSKMPalettizerConfig",
            "SKMPalettizer",
            "SKMPalettizerConfig",
        ]
        visible_modules = _get_visible_items(coremltools.optimize.torch.palettization)
        _check_visible_modules(visible_modules, expected)
        # coremltools.optimize.torch.palettization.palettizer.*
        expected = [
            "Palettizer",
            "DKMPalettizer",
        ]
        visible_modules = _get_visible_items(coremltools.optimize.torch.palettization.palettizer)
        _check_visible_modules(visible_modules, expected)

    def test_pruning_module(self):
        # coremltools.optimize.torch.pruning.*
        expected = [
            "ConstantSparsityScheduler",
            "MagnitudePruner",
            "MagnitudePrunerConfig",
            "ModuleMagnitudePrunerConfig",
            "PolynomialDecayScheduler",
            "magnitude_pruner",
            "pruning_scheduler",
        ]
        visible_modules = _get_visible_items(coremltools.optimize.torch.pruning)
        _check_visible_modules(visible_modules, expected)

    def test_quantization_module(self):
        # coremltools.optimize.torch.quantization.*
        expected = [
            "LinearQuantizer",
            "LinearQuantizerConfig",
            "ModuleLinearQuantizerConfig",
            "ObserverType",
            "QuantizationScheme",
            "quantizer",
            "quantization_config",
            "modules",
            "ModulePostTrainingQuantizerConfig",
            "PostTrainingQuantizer",
            "PostTrainingQuantizerConfig",
            "post_training_quantization",
        ]
        visible_modules = _get_visible_items(coremltools.optimize.torch.quantization)
        _check_visible_modules(visible_modules, expected)
        # coremltools.optimize.torch.quantization.LinearQuantizer.*
        expected = [
            "finalize",
            "prepare",
            "step",
            "report",
            "supported_modules",
        ]
        visible_modules = _get_visible_items(
            coremltools.optimize.torch.quantization.LinearQuantizer
        )
        _check_visible_modules(visible_modules, expected)
        # coremltools.optimize.torch.quantization.quantizer.*
        expected = [
            "Quantizer",
            "LinearQuantizer",
        ]
        visible_modules = _get_visible_items(coremltools.optimize.torch.quantization.quantizer)
        _check_visible_modules(visible_modules, expected)

    def test_layerwise_compression_module(self):
        expected = [
            "algorithms",
            "LayerwiseCompressionAlgorithm",
            "LayerwiseCompressionAlgorithmConfig",
            "SparseGPT",
            "GPTQ",
            "ModuleGPTQConfig",
            "ModuleSparseGPTConfig",
            "input_cacher",
            "FirstLayerInputCacher",
            "DefaultInputCacher",
            "GPTFirstLayerInputCacher",
            "layerwise_compressor",
            "LayerwiseCompressor",
            "LayerwiseCompressorConfig",
        ]
        visible_modules = _get_visible_items(coremltools.optimize.torch.layerwise_compression)
        _check_visible_modules(visible_modules, expected)
