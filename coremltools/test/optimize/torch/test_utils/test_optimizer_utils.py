#  Copyright (c) 2025, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch.nn as nn

import coremltools.optimize.torch
from coremltools.optimize.torch._utils.optimizer_utils import (
    _ConfigToOptimizerRegistry,
    _ModuleToOptConfigRegistry,
    is_supported_module_for_config,
)
from coremltools.optimize.torch.layerwise_compression import LayerwiseCompressor
from coremltools.optimize.torch.layerwise_compression.algorithms import ModuleGPTQConfig
from coremltools.optimize.torch.optimization_config import (
    ModuleOptimizationConfig,
    OptimizationConfig,
)
from coremltools.optimize.torch.palettization import (
    ModulePostTrainingPalettizerConfig,
    PostTrainingPalettizer,
)
from coremltools.optimize.torch.palettization.sensitive_k_means import (
    ModuleSKMPalettizerConfig,
    SKMPalettizer,
)
from coremltools.test.optimize.torch.utils import get_classes_recursively


@pytest.mark.parametrize(
    "module_config_cls, optimizer",
    [
        (ModulePostTrainingPalettizerConfig, PostTrainingPalettizer),
        (ModuleSKMPalettizerConfig, SKMPalettizer),
        (ModuleGPTQConfig, LayerwiseCompressor),
    ],
)
def test_optimizer_config_registry(module_config_cls, optimizer):
    opt_cfg_cls = _ModuleToOptConfigRegistry.get_opt_cfg(module_config_cls)
    opt_cls = _ConfigToOptimizerRegistry.get_optimizer(opt_cfg_cls)
    assert opt_cls == optimizer


@pytest.mark.parametrize(
    "module, config_cls, is_supported",
    [
        (nn.Conv2d(32, 16, 3), ModulePostTrainingPalettizerConfig, True),
        (nn.ReLU(), ModulePostTrainingPalettizerConfig, False),
    ],
)
def test_is_supported_config(module, config_cls, is_supported):
    assert is_supported_module_for_config(module, config_cls) == is_supported


def test_optimization_config_registry():
    skipped_configs = [ModuleOptimizationConfig]
    all_coremltools_optimize_torch_classes = get_classes_recursively(coremltools.optimize.torch)
    module_config_classes = [
        c
        for n, c in all_coremltools_optimize_torch_classes.items()
        if issubclass(c, coremltools.optimize.torch.optimization_config.ModuleOptimizationConfig)
        and c not in skipped_configs
    ]

    unregistered_config_classes = []
    for module_cfg_cls in module_config_classes:
        try:
            opt_cfg_cls = _ModuleToOptConfigRegistry.get_opt_cfg(module_cfg_cls)
        except NotImplementedError:
            unregistered_config_classes.append(module_cfg_cls)

    assert (
        len(unregistered_config_classes) == 0
    ), f"No optimization config registered for following module optimization config classes: {unregistered_config_classes}"


def test_optimizer_registry():
    skipped_configs = [OptimizationConfig]
    all_coremltools_optimize_torch_classes = get_classes_recursively(coremltools.optimize.torch)
    optimization_config_classes = [
        c
        for n, c in all_coremltools_optimize_torch_classes.items()
        if issubclass(c, coremltools.optimize.torch.optimization_config.OptimizationConfig)
        and c not in skipped_configs
    ]

    unregistered_config_classes = []
    for opt_cfg_cls in optimization_config_classes:
        try:
            opt_cls = _ConfigToOptimizerRegistry.get_optimizer(opt_cfg_cls)
        except NotImplementedError:
            unregistered_config_classes.append(opt_cfg_cls)

    assert (
        len(unregistered_config_classes) == 0
    ), f"No optimizer registered for following config classes: {unregistered_config_classes}"
