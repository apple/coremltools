#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict

import cattrs
import pytest
import torch
import torch.ao.quantization
import torch.nn as nn
import torch.nn.intrinsic
import torch.nn.intrinsic.qat
import torch.nn.quantized
import torch.nn.quantized.modules.utils

from coremltools.optimize.torch.quantization import (
    LinearQuantizer,
    LinearQuantizerConfig,
    ModuleLinearQuantizerConfig,
)


@pytest.mark.parametrize(
    "option_and_value", [
        ("weight_dtype", torch.int32),
        ("activation_dtype", torch.int8),
        ("milestones", [0, 2])
    ]
)
def test_config_illegal_options(option_and_value):
    option, value = option_and_value
    with pytest.raises(cattrs.errors.ClassValidationError):
        LinearQuantizerConfig.from_dict({"global_config": {option: value}})


@pytest.mark.parametrize(
    "config_dict",
    [
        {"module_type_configs": {nn.Linear: {"weight_dtype": torch.quint8}}},
        {"module_name_configs": {"conv2d": {"weight_dtype": torch.quint8}}},
        {"global_config": {"weight_dtype": torch.quint8}},
        {},
    ],
)
def test_linear_quantizer_config_global_config_set(config_dict):
    config = LinearQuantizerConfig.from_dict(config_dict)
    if len(config_dict) == 0:
        assert config.global_config == ModuleLinearQuantizerConfig()
    else:
        keys = ["global_config", "module_type_configs", "module_name_configs"]
        for key in keys:
            if key not in config_dict:
                param_in_config = getattr(config, key)
                assert param_in_config is None or len(param_in_config) == 0
        if "global_config" in config_dict:
            assert config.global_config.weight_dtype == config_dict["global_config"]["weight_dtype"]
        if "module_name_configs" in config_dict:
            for key in config_dict["module_name_configs"]:
                assert config.module_name_configs[key].weight_dtype == \
                       config_dict["module_name_configs"][key]["weight_dtype"]
        if "module_type_configs" in config_dict:
            for key in config_dict["module_type_configs"]:
                assert config.module_type_configs[key].weight_dtype == \
                       config_dict["module_type_configs"][key]["weight_dtype"]


@pytest.mark.parametrize(
    "config_dict",
    [
        {
            "global_config": {"quantization_scheme": "affine"},
            "module_name_configs": {"conv1": {"quantization_scheme": "symmetric"}},
        },
        {
            "global_config": {"quantization_scheme": "affine"},
            "module_type_configs": {nn.Linear: {"quantization_scheme": "symmetric"}},
        },
        {
            "module_name_configs": {
                "conv1": {"quantization_scheme": "affine"},
                "conv2": {"quantization_scheme": "symmetric"},
            }
        },
        {
            "module_type_configs": {
                nn.Linear: {"quantization_scheme": "symmetric"},
                "Conv2d": {"quantization_scheme": "affine"},
            }
        },
        {
            "module_type_configs": {nn.Linear: {"quantization_scheme": "symmetric"}},
            "module_name_configs": {"conv1": {"quantization_scheme": "affine"}},
        },
        {"global_config": {"activation_dtype": "quint8", "weight_dtype": "float32"}},
        {
            "module_name_configs": {
                "conv1": {"activation_dtype": "quint8", "weight_dtype": "float32"}
            }
        },
        {
            "module_name_configs": {
                "Conv2d": {"activation_dtype": "quint8", "weight_dtype": "float32"}
            }
        },
    ],
)
def test_linear_quantizer_config_failure_modes(config_dict):
    with pytest.raises(Exception):
        LinearQuantizerConfig.from_dict(config_dict)


def test_linear_quantizer_config_different_config_success():
    config_dict = {
        "global_config": {"quantization_scheme": "affine"},
        "module_name_configs": {
            "conv1": {"quantization_scheme": "affine"},
            "conv2": {"quantization_scheme": "affine"},
        },
        "module_type_configs": {nn.Linear: {"quantization_scheme": "affine"}},
    }
    LinearQuantizerConfig.from_dict(config_dict)


@pytest.mark.parametrize(
    "config_dict",
    [
        {
            "global_config": {"quantization_scheme": "affine"},
            "module_name_configs": {
                "conv1": {"quantization_scheme": "affine"},
                "conv2": {"quantization_scheme": "affine"},
            },
            "module_type_configs": {nn.Linear: {"quantization_scheme": "affine"}},
        },
        {
            "module_name_configs": {
                "conv1": {"quantization_scheme": "affine"},
                "conv2": {"quantization_scheme": "affine"},
            }
        },
        {"module_type_configs": {nn.Linear: {"quantization_scheme": "affine"}}},
        {},
    ],
)
def test_linear_quantizer_quantization_scheme_setting(config_dict):
    model = nn.Sequential(OrderedDict({
        'conv': nn.Conv2d(1, 20, (3, 3)),
        'relu': nn.ReLU(),
    }))
    config = LinearQuantizerConfig.from_dict(config_dict)
    quantizer = LinearQuantizer(model, config)

    def_quantization_scheme = ModuleLinearQuantizerConfig().quantization_scheme.value
    quantization_scheme = quantizer._quantization_scheme.value
    if len(config_dict) == 0:
        assert def_quantization_scheme == quantization_scheme
    else:
        assert quantization_scheme == "affine"


@pytest.mark.parametrize("quantization_scheme", ["symmetric", "affine"])
def test_activation_defaults(quantization_scheme):
    model = nn.Sequential(OrderedDict({
        'conv': nn.Conv2d(1, 20, (3, 3)),
        'relu': nn.ReLU(),
    }))
    config = LinearQuantizerConfig.from_dict(
        {"global_config": {
            "quantization_scheme": quantization_scheme,
            "milestones": [0, 2, 3, 3],
        }}
    )
    quantizer = LinearQuantizer(model, config)
    model = quantizer.prepare(example_inputs=torch.randn(1, 1, 28, 28))

    assert isinstance(model.conv, torch.nn.intrinsic.qat.ConvReLU2d)
    assert model.activation_post_process_0.dtype == torch.quint8
    if quantization_scheme == "symmetric":
        assert model.activation_post_process_0.qscheme == torch.per_tensor_symmetric
    else:
        assert model.activation_post_process_0.qscheme == torch.per_tensor_affine
    assert model.activation_post_process_1.dtype == torch.quint8
    assert model.activation_post_process_1.qscheme == torch.per_tensor_affine


@pytest.mark.parametrize("quantization_scheme", ["symmetric", "affine"])
def test_quantizer_step_mechanism(quantization_scheme):
    model = nn.Sequential(OrderedDict({
        'conv': nn.Conv2d(1, 20, (3, 3)),
        'bn': nn.BatchNorm2d(20),
        'relu': nn.ReLU(),
    }))

    config = LinearQuantizerConfig.from_dict(
        {"global_config": {
            "quantization_scheme": quantization_scheme,
            "milestones": [0, 1, 2, 3],
        }}
    )
    quantizer = LinearQuantizer(model, config)
    model = quantizer.prepare(example_inputs=torch.randn(1, 1, 28, 28))

    assert not model.activation_post_process_0.observer_enabled
    assert not model.activation_post_process_0.fake_quant_enabled
    assert not model.activation_post_process_1.observer_enabled
    assert not model.activation_post_process_1.fake_quant_enabled

    for idx in range(4):
        quantizer.step()
        if idx == 0:
            assert not model.conv.freeze_bn
            assert model.activation_post_process_0.observer_enabled
            assert not model.activation_post_process_0.fake_quant_enabled
            assert model.activation_post_process_1.observer_enabled
            assert not model.activation_post_process_1.fake_quant_enabled
        if idx == 1:
            assert not model.conv.freeze_bn
            assert model.activation_post_process_0.observer_enabled
            assert model.activation_post_process_0.fake_quant_enabled
            assert model.activation_post_process_1.observer_enabled
            assert model.activation_post_process_1.fake_quant_enabled
        if idx == 2:
            assert not model.conv.freeze_bn
            assert not model.activation_post_process_0.observer_enabled
            assert model.activation_post_process_0.fake_quant_enabled
            assert not model.activation_post_process_1.observer_enabled
            assert model.activation_post_process_1.fake_quant_enabled
        if idx == 3:
            assert model.conv.freeze_bn
            assert not model.activation_post_process_0.observer_enabled
            assert model.activation_post_process_0.fake_quant_enabled
            assert not model.activation_post_process_1.observer_enabled
            assert model.activation_post_process_1.fake_quant_enabled
