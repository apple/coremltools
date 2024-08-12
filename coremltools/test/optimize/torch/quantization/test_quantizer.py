#  Copyright (c) 2024, Apple Inc. All rights reserved.
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

import coremltools.optimize.torch.quantization.modules.qat_modules as _qat
from coremltools.optimize.torch._utils.metadata_utils import CompressionMetadata, CompressionType
from coremltools.optimize.torch.quantization import (
    LinearQuantizer,
    LinearQuantizerConfig,
    ModuleLinearQuantizerConfig,
    QuantizationScheme,
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
        {"module_type_configs": {nn.ConvTranspose2d: {"weight_dtype": torch.quint8}}},
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
                "ConvTranspose2d": {"quantization_scheme": "affine"},
            }
        },
        {
            "module_type_configs": {nn.Linear: {"quantization_scheme": "symmetric"}},
            "module_name_configs": {"conv1": {"quantization_scheme": "affine"}},
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
        "module_type_configs": {
            nn.Linear: {"quantization_scheme": "affine"},
            nn.ConvTranspose2d: {"quantization_scheme": "affine"},
        },
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


@pytest.mark.parametrize(
    "model_config",
    [
        (
            nn.Sequential(
                OrderedDict(
                    {
                        "conv": nn.Conv2d(1, 20, (3, 3)),
                        "relu": nn.ReLU(),
                    }
                )
            ),
            torch.nn.intrinsic.qat.ConvReLU2d,
        ),
        (
            nn.Sequential(
                OrderedDict(
                    {
                        "conv": nn.ConvTranspose2d(1, 20, (3, 3)),
                        "relu": nn.ReLU(),
                    }
                )
            ),
            _qat.ConvTransposeAct2d,
        ),
    ],
)
@pytest.mark.parametrize("quantization_scheme", ["symmetric", "affine"])
def test_activation_defaults(quantization_scheme, model_config):

    config = LinearQuantizerConfig.from_dict(
        {"global_config": {
            "quantization_scheme": quantization_scheme,
            "milestones": [0, 2, 3, 3],
        }}
    )

    model, model_conv_instance = model_config
    quantizer = LinearQuantizer(model, config)
    model = quantizer.prepare(example_inputs=(torch.randn(1, 1, 28, 28),))

    assert isinstance(model.conv, model_conv_instance)
    assert model.activation_post_process_0.dtype == torch.quint8
    if quantization_scheme == "symmetric":
        assert model.activation_post_process_0.qscheme == torch.per_tensor_symmetric
    else:
        assert model.activation_post_process_0.qscheme == torch.per_tensor_affine
    assert model.activation_post_process_1.dtype == torch.quint8
    assert model.activation_post_process_1.qscheme == torch.per_tensor_affine


@pytest.mark.parametrize(
    "model_config",
    [
        (
            nn.Sequential(
                OrderedDict(
                    {
                        "conv": nn.Conv2d(1, 20, (3, 3)),
                        "bn": nn.BatchNorm2d(20),
                        "relu": nn.ReLU(),
                    }
                )
            ),
            True,
        ),
        (
            nn.Sequential(
                OrderedDict(
                    {
                        "conv": nn.ConvTranspose2d(1, 20, (3, 3)),
                        "bn": nn.BatchNorm2d(20),
                        "relu": nn.ReLU(),
                    }
                )
            ),
            False,
        ),
    ],
)
@pytest.mark.parametrize("quantization_scheme", ["symmetric", "affine"])
def test_quantizer_step_mechanism(quantization_scheme, model_config):

    config = LinearQuantizerConfig.from_dict(
        {"global_config": {
            "quantization_scheme": quantization_scheme,
            "milestones": [0, 1, 2, 3],
        }}
    )

    model, pytorch_builtin_mod = model_config
    quantizer = LinearQuantizer(model, config)
    model = quantizer.prepare(example_inputs=(torch.randn(1, 1, 28, 28),))

    if pytorch_builtin_mod:
        bn_module_to_check = model.conv
    else:
        bn_module_to_check = model.conv.conv

    assert not model.activation_post_process_0.observer_enabled
    assert not model.activation_post_process_0.fake_quant_enabled
    assert not model.activation_post_process_1.observer_enabled
    assert not model.activation_post_process_1.fake_quant_enabled

    for idx in range(4):
        quantizer.step()
        if idx == 0:
            assert not getattr(bn_module_to_check, "freeze_bn")
            assert model.activation_post_process_0.observer_enabled
            assert not model.activation_post_process_0.fake_quant_enabled
            assert model.activation_post_process_1.observer_enabled
            assert not model.activation_post_process_1.fake_quant_enabled
        if idx == 1:
            assert not getattr(bn_module_to_check, "freeze_bn")
            assert model.activation_post_process_0.observer_enabled
            assert model.activation_post_process_0.fake_quant_enabled
            assert model.activation_post_process_1.observer_enabled
            assert model.activation_post_process_1.fake_quant_enabled
        if idx == 2:
            assert not getattr(bn_module_to_check, "freeze_bn")
            assert not model.activation_post_process_0.observer_enabled
            assert model.activation_post_process_0.fake_quant_enabled
            assert not model.activation_post_process_1.observer_enabled
            assert model.activation_post_process_1.fake_quant_enabled
        if idx == 3:
            assert getattr(bn_module_to_check, "freeze_bn")
            assert not model.activation_post_process_0.observer_enabled
            assert model.activation_post_process_0.fake_quant_enabled
            assert not model.activation_post_process_1.observer_enabled
            assert model.activation_post_process_1.fake_quant_enabled


@pytest.mark.parametrize(
    "model_dict",
    [
        OrderedDict(
            {
                "conv": nn.Conv2d(1, 20, (3, 3)),
                "bn": nn.BatchNorm2d(20),
                "relu": nn.ReLU(),
            }
        ),
        OrderedDict(
            {
                "conv": nn.ConvTranspose2d(1, 20, (3, 3)),
                "bn": nn.BatchNorm2d(20),
                "relu": nn.ReLU(),
            }
        ),
    ],
)
def test_preserved_attributes(model_dict):
    """
    Test if methods and attributes on the model are preserved by passing
    preserved_attributes to the config.
    """

    class MyModel(nn.Sequential):
        def __init__(self, model_dict):
            super().__init__(model_dict)
            self.conv.weight.data = torch.ones_like(self.conv.weight.data)

        def my_method(self):
            return self.weight + torch.ones_like(self.weight)

        @property
        def weight(self):
            return (
                self.conv.weight
                if hasattr(self.conv, "weight")
                else self.conv.get_submodule("0").weight
            )

    preserved_attrs = ["key_1", "key_2", "my_method", "weight"]

    model = MyModel(model_dict)
    model.key_1 = 5
    model.key_2 = torch.tensor(5)

    config = LinearQuantizerConfig.from_dict(
        {
            "global_config": {
                "milestones": [0, 3, 4, 5],
            },
            "preserved_attributes": preserved_attrs,
        }
    )
    quantizer_1 = LinearQuantizer(model, LinearQuantizerConfig())
    prepared_model = quantizer_1.prepare(example_inputs=(torch.randn(1),), inplace=False)
    for attr in preserved_attrs:
        assert not hasattr(prepared_model, attr)

    quantizer_2 = LinearQuantizer(model, config)
    prepared_model = quantizer_2.prepare(example_inputs=(torch.randn(1),), inplace=False)
    for attr in preserved_attrs:
        assert hasattr(prepared_model, attr)
    assert torch.all(
        prepared_model.my_method() == 2 * torch.ones_like(prepared_model.conv.weight.data)
    )

    quantizer_2.step()
    prepared_model(torch.randn(2, 1, 28, 28))
    final_model = quantizer_2.finalize()
    for attr in preserved_attrs:
        assert hasattr(final_model, attr)
    assert torch.all(
        final_model.my_method()
        == final_model.weight.data + torch.ones_like(prepared_model.weight.data)
    )


@pytest.mark.optional
@pytest.mark.parametrize("algorithm", ["vanilla", "learnable"])
@pytest.mark.parametrize("weight_dtype", ["qint8", "quint8", "qint4", "quint4"])
@pytest.mark.parametrize("weight_per_channel", [True, False])
@pytest.mark.parametrize(
    "quantization_scheme", [QuantizationScheme.symmetric, QuantizationScheme.affine]
)
def test_linear_quantizer_report(
    mnist_model_conv_transpose,
    algorithm,
    weight_dtype,
    weight_per_channel,
    quantization_scheme,
):
    print("\nTESTING REPORT WITH")
    print("ALGORITHM", algorithm)
    print("WEIGHT_DTYPE", weight_dtype)
    print("WEIGHT_PER_CHANNEL", weight_per_channel)
    print("QUANTIZATION_SCHEME", quantization_scheme)

    config = LinearQuantizerConfig.from_dict(
        {
            "global_config": {
                "milestones": [0, 1, 1, 3],
                "algorithm": algorithm,
                "weight_dtype": weight_dtype,
                "weight_per_channel": weight_per_channel,
                "quantization_scheme": quantization_scheme,
            },
            "module_name_configs": {
                "dense2": {
                    "milestones": [0, 1, 1, 3],
                    "activation_dtype": torch.float32,
                    "algorithm": algorithm,
                    "weight_dtype": weight_dtype,
                    "weight_per_channel": weight_per_channel,
                    "quantization_scheme": quantization_scheme,
                }
            },
        }
    )

    quantizer = LinearQuantizer(mnist_model_conv_transpose, config)
    prepared_model = quantizer.prepare(example_inputs=(torch.randn(1, 1, 28, 28),))

    report = quantizer.report()

    print("\nREPORT\n" + str(report))


@pytest.mark.parametrize("dtype", ["qint4", "qint8"])
@pytest.mark.parametrize("scheme", ["symmetric", "affine"])
@pytest.mark.parametrize("conv_transpose", [False, True])
def test_compression_metadata(dtype, scheme, conv_transpose):
    """
    Test that calling finalize on the module leads to compression metadata being added to the model
    """
    model = nn.Sequential(
        OrderedDict(
            [
                (
                    "conv1",
                    (nn.Conv2d(1, 20, 3) if not conv_transpose else nn.ConvTranspose2d(1, 20, 3)),
                ),
                ("fc1", nn.Linear(20, 100)),
            ]
        )
    )
    config = LinearQuantizerConfig.from_dict(
        {
            "module_name_configs": {
                "conv1": {
                    "weight_dtype": dtype,
                    "quantization_scheme": scheme,
                },
                "fc1": None,
            }
        }
    )
    quantizer = LinearQuantizer(model, config)
    quantizer.prepare(inplace=True, example_inputs=(torch.randn(1, 1, 28, 28),))
    for _ in range(4):
        quantizer.step()
    model = quantizer.finalize()

    # Verify metadata version is added to model
    assert "_COREML_/metadata_version" in model.state_dict()

    # Verify compression metadata is added for conv1
    metadata_dict = CompressionMetadata.from_state_dict(model.conv1.state_dict())
    assert len(metadata_dict) == 1
    assert "weight" in metadata_dict

    metadata = metadata_dict["weight"]
    assert metadata.compression_type == [CompressionType.quantization.value]
    assert metadata.quantization_n_bits == 4 if dtype == "qint4" else 8
    scale_zero_point_shape = (20, 1) if not conv_transpose else (1, 20)
    assert metadata.quantization_scale.shape == scale_zero_point_shape
    assert metadata.zero_point.shape == scale_zero_point_shape
    if scheme == "symmetric":
        assert torch.all(metadata.zero_point == 0)

    # # Verify no compression metadata is added for fc1
    metadata_dict = CompressionMetadata.from_state_dict(model.fc1.state_dict())
    assert len(metadata_dict) == 0
