#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import math
import operator
from collections import OrderedDict
from typing import List

import pytest
import torch
import torch.ao.nn.quantized.reference
import torch.ao.quantization
import torch.nn as nn
import torch.nn.intrinsic
import torch.nn.intrinsic.qat
import torch.nn.qat
import torch.nn.quantized

from coremltools.optimize.torch.quantization import (
    LinearQuantizer,
    LinearQuantizerConfig,
    ModuleLinearQuantizerConfig,
)
from coremltools.optimize.torch.quantization._backend_config import _mod_activations
from coremltools.optimize.torch.quantization._qconfig_mapping import _QConfigMappingBuilder
from coremltools.optimize.torch.quantization._utils import (
    find_module,
    get_quant_range,
    is_activation_post_process,
)
from coremltools.optimize.torch.quantization.modules import fused_modules as _fused
from coremltools.optimize.torch.quantization.modules import qat_modules as _qat
from coremltools.optimize.torch.quantization.modules import quantized_modules as _quantized
from coremltools.optimize.torch.quantization.quantization_config import QuantizationScheme


def get_configs_for_qscheme(
    activation_dtype=torch.quint8,
    weight_per_channel=True,
    weight_dtype=torch.qint8,
) -> List[LinearQuantizerConfig]:
    return [
        LinearQuantizerConfig.from_dict(
            {
                "global_config": {
                    "quantization_scheme": QuantizationScheme.symmetric,
                    "milestones": [0, 0, 10, 10],
                    "weight_dtype": weight_dtype,
                    "activation_dtype": activation_dtype,
                    "weight_per_channel": weight_per_channel,
                }
            }
        ),
        LinearQuantizerConfig.from_dict(
            {
                "global_config": {
                    "quantization_scheme": QuantizationScheme.affine,
                    "milestones": [0, 0, 10, 10],
                    "weight_dtype": weight_dtype,
                    "activation_dtype": activation_dtype,
                    "weight_per_channel": weight_per_channel,
                }
            }
        ),
    ]


def quantize_model(model, data, config=None):
    quantizer = LinearQuantizer(model, config)
    prepared_model = quantizer.prepare(example_inputs=(data,), inplace=False)
    quantizer.step()
    prepared_model(data)
    return prepared_model, quantizer


def _verify_quant_range(fake_quant, weight_n_bits, weight_dtype):
    quant_min, quant_max = get_quant_range(n_bits=weight_n_bits, dtype=weight_dtype)
    assert fake_quant.quant_min == quant_min
    assert fake_quant.quant_max == quant_max


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
            True,
            torch.nn.intrinsic.qat.ConvReLU2d,
            torch.ao.nn.intrinsic.ConvReLU2d,
            torch.ao.nn.quantized.reference.Conv2d,
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
            False,
            _qat.ConvTransposeAct2d,
            _quantized.QuantizedConvTransposeAct2d,
            torch.ao.nn.quantized.reference.ConvTranspose2d,
        ),
    ],
)
@pytest.mark.parametrize(
    "config",
    get_configs_for_qscheme()
    + get_configs_for_qscheme(weight_per_channel=False)
    + get_configs_for_qscheme(weight_dtype="qint8")
    + get_configs_for_qscheme(weight_dtype=torch.quint8),
)
def test_conv_relu_fusion(config, model_config):

    (
        model,
        pytorch_builtin_mod,
        qat_mod_type,
        fused_quant_mod_type,
        ref_quant_mod_type,
    ) = model_config

    data = torch.randn(1, 1, 28, 28)

    prepared_model, quantizer = quantize_model(model, data, config)

    assert isinstance(prepared_model.conv, qat_mod_type)
    _verify_quant_range(
        prepared_model.conv.weight_fake_quant,
        weight_n_bits=config.global_config.weight_n_bits,
        weight_dtype=config.global_config.weight_dtype,
    )

    converted_model = quantizer.finalize(inplace=False)

    assert isinstance(converted_model.conv, fused_quant_mod_type)
    assert isinstance(
        converted_model.conv[0] if pytorch_builtin_mod else converted_model.conv.conv,
        ref_quant_mod_type,
    )


@pytest.mark.parametrize(
    "config",
    get_configs_for_qscheme()
    + get_configs_for_qscheme(weight_per_channel=False)
    + get_configs_for_qscheme(weight_dtype="qint4")
    + get_configs_for_qscheme(weight_dtype="quint4"),
)
@pytest.mark.parametrize("activation_fn", list(_mod_activations))
@pytest.mark.parametrize("conv_transpose", [False, True])
def test_conv_act_fusion(config, activation_fn, conv_transpose):
    model = nn.Sequential(
        OrderedDict(
            {
                "conv": (
                    nn.Conv2d(1, 20, (3, 3))
                    if not conv_transpose
                    else nn.ConvTranspose2d(1, 20, (3, 3))
                ),
                "act": activation_fn(),
            }
        )
    )
    data = torch.randn(1, 1, 28, 28)

    prepared_model, quantizer = quantize_model(model, data, config)

    if not conv_transpose:
        assert isinstance(prepared_model.conv, _qat.ConvAct2d)
    else:
        assert isinstance(prepared_model.conv, _qat.ConvTransposeAct2d)

    assert isinstance(prepared_model.conv.act, activation_fn)
    _verify_quant_range(
        prepared_model.conv.conv.weight_fake_quant,
        weight_n_bits=config.global_config.weight_n_bits,
        weight_dtype=config.global_config.weight_dtype,
    )

    converted_model = quantizer.finalize(inplace=False)

    if not conv_transpose:
        assert isinstance(converted_model.conv, _quantized.QuantizedConvAct2d)
    else:
        assert isinstance(converted_model.conv, _quantized.QuantizedConvTransposeAct2d)

    assert isinstance(converted_model.conv.act, activation_fn)


@pytest.mark.parametrize(
    "config",
    get_configs_for_qscheme()
    + get_configs_for_qscheme(weight_per_channel=False)
    + get_configs_for_qscheme(weight_dtype="qint4")
    + get_configs_for_qscheme(weight_dtype="quint4"),
)
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
            torch.nn.intrinsic.qat.ConvBnReLU2d,
            torch.ao.nn.intrinsic.ConvReLU2d,
            torch.ao.nn.quantized.reference.Conv2d,
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
            _qat.ConvTransposeBnAct2d,
            _quantized.QuantizedConvTransposeAct2d,
            torch.ao.nn.quantized.reference.ConvTranspose2d,
        ),
    ],
)
def test_conv_bn_relu_fusion(config, model_config):

    (
        model,
        pytorch_builtin_mod,
        qat_mod_type,
        fused_quant_mod_type,
        ref_quant_mod_type,
    ) = model_config
    data = torch.randn(1, 1, 28, 28)

    prepared_model, quantizer = quantize_model(model, data, config)

    assert isinstance(prepared_model.conv, qat_mod_type)
    _verify_quant_range(
        prepared_model.conv.weight_fake_quant,
        weight_n_bits=config.global_config.weight_n_bits,
        weight_dtype=config.global_config.weight_dtype,
    )

    converted_model = quantizer.finalize(inplace=False)

    assert isinstance(converted_model.conv, fused_quant_mod_type)
    assert isinstance(
        converted_model.conv[0] if pytorch_builtin_mod else converted_model.conv.conv,
        ref_quant_mod_type,
    )


@pytest.mark.parametrize(
    "config",
    get_configs_for_qscheme()
    + get_configs_for_qscheme(weight_per_channel=False)
    + get_configs_for_qscheme(weight_dtype="qint4")
    + get_configs_for_qscheme(weight_dtype="quint4"),
)
@pytest.mark.parametrize("activation_fn", list(_mod_activations))
@pytest.mark.parametrize("conv_transpose", [False, True])
def test_conv_bn_act_fusion(config, activation_fn, conv_transpose):
    model = nn.Sequential(
        OrderedDict(
            {
                "conv": (
                    nn.Conv2d(1, 20, (3, 3))
                    if not conv_transpose
                    else nn.ConvTranspose2d(1, 20, (3, 3))
                ),
                "bn": nn.BatchNorm2d(20),
                "act": activation_fn(),
            }
        )
    )
    data = torch.randn(1, 1, 28, 28)

    prepared_model, quantizer = quantize_model(model, data, config)

    if not conv_transpose:
        assert isinstance(prepared_model.conv, _qat.ConvBnAct2d)
    else:
        assert isinstance(prepared_model.conv, _qat.ConvTransposeBnAct2d)

    assert isinstance(prepared_model.conv.act, activation_fn)
    _verify_quant_range(
        prepared_model.conv.conv.weight_fake_quant,
        weight_n_bits=config.global_config.weight_n_bits,
        weight_dtype=config.global_config.weight_dtype,
    )

    converted_model = quantizer.finalize(inplace=False)

    if not conv_transpose:
        assert isinstance(converted_model.conv, _quantized.QuantizedConvAct2d)
    else:
        assert isinstance(converted_model.conv, _quantized.QuantizedConvTransposeAct2d)

    assert isinstance(converted_model.conv.act, activation_fn)


@pytest.mark.parametrize(
    "config",
    get_configs_for_qscheme()
    + get_configs_for_qscheme(weight_per_channel=False)
    + get_configs_for_qscheme(weight_dtype="qint4")
    + get_configs_for_qscheme(weight_dtype="quint4"),
)
def test_linear_relu_fusion(config):
    model = nn.Sequential(OrderedDict({"linear": nn.Linear(20, 100), "act": nn.ReLU()}))
    data = torch.randn(1, 20)

    prepared_model, quantizer = quantize_model(model, data, config)

    assert isinstance(prepared_model.linear, torch.nn.intrinsic.qat.LinearReLU)
    _verify_quant_range(
        prepared_model.linear.weight_fake_quant,
        weight_n_bits=config.global_config.weight_n_bits,
        weight_dtype=config.global_config.weight_dtype,
    )

    converted_model = quantizer.finalize(inplace=False)

    assert isinstance(converted_model.linear, torch.ao.nn.intrinsic.LinearReLU)
    assert isinstance(converted_model.linear[0], torch.ao.nn.quantized.reference.Linear)


@pytest.mark.parametrize(
    "config",
    get_configs_for_qscheme()
    + get_configs_for_qscheme(weight_per_channel=False)
    + get_configs_for_qscheme(weight_dtype="qint4")
    + get_configs_for_qscheme(weight_dtype="quint4"),
)
@pytest.mark.parametrize("activation_fn", list(_mod_activations))
def test_linear_act_fusion(config, activation_fn):
    model = nn.Sequential(OrderedDict({
        'linear': nn.Linear(20, 100),
        'act': activation_fn(),
    }))
    data = torch.randn(1, 20)

    prepared_model, quantizer = quantize_model(model, data, config)

    assert isinstance(prepared_model.linear, _qat.LinearAct)
    assert isinstance(prepared_model.linear.act, activation_fn)
    _verify_quant_range(
        prepared_model.linear.linear.weight_fake_quant,
        weight_n_bits=config.global_config.weight_n_bits,
        weight_dtype=config.global_config.weight_dtype,
    )

    converted_model = quantizer.finalize(inplace=False)

    assert isinstance(converted_model.linear, _quantized.QuantizedLinearAct)
    assert isinstance(converted_model.linear.act, activation_fn)


@pytest.mark.parametrize("activation_fn", [torch.nn.ReLU, torch.nn.ReLU6])
@pytest.mark.parametrize(
    "layer_and_data",
    [
        [nn.Conv2d(1, 20, (3, 3)), torch.randn(1, 1, 28, 28)],
        [nn.ConvTranspose2d(1, 20, (3, 3)), torch.randn(1, 1, 28, 28)],
        [nn.Linear(20, 100), torch.randn(1, 20)],
    ],
)
@pytest.mark.parametrize("bn", [nn.BatchNorm2d(20), None])
def test_single_act_qscheme_for_symmetric(activation_fn, layer_and_data, bn):
    """
    Tests that when qscheme is symmetric, always affine layers have affine qscheme
    """
    layer, data = layer_and_data
    if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d)) and bn is not None:
        model = nn.Sequential(
            OrderedDict(
                {
                    "layer": layer,
                    "bn": bn,
                    "act": activation_fn(),
                }
            )
        )
    else:
        model = nn.Sequential(OrderedDict({
            'layer': layer,
            'act': activation_fn(),
        }))

    prepared_model, _ = quantize_model(model, data)

    assert prepared_model.activation_post_process_0.qscheme == torch.per_tensor_symmetric
    assert prepared_model.activation_post_process_1.qscheme == torch.per_tensor_affine


@pytest.mark.parametrize(
    "activation_fn",
    [torch.nn.Hardsigmoid, torch.nn.Sigmoid, torch.nn.Softmax, torch.nn.Tanh],
)
@pytest.mark.parametrize(
    "layer_and_data",
    [
        [nn.Conv2d(1, 20, (3, 3)), torch.randn(1, 1, 28, 28)],
        [nn.ConvTranspose2d(1, 20, (3, 3)), torch.randn(1, 1, 28, 28)],
        [nn.Linear(20, 100), torch.randn(1, 20)],
    ],
)
@pytest.mark.parametrize("bn", [nn.BatchNorm2d(20), None])
@pytest.mark.parametrize("config", get_configs_for_qscheme())
def test_single_fixed_qparams_act_for_symmetric(
    activation_fn, layer_and_data, bn, config
):
    """
    Tests that when qscheme is symmetric, the qparams of fixed qparam ops are maintained
    """
    layer, data = layer_and_data
    if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d)) and bn is not None:
        model = nn.Sequential(
            OrderedDict(
                {
                    "layer": layer,
                    "bn": bn,
                    "act": activation_fn(),
                }
            )
        )
    else:
        model = nn.Sequential(OrderedDict({
            'layer': layer,
            'act': activation_fn(),
        }))

    prepared_model, _ = quantize_model(model, data)

    builder = _QConfigMappingBuilder()
    qconfig = builder.get_default_qconfig_mapping(
        QuantizationScheme.symmetric,
        ModuleLinearQuantizerConfig(),
    ).object_type_qconfigs[activation_fn]

    assert prepared_model.activation_post_process_1.scale == qconfig.activation().scale
    assert prepared_model.activation_post_process_1.zero_point == qconfig.activation().zero_point


@pytest.mark.parametrize("activation_fn", [nn.ReLU, nn.ReLU6])
@pytest.mark.parametrize("conv_transpose", [False, True])
def test_dropout_affine_input(activation_fn, conv_transpose):
    model = nn.Sequential(
        OrderedDict(
            {
                "conv": (
                    nn.Conv2d(1, 20, (3, 3))
                    if not conv_transpose
                    else nn.ConvTranspose2d(1, 20, (3, 3))
                ),
                "relu": activation_fn(),
                "dropout": nn.Dropout2d(),
                "leaky_relu": nn.LeakyReLU(),
            }
        )
    )
    data = torch.randn(1, 1, 28, 28)

    prepared_model, _ = quantize_model(model, data)

    assert prepared_model.activation_post_process_1.qscheme == torch.per_tensor_affine
    assert not hasattr(prepared_model, "activation_post_process_2")
    assert prepared_model.activation_post_process_3.qscheme == torch.per_tensor_symmetric


def test_sequential_network_config_for_symmetric(mnist_model_quantization):
    """
    Tests a sequential network with multiple modules is configured correctly.
    This network has layers where input and output observers are shared. We test
    that for these layers, we set acitvation quantizer correctly for always affine layers
    """
    data = torch.randn(1, 1, 28, 28)
    prepared_model, quantizer = quantize_model(mnist_model_quantization, data)

    # verify module fusion
    assert isinstance(prepared_model.conv1, _qat.ConvBnAct2d)
    assert isinstance(prepared_model.conv2, _qat.ConvAct2d)
    assert isinstance(prepared_model.conv_transpose1, _qat.ConvTransposeBnAct2d)
    assert isinstance(prepared_model.conv_transpose2, _qat.ConvTransposeAct2d)
    assert isinstance(prepared_model.dense1, _qat.LinearAct)
    assert isinstance(prepared_model.dense2, _qat.LinearAct)

    # verify activation quantizers
    # after input
    assert prepared_model.activation_post_process_0.qscheme == torch.per_tensor_symmetric
    # after conv1
    assert prepared_model.activation_post_process_1.qscheme == torch.per_tensor_affine
    # after pool, this is shared with output of conv1
    assert id(prepared_model.activation_post_process_1) == id(prepared_model.activation_post_process_2)
    # after conv2
    assert prepared_model.activation_post_process_3.qscheme == torch.per_tensor_affine
    # after pool, shared with output of conv2
    assert id(prepared_model.activation_post_process_3) == id(prepared_model.activation_post_process_4)
    # after conv_transpose1
    assert prepared_model.activation_post_process_5.qscheme == torch.per_tensor_affine
    # after pool, shared with output of conv_transpose1
    assert id(prepared_model.activation_post_process_5) == id(
        prepared_model.activation_post_process_6
    )
    # after conv_transpose2
    assert prepared_model.activation_post_process_7.qscheme == torch.per_tensor_symmetric
    # after flatten, shared with the output of conv_transpose2
    assert id(prepared_model.activation_post_process_7) == id(
        prepared_model.activation_post_process_8
    )
    # after linear1
    assert prepared_model.activation_post_process_9.qscheme == torch.per_tensor_affine
    # after dropout
    # we remove activation post process after dropout layer
    assert not hasattr(prepared_model, "activation_post_process_10")
    # after linear2, logsoftmax
    assert prepared_model.activation_post_process_11.qscheme == torch.per_tensor_symmetric

    # convert model and test fusion
    converted_model = quantizer.finalize(inplace=False)

    # assert converted module fusion
    assert isinstance(converted_model.conv1, _quantized.QuantizedConvAct2d)
    assert isinstance(converted_model.conv2, _quantized.QuantizedConvAct2d)
    assert isinstance(converted_model.conv_transpose1, _quantized.QuantizedConvTransposeAct2d)
    assert isinstance(converted_model.conv_transpose2, _quantized.QuantizedConvTransposeAct2d)
    assert isinstance(converted_model.dense1, _quantized.QuantizedLinearAct)
    assert isinstance(converted_model.dense2, _quantized.QuantizedLinearAct)


class ConvBlock(nn.Module):
    def __init__(self, conv_transpose, activation):
        super().__init__()
        if conv_transpose:
            self.conv = nn.ConvTranspose2d(1, 20, (3, 3), padding=1)
        else:
            self.conv = nn.Conv2d(1, 20, (3, 3), padding="same")
        self.activation = activation

    def forward(self, x):
        return self.activation(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, conv_transpose: bool, activation: nn.Module):
        super().__init__()
        self.conv = ConvBlock(conv_transpose, activation)

    def forward(self, x):
        return x + self.conv(x)


@pytest.mark.parametrize("activation_fn", [torch.nn.functional.relu, torch.nn.functional.relu_])
@pytest.mark.parametrize("conv_transpose", [False, True])
def test_functional_relu_qscheme_for_symmetric(activation_fn, conv_transpose):
    class Model(nn.Module):
        def __init__(self, conv_transpose):
            super().__init__()
            if not conv_transpose:
                self.conv1 = nn.Conv2d(1, 20, (3, 3), padding="same")
                self.conv2 = nn.Conv2d(20, 20, (3, 3), padding="same")
            else:
                self.conv1 = nn.Conv2d(1, 20, (3, 3), padding=1)
                self.conv2 = nn.Conv2d(20, 20, (3, 3), padding=1)

        def forward(self, x):
            return self.conv2(activation_fn(self.conv1(x)))

    model = Model(conv_transpose)
    data = torch.randn(1, 1, 28, 28)

    prepared_model, _ = quantize_model(model, data)

    if activation_fn == torch.nn.functional.relu:
        assert prepared_model.activation_post_process_1.qscheme == torch.per_tensor_affine
    else:
        assert prepared_model.activation_post_process_2.qscheme == torch.per_tensor_affine


@pytest.mark.parametrize("activation_fn", [torch.nn.ReLU, torch.nn.ReLU6])
@pytest.mark.parametrize("conv_transpose", [False, True])
def test_addition_of_uint_and_uint_for_symmetric(activation_fn, conv_transpose):
    model = nn.Sequential(
        OrderedDict(
            {
                "previous_activation": nn.ReLU(),
                "res_block": ResidualBlock(conv_transpose, activation_fn()),
            }
        )
    )
    data = torch.randn(1, 1, 28, 28)

    prepared_model, _ = quantize_model(model, data)

    assert prepared_model.activation_post_process_0.qscheme == torch.per_tensor_symmetric
    affine_acts = [prepared_model.activation_post_process_1,
                   prepared_model.activation_post_process_2, prepared_model.activation_post_process_3]
    for act in affine_acts:
        assert act.qscheme == torch.per_tensor_affine


@pytest.mark.parametrize("activation_fn", [torch.nn.ReLU, torch.nn.ReLU6])
@pytest.mark.parametrize("conv_transpose", [False, True])
def test_addition_of_int_and_uint_for_symmetric(activation_fn, conv_transpose):
    model = nn.Sequential(
        OrderedDict(
            {
                "previous_activation": nn.LeakyReLU(),
                "res_block": ResidualBlock(conv_transpose, activation_fn()),
            }
        )
    )
    data = torch.randn(1, 1, 28, 28)

    prepared_model, _ = quantize_model(model, data)

    # relu shares observer with input, so input is affine as well
    symmetric_acts = [prepared_model.activation_post_process_0, prepared_model.activation_post_process_1,
                      prepared_model.activation_post_process_3]
    for act in symmetric_acts:
        assert act.qscheme == torch.per_tensor_symmetric
    # output of conv block is still affine
    assert prepared_model.activation_post_process_2.qscheme == torch.per_tensor_affine


class ComplexAdd(nn.Module):
    """
    a (affine)
              + ->  c (symmetric)
    b (symmetric)
                                + -> g (symmetric)
    d (affine)
              +  -> f (affine)
    e (affine)
    """

    def __init__(self, activation_fn):
        super().__init__()
        self.lrelu = nn.LeakyReLU()
        self.relu1 = activation_fn()
        self.relu2 = activation_fn()
        self.relu3 = activation_fn()

    def forward(self, x):
        a = self.relu1(x)
        b = self.lrelu(x)
        d = self.relu2(x)
        e = self.relu3(x)
        c = a + b
        f = d + e
        g = c + f
        return g


@pytest.mark.parametrize("activation_fn", [torch.nn.ReLU, torch.nn.ReLU6])
def test_complex_add(activation_fn):
    model = ComplexAdd(activation_fn)
    data = torch.randn(1, 1, 28, 28)

    prepared_model, _ = quantize_model(model, data)

    symmetric_acts = [prepared_model.activation_post_process_0, prepared_model.activation_post_process_2,
                      prepared_model.activation_post_process_5, prepared_model.activation_post_process_7]
    for act in symmetric_acts:
        assert act.qscheme == torch.per_tensor_symmetric
    affine_acts = [prepared_model.activation_post_process_1, prepared_model.activation_post_process_3,
                   prepared_model.activation_post_process_4, prepared_model.activation_post_process_6]
    for act in affine_acts:
        assert act.qscheme == torch.per_tensor_affine


class ComplexConcatAdd(nn.Module):
    """
    conv_c (uint)  --- c.
                        .`-- concat
                   .--a2
    conv_a (uint) `
                   `--a1 `-- add
    conv_b (int)  ---- b `
    """

    def __init__(self, conv_transpose, activation_fn):
        super().__init__()
        self.conv_a = ConvBlock(conv_transpose, activation_fn())
        self.conv_b = ConvBlock(conv_transpose, nn.LeakyReLU())
        self.conv_c = ConvBlock(conv_transpose, activation_fn())

    def forward(self, x):
        a1 = self.conv_a(x)
        b = self.conv_b(x)
        ab = a1 + b
        c = self.conv_c(x)
        ac = torch.cat([a1, c])
        return ab, ac


@pytest.mark.parametrize("activation_fn", [torch.nn.ReLU, torch.nn.ReLU6])
@pytest.mark.parametrize("conv_transpose", [False, True])
def test_complex_concat_add(activation_fn, conv_transpose):
    model = ComplexConcatAdd(conv_transpose, activation_fn)
    data = torch.randn(1, 1, 28, 28)

    prepared_model, _ = quantize_model(model, data)

    symmetric_acts = [prepared_model.activation_post_process_0, prepared_model.activation_post_process_2,
                      prepared_model.activation_post_process_3]
    for act in symmetric_acts:
        assert act.qscheme == torch.per_tensor_symmetric
    affine_acts = [prepared_model.activation_post_process_1, prepared_model.activation_post_process_4,
                   prepared_model.activation_post_process_5]
    for act in affine_acts:
        assert act.qscheme == torch.per_tensor_affine


class ConcatBlock(nn.Module):
    def __init__(self, conv_transpose: bool, *activations: torch.nn.Module):
        super().__init__()
        self.branches = nn.ModuleList(ConvBlock(conv_transpose, act) for act in activations)

    def forward(self, x):
        return torch.cat(list(f(x) for f in self.branches))


@pytest.mark.parametrize("activation_fn", [torch.nn.ReLU, torch.nn.ReLU6, torch.nn.LeakyReLU])
@pytest.mark.parametrize("conv_transpose", [False, True])
def test_concat_uint_and_int(activation_fn, conv_transpose):
    model = ConcatBlock(conv_transpose, activation_fn(), nn.Identity())
    data = torch.randn(1, 1, 28, 28)

    prepared_model, _ = quantize_model(model, data)

    symmetric_acts = [prepared_model.activation_post_process_0, prepared_model.activation_post_process_2]
    for act in symmetric_acts:
        assert act.qscheme == torch.per_tensor_symmetric
    # these are inputs and output of cat layer, they all share same activation quantization
    other_acts = [prepared_model.activation_post_process_1, prepared_model.activation_post_process_3,
                  prepared_model.activation_post_process_4]
    for act in other_acts:
        if isinstance(activation_fn(), (torch.nn.ReLU, torch.nn.ReLU6)):
            assert act.qscheme == torch.per_tensor_affine
        else:
            assert act.qscheme == torch.per_tensor_symmetric

    assert id(prepared_model.activation_post_process_1) == id(prepared_model.activation_post_process_3)
    assert id(prepared_model.activation_post_process_3) == id(prepared_model.activation_post_process_4)


@pytest.mark.parametrize(
    "config", get_configs_for_qscheme(activation_dtype=torch.float32)
)
@pytest.mark.parametrize("activation_fn", list(_mod_activations) + [nn.ReLU])
@pytest.mark.parametrize("bn", [nn.BatchNorm2d(20), None])
@pytest.mark.parametrize("conv_transpose", [False, True])
def test_conv_weight_only_quantization(config, activation_fn, bn, conv_transpose):
    if bn is not None:
        model = nn.Sequential(
            OrderedDict(
                {
                    "layer": (
                        nn.Conv2d(1, 20, (3, 3))
                        if not conv_transpose
                        else nn.ConvTranspose2d(1, 20, (3, 3))
                    ),
                    "bn": bn,
                    "act": activation_fn(),
                }
            )
        )
    else:
        model = nn.Sequential(
            OrderedDict(
                {
                    "layer": (
                        nn.Conv2d(1, 20, (3, 3))
                        if not conv_transpose
                        else nn.ConvTranspose2d(1, 20, (3, 3))
                    ),
                    "act": activation_fn(),
                }
            )
        )
    data = torch.randn(1, 1, 28, 28)

    prepared_model, quantizer = quantize_model(model, data, config)

    if bn is not None:
        if conv_transpose:
            assert isinstance(prepared_model.layer, _qat.ConvTransposeBnAct2d)
        else:
            assert isinstance(prepared_model.layer, _qat.ConvBnAct2d) or isinstance(
                prepared_model.layer, torch.nn.intrinsic.qat.ConvBnReLU2d
            )
    else:
        if conv_transpose:
            assert isinstance(prepared_model.layer, _qat.ConvTransposeAct2d)
        else:
            assert isinstance(prepared_model.layer, _qat.ConvAct2d) or isinstance(
                prepared_model.layer, torch.nn.intrinsic.qat.ConvReLU2d
            )

    assert len(list(prepared_model.children())) == 1

    converted_model = quantizer.finalize(inplace=False)

    if conv_transpose:
        assert isinstance(converted_model.layer, _quantized.QuantizedConvTransposeAct2d)
    else:
        assert isinstance(converted_model.layer, _quantized.QuantizedConvAct2d) or isinstance(
            converted_model.layer[0], torch.ao.nn.quantized.reference.Conv2d
        )


@pytest.mark.parametrize("config", get_configs_for_qscheme(weight_dtype=torch.float32))
@pytest.mark.parametrize("activation_fn", list(_mod_activations) + [nn.ReLU])
@pytest.mark.parametrize("bn", [nn.BatchNorm2d(20), None])
def test_conv_activation_only_quantization(config, activation_fn, bn):
    if bn is not None:
        model = nn.Sequential(
            OrderedDict(
                {
                    "layer": nn.Conv2d(1, 20, (3, 3)),
                    "bn": bn,
                    "act": activation_fn(),
                }
            )
        )
    else:
        model = nn.Sequential(
            OrderedDict(
                {
                    "layer": nn.Conv2d(1, 20, (3, 3)),
                    "act": activation_fn(),
                }
            )
        )
    data = torch.randn(1, 1, 28, 28)

    prepared_model, quantizer = quantize_model(model, data, config)

    if bn is not None:
        assert isinstance(prepared_model.layer, _qat.ConvBnAct2d) or isinstance(
            prepared_model.layer, torch.nn.intrinsic.qat.ConvBnReLU2d
        )
    else:
        assert isinstance(prepared_model.layer, _qat.ConvAct2d) or isinstance(
            prepared_model.layer, torch.nn.intrinsic.qat.ConvReLU2d
        )

    assert len(list(prepared_model.children())) == 3
    assert isinstance(
        prepared_model.get_submodule("activation_post_process_0"),
        torch.ao.quantization.FakeQuantize,
    )
    assert isinstance(
        prepared_model.get_submodule("activation_post_process_1"),
        torch.ao.quantization.FakeQuantize,
    )

    converted_model = quantizer.finalize(inplace=False)

    assert isinstance(converted_model.layer, _quantized.QuantizedConvAct2d) or isinstance(
        converted_model.layer[0], torch.nn.Conv2d
    )


@pytest.mark.parametrize(
    "config", get_configs_for_qscheme(activation_dtype=torch.float32)
)
@pytest.mark.parametrize("activation_fn", list(_mod_activations) + [nn.ReLU])
def test_linear_weight_only_quantization(config, activation_fn):
    model = nn.Sequential(
        OrderedDict(
            {
                "layer": nn.Linear(20, 100),
                "act": activation_fn(),
            }
        )
    )
    data = torch.randn(1, 20)

    prepared_model, quantizer = quantize_model(model, data, config)

    assert isinstance(prepared_model.layer, _qat.LinearAct) or isinstance(
        prepared_model.layer, torch.nn.intrinsic.qat.LinearReLU
    )

    assert len(list(prepared_model.children())) == 1

    converted_model = quantizer.finalize(inplace=False)

    assert isinstance(
        converted_model.layer, _quantized.QuantizedLinearAct
    ) or isinstance(converted_model.layer[0], torch.ao.nn.quantized.reference.Linear)


@pytest.mark.parametrize("config", get_configs_for_qscheme(weight_dtype=torch.float32))
@pytest.mark.parametrize("activation_fn", list(_mod_activations) + [nn.ReLU])
def test_linear_activation_only_quantization(config, activation_fn):
    model = nn.Sequential(
        OrderedDict(
            {
                "layer": nn.Linear(20, 100),
                "act": activation_fn(),
            }
        )
    )
    data = torch.randn(1, 20)

    prepared_model, quantizer = quantize_model(model, data, config)

    assert isinstance(prepared_model.layer, _qat.LinearAct) or isinstance(
        prepared_model.layer, torch.nn.intrinsic.qat.LinearReLU
    )

    assert len(list(prepared_model.children())) == 3
    assert isinstance(
        prepared_model.get_submodule("activation_post_process_0"),
        torch.ao.quantization.FakeQuantize,
    )
    assert isinstance(
        prepared_model.get_submodule("activation_post_process_1"),
        torch.ao.quantization.FakeQuantize,
    )

    converted_model = quantizer.finalize(inplace=False)

    assert isinstance(converted_model.layer, _quantized.QuantizedLinearAct) or isinstance(
        converted_model.layer[0], torch.nn.Linear
    )


# @pytest.mark.parametrize("activation_dtype", [torch.float32, torch.quint8])
# TODO: Fix quantization of embedding layer when activation dtype is quint8
@pytest.mark.parametrize("activation_dtype", [torch.float32])
def test_embedding_layer_quantization(activation_dtype):
    model = nn.Sequential(
        OrderedDict(
            {
                "embedding": nn.Embedding(10, 10),
                "linear": nn.Linear(10, 10),
            }
        )
    )
    data = torch.randint(0, 10, (1, 10))

    configs = get_configs_for_qscheme(activation_dtype)

    for config in configs:
        prepared_model, quantizer = quantize_model(model, data, config)

        assert isinstance(prepared_model.embedding, torch.nn.qat.Embedding)

        if activation_dtype == torch.float32:
            assert len(list(prepared_model.children())) == 2
        else:
            assert len(list(prepared_model.children())) == 4
            assert prepared_model.activation_post_process_0.dtype == torch.quint8
            assert prepared_model.activation_post_process_1.dtype == torch.quint8

        if config.global_config.quantization_scheme == QuantizationScheme.symmetric:
            assert (
                prepared_model.embedding.weight_fake_quant.qscheme
                == torch.per_channel_symmetric
            )
        else:
            assert (
                prepared_model.embedding.weight_fake_quant.qscheme
                == torch.per_channel_affine
            )

        converted_model = quantizer.finalize(inplace=False)

        assert isinstance(
            converted_model.embedding, torch.ao.nn.quantized.reference.Embedding
        )
        assert isinstance(
            converted_model.linear, torch.ao.nn.quantized.reference.Linear
        )


@pytest.mark.parametrize("config", get_configs_for_qscheme())
@pytest.mark.parametrize("activation_fn", list(_mod_activations) + [nn.ReLU])
@pytest.mark.parametrize(
    "elementwise_op",
    [operator.add, torch.add, operator.mul, torch.mul, torch.matmul, torch.einsum],
)
@pytest.mark.parametrize("conv_transpose", [False, True])
def test_elementwise_op_act_fusion(config, activation_fn, elementwise_op, conv_transpose):
    class ElementWiseActModule(torch.nn.Module):
        def __init__(self, conv_transpose):
            super().__init__()
            if conv_transpose:
                self.conv1 = torch.nn.ConvTranspose2d(48, 48, (3, 3), (1, 1), padding=(1, 1))
            else:
                self.conv1 = torch.nn.Conv2d(48, 48, (3, 3), (1, 1), padding=(1, 1))
            self.act = activation_fn()

        def forward(self, x):
            if elementwise_op == torch.einsum:
                return self.act(
                    elementwise_op("bkhq,bchk->bchq", x.transpose(1, 3), self.conv1(x))
                )
            return self.act(elementwise_op(x, self.conv1(x)))

    model = ElementWiseActModule(conv_transpose)
    data = torch.randn(1, 48, 224, 224)

    prepared_model, quantizer = quantize_model(model, data, config)

    for node in prepared_model.graph.nodes:
        if node.op == "call_function":
            assert isinstance(find_module(prepared_model, node.next), activation_fn)
            assert is_activation_post_process(
                find_module(prepared_model, node.next.next)
            )


@pytest.mark.parametrize("quantization_scheme", ["symmetric", "affine"])
@pytest.mark.parametrize(
    "skipped_layers",
    [
        ["conv1", "pool1"],
        ["conv2", "pool1", "pool2"],
        ["dense1", "flatten", "dropout"],
        ["dense2", "dropout"],
        ["conv_transpose1", "pool2", "pool3"],
        ["conv_transpose2", "pool3", "flatten"],
    ],
)
def test_skipping_quantization_for_layers(
    mnist_model_quantization, quantization_scheme, skipped_layers
):
    config_s = LinearQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": quantization_scheme,
                "milestones": [0, 0, 100, 100],
            },
            "module_name_configs": {
                skipped_layer: None for skipped_layer in skipped_layers
            },
        }
    )
    config_f = LinearQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": quantization_scheme,
                "milestones": [0, 0, 100, 100],
            }
        }
    )
    data = torch.randn(1, 1, 28, 28)
    prepared_model_s, quantizer_s = quantize_model(
        mnist_model_quantization, data, config_s
    )
    prepared_model_f, quantizer_f = quantize_model(
        mnist_model_quantization, data, config_f
    )

    skipped_mod_name = skipped_layers[0]
    skipped_mod = mnist_model_quantization.get_submodule(skipped_mod_name)
    if isinstance(skipped_mod, nn.Conv2d):
        submod_s = prepared_model_s.get_submodule(skipped_mod_name)
        submod_f = prepared_model_f.get_submodule(skipped_mod_name)
        assert isinstance(submod_s, _fused.ConvBnAct2d) or isinstance(
            submod_s, _fused.ConvAct2d
        )
        assert not hasattr(submod_s.conv, "weight_fake_quant")
        assert isinstance(submod_f, _qat.ConvBnAct2d) or isinstance(
            submod_f, _qat.ConvAct2d
        )
        assert hasattr(submod_f.conv, "weight_fake_quant")
    elif isinstance(skipped_mod, nn.Linear):
        submod_s = prepared_model_s.get_submodule(skipped_mod_name)
        submod_f = prepared_model_f.get_submodule(skipped_mod_name)
        assert isinstance(submod_s, _fused.LinearAct)
        assert not hasattr(submod_s.linear, "weight_fake_quant")
        assert isinstance(submod_f, _qat.LinearAct)
        assert hasattr(submod_f.linear, "weight_fake_quant")

    for node in prepared_model_s.graph.nodes:
        if node.target == skipped_mod_name:
            for consumer in node.users:
                assert "activation_post_process" not in consumer.target
            for producer in node.args:
                assert "activation_post_process" not in producer.target

    for node in prepared_model_f.graph.nodes:
        if node.target == skipped_mod_name:
            for consumer in node.users:
                assert "activation_post_process" in consumer.target
            for producer in node.args:
                if producer.target != "dropout":
                    # for some nodes, if producer is dropout, we won't have activation post process
                    assert "activation_post_process" in producer.target


class MySoftmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Reduction max
        max_x = x.max(dim=self.dim, keepdim=True).values
        # EW sub
        x -= max_x
        # Scale for EXP to EXP2, Activation EXP2
        scaled_x = x * (1 / math.log(2))
        exp_act = torch.exp2(scaled_x)
        # Reduction Sum + Inv
        exp_sum_inv = 1 / exp_act.sum(dim=self.dim, keepdims=True)
        # EW Mult
        return exp_act * exp_sum_inv


def test_softmax_breakdown():
    model = MySoftmax(1)
    input = torch.rand(2, 77, 1, 4096)

    quantizer = LinearQuantizer(model, LinearQuantizerConfig())
    prepared_model = quantizer.prepare(example_inputs=input)
    nodes = list(prepared_model.graph.nodes)
    """
    Ensure fake quantize layers are inserted at the right places in the graph

    # Reduction max,
    # q / dq
    # EW sub,
    # q / dq
    # Scale for EXP to EXP2, Activation EXP2,
    # q / dq
    # Reduction Sum + Inv,
    # q / dq
    # EW Mult
    """
    assert nodes[0].name == "x"
    assert nodes[1].name == "activation_post_process_0"
    assert nodes[2].name == "max_1"
    assert nodes[3].name == "getattr_1"
    assert nodes[4].name == "activation_post_process_1"
    assert nodes[5].name == "sub"
    assert nodes[6].name == "activation_post_process_2"
    assert nodes[7].name == "mul"
    assert nodes[8].name == "exp2"
    assert nodes[9].name == "activation_post_process_3"
    assert nodes[10].name == "sum_1"
    assert nodes[11].name == "truediv"
    assert nodes[12].name == "activation_post_process_4"
    assert nodes[13].name == "mul_1"
    assert nodes[14].name == "activation_post_process_5"
    assert nodes[15].name == "output"
