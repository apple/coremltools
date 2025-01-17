#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from collections import OrderedDict
from typing import Dict, Optional

import pytest
import torch
import torch.nn as nn

from torch.fx import Node

from coremltools.optimize.torch.quantization.quantization_config import (
    LinearQuantizerConfig,
    QuantizationScheme,
)
from coremltools._deps import _HAS_TORCH_EXPORT_API
if _HAS_TORCH_EXPORT_API:
    from torch._export import capture_pre_autograd_graph
    from torch.ao.quantization.quantize_pt2e import (
        convert_pt2e,
        prepare_pt2e,
        prepare_qat_pt2e,
    )

_TORCH_VERSION = torch.__version__
_EXPECTED_TORCH_VERSION = '2.2.0'
if _TORCH_VERSION >= _EXPECTED_TORCH_VERSION:
    from coremltools.optimize.torch.quantization._coreml_quantizer import CoreMLQuantizer


activations = {
    nn.ReLU: {
        True: torch.ops.aten.relu_.default,
        False: torch.ops.aten.relu.default,
    },
    nn.ReLU6: {
        True: torch.ops.aten.hardtanh_.default,
        False: torch.ops.aten.hardtanh.default,
    },
    nn.LeakyReLU: {
        True: torch.ops.aten.leaky_relu_.default,
        False: torch.ops.aten.leaky_relu.default,
    },
    nn.SiLU: {
        True: torch.ops.aten.silu_.default,
        False: torch.ops.aten.silu.default,
    },
    nn.ELU: {
        True: torch.ops.aten.elu_.default,
        False: torch.ops.aten.elu.default,
    },
    nn.CELU: {
        True: torch.ops.aten.celu_.default,
        False: torch.ops.aten.celu.default,
    },
    nn.SELU: {
        True: torch.ops.aten.selu_.default,
        False: torch.ops.aten.selu.default,
    },
    nn.Mish: {
        True: torch.ops.aten.mish_.default,
        False: torch.ops.aten.mish.default,
    },
    nn.Hardtanh: {
        True: torch.ops.aten.hardtanh_.default,
        False: torch.ops.aten.hardtanh.default,
    },
    nn.Hardswish: {
        True: torch.ops.aten.hardswish_.default,
        False: torch.ops.aten.hardswish.default,
    },
    nn.Hardsigmoid: {
        True: torch.ops.aten.hardsigmoid_.default,
        False: torch.ops.aten.hardsigmoid.default,
    },
    nn.GELU: {
        False: torch.ops.aten.gelu.default,
    },
    nn.Sigmoid: {
        False: torch.ops.aten.sigmoid.default,
    },
    nn.LogSigmoid: {
        False: torch.ops.aten.log_sigmoid.default,
    },
    nn.Tanh: {
        False: torch.ops.aten.tanh.default,
    },
}


@pytest.fixture(scope="module")
def model_for_quant() -> torch.nn.Module:
    model_dict = OrderedDict()
    activation_dict = {}
    idx = 0
    start_idx = idx
    for act_fn in activations:
        for inplace in activations[act_fn].keys():
            inp_channels = 1 if idx == start_idx else 20
            model_dict[f"conv_{idx}"] = torch.nn.Conv2d(
                inp_channels, 20, (3, 3), padding=(1, 1)
            )
            model_dict[f"act_{idx}"] = act_fn(inplace=inplace) if inplace else act_fn()
            activation_dict[idx] = activations[act_fn][inplace]
            idx += 1
            model_dict[f"conv_{idx}"] = torch.nn.Conv2d(20, 20, (3, 3), padding=(1, 1))
            model_dict[f"bn_{idx}"] = nn.BatchNorm2d(20)
            model_dict[f"act_{idx}"] = act_fn(inplace=inplace) if inplace else act_fn()
            activation_dict[idx] = activations[act_fn][inplace]
            idx += 1

    model_dict["flatten"] = torch.nn.Flatten(start_dim=2)
    start_idx = idx
    for act_fn in activations:
        for inplace in activations[act_fn].keys():
            inp_channels = 784 if idx == start_idx else 20
            model_dict[f"lin_{idx}"] = nn.Linear(inp_channels, 20)
            model_dict[f"act_{idx}"] = act_fn(inplace=inplace) if inplace else act_fn()
            activation_dict[idx] = activations[act_fn][inplace]
            idx += 1
            model_dict[f"lin_{idx}"] = nn.Linear(20, 20)
            model_dict[f"bn_{idx}"] = nn.BatchNorm1d(20)
            model_dict[f"act_{idx}"] = act_fn(inplace=inplace) if inplace else act_fn()
            activation_dict[idx] = activations[act_fn][inplace]
            idx += 1
    return nn.Sequential(model_dict)


def get_node_map(model: torch.fx.GraphModule) -> Dict[str, Node]:
    """
    Return a dictionary of node name to node
    """
    node_map = {}
    for node in model.graph.nodes:
        node_map[node.name] = node
    return node_map


@pytest.fixture(scope="module")
def config(request) -> LinearQuantizerConfig:
    quantization_scheme, weight_per_channel, activation_dtype = request.param
    return LinearQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": quantization_scheme,
                "milestones": [0, 0, 10, 10],
                "activation_dtype": activation_dtype,
                "weight_dtype": torch.qint8,
                "weight_per_channel": weight_per_channel,
            }
        }
    )


def quantize_model(
    model: nn.Module,
    data: torch.Tensor,
    quantization_config: Optional[LinearQuantizerConfig] = None,
    is_qat: bool = True,
):
    quantizer = CoreMLQuantizer(quantization_config)
    exported_model = capture_pre_autograd_graph(model, (data,))
    if is_qat:
        prepared_model = prepare_qat_pt2e(exported_model, quantizer)
    else:
        prepared_model = prepare_pt2e(exported_model, quantizer)
    prepared_model(data)
    converted_model = convert_pt2e(prepared_model, use_reference_representation=False)
    return converted_model


@pytest.mark.parametrize(
    "config",
    [
        (QuantizationScheme.symmetric, True, torch.quint8),
        (QuantizationScheme.symmetric, True, torch.float32),
    ],
    indirect=True,
)
@pytest.mark.parametrize("is_qat", [True, False])
@pytest.mark.skipif(not _HAS_TORCH_EXPORT_API or _TORCH_VERSION < _EXPECTED_TORCH_VERSION,
                    reason="This test requires PyTorch Export APIs and PyTorch >= 2.2.0.")
def test_weight_module_act_fusion(model_for_quant, is_qat, config):
    model = model_for_quant
    data = torch.randn(2, 1, 28, 28)
    converted_model = quantize_model(model, data, config, is_qat=is_qat)

    node_map = get_node_map(converted_model)
    mod_nodes = [torch.ops.aten.conv2d.default, torch.ops.aten.linear.default]
    activation_dtype = config.global_config.activation_dtype

    for node_name, node in node_map.items():
        if node.target in mod_nodes:
            if activation_dtype == torch.float32:
                assert (
                    node.args[0].target
                    != torch.ops.quantized_decomposed.dequantize_per_tensor.default
                )
            else:
                assert (
                    node.args[0].target
                    == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                )

            assert (
                node.args[1].target
                == torch.ops.quantized_decomposed.dequantize_per_channel.default
            )
            assert len(node.users) == 1
            act_node = list(node.users.keys())[0]
            if act_node.target == torch.ops.aten._native_batch_norm_legit.default:
                act_node = act_node.next.next
            assert len(act_node.users) == 1
            post_act_node = list(act_node.users.keys())[0]
            if activation_dtype == torch.float32:
                assert (
                    post_act_node.target
                    != torch.ops.quantized_decomposed.quantize_per_tensor.default
                )
            else:
                assert (
                    post_act_node.target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                )
    # necessary to clear cache, otherwise tests fail with cache_size_limit reached
    torch._dynamo.reset()
