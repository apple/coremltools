#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import math
import operator as _operator
from collections import defaultdict
from enum import Enum as _Enum
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import torch as _torch
import torch.ao.quantization as _aoquant
import torch.fx as _fx
from torch.ao.nn.quantized.reference.modules.utils import _quantize_and_dequantize_weight_decomposed
from torch.ao.quantization.backend_config import BackendConfig as _BackendConfig
from torch.ao.quantization.backend_config import ObservationType as _ObservationType

from coremltools.optimize.torch._utils.metadata_utils import (
    CompressionMetadata as _CompressionMetadata,
)
from coremltools.optimize.torch._utils.version_utils import is_torch_2 as _is_torch_2


def is_per_channel_quant(qscheme: _torch.qscheme) -> bool:
    """
    Returns True if provided qscheme is for per-channel quantization. Otherwise returns False.
    """
    return qscheme in [_torch.per_channel_symmetric, _torch.per_channel_affine]


def is_symmetric_quant(qscheme: _torch.qscheme) -> bool:
    """
    Returns True if provided qscheme is for symmetric quantization. Otherwise returns False.
    """
    return qscheme in [_torch.per_tensor_symmetric, _torch.per_channel_symmetric]


def is_pytorch_defined_observer(observer: _aoquant.ObserverBase):
    """
    Returns True if provided observer instance is defined by PyTorch. Otherwise returns False.
    """

    checklist = (
        _aoquant.MinMaxObserver,
        _aoquant.PerChannelMinMaxObserver,
        _aoquant.MovingAverageMinMaxObserver,
        _aoquant.MovingAveragePerChannelMinMaxObserver,
        _aoquant.HistogramObserver,
        _aoquant.PlaceholderObserver,
        _aoquant.NoopObserver,
        _aoquant.FixedQParamsObserver,
    )
    return isinstance(observer, checklist)


class CombinationOpType(_Enum):
    Add = "add"
    Mul = "mul"
    Concat = "concat"
    AddReLU = "add_relu"


def find_target(model, target_name):
    """
    Finds the module in model which is referenced by the target_name.
    target_name is in the form of `mod_a.mod_b.mod_c`
    """
    current_obj = model
    for attr in target_name.split("."):
        current_obj = getattr(current_obj, attr)
    return current_obj


def find_module(model: _torch.nn.Module, node: _fx.Node):
    """
    Finds module corresponding to the node.
    """
    if hasattr(node, "op") and node.op == "call_module":
        return find_target(model, node.target)
    return None


def is_add(node: _fx.Node):
    """
    Returns True if node is an add op
    """
    if node.op == "call_function":
        return node.target == _operator.add or node.target == _torch.add
    return False


def is_mul(node: _fx.Node):
    """
    Returns True if node is a mul op
    """
    if node.op == "call_function":
        return node.target == _operator.mul or node.target == _torch.mul
    return False


def is_concat(node: _fx.Node):
    """
    Returns True if node is a concat op
    """
    if node.op == "call_function":
        return node.target == _torch.cat
    return False


def is_relu(node: _fx.Node) -> bool:
    """
    Returns True if node is a relu op
    """
    if node.op == "call_function":
        return node.target == _torch.nn.functional.relu
    return False


def is_add_relu(node: _fx.Node) -> bool:
    """
    Returns True if node is a add-relu op
    """
    return is_relu(node) and len(node.args) == 1 and is_add(node.args[0])


def combine_op_type(node: _fx.Node) -> _Optional[CombinationOpType]:
    """
    Returns type of combination op at this node -> add, mul, add-relu or concat
    """
    if is_add(node):
        return CombinationOpType.Add
    elif is_mul(node):
        return CombinationOpType.Mul
    elif is_add_relu(node):
        return CombinationOpType.AddReLU
    elif is_concat(node):
        return CombinationOpType.Concat
    return None


def is_activation_post_process(module: _torch.nn.Module) -> bool:
    """
    Returns true if a module is an activation post process module.
    """
    return isinstance(module, _aoquant.FakeQuantizeBase)


def is_quantized(module: _aoquant.FakeQuantizeBase):
    """
    Returns true if activation post process module uses integer dtypes.
    """
    if hasattr(module, "activation_post_process"):
        return module.activation_post_process.dtype in [_torch.qint8, _torch.quint8]
    return False


def group_activation_quantization_modules_by_id(
    model: _fx.GraphModule,
) -> _Dict[int, _List[_fx.Node]]:
    """
    Groups activation post process layers by their ids. This is useful
    because multiple activation post process modules in a traced graph may
    point to the same module.
    """
    groups = defaultdict(list)
    for node in model.graph.nodes:
        if node.op == "call_module":
            module = find_target(model, node.target)
            if is_activation_post_process(module) and is_quantized(module):
                groups[id(module)].append(node)
    return groups


def get_share_qparams_ops(backend_config: _BackendConfig):
    """
    Returns list of ops which share qparams with input.
    """

    configs = (
        backend_config._pattern_complex_format_to_config
        if _is_torch_2()
        else backend_config.configs
    )

    return [
        op
        for op in configs
        if configs[op].observation_type == _ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    ]


def get_quant_range(n_bits: int, dtype: _torch.dtype) -> _Tuple[int, int]:
    """
    Returns quant_max and quant_min values for a given quantization n_bits.
    """
    max_q = 2**n_bits
    if dtype in [_torch.quint8, _torch.uint8]:
        quant_min = 0
        quant_max = max_q - 1
    else:
        quant_min = -max_q / 2
        quant_max = max_q / 2 - 1
    return int(quant_min), int(quant_max)


def get_n_bits_from_range(quant_min: int, quant_max: int) -> int:
    """
    Returns quantization n_bits for given quantization range
    """
    n_bits = int(math.log2(quant_max + 1))
    if quant_min < 0:
        n_bits += 1

    return n_bits


def register_compression_metadata(submodule):
    metadata = _CompressionMetadata("weight")
    metadata.compression_type = ["quantization"]
    metadata.quantization_n_bits = get_n_bits_from_range(
        submodule.weight_quant_min, submodule.weight_quant_max
    )
    metadata.quantization_scale = (
        submodule.weight_scale.detach().clone().unsqueeze(-1)
        if submodule.weight_axis == 0
        else submodule.weight_scale.detach().clone().unsqueeze(-1).transpose(1, 0)
    )
    metadata.zero_point = (
        submodule.weight_zero_point.detach().clone().unsqueeze(-1)
        if submodule.weight_axis == 0
        else submodule.weight_zero_point.detach().clone().unsqueeze(-1).transpose(1, 0)
    )
    metadata.register(submodule)


def pre_apply_weight_quant(model: _torch.nn.Module):
    for module in model.modules():
        if isinstance(
            module,
            _torch.ao.nn.quantized.reference.modules.utils.ReferenceQuantizedModule,
        ):
            weight = _quantize_and_dequantize_weight_decomposed(
                module.weight,
                module.weight_qscheme,
                module.weight_dtype,
                module.weight_scale,
                module.weight_zero_point,
                module.weight_axis_int,
                module.weight_quant_min,
                module.weight_quant_max,
            )
            module.weight.detach().copy_(weight)
