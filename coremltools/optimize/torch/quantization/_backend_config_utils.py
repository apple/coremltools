#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy as _deepcopy
from typing import Any as _Any
from typing import Callable as _Callable
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Type as _Type

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
from torch.ao.quantization.backend_config import BackendPatternConfig as _BackendPatternConfig
from torch.ao.quantization.backend_config import DTypeConfig as _DTypeConfig
from torch.ao.quantization.backend_config import DTypeWithConstraints as _DTypeWithConstraints
from torch.ao.quantization.backend_config import ObservationType as _ObservationType

from coremltools.optimize.torch._utils.version_utils import is_torch_2 as _is_torch_2

act_quant_dtype_configs = [
    # int input and output
    _DTypeConfig(
        input_dtype=_torch.quint8,
        output_dtype=_torch.quint8,
    ),
    # int input, float output
    _DTypeConfig(
        input_dtype=_torch.quint8,
        output_dtype=_torch.float,
    ),
    # float input, int output
    _DTypeConfig(
        input_dtype=_torch.float,
        output_dtype=_torch.quint8,
    ),
]


weighted_dtype_configs = [
    # weight int, act float, weight dtype signed
    _DTypeConfig(
        input_dtype=_torch.float,
        output_dtype=_torch.float,
        weight_dtype=_torch.qint8,
        bias_dtype=_torch.float,
    ),
    # weight int, act float, weight dtype unsigned
    _DTypeConfig(
        input_dtype=_torch.float,
        output_dtype=_torch.float,
        weight_dtype=_torch.quint8,
        bias_dtype=_torch.float,
    ),
    # weight int, act int, weight dtype signed
    _DTypeConfig(
        input_dtype=_torch.quint8,
        output_dtype=_torch.quint8,
        weight_dtype=_torch.qint8,
        bias_dtype=_torch.float,
    ),
    # weight int, act int, weight dtype unsigned
    _DTypeConfig(
        input_dtype=_torch.quint8,
        output_dtype=_torch.quint8,
        weight_dtype=_torch.quint8,
        bias_dtype=_torch.float,
    ),
]


def get_fuser_method(constructor):
    """
    Creates fuser method from class constructor of fused modules.
    """
    if _is_torch_2():

        def fuser_method(is_qat, m1, m2):
            if isinstance(m1, tuple):
                m0, m1 = m1
                return constructor(m1, m0, m2)
            return constructor(m1, m2)

    else:

        def fuser_method(is_qat, m1, m2):
            if isinstance(m2, tuple):
                m2, m3 = m2
                return constructor(m3, m2, m1)
            return constructor(m2, m1)

    return fuser_method


def get_fusion_pattern(pattern: _Tuple[_Any, _Any]) -> _Tuple[_Any, _Any]:
    """
    Swaps fusion pattern if torch version is >= 2.0.
    """
    if _is_torch_2():
        return pattern[1], pattern[0]
    else:
        return pattern


def fused_mod_config(
    mod: _Type[_nn.Module],
    fused_mod: _Type[_nn.Module],
    qat_mod: _Type[_nn.Module],
    ref_quant_mod: _Type[_nn.Module],
    input_output_observed: _Optional[bool] = None,
) -> _BackendPatternConfig:
    """
    Returns backend pattern config for fused modules.
    """
    config = (
        _BackendPatternConfig(fused_mod)
        .set_observation_type(_ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs(weighted_dtype_configs)
        .set_root_module(mod)
        .set_qat_module(qat_mod)
        .set_reference_quantized_module(ref_quant_mod)
    )
    if input_output_observed is not None:
        if _is_torch_2():
            config.set_observation_type(_ObservationType.INPUT_OUTPUT_NOT_OBSERVED)
        else:
            config._input_output_observed = False
    return config


def qat_mod_config(
    mod: _Type[_nn.Module], qat_mod: _Type[_nn.Module], ref_quant_mod: _Type[_nn.Module]
) -> _BackendPatternConfig:
    """
    Returns backend pattern config for QAT modules.
    """
    return (
        _BackendPatternConfig(qat_mod)
        .set_observation_type(_ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs(weighted_dtype_configs)
        .set_root_module(mod)
        .set_reference_quantized_module(ref_quant_mod)
    )


def weighted_configs(
    mod: _Type[_nn.Module],
    func_mod: _Optional[_Callable],
    qat_mod: _Type[_nn.Module],
    ref_quant_mod: _Type[_nn.Module],
    input_output_observed: _Optional[bool] = None,
) -> _List[_BackendPatternConfig]:
    """
    Returns backend pattern configs for modules which have a weight associated with them,
    such as convolution, linear, embedding, etc.
    """
    configs = [
        # conv/linear module
        fused_mod_config(
            mod=mod,
            fused_mod=mod,
            qat_mod=qat_mod,
            ref_quant_mod=ref_quant_mod,
            input_output_observed=input_output_observed,
        ),
        # qat conv/linear
        qat_mod_config(mod=mod, qat_mod=qat_mod, ref_quant_mod=ref_quant_mod),
    ]
    if func_mod is not None:
        configs += [
            # functional
            _BackendPatternConfig(func_mod)
            .set_observation_type(_ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
            .set_dtype_configs(weighted_dtype_configs)
            ._set_input_type_to_index({"weight": 1, "bias": 2}),
        ]
    return configs


def weighted_relu_configs(
    mod: _Type[_nn.Module],
    func_mod: _Callable,
    fused_mod: _Type[_nn.Module],
    ref_quant_mod: _Type[_nn.Module],
    qat_mod: _Type[_nn.Module],
) -> _List[_BackendPatternConfig]:
    """
    Returns backend pattern configs for the following sequence of ops:

    input -> mod -> relu -> output

    where mod is a module with a weight associated with it, such as convolution and linear.
    """
    return [
        # conv/linear module + relu func/module
        *[
            _BackendPatternConfig(get_fusion_pattern((act, mod)))
            .set_dtype_configs(weighted_dtype_configs)
            .set_fuser_method(get_fuser_method(fused_mod))
            .set_fused_module(fused_mod)
            for act in [_nn.ReLU, _F.relu]
        ],
        # conv/linear func + relu func/module
        *[
            _BackendPatternConfig(get_fusion_pattern((act, func_mod)))
            .set_observation_type(_ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
            .set_dtype_configs(weighted_dtype_configs)
            for act in [_nn.ReLU, _F.relu]
        ],
        # conv/linear + relu fused
        fused_mod_config(
            mod=mod, fused_mod=fused_mod, qat_mod=qat_mod, ref_quant_mod=ref_quant_mod
        ),
        # qat conv/linear + relu
        qat_mod_config(mod=mod, qat_mod=qat_mod, ref_quant_mod=ref_quant_mod),
    ]


def weighted_act_configs(
    mod: _Type[_nn.Module],
    func_mod: _Callable,
    act: _Type[_nn.Module],
    fused_mod: _Type[_nn.Module],
    qat_mod: _Type[_nn.Module],
    ref_quant_mod: _Type[_nn.Module],
) -> _List[_BackendPatternConfig]:
    """
    Returns backend pattern configs for the following sequence of ops:

    input -> mod -> activation -> output

    where mod is a module with a weight associated with it, such as convolution and linear.
    """
    return [
        # conv/linear module + act module
        _BackendPatternConfig(get_fusion_pattern((act, mod)))
        .set_dtype_configs(weighted_dtype_configs)
        .set_fuser_method(get_fuser_method(fused_mod))
        .set_fused_module(fused_mod),
        # conv/linear func + act module
        _BackendPatternConfig(get_fusion_pattern((act, func_mod)))
        .set_observation_type(_ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs(weighted_dtype_configs),
        # conv/linear + act fused
        fused_mod_config(
            mod=fused_mod,
            fused_mod=fused_mod,
            qat_mod=qat_mod,
            ref_quant_mod=ref_quant_mod,
        ),
        # qat conv/linear + act
        qat_mod_config(mod=fused_mod, qat_mod=qat_mod, ref_quant_mod=ref_quant_mod),
    ]


def weighted_bn_configs(
    mod: _Type[_nn.Module],
    bn_mod: _Type[_nn.Module],
    fused_mod: _Type[_nn.Module],
    ref_quant_mod: _Type[_nn.Module],
    qat_mod: _Type[_nn.Module],
) -> _List[_BackendPatternConfig]:
    """
    Returns backend pattern configs for the following sequence of ops:

    input -> mod -> batch_norm -> output

    where mod is a module with a weight associated with it, such as convolution and linear.
    """
    return [
        # conv module + bn module
        _BackendPatternConfig(get_fusion_pattern((bn_mod, mod)))
        .set_dtype_configs(weighted_dtype_configs)
        .set_fuser_method(get_fuser_method(fused_mod))
        .set_fused_module(fused_mod),
        # conv + bn fused
        fused_mod_config(
            mod=mod, fused_mod=fused_mod, qat_mod=qat_mod, ref_quant_mod=ref_quant_mod
        ),
        # qat conv + bn
        qat_mod_config(mod=mod, qat_mod=qat_mod, ref_quant_mod=ref_quant_mod),
    ]


def weighted_bn_relu_configs(
    mod: _Type[_nn.Module],
    bn_mod: _Type[_nn.Module],
    fused_mod: _Type[_nn.Module],
    ref_quant_mod: _Type[_nn.Module],
    qat_mod: _Type[_nn.Module],
) -> _List[_BackendPatternConfig]:
    """
    Returns backend pattern configs for the following sequence of ops:

    input -> mod -> batch_norm -> relu -> output

    where mod is a module with a weight associated with it, such as convolution and linear.
    """
    return [
        # conv module + bn module + relu func/module
        *[
            _BackendPatternConfig(get_fusion_pattern((act, (bn_mod, mod))))
            .set_dtype_configs(weighted_dtype_configs)
            .set_fuser_method(get_fuser_method(fused_mod))
            .set_fused_module(fused_mod)
            for act in [_nn.ReLU, _F.relu]
        ],
        # conv + bn + relu fused
        fused_mod_config(
            mod=mod, fused_mod=fused_mod, qat_mod=qat_mod, ref_quant_mod=ref_quant_mod
        ),
        # qat conv + bn + relu
        qat_mod_config(mod=mod, qat_mod=qat_mod, ref_quant_mod=ref_quant_mod),
    ]


def weighted_bn_act_configs(
    mod: _Type[_nn.Module],
    act: _Type[_nn.Module],
    bn_mod: _Type[_nn.Module],
    root_mod: _Type[_nn.Module],
    fused_mod: _Type[_nn.Module],
    ref_quant_mod: _Type[_nn.Module],
    qat_mod: _Type[_nn.Module],
) -> _List[_BackendPatternConfig]:
    """
    Returns backend pattern configs for the following sequence of ops:

    input -> mod -> batch_norm -> activation -> output

    where mod is a module with a weight associated with it, such as convolution and linear.
    """
    return [
        # conv module + bn module + act module
        _BackendPatternConfig(get_fusion_pattern((act, (bn_mod, mod))))
        .set_dtype_configs(weighted_dtype_configs)
        .set_fuser_method(get_fuser_method(fused_mod))
        .set_fused_module(fused_mod),
        # conv + bn + act fused
        fused_mod_config(
            mod=root_mod,
            fused_mod=fused_mod,
            qat_mod=qat_mod,
            ref_quant_mod=ref_quant_mod,
        ),
        # qat conv + bn + act
        qat_mod_config(mod=root_mod, qat_mod=qat_mod, ref_quant_mod=ref_quant_mod),
    ]


def binary_op_configs(ops: _List[_Any]) -> _List[_BackendPatternConfig]:
    """
    Returns backend pattern configs for the following sequence of ops:

    input_1 -->
                  operator --> output
    input_2 -->

    where operator is a binary operator such as add, multiply or matmul.
    """
    return [
        _BackendPatternConfig(op)
        .set_dtype_configs(act_quant_dtype_configs)
        .set_observation_type(_ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        for op in ops
    ]


def binary_op_act_configs(ops: _List[_Any], acts: _List[_Any]) -> _List[_BackendPatternConfig]:
    """
    Returns backend pattern configs for the following sequence of ops:

    input_1 -->
                  operator --> act --> output
    input_2 -->

    where operator is a binary operator such as add or multiply.
    """
    configs = []
    for op in ops:
        configs.extend(
            [
                _BackendPatternConfig(get_fusion_pattern((act, op)))
                .set_dtype_configs(act_quant_dtype_configs)
                .set_observation_type(_ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
                for act in acts
            ]
        )
    return configs


def share_observer_configs(ops: _List[_Any]) -> _List[_BackendPatternConfig]:
    """
    Returns backend pattern configs for ops which do not change the scale or
    zero-point of the input tensor and thus can share the same qparams.
    """
    return [
        _BackendPatternConfig(op)
        .set_observation_type(_ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs(act_quant_dtype_configs)
        for op in ops
    ]


def activation_configs(
    ops: _List[_Any], constraints: _Optional[_DTypeWithConstraints] = None
) -> _List[_BackendPatternConfig]:
    """
    Returns backend pattern configs for default ops like activations which
    do not have an associated weight but can alter the scale and zero point of
    the input tensor.
    """
    dtype_configs = []
    for act_dtype in act_quant_dtype_configs:
        new_act_dtype = _deepcopy(act_dtype)
        if act_dtype.output_dtype == _torch.quint8 and constraints is not None:
            new_act_dtype.output_dtype_with_constraints = constraints
        dtype_configs.append(new_act_dtype)
    return [
        _BackendPatternConfig(op)
        .set_observation_type(_ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs(dtype_configs)
        for op in ops
    ]


def bn_relu(
    mod: _Type[_nn.Module],
    fused_mod: _Type[_nn.Module],
) -> _List[_BackendPatternConfig]:
    """
    Returns backend pattern configs for the following sequence of ops:

    input -> batch_norm -> relu -> output
    """
    return [
        # bn module + relu func/module
        *[
            _BackendPatternConfig(get_fusion_pattern((act, mod)))
            .set_dtype_configs(weighted_dtype_configs)
            .set_fuser_method(get_fuser_method(fused_mod))
            .set_fused_module(fused_mod)
            for act in [_nn.ReLU, _F.relu]
        ]
    ] + activation_configs(
        ops=[fused_mod]
    )  # fused bn + relu
