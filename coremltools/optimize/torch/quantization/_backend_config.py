#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import operator as _operator
from typing import Any as _Any
from typing import List as _List
from typing import Set as _Set

import torch as _torch
import torch.ao.nn.qat as _nnq
import torch.ao.nn.quantized.reference as _nnr
import torch.nn as _nn
import torch.nn.functional as _F
import torch.nn.intrinsic as _nni
import torch.nn.intrinsic.qat as _nniq
from torch.ao.quantization.backend_config import BackendConfig as _BackendConfig
from torch.ao.quantization.backend_config import BackendPatternConfig as _BackendPatternConfig
from torch.ao.quantization.backend_config import DTypeWithConstraints as _DTypeWithConstraints

import coremltools.optimize.torch.quantization.modules.conv_transpose as _qconv_transpose
import coremltools.optimize.torch.quantization.modules.conv_transpose_fused as _qconv_transpose_fused
import coremltools.optimize.torch.quantization.modules.fused_modules as _fused
import coremltools.optimize.torch.quantization.modules.qat_modules as _qat
import coremltools.optimize.torch.quantization.modules.quantized_modules as _quantized
from coremltools.optimize.torch._utils.version_utils import is_torch_2 as _is_torch_2
from coremltools.optimize.torch.quantization._backend_config_utils import (
    activation_configs as _activation_configs,
)
from coremltools.optimize.torch.quantization._backend_config_utils import (
    binary_op_act_configs as _binary_op_relu_configs,
)
from coremltools.optimize.torch.quantization._backend_config_utils import (
    binary_op_configs as _binary_op_configs,
)
from coremltools.optimize.torch.quantization._backend_config_utils import bn_relu as _bn_relu
from coremltools.optimize.torch.quantization._backend_config_utils import (
    share_observer_configs as _share_observer_configs,
)
from coremltools.optimize.torch.quantization._backend_config_utils import (
    weighted_act_configs as _weighted_act_configs,
)
from coremltools.optimize.torch.quantization._backend_config_utils import (
    weighted_bn_act_configs as _weighted_bn_act_configs,
)
from coremltools.optimize.torch.quantization._backend_config_utils import (
    weighted_bn_configs as _weighted_bn_configs,
)
from coremltools.optimize.torch.quantization._backend_config_utils import (
    weighted_bn_relu_configs as _weighted_bn_relu_configs,
)
from coremltools.optimize.torch.quantization._backend_config_utils import (
    weighted_configs as _weighted_configs,
)
from coremltools.optimize.torch.quantization._backend_config_utils import (
    weighted_relu_configs as _weighted_relu_configs,
)

# module based activations
_mod_activations = (
    _nn.PReLU,
    _nn.RReLU,
    _nn.ReLU6,
    _nn.LeakyReLU,
    _nn.Sigmoid,
    _nn.LogSigmoid,
    _nn.Hardsigmoid,
    _nn.SiLU,
    _nn.ELU,
    _nn.CELU,
    _nn.SELU,
    _nn.GLU,
    _nn.Mish,
    _nn.GELU,
    _nn.Tanh,
    _nn.Hardtanh,
    _nn.Softmax,
    _nn.LogSoftmax,
    _nn.Hardswish,
)

# functional activations
_func_activations = (
    _F.prelu,
    _F.rrelu_,
    _F.rrelu,
    _F.relu6,
    _F.leaky_relu,
    _F.leaky_relu_,
    _F.logsigmoid,
    _F.silu,
    _F.elu,
    _F.elu_,
    _F.celu,
    _F.celu_,
    _F.selu,
    _F.selu_,
    _F.glu,
    _F.mish,
    _F.gelu,
    _F.hardtanh,
    _F.hardtanh_,
    _F.log_softmax,
    _F.hardswish,
)

# ReLU activations
_relu_activations = (
    _nn.ReLU,
    _F.relu,
)

# layers which have a fixed output range and hence use fixed qparams
_fixed_qparams_modules = {
    _torch.nn.Hardsigmoid,
    _torch.nn.functional.hardsigmoid,
    "hardsigmoid",
    "hardsigmoid_",
    _torch.nn.Sigmoid,
    _torch.sigmoid,
    "sigmoid",
    "sigmoid_",
    _torch.nn.Softmax,
    _torch.nn.Tanh,
    _torch.tanh,
    "tanh",
    "tanh_",
}


class _BackendConfigRegistry:
    """
    A registry of quantization patterns.
    """

    backend_config: _BackendConfig = _BackendConfig()
    supported_modules: _Set[_Any] = set()

    @classmethod
    def register(cls):
        def inner_wrapper(wrapped_fn):
            backend_pattern_configs: _List[_BackendPatternConfig] = wrapped_fn()
            for config in backend_pattern_configs:
                if not isinstance(config.pattern, tuple):
                    cls.supported_modules.add(config.pattern)
            cls.backend_config.set_backend_pattern_configs(backend_pattern_configs)
            return wrapped_fn

        return inner_wrapper


@_BackendConfigRegistry.register()
def _conv1d_act() -> _List[_BackendPatternConfig]:
    """
    float: Conv1d -> Act
    qat: FakeQuant -> qat.ConvAct1d -> FakeQuant
    """
    configs = _weighted_relu_configs(
        mod=_nn.Conv1d,
        func_mod=_F.conv1d,
        fused_mod=_nni.ConvReLU1d,
        qat_mod=_nniq.ConvReLU1d,
        ref_quant_mod=_nnr.Conv1d,
    )
    for act in _mod_activations:
        configs.extend(
            _weighted_act_configs(
                mod=_nn.Conv1d,
                func_mod=_F.conv1d,
                act=act,
                fused_mod=_fused.ConvAct1d,
                qat_mod=_qat.ConvAct1d,
                ref_quant_mod=_quantized.QuantizedConvAct1d,
            )
        )
    return configs


@_BackendConfigRegistry.register()
def _conv2d_act() -> _List[_BackendPatternConfig]:
    """
    float: Conv2d -> Act
    qat: FakeQuant -> qat.ConvAct2d -> FakeQuant
    """
    configs = _weighted_relu_configs(
        mod=_nn.Conv2d,
        func_mod=_F.conv2d,
        fused_mod=_nni.ConvReLU2d,
        qat_mod=_nniq.ConvReLU2d,
        ref_quant_mod=_nnr.Conv2d,
    )
    for act in _mod_activations:
        configs.extend(
            _weighted_act_configs(
                mod=_nn.Conv2d,
                func_mod=_F.conv2d,
                act=act,
                fused_mod=_fused.ConvAct2d,
                qat_mod=_qat.ConvAct2d,
                ref_quant_mod=_quantized.QuantizedConvAct2d,
            )
        )
    return configs


@_BackendConfigRegistry.register()
def _conv3d_act() -> _List[_BackendPatternConfig]:
    """
    float: Conv3d -> Act
    qat: FakeQuant -> qat.ConvAct3d -> FakeQuant
    """
    configs = _weighted_relu_configs(
        mod=_nn.Conv3d,
        func_mod=_F.conv3d,
        fused_mod=_nni.ConvReLU3d,
        qat_mod=_nniq.ConvReLU3d,
        ref_quant_mod=_nnr.Conv3d,
    )
    for act in _mod_activations:
        configs.extend(
            _weighted_act_configs(
                mod=_nn.Conv3d,
                func_mod=_F.conv3d,
                act=act,
                fused_mod=_fused.ConvAct3d,
                qat_mod=_qat.ConvAct3d,
                ref_quant_mod=_quantized.QuantizedConvAct3d,
            )
        )
    return configs


@_BackendConfigRegistry.register()
def _conv_transpose1d_act() -> _List[_BackendPatternConfig]:
    """
    float: ConvTranspose1d -> Act
    qat: FakeQuant -> qat.ConvTransposeAct1d -> FakeQuant
    """
    configs = []
    for act in _mod_activations + _relu_activations:
        configs.extend(
            _weighted_act_configs(
                mod=_nn.ConvTranspose1d,
                func_mod=_F.conv_transpose1d,
                act=act,
                fused_mod=_fused.ConvTransposeAct1d,
                qat_mod=_qat.ConvTransposeAct1d,
                ref_quant_mod=_quantized.QuantizedConvTransposeAct1d,
            )
        )
    return configs


@_BackendConfigRegistry.register()
def _conv_transpose2d_act() -> _List[_BackendPatternConfig]:
    """
    float: ConvTranspose2d -> Act
    qat: FakeQuant -> qat.ConvTransposeAct2d -> FakeQuant
    """
    configs = []
    for act in _mod_activations + _relu_activations:
        configs.extend(
            _weighted_act_configs(
                mod=_nn.ConvTranspose2d,
                func_mod=_F.conv_transpose2d,
                act=act,
                fused_mod=_fused.ConvTransposeAct2d,
                qat_mod=_qat.ConvTransposeAct2d,
                ref_quant_mod=_quantized.QuantizedConvTransposeAct2d,
            )
        )
    return configs


@_BackendConfigRegistry.register()
def _conv_transpose3d_act() -> _List[_BackendPatternConfig]:
    """
    float: ConvTranspose3d -> Act
    qat: FakeQuant -> qat.ConvTransposeAct3d -> FakeQuant
    """
    configs = []
    for act in _mod_activations + _relu_activations:
        configs.extend(
            _weighted_act_configs(
                mod=_nn.ConvTranspose3d,
                func_mod=_F.conv_transpose3d,
                act=act,
                fused_mod=_fused.ConvTransposeAct3d,
                qat_mod=_qat.ConvTransposeAct3d,
                ref_quant_mod=_quantized.QuantizedConvTransposeAct3d,
            )
        )
    return configs


@_BackendConfigRegistry.register()
def _linear_act() -> _List[_BackendPatternConfig]:
    """
    float: Linear -> Act
    qat: FakeQuant -> qat.LinearAct -> FakeQuant
    """
    configs = _weighted_relu_configs(
        mod=_nn.Linear,
        func_mod=_F.linear,
        fused_mod=_nni.LinearReLU,
        qat_mod=_nniq.LinearReLU,
        ref_quant_mod=_nnr.Linear,
    )
    for act in _mod_activations:
        configs.extend(
            _weighted_act_configs(
                mod=_nn.Linear,
                func_mod=_F.linear,
                act=act,
                fused_mod=_fused.LinearAct,
                qat_mod=_qat.LinearAct,
                ref_quant_mod=_quantized.QuantizedLinearAct,
            )
        )
    return configs


@_BackendConfigRegistry.register()
def _conv1d_bn() -> _List[_BackendPatternConfig]:
    """
    float: Conv1d -> BatchNorm1d
    qat: FakeQuant -> qat.ConvBn1d -> FakeQuant
    """
    return _weighted_bn_configs(
        mod=_nn.Conv1d,
        bn_mod=_nn.BatchNorm1d,
        fused_mod=_nni.ConvBn1d,
        qat_mod=_nniq.ConvBn1d,
        ref_quant_mod=_nnr.Conv1d,
    )


@_BackendConfigRegistry.register()
def _conv2d_bn() -> _List[_BackendPatternConfig]:
    """
    float: Conv2d -> BatchNorm2d
    qat: FakeQuant -> qat.ConvBn2d -> FakeQuant
    """
    return _weighted_bn_configs(
        mod=_nn.Conv2d,
        bn_mod=_nn.BatchNorm2d,
        fused_mod=_nni.ConvBn2d,
        qat_mod=_nniq.ConvBn2d,
        ref_quant_mod=_nnr.Conv2d,
    )


@_BackendConfigRegistry.register()
def _conv3d_bn() -> _List[_BackendPatternConfig]:
    """
    float: Conv3d -> BatchNorm3d
    qat: FakeQuant -> qat.ConvBn3d -> FakeQuant
    """
    return _weighted_bn_configs(
        mod=_nn.Conv3d,
        bn_mod=_nn.BatchNorm3d,
        fused_mod=_nni.ConvBn3d,
        qat_mod=_nniq.ConvBn3d,
        ref_quant_mod=_nnr.Conv3d,
    )


@_BackendConfigRegistry.register()
def _conv_transpose1d_bn() -> _List[_BackendPatternConfig]:
    """
    float: ConvTranspose1d -> BatchNorm1d
    qat: FakeQuant -> qat.ConvTransposeBn1d -> FakeQuant
    """
    return _weighted_bn_configs(
        mod=_nn.ConvTranspose1d,
        bn_mod=_nn.BatchNorm1d,
        fused_mod=_fused.ConvTransposeBn1d,
        qat_mod=_qconv_transpose_fused.ConvTransposeBn1d,
        ref_quant_mod=_nnr.ConvTranspose1d,
    )


@_BackendConfigRegistry.register()
def _conv_transpose2d_bn() -> _List[_BackendPatternConfig]:
    """
    float: ConvTranspose2d -> BatchNorm2d
    qat: FakeQuant -> qat.ConvTransposeBn2d -> FakeQuant
    """
    return _weighted_bn_configs(
        mod=_nn.ConvTranspose2d,
        bn_mod=_nn.BatchNorm2d,
        fused_mod=_fused.ConvTransposeBn2d,
        qat_mod=_qconv_transpose_fused.ConvTransposeBn2d,
        ref_quant_mod=_nnr.ConvTranspose2d,
    )


@_BackendConfigRegistry.register()
def _conv_transpose3d_bn() -> _List[_BackendPatternConfig]:
    """
    float: ConvTranspose3d -> BatchNorm3d
    qat: FakeQuant -> qat.ConvTransposeBn3d -> FakeQuant
    """
    return _weighted_bn_configs(
        mod=_nn.ConvTranspose3d,
        bn_mod=_nn.BatchNorm3d,
        fused_mod=_fused.ConvTransposeBn3d,
        qat_mod=_qconv_transpose_fused.ConvTransposeBn3d,
        ref_quant_mod=_nnr.ConvTranspose3d,
    )


@_BackendConfigRegistry.register()
def _linear_bn() -> _List[_BackendPatternConfig]:
    """
    float: Linear -> BatchNorm1d
    qat: FakeQuant -> qat.LinearBn1d -> FakeQuant
    """
    return _weighted_bn_configs(
        mod=_nn.Linear,
        bn_mod=_nn.BatchNorm1d,
        fused_mod=_nni.LinearBn1d,
        qat_mod=_nniq.LinearBn1d,
        ref_quant_mod=_nnr.Linear,
    )


@_BackendConfigRegistry.register()
def _conv1d_bn_act() -> _List[_BackendPatternConfig]:
    """
    float: Conv1d -> BatchNorm1d -> Act
    qat: FakeQuant -> qat.ConvBnAct1d -> FakeQuant
    """
    configs = _weighted_bn_relu_configs(
        mod=_nn.Conv1d,
        bn_mod=_nn.BatchNorm1d,
        fused_mod=_nni.ConvBnReLU1d,
        qat_mod=_nniq.ConvBnReLU1d,
        ref_quant_mod=_nnr.Conv1d,
    )
    for act in _mod_activations:
        configs.extend(
            _weighted_bn_act_configs(
                mod=_nn.Conv1d,
                act=act,
                bn_mod=_nn.BatchNorm1d,
                root_mod=_nni.ConvBn1d,
                fused_mod=_fused.ConvBnAct1d,
                qat_mod=_qat.ConvBnAct1d,
                ref_quant_mod=_quantized.QuantizedConvAct1d,
            )
        )
    return configs


@_BackendConfigRegistry.register()
def _conv2d_bn_act() -> _List[_BackendPatternConfig]:
    """
    float: Conv2d -> BatchNorm2d -> Act
    qat: FakeQuant -> qat.ConvBnAct2d -> FakeQuant
    """
    configs = _weighted_bn_relu_configs(
        mod=_nn.Conv2d,
        bn_mod=_nn.BatchNorm2d,
        fused_mod=_nni.ConvBnReLU2d,
        qat_mod=_nniq.ConvBnReLU2d,
        ref_quant_mod=_nnr.Conv2d,
    )
    for act in _mod_activations:
        configs.extend(
            _weighted_bn_act_configs(
                mod=_nn.Conv2d,
                act=act,
                bn_mod=_nn.BatchNorm2d,
                root_mod=_nni.ConvBn2d,
                fused_mod=_fused.ConvBnAct2d,
                qat_mod=_qat.ConvBnAct2d,
                ref_quant_mod=_quantized.QuantizedConvAct2d,
            )
        )
    return configs


@_BackendConfigRegistry.register()
def _conv3d_bn_act() -> _List[_BackendPatternConfig]:
    """
    float: Conv3d -> BatchNorm3d -> Act
    qat: FakeQuant -> qat.ConvBnAct3d -> FakeQuant
    """
    configs = _weighted_bn_relu_configs(
        mod=_nn.Conv3d,
        bn_mod=_nn.BatchNorm3d,
        fused_mod=_nni.ConvBnReLU3d,
        qat_mod=_nniq.ConvBnReLU3d,
        ref_quant_mod=_nnr.Conv3d,
    )
    for act in _mod_activations:
        configs.extend(
            _weighted_bn_act_configs(
                mod=_nn.Conv3d,
                act=act,
                bn_mod=_nn.BatchNorm3d,
                root_mod=_nni.ConvBn3d,
                fused_mod=_fused.ConvBnAct3d,
                qat_mod=_qat.ConvBnAct3d,
                ref_quant_mod=_quantized.QuantizedConvAct3d,
            )
        )
    return configs


@_BackendConfigRegistry.register()
def _conv_transpose1d_bn_act() -> _List[_BackendPatternConfig]:
    """
    float: ConvTranspose1d -> BatchNorm1d -> Act
    qat: FakeQuant -> qat.ConvTransposeBnAct1d -> FakeQuant
    """
    configs = []
    for act in _mod_activations + _relu_activations:
        configs.extend(
            _weighted_bn_act_configs(
                mod=_nn.ConvTranspose1d,
                act=act,
                bn_mod=_nn.BatchNorm1d,
                root_mod=_fused.ConvTransposeBn1d,
                fused_mod=_fused.ConvTransposeBnAct1d,
                qat_mod=_qat.ConvTransposeBnAct1d,
                ref_quant_mod=_quantized.QuantizedConvTransposeAct1d,
            )
        )
    return configs


@_BackendConfigRegistry.register()
def _conv_transpose2d_bn_act() -> _List[_BackendPatternConfig]:
    """
    float: ConvTranspose2d -> BatchNorm2d -> Act
    qat: FakeQuant -> qat.ConvTransposeBnAct2d -> FakeQuant
    """
    configs = []
    for act in _mod_activations + _relu_activations:
        configs.extend(
            _weighted_bn_act_configs(
                mod=_nn.ConvTranspose2d,
                act=act,
                bn_mod=_nn.BatchNorm2d,
                root_mod=_fused.ConvTransposeBn2d,
                fused_mod=_fused.ConvTransposeBnAct2d,
                qat_mod=_qat.ConvTransposeBnAct2d,
                ref_quant_mod=_quantized.QuantizedConvTransposeAct2d,
            )
        )
    return configs


@_BackendConfigRegistry.register()
def _conv_transpose3d_bn_act() -> _List[_BackendPatternConfig]:
    """
    float: ConvTranspose3d -> BatchNorm3d -> Act
    qat: FakeQuant -> qat.ConvTransposeBnAct3d -> FakeQuant
    """
    configs = []
    for act in _mod_activations + _relu_activations:
        configs.extend(
            _weighted_bn_act_configs(
                mod=_nn.ConvTranspose3d,
                act=act,
                bn_mod=_nn.BatchNorm3d,
                root_mod=_fused.ConvTransposeBn3d,
                fused_mod=_fused.ConvTransposeBnAct3d,
                qat_mod=_qat.ConvTransposeBnAct3d,
                ref_quant_mod=_quantized.QuantizedConvTransposeAct3d,
            )
        )
    return configs


@_BackendConfigRegistry.register()
def _conv1d() -> _List[_BackendPatternConfig]:
    """
    float: Conv1d
    qat: FakeQuant -> qat.Conv1d -> FakeQuant
    """
    return _weighted_configs(
        mod=_nn.Conv1d,
        func_mod=_F.conv1d,
        qat_mod=_nnq.Conv1d,
        ref_quant_mod=_nnr.Conv1d,
    )


@_BackendConfigRegistry.register()
def _conv2d() -> _List[_BackendPatternConfig]:
    """
    float: Conv2d
    qat: FakeQuant -> qat.Conv2d -> FakeQuant
    """
    return _weighted_configs(
        mod=_nn.Conv2d,
        func_mod=_F.conv2d,
        qat_mod=_nnq.Conv2d,
        ref_quant_mod=_nnr.Conv2d,
    )


@_BackendConfigRegistry.register()
def _conv3d() -> _List[_BackendPatternConfig]:
    """
    float: Conv3d
    qat: FakeQuant -> qat.Conv3d -> FakeQuant
    """
    return _weighted_configs(
        mod=_nn.Conv3d,
        func_mod=_F.conv3d,
        qat_mod=_nnq.Conv3d,
        ref_quant_mod=_nnr.Conv3d,
    )


@_BackendConfigRegistry.register()
def _conv_transpose1d() -> _List[_BackendPatternConfig]:
    """
    float: ConvTranspose1d
    qat: FakeQuant -> qat.ConvTranspose1d -> FakeQuant
    """
    return _weighted_configs(
        mod=_nn.ConvTranspose1d,
        func_mod=_F.conv_transpose1d,
        qat_mod=_qconv_transpose.ConvTranspose1d,
        ref_quant_mod=_nnr.ConvTranspose1d,
    )


@_BackendConfigRegistry.register()
def _conv_transpose2d() -> _List[_BackendPatternConfig]:
    """
    float: ConvTranspose2d
    qat: FakeQuant -> qat.ConvTranspose2d -> FakeQuant
    """
    return _weighted_configs(
        mod=_nn.ConvTranspose2d,
        func_mod=_F.conv_transpose2d,
        qat_mod=_qconv_transpose.ConvTranspose2d,
        ref_quant_mod=_nnr.ConvTranspose2d,
    )


@_BackendConfigRegistry.register()
def _conv_transpose3d() -> _List[_BackendPatternConfig]:
    """
    float: ConvTranspose3d
    qat: FakeQuant -> qat.ConvTranspose3d -> FakeQuant
    """
    return _weighted_configs(
        mod=_nn.ConvTranspose3d,
        func_mod=_F.conv_transpose3d,
        qat_mod=_qconv_transpose.ConvTranspose3d,
        ref_quant_mod=_nnr.ConvTranspose3d,
    )


@_BackendConfigRegistry.register()
def _linear() -> _List[_BackendPatternConfig]:
    """
    float: Linear
    qat: FakeQuant -> qat.Linear -> FakeQuant
    """
    return _weighted_configs(
        mod=_nn.Linear,
        func_mod=_F.linear,
        qat_mod=_nnq.Linear,
        ref_quant_mod=_nnr.Linear,
    )


@_BackendConfigRegistry.register()
def _embedding() -> _List[_BackendPatternConfig]:
    """
    float: Embedding
    qat: qat.Embedding
    """
    return _weighted_configs(
        mod=_nn.Embedding,
        func_mod=None,
        qat_mod=_nnq.Embedding,
        ref_quant_mod=_nnr.Embedding,
        input_output_observed=False,
    )


@_BackendConfigRegistry.register()
def _embedding_bag() -> _List[_BackendPatternConfig]:
    """
    float: EmbeddingBag
    qat: qat.EmbeddingBag
    """
    return _weighted_configs(
        mod=_nn.EmbeddingBag,
        func_mod=None,
        qat_mod=_nnq.EmbeddingBag,
        ref_quant_mod=_nnr.EmbeddingBag,
        input_output_observed=False,
    )


# n-ary ops
@_BackendConfigRegistry.register()
def _identity() -> _List[_BackendPatternConfig]:
    return _share_observer_configs(ops=[_nn.Identity])


@_BackendConfigRegistry.register()
def _add_act() -> _List[_BackendPatternConfig]:
    """
    float:
    input_1 ->
                add -> Act -> output
    input_2 ->

    qat:
    FakeQuant ->
                 add -> Act -> FakeQuant
    FakeQuant ->
    """
    acts = _mod_activations + _func_activations + (_nn.ReLU, _F.relu, _torch.relu)
    return _binary_op_relu_configs(ops=[_operator.add, _torch.add], acts=list(acts))


@_BackendConfigRegistry.register()
def _mul_act() -> _List[_BackendPatternConfig]:
    """
    float:
    input_1 ->
                mul -> Act -> output
    input_2 ->

    qat:
    FakeQuant ->
                 mul -> Act -> FakeQuant
    FakeQuant ->
    """
    acts = _mod_activations + _func_activations + (_nn.ReLU, _F.relu, _torch.relu)
    return _binary_op_relu_configs(ops=[_operator.mul, _torch.mul], acts=list(acts))


@_BackendConfigRegistry.register()
def _matmul_act() -> _List[_BackendPatternConfig]:
    """
    float:
    input_1 ->
                matmul -> Act -> output
    input_2 ->

    qat:
    FakeQuant ->
                 matmul -> Act -> FakeQuant
    FakeQuant ->
    """
    acts = _mod_activations + _func_activations + (_nn.ReLU, _F.relu, _torch.relu)
    return _binary_op_relu_configs(ops=[_torch.matmul], acts=list(acts))


@_BackendConfigRegistry.register()
def _add() -> _List[_BackendPatternConfig]:
    """
    float:
    input_1 ->
                add -> output
    input_2 ->

    qat:
    FakeQuant ->
                 add -> FakeQuant
    FakeQuant ->
    """
    return _binary_op_configs(ops=[_operator.add, _torch.add])


@_BackendConfigRegistry.register()
def _mul() -> _List[_BackendPatternConfig]:
    """
    float:
    input_1 ->
                mul -> output
    input_2 ->

    qat:
    FakeQuant ->
                 mul -> FakeQuant
    FakeQuant ->
    """
    return _binary_op_configs(ops=[_operator.mul, _torch.mul])


@_BackendConfigRegistry.register()
def _matmul() -> _List[_BackendPatternConfig]:
    """
    float:
    input_1 ->
                matmul -> output
    input_2 ->

    qat:
    FakeQuant ->
                 matmul -> FakeQuant
    FakeQuant ->
    """
    return _binary_op_configs(ops=[_torch.matmul])


@_BackendConfigRegistry.register()
def _cat() -> _List[_BackendPatternConfig]:
    """
    float:
    input_1 ->
                cat -> output
    input_2 ->

    qat:
    FakeQuant ->
                 cat -> FakeQuant
    FakeQuant ->

    The number of inputs is not restricted to 2.
    All FakeQuant(s) share the same scale and zero point
    """
    return _share_observer_configs(ops=[_torch.cat])


# pooling layers
@_BackendConfigRegistry.register()
def _max_pool1d() -> _List[_BackendPatternConfig]:
    """
    float: MaxPool1d
    qat: FakeQuant -> MaxPool1d -> FakeQuant

    FakeQuant(s) share the same scale and zero point
    """
    return _share_observer_configs(ops=[_nn.MaxPool1d, _F.max_pool1d])


@_BackendConfigRegistry.register()
def _max_pool2d() -> _List[_BackendPatternConfig]:
    """
    float: MaxPool2d
    qat: FakeQuant -> MaxPool2d -> FakeQuant

    FakeQuant(s) share the same scale and zero point
    """
    return _share_observer_configs(ops=[_nn.MaxPool2d, _F.max_pool2d])


@_BackendConfigRegistry.register()
def _max_pool3d() -> _List[_BackendPatternConfig]:
    """
    float: MaxPool3d
    qat: FakeQuant -> MaxPool3d -> FakeQuant

    FakeQuant(s) share the same scale and zero point
    """
    return _share_observer_configs(ops=[_nn.MaxPool3d, _F.max_pool3d])


@_BackendConfigRegistry.register()
def _adaptive_avg_pool1d() -> _List[_BackendPatternConfig]:
    """
    float: AdaptiveAvgPool1d
    qat: FakeQuant -> AdaptiveAvgPool1d -> FakeQuant

    FakeQuant(s) share the same scale and zero point
    """
    return _share_observer_configs(
        ops=[_nn.AdaptiveAvgPool1d, _F.adaptive_avg_pool1d, _torch.adaptive_avg_pool1d]
    )


@_BackendConfigRegistry.register()
def _adaptive_avg_pool2d() -> _List[_BackendPatternConfig]:
    """
    float: AdaptiveAvgPool2d
    qat: FakeQuant -> AdaptiveAvgPool2d -> FakeQuant

    FakeQuant(s) share the same scale and zero point
    """
    return _share_observer_configs(ops=[_nn.AdaptiveAvgPool2d, _F.adaptive_avg_pool2d])


@_BackendConfigRegistry.register()
def _adaptive_avg_pool3d() -> _List[_BackendPatternConfig]:
    """
    float: AdaptiveAvgPool3d
    qat: FakeQuant -> AdaptiveAvgPool3d -> FakeQuant

    FakeQuant(s) share the same scale and zero point
    """
    return _share_observer_configs(ops=[_nn.AdaptiveAvgPool3d, _F.adaptive_avg_pool3d])


@_BackendConfigRegistry.register()
def _avg_pool1d() -> _List[_BackendPatternConfig]:
    """
    float: AvgPool1d
    qat: FakeQuant -> AvgPool1d -> FakeQuant

    FakeQuant(s) share the same scale and zero point
    """
    return _share_observer_configs(
        ops=[_nn.AvgPool1d, _F.avg_pool1d, _torch.avg_pool1d, _torch.mean]
    )


@_BackendConfigRegistry.register()
def _avg_pool2d() -> _List[_BackendPatternConfig]:
    """
    float: AvgPool2d
    qat: FakeQuant -> AvgPool2d -> FakeQuant

    FakeQuant(s) share the same scale and zero point
    """
    return _share_observer_configs(ops=[_nn.AvgPool2d, _F.avg_pool2d, _torch._C._nn.avg_pool2d])


@_BackendConfigRegistry.register()
def _avg_pool3d() -> _List[_BackendPatternConfig]:
    """
    float: AvgPool3d
    qat: FakeQuant -> AvgPool3d -> FakeQuant

    FakeQuant(s) share the same scale and zero point
    """
    return _share_observer_configs(ops=[_nn.AvgPool3d, _F.avg_pool3d, _torch._C._nn.avg_pool3d])


# memory movement ops
@_BackendConfigRegistry.register()
def _flatten() -> _List[_BackendPatternConfig]:
    """
    float: AvgPool1d
    qat: FakeQuant -> Flatten -> FakeQuant

    FakeQuant(s) share the same scale and zero point
    """
    return _share_observer_configs(ops=[_nn.Flatten, _torch.flatten])


# norm layers
@_BackendConfigRegistry.register()
def _bn() -> _List[_BackendPatternConfig]:
    """
    float: BatchNorm
    qat: FakeQuant -> BatchNorm -> FakeQuant
    """
    return _activation_configs(ops=[_nn.BatchNorm1d, _nn.BatchNorm2d, _nn.BatchNorm3d])


@_BackendConfigRegistry.register()
def _bn2d_relu() -> _List[_BackendPatternConfig]:
    """
    float: BatchNorm2d -> ReLU
    qat: FakeQuant -> BNReLU2d -> FakeQuant
    """
    return _bn_relu(mod=_nn.BatchNorm2d, fused_mod=_nni.BNReLU2d)


@_BackendConfigRegistry.register()
def _bn3d_relu() -> _List[_BackendPatternConfig]:
    """
    float: BatchNorm3d -> ReLU
    qat: FakeQuant -> BNReLU3d -> FakeQuant
    """
    return _bn_relu(mod=_nn.BatchNorm3d, fused_mod=_nni.BNReLU3d)


# activations
@_BackendConfigRegistry.register()
def _softmax() -> _List[_BackendPatternConfig]:
    """
    float: Softmax
    qat: FakeQuant -> Softmax -> FakeQuant

    FakeQuant at the output has fixed qparams.
    """
    constraints = (
        _DTypeWithConstraints(
            dtype=_torch.quint8,
            quant_min_lower_bound=0,
            quant_max_upper_bound=255,
            scale_exact_match=1.0 / 256.0,
            zero_point_exact_match=0,
        )
        if _is_torch_2()
        else None
    )
    return _activation_configs(ops=[_nn.Softmax], constraints=constraints)


@_BackendConfigRegistry.register()
def _sigmoid() -> _List[_BackendPatternConfig]:
    """
    float: Sigmoid
    qat: FakeQuant -> Sigmoid -> FakeQuant

    FakeQuant at the output has fixed qparams.
    """
    constraints = (
        _DTypeWithConstraints(
            dtype=_torch.quint8,
            quant_min_lower_bound=0,
            quant_max_upper_bound=255,
            scale_exact_match=1.0 / 256.0,
            zero_point_exact_match=0,
        )
        if _is_torch_2()
        else None
    )
    return _activation_configs(ops=[_nn.Sigmoid, _F.sigmoid], constraints=constraints)


@_BackendConfigRegistry.register()
def _hardsigmoid() -> _List[_BackendPatternConfig]:
    """
    float: Hardsigmoid
    qat: FakeQuant -> Hardsigmoid -> FakeQuant

    FakeQuant at the output has fixed qparams.
    """
    constraints = (
        _DTypeWithConstraints(
            dtype=_torch.quint8,
            quant_min_lower_bound=0,
            quant_max_upper_bound=255,
            scale_exact_match=1.0 / 256.0,
            zero_point_exact_match=0,
        )
        if _is_torch_2()
        else None
    )
    return _activation_configs(ops=[_nn.Hardsigmoid, _F.hardsigmoid], constraints=constraints)


@_BackendConfigRegistry.register()
def _tanh() -> _List[_BackendPatternConfig]:
    """
    float: Tanh
    qat: FakeQuant -> Tanh -> FakeQuant

    FakeQuant at the output has fixed qparams.
    """
    constraints = (
        _DTypeWithConstraints(
            dtype=_torch.quint8,
            quant_min_lower_bound=0,
            quant_max_upper_bound=255,
            scale_exact_match=2.0 / 256.0,
            zero_point_exact_match=128,
        )
        if _is_torch_2()
        else None
    )
    return _activation_configs(ops=[_nn.Tanh, _F.tanh], constraints=constraints)


@_BackendConfigRegistry.register()
def _activations() -> _List[_BackendPatternConfig]:
    """
    float: Act
    qat: FakeQuant -> Act -> FakeQuant
    """
    ops = [op for op in _mod_activations if op not in _fixed_qparams_modules]
    ops += [
        _nn.ReLU,
        _F.relu,
        _F.relu_,
    ] + list(_func_activations)
    return _activation_configs(ops=ops)


def get_backend_config() -> _BackendConfig:
    """
    Returns backend config encoding information about how quantization
    layers are inserted in a module.
    """
    return _BackendConfigRegistry.backend_config


def get_supported_modules() -> _List[_Any]:
    """
    Returns a tuple of modules which are supported for quantization
    aware training.
    """
    return tuple(_BackendConfigRegistry.supported_modules)
