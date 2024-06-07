#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Any as _Any
from typing import Optional as _Optional

import torch as _torch
import torch.ao.quantization as _aoquant
import torch.nn as _nn

from coremltools.optimize.torch.quantization._backend_config import _fixed_qparams_modules
from coremltools.optimize.torch.quantization._backend_config import (
    get_supported_modules as _get_supported_modules,
)
from coremltools.optimize.torch.quantization._utils import get_quant_range as _get_quant_range
from coremltools.optimize.torch.quantization.quantization_config import (
    LinearQuantizerConfig as _LinearQuantizerConfig,
)
from coremltools.optimize.torch.quantization.quantization_config import (
    ModuleLinearQuantizerConfig as _ModuleLinearQuantizerConfig,
)
from coremltools.optimize.torch.quantization.quantization_config import (
    ObserverType as _ObserverType,
)
from coremltools.optimize.torch.quantization.quantization_config import (
    QuantizationScheme as _QuantizationScheme,
)
from coremltools.optimize.torch.quantization.quantization_config import (
    _default_quantization_options,
)


class _QConfigMappingBuilder:
    """
    Builds py:class:`QConfigMapping` from :py:class:`LinearQuantizerConfig`.
    """

    @staticmethod
    def _get_default_qconfig_from_quantization_scheme(
        quantization_scheme: _QuantizationScheme,
    ) -> _aoquant.QConfig:
        """
        Returns default QConfig for a given quantization types
        """
        return _aoquant.QConfig(
            activation=_aoquant.FakeQuantize.with_args(
                observer=_ObserverType.get_observer(
                    _default_quantization_options["observer"], is_per_channel=False
                ),
                dtype=_default_quantization_options["activation_dtype"],
                qscheme=_QuantizationScheme.get_qscheme(quantization_scheme, is_per_channel=False),
            ),
            weight=_aoquant.FakeQuantize.with_args(
                observer=_ObserverType.get_observer(
                    _default_quantization_options["observer"],
                    is_per_channel=_default_quantization_options["weight_per_channel"],
                ),
                dtype=_default_quantization_options["weight_dtype"],
                qscheme=_QuantizationScheme.get_qscheme(
                    quantization_scheme,
                    is_per_channel=_default_quantization_options["weight_per_channel"],
                ),
            ),
        )

    @staticmethod
    def _adjust_qconfig_for_module_type(mod_type: _Any, qconfig: _aoquant.QConfig):
        """
        Enforces Embedding layers to use float qparams, because that's preferred
        by prepare_qat_fx.
        """
        if mod_type == _torch.nn.Embedding:
            weight = qconfig.weight()
            return _aoquant.QConfig(
                activation=_aoquant.NoopObserver.with_args(dtype=_torch.float),
                weight=_aoquant.FakeQuantize.with_args(
                    observer=type(weight.activation_post_process),
                    dtype=weight.dtype,
                    qscheme=_torch.per_channel_affine_float_qparams,
                ),
            )
        return qconfig

    @staticmethod
    def _get_module_names_for_setting_qconfig(model: _nn.Module, mod_name: str):
        """
        When layers are fused and we want to skip quantization for a convolution
        or linear layer, we need to set the qconfig for the layer being fused as None
        as well.
        """
        try:
            submod = model.get_submodule(mod_name)
        except AttributeError:
            return (mod_name,)

        if isinstance(submod, _torch.nn.Conv2d):
            return mod_name, f"{mod_name}.conv"
        elif isinstance(submod, _torch.nn.Linear):
            return mod_name, f"{mod_name}.linear"
        return (mod_name,)

    @staticmethod
    def _create_qconfig_from_quantization_config(
        quantization_config: _ModuleLinearQuantizerConfig,
    ) -> _Optional[_aoquant.QConfig]:
        """
        Creates a :py:class:`QConfig` from ``quantization_config``
        """
        if quantization_config.weight_dtype == _torch.float32:
            return None
        if quantization_config.activation_dtype == _torch.float32:
            activation_qconfig = _aoquant.NoopObserver.with_args(
                dtype=_torch.float,
            )
        else:
            activation_qconfig = _aoquant.FakeQuantize.with_args(
                observer=_ObserverType.get_observer(
                    quantization_config.activation_observer,
                    is_per_channel=False,
                ),
                dtype=quantization_config.activation_dtype,
                qscheme=_QuantizationScheme.get_qscheme(
                    quantization_config.quantization_scheme,
                    is_per_channel=False,
                ),
            )

        quant_min, quant_max = (
            _get_quant_range(
                n_bits=quantization_config.weight_n_bits,
                dtype=quantization_config.weight_dtype,
            )
            if quantization_config.weight_n_bits < 8
            else (None, None)
        )

        weight_qconfig = _aoquant.FakeQuantize.with_args(
            observer=_ObserverType.get_observer(
                quantization_config.weight_observer,
                is_per_channel=quantization_config.weight_per_channel,
            ),
            dtype=quantization_config.weight_dtype,
            qscheme=_QuantizationScheme.get_qscheme(
                quantization_config.quantization_scheme,
                is_per_channel=quantization_config.weight_per_channel,
            ),
            quant_min=quant_min,
            quant_max=quant_max,
        )
        return _aoquant.QConfig(activation=activation_qconfig, weight=weight_qconfig)

    def get_default_qconfig_mapping(
        self,
        quantization_scheme: _QuantizationScheme,
        qconfig: _Optional[_aoquant.QConfig] = None,
    ) -> _aoquant.QConfigMapping:
        """
        Returns default QconfigMapping for a given quantization scheme. If a qconfig is passed,
        it is used as the default qconfig instead.
        """
        supported_modules = list(set(_get_supported_modules()) - set(_fixed_qparams_modules))
        # Add _FakeQuantize to ensure all fused ops have same qconfig
        supported_modules.append(_aoquant.FakeQuantize)

        qconfig_mapping = _aoquant.QConfigMapping()
        default_qconfig_mapping = _aoquant.get_default_qat_qconfig_mapping()

        # copy qconfig mapping for fixed qparams
        for key in default_qconfig_mapping.object_type_qconfigs:
            if key in _fixed_qparams_modules:
                qconfig_mapping.set_object_type(
                    key, default_qconfig_mapping.object_type_qconfigs[key]
                )

        qconfig = (
            self._get_default_qconfig_from_quantization_scheme(quantization_scheme)
            if qconfig is None
            else qconfig
        )

        qconfig_mapping.set_global(qconfig)
        for mod_type in supported_modules:
            qconfig_mapping.set_object_type(
                mod_type,
                self._adjust_qconfig_for_module_type(mod_type, qconfig),
            )
        return qconfig_mapping

    def get_qconfig_mapping_from_quantization_config(
        self,
        model: _nn.Module,
        quantization_config: _LinearQuantizerConfig,
        quantization_scheme: _QuantizationScheme,
    ) -> _aoquant.QConfigMapping:
        """
        Builds py:class:`QConfigMapping` from :py:class:`LinearQuantizerConfig`.
        """
        qconfig_mapping = self.get_default_qconfig_mapping(quantization_scheme)
        if quantization_config.global_config is not None:
            qconfig_mapping = self.get_default_qconfig_mapping(
                quantization_scheme,
                self._create_qconfig_from_quantization_config(quantization_config.global_config),
            )
        for mod_type, config in quantization_config.module_type_configs.items():
            qconfig = (
                self._create_qconfig_from_quantization_config(config)
                if config is not None
                else config
            )
            qconfig = (
                self._adjust_qconfig_for_module_type(mod_type, qconfig)
                if qconfig is not None
                else qconfig
            )
            qconfig_mapping = qconfig_mapping.set_object_type(mod_type, qconfig)
        for mod_name, config in quantization_config.module_name_configs.items():
            qconfig = (
                self._create_qconfig_from_quantization_config(config)
                if config is not None
                else config
            )
            mod_names = self._get_module_names_for_setting_qconfig(model, mod_name)
            for mn in mod_names:
                qconfig_mapping = qconfig_mapping.set_module_name(mn, qconfig)
        return qconfig_mapping
