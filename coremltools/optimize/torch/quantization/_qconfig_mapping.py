#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging as _logging
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Type as _Type

import torch as _torch
import torch.ao.quantization as _aoquant
import torch.nn as _nn

from coremltools.optimize.torch.quantization._backend_config import _fixed_qparams_modules
from coremltools.optimize.torch.quantization._backend_config import (
    get_supported_modules as _get_supported_modules,
)
from coremltools.optimize.torch.quantization._utils import get_quant_range as _get_quant_range
from coremltools.optimize.torch.quantization.modules.observers import NoopObserver as _NoopObserver
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

_logger = _logging.getLogger(__name__)


class _QConfigMappingBuilder:
    """
    Builds py:class:`QConfigMapping` from :py:class:`LinearQuantizerConfig`.
    """

    _observer_cls: _Type = _ObserverType

    @classmethod
    def _get_fake_quantize_class(
        cls, quantization_config: _Optional[_ModuleLinearQuantizerConfig] = None
    ) -> _Type[_aoquant.FakeQuantizeBase]:
        return _aoquant.FakeQuantize

    @classmethod
    def _create_fake_quantize_partial_from_kwargs(
        cls,
        is_weight: bool,
        observer_cls: _aoquant.ObserverBase,
        dtype: _torch.dtype,
        qscheme: _torch.qscheme,
        weight_per_channel: bool = False,
        quant_min: _Optional[int] = None,
        quant_max: _Optional[int] = None,
        ch_axis: _Optional[int] = None,
        quantization_config: _Optional[_ModuleLinearQuantizerConfig] = None,
    ) -> _Callable:
        fq_kwargs = dict()
        fq_kwargs["observer"] = observer_cls
        fq_kwargs["dtype"] = dtype
        fq_kwargs["qscheme"] = qscheme

        if is_weight and weight_per_channel:
            fq_kwargs["ch_axis"] = (
                ch_axis if ch_axis else _default_quantization_options["weight_ch_axis"]
            )

        if quant_min is not None:
            fq_kwargs["quant_min"] = quant_min

        if quant_max is not None:
            fq_kwargs["quant_max"] = quant_max

        fq_class = cls._get_fake_quantize_class(quantization_config)
        return fq_class.with_args(**fq_kwargs)

    @classmethod
    def _get_default_qconfig_from_quantization_scheme(
        cls,
        quantization_scheme: _QuantizationScheme,
        quantization_config: _ModuleLinearQuantizerConfig,
    ) -> _aoquant.QConfig:
        """
        Returns default QConfig for a given quantization scheme
        """
        act_observer = cls._observer_cls.get_observer(
            _default_quantization_options["observer"], is_per_channel=False
        )
        act_dtype = _default_quantization_options["activation_dtype"]
        act_qscheme = _QuantizationScheme.get_qscheme(quantization_scheme, is_per_channel=False)

        weight_observer = cls._observer_cls.get_observer(
            _default_quantization_options["observer"],
            is_per_channel=_default_quantization_options["weight_per_channel"],
        )
        weight_dtype = _default_quantization_options["weight_dtype"]
        weight_qscheme = _QuantizationScheme.get_qscheme(
            quantization_scheme,
            is_per_channel=_default_quantization_options["weight_per_channel"],
        )

        return _aoquant.QConfig(
            activation=cls._create_fake_quantize_partial_from_kwargs(
                False,
                act_observer,
                act_dtype,
                act_qscheme,
                False,
                quantization_config=quantization_config,
            ),
            weight=cls._create_fake_quantize_partial_from_kwargs(
                True,
                weight_observer,
                weight_dtype,
                weight_qscheme,
                weight_per_channel=_default_quantization_options["weight_per_channel"],
                quantization_config=quantization_config,
            ),
        )

    @classmethod
    def _adjust_qconfig_for_module_type(
        cls,
        mod_type: _Any,
        qconfig: _aoquant.QConfig,
        quantization_config: _ModuleLinearQuantizerConfig,
    ) -> _aoquant.QConfig:
        """
        Enforces Embedding layers to use float qparams, because that's preferred by prepare_qat_fx.
        Overwrites ch_axis for ConvTranspose layers if qscheme is not per_tensor
        """
        if mod_type in [
            _torch.nn.Embedding,
            _torch.nn.ConvTranspose1d,
            _torch.nn.ConvTranspose2d,
            _torch.nn.ConvTranspose3d,
        ]:
            weight = qconfig.weight()
            weight_dtype = weight.dtype
            if weight_dtype == _torch.float:
                return qconfig

            weight_per_channel = _default_quantization_options["weight_per_channel"]
            weight_observer = type(weight.activation_post_process)

            if mod_type == _torch.nn.Embedding:
                ch_axis = None
                weight_qscheme = _torch.per_channel_affine_float_qparams
                # we do not want to quantize inputs to Embedding layer because they are integers
                activation_config = _NoopObserver.with_args(dtype=_torch.float)
            else:
                if hasattr(weight, "qscheme") and weight.qscheme not in [
                    _torch.per_tensor_affine,
                    _torch.per_tensor_symmetric,
                ]:
                    ch_axis = 1
                    weight_qscheme = weight.qscheme
                    # preserve activation config for ConvTranspose ops
                    activation_config = qconfig.activation
                    weight_per_channel = (
                        weight_per_channel
                        if quantization_config is None
                        else quantization_config.weight_per_channel
                    )
                else:
                    return qconfig

            return _aoquant.QConfig(
                activation=activation_config,
                weight=cls._create_fake_quantize_partial_from_kwargs(
                    True,
                    weight_observer,
                    weight_dtype,
                    weight_qscheme,
                    weight_per_channel=weight_per_channel,
                    quant_min=weight.quant_min,
                    quant_max=weight.quant_max,
                    ch_axis=ch_axis,
                    quantization_config=quantization_config,
                ),
            )
        return qconfig

    @staticmethod
    def _get_module_names_for_setting_qconfig(model: _nn.Module, mod_name: str) -> _Tuple[str, ...]:
        """
        When layers are fused and we want to skip quantization for a convolution
        or linear layer, we need to set the qconfig for the layer being fused as None
        as well.
        """
        try:
            submod = model.get_submodule(mod_name)
        except AttributeError:
            return (mod_name,)

        if isinstance(submod, _torch.nn.Conv2d) or isinstance(submod, _torch.nn.ConvTranspose2d):
            return mod_name, f"{mod_name}.conv"
        elif isinstance(submod, _torch.nn.Linear):
            return mod_name, f"{mod_name}.linear"
        return (mod_name,)

    @classmethod
    def _create_qconfig_from_quantization_config(
        cls,
        quantization_config: _ModuleLinearQuantizerConfig,
    ) -> _Optional[_aoquant.QConfig]:
        """
        Creates a :py:class:`QConfig` from ``quantization_config``
        """
        if quantization_config.activation_dtype == _torch.float32:
            activation_qconfig = _NoopObserver.with_args(
                dtype=_torch.float,
            )
        else:
            act_observer = cls._observer_cls.get_observer(
                quantization_config.activation_observer, is_per_channel=False
            )
            act_dtype = quantization_config.activation_dtype
            act_qscheme = _QuantizationScheme.get_qscheme(
                quantization_config.quantization_scheme, is_per_channel=False
            )

            activation_qconfig = cls._create_fake_quantize_partial_from_kwargs(
                False,
                act_observer,
                act_dtype,
                act_qscheme,
                quantization_config=quantization_config,
            )

        if quantization_config.weight_dtype == _torch.float32:
            weight_qconfig = _NoopObserver.with_args(
                dtype=_torch.float,
            )
        else:
            quant_min, quant_max = (
                _get_quant_range(
                    n_bits=quantization_config.weight_n_bits,
                    dtype=quantization_config.weight_dtype,
                )
                if quantization_config.weight_n_bits < 8
                else (None, None)
            )

            weight_observer = cls._observer_cls.get_observer(
                quantization_config.weight_observer,
                is_per_channel=quantization_config.weight_per_channel,
            )
            weight_dtype = quantization_config.weight_dtype
            weight_qscheme = _QuantizationScheme.get_qscheme(
                quantization_config.quantization_scheme,
                is_per_channel=quantization_config.weight_per_channel,
            )

            weight_qconfig = cls._create_fake_quantize_partial_from_kwargs(
                True,
                weight_observer,
                weight_dtype,
                weight_qscheme,
                weight_per_channel=quantization_config.weight_per_channel,
                quant_min=quant_min,
                quant_max=quant_max,
                quantization_config=quantization_config,
            )

        return _aoquant.QConfig(activation=activation_qconfig, weight=weight_qconfig)

    def _get_supported_modules(self):
        supported_modules = list(set(_get_supported_modules()) - set(_fixed_qparams_modules))
        # Add _FakeQuantize, NoopObserver to ensure all fused ops have same qconfig
        supported_modules.append(_aoquant.FakeQuantize)
        supported_modules.append(_NoopObserver)
        return supported_modules

    def get_default_qconfig_mapping(
        self,
        quantization_scheme: _QuantizationScheme,
        quantization_config: _Optional[_ModuleLinearQuantizerConfig] = None,
        qconfig: _Optional[_aoquant.QConfig] = None,
    ) -> _aoquant.QConfigMapping:
        """
        Returns default QConfigMapping for a given quantization scheme. If a qconfig is passed,
        it is used as the default qconfig instead.
        """
        supported_modules = self._get_supported_modules()

        qconfig_mapping = _aoquant.QConfigMapping()
        default_qconfig_mapping = _aoquant.get_default_qat_qconfig_mapping()

        # copy qconfig mapping for fixed qparams
        for key in default_qconfig_mapping.object_type_qconfigs:
            if key in _fixed_qparams_modules:
                qconfig_mapping.set_object_type(
                    key, default_qconfig_mapping.object_type_qconfigs[key]
                )

        qconfig = (
            self._get_default_qconfig_from_quantization_scheme(
                quantization_scheme, quantization_config
            )
            if qconfig is None
            else qconfig
        )

        qconfig_mapping.set_global(qconfig)
        for mod_type in supported_modules:
            qconfig_mapping.set_object_type(
                mod_type,
                self._adjust_qconfig_for_module_type(mod_type, qconfig, quantization_config),
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
        qconfig = (
            self._create_qconfig_from_quantization_config(quantization_config.global_config)
            if quantization_config.global_config is not None
            else None
        )

        qconfig_mapping = self.get_default_qconfig_mapping(
            quantization_scheme=quantization_scheme,
            quantization_config=quantization_config.global_config,
            qconfig=qconfig,
        )

        for mod_type, config in quantization_config.module_type_configs.items():
            qconfig = (
                self._create_qconfig_from_quantization_config(config)
                if config is not None
                else config
            )
            qconfig = (
                self._adjust_qconfig_for_module_type(mod_type, qconfig, config)
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
            try:
                submod = model.get_submodule(mod_name)
                qconfig = (
                    self._adjust_qconfig_for_module_type(type(submod), qconfig, config)
                    if qconfig is not None
                    else qconfig
                )
            except AttributeError:
                _logger.warning(
                    f"Could not find a submodule with name {mod_name}. "
                    f"If the name corresponded to something other than a module, "
                    f"this message can be ignored. Otherwise, it's possible "
                    f"the module name was not correctly specified in the config."
                )

            mod_names = self._get_module_names_for_setting_qconfig(model, mod_name)
            for mn in mod_names:
                qconfig_mapping = qconfig_mapping.set_module_name(mn, qconfig)
        return qconfig_mapping
