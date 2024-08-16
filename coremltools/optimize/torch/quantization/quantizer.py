#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy as _copy
import logging as _logging
from typing import Any as _Any
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Type as _Type

import torch as _torch
import torch.ao.quantization as _aoquant
from torch.ao.quantization.fx.custom_config import ConvertCustomConfig as _ConvertCustomConfig
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig as _PrepareCustomConfig
from torch.ao.quantization.quantize_fx import convert_to_reference_fx as _convert_to_reference_fx

import coremltools.optimize.torch.quantization.modules.qat_modules as _qat
from coremltools.optimize.torch._utils.math_utils import rmse_error as _rmse_error
from coremltools.optimize.torch._utils.metadata_utils import (
    register_metadata_version as _register_metadata_version,
)
from coremltools.optimize.torch._utils.torch_utils import get_eval_model as _get_eval_model
from coremltools.optimize.torch.base_model_optimizer import (
    BaseTrainingTimeModelOptimizer as _BaseTrainingTimeModelOptimizer,
)
from coremltools.optimize.torch.base_model_optimizer import _Report
from coremltools.optimize.torch.quantization._backend_config import (
    get_backend_config as _get_backend_config,
)
from coremltools.optimize.torch.quantization._backend_config import (
    get_supported_modules as _get_supported_modules,
)
from coremltools.optimize.torch.quantization._configure import (
    QATConfigurationHandler as _QATConfigurationHandler,
)
from coremltools.optimize.torch.quantization._qconfig_mapping import _QConfigMappingBuilder
from coremltools.optimize.torch.quantization._utils import (
    is_per_channel_quant as _is_per_channel_quant,
)
from coremltools.optimize.torch.quantization._utils import is_symmetric_quant as _is_symmetric_quant
from coremltools.optimize.torch.quantization._utils import (
    pre_apply_weight_quant as _pre_apply_weight_quant,
)
from coremltools.optimize.torch.quantization._utils import (
    register_compression_metadata as _register_compression_metadata,
)
from coremltools.optimize.torch.quantization.quantization_config import (
    LinearQuantizerConfig as _LinearQuantizerConfig,
)
from coremltools.optimize.torch.quantization.quantization_config import (
    ModuleLinearQuantizerConfig as _ModuleLinearQuantizerConfig,
)

_logger = _logging.getLogger(__name__)


class Quantizer(_BaseTrainingTimeModelOptimizer):
    pass


class LinearQuantizer(Quantizer):
    """
    Perform quantization aware training (QAT) of models. This algorithm simulates the effects of
    quantization during training, by quantizing and dequantizing the weights and/or activations during
    the model's forward pass. The forward and backward pass computations are conducted in ``float`` dtype,
    however, these ``float`` values follow the constraints imposed by ``int8`` and ``quint8`` dtypes. Thus,
    this algorithm adjusts the model's weights while closely simulating the numerics which get
    executed during quantized inference, allowing model's weights to adjust to quantization
    constraints.

    For more details, please refer to  `Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only
    Inference <https://arxiv.org/pdf/1712.05877.pdf>`_.

    Example:

            .. code-block:: python

                import torch.nn as nn
                from coremltools.optimize.torch.quantization import (
                    LinearQuantizer,
                    LinearQuantizerConfig,
                )

                model = nn.Sequential(
                    OrderedDict(
                        {
                            "conv": nn.Conv2d(1, 20, (3, 3)),
                            "relu1": nn.ReLU(),
                            "conv2": nn.Conv2d(20, 20, (3, 3)),
                            "relu2": nn.ReLU(),
                        }
                    )
                )

                loss_fn = define_loss()

                # initialize the quantizer
                config = LinearQuantizerConfig.from_dict(
                    {
                        "global_config": {
                            "quantization_scheme": "symmetric",
                            "milestones": [0, 100, 400, 400],
                        }
                    }
                )

                quantizer = LinearQuantizer(model, config)

                # prepare the model to insert FakeQuantize layers for QAT
                model = quantizer.prepare()

                # use quantizer in your PyTorch training loop
                for inputs, labels in data:
                    output = model(inputs)
                    loss = loss_fn(output, labels)
                    loss.backward()
                    optimizer.step()
                    quantizer.step()

                # convert operations to their quantized counterparts using parameters learned via QAT
                model = quantizer.finalize(inplace=True)

    Args:
        model (:obj:`torch.nn.Module`): Module to be trained.
        config (:py:class:`_LinearQuantizerConfig`): Config that specifies how
            different submodules in the model will be quantized.
            Default config is used when passed as ``None``.
    """
    _supported_modules: _Tuple = tuple(_get_supported_modules())
    _qconfig_mapping_builder_cls: _Type = _QConfigMappingBuilder
    _qat_configuration_handler_cls: _Type = _QATConfigurationHandler

    def __init__(self, model: _torch.nn.Module, config: _Optional[_LinearQuantizerConfig] = None):
        config = _LinearQuantizerConfig() if config is None else config
        super().__init__(model, config)
        global_config = self._construct_global_config()
        self._is_prepared = False
        self._quantization_scheme = global_config.quantization_scheme
        self._milestones = global_config.milestones
        qmapping_builder = self._qconfig_mapping_builder_cls()
        self._qconfig_mapping = qmapping_builder.get_qconfig_mapping_from_quantization_config(
            model=self._model,
            quantization_config=self._config,
            quantization_scheme=self._quantization_scheme,
        )

    def _construct_global_config(self) -> _ModuleLinearQuantizerConfig:
        if self._config.global_config is not None:
            return self._config.global_config
        for _, config in self._config.module_type_configs.items():
            if config is not None:
                return config
        for _, config in self._config.module_name_configs.items():
            if config is not None:
                return config
        return _ModuleLinearQuantizerConfig()

    def prepare(self, example_inputs: _Tuple[_Any, ...], inplace: bool = False) -> _torch.nn.Module:
        """
        Prepares the model for quantization aware training by inserting
        :py:class:`torch.ao.quantization.FakeQuantize` layers in the model in appropriate places.

        Args:
            example_inputs (:obj:`Tuple[Any, ...]`): Example inputs for forward function of the model,
                tuple of positional args (keyword args can be passed as positional args as well)
            inplace (:obj:`bool`): If ``True``, model transformations are carried out in-place and
                the original module is mutated, otherwise a copy of the model is mutated and returned.

        .. note::
            This method uses `prepare_qat_fx method <https://pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_fx.prepare_qat_fx.html#torch.ao.quantization.quantize_fx.prepare_qat_fx>`_
            to insert quantization layers and the returned model is a :py:class:`torch.fx.GraphModule`.
            Some models, like those with dynamic control flow, may not be trace-able into a
            :py:class:`torch.fx.GraphModule`. Please follow directions in `Limitations of Symbolic Tracing <https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing>`_
            to update your model first before using :py:class:`LinearQuantizer` algorithm.

        """
        if self._is_prepared:
            _logger.warning(
                "Model has already been prepared for QAT. This API call "
                "will be a no-op."
            )
            return self._model
        model = self._get_model_for_compression(inplace=inplace)
        model.train()
        prepare_custom_config = _PrepareCustomConfig().set_non_traceable_module_names(
            self._config.non_traceable_module_names
        )
        prepare_custom_config = prepare_custom_config.set_preserved_attributes(
            self._config.preserved_attributes
        )
        qat_handler = self._qat_configuration_handler_cls(
            prepare_custom_config=prepare_custom_config,
            qconfig_mapping=self._qconfig_mapping,
            backend_config=_get_backend_config(),
            quantization_scheme=self._quantization_scheme,
        )
        prepared_model = qat_handler.prepare(model, example_inputs)
        if self._milestones is not None:
            prepared_model.apply(_aoquant.disable_observer)
            prepared_model.apply(_aoquant.disable_fake_quant)
        self._model = prepared_model
        self._is_prepared = True
        return prepared_model

    def step(self):
        """
        Steps through the milestones defined for this quantizer.

        The first milestone corresponds to enabling observers, the second
        to enabling fake quantization simulation, the third
        to disabling observers, and the last to freezing batch norm statistics.

        .. note::
            If milestones argument is set as ``None``, this method is a no-op.

        .. note::
            In order to not use a particular milestone, its value can be set as ``-1``.
        """
        if not self._is_prepared:
            _logger.warning(
                "Model has not been prepared for QAT. This API call "
                "will be a no-op. prepare method must be called before "
                "a call to the step method."
            )
            return
        if self._milestones is None:
            return
        else:
            if self._step_count == self._milestones[0]:
                self._model.apply(_aoquant.enable_observer)
            if self._step_count == self._milestones[1]:
                self._model.apply(_aoquant.enable_fake_quant)
            if self._step_count == self._milestones[2]:
                self._model.apply(_aoquant.disable_observer)
            if self._step_count == self._milestones[3]:
                self._model.apply(_qat.freeze_bn_stats)
        self._step_count += 1

    def finalize(
        self, model: _Optional[_torch.nn.Module] = None, inplace: bool = False
    ) -> _torch.nn.Module:
        """
        Prepares the model for export.

        Args:
            model (:py:class:`_torch.nn.Module`): Model to be finalized.
            inplace (:obj:`bool`): If ``True``, model transformations are carried out in-place and
                the original module is mutated; otherwise, a copy of the model is mutated and returned.

        .. note::
            Once the model is finalized with ``in_place = True``, it may not be
            runnable on the GPU.
        """
        if not self._is_prepared:
            _logger.warning(
                "Model has not been prepared for QAT. This API call "
                "will be a no-op. prepare method must be called before "
                "a call to the finalize method."
            )
            return self._model
        if model is None:
            model = self._model
        if not inplace:
            model = _copy.deepcopy(model)
        model.eval()
        convert_custom_config = _ConvertCustomConfig().set_preserved_attributes(
            self._config.preserved_attributes
        )
        finalized_model = _convert_to_reference_fx(
            model,
            convert_custom_config=convert_custom_config,
            qconfig_mapping=self._qconfig_mapping,
            backend_config=_get_backend_config(),
        )

        # PyTorch fx QAT does not properly handle the clipping of < 8-bit weights during
        # finalization so have to apply the utility method below after finalization to clip
        # the de-quantized weights.
        _pre_apply_weight_quant(finalized_model)

        _register_metadata_version(finalized_model)
        for name, submodule in finalized_model.named_modules(remove_duplicate=True):
            if hasattr(submodule, "weight_scale"):
                submod_config = self._config.get_module_config(name, submodule)
                _register_compression_metadata(submodule, submod_config)

        if model is None:
            self._model = finalized_model
        return finalized_model

    def report(self) -> _Report:
        """
        Returns a dictionary with important statistics related to current state of quantization.
        Each key in the dictionary corresponds to a module name, and the
        value is a dictionary containing the statistics such as scale, zero point,
        number of parameters, and so on.

        Note that error will be nan and #params will be -1 for activations.
        """

        report = _Report()
        with _get_eval_model(self._model) as model:
            with _torch.no_grad():
                for name, module in model.named_modules(remove_duplicate=True):

                    if (
                        hasattr(module, "weight_fake_quant")
                        and module.weight_fake_quant is not None
                    ):
                        module_summary = dict()

                        module_summary["type"] = "weight"

                        module_summary["device"] = module.weight.device

                        qscheme = module.weight_fake_quant.qscheme
                        module_summary["qscheme"] = (
                            "symmetric" if _is_symmetric_quant(qscheme) else "affine"
                        )
                        module_summary["per_channel"] = _is_per_channel_quant(qscheme)

                        qweight = module.weight_fake_quant.forward(module.weight.detach())

                        module_summary["dtype"] = module.weight_fake_quant.dtype
                        module_summary["qmin"] = module.weight_fake_quant.quant_min
                        module_summary["qmax"] = module.weight_fake_quant.quant_max

                        module_summary["error"] = _rmse_error(
                            module.weight.detach(), qweight
                        ).item()
                        module_summary["#params"] = int(_torch.numel(qweight))
                        report[name] = module_summary

                    elif (
                        not name.endswith(".weight_fake_quant")
                        and isinstance(module, _aoquant.FakeQuantize)
                        and hasattr(module, "activation_post_process")
                        and module.activation_post_process is not None
                    ):
                        module_summary = dict()

                        module_summary["type"] = "activation"

                        scale, zp = module.activation_post_process.calculate_qparams()
                        module_summary["device"] = scale.device

                        qscheme = module.qscheme
                        module_summary["qscheme"] = (
                            "symmetric" if _is_symmetric_quant(qscheme) else "affine"
                        )
                        module_summary["per_channel"] = _is_per_channel_quant(qscheme)

                        module_summary["dtype"] = module.dtype
                        module_summary["qmin"] = module.quant_min
                        module_summary["qmax"] = module.quant_max

                        module_summary["error"] = float("nan")
                        module_summary["#params"] = -1
                        report[name] = module_summary
        return report
