#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# Original implementation from https://github.com/IST-DASLab/sparsegpt
# Copyright 2023 IST Austria Distributed Algorithms and Systems Lab. All Rights Reserved.

import logging as _logging
import re as _re
from collections import OrderedDict as _OrderedDict
from contextlib import contextmanager as _contextmanager
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import Iterable as _Iterable
from typing import List as _List
from typing import NewType as _NewType
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

import cattrs as _cattrs
import torch as _torch
import torch.nn as _nn
from attr import define as _define
from attr import field as _field
from attrs import validators as _validators

from coremltools.optimize.torch._utils.metadata_utils import (
    register_metadata_version as _register_metadata_version,
)
from coremltools.optimize.torch._utils.report_utils import (
    compute_post_training_report as _compute_post_training_report,
)
from coremltools.optimize.torch._utils.torch_utils import get_atomic_layers as _get_atomic_layers
from coremltools.optimize.torch._utils.torch_utils import get_eval_model as _get_eval_model
from coremltools.optimize.torch.base_model_optimizer import (
    BaseDataCalibratedModelOptimizer as _BaseDataCalibratedModelOptimizer,
)
from coremltools.optimize.torch.base_model_optimizer import _Report
from coremltools.optimize.torch.layerwise_compression.algorithms import (
    LayerwiseCompressionAlgorithm as _LayerwiseCompressionAlgorithm,
)
from coremltools.optimize.torch.layerwise_compression.algorithms import (
    LayerwiseCompressionAlgorithmConfig as _LayerwiseCompressionAlgorithmConfig,
)
from coremltools.optimize.torch.layerwise_compression.input_cacher import (
    FirstLayerInputCacher as _FirstLayerInputCacher,
)
from coremltools.optimize.torch.optimization_config import OptimizationConfig as _OptimizationConfig

_logger = _logging.getLogger(__name__)


_ModuleTypeConfigType = _NewType(
    "ModuleTypeConfigType",
    _Dict[_Union[_Callable, str], _Optional[_LayerwiseCompressionAlgorithmConfig]],
)


_SUPPORTED_MODULES = [_torch.nn.Conv2d, _torch.nn.Linear]


@_define
class LayerwiseCompressorConfig(_OptimizationConfig):
    """
    Configuration class for specifying how different submodules of a model are
    compressed by :py:class:`LayerwiseCompressor`. Note that only sequential models are supported.

    Args:
        layers (:obj:`list` of :py:class:`torch.nn.Module` or :obj:`str`): List of layers
            to be compressed. When items in the list are :obj:`str`, the string can be a regex
            or the exact name of the module. The layers listed should be immediate child modules
            of the parent container :py:class:`torch.nn.Sequential` model, and they should be contiguous.
            That is, the output of layer ``n`` should be the input to layer ``n+1``.
        global_config (:py:class:`ModuleGPTQConfig` or :py:class:`ModuleSparseGPTConfig`): Config to be applied globally
            to all supported modules. Missing values are chosen from the default config.
        module_type_configs (:obj:`dict` of :obj:`str` to :py:class:`ModuleGPTQConfig` or :py:class:`ModuleSparseGPTConfig`):
            Module type configs applied to a specific module class, such as :py:class:`torch.nn.Linear`.
            The keys can be either strings or module classes.
        module_name_configs (:obj:`dict` of :obj:`str` to :py:class:`ModuleGPTQConfig` or :py:class:`ModuleSparseGPTConfig`):
            Module-level configs applied to specific modules. The name of the module must either be a regex or
            a fully qualified name that can be used to fetch it from the top level module using the
            ``module.get_submodule(target)`` method.
        input_cacher (:obj:`str` or :py:class:`FirstLayerInputCacher`): Cacher object that caches inputs which are then
            fed to the first layer set up for compression.
        calibration_nsamples (:obj:`int`): Number of samples to be used for calibration.
    """

    layers: _Optional[_Union[_List[_Union[_nn.Module, str]], _nn.ModuleList]] = _field(
        default=None,
        validator=_validators.optional(
            _validators.deep_iterable(
                member_validator=_validators.instance_of((_nn.Module, str)),
                iterable_validator=_validators.instance_of((list, _nn.ModuleList)),
            )
        ),
    )
    global_config: _Optional[_LayerwiseCompressionAlgorithmConfig] = _field(
        default=None,
        validator=_validators.optional(
            _validators.instance_of(_LayerwiseCompressionAlgorithmConfig)
        ),
    )
    module_type_configs: _ModuleTypeConfigType = _field(
        factory=_OrderedDict,
        validator=_validators.deep_mapping(
            key_validator=_validators.instance_of((str, _Callable)),
            value_validator=_validators.optional(
                _validators.instance_of(_LayerwiseCompressionAlgorithmConfig)
            ),
            mapping_validator=_validators.instance_of(dict),
        ),
    )
    module_name_configs: _Dict[str, _Optional[_LayerwiseCompressionAlgorithmConfig]] = _field(
        factory=_OrderedDict,
        validator=_validators.deep_mapping(
            key_validator=_validators.instance_of(str),
            value_validator=_validators.optional(
                _validators.instance_of(_LayerwiseCompressionAlgorithmConfig)
            ),
            mapping_validator=_validators.instance_of(dict),
        ),
    )
    input_cacher: str = _field(default="default", converter=_FirstLayerInputCacher.get_class)
    calibration_nsamples: int = _field(default=128, validator=_validators.instance_of(int))

    @classmethod
    def from_dict(cls, config_dict: _Dict[str, _Any]) -> "LayerwiseCompressorConfig":
        super().from_dict(config_dict)
        converter = _cattrs.Converter(forbid_extra_keys=True)
        converter.register_structure_hook(
            _Optional[_Union[_List[_Union[_nn.Module, str]], _nn.ModuleList]],
            lambda obj, type: obj,
        )
        converter.register_structure_hook(
            _LayerwiseCompressionAlgorithmConfig,
            lambda obj, type: _LayerwiseCompressionAlgorithmConfig.get_class(
                obj["algorithm"]
            ).from_dict(obj),
        )
        converter.register_structure_hook(
            _ModuleTypeConfigType,
            lambda module_type_config, type: {
                key: _LayerwiseCompressionAlgorithmConfig.get_class(val["algorithm"]).from_dict(val)
                if val is not None
                else None
                for key, val in module_type_config.items()
            },
        )
        return converter.structure_attrs_fromdict(config_dict, cls)

    def get_layers(self, model: _nn.Module):
        if self.layers is None:
            for module_name, module in model.named_children():
                yield module_name, module
        else:
            yielded = set()
            for module_name, module in model.named_modules(remove_duplicate=True):
                for layer in self.layers:
                    if isinstance(layer, str) and _re.fullmatch(layer, module_name):
                        if module_name not in yielded:
                            yielded.add(module_name)
                            yield module_name, module
                    elif module == layer:
                        if module_name not in yielded:
                            yielded.add(module_name)
                            yield module_name, module


@_contextmanager
def _set_torch_flags():
    # TODO: Copied from original implementation; determine if this is necessary
    cuda_matmul_tf32 = _torch.backends.cuda.matmul.allow_tf32
    cudnn_allow_tf32 = _torch.backends.cudnn.allow_tf32
    try:
        _torch.backends.cuda.matmul.allow_tf32 = False
        _torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        _torch.backends.cuda.matmul.allow_tf32 = cuda_matmul_tf32
        _torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32


class LayerwiseCompressor(_BaseDataCalibratedModelOptimizer):
    """
    A post-training compression algorithm which compresses a sequential model layer by layer
    by minimizing the quantization error while quantizing the weights. The implementation
    supports two variations of this algorithm:

    1) `Generative Pre-Trained Transformer Quantization (GPTQ) <https://arxiv.org/pdf/2210.17323.pdf>`_
    2) `Sparse Generative Pre-Trained Transformer (SparseGPT) <https://arxiv.org/pdf/2301.00774.pdf>`_

    At a high level, it compresses weights of a model layer by layer
    by minimizing the L2 norm of the difference between the original activations and
    activations obtained from compressing the weights of a layer. The activations
    are computed using a few samples of training data.

    Only sequential models are supported, where the output of one layer feeds into the
    input of the next layer.

    For HuggingFace models, disable the ``use_cache`` config. This is used to speed up decoding,
    but to generalize forward pass for :py:class:`LayerwiseCompressor` algorithms across all
    model types, the behavior must be disabled.

    Example:

            .. code-block:: python

                import torch.nn as nn
                from coremltools.optimize.torch.layerwise_compression import (
                    LayerwiseCompressor,
                    LayerwiseCompressorConfig,
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

                dataloder = load_calibration_data()

                # initialize the quantizer
                config = LayerwiseCompressorConfig.from_dict(
                    {
                        "global_config": {
                            "algorithm": "gptq",
                            "weight_dtype": "int4",
                        },
                        "input_cacher": "default",
                        "calibration_nsamples": 16,
                    }
                )

                compressor = LayerwiseCompressor(model, config)

                compressed_model = compressor.compress(dataloader)

    Args:
        model (:obj:`torch.nn.Module`): Module to be compressed.
        config (:py:class:`LayerwiseCompressorConfig`): Config that specifies how
            different submodules in the model will be compressed.
    """

    _supported_modules: _Tuple = tuple(_SUPPORTED_MODULES)

    def __init__(self, model: _nn.Module, config: LayerwiseCompressorConfig):
        super().__init__(model, config)
        self._input_cacher = self._config.input_cacher(
            self._model,
            self._config.layers,
        )

    @staticmethod
    def _forward_layer(layer, inputs, kwarg_inputs, outputs) -> _List:
        """
        Perform forward pass on layer and store outputs.
        """
        for j, inp in enumerate(inputs):
            if isinstance(inp, _torch.Tensor):
                inp = (inp,)
            outputs[j] = layer(*inp, **kwarg_inputs)
        return outputs

    def _get_cached_inputs(
        self, dataloader: _Iterable, device: str
    ) -> _Tuple[_List[_torch.Tensor], _Dict[str, _torch.Tensor]]:
        """
        Cache the inputs and keyword arguments up till the first layer set up for compression
        """
        inputs, kwarg_inputs = self._input_cacher.cache(
            dataloader=dataloader,
            nsamples=self._config.calibration_nsamples,
            device=device,
        )
        return inputs, kwarg_inputs

    def _get_layers_to_compress(self) -> _Dict[str, _nn.Module]:
        """
        Returns a list of layers to be compressed
        """
        return self._config.get_layers(self._model)

    def _init_and_config_layer(
        self, atomic_layer_name, atomic_layer
    ) -> _Optional[_LayerwiseCompressionAlgorithm]:
        """
        Initializes and configures the compression algorithm for a given
        atomic layer. Returns the initialized and configured compression
        algorithm object
        """
        layer_config = self._config.get_module_config(atomic_layer_name, atomic_layer)
        if layer_config is not None:
            algo_class = _LayerwiseCompressionAlgorithm.get_class(layer_config.algorithm)
            try:
                return algo_class(atomic_layer, layer_config)
            except ValueError as error:
                _logger.info(f"Skipping compression for {atomic_layer_name}. Reason={error}")
        return None

    def _register_activation_processing_hook(
        self, atomic_layer, compressor_obj
    ) -> _torch.utils.hooks.RemovableHandle:
        """
        Registers a forward hook on the layer for performing computation
        using the inputs to acquire statistics. Returns the handle for
        the forward hook
        """

        def activation_processing_hook(_, inp, out):
            compressor_obj.add_batch(inp[0].data, out.data)

        return atomic_layer.register_forward_hook(activation_processing_hook)

    @_torch.no_grad()
    def _compress_impl(self, dataloader: _Iterable, device: str) -> _nn.Module:
        """
        Compresses a model layerwise using the following steps:
        1) Compute inputs to the first layer which is set up for compression using input cacher
        2) For each layer, find submodules which are supported for compression and install compression
           hooks.
        3) Run forward pass through each layer, compute activation statistics and use them to
           compress weights.
        4) Compute updated outputs using compressed weights to propagate quantization error
           to the next layer and set them up as inputs to next layer.
        """
        inputs, kwarg_inputs = self._get_cached_inputs(dataloader, device)
        outputs = [None for _ in inputs]

        # compress the layers one by one
        for layer_idx, (parent_layer_name, layer) in enumerate(self._get_layers_to_compress()):
            layer.to(device)
            atomic_layers_dict = _get_atomic_layers(
                layer,
                layer_types=self._supported_modules,
                name_prefix=parent_layer_name,
            )

            # dict mapping layer_name -> compression algorithm object
            compression_algo_objects_dict = dict()

            # dict mapping layer_name -> forward hook handle
            layer_hooks = []

            for atomic_layer_name, atomic_layer in atomic_layers_dict.items():
                obj = self._init_and_config_layer(atomic_layer_name, atomic_layer)

                if obj is not None:
                    compression_algo_objects_dict[atomic_layer_name] = obj

                    layer_hooks.append(self._register_activation_processing_hook(atomic_layer, obj))

            # Compute statistics on the activations using the activation processing hooks
            outputs = self._forward_layer(
                layer,
                inputs,
                kwarg_inputs,
                outputs,
            )

            # Remove the activation processing hooks
            for h in layer_hooks:
                h.remove()

            # compress the layers
            _logger.info(f"Layer {layer_idx}")
            for (
                atomic_layer_name,
                compressor_algo,
            ) in compression_algo_objects_dict.items():
                _logger.info(f"Compressing {atomic_layer_name}")
                compressor_algo.compress()
                compressor_algo.cleanup()

            del compression_algo_objects_dict

            # feed the previous layer's outputs to this layer
            outputs = self._forward_layer(
                layer,
                inputs,
                kwarg_inputs,
                outputs,
            )

            # free memory
            layer.cpu()
            del layer
            _torch.cuda.empty_cache()

            # interchange inputs and outputs
            inputs, outputs = outputs, inputs

        _register_metadata_version(self._model)
        return self._model

    def compress(self, dataloader: _Iterable, device: str, inplace: bool = False) -> _nn.Module:
        """
        Compresses model using samples from ``dataloader``.

        Args:
            dataloader (:py:class:`Iterable`): An iterable where each element
                is an input to the model to be compressed.
            device (:obj:`str`): Device string for device to run compression on.
            inplace (:obj:`bool`): If ``True``, model transformations are carried out in-place and
                the original module is mutated, otherwise a copy of the model is mutated and returned.
                Defaults to ``False``.
        """
        self._model = super().compress(dataloader=dataloader, inplace=inplace)
        with _get_eval_model(self._model):
            with _set_torch_flags():
                return self._compress_impl(dataloader, device)

    def report(self) -> _Report:
        return _compute_post_training_report(
            self._uncompressed_model,
            self._model,
            supported_modules=self._supported_modules,
        )
