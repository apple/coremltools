#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging as _logging
from collections import OrderedDict as _OrderedDict
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import NewType as _NewType
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

import cattrs as _cattrs
import torch as _torch
from attr import define as _define
from attr import field as _field
from attrs import validators as _validators

from coremltools.optimize.torch._utils.k_means import KMeansConfig as _KMeansConfig
from coremltools.optimize.torch._utils.k_means import (
    KMeansSupportedModulesRegistry as _KMeansSupportedModulesRegistry,
)
from coremltools.optimize.torch._utils.k_means import ParallelKMeans as _ParallelKMeans
from coremltools.optimize.torch._utils.k_means import SequentialKMeans as _SequentialKMeans
from coremltools.optimize.torch._utils.report_utils import (
    compute_post_training_report as _compute_post_training_report,
)
from coremltools.optimize.torch._utils.torch_utils import (
    maybe_convert_str_to_dtype as _maybe_convert_str_to_dtype,
)
from coremltools.optimize.torch._utils.torch_utils import (
    maybe_convert_str_to_mod_type as _maybe_convert_str_to_mod_type,
)
from coremltools.optimize.torch._utils.validation_utils import (
    validate_param_config as _validate_param_config,
)
from coremltools.optimize.torch.base_model_optimizer import (
    BasePostTrainingModelOptimizer as _BasePostTrainingModelOptimizer,
)
from coremltools.optimize.torch.base_model_optimizer import _Report
from coremltools.optimize.torch.optimization_config import (
    ModuleOptimizationConfig as _ModuleOptimizationConfig,
)
from coremltools.optimize.torch.optimization_config import OptimizationConfig as _OptimizationConfig
from coremltools.optimize.torch.optimization_config import (
    PalettizationGranularity,
    _structure_from_dict_hook_factory,
)

_logger = _logging.getLogger(__name__)


@_define
class ModulePostTrainingPalettizerConfig(_ModuleOptimizationConfig):
    """
    Configuration class for specifying global and module-level palettization options for
    :py:class:`PostTrainingPalettizerConfig` algorithm.

    Args:
        n_bits (:obj:`int`): Number of bits to use for palettizing the weights. Defaults to ``4``.
        lut_dtype (:py:class:`torch.dtype`): The dtype to use for representing each element in lookup tables.
            When value is ``None``, no quantization is performed. Supported values are :py:class:`torch.int8` and
            :py:class:`torch.uint8`. Defaults to ``None``.
        granularity (:py:class:`PalettizationGranularity`) – Granularity for palettization.
            One of ``per_tensor`` or ``per_grouped_channel``. Defaults to ``per_tensor``.
        group_size (:obj:`int`): Specify the number of channels in a group.
            Only effective when granularity is ``per_grouped_channel``.
        channel_axis (:obj:`int`): Specify the channel axis to form a group of channels.
            Only effective when granularity is ``per_grouped_channel``. Defaults to output channel axis.
        cluster_dim (:obj:`int`): The dimension of centroids for each lookup table.
            The centroid is a scalar by default. When ``cluster_dim > 1``, it indicates 2-D clustering,
            and each ``cluster_dim`` length of weight vectors along the output channel are palettized
            using the same 2-D centroid. The length of each entry in the lookup tables is equal to ``cluster_dim``.
        enable_per_channel_scale (:obj:`bool`): When set to ``True``, weights are normalized along the output channels
            using per-channel scales before being palettized. This is not supported with ``cluster_dim > 1``.

    This class supports two different configurations to structure the palettization:

    1. **Per-tensor palettization**:  This is the default configuration where the whole tensor shares a single lookup
    table. The ``granularity`` is set to ``per_tensor``, and ``group_size`` is ``None``.

    2. **Per-grouped-channel palettization**: In this configuration, the number of channels ``group_size`` along
    ``channel_axis`` share the same lookup table. For example, for a weight matrix of shape ``(16, 25)``, if we provide
    ``group_size = 8``, the shape of the lookup table would be ``(2, 2^n_bits)``.

    .. note::
        Grouping is currently only supported along either the input or output channel axis.
    """

    n_bits: _Optional[int] = _field(
        default=4, validator=_validators.optional(_validators.instance_of(int))
    )
    lut_dtype: _torch.dtype = _field(
        default=None,
        converter=lambda val: _maybe_convert_str_to_dtype(val) if val else val,
        validator=_validators.optional(
            [
                _validators.instance_of(_torch.dtype),
                _validators.in_([_torch.int8, _torch.uint8]),
            ]
        ),
    )
    granularity: PalettizationGranularity = _field(
        default="per_tensor",
        converter=PalettizationGranularity,
        validator=_validators.in_(PalettizationGranularity),
    )
    group_size: _Optional[int] = _field(
        default=None, validator=_validators.optional(_validators.instance_of(int))
    )
    channel_axis: int = _field(
        default=0,
        validator=_validators.optional([_validators.instance_of(int), _validators.in_([0, 1])]),
    )
    cluster_dim: _Optional[int] = _field(
        default=None, validator=_validators.optional(_validators.instance_of(int))
    )
    enable_per_channel_scale: _Optional[bool] = _field(
        default=False, validator=_validators.optional(_validators.instance_of(bool))
    )

    @group_size.validator
    def per_grouped_channel_granularity(self, attribute, value):
        if self.granularity == PalettizationGranularity.per_grouped_channel:
            assert (
                value is not None
            ), "group_size has to be specified along with per_grouped_channel granularity."
            assert value > 0, "group_size should be greater than zero"
        else:
            assert value is None, "group_size can't be specified along with per_tensor granularity."

    @cluster_dim.validator
    def no_per_channel_scale(self, attribute, value):
        if value and value > 1:
            assert (
                self.enable_per_channel_scale == False
            ), f"Enabling per_channel_scale is not supported for cluster_dim={value} larger than 1"


_ModuleTypeConfigType = _NewType(
    "ModuleTypeConfigType",
    _Dict[_Union[_Callable, str], _Optional[ModulePostTrainingPalettizerConfig]],
)


@_define
class PostTrainingPalettizerConfig(_OptimizationConfig):
    """
    Configuration class for specifying how different submodules of a model
    should be post-training palettized by :py:class:`PostTrainingPalettizer`.

    Args:
        global_config (:py:class:`ModulePostTrainingPalettizerConfig`): Config to be applied globally
            to all supported modules.
        module_type_configs (:obj:`dict` of :obj:`str` to :py:class:`ModulePostTrainingPalettizerConfig`):
            Module type configs applied to a specific module class, such as :py:class:`torch.nn.Linear`.
            The keys can be either strings or module classes.
        module_name_configs (:obj:`dict` of :obj:`str` to :py:class:`ModulePostTrainingPalettizerConfig`):
            Module name configs applied to specific modules. This can be a dictionary with module names pointing to their
            corresponding :py:class:`ModulePostTrainingPalettizerConfig`.
    """

    global_config: _Optional[ModulePostTrainingPalettizerConfig] = _field(
        default=None,
        validator=_validators.optional(_validators.instance_of(ModulePostTrainingPalettizerConfig)),
    )
    module_type_configs: _ModuleTypeConfigType = _field(
        factory=_OrderedDict,
        validator=_validators.deep_mapping(
            key_validator=_validators.instance_of((str, _Callable)),
            value_validator=_validators.optional(
                _validators.instance_of(ModulePostTrainingPalettizerConfig)
            ),
            mapping_validator=_validators.instance_of(dict),
        ),
    )
    module_name_configs: _Dict[str, _Optional[ModulePostTrainingPalettizerConfig]] = _field(
        factory=_OrderedDict,
        validator=_validators.deep_mapping(
            key_validator=_validators.instance_of(str),
            value_validator=_validators.optional(
                _validators.instance_of(ModulePostTrainingPalettizerConfig)
            ),
            mapping_validator=_validators.instance_of(dict),
        ),
    )

    def __attrs_post_init__(self):
        if (
            self.global_config is None
            and len(self.module_type_configs) == 0
            and len(self.module_name_configs) == 0
        ):
            self.global_config = ModulePostTrainingPalettizerConfig()
        self.module_type_configs = {
            _maybe_convert_str_to_mod_type(key): val
            for key, val in self.module_type_configs.items()
        }

    @classmethod
    def from_dict(cls, config_dict: _Dict[str, _Any]) -> "PostTrainingPalettizerConfig":
        super().from_dict(config_dict)
        converter = _cattrs.Converter(forbid_extra_keys=True)
        converter.register_structure_hook(
            _ModuleTypeConfigType,
            _structure_from_dict_hook_factory(ModulePostTrainingPalettizerConfig),
        )
        return converter.structure_attrs_fromdict(config_dict, cls)


class PostTrainingPalettizer(_BasePostTrainingModelOptimizer):
    """
    Perform post-training palettization on a torch model. Post palettization, all the weights in supported
    layers point to elements in a lookup table after performing a k-means operation.

    Example:

            .. code-block:: python

                import torch.nn as nn
                from coremltools.optimize.torch.palettization import (
                    PostTrainingPalettizerConfig,
                    PostTrainingPalettizer,
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

                # initialize the palettizer
                config = PostTrainingPalettizerConfig.from_dict(
                    {
                        "global_config": {
                            "n_bits": 4,
                        },
                    }
                )

                ptpalettizer = PostTrainingPalettizer(model, config)
                palettized_model = ptpalettizer.compress()

    Args:
        model (:obj:`torch.nn.Module`): Module to be compressed.
        config (:py:class:`PostTrainingPalettizerConfig`): Config that specifies how
            different submodules in the model will be palettized.
    """

    _supported_modules: _Tuple = _KMeansSupportedModulesRegistry.get_supported_modules()

    def __init__(self, model: _torch.nn.Module, config: PostTrainingPalettizerConfig = None):
        config = PostTrainingPalettizerConfig() if config is None else config
        super().__init__(model, config)

    def compress(self, num_kmeans_workers: int = 1, inplace: bool = False) -> _torch.nn.Module:
        """
        The compress method performs a `k-means` operation on all supported modules.
        Args:
            num_kmeans_workers (:obj:`int`): Number of worker processes used for
                performing post-training palettization. Defaults to ``1``.
            inplace (:obj:`bool`): If ``True``, model transformations are carried out in-place and
                the original module is mutated, otherwise a copy of the model is mutated and returned.
                Defaults to ``False``.
        """
        self._model = super().compress(inplace=inplace)
        kmeans_config_dict = dict()
        for name, submodule in self._model.named_modules():
            submod_config = self._config.get_module_config(name, submodule)
            if submod_config is None:
                continue

            k_means_module_cls = _KMeansSupportedModulesRegistry.get_kmeans_module(submodule)
            if k_means_module_cls is None:
                continue

            for param_name in k_means_module_cls.parameter_names:
                # Validate configuration for parameter
                param = submodule.get_parameter(param_name)
                updated_config = _validate_param_config(
                    name + "." + param_name,
                    param,
                    submodule,
                    submod_config,
                    ["palettization_group_size", "palettization_cluster_dim"],
                )
                if not updated_config:
                    continue

                if name not in kmeans_config_dict:
                    kmeans_config_dict[name] = {}

                kmeans_config_dict[name][param_name] = _KMeansConfig(
                    n_bits=updated_config.n_bits,
                    axis=updated_config.channel_axis,
                    lut_dtype=updated_config.lut_dtype,
                    block_size=updated_config.group_size,
                    cluster_dim=updated_config.cluster_dim,
                    enable_per_channel_scale=updated_config.enable_per_channel_scale,
                )

        if num_kmeans_workers > 1:
            return _ParallelKMeans.cluster_weights(
                self._model, kmeans_config_dict, num_workers=num_kmeans_workers
            )
        else:
            return _SequentialKMeans.cluster_weights(self._model, kmeans_config_dict)

    def report(self) -> _Report:
        return _compute_post_training_report(
            self._uncompressed_model,
            self._model,
            supported_modules=self._supported_modules,
        )
