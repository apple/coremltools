#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict as _OrderedDict
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import List as _List
from typing import NewType as _NewType
from typing import Optional as _Optional
from typing import Union as _Union

import cattrs as _cattrs
import torch as _torch
import torch.nn as _nn
from attr import define as _define
from attr import field as _field
from attrs import validators as _validators

from coremltools.optimize.torch._utils.torch_utils import (
    maybe_convert_str_to_dtype as _maybe_convert_str_to_dtype,
)
from coremltools.optimize.torch.optimization_config import (
    ModuleOptimizationConfig as _ModuleOptimizationConfig,
)
from coremltools.optimize.torch.optimization_config import OptimizationConfig as _OptimizationConfig
from coremltools.optimize.torch.optimization_config import _validate_module_type_keys_factory

# Default advanced options for palettization
DEFAULT_PALETTIZATION_ADVANCED_OPTIONS = {
    "partition_size": 2000000000,
    "cluster_permute": None,
    "palett_max_mem": 1.0,
    "kmeans_max_iter": 3,
    "prune_threshold": 0.0,
    "kmeans_init": "cpu.kmeans++",
    "kmeans_opt1d_threshold": 1024,
    "enforce_zero": False,
    "palett_mode": "dkm",
    "palett_tau": 0.0001,
    "palett_epsilon": 0.0001,
    "palett_lambda": 0.0,
    "add_extra_centroid": False,
    "palett_cluster_tol": 0.05,
}


DEFAULT_PALETTIZATION_OPTIONS = {
    "quant_min": -128,
    "quant_max": 127,
    "dtype": _torch.qint8,
    "cluster_dtype": "f32",
    "weight_threshold": 2048,
    "milestone": 0,
    "quantize_activations": False,
}


_default_palettization_scheme = {
    **DEFAULT_PALETTIZATION_OPTIONS,
    **DEFAULT_PALETTIZATION_ADVANCED_OPTIONS,
}


# Default scheme for palettization
DEFAULT_PALETTIZATION_SCHEME = {
    _nn.Linear: {"n_bits": 4, "cluster_dim": 1, **_default_palettization_scheme},
    _nn.Conv1d: {"n_bits": 2, "cluster_dim": 1, **_default_palettization_scheme},
    _nn.Conv2d: {"n_bits": 2, "cluster_dim": 1, **_default_palettization_scheme},
    _nn.Conv3d: {"n_bits": 2, "cluster_dim": 1, **_default_palettization_scheme},
    _nn.LayerNorm: {"n_bits": 2, "cluster_dim": 1, **_default_palettization_scheme},
    _nn.MultiheadAttention: {"n_bits": 2, "cluster_dim": 1, **_default_palettization_scheme},
    _nn.Embedding: {"n_bits": 2, "cluster_dim": 1, **_default_palettization_scheme},
}


# Pytorch modules from torch.ao.quantization.quantization_mappings.DEFAULT_QAT_MODULE_MAPPINGS that are supported
# for palettization
SUPPORTED_PYTORCH_QAT_MODULES = (_nn.Linear, _nn.Conv2d, _nn.Conv3d)


@_define
class ModuleDKMPalettizerConfig(_ModuleOptimizationConfig):
    r"""
    Configuration class for specifying global and module-level options for palettization
    algorithm implemented in :py:class:`DKMPalettizer`.

    The parameters specified in this config control the DKM algorithm, described in
    `DKM: Differentiable K-Means Clustering Layer for Neural Network Compression
    <https://arxiv.org/abs/2108.12659>`_.

    For most use cases, the only parameters you need to specify are ``n_bits``,  ``weight_threshold``,
    and ``milestone``.

    .. note::
        Most of the parameters in this class are meant for advanced use cases and for further fine-tuning the
        DKM algorithm. The default values usually work for a majority of tasks.

    .. note::
        Change the following parameters only when you use activation quantization in conjunction with
        DKM weight palettization: ``quant_min``, ``quant_max``, ``dtype``, and ``quantize_activations``.

    Args:
        n_bits (:obj:`int`): Number of clusters. The number of clusters used is :math:`2^{n\_bits}`.
            Defaults to ``4`` for linear layers and ``2`` for all other layers.
        weight_threshold (:obj:`int`): A module is only palettized if the number of elements in
            its weight matrix exceeds ``weight_threshold``. Defaults to ``2048``.
        milestone (:obj:`int`): Step or epoch at which palettization begins. Defaults to ``0``.
        cluster_dim (:obj:`int`, ``optional``): The dimension of each cluster. Defaults to ``1``.
        quant_min: (:obj:`int`, ``optional``): The minimum value for each element in the weight clusters if they are
            quantized. Defaults to ``-128``.
        quant_max: (:obj:`int`, ``optional``): The maximum value for each element in the weight clusters if they are
            quantized. Defaults to ``127``
        dtype (:py:class:`torch.dtype`, ``optional``): The ``dtype`` to use for quantizing the activations. Only applies
            when ``quantize_activations`` is ``True``. Defaults to :py:class:`torch.qint8`.
        cluster_dtype (:obj:`str`, ``optional``): ``dtype`` to use for quantizing the clusters. Allowed options are
            ``'i8'``, ``'u8'``, ``'f16'``, ``'bf16'``, ``'f32'``.  Defaults to ``'f32'``, i.e.,
            by default, the clusters aren't quantized.
        quantize_activations (:obj:`bool`, ``optional``): When ``True``, the activation are quantized.
            Defaults to ``False``.
        partition_size (:obj:`int`, ``optional``): partition_size helps in per channel palettization.
            Defaults to ``2000000000``.
        cluster_permute (:obj:`tuple`, ``optional``): Permutation order to apply to weight partitions.
            Defaults to ``None``.
        palett_max_mem (:obj:`float`, ``optional``): Proportion of available GPU memory that should be used for
            palettization. Defaults to ``1.0``.
        kmeans_max_iter (:obj:`int`, ``optional``): Maximum number of differentiable ``k-means`` iterations.
            Defaults to ``3``.
        prune_threshold (:obj:`float`, ``optional``): Hard-shrinks weights between [``-prune_threshold``,
            ``prune_threshold``] to zero. Useful for joint pruning and palettization. Defaults to ``0.0``.
        kmeans_init (:obj:`str`, ``optional``): ``k-means`` algorithm to use. Allowed options are
            ``efficient_kmeans``, ``cpu.kmeans++`` and ``kmeans_pp``. Defaults to ``cpu.kmeans++``.
        kmeans_opt1d_threshold (:obj:`int`, ``optional``): Channel threshold to decide if ``opt1d kmeans``
            should be used. Defaults to ``1024``.
        enforce_zero (:obj:`bool`, ``optional``): If ``True``, enforces the LUT centroid which is closest to
            the origin to be fixed to zero. Defaults to ``False``.
        palett_mode (:obj:`str`, ``optional``): Criteria to calculate attention during ``k-means``. Allowed options are
            ``gsm``, ``dkm`` and ``hard``. Defaults to ``dkm``.
        palett_tau (:obj:`float`, ``optional``): Temperature factor for softmax used in DKM algorithm.
            Defaults to ``0.0001``.
        palett_epsilon (:obj:`float`, ``optional``): Distance threshold for clusters between ``k-means`` iterations.
            Defaults to ``0.0001``.
        palett_lambda (:obj:`float`, ``optional``): Reduces effects of outliers during centroid calculation.
            Defaults to ``0.0``.
        add_extra_centroid (:obj:`bool`, ``optional``): If ``True``, adds an extra centroid to the LUT.
            Defaults to ``False``.
        palett_cluster_tol (:obj:`float`, ``optional``): Tolerance for non-unique centroids in the LUT.
            The higher the number, the more tolerance for non-unique centroids. Defaults to ``0.05``.
    """
    n_bits: _Optional[int] = _field(
        default=None, validator=_validators.optional(_validators.instance_of(int))
    )
    weight_threshold: int = _field(
        default=DEFAULT_PALETTIZATION_OPTIONS["weight_threshold"],
        validator=_validators.instance_of(int),
    )
    milestone: int = _field(
        default=DEFAULT_PALETTIZATION_OPTIONS["milestone"],
        validator=_validators.instance_of(int),
    )
    cluster_dim: _Optional[int] = _field(
        default=None, validator=_validators.optional(_validators.instance_of(int))
    )
    quant_min: int = _field(
        default=DEFAULT_PALETTIZATION_OPTIONS["quant_min"],
        validator=_validators.instance_of(int),
    )
    quant_max: int = _field(
        default=DEFAULT_PALETTIZATION_OPTIONS["quant_max"],
        validator=_validators.instance_of(int),
    )
    dtype: _torch.dtype = _field(
        default=DEFAULT_PALETTIZATION_OPTIONS["dtype"],
        converter=_maybe_convert_str_to_dtype,
        validator=_validators.instance_of(_torch.dtype),
    )
    cluster_dtype: str = _field(
        default=DEFAULT_PALETTIZATION_OPTIONS["cluster_dtype"],
        validator=_validators.instance_of(str),
    )
    quantize_activations: bool = _field(
        default=DEFAULT_PALETTIZATION_OPTIONS["quantize_activations"],
        validator=_validators.instance_of(bool),
    )
    partition_size: int = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["partition_size"],
        validator=_validators.instance_of(int),
    )
    cluster_permute: _Optional[tuple] = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["cluster_permute"],
        validator=_validators.optional(_validators.instance_of(tuple)),
    )
    palett_max_mem: float = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_max_mem"],
        validator=_validators.instance_of(float),
    )
    kmeans_max_iter: int = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["kmeans_max_iter"],
        validator=_validators.instance_of(int),
    )
    prune_threshold: float = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["prune_threshold"],
        validator=_validators.instance_of(float),
    )
    kmeans_init: str = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["kmeans_init"],
        validator=_validators.instance_of(str),
    )
    kmeans_opt1d_threshold: int = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["kmeans_opt1d_threshold"],
        validator=_validators.instance_of(int),
    )
    enforce_zero: bool = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["enforce_zero"],
        validator=_validators.instance_of(bool),
    )
    palett_mode: str = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_mode"],
        validator=_validators.instance_of(str),
    )
    palett_tau: float = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_tau"],
        validator=_validators.instance_of(float),
    )
    palett_epsilon: float = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_epsilon"],
        validator=_validators.instance_of(float),
    )
    palett_lambda: float = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_lambda"],
        validator=_validators.instance_of(float),
    )
    add_extra_centroid: bool = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["add_extra_centroid"],
        validator=_validators.instance_of(bool),
    )
    palett_cluster_tol: float = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_cluster_tol"],
        validator=_validators.instance_of(float),
    )


_default_module_type_configs = _OrderedDict(
    {
        key: ModuleDKMPalettizerConfig.from_dict(val)
        for key, val in DEFAULT_PALETTIZATION_SCHEME.items()
    }
)


_GlobalConfigType = _NewType(
    "GlobalConfigType",
    _Union[
        _Optional[ModuleDKMPalettizerConfig],
        _List[_Optional[ModuleDKMPalettizerConfig]],
    ],
)
_ModuleTypeConfigType = _NewType(
    "ModuleTypeConfigType", _Dict[_Union[_Callable, str], _GlobalConfigType]
)
_ModuleNameConfigType = _NewType(
    "ModuleNameConfigType", _Dict[str, _Optional[ModuleDKMPalettizerConfig]]
)


def _validate_dkm_config_type(instance, attribute, value):
    if value is not None:
        if isinstance(value, list):
            return _validators.deep_iterable(
                member_validator=_validators.optional(
                    _validators.instance_of(ModuleDKMPalettizerConfig)
                ),
                iterable_validator=_validators.instance_of(list),
            )(instance, attribute, value)
        else:
            return _validators.optional(_validators.instance_of(ModuleDKMPalettizerConfig))(
                instance, attribute, value
            )


@_define
class DKMPalettizerConfig(_OptimizationConfig):
    """
    Configuration for specifying how different submodules of a model are palettized by
    :py:class:`DKMPalettizer`.

    The ``module_type_configs`` parameter can accept a list of :py:class:`ModuleDKMPalettizerConfig`
    as values for a given module type. The list can specify
    different parameters for different ``weight_threshold`` values. This is useful if
    you want to apply different configs to layers of the same type with weights of different sizes.

    For example, to use ``4`` -bit palettization for weights with more than ``1000`` elements and
    ``2`` -bit palettization for weights with more than ``300`` but less than ``1000`` elements,
    create a config as follows:

    .. code-block:: python

        custom_config = {
            nn.Conv2d: [
                {"n_bits": 4, "cluster_dim": 4, "weight_threshold": 1000},
                {"n_bits": 2, "cluster_dim": 2, "weight_threshold": 300},
            ]
        }
        config = DKMPalettizerConfig.from_dict({"module_type_configs": custom_config})

    Args:
        global_config (:py:class:`ModuleDKMPalettizerConfig`): Config to be applied globally
            to all supported modules. Missing values are chosen from the default config.
        module_type_configs (:obj:`dict` of :obj:`str` to :py:class:`ModuleDKMPalettizerConfig`):
            Module type level configs applied to a specific module class, such as :py:class:`torch.nn.Linear`.
            The keys can be either strings or module classes. When ``module_type_config`` is set to ``None``
            for a module type, it is not palettized.
        module_name_configs (:obj:`dict` of :obj:`str` to :py:class:`ModuleDKMPalettizerConfig`):
            Module level configs applied to specific modules.
            The name of the module must be a fully qualified name that can be used to fetch it
            from the top level module using the ``module.get_submodule(target)`` method. When
            ``module_name_config`` is set to ``None`` for a module, it is not palettized.
    """

    global_config: _GlobalConfigType = _field(default=None, validator=_validate_dkm_config_type)
    module_type_configs: _ModuleTypeConfigType = _field(
        factory=_OrderedDict,
        validator=_validators.deep_mapping(
            key_validator=_validators.and_(
                _validators.instance_of((str, _Callable)),
                _validate_module_type_keys_factory(list(DEFAULT_PALETTIZATION_SCHEME.keys())),
            ),
            value_validator=_validate_dkm_config_type,
            mapping_validator=_validators.instance_of(dict),
        ),
    )
    module_name_configs: _ModuleNameConfigType = _field(
        factory=_OrderedDict,
        validator=_validators.deep_mapping(
            key_validator=_validators.instance_of(str),
            value_validator=_validators.optional(
                _validators.instance_of(ModuleDKMPalettizerConfig)
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
            self.module_type_configs = _default_module_type_configs
        self._sort_configs_by_weight_threshold(self.global_config)
        for ctype, config in self.module_type_configs.items():
            self.set_module_type(ctype, self._sort_configs_by_weight_threshold(config))
        for name, config in self.module_name_configs.items():
            self.set_module_type(name, self._sort_configs_by_weight_threshold(config))

    @classmethod
    def from_dict(cls, config_dict: _Dict[str, _Any]) -> "DKMPalettizerConfig":
        super().from_dict(config_dict)
        converter = _cattrs.Converter(forbid_extra_keys=True)
        converter.register_structure_hook(_ModuleTypeConfigType, _structure_from_dict_hook)
        converter.register_structure_hook(_ModuleNameConfigType, _structure_from_dict_hook)
        converter.register_structure_hook(_GlobalConfigType, _structure_dkm_config_hook)
        return converter.structure_attrs_fromdict(config_dict, cls)

    @staticmethod
    def _sort_configs_by_weight_threshold(config: _GlobalConfigType):
        if isinstance(config, list):
            return sorted(config, key=lambda x: x.weight_threshold)
        return config


def _structure_dkm_config_hook(
    config_dict: _Union[_List[_Dict[str, _Any]], _Dict[str, _Any]], type: _Any
):
    if isinstance(config_dict, list):
        return [ModuleDKMPalettizerConfig.from_dict(cd) for cd in config_dict]
    return ModuleDKMPalettizerConfig.from_dict(config_dict)


def _structure_from_dict_hook(module_type_dict: _Dict[_Union[_Callable, str], _Any], type: _Any):
    return_dict = _OrderedDict()
    for key, value in module_type_dict.items():
        if value is None:
            return_dict[key] = None
        else:
            return_dict[key] = _structure_dkm_config_hook(value, type)
    return return_dict
