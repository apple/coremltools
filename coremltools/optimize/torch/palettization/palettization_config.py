#  Copyright (c) 2024, Apple Inc. All rights reserved.
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
from coremltools.optimize.torch.optimization_config import (
    PalettizationGranularity,
    _deprecated_field,
    _validate_module_type_keys_factory,
)

# Default advanced options for palettization
DEFAULT_PALETTIZATION_ADVANCED_OPTIONS = {
    "cluster_permute": None,
    "palett_max_mem": 1.0,
    "kmeans_max_iter": 3,
    "prune_threshold": 1e-7,
    "kmeans_init": "auto",
    "kmeans_opt1d_threshold": 1024,
    "enforce_zero": False,
    "palett_mode": "dkm",
    "palett_tau": 0.0001,
    "palett_epsilon": 0.0001,
    "palett_lambda": 0.0,
    "add_extra_centroid": False,
    "palett_cluster_tol": 0.0,
    "palett_min_tsize": 64 * 1024,
    "palett_unique": False,
    "palett_shard": False,
    "palett_batch_mode": False,
    "palett_dist": False,
    "per_channel_scaling_factor_scheme": "min_max",
    "percentage_palett_enable": 1.0,
    "kmeans_batch_threshold": 4,
    "kmeans_n_init": 100,
    "zero_threshold": 1e-7,
    "kmeans_error_bnd": 0.0,
    "channel_axis": 0,
}


DEFAULT_PALETTIZATION_OPTIONS = {
    "quant_min": -128,
    "quant_max": 127,
    "dtype": _torch.qint8,
    "lut_dtype": "f32",
    "weight_threshold": 2048,
    "milestone": 0,
    "quantize_activations": False,
    "enable_per_channel_scale": False,
    "granularity": "per_tensor",
    "group_size": None,
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
            its weight matrix exceeds ``weight_threshold``. If there are multiple weights in a
            module (like :py:class:`torch.nn.MultiheadAttention`), all of them must have
            more elements than the ``weight_threshold`` for the module to be palettized.
            Defaults to ``2048``.
        granularity (:py:class:`PalettizationGranularity`) – Granularity for palettization.
            One of ``per_tensor`` or ``per_grouped_channel``. Defaults to ``per_tensor``.
        group_size (:obj:`int`): Specify the number of channels in a group.
            Only effective when granularity is ``per_grouped_channel``.
        channel_axis (:obj:`int`): Specify the channel axis to form a group of channels.
            Only effective when granularity is ``per_grouped_channel``. Defaults to output channel axis. For now, only
            output channel axis is supported by DKM.
        enable_per_channel_scale (:obj:`bool`): When set to ``True``, per channel scaling is used along the channel
            dimension.
        milestone (:obj:`int`): Step or epoch at which palettization begins. Defaults to ``0``.
        cluster_dim (:obj:`int`, ``optional``): The dimension of each cluster.
        quant_min: (:obj:`int`, ``optional``): The minimum value for each element in the weight clusters if they are
            quantized. Defaults to ``-128``.
        quant_max: (:obj:`int`, ``optional``): The maximum value for each element in the weight clusters if they are
            quantized. Defaults to ``127``
        dtype (:py:class:`torch.dtype`, ``optional``): The ``dtype`` to use for quantizing the activations. Only applies
            when ``quantize_activations`` is ``True``. Defaults to :py:class:`torch.qint8`.
        lut_dtype (:obj:`str`, ``optional``): ``dtype`` to use for quantizing the clusters. Allowed options are
            ``'i8'``, ``'u8'``, ``'f16'``, ``'bf16'``, ``'f32'``.  Defaults to ``'f32'``, i.e.,
            by default, the clusters aren't quantized.
        quantize_activations (:obj:`bool`, ``optional``): When ``True``, the activation are quantized.
            Defaults to ``False``.
        cluster_permute (:obj:`tuple`, ``optional``): Permutation order to apply to weight partitions.
            Defaults to ``None``.
        palett_max_mem (:obj:`float`, ``optional``): Proportion of available GPU memory that should be used for
            palettization. Defaults to ``1.0``.
        kmeans_max_iter (:obj:`int`, ``optional``): Maximum number of differentiable ``k-means`` iterations.
            Defaults to ``3``.
        prune_threshold (:obj:`float`, ``optional``): Hard-shrinks weights between [``-prune_threshold``,
            ``prune_threshold``] to zero. Useful for joint pruning and palettization. Defaults to ``1e-7``.
        kmeans_init (:obj:`str`, ``optional``): ``k-means`` algorithm to use. Allowed options are
            ``opt1d``, ``cpu.kmeans++`` and ``kmeans++``. Defaults to ``auto``.
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
            The higher the number, the more tolerance for non-unique centroids. Defaults to ``0.0``.
        palett_min_tsize (:obj:`int`, ``optional``): Weight threshold beyond which to use custom packing and unpacking
            hook for autograd. Defaults to ``64*1024``.
        palett_unique (:obj:`bool`, ``optional``): If ``True``, reduces the attention map by leveraging the fact that
            FP16 only has ``2^16`` unique values. Useful for Large Models like LLMs where attention maps can be huge.
            Defaults to ``False``. More details can be found `eDKM: An Efficient and Accurate Train-time Weight
            Clustering for Large Language Models <https://arxiv.org/pdf/2309.00964.pdf>`_ .
        palett_shard (:obj:`bool`, ``optional``): If ``True``, the index list is sharded across GPUs.
            Defaults to ``False``. More details can be found `eDKM: An Efficient and Accurate Train-time Weight
            Clustering for Large Language Models <https://arxiv.org/pdf/2309.00964.pdf>`_ .
        palett_batch_mode (:obj:`bool`, ``optional``): If ``True``, performs batch DKM across different partitions
            created for different blocks. Defaults to ``False``. More details can be found `eDKM: An Efficient and Accurate Train-time Weight
            Clustering for Large Language Models <https://arxiv.org/pdf/2309.00964.pdf>`_ .
        palett_dist (:obj:`bool`, ``optional``): If ``True``, performs distributed distance calculation in batch_mode if
            distributed torch is available. Defaults to ``False``.
        per_channel_scaling_factor_scheme (:obj:`str`, ``optional``): Criteria to calculate the
            ``per_channel_scaling_factor``. Allowed options are ``min_max`` and ``abs``. Defaults to ``min_max``.
        percentage_palett_enable (:obj:`float`, ``optional``): Percentage partitions to enable for DKM.
                    Defaults to ``1.0``.
        kmeans_batch_threshold (:obj:`int`, ``optional``): Threshold to decide at what num_partitions to go through with
            sharded centroids list. num_partitions is calculated by dividing the channel size with the group_size
            provided. If the kmeans_batch_threshold, the algorithm resorts to performing distirbuted kmeans for lower
            partition numbers, given that num_partition number of GPUs are available. Defaults to ``4``.
        kmeans_n_init (:obj:`int`, ``optional``): Number of time the k-means algorithm will be run with different
            centroid seeds. The final results will be the best output of kmeans_n_init consecutive runs in terms of inertia.
        zero_threshold (:obj:`int`, ``optional``): Zero threshold to be used to decide min value of clamp for softmax
            . Defaults to ``1e-7``.
        kmeans_error_bnd (:obj:`float`, ``optional``): Distance threshold to decide at what distance between parameters
            and clusters to stop the kmeans operation. Defaults to ``0.0``.

    This class supports few different configurations to structure the palettization:

    1. **Per-tensor palettization**:  This is the default configuration where the whole tensor shares a single look-up
    table. The ``granularity`` is set to ``per_tensor`` and ``group_size`` is ``None``.

    2. **Per-grouped-channel palettization**: In this configuration, ``group_size`` number of channels along
    ``channel_axis`` share the same look-up table. For example, for a weight matrix of shape ``(16, 25)``, if we provide
     ``group_size = 8``, the shape of the look-up table would be ``(2, 2^n_bits)``.

    NOTE: Currently grouping is only supported along output channel axis.
    """
    n_bits: _Optional[int] = _field(
        default=None, validator=_validators.optional(_validators.instance_of(int))
    )
    weight_threshold: int = _field(
        default=DEFAULT_PALETTIZATION_OPTIONS["weight_threshold"],
        validator=_validators.instance_of(int),
    )
    granularity: PalettizationGranularity = _field(
        default=DEFAULT_PALETTIZATION_OPTIONS["granularity"],
        converter=PalettizationGranularity,
        validator=_validators.in_(PalettizationGranularity),
    )
    group_size: _Optional[int] = _field(
        default=DEFAULT_PALETTIZATION_OPTIONS["group_size"],
        validator=_validators.optional(_validators.instance_of(int)),
    )
    channel_axis: int = _field(
        default=0,
        validator=_validators.optional([_validators.instance_of(int), _validators.in_([0])]),
    )
    enable_per_channel_scale: bool = _field(
        default=DEFAULT_PALETTIZATION_OPTIONS["enable_per_channel_scale"],
        validator=_validators.instance_of(bool),
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
        validator=[
            _validators.instance_of(_torch.dtype),
            _validators.in_([_torch.qint8, _torch.quint8, _torch.float32]),
        ],
    )
    lut_dtype: str = _field(
        default=DEFAULT_PALETTIZATION_OPTIONS["lut_dtype"],
        validator=_validators.instance_of(str),
    )
    quantize_activations: bool = _field(
        default=DEFAULT_PALETTIZATION_OPTIONS["quantize_activations"],
        validator=_validators.instance_of(bool),
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
    palett_min_tsize: int = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_min_tsize"],
        validator=_validators.instance_of(int),
    )
    palett_unique: bool = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_unique"],
        validator=_validators.instance_of(bool),
    )
    palett_shard: bool = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_shard"],
        validator=_validators.instance_of(bool),
    )
    palett_batch_mode: bool = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_batch_mode"],
        validator=_validators.instance_of(bool),
    )
    palett_dist: bool = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_dist"],
        validator=_validators.instance_of(bool),
    )
    per_channel_scaling_factor_scheme: str = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["per_channel_scaling_factor_scheme"],
        validator=_validators.and_(
            _validators.instance_of(str), _validators.in_(["min_max", "abs"])
        ),
    )
    percentage_palett_enable: float = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["percentage_palett_enable"],
        validator=_validators.instance_of(float),
    )
    kmeans_batch_threshold: int = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["kmeans_batch_threshold"],
        validator=_validators.instance_of(int),
    )
    kmeans_n_init: int = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["kmeans_n_init"],
        validator=_validators.instance_of(int),
    )
    zero_threshold: float = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["zero_threshold"],
        validator=_validators.instance_of(float),
    )
    kmeans_error_bnd: float = _field(
        default=DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["kmeans_error_bnd"],
        validator=_validators.instance_of(float),
    )
    partition_size: int = _deprecated_field(
        message=(
            "partition_size is being deprecated and will be removed in "
            "future versions. Please use group_size parameter instead."
        )
    )
    cluster_dtype: str = _deprecated_field(
        message=(
            "cluster_dtype is being deprecated and will be removed in "
            "future versions. Please use lut_dtype parameter instead."
        )
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
            value_validator=_validate_dkm_config_type,
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
            self.set_module_name(name, self._sort_configs_by_weight_threshold(config))

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
