#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging as _logging
import queue as _queue
from abc import abstractmethod as _abstractmethod
from typing import Any as _Any
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Type as _Type
from typing import Union as _Union

import torch as _torch
import torch.multiprocessing as _mp
from attr import define as _define

from coremltools._deps import _kmeans1d
from coremltools.converters.mil.mil.ops.defs.iOS18 import (
    constexpr_blockwise_shift_scale as _quantize_op,
)
from coremltools.optimize.coreml._utils import compute_qparams as _compute_qparams
from coremltools.optimize.torch._utils.metadata_utils import (
    CompressionMetadata as _CompressionMetadata,
)
from coremltools.optimize.torch._utils.metadata_utils import (
    register_metadata_version as _register_metadata_version,
)
from coremltools.optimize.torch._utils.python_utils import ClassRegistryMixin as _ClassRegistryMixin
from coremltools.optimize.torch._utils.torch_utils import (
    get_atomic_layers,
    get_n_bits_from_dtype,
    is_signed_dtype,
)
from coremltools.optimize.torch.palettization._efficient_kmeans import _EfficientKMeans

_logger = _logging.getLogger(__name__)


@_define(frozen=True)
class KMeansConfig:
    n_bits: int = 4
    axis: int = 0
    lut_dtype: _torch.dtype = None
    block_size: _Optional[int] = None
    cluster_dim: _Optional[int] = None
    enable_per_channel_scale: bool = False
    mask: _Optional[_torch.Tensor] = None
    importance: _Optional[_torch.Tensor] = None


class KMeansSupportedModulesRegistry(_ClassRegistryMixin):
    """
    A registry of :py:class:`KMeansModule` classes
    """

    REGISTRY: _Dict[str, _Type["KMeansModule"]]

    @classmethod
    def get_kmeans_module(cls, module: _torch.nn.Module) -> _Optional[_Type["KMeansModule"]]:
        """
        Returns the :py:class:`KMeansModule` class which implements k-means
        for the given module.
        """
        for _, layer_cls in cls.REGISTRY.items():
            if layer_cls.is_supported_module(module):
                return layer_cls
        return None

    @classmethod
    def get_supported_modules(cls) -> _Tuple[_Type[_torch.nn.Module]]:
        """
        Returns all supported module types for k-means.
        """
        return tuple(layer_cls.layer_type for _, layer_cls in cls.REGISTRY.items())


class KMeansModule:
    """
    An interface for adding support for a given module class for running
    k-means. Implements methods to retrieve parameters which can be clustered
    and to update them with new values after clustering.
    """

    layer_type: _Type[
        _torch.nn.Module
    ]  # The layer type which this interface supports clustering for
    parameter_names: _List[str] = []  # List of parameters which are clustered for this layer type

    def __init_subclass__(cls):
        KMeansSupportedModulesRegistry.register(cls.__name__)(cls)

    def __init__(self, module: _torch.nn.Module, config: _Dict[str, KMeansConfig]):
        self.module = module
        self.config = config
        self._parameter_metadata = None
        self._init_parameter_metadata()

    @_abstractmethod
    def _init_parameter_metadata(self):
        """
        Initialize metadata for k-means clustering for this layer type.
        The metadata is a dictionary from parameter name to a dictionary
        of metadata name and its value. This method should add the shape of
        the parameters as the metadata for each parameter which
        should be clustered.
        """

    @_abstractmethod
    def _get_parameters_impl(self) -> _Dict[str, _torch.Tensor]:
        """
        Returns a dictionary of parameter name to the parameter tensor
        which should be clustered for this layer type.
        """

    @_abstractmethod
    def _update_parameters_impl(self, param_name: str, new_value: _torch.Tensor):
        """
        Update the parameter corresponding to this parameter name with the
        new value after reshaping to original parameter shape.
        """

    @_abstractmethod
    def _reshape_for_kmeans(self, param_name: str, value: _torch.Tensor) -> _torch.Tensor:
        """
        Reshape any value of original parameter shape to flattened shape for k-means.
        """

    @_abstractmethod
    def _reshape_to_original(self, param_name: str, value: _torch.Tensor) -> _torch.Tensor:
        """
        Reshape any value flattened for k-means back to original parameter shape.
        """

    def _compute_lut_and_indices(self, param_name: str, param: _torch.Tensor):
        """
        Compute LUT and indices from parameter.
        For 4-bit palettization and param shape (32, 16, 3, 3),
        Case-1: If block_size = 4 and axis = 0, then LUT has shape (8, 1, 1, 1, 16, 1)
        Case-2: If block_size = 4 and axis = 1, then LUT has shape (1, 4, 1, 1, 16, 1)
        Case-3: If cluster_dim = 4, then LUT has shape (1, 1, 1, 1, 16, 4)
        """
        axis = self.config[param_name].axis
        num_channels = param.shape[axis]
        mask = self.config[param_name].mask
        block_size = self.config[param_name].block_size
        block_size = num_channels if block_size is None else block_size
        cluster_dim = self.config[param_name].cluster_dim
        orig_param_shape = self._parameter_metadata[param_name]["shape"]
        cluster_dim = 1 if cluster_dim is None else cluster_dim

        lut, indices = [], []
        if cluster_dim == 1:
            # Scalar palettization
            for block_idx in range(0, num_channels, block_size):
                if axis == 0:
                    lut_idx, ind_idx = _torch.unique(
                        param[block_idx : block_idx + block_size, :],
                        return_inverse=True,
                    )
                else:
                    lut_idx, ind_idx = _torch.unique(
                        param[:, block_idx : block_idx + block_size],
                        return_inverse=True,
                    )

                # Ensure param was correctly palettized
                # Unless a mask was applied, number of unique values cannot exceed 2^nbits
                max_unique_val = 2 ** self.config[param_name].n_bits
                assert mask is not None or len(lut_idx) <= max_unique_val, (
                    f"Found more than expected unique values in {self.module} "
                    f"for {param_name}, expected <= {max_unique_val}, found = {len(lut_idx)}"
                )
                # Pad lut with zeros if fewer than 2^n_bit unique values are found
                if len(lut_idx) < max_unique_val:
                    padded_lut_idx = _torch.zeros(max_unique_val)
                    padded_lut_idx[: len(lut_idx)] = lut_idx
                    lut_idx = padded_lut_idx

                lut.append(lut_idx)
                indices.append(ind_idx)

            lut = _torch.stack(lut).unsqueeze(1 - axis).unsqueeze(-1)
            indices = _torch.cat(indices, dim=axis)
            indices = self._reshape_to_original(param_name, indices)
        else:
            # Vector palettization
            # Reshape param for 2D clustering
            if axis == 0:
                param_reshaped = param.transpose(0, 1).reshape(-1, cluster_dim)
            else:
                param_reshaped = param.reshape(-1, cluster_dim)
            lut, indices = _torch.unique(param_reshaped, dim=0, return_inverse=True)

            # Undo reshaping in indices done for 2D clustering
            if axis == 0:
                indices = indices.reshape(param.shape[0] // cluster_dim, param.shape[1])
            else:
                indices = indices.reshape(param.shape[0], param.shape[1] // cluster_dim)

        # Incorporate param dimensions in lut shape
        for i in range(len(orig_param_shape) - lut.dim() + 2):
            lut = lut.unsqueeze(-3)

        return lut, indices

    def _scale_by_per_channel_scale(self, param_name: str, param: _torch.Tensor) -> _torch.Tensor:
        """
        Compute per channel scales for scaling the parameter in the range ``[-1, 1]``
        and store them in the parameter metadata. Also scale the parameter using
        the computed scales.
        """
        if self.config[param_name].enable_per_channel_scale:
            flattened_param = param.flatten(1)
            per_channel_scale = _torch.max(_torch.abs(flattened_param), dim=1, keepdim=True).values
            # Handle zero scales
            per_channel_scale[per_channel_scale == 0] = 1
            flattened_param /= per_channel_scale
            param = flattened_param.reshape(param.shape)
            self._parameter_metadata[param_name]["per_channel_scale"] = per_channel_scale
        return param

    def _get_compression_metadata(
        self, param_name: str, param: _torch.Tensor
    ) -> _CompressionMetadata:
        """
        Return compression metadata to be stored in the model for this parameter
        """
        metadata = _CompressionMetadata(param_name)
        compression_type = ["palettization"]
        # LUT
        metadata.lut, _ = self._compute_lut_and_indices(param_name, param)
        # Per channel scale
        if self.config[param_name].enable_per_channel_scale:
            per_channel_scale = self._parameter_metadata[param_name]["per_channel_scale"]
            reshaped_param = self._reshape_to_original(param_name, param)
            for _ in range(reshaped_param.dim() - per_channel_scale.dim()):
                per_channel_scale = per_channel_scale.unsqueeze(-1)
            metadata.palettization_scale = per_channel_scale
        # LUT quantization
        if self.config[param_name].lut_dtype is not None:
            dtype = self.config[param_name].lut_dtype
            compression_type.append("quantization")
            metadata.quantization_n_bits = get_n_bits_from_dtype(dtype)
            scale = self._parameter_metadata[param_name]["lut_quantization_scale"]
            # match scale rank to lut rank
            for i in range(metadata.lut.dim() - scale.dim()):
                scale = scale.unsqueeze(-1)
            metadata.quantization_scale = scale
            zp = self._parameter_metadata[param_name]["lut_quantization_zp"]
            if zp is not None:
                # match zp rank to lut rank
                for i in range(metadata.lut.dim() - zp.dim()):
                    zp = zp.unsqueeze(-1)
                metadata.zero_point = zp
        # vector axis for cluster_dim > 1
        cluster_dim = self.config[param_name].cluster_dim
        if cluster_dim is not None and cluster_dim > 1:
            metadata.vector_axis = self.config[param_name].axis
        # Compression type
        metadata.compression_type = compression_type
        return metadata

    def _register_compression_metadata(self, param_name: str, param: _torch.Tensor):
        """
        Register compression metadata on the model so that it can be serialized.
        """
        metadata = self._get_compression_metadata(param_name, param)
        metadata.register(self.module)

    def _unscale_by_per_channel_scale(self, param_name: str, param: _torch.Tensor) -> _torch.Tensor:
        """
        Re-scale the parameter with ``param_name`` back to its original range by multiplying
        per channel scales.
        """
        if self.config[param_name].enable_per_channel_scale:
            per_channel_scale = self._parameter_metadata[param_name]["per_channel_scale"]
            flattened_param = param.flatten(1)
            flattened_param *= per_channel_scale
            param = flattened_param.reshape(param.shape)
        return param

    @classmethod
    def is_supported_module(cls, module: _torch.nn.Module) -> bool:
        """
        Returns ``True`` if clustering this module is supported by this interface.
        """
        return isinstance(module, cls.layer_type)

    def get_parameters(self) -> _Dict[str, _torch.Tensor]:
        """
        Returns a dictionary of parameter name to the parameter tensor
        which should be clustered for this layer type. Scales the weights
        in the range ``[-1, 1]`` if ``per_channel_scale`` is enabled.
        """
        return self._get_parameters_impl()

    def update_parameters(self, param_name: str, new_value: _torch.Tensor):
        """
        Update the parameter corresponding to this parameter name with the
        new value.
        """
        self._register_compression_metadata(param_name, new_value)
        self._update_parameters_impl(param_name, new_value)

    def get_param_config(self, param_name: str, param: _torch.Tensor) -> KMeansConfig:
        """
        Returns KMeansConfig for the specified parameter
        """
        config = self.config[param_name]
        block_size = param.shape[config.axis] if config.block_size is None else config.block_size
        cluster_dim = 1 if config.cluster_dim is None else config.cluster_dim
        importance = self._reshape_for_kmeans(param_name, config.importance)
        mask = self._reshape_for_kmeans(param_name, config.mask)

        return KMeansConfig(
            n_bits=config.n_bits,
            axis=config.axis,
            lut_dtype=config.lut_dtype,
            block_size=block_size,
            cluster_dim=cluster_dim,
            enable_per_channel_scale=config.enable_per_channel_scale,
            mask=mask,
            importance=importance,
        )


class Linear(KMeansModule):
    layer_type: _Type = _torch.nn.Linear
    parameter_names: _List[str] = ["weight"]

    def _init_parameter_metadata(self):
        self._parameter_metadata = {
            "weight": {
                "shape": self.module.weight.shape,
            }
        }

    def _get_parameters_impl(self):
        scaled_param = self._scale_by_per_channel_scale("weight", self.module.weight.data)
        return {"weight": self._reshape_for_kmeans("weight", scaled_param)}

    def _update_parameters_impl(self, param_name: str, new_value: _torch.Tensor):
        param = self._reshape_to_original(param_name, new_value)
        self.module.weight.data = self._unscale_by_per_channel_scale(param_name, param)

    def _reshape_for_kmeans(self, param_name: str, value: _torch.Tensor) -> _torch.Tensor:
        return value

    def _reshape_to_original(self, param_name: str, value: _torch.Tensor) -> _torch.Tensor:
        return value


class Embedding(KMeansModule):
    layer_type: _Type = _torch.nn.Embedding
    parameter_names: _List[str] = ["weight"]

    def _init_parameter_metadata(self):
        self._parameter_metadata = {
            "weight": {
                "shape": self.module.weight.shape,
            }
        }

    def _get_parameters_impl(self):
        scaled_param = self._scale_by_per_channel_scale("weight", self.module.weight.data)
        return {"weight": self._reshape_for_kmeans("weight", scaled_param)}

    def _update_parameters_impl(self, param_name: str, new_value: _torch.Tensor):
        param = self._reshape_to_original(param_name, new_value)
        self.module.weight.data = self._unscale_by_per_channel_scale(param_name, param)

    def _reshape_for_kmeans(self, param_name: str, value: _torch.Tensor) -> _torch.Tensor:
        return value

    def _reshape_to_original(self, param_name: str, value: _torch.Tensor) -> _torch.Tensor:
        return value


class Conv2d(KMeansModule):
    layer_type: _Type = _torch.nn.Conv2d
    parameter_names: _List[str] = ["weight"]

    def _init_parameter_metadata(self):
        self._parameter_metadata = {
            "weight": {
                "shape": self.module.weight.shape,
            }
        }

    def _get_parameters_impl(self):
        scaled_param = self._scale_by_per_channel_scale("weight", self.module.weight.data)
        return {"weight": self._reshape_for_kmeans("weight", scaled_param)}

    def _update_parameters_impl(self, param_name: str, new_value: _torch.Tensor):
        param = self._reshape_to_original(param_name, new_value)
        self.module.weight.data = self._unscale_by_per_channel_scale(param_name, param)

    def _reshape_for_kmeans(self, param_name: str, value: _torch.Tensor) -> _torch.Tensor:
        if value is None:
            return value

        if self.config[param_name].axis == 0:
            new_value = value.flatten(1)
        else:
            new_value = value.transpose(0, 1).flatten(1).transpose(0, 1)

        return new_value

    def _reshape_to_original(self, param_name: str, value: _torch.Tensor) -> _torch.Tensor:
        if value is None:
            return value

        weight_shape = self._parameter_metadata[param_name]["shape"]
        if self.config[param_name].axis == 0:
            new_value = value.reshape(weight_shape)
        else:
            new_value = (
                value.transpose(0, 1)
                .reshape(
                    (
                        weight_shape[1],
                        weight_shape[0],
                        weight_shape[2],
                        weight_shape[3],
                    )
                )
                .transpose(0, 1)
            )
        return new_value


class MultiheadAttention(KMeansModule):
    layer_type: _Type = _torch.nn.MultiheadAttention
    parameter_names: _List[str] = ["in_proj_weight"]

    def _init_parameter_metadata(self):
        self._parameter_metadata = {
            "in_proj_weight": {
                "shape": self.module.in_proj_weight.shape,
            },
        }

    def _get_parameters_impl(self):
        scaled_param = self._scale_by_per_channel_scale(
            "in_proj_weight", self.module.in_proj_weight.data
        )
        return {"in_proj_weight": self._reshape_for_kmeans("in_proj_weight", scaled_param)}

    def _update_parameters_impl(self, param_name: str, new_value: _torch.Tensor):
        param = self._reshape_to_original(param_name, new_value)
        self.module.in_proj_weight.data = self._unscale_by_per_channel_scale(param_name, param)

    def _reshape_for_kmeans(self, param_name: str, value: _torch.Tensor) -> _torch.Tensor:
        return value

    def _reshape_to_original(self, param_name: str, value: _torch.Tensor) -> _torch.Tensor:
        return value


class KMeans:
    @classmethod
    def _get_block_to_cluster(cls, weight: _torch.Tensor, config: KMeansConfig, block_idx: int):
        """
        Extract block weight to cluster.
        """
        if config.axis == 0:
            block_importance = (
                config.importance[block_idx : block_idx + config.block_size, :]
                if config.importance is not None
                else None
            )
            block_weight = weight[block_idx : block_idx + config.block_size, :]
            block_mask = (
                config.mask[block_idx : block_idx + config.block_size, :].flatten()
                if config.mask is not None
                else None
            )
        else:
            block_importance = (
                config.importance[:, block_idx : block_idx + config.block_size]
                if config.importance is not None
                else None
            )
            block_weight = weight[:, block_idx : block_idx + config.block_size]
            block_mask = (
                config.mask[:, block_idx : block_idx + config.block_size].flatten()
                if config.mask is not None
                else None
            )
        return block_weight, block_importance, block_mask

    @classmethod
    def _cluster_weights_with_masking(
        cls,
        block_weight: _torch.Tensor,
        block_importance: _torch.Tensor,
        block_mask: _torch.Tensor,
        config: KMeansConfig,
    ) -> _Tuple[_Optional[_torch.Tensor], _Optional[_torch.Tensor]]:
        """
        Cluster block weight with clustering only applied to masked weight elements.
        """
        num_clusters = 2**config.n_bits

        block_weight_flatten = block_weight.flatten()
        block_weight_flatten_masked = block_weight_flatten[block_mask]

        if len(block_weight_flatten_masked) > 0:
            if block_importance is not None:
                block_importance_flatten = block_importance.flatten()
                kmeans_results = _kmeans1d.cluster(
                    block_weight_flatten_masked.numpy(),
                    num_clusters,
                    weights=block_importance_flatten[block_mask].numpy(),
                )
            else:
                kmeans_results = _kmeans1d.cluster(
                    block_weight_flatten_masked.numpy(), num_clusters
                )
            return _torch.tensor(kmeans_results.centroids), _torch.tensor(kmeans_results.clusters)

        return None, None

    @classmethod
    def _cluster_weights_1d(
        cls,
        block_weight: _torch.Tensor,
        block_importance: _torch.Tensor,
        config: KMeansConfig,
    ) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        """
        Cluster weights such that each centroid is a 1d scalar, i.e., cluster_dim == 1.
        """
        num_clusters = 2**config.n_bits

        block_weight_flatten = block_weight.flatten()
        if block_importance is not None:
            block_importance_flatten = block_importance.flatten()
            kmeans_results = _kmeans1d.cluster(
                block_weight_flatten.numpy(),
                num_clusters,
                weights=block_importance_flatten.numpy(),
            )
        else:
            kmeans_results = _kmeans1d.cluster(block_weight_flatten.numpy(), num_clusters)
        return _torch.tensor(kmeans_results.centroids), _torch.tensor(kmeans_results.clusters)

    @classmethod
    def _cluster_weights_2d(
        cls,
        block_weight: _torch.Tensor,
        block_importance: _torch.Tensor,
        config: KMeansConfig,
        rank: int,
    ) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        """
        Cluster weights such that each centroid is a 2d vector, i.e., cluster_dim > 1.
        If axis = 0: vectors are chosen with elements along the output channel dimension.
        Example:
            weight = [
                       [1, 2, 3, 4],
                       [5, 6, 7, 8],
                    ]
            axis = 0
            ========
            clustering is done for the 4 points below:
                [
                  [1, 5], ---> point 1
                  [2, 6], ---> point 2
                  [3, 7], ---> point 3
                  [4, 8], ---> point 4
                ]
            axis = 1
            ========
            clustering is done for the 4 points below:
                [
                  [1, 2], ---> point 1
                  [3, 4], ---> point 2
                  [5, 6], ---> point 3
                  [7, 8], ---> point 4
                ]
        """
        num_clusters = 2**config.n_bits

        # Convert weight from N-D to 2-D.
        # Apply 2-D k-means clustering on 2-D weights.
        if config.axis == 0:
            # (C_out, C_in, H, W) -> (C_in, C_out, H, W)
            # (C_in, C_out, H, W) -> (C_in * H * W * C_out // cluster_dim, cluster_dim)
            weight_2d = block_weight.transpose(0, 1).reshape(-1, config.cluster_dim)
            importance_2d = (
                block_importance.transpose(0, 1)
                .reshape(-1, config.cluster_dim)
                .sum(dim=1, keepdim=True)
                if block_importance is not None
                else None
            )
        else:
            # (C_out, C_in, H, W) -> (C_in, C_out * H * W) -> (C_out * H * W, C_in)
            # (C_out * H * W, C_in) -> (C_out * H * W * C_in // cluster_dim, cluster_dim)
            weight_2d = block_weight.reshape(-1, config.cluster_dim)
            importance_2d = (
                block_importance.reshape(-1, config.cluster_dim).sum(dim=1, keepdim=True)
                if block_importance is not None
                else None
            )

        # Optionally move tensors to GPU
        if _torch.cuda.is_available():
            device_id = rank % _torch.cuda.device_count()
            weight_2d = weight_2d.to(f"cuda:{device_id}")
            importance_2d = (
                importance_2d.to(f"cuda:{device_id}") if importance_2d is not None else None
            )

        kmeans_results = _EfficientKMeans(
            n_clusters=num_clusters,
            init="kmeans++",
            n_init=5,
            max_iter=300,
        ).fit(weight_2d, sample_weight=importance_2d)

        weight_2d.cpu()
        if importance_2d is not None:
            importance_2d.cpu()

        return kmeans_results.cluster_centers_.cpu(), kmeans_results.labels_.cpu()

    @classmethod
    def _update_clustered_block_weight(
        cls,
        block_weight: _torch.Tensor,
        block_mask: _torch.Tensor,
        depalett_block_weight: _torch.Tensor,
        new_weight: _torch.Tensor,
        config: KMeansConfig,
        block_idx: int,
    ):
        """
        Write back clustered weight in new weight.
        """
        block_weight_flatten = block_weight.flatten()

        if block_mask is not None:
            new_block_weight = block_weight_flatten.clone()
            new_block_weight[block_mask] = depalett_block_weight
            new_block_weight = new_block_weight.reshape(block_weight.shape)
        else:
            if config.axis == 1 or config.cluster_dim == 1:
                new_block_weight = depalett_block_weight.reshape(block_weight.shape)
            else:
                # need to reshape back for cluster_dim > 1 and axis = 0
                new_block_weight = depalett_block_weight.reshape(
                    block_weight.shape[1], block_weight.shape[0]
                ).transpose(0, 1)

        if config.axis == 0:
            new_weight[block_idx : block_idx + config.block_size, :] = new_block_weight
        else:
            new_weight[:, block_idx : block_idx + config.block_size] = new_block_weight

    @classmethod
    @_torch.no_grad()
    def _cluster_weights_worker(
        cls,
        rank: int,
        work_q: _Union[_mp.Queue, _queue.Queue],
        results_q: _Union[_mp.Queue, _queue.Queue],
    ):
        while True:
            try:
                (
                    layer_name,
                    weight_name,
                    weight,
                    config,
                ) = work_q.get_nowait()
            except _queue.Empty:
                break

            _logger.info(f"Starting to process layer {layer_name}")

            new_weight = _torch.zeros_like(weight, dtype=weight.dtype)

            _logger.info(
                f"Number of blocks in {layer_name}.{weight_name}: {weight.shape[config.axis] // config.block_size}"
            )

            lut_quant_scale = []
            lut_quant_zp = []

            for block_idx in range(0, weight.shape[config.axis], config.block_size):
                block_weight, block_importance, block_mask = cls._get_block_to_cluster(
                    weight, config, block_idx
                )

                if block_mask is not None:
                    if config.cluster_dim == 1:
                        centroids, clusters = cls._cluster_weights_with_masking(
                            block_weight, block_importance, block_mask, config
                        )
                    else:
                        # Masking not supported for cluster_dim > 1
                        centroids, clusters = None, None
                        _logger.info(
                            f"Skipping palettizing layer: {layer_name} with "
                            f"cluster_dim: {config.cluster_dim} and mask, because "
                            f"vector palettization with masking is not supported."
                        )
                        new_weight = weight.clone()
                else:
                    if config.cluster_dim == 1:
                        centroids, clusters = cls._cluster_weights_1d(
                            block_weight, block_importance, config
                        )
                    else:
                        centroids, clusters = cls._cluster_weights_2d(
                            block_weight, block_importance, config, rank
                        )
                if centroids is not None and clusters is not None:
                    # quantize LUT
                    if config.lut_dtype is not None:
                        centroids, scale, zp = cls._quantize_centroids(config.lut_dtype, centroids)
                        lut_quant_scale.append(scale)
                        if zp:
                            lut_quant_zp.append(zp)

                    depalett_block_weight = centroids[clusters].to(weight.dtype)

                    cls._update_clustered_block_weight(
                        block_weight,
                        block_mask,
                        depalett_block_weight,
                        new_weight,
                        config,
                        block_idx,
                    )

            # Combine quantization scales / zp for all LUTs into single tensor
            scale, zp = None, None
            if config.lut_dtype is not None and len(lut_quant_scale) > 0:
                scale = _torch.stack(lut_quant_scale, dim=config.axis)
                if len(lut_quant_zp) > 0:
                    zp = _torch.stack(lut_quant_zp, dim=config.axis)

            _logger.info(f"Finished processing {weight_name} in layer {layer_name} successfully")

            results_q.put((layer_name, weight_name, new_weight, scale, zp))

        _logger.info("Process done, work queue is empty")

    @classmethod
    def _quantize_centroids(cls, dtype: _torch.dtype, centroids: _torch.Tensor):
        centroids = centroids.numpy()
        ret = _compute_qparams(
            weight=centroids,
            nbits=get_n_bits_from_dtype(dtype),
            quantization_mode="LINEAR_SYMMETRIC",
            dtype=centroids.dtype,
            block_sizes=[0] * centroids.ndim,
            signed=is_signed_dtype(dtype),
        )

        if ret is None:
            _logger.warning(f"Unable to quantize centroids {centroids}")
            return

        quant_centroids, scale, zp = ret
        dequant_centroids = _quantize_op.decompress(
            quant_centroids,
            scale,
            zp,
        )

        # Convert back to torch tensors
        dequant_centroids = _torch.from_numpy(dequant_centroids)
        scale = _torch.from_numpy(scale)
        if zp is not None:
            zp = _torch.from_numpy(zp)

        return dequant_centroids, scale, zp

    @classmethod
    def _get_weights_to_cluster(
        cls,
        model: _torch.nn.Module,
        work_q: _Union[_mp.Queue, _queue.Queue],
        config: _Union[_Dict[str, _Dict[str, KMeansConfig]], KMeansConfig] = KMeansConfig(),
    ) -> _Tuple[_Dict[str, KMeansModule], _Dict[str, _Any]]:
        if not isinstance(config, dict):
            layers_to_cluster = get_atomic_layers(
                model,
                layer_types=list(KMeansSupportedModulesRegistry.get_supported_modules()),
                name_prefix="",
            )
            config_dict = {}
            for layer_name, layer in layers_to_cluster.items():
                layer_config = {}
                for param_name in KMeansSupportedModulesRegistry.get_kmeans_module(
                    layer
                ).parameter_names:
                    layer_config[param_name] = config
                config_dict[layer_name] = layer_config
        else:
            layers_to_cluster = {
                layer_name: model.get_submodule(layer_name) for layer_name, _ in config.items()
            }
            config_dict = config

        k_means_module_map = dict()

        param_dict = {}
        for layer_name, layer in layers_to_cluster.items():
            layer_config = config_dict[layer_name]

            k_means_module_cls = KMeansSupportedModulesRegistry.get_kmeans_module(layer)
            k_means_module: KMeansModule = k_means_module_cls(layer, layer_config)

            k_means_module_map[layer_name] = k_means_module

            for param_name, param in k_means_module.get_parameters().items():
                param_config = k_means_module.get_param_config(param_name, param)
                work_q.put((layer_name, param_name, param, param_config))
                param_dict[f"{layer_name}${param_name}"] = (param, param_config)

        return k_means_module_map, param_dict

    @classmethod
    def _prepare_worker_processes(
        cls, num_workers: int
    ) -> _Tuple[
        _Union[_mp.Queue, _queue.Queue],
        _Union[_mp.Queue, _queue.Queue],
        _Optional[_List[_mp.Process]],
    ]:
        raise NotImplementedError("This method is not implemented by base class.")

    @classmethod
    def _run_worker_processes(
        cls,
        work_q: _Union[_mp.Queue, _queue.Queue],
        results_q: _Union[_mp.Queue, _queue.Queue],
        worker_processes: _Optional[_List[_mp.Process]],
    ):
        raise NotImplementedError("This method is not implemented by base class.")

    @classmethod
    def _join_worker_processes(cls, worker_processes: _Optional[_List[_mp.Process]]):
        raise NotImplementedError("This method is not implemented by base class.")

    @classmethod
    @_torch.no_grad()
    def cluster_weights(
        cls,
        model: _torch.nn.Module,
        config: _Union[_Dict[str, _Dict[str, KMeansConfig]], KMeansConfig] = KMeansConfig(),
        num_workers: int = 1,
    ) -> _torch.nn.Module:
        work_q, results_q, worker_processes = cls._prepare_worker_processes(num_workers)
        k_means_module_map, param_dict = cls._get_weights_to_cluster(
            model=model,
            work_q=work_q,
            config=config,
        )

        num_params = len(param_dict)
        remaining_params = param_dict

        def _worker_loop() -> None:
            cls._run_worker_processes(work_q, results_q, worker_processes)
            num_params_left = len(remaining_params)
            num_errors = 0
            last_chance = False
            while remaining_params:
                try:
                    layer_name, param_name, new_value, scale, zp = results_q.get(timeout=10)
                except _queue.Empty:
                    if worker_processes is not None:
                        # This if path is for ParallelKMeans
                        # Check if workers are still running, in which case they may still be chewing on data and we
                        # need to wait. Also identify if any worker died (maybe it has been killed for OOM) and count
                        # it as an error
                        for proc in list(worker_processes):
                            if not proc.is_alive():
                                proc.join()
                                if proc.exitcode != 0:
                                    _logger.error(
                                        f"Process {proc} exited with exit code {proc.exitcode}"
                                    )
                                    num_errors += 1
                        alive_processes = sum(proc.is_alive() for proc in worker_processes)
                        if not alive_processes:
                            if last_chance:
                                _logger.info(
                                    f"All processes are done, but queue is empty, which is unexpected. Expecting to "
                                    f"receive {num_params_left} more param(s). Will end now."
                                )
                                break
                            else:
                                last_chance = True
                                continue
                        _logger.info(
                            f"Result queue is empty, but {alive_processes} process(es) is / are still alive, "
                            f"continuing..."
                        )
                        continue
                    else:
                        # This else path is for SequentialKMeans
                        if not last_chance:
                            last_chance = True
                            continue
                        else:
                            raise ValueError(
                                f"Queue is empty, which is unexpected. Expecting to receive {num_params_left} more "
                                f"param(s)."
                            )
                else:
                    _logger.info(f"Progress: {100 * (1.0 - (num_params_left / num_params)):.2f} %")
                    k_means_module = k_means_module_map[layer_name]
                    k_means_module._parameter_metadata[param_name]["lut_quantization_scale"] = scale
                    k_means_module._parameter_metadata[param_name]["lut_quantization_zp"] = zp
                    k_means_module.update_parameters(param_name, new_value)
                    remaining_params.pop(f"{layer_name}${param_name}")
                    # Even though it might not have succeeded
                    num_params_left -= 1

            _logger.info("joining worker processes")
            cls._join_worker_processes(worker_processes)

        _worker_loop()

        if remaining_params:
            _logger.error(
                f"The {len(remaining_params)} following params of following layers were not successfully palettized and"
                f" a new palettization will be attempted using a single worker: {', '.join(sorted(remaining_params))}"
            )
            work_q, results_q, worker_processes = cls._prepare_worker_processes(
                num_workers=1
            )  # Running the remaining params with 1 worker as that is more stable
            for current_param, param_tuple in remaining_params.items():
                layer_name, param_name = current_param.split("$")
                work_q.put((layer_name, param_name, param_tuple[0], param_tuple[1]))

            _worker_loop()

            if remaining_params:
                raise RuntimeError(
                    f"Even after rerunning all failed layers with a single worker, {len(remaining_params)} are "
                    f"still missing: {', '.join(sorted(remaining_params))}"
                )
            else:
                _logger.info(
                    "After rerunning all failed layers with a single worker, all palettizations succeeded"
                )

        _register_metadata_version(model)
        return model


class ParallelKMeans(KMeans):
    @classmethod
    def _prepare_worker_processes(
        cls,
        num_workers: int,
    ) -> _Tuple[
        _Union[_mp.Queue, _queue.Queue],
        _Union[_mp.Queue, _queue.Queue],
        _Optional[_List[_mp.Process]],
    ]:
        ctx = _mp.get_context("spawn")
        manager = ctx.Manager()
        work_q = manager.Queue()
        results_q = manager.Queue()

        worker_processes = [
            ctx.Process(
                target=cls._cluster_weights_worker,
                args=(rank, work_q, results_q),
                name=f"Process-{rank}",
                daemon=True,
            )
            for rank in range(num_workers)
        ]
        return work_q, results_q, worker_processes

    @classmethod
    def _run_worker_processes(
        cls,
        work_q: _Union[_mp.Queue, _queue.Queue],
        results_q: _Union[_mp.Queue, _queue.Queue],
        worker_processes: _Optional[_List[_mp.Process]],
    ):
        for worker_process in worker_processes:
            worker_process.start()
            _logger.info(f"Started {worker_process.name} for clustering weights.")

    @classmethod
    def _join_worker_processes(cls, worker_processes: _Optional[_List[_mp.Process]]):
        for worker_process in worker_processes:
            worker_process.join()
            _logger.info(f"Finished {worker_process.name}.")


class SequentialKMeans(KMeans):
    @classmethod
    def _prepare_worker_processes(
        cls, num_workers: int
    ) -> _Tuple[
        _Union[_mp.Queue, _queue.Queue],
        _Union[_mp.Queue, _queue.Queue],
        _Optional[_List[_mp.Process]],
    ]:
        work_q = _queue.Queue()
        results_q = _queue.Queue()
        return work_q, results_q, None

    @classmethod
    def _run_worker_processes(
        cls,
        work_q: _Union[_mp.Queue, _queue.Queue],
        results_q: _Union[_mp.Queue, _queue.Queue],
        worker_processes: _Optional[_List[_mp.Process]],
    ):
        cls._cluster_weights_worker(0, work_q, results_q)

    @classmethod
    def _join_worker_processes(cls, worker_processes: _Optional[_List[_mp.Process]]):
        return
