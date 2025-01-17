#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Dict as _Dict
from typing import Type as _Type

import torch as _torch
import torch.nn as _nn

from coremltools.optimize.torch._utils.metadata_utils import (
    CompressionMetadata as _CompressionMetadata,
)
from coremltools.optimize.torch._utils.python_utils import ClassRegistryMixin as _ClassRegistryMixin
from coremltools.optimize.torch.palettization._supported_modules import (
    Conv1d,
    Conv2d,
    Conv3d,
    Embedding,
    LayerNorm,
    Linear,
    MultiheadAttention,
)


class PalettizationConverterRegistry(_ClassRegistryMixin):
    """
    A registry of converts for converting a palettized module to a
    finalized module with de-palettized parameters.
    """


class PalettizationConverterBase(_nn.Module):
    """
    PalettizationCustomConverterBase is the base class for palettized model conversion. It implements the
    get_finalized_weights method which returns the palettized weights from ``LUT`` and ``indices``
    post-palettization.
    """

    _OBSERVED_MODULE: _Type

    def __init_subclass__(cls):
        PalettizationConverterRegistry.register(cls._OBSERVED_MODULE)(cls)

    @classmethod
    def validate_observed_module(cls, observed_module: _nn.Module):
        assert hasattr(
            observed_module, "qconfig"
        ), f"Module {type(observed_module)} has no attribute qconfig"
        assert hasattr(observed_module, "activation_post_process"), (
            f"Module {type(observed_module)} has no " f"attribute activation_post_process"
        )
        assert hasattr(observed_module, "weight_fake_quant"), (
            f"Module {type(observed_module)} has no attribute " f"weight_fake_quant"
        )

    @classmethod
    def get_finalized_weights(cls, observed_module: _nn.Module) -> _torch.Tensor:
        if observed_module.weight_fake_quant.partitions:
            return observed_module.weight_fake_quant.forward(observed_module.weight.detach())
        return observed_module.weight

    @classmethod
    def add_metadata(cls, observed_module: _nn.Module, return_module: _nn.Module):
        for dir_key in dir(observed_module):
            if "_fake_quant" in dir_key:
                if not isinstance(getattr(observed_module, dir_key).centroids[0], _torch.Tensor):
                    break
                param_name = dir_key.replace("_fake_quant", "")
                compression_metadata = _CompressionMetadata(param_name)
                compression_metadata.compression_type = ["palettization"]
                lut = _torch.stack(getattr(observed_module, dir_key).centroids, dim=0)
                for i in range(observed_module.weight.dim() + 2 - lut.dim()):
                    lut = lut.unsqueeze(-3)
                compression_metadata.lut = lut
                if getattr(observed_module, dir_key).cluster_dim > 1:
                    vector_ch_axis = getattr(observed_module, dir_key).vector_ch_axis
                    compression_metadata.vector_axis = vector_ch_axis
                if getattr(observed_module, dir_key).enable_per_channel_scale:
                    per_channel_scaling_factor = getattr(
                        observed_module, dir_key
                    ).per_channel_scaling_factor
                    for _ in range(observed_module.weight.dim() - per_channel_scaling_factor.dim()):
                        per_channel_scaling_factor = per_channel_scaling_factor.unsqueeze(-1)
                    compression_metadata.palettization_scale = per_channel_scaling_factor
                compression_metadata.register(return_module)

    @classmethod
    def from_observed(cls, observed_module: _nn.Module) -> _nn.Module:
        """
        The classes that base-class this class will have to implement the ``from_observed`` method to tell the
        convert method what type of a module to return through Pytorch's conversion.
        """
        raise NotImplementedError()


class LinearPalettizationConverter(PalettizationConverterBase):
    """
    Conversion class for Linear.
    """

    _OBSERVED_MODULE: _Type = Linear

    @classmethod
    def from_observed(cls, observed_module: _nn.Module) -> _nn.Module:
        cls.validate_observed_module(observed_module)
        finalized_weights = cls.get_finalized_weights(observed_module)
        return_module = _nn.Linear(
            in_features=observed_module.in_features,
            out_features=observed_module.out_features,
            bias=observed_module.bias is not None,
            device=observed_module.device if hasattr(observed_module, "device") else None,
            dtype=observed_module.dtype if hasattr(observed_module, "dtype") else None,
        )
        return_module.weight = _nn.Parameter(finalized_weights)
        cls.add_metadata(observed_module, return_module)
        if observed_module.bias is not None:
            return_module.bias = _nn.Parameter(observed_module.bias.detach())
        return_module.activation_post_process = observed_module.activation_post_process
        return return_module


class Conv1dPalettizationConverter(PalettizationConverterBase):
    """
    Conversion class for Conv2d.
    """

    _OBSERVED_MODULE: _Type = Conv1d

    @classmethod
    def from_observed(cls, observed_module: _nn.Module) -> _nn.Module:
        cls.validate_observed_module(observed_module)
        finalized_weights = cls.get_finalized_weights(observed_module)
        return_module = _nn.Conv1d(
            in_channels=observed_module.in_channels,
            out_channels=observed_module.out_channels,
            kernel_size=observed_module.kernel_size,
            stride=observed_module.stride,
            padding=observed_module.padding,
            dilation=observed_module.dilation,
            groups=observed_module.groups,
            bias=observed_module.bias is not None,
            padding_mode=observed_module.padding_mode,
            device=observed_module.device if hasattr(observed_module, "device") else None,
            dtype=observed_module.dtype if hasattr(observed_module, "dtype") else None,
        )
        return_module.weight = _nn.Parameter(finalized_weights)
        cls.add_metadata(observed_module, return_module)
        if observed_module.bias is not None:
            return_module.bias = _nn.Parameter(observed_module.bias.detach())
        return_module.activation_post_process = observed_module.activation_post_process
        return return_module


class Conv2dPalettizationConverter(PalettizationConverterBase):
    """
    Conversion class for Conv2d.
    """

    _OBSERVED_MODULE: _Type = Conv2d

    @classmethod
    def from_observed(cls, observed_module: _nn.Module) -> _nn.Module:
        cls.validate_observed_module(observed_module)
        finalized_weights = cls.get_finalized_weights(observed_module)
        return_module = _nn.Conv2d(
            in_channels=observed_module.in_channels,
            out_channels=observed_module.out_channels,
            kernel_size=observed_module.kernel_size,
            stride=observed_module.stride,
            padding=observed_module.padding,
            dilation=observed_module.dilation,
            groups=observed_module.groups,
            bias=observed_module.bias is not None,
            padding_mode=observed_module.padding_mode,
            device=observed_module.device if hasattr(observed_module, "device") else None,
            dtype=observed_module.dtype if hasattr(observed_module, "dtype") else None,
        )
        return_module.weight = _nn.Parameter(finalized_weights)
        cls.add_metadata(observed_module, return_module)
        if observed_module.bias is not None:
            return_module.bias = _nn.Parameter(observed_module.bias.detach())
        return_module.activation_post_process = observed_module.activation_post_process
        return return_module


class Conv3dPalettizationConverter(PalettizationConverterBase):
    """
    Conversion class for Conv3d.
    """

    _OBSERVED_MODULE: _Type = Conv3d

    @classmethod
    def from_observed(cls, observed_module: _nn.Module) -> _nn.Module:
        cls.validate_observed_module(observed_module)
        finalized_weights = cls.get_finalized_weights(observed_module)
        return_module = _nn.Conv3d(
            in_channels=observed_module.in_channels,
            out_channels=observed_module.out_channels,
            kernel_size=observed_module.kernel_size,
            stride=observed_module.stride,
            padding=observed_module.padding,
            dilation=observed_module.dilation,
            groups=observed_module.groups,
            bias=observed_module.bias is not None,
            padding_mode=observed_module.padding_mode,
            device=observed_module.device if hasattr(observed_module, "device") else None,
            dtype=observed_module.dtype if hasattr(observed_module, "dtype") else None,
        )
        return_module.weight = _nn.Parameter(finalized_weights)
        cls.add_metadata(observed_module, return_module)
        if observed_module.bias is not None:
            return_module.bias = _nn.Parameter(observed_module.bias.detach())
        return_module.activation_post_process = observed_module.activation_post_process
        return return_module


class LayerNormPalettizationConverter(PalettizationConverterBase):
    """
    Conversion class for LayerNorm.
    """

    _OBSERVED_MODULE: _Type = LayerNorm

    @classmethod
    def from_observed(cls, observed_module: _nn.Module):
        cls.validate_observed_module(observed_module)
        finalized_weights = cls.get_finalized_weights(observed_module)
        return_module = _nn.LayerNorm(
            normalized_shape=observed_module.normalized_shape,
            eps=observed_module.eps,
            elementwise_affine=observed_module.elementwise_affine,
            device=observed_module.device if hasattr(observed_module, "device") else None,
            dtype=observed_module.dtype if hasattr(observed_module, "dtype") else None,
        )
        if observed_module.elementwise_affine:
            return_module.weight = _nn.Parameter(finalized_weights)
            if observed_module.bias:
                return_module.bias = _nn.Parameter(observed_module.bias.detach())
        cls.add_metadata(observed_module, return_module)
        return_module.activation_post_process = observed_module.activation_post_process
        return return_module


class MultiheadAttentionPalettizationConverter(PalettizationConverterBase):
    """
    Conversion class for MultiheadAttention.
    """

    _OBSERVED_MODULE: _Type = MultiheadAttention

    @classmethod
    def validate_observed_module(cls, observed_module: _nn.Module):
        assert hasattr(
            observed_module, "qconfig"
        ), f"Module {type(observed_module)} has no attribute qconfig"
        assert hasattr(observed_module, "activation_post_process"), (
            f"Module {type(observed_module)} has no " f"attribute activation_post_process"
        )

        assert hasattr(observed_module.out_proj, "weight_fake_quant"), (
            f"Module {type(observed_module.out_proj)} has no attribute " f"q_proj_weight_fake_quant"
        )
        if not observed_module._qkv_same_embed_dim:
            assert hasattr(observed_module, "q_proj_weight_fake_quant"), (
                f"Module {type(observed_module)} has no attribute " f"q_proj_weight_fake_quant"
            )
            assert hasattr(observed_module, "k_proj_weight_fake_quant"), (
                f"Module {type(observed_module)} has no attribute " f"k_proj_weight_fake_quant"
            )
            assert hasattr(observed_module, "v_proj_weight_fake_quant"), (
                f"Module {type(observed_module)} has no attribute " f"v_proj_weight_fake_quant"
            )
        else:
            assert hasattr(observed_module, "in_proj_weight_fake_quant"), (
                f"Module {type(observed_module)} has no attribute " f"in_proj_weight_fake_quant"
            )

    @classmethod
    def from_observed(cls, observed_module: _nn.Module) -> _nn.Module:
        cls.validate_observed_module(observed_module)
        add_bias_kv = observed_module.bias_k is not None and observed_module.bias_v is not None
        bias = (
            observed_module.out_proj.bias is not None and observed_module.in_proj_bias is not None
        )
        return_module = _nn.MultiheadAttention(
            embed_dim=observed_module.embed_dim,
            num_heads=observed_module.num_heads,
            dropout=observed_module.dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=observed_module.add_zero_attn,
            kdim=observed_module.kdim,
            vdim=observed_module.vdim,
            batch_first=observed_module.batch_first,
            device=observed_module.device if hasattr(observed_module, "device") else None,
            dtype=observed_module.dtype if hasattr(observed_module, "dtype") else None,
        )
        if not observed_module._qkv_same_embed_dim:
            return_module.q_proj_weight = _nn.Parameter(
                observed_module.q_proj_weight_fake_quant.forward(
                    observed_module.q_proj_weight.detach()
                )
            )
            return_module.k_proj_weight = _nn.Parameter(
                observed_module.k_proj_weight_fake_quant.forward(
                    observed_module.k_proj_weight.detach()
                )
            )
            return_module.v_proj_weight = _nn.Parameter(
                observed_module.v_proj_weight_fake_quant.forward(
                    observed_module.v_proj_weight.detach()
                )
            )
        else:
            return_module.in_proj_weight = _nn.Parameter(
                observed_module.in_proj_weight_fake_quant.forward(
                    observed_module.in_proj_weight.detach()
                )
            )
        return_module.out_proj.weight = _nn.Parameter(
            observed_module.out_proj.weight_fake_quant.forward(
                observed_module.out_proj.weight.detach()
            )
        )
        if bias:
            return_module.out_proj.bias = _nn.Parameter(observed_module.out_proj.bias.detach())
            return_module.in_proj_bias = _nn.Parameter(observed_module.in_proj_bias.detach())
        if add_bias_kv:
            return_module.bias_k = _nn.Parameter(observed_module.bias_k.detach())
            return_module.bias_v = _nn.Parameter(observed_module.bias_v.detach())
        cls.add_metadata(observed_module, return_module)
        return_module.activation_post_process = observed_module.activation_post_process
        return return_module


class EmbeddingPalettizationConverter(PalettizationConverterBase):
    """
    Conversion class for Embedding.
    """

    _OBSERVED_MODULE: _Type = Embedding

    @classmethod
    def from_observed(cls, observed_module: _nn.Module) -> _nn.Module:
        cls.validate_observed_module(observed_module)
        finalized_weights = cls.get_finalized_weights(observed_module)
        return_module = _nn.Embedding(
            num_embeddings=observed_module.num_embeddings,
            embedding_dim=observed_module.embedding_dim,
            padding_idx=observed_module.padding_idx,
            max_norm=observed_module.max_norm,
            norm_type=observed_module.norm_type,
            scale_grad_by_freq=observed_module.scale_grad_by_freq,
            sparse=observed_module.sparse,
            _weight=None,
            device=observed_module.device if hasattr(observed_module, "device") else None,
            dtype=observed_module.dtype if hasattr(observed_module, "dtype") else None,
        )
        return_module.weight = _nn.Parameter(finalized_weights)
        cls.add_metadata(observed_module, return_module)
        return_module.activation_post_process = observed_module.activation_post_process
        return return_module


def get_conversion_custom_config_dict() -> _Dict[str, _Dict[_Type, _Type]]:
    """
    Returns a dictionary to mapping palettized modules to custom conversion class.
    A custom conversion class derives from :py:class:`PalettizationCustomConverterBase` and
    implements a ``from_observed`` method which creates de-palettized from palettized modules.
    """
    return {"observed_to_quantized_custom_module_class": PalettizationConverterRegistry.REGISTRY}
