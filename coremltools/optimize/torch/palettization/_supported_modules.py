#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F

from .palettization_config import SUPPORTED_PYTORCH_QAT_MODULES


def _get_palettization_qat_mappings():
    """
    _get_palettization_qat_mappings creates qat_module_mappings supported by coremltools.optimize.torch for palettization. We
    support three modules already in DEFAULT_QAT_MODULE_MAPPINGS, namely, nn.Linear, nn.Conv2d and
    nn.Conv3d. Additionally, we have added support for preparation of nn.Conv1d, nn.LayerNorm,
    nn.MultiheadAttention and nn.Embedding modules.
    """
    qat_module_mappings = (
        _torch.quantization.quantization_mappings.get_default_qat_module_mappings()
    )
    for k in list(qat_module_mappings.keys()):
        if k not in SUPPORTED_PYTORCH_QAT_MODULES:
            del qat_module_mappings[k]
    qat_module_mappings[Conv1d._FLOAT_MODULE] = Conv1d
    qat_module_mappings[LayerNorm._FLOAT_MODULE] = LayerNorm
    qat_module_mappings[MultiheadAttention._FLOAT_MODULE] = MultiheadAttention
    qat_module_mappings[Embedding._FLOAT_MODULE] = Embedding

    return qat_module_mappings


class Conv1d(_nn.Conv1d):
    _FLOAT_MODULE = _nn.Conv1d

    def forward(self, input):
        qweight = self.weight_fake_quant(self.weight)
        if self.padding_mode != "zeros":
            return _F.conv1d(
                _F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                qweight,
                self.bias,
                self.stride,
                (0,),
                self.dilation,
                self.groups,
            )
        return _F.conv1d(
            input, qweight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module, either produced by torch.quantization utilities
        or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat." + cls.__name__ + ".from_float only works for " + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"

        qconfig = mod.qconfig
        qat = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
        )
        qat.qconfig = qconfig
        qat.weight_fake_quant = qconfig.weight()

        wnorm = None

        for k, hook in mod._forward_pre_hooks.items():
            if "WeightNorm" in str(hook):
                wnorm = hook

        if wnorm:
            qat = _nn.utils.weight_norm(qat, name=wnorm.name, dim=wnorm.dim)

        for name, param in mod.named_parameters(recurse=False):
            setattr(qat, name, param)

        if wnorm:
            _nn.utils.remove_weight_norm(mod)

        return qat


class LayerNorm(_nn.LayerNorm):
    _FLOAT_MODULE = _nn.LayerNorm

    def forward(self, input):
        return _F.layer_norm(
            input,
            self.normalized_shape,
            self.weight_fake_quant(self.weight) if self.elementwise_affine else self.weight,
            self.bias,
            self.eps,
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module, either produced by torch.quantization utilities
        or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat." + cls.__name__ + ".from_float only works for " + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"

        assert isinstance(
            mod.weight, _nn.Parameter
        ), "CANNOT be prepared for palettization: weight is NOT learnable"

        qconfig = mod.qconfig
        qat = cls(mod.normalized_shape, eps=mod.eps, elementwise_affine=mod.elementwise_affine)
        qat.qconfig = qconfig

        if qat.elementwise_affine:
            qat.weight_fake_quant = qconfig.weight()

        for name, param in mod.named_parameters(recurse=False):
            setattr(qat, name, param)

        assert qat.elementwise_affine == (qat.weight is not None)
        return qat


class Embedding(_nn.Embedding):
    _FLOAT_MODULE = _nn.Embedding

    def forward(self, input):
        qweight = self.weight_fake_quant(self.weight)
        return _F.embedding(
            input,
            qweight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module, either produced by torch.quantization utilities
        or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat." + cls.__name__ + ".from_float only works for " + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"

        assert isinstance(
            mod.weight, _nn.Parameter
        ), "CANNOT be prepared for palettization: weight is NOT learnable"

        qconfig = mod.qconfig
        qat = cls(
            mod.num_embeddings,
            mod.embedding_dim,
            mod.padding_idx,
            max_norm=mod.max_norm,
            norm_type=mod.norm_type,
            scale_grad_by_freq=mod.scale_grad_by_freq,
            sparse=mod.sparse,
            _weight=None,
        )
        qat.qconfig = qconfig
        qat.weight_fake_quant = qconfig.weight()

        for name, param in mod.named_parameters(recurse=False):
            setattr(qat, name, param)

        return qat


class MultiheadAttention(_nn.MultiheadAttention):
    _FLOAT_MODULE = _nn.MultiheadAttention

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            # Ensure that that the "is" property is maintained
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = _F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight_fake_quant(self.out_proj.weight),
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight_fake_quant(self.q_proj_weight),
                k_proj_weight=self.k_proj_weight_fake_quant(self.k_proj_weight),
                v_proj_weight=self.v_proj_weight_fake_quant(self.v_proj_weight),
            )
        else:
            attn_output, attn_output_weights = _F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight_fake_quant(self.in_proj_weight),
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight_fake_quant(self.out_proj.weight),
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    @classmethod
    def from_float(cls, mod):
        r"""Create a palettization module from a float module or qparams_dict

        Args: `mod` a float module, either produced by torch.quantization utilities
        or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat." + cls.__name__ + ".from_float only works for " + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"

        qconfig = mod.qconfig
        qat = cls(
            mod.embed_dim,
            mod.num_heads,
            mod.dropout,
            batch_first=mod.batch_first,
            bias=hasattr(mod, "in_proj_bias"),
            add_bias_kv=mod.bias_k is not None,
            add_zero_attn=mod.add_zero_attn,
            kdim=mod.kdim,
            vdim=mod.vdim,
        )
        qat.qconfig = qconfig
        if not qat._qkv_same_embed_dim:
            qat.q_proj_weight_fake_quant = qconfig.weight()
            qat.k_proj_weight_fake_quant = qconfig.weight()
            qat.v_proj_weight_fake_quant = qconfig.weight()
        else:
            qat.in_proj_weight_fake_quant = qconfig.weight()

        qat.out_proj.weight_fake_quant = qconfig.weight()

        for name, param in mod.named_parameters(recurse=False):
            setattr(qat, name, param)

        for name, param in mod.out_proj.named_parameters(recurse=False):
            setattr(qat.out_proj, name, param)

        return qat


def get_palettizable_parameters(module):
    """
    Return a list of parameters of the module which can be palettized
    """
    if isinstance(module, _nn.MultiheadAttention):
        if not module._qkv_same_embed_dim:
            return [
                (module.out_proj.weight, "out_proj.weight"),
                (module.q_proj_weight, "q_proj_weight"),
                (module.k_proj_weight, "k_proj_weight"),
                (module.v_proj_weight, "v_proj_weight"),
            ]
        else:
            return [
                (module.in_proj_weight, "in_proj_weight"),
                (module.out_proj.weight, "out_proj.weight"),
            ]
    return [(module.weight, "weight")]
