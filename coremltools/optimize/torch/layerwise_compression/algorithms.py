#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# Original implementation from https://github.com/IST-DASLab/sparsegpt
# Copyright 2023 IST Austria Distributed Algorithms and Systems Lab. All Rights Reserved.

import copy as _copy
import logging as _logging
import math as _math
import time as _time
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
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
    CompressionMetadata as _CompressionMetadata,
)
from coremltools.optimize.torch._utils.python_utils import ClassRegistryMixin as _ClassRegistryMixin
from coremltools.optimize.torch._utils.torch_utils import (
    get_n_bits_from_dtype as _get_n_bits_from_dtype,
)
from coremltools.optimize.torch._utils.torch_utils import (
    maybe_convert_str_to_dtype as _maybe_convert_str_to_dtype,
)
from coremltools.optimize.torch.layerwise_compression._quant import Quantizer as _Quantizer
from coremltools.optimize.torch.layerwise_compression._quant import _normal_float_palette
from coremltools.optimize.torch.layerwise_compression._quant import quantize as _quantize
from coremltools.optimize.torch.optimization_config import (
    ModuleOptimizationConfig as _ModuleOptimizationConfig,
)
from coremltools.optimize.torch.optimization_config import QuantizationGranularity
from coremltools.optimize.torch.quantization.quantization_config import (
    QuantizationScheme as _QuantizationScheme,
)

_logger = _logging.getLogger(__name__)


class LayerwiseCompressionAlgorithmConfig(_ABC, _ClassRegistryMixin, _ModuleOptimizationConfig):
    """
    A template class and registry for configuration classes to be used
    with :py:class:`LayerwiseCompressionAlgorithm`.
    """

    pass

@LayerwiseCompressionAlgorithmConfig.register("gptq")
@_define
class ModuleGPTQConfig(LayerwiseCompressionAlgorithmConfig):
    """
    Configuration class for specifying global and module-level compression options for the
    `Generative Pre-Trained Transformer Quantization (GPTQ) <https://arxiv.org/pdf/2210.17323.pdf>`_ algorithm.

    Args:
        weight_dtype (:py:class:`torch.dtype`): The dtype to use for quantizing the weights. The number of bits used
            for quantization is inferred from the dtype. When dtype is set to :py:class:`torch.float32`, the weights
            corresponding to that layer are not quantized. Defaults to :py:class:`torch.uint8`, which corresponds to
            8-bit quantization.
        granularity (:py:class:`QuantizationGranularity`): Specifies the granularity at which quantization parameters
            will be computed. Can be one of ``per_channel``, ``per_tensor`` or ``per_block``. When using ``per_block``,
            ``block_size`` argument must be specified. Defaults to ``per_channel``.
        quantization_scheme (:py:class:`~.coremltools.optimize.torch.quantization.quantization_config.QuantizationScheme`): Type of
            quantization configuration to use. When this parameter is set to ``QuantizationScheme.symmetric``, all
            weights are quantized with zero point as zero. When it is set to ``QuantizationScheme.affine``, zero point
            can be set anywhere in the range of values allowed for the quantized weight.
            Defaults to ``QuantizationScheme.symmetric``.
        block_size (:obj:`int`): When ``block_size`` is specified, ``block_size`` number of values will share the same quantization
            parameters of scale, as well as the same zero point when applicable, across the input channel axis. Defaults to ``None``.
        enable_normal_float (:obj:`bool`): When ``True``, normal float format is used for quantization. It's
            only supported when ``weight_dtype`` is equal to ``int3`` and ``int4``. Defaults to ``False``.
        hessian_dampening (:obj:`float`): Dampening factor added to the diagonal of the
            Hessian used by GPTQ algorithm. Defaults to ``0.01``.
        use_activation_order_heuristic (:obj:`bool`): When ``True``, columns of weight are sorted
            in descending order of values of Hessian diagonal elements. Defaults to ``True``.
        processing_group_size (:obj:`int`): The weights are updated in
            blocks of size ``processing_group_size``. Defaults to ``128``.

    .. note:
        Blocking is currently limited to the input channel axis for GPTQ. For a linear layer of shape `(C_o x C_i)`, and ``block_size`` `B`,
        the quantization scales will have shape `(C_o x C_i/B)`. For a 2D conv layer of shape `(C_o x C_i x H x W)`, the
        quantization scales will have shape `(C_o x C_i/B x 1 x 1)`.
    """

    weight_dtype: _Union[str, _torch.dtype] = _field(
        default="uint8",
    )
    granularity: QuantizationGranularity = _field(
        default="per_channel",
        converter=QuantizationGranularity,
        validator=_validators.in_(QuantizationGranularity),
    )
    quantization_scheme: _QuantizationScheme = _field(
        default="symmetric",
        converter=_QuantizationScheme,
        validator=_validators.in_(_QuantizationScheme),
    )
    block_size: _Optional[int] = _field(
        default=None, validator=_validators.optional(_validators.instance_of(int))
    )
    enable_normal_float: bool = _field(default=False, validator=_validators.instance_of(bool))
    hessian_dampening: float = _field(default=0.01, validator=_validators.instance_of(float))
    use_activation_order_heuristic: bool = _field(
        default=False, validator=_validators.instance_of(bool)
    )
    processing_group_size: int = _field(default=128, validator=_validators.instance_of(int))
    algorithm: str = _field(default="gptq", validator=_validators.in_("gptq"))

    def __attrs_post_init__(self):
        self.weight_n_bits = _get_n_bits_from_dtype(self.weight_dtype)
        self.weight_dtype = _maybe_convert_str_to_dtype(self.weight_dtype)
        if self.weight_dtype not in [_torch.uint8, _torch.float32]:
            raise ValueError(
                f"weight_dtype must be one of (torch.uint8, torch.float32) not {self.weight_dtype}"
            )

    @classmethod
    def from_dict(cls, config_dict):
        converter = _cattrs.Converter(forbid_extra_keys=True)
        converter.register_structure_hook(
            _Union[str, _torch.dtype],
            lambda obj, type: obj,
        )
        return converter.structure_attrs_fromdict(config_dict, cls)


@LayerwiseCompressionAlgorithmConfig.register("sparse_gpt")
@_define
class ModuleSparseGPTConfig(LayerwiseCompressionAlgorithmConfig):
    """
    Configuration class for specifying global and module-level compression options for the
    `Sparse Generative Pre-Trained Transformer (SparseGPT) <https://arxiv.org/pdf/2301.00774.pdf>`_ algorithm.

    Args:
        target_sparsity (:obj:`float`): Fraction of weight elements to set to ``0``. Defaults to
            ``0.5``.
        n_m_ratio (:obj:`tuple` of :obj:`int`): A tuple of two integers which specify how ``n:m`` pruning should be
            applied. In ``n:m`` pruning, out of every ``m`` elements, ``n`` with lowest magnitude are set to
            zero. When ``n_m_ratio`` is not ``None``, the value of ``target_sparsity`` is ignored and the actual
            target sparsity is determined by the ``n:m`` ratio.
        weight_dtype (:py:class:`torch.dtype`): The dtype to use for quantizing the weights. The number of bits used
            for quantization is inferred from the dtype. When dtype is set to :py:class:`torch.float32`, the weights
            corresponding to that layer are not quantized. Defaults to :py:class:`torch.float32`, which corresponds to
            no quantization.
        quantization_granularity (:py:class:`QuantizationGranularity`): Specifies the granularity at which quantization parameters
            will be computed. Can be one of ``per_channel``, ``per_tensor`` or ``per_block``. When using ``per_block``,
            ``block_size`` argument must be specified. Defaults to ``per_channel``.
        quantization_scheme (:py:class:`~.coremltools.optimize.torch.quantization.quantization_config.QuantizationScheme`): Type of
            quantization configuration to use. When this parameter is set to ``QuantizationScheme.symmetric``, all
            weights are quantized with zero point as zero. When it is set to ``QuantizationScheme.affine``, zero point
            can be set anywhere in the range of values allowed for the quantized weight.
            Defaults to ``QuantizationScheme.symmetric``.
        enable_normal_float (:obj:`bool`): When ``True``, normal float format is used for quantization. It's
            only supported for ``weight_dtype`` is equal to ``int3`` and ``int4``.
        hessian_dampening (:obj:`float`): Dampening factor added to the diagonal of the
            Hessian used by GPTQ algorithm. Defaults to ``0.01``.
        processing_group_size (:obj:`int`): The weights are updated in
            blocks of size processing_group_size. Defaults to ``128``.
    """

    target_sparsity: float = _field(default=0.5, validator=_validators.instance_of(float))
    n_m_ratio: _Optional[_Tuple[int, int]] = _field(
        default=None,
        validator=_validators.optional(
            _validators.deep_iterable(
                member_validator=_validators.instance_of(int),
                iterable_validator=_validators.instance_of((tuple, list)),
            )
        ),
    )
    weight_dtype: _Union[str, _torch.dtype] = _field(
        default="uint8",
    )
    quantization_granularity: QuantizationGranularity = _field(
        default="per_channel",
        converter=QuantizationGranularity,
        validator=_validators.in_(QuantizationGranularity),
    )
    quantization_scheme: _QuantizationScheme = _field(
        default="symmetric",
        converter=_QuantizationScheme,
        validator=_validators.in_(_QuantizationScheme),
    )

    enable_normal_float: bool = _field(default=False, validator=_validators.instance_of(bool))
    hessian_dampening: float = _field(default=0.01, validator=_validators.instance_of(float))
    processing_group_size: int = _field(default=128, validator=_validators.instance_of(int))
    algorithm: str = _field(default="sparse_gpt", validator=_validators.in_("sparse_gpt"))

    def __attrs_post_init__(self):
        self.weight_n_bits = _get_n_bits_from_dtype(self.weight_dtype)
        self.weight_dtype = _maybe_convert_str_to_dtype(self.weight_dtype)
        if self.weight_dtype not in [_torch.uint8, _torch.float16, _torch.float32]:
            raise ValueError(
                f"weight_dtype must be one of (torch.uint8, torch.float16, torch.float32) not {self.weight_dtype}"
            )

    @classmethod
    def from_dict(cls, config_dict):
        converter = _cattrs.Converter(forbid_extra_keys=True)
        converter.register_structure_hook(
            _Union[str, _torch.dtype],
            lambda obj, type: obj,
        )
        return converter.structure_attrs_fromdict(config_dict, cls)


class LayerwiseCompressionAlgorithm(_ClassRegistryMixin):
    """
    A template class for implementing layerwise compression algorithms
    to be used with :py:class:`LayerwiseCompressor`.
    """

    @_abstractmethod
    def add_batch(self, inp: _torch.Tensor, out: _torch.Tensor) -> None:
        """
        Perform computation on a batch of data to acquire statistics before
        compression.
        """
        raise NotImplementedError("Method not implemented in base class.")

    @_abstractmethod
    def cleanup(self) -> None:
        """
        Reset the state of the compression algorithm object and free GPU memory.
        """
        raise NotImplementedError("Method not implemented in base class.")

    @_abstractmethod
    def compress(self) -> None:
        """
        Compress the weights of the layer.
        """
        raise NotImplementedError("Method not implemented in base class.")


class OBSCompressionAlgorithm(LayerwiseCompressionAlgorithm):
    """
    A compression algorithm which uses the Hessian of the reconstruction loss
    to compress a weight matrix of a given layer. Based on the
    optimal brain surgeon paradigm described in `Optimal Brain Compression:
    A Framework for Accurate Post-Training Quantization and Pruning
    <https://arxiv.org/pdf/2208.11580.pdf>`_.
    """

    def __init__(self, layer: _nn.Module, config: LayerwiseCompressionAlgorithmConfig):
        self._layer = layer
        self._device = self._layer.weight.device
        self._nsamples = 0
        self._config = config
        weight = self._layer.weight.data
        if isinstance(self._layer, _nn.Conv2d):
            weight = weight.flatten(1)
        self._dim = weight.dim()
        self._rows = weight.shape[0]
        self._columns = weight.shape[1]
        self._hessian = _torch.zeros((self._columns, self._columns), device=self._device)

    @_abstractmethod
    def _init_parameters(self, config: LayerwiseCompressionAlgorithmConfig):
        """
        Initialize parameters of the algorithm from config.
        """
        raise NotImplementedError("Method not implemented in base class.")

    def add_batch(self, inp: _torch.Tensor, out: _torch.Tensor):
        self._compute_hessian(inp, out)

    def _compute_hessian(self, inp: _torch.Tensor, out: _torch.Tensor):
        """
        Compute Hessian of the L2 loss between the original output
        of the layer and the output computed using compressed weights.
        """
        self._inp1 = inp
        self._out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self._layer, _nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self._layer, _nn.Conv2d):
            unfold = _nn.Unfold(
                self._layer.kernel_size,
                dilation=self._layer.dilation,
                padding=self._layer.padding,
                stride=self._layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self._hessian *= self._nsamples / (self._nsamples + tmp)
        self._nsamples += tmp
        inp = _math.sqrt(2 / self._nsamples) * inp.float()
        self._hessian += inp.matmul(inp.t())

    @_abstractmethod
    def _compress_impl(self):
        """
        Implementation of the compression algorithm
        """
        raise NotImplementedError("Method not implemented in base class.")

    def compress(self):
        self._compress_impl()
        # NOTE: Currently algorithm assumes weight parameter is available for all layers
        # and the only parameter that gets updated
        metadata = self._get_compression_metadata("weight", self._layer.weight)
        metadata.register(self._layer)

    def cleanup(self):
        self._inp1 = None
        self._out1 = None
        self._nsamples = 0
        _torch.cuda.empty_cache()
        self._hessian = None

    @_abstractmethod
    def _get_compression_metadata(self, param_name, param):
        raise NotImplementedError("Method not implemented in base class.")

    def _store_quantization_params(self, quantizer: _Quantizer):
        if quantizer is not None:
            scale = quantizer.scale
            scale_store = _torch.empty_like(scale, device=_torch.device("cpu")).copy_(scale)
            self._scale.append(scale_store)
            if not self._enable_normal_float:
                zero_point = quantizer.zero_point
                zero_point_store = _torch.empty_like(zero_point, device=_torch.device("cpu")).copy_(
                    zero_point
                )
                self._zero_point.append(zero_point_store)


@LayerwiseCompressionAlgorithm.register("gptq")
class GPTQ(OBSCompressionAlgorithm):
    """
    A post-training compression algorithm based on the paper
    `GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
    <https://arxiv.org/pdf/2210.17323.pdf>`_.

    Args:
        layer (:obj:`torch.nn.Module`): Module to be compressed.
        config (:py:class:`ModuleGPTQConfig`): Config specifying hyperparameters
            for the GPTQ algorithm.
    """

    def __init__(self, layer: _nn.Module, config: ModuleGPTQConfig):
        super().__init__(layer, config)
        self._init_parameters(config)

    def _init_parameters(self, config: ModuleGPTQConfig):
        # Defaults to blocking along input channel axis
        self._block_size = config.block_size
        if self._block_size is not None and self._columns % self._block_size != 0:
            raise ValueError(
                f"Block size must completely divide the axis along which blocking is done: {self._columns} % {self._block_size} != 0"
            )
        self._weight_n_bits = config.weight_n_bits
        self._processing_group_size = config.processing_group_size
        self._enable_normal_float = config.enable_normal_float
        self._hessian_dampening = config.hessian_dampening
        self._use_activation_order_heuristic = config.use_activation_order_heuristic
        # static grouping leads to all quantization parameters being pre-computed,
        # rather than dynamically during algorithm execution. This is necessary when
        # activation_order_heuristic is turned on to make sure the model is still exportable
        self._enable_static_blocking = config.use_activation_order_heuristic
        self._quantizer = None
        if self._weight_n_bits < 16:
            per_channel = config.granularity in [
                QuantizationGranularity.per_channel,
                QuantizationGranularity.per_block,
            ]
            self._quantizer = _Quantizer(
                n_bits=self._weight_n_bits,
                per_channel=per_channel,
                symmetric=config.quantization_scheme == _QuantizationScheme.symmetric,
                enable_normal_float=config.enable_normal_float,
            )
            self._scale = []
            self._zero_point = []

    def _compress_impl(self):
        weight = self._layer.weight.data.clone()
        if isinstance(self._layer, _nn.Conv2d):
            if self._block_size is not None:
                self._block_size = self._block_size * weight.shape[2] * weight.shape[3]
            weight = weight.flatten(1)

        weight = weight.float()

        tick = _time.time()

        if not self._quantizer.ready():
            self._quantizer.find_params(weight, weight=True)
            if self._block_size is None:
                self._store_quantization_params(self._quantizer)

        hessian = self._hessian
        del self._hessian
        dead = _torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        blocks = []
        if self._enable_static_blocking and self._block_size is not None:
            for i in range(0, self._columns, self._block_size):
                quantizer = _copy.deepcopy(self._quantizer)
                quantizer.find_params(weight[:, i : (i + self._block_size)], weight=True)
                blocks.append(quantizer)
                self._store_quantization_params(quantizer)

        perm = None
        if self._use_activation_order_heuristic:
            perm = _torch.argsort(_torch.diag(hessian), descending=True)
            weight = weight[:, perm]
            hessian = hessian[perm][:, perm]

        losses = _torch.zeros_like(weight)
        quant_weight = _torch.zeros_like(weight)

        damp = self._hessian_dampening * _torch.mean(_torch.diag(hessian))
        diag = _torch.arange(self._columns, device=self._device)
        hessian[diag, diag] += damp
        hessian = _torch.linalg.cholesky(hessian)
        hessian = _torch.cholesky_inverse(hessian)
        hessian = _torch.linalg.cholesky(hessian, upper=True)
        hessian_inverse = hessian

        for i1 in range(0, self._columns, self._processing_group_size):
            i2 = min(i1 + self._processing_group_size, self._columns)
            count = i2 - i1

            weight_block = weight[:, i1:i2].clone()
            quant_weight_block = _torch.zeros_like(weight_block)
            error_block = _torch.zeros_like(weight_block)
            losses_block = _torch.zeros_like(weight_block)
            hessian_inverse_block = hessian_inverse[i1:i2, i1:i2]

            for i in range(count):
                w = weight_block[:, i]
                d = hessian_inverse_block[i, i]

                if self._block_size is not None:
                    if self._enable_static_blocking:
                        idx = perm[i1 + i]
                        self._quantizer = blocks[idx // self._block_size]
                    else:
                        if (i1 + i) % self._block_size == 0:
                            self._quantizer.find_params(
                                weight[:, (i1 + i) : (i1 + i + self._block_size)],
                                weight=True,
                            )
                            self._store_quantization_params(self._quantizer)

                q = _quantize(
                    w.unsqueeze(1),
                    self._quantizer.scale,
                    self._quantizer.zero_point,
                    self._quantizer.max_q,
                    self._enable_normal_float,
                ).flatten()
                quant_weight_block[:, i] = q
                losses_block[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                weight_block[:, i:] -= err1.unsqueeze(1).matmul(
                    hessian_inverse_block[i, i:].unsqueeze(0)
                )
                error_block[:, i] = err1

            quant_weight[:, i1:i2] = quant_weight_block
            losses[:, i1:i2] = losses_block / 2

            weight[:, i2:] -= error_block.matmul(hessian_inverse[i1:i2, i2:])

        if _torch.cuda.is_available():
            _torch.cuda.synchronize()

        _logger.info(
            "time %.2f, weight quantization error %.2f"
            % (_time.time() - tick, _torch.sum(losses).item())
        )

        if self._use_activation_order_heuristic:
            inverse_perm = _torch.argsort(perm)
            quant_weight = quant_weight[:, inverse_perm]

        self._layer.weight.data = quant_weight.reshape(self._layer.weight.shape).to(
            self._layer.weight.data.dtype
        )
        _logger.debug(
            "quantization error in output activations = %.2f"
            % (_torch.sum((self._layer(self._inp1) - self._out1) ** 2))
        )

    def _get_compression_metadata(self, param_name, param):
        metadata = _CompressionMetadata(param_name)

        scale = _torch.cat(self._scale, dim=1)
        if self._enable_normal_float:
            metadata.compression_type = ["palettization"]
            metadata.lut = _normal_float_palette[self._weight_n_bits].unsqueeze(-1)
            for _ in range(param.dim()):
                metadata.lut = metadata.lut.unsqueeze(0)
            metadata.palettization_scale = scale
        else:
            metadata.compression_type = ["quantization"]
            metadata.quantization_n_bits = self._weight_n_bits
            metadata.quantization_scale = scale
            metadata.zero_point = _torch.cat(self._zero_point, dim=1)

        return metadata


@LayerwiseCompressionAlgorithm.register("sparse_gpt")
class SparseGPT(OBSCompressionAlgorithm):
    """
    A post-training compression algorithm based on the paper
    `SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot
    <https://arxiv.org/pdf/2301.00774.pdf>`_.

    Args:
        layer (:obj:`torch.nn.Module`): Module to be compressed.
        config (:py:class:`ModuleSparseGPTConfig`): Config specifying hyper-parameters
            for the SparseGPT algorithm.
    """

    def __init__(self, layer: _nn.Module, config: ModuleSparseGPTConfig):
        super().__init__(layer, config)
        self._init_parameters(config)

    def _init_parameters(self, config: ModuleSparseGPTConfig):
        self._target_sparsity = config.target_sparsity
        self._weight_n_bits = config.weight_n_bits
        self._n_m_ratio = config.n_m_ratio
        self._processing_group_size = config.processing_group_size
        self._enable_normal_float = config.enable_normal_float
        self._hessian_dampening = config.hessian_dampening
        self._quantizer = None
        if self._weight_n_bits < 16:
            per_channel = config.quantization_granularity in [
                QuantizationGranularity.per_channel,
                QuantizationGranularity.per_block,
            ]
            self._quantizer = _Quantizer(
                n_bits=self._weight_n_bits,
                per_channel=per_channel,
                symmetric=config.quantization_scheme == _QuantizationScheme.symmetric,
                enable_normal_float=config.enable_normal_float,
            )
            self._scale = []
            self._zero_point = []
        if self._n_m_ratio is not None:
            self._prune_n, self._prune_m = self._n_m_ratio
        else:
            self._prune_n, self._prune_m = 0, 0

    def _compress_impl(self):
        weight = self._layer.weight.data.clone()
        if isinstance(self._layer, _nn.Conv2d):
            weight = weight.flatten(1)

        if self._config.weight_dtype in [_torch.float32, _torch.float16]:
            weight = weight.to(self._config.weight_dtype)

        if self._quantizer is not None and not self._quantizer.ready():
            self._quantizer.find_params(weight, weight=True)
            self._store_quantization_params(self._quantizer)

        tick = _time.time()

        hessian = self._hessian

        del self._hessian
        dead = _torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        losses = _torch.zeros(self._rows, device=self._device)

        damp = self._hessian_dampening * _torch.mean(_torch.diag(hessian))
        diag = _torch.arange(self._columns, device=self._device)
        hessian[diag, diag] += damp
        hessian = _torch.linalg.cholesky(hessian)
        hessian = _torch.cholesky_inverse(hessian)
        hessian = _torch.linalg.cholesky(hessian, upper=True)
        hessian_inverse = hessian

        # Hessian computation happens in float32, and _torch.linalg.cholesky does not support float16, so we cast here
        if self._config.weight_dtype in [_torch.float32, _torch.float16]:
            hessian_inverse = hessian_inverse.to(self._config.weight_dtype)

        mask = None

        for i1 in range(0, self._columns, self._processing_group_size):
            i2 = min(i1 + self._processing_group_size, self._columns)
            count = i2 - i1

            weight_block = weight[:, i1:i2].clone()
            quant_weight_block = _torch.zeros_like(weight_block)
            error_block = _torch.zeros_like(weight_block)
            losses_block = _torch.zeros_like(weight_block)
            hessian_inverse_block = hessian_inverse[i1:i2, i1:i2]

            if self._prune_n == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = (
                        weight_block**2
                        / (_torch.diag(hessian_inverse_block).reshape((1, -1))) ** 2
                    )
                    thresh = _torch.sort(tmp.flatten())[0][int(tmp.numel() * self._target_sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = _torch.zeros_like(weight_block) == 1

            for i in range(count):
                w = weight_block[:, i]
                d = hessian_inverse_block[i, i]

                if self._prune_n != 0 and i % self._prune_m == 0:
                    tmp = (
                        weight_block[:, i : (i + self._prune_m)] ** 2
                        / (
                            _torch.diag(hessian_inverse_block)[i : (i + self._prune_m)].reshape(
                                (1, -1)
                            )
                        )
                        ** 2
                    )
                    mask1.scatter_(
                        1,
                        i + _torch.topk(tmp, self._prune_n, dim=1, largest=False)[1],
                        True,
                    )

                q = w.clone()
                q[mask1[:, i]] = 0

                if self._quantizer is not None:
                    q = _quantize(
                        q.unsqueeze(1),
                        self._quantizer.scale,
                        self._quantizer.zero_point,
                        self._quantizer.max_q,
                        self._enable_normal_float,
                    ).flatten()

                quant_weight_block[:, i] = q
                losses_block[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                weight_block[:, i:] -= err1.unsqueeze(1).matmul(
                    hessian_inverse_block[i, i:].unsqueeze(0)
                )
                error_block[:, i] = err1

            weight[:, i1:i2] = quant_weight_block
            losses += _torch.sum(losses_block, 1) / 2

            weight[:, i2:] -= error_block.matmul(hessian_inverse[i1:i2, i2:])

        if _torch.cuda.is_available():
            _torch.cuda.synchronize()

        _logger.info(
            "time %.2f, weight quantization error %.2f"
            % (_time.time() - tick, _torch.sum(losses).item())
        )

        self._layer.weight.data = weight.reshape(self._layer.weight.shape).to(
            self._layer.weight.data.dtype
        )
        _logger.debug(
            "quantization error in output activations = %.2f"
            % (_torch.sum((self._layer(self._inp1) - self._out1) ** 2))
        )

    def _get_compression_metadata(self, param_name, param):
        metadata = _CompressionMetadata(param_name)
        compression_type = ["pruning"]

        if not self._quantizer:
            metadata.compression_type = compression_type
            return metadata

        scale = _torch.cat(self._scale, dim=1)
        if self._enable_normal_float:
            compression_type.append("palettization")
            metadata.lut = _normal_float_palette[self._weight_n_bits].unsqueeze(-1)
            for _ in range(param.dim()):
                metadata.lut = metadata.lut.unsqueeze(0)
            metadata.palettization_scale = scale
        else:
            compression_type.append("quantization")
            metadata.quantization_n_bits = self._weight_n_bits
            metadata.quantization_scale = scale
            metadata.zero_point = _torch.cat(self._zero_point, dim=1)

        metadata.compression_type = compression_type
        return metadata
