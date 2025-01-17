#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging as _logging
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import cast as _cast

import torch as _torch

from coremltools.optimize.torch._utils.joint_compression_utils import (
    is_palettized_module as _is_palettized_module,
)
from coremltools.optimize.torch._utils.joint_compression_utils import (
    is_quantized_module as _is_quantized_module,
)
from coremltools.optimize.torch._utils.metadata_utils import (
    CompressionMetadata as _CompressionMetadata,
)
from coremltools.optimize.torch._utils.torch_utils import get_atomic_layers as _get_atomic_layers

logger = _logging.getLogger(__name__)


def lerp(v0, v1, t):
    return v0 + (v1 - v0) * t


def spline(v0, v1, t, power):
    one_m_t = 1.0 - t
    x = one_m_t**power
    return lerp(v1, v0, x)


def magnitude_ranked_mask(
    weights: _torch.Tensor, sparsity_fraction: float, block_size: int, granularity: str
) -> _torch.Tensor:
    """
    Compute a binary mask for pruning based on magnitude-based ranking
    If granularity is `per_scalar`, L1 norm is used. L2 is used otherwise
    """
    shape = weights.shape
    rank = len(shape)

    # rank 1: flattened global unstructured weights, rank 2: torch.Linear, rank 3: torch.Conv1d,
    # rank 4: torch.Conv2d, rank 5: torch.Conv3d
    assert rank in [1, 2, 3, 4, 5], f"weights tensor rank must be in [1, 2, 3, 4, 5], got {rank}"

    if granularity == "per_scalar" or rank == 2:
        magnitude_map = weights.abs()
        nb_weight_components = weights.numel()

    elif rank in [3, 4, 5]:
        if granularity == "per_kernel":
            start_dim = 2
            nb_weight_components = shape[0] * shape[1]
        elif granularity == "per_channel":
            start_dim = 1
            nb_weight_components = shape[0]
        else:
            raise ValueError(f"Unsupported granularity for magnitude_ranked_mask: {granularity}")

        # Compute L2 norm per weight slice (as defined by the granularity)
        magnitude_map = _torch.norm(weights.flatten(start_dim), dim=-1)
        for _ in range(rank - start_dim):
            magnitude_map = magnitude_map.unsqueeze(-1)

    if block_size > 1:
        ch_shape = shape[0]
        if ch_shape % block_size != 0:
            # Since the number of channels isn't divisible by block size,
            # we shall pad the channels so that it is divisible
            pad_shape = list(magnitude_map.shape)
            pad_shape[0] = block_size - ch_shape % block_size
            magnitude_map = _torch.cat(
                [magnitude_map, _torch.zeros(pad_shape, device=magnitude_map.device)], dim=0
            )
            ch_shape = magnitude_map.shape[0]
            assert ch_shape % block_size == 0

        # Reshape to expose the "block" sub-axis
        s = list(magnitude_map.shape)  # block exposed shape
        s.insert(1, block_size)
        s[0] = int(s[0] / block_size)
        f = [-1] * len(s)  # expand factors to recover orig shape
        f[1] = block_size
        magnitude_map = (
            magnitude_map.view(s)
            .pow(2)
            .sum(1, keepdim=True)
            .sqrt()
            .expand(f)
            .contiguous()
            .view(magnitude_map.shape)
        )

        # Reshape to original shape in case of padding
        magnitude_map = magnitude_map[: shape[0]]

    nb_nonzero = _torch.ceil(
        _torch.as_tensor(nb_weight_components, dtype=_torch.float32) * (1 - sparsity_fraction)
    ).int()

    # handle special case when sparsity_fraction = 1.0
    if nb_nonzero == 0:
        thr = 1.0 + magnitude_map.flatten().max()
    else:
        thr = (
            magnitude_map.flatten().sort()[0].flip(0)[nb_nonzero - 1]
        )  # produces same mask for 1.0 and 0.0 sparsity

    mask = _torch.greater_equal(magnitude_map, thr)

    return mask


def n_m_mask(weights: _torch.Tensor, nm: _Tuple[int, int], dim: _Optional[int] = 1):
    """
    Create a n:m sparsity mask.
    """
    shape = weights.shape
    permuted_shape = shape
    rank = len(shape)
    num_zeros, block_size = nm
    mask_value = 0.0

    assert num_zeros < block_size, (
        f"n (number of zeros) = {num_zeros} must be " f"less than m (block size) = {block_size}"
    )

    assert dim in [0, 1], (
        f"n:m mask is supported along dimensions (0, 1), "
        f"corresponding to input and output channels. Received "
        f"dim = {dim}"
    )
    # rank 2: torch.Linear, rank 3: torch.Conv1d,
    # rank 4: torch.Conv2d, rank 5: torch.Conv3d
    assert rank in [2, 3, 4, 5], f"weights tensor rank must be in [2, 3, 4, 5], got {rank}"

    # num_non_zeros = block_size - num_zeros

    # if n:m is required along C_o, flip C_i and C_o
    if dim == 0:
        weights = _torch.permute(weights, [1, 0] + list(range(2, rank)))
    # transform to A x C_i
    # For Conv1D: C_o x C_i x H         ==>         H x C_o x C_i ==> H*C_o     x C_i
    # For Conv2D: C_o x C_i x H x W     ==>     H x W x C_o x C_i ==> H*W*C_o   x C_i
    # For Conv3D: C_o x C_i x H x W x D ==> H x W x D x C_o x C_i ==> H*W*D*C_o x C_i
    if rank > 2:
        permute_array = list(range(2, rank)) + [0, 1]
        weights = _torch.permute(weights, permute_array)
        permuted_shape = weights.shape
        weights = _torch.reshape(weights, (-1, weights.shape[-1]))

    abs_weights = weights.abs()
    padding_size = block_size - abs_weights.shape[-1] % block_size
    abs_weights_pad = _torch.nn.functional.pad(abs_weights, (0, padding_size), mode="constant")

    num_blocks = abs_weights_pad.numel() // block_size
    weights_blocks = abs_weights_pad.view(num_blocks, block_size)

    indices = _torch.argsort(weights_blocks, dim=1)[:, :num_zeros]
    sparsity_mask = _torch.ones([num_blocks, block_size], device=weights.device)
    sparsity_mask.scatter_(dim=1, index=indices, value=mask_value)
    sparsity_mask = sparsity_mask.view(abs_weights_pad.shape)
    sparsity_mask = sparsity_mask[:, : abs_weights.shape[-1]]

    # revert changes to mask shape to achieve same size as original weight
    if rank > 2:
        sparsity_mask = _torch.reshape(sparsity_mask, permuted_shape)
        permute_array = [rank - 2, rank - 1] + list(range(0, rank - 2))
        sparsity_mask = _torch.permute(sparsity_mask, permute_array)
    if dim == 0:
        sparsity_mask = _torch.permute(sparsity_mask, [1, 0] + list(range(2, rank)))

    return sparsity_mask


def block2_sparsity(weight: _torch.Tensor) -> _torch.Tensor:
    n = weight.size(0)
    assert n % 2 == 0
    return weight.flatten(1).view(n // 2, 2, -1).sum(1).eq(0.0).float().mean().item()


def structured_sparsity(weight: _torch.Tensor) -> _torch.Tensor:
    return weight.flatten(1).sum(1).eq(0.0).float().mean().item()


def unstructured_sparsity(weight: _torch.Tensor) -> _torch.Tensor:
    return weight.eq(0.0).float().mean().item()


def get_global_sparsity_summaries(
    layer_sparsities: _List[_torch.Tensor], layer_numel: _List[int]
) -> float:
    assert len(layer_sparsities) == len(layer_numel)

    weighted_sum, denom = 0.0, 0.0
    for sparsity, numel in zip(layer_sparsities, layer_numel):
        if sparsity >= 0.0:
            denom += numel
            weighted_sum += numel * _cast(float, sparsity)

    if _torch.all(_torch.tensor(layer_sparsities) < 0):
        # to indicate the sparsity type is not applicable
        return -1

    assert denom > 0.0
    return weighted_sum / denom


def validate_allowed_granularity_values(instance, attribute, value):
    if value is None:
        return
    allowed_values = ["per_scalar", "per_kernel", "per_channel", "per_layer"]
    if value not in allowed_values:
        raise ValueError(
            f"Allowed values for granularity are: {', '.join(allowed_values)}. "
            f"Received: {value}"
        )


def register_compression_metadata(submodule, pruner_info, supported_modules):
    config = pruner_info.config
    compression_type = ["pruning"]

    # Identify joint compression cases
    if _is_quantized_module(pruner_info.module):
        compression_type += ["quantization"]
        pruning_support_layers = _get_atomic_layers(submodule, supported_modules)
        submodule = list(pruning_support_layers.values())[0]
    elif _is_palettized_module(pruner_info.module):
        compression_type += ["palettization"]

    param_name = config.param_name
    metadata = _CompressionMetadata(param_name)
    metadata.compression_type = compression_type
    metadata.register(submodule, override_compression_type=(len(compression_type) > 1))
