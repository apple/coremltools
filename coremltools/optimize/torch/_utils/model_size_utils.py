#  Copyright (c) 2025, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import math

import torch.nn

from coremltools.optimize.torch._utils.metadata_utils import CompressionMetadata, CompressionType


def n_bits_from_lut(lut):
    n_bits = math.log2(lut.shape[-2])
    assert n_bits.is_integer()
    return int(n_bits)


def get_compressed_size(module: torch.nn.Module) -> int:
    """
    Returns size in bits for a compressed module
    """
    compression_metadata = CompressionMetadata.from_state_dict(module.state_dict())
    size = 0
    for param_name, param in module.named_parameters():
        if param_name in compression_metadata:
            metadata = compression_metadata[param_name]
            compression_types = [CompressionType(x) for x in metadata.compression_type]

            if compression_types == [CompressionType.palettization] or compression_types == [
                CompressionType.palettization,
                CompressionType.quantization,
            ]:
                n_bits = n_bits_from_lut(metadata.lut)
                lut_precision = 16 if metadata.palettization_scale is None else 8
                param = module.get_parameter(param_name)
                size += metadata.lut.numel() * lut_precision + param.numel() * n_bits

        else:
            size += param.numel() * torch.finfo(param.dtype).bits

    return size
