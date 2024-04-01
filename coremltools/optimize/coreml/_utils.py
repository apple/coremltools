# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from typing import Tuple, Optional


def get_quant_range(n_bits: int, signed: bool, mode: str) -> Tuple[int, int]:
    """
    Utility to get the quantization range for a given quantization config
    Adapted from phoenix/quatization/_utils.py
    """
    max_q = 2**n_bits
    if not signed:
        quant_min = 0
        quant_max = max_q - 1
        if mode == "LINEAR_SYMMETRIC":
            quant_max -= 1
    else:
        quant_min = -max_q / 2
        quant_max = max_q / 2 - 1
        if mode == "LINEAR_SYMMETRIC":
            quant_min += 1
    return int(quant_min), int(quant_max)


def quantize_weight(
    weight: np.ndarray,
    axes: Tuple[int, ...],
    nbits: int,
    signed: bool,
    quantization_mode: str,
    dtype: np.dtype,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Get quantized data along with metadata (scale, zero_point)."""
    if not np.issubdtype(weight.dtype, np.floating):
        raise ValueError("Only floating numpy arrays are supported.")

    val_min = np.amin(weight, axis=axes, keepdims=True)
    val_max = np.amax(weight, axis=axes, keepdims=True)

    q_val_min, q_val_max = get_quant_range(nbits, signed, quantization_mode)

    zero_point = None
    if quantization_mode == "LINEAR_SYMMETRIC":
        # For the linear_symmetric quantization_mode, the range is symmetrical to 0
        max_abs = np.maximum(np.abs(val_min), np.abs(val_max))
        val_min = -max_abs
        val_max = max_abs

        if not signed:
            zero_point_shift = q_val_max // 2
            zero_point = zero_point_shift * np.ones(val_min.shape)
    else:
        assert quantization_mode == "LINEAR"
        # For the linear quantization_mode, we need to make sure the data range contains `0`
        val_min = np.minimum(0.0, val_min)
        val_max = np.maximum(0.0, val_max)
        zero_point = (q_val_min * val_max - q_val_max * val_min) / (val_max - val_min)
        zero_point = np.round(zero_point)
        zero_point = np.clip(zero_point, q_val_min, q_val_max)

    scale = (val_max - val_min) / (q_val_max - q_val_min)
    quantized_data = np.round(weight / scale)
    if zero_point is not None:
        quantized_data += zero_point
        zero_point = zero_point.squeeze().astype(dtype)
    quantized_data = np.clip(quantized_data, q_val_min, q_val_max).astype(dtype)
    scale = scale.astype(weight.dtype).squeeze()

    return quantized_data, scale, zero_point
