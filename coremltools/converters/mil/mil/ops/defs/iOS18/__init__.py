#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil._deployment_compatibility import AvailableTarget as target
from coremltools.converters.mil.mil.ops.registry import SSAOpRegistry

# Ensure op registrations recognize the new opset.
_IOS18_TARGET = target.iOS18

from .compression import (
    constexpr_blockwise_shift_scale,
    constexpr_lut_to_dense,
    constexpr_lut_to_sparse,
    constexpr_sparse_blockwise_shift_scale,
    constexpr_sparse_to_dense,
)
from .recurrent import gru
from .states import read_state
from .tensor_transformation import slice_update
from .transformers import scaled_dot_product_attention
