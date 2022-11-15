#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
from coremltools.converters.mil._deployment_compatibility import \
    AvailableTarget as target

_IOS16_TARGET = target.iOS16

from .constexpr_ops import (constexpr_affine_dequantize, constexpr_cast,
                            constexpr_lut_to_dense, constexpr_sparse_to_dense)
from .image_resizing import crop_resize, resample, upsample_bilinear
from .scatter_gather import gather, gather_nd
from .tensor_operation import fill_like, topk
from .tensor_transformation import pixel_unshuffle, reshape_like
