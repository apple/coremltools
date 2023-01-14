#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools._deps import _HAS_PADDLE

register_paddle_op = None

if _HAS_PADDLE:
    from .dialect_ops import (paddle_tensor_assign, paddle_upsample_bilinear,
                              paddle_upsample_nearest_neighbor)
    from .load import load
    from .paddle_op_registry import register_paddle_op
