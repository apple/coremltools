#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools._deps import _IMPORT_CT_OPTIMIZE_TORCH

from . import coreml

if _IMPORT_CT_OPTIMIZE_TORCH:
    from . import torch
