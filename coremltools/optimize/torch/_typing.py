#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Any as _Any
from typing import Callable as _Callable
from typing import Dict as _Dict

import torch as _torch

ParamsDict = _Dict[str, _Any]
TensorCallable = _Callable[[_torch.Tensor], _torch.Tensor]
