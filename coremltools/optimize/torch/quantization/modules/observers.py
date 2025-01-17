#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Any as _Any
from typing import Dict as _Dict

import torch as _torch
import torch.ao.quantization as _aoquant


class NoopObserver(_aoquant.NoopObserver):
    """
    Extends aoquant.NoopObserver to add support for accepting factory_kwargs which are
    passed to it during qconfig.weight() creation in QAT Conv/Linear modules.
    """

    def __init__(
        self,
        dtype: _torch.dtype = _torch.float16,
        custom_op_name: str = "",
        factory_kwargs: _Dict[str, _Any] = None,
    ):
        super().__init__(dtype, custom_op_name)
