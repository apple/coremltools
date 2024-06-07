#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.optimize.torch import (
    base_model_optimizer,
    layerwise_compression,
    optimization_config,
    palettization,
    pruning,
    quantization,
)

from ._logging import init_root_logger as _init_root_logger

_logger = _init_root_logger()
