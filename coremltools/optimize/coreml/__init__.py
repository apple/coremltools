#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from ._config import (
    OpActivationLinearQuantizerConfig,
    OpLinearQuantizerConfig,
    OpMagnitudePrunerConfig,
    OpPalettizerConfig,
    OpThresholdPrunerConfig,
    OptimizationConfig,
)
from ._post_training_quantization import (
    activations_quantization,
    CoreMLOpMetaData,
    CoreMLWeightMetaData,
    decompress_weights,
    get_weights_metadata,
    linear_quantize_weights,
    palettize_weights,
    prune_weights,
)

# Import to make sure the pass is registered.
from ._quantization_passes import activations_quantization as _activations_quantization