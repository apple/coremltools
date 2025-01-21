#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# TODO: coreml.experimental.linear_quantize_activations has been deprecated and related code will be removed in rdar://140233515.
from ._config import OpActivationLinearQuantizerConfig
from ._post_training_quantization import linear_quantize_activations
