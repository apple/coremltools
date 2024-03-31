# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Core ML is an Apple framework which allows developers to simply and easily integrate machine
learning (ML) models into apps running on Apple devices (including iOS, watchOS, macOS, and
tvOS). Core ML introduces a public file format (.mlmodel) for a broad set of ML methods
including deep neural networks (both convolutional and recurrent), tree ensembles with boosting,
and generalized linear models. Models in this format can be directly integrated into apps
through Xcode.

Coremltools is a python package for creating, examining, and testing models in the .mlpackage
and .mlmodel formats. In particular, it can be used to:

* Convert existing models to .mlpackage or .mlmodel formats from popular machine learning tools including:
     PyTorch, TensorFlow, scikit-learn, XGBoost and libsvm.
* Express models in .mlpackage and .mlmodel formats through a simple API.
* Make predictions with .mlpackage and .mlmodel files (on macOS).

For more information: http://developer.apple.com/documentation/coreml
"""

from enum import Enum as _Enum
from logging import getLogger as _getLogger

from .version import __version__

_logger = _getLogger(__name__)

# This is the basic Core ML specification format understood by iOS 11.0
SPECIFICATION_VERSION = 1

# New versions for iOS 11.2 features. Models which use these features should have these
# versions, but models created from this coremltools which do not use the features can
# still have the basic version.
_MINIMUM_CUSTOM_LAYER_SPEC_VERSION = 2
_MINIMUM_FP16_SPEC_VERSION = 2

# New versions for iOS 12.0 features. Models which use these features should have these
# versions, but models created from this coremltools which do not use the features can
# still have the basic version.
_MINIMUM_CUSTOM_MODEL_SPEC_VERSION = 3
_MINIMUM_QUANTIZED_MODEL_SPEC_VERSION = 3
_MINIMUM_FLEXIBLE_SHAPES_SPEC_VERSION = 3

# New versions for iOS 13.0.
_MINIMUM_NDARRAY_SPEC_VERSION = 4
_MINIMUM_NEAREST_NEIGHBORS_SPEC_VERSION = 4
_MINIMUM_LINKED_MODELS_SPEC_VERSION = 4
_MINIMUM_UPDATABLE_SPEC_VERSION = 4
_SPECIFICATION_VERSION_IOS_13 = 4

# New versions for iOS 14.0
_SPECIFICATION_VERSION_IOS_14 = 5

# New versions for iOS 15.0
_SPECIFICATION_VERSION_IOS_15 = 6

# New versions for iOS 16.0
_SPECIFICATION_VERSION_IOS_16 = 7

# New versions for iOS 17.0
_SPECIFICATION_VERSION_IOS_17 = 8


class ComputeUnit(_Enum):
    '''
    The set of processing-unit configurations the model can use to make predictions.
    '''
    ALL = 1  # Allows the model to use all compute units available, including the neural engine
    CPU_AND_GPU = 2 # Allows the model to use both the CPU and GPU, but not the neural engine
    CPU_ONLY = 3 # Limit the model to only use the CPU
    CPU_AND_NE = 4 # Allows the model to use both the CPU and neural engine, but not the GPU.
                   # Only available on macOS >= 13.0

# A dictionary that maps the CoreML model specification version to the MLProgram/MIL opset string
_OPSET = {
    _SPECIFICATION_VERSION_IOS_13: "CoreML3",
    _SPECIFICATION_VERSION_IOS_14: "CoreML4",
    _SPECIFICATION_VERSION_IOS_15: "CoreML5",
    _SPECIFICATION_VERSION_IOS_16: "CoreML6",
    _SPECIFICATION_VERSION_IOS_17: "CoreML7",
}

# Default specification version for each backend
_LOWEST_ALLOWED_SPECIFICATION_VERSION_FOR_NEURALNETWORK = _SPECIFICATION_VERSION_IOS_13
_LOWEST_ALLOWED_SPECIFICATION_VERSION_FOR_MILPROGRAM = _SPECIFICATION_VERSION_IOS_15


# expose sub packages as directories
from . import converters, models, optimize, proto
# expose unified converter in coremltools package level
from .converters import ClassifierConfig
from .converters import ColorLayout as colorlayout
from .converters import EnumeratedShapes, ImageType, RangeDim, Shape, TensorType, convert
from .converters.mil._deployment_compatibility import AvailableTarget as target
from .converters.mil.mil.passes.defs import quantization as transform
from .converters.mil.mil.passes.defs.quantization import ComputePrecision as precision
from .converters.mil.mil.passes.pass_pipeline import PassPipeline
from .models import utils
from .models.ml_program import compression_utils

try:
    from . import libcoremlpython
except:
    pass

# Time profiling for functions in coremltools package, decorated with @profile
import os as _os
import sys as _sys

from .converters._profile_utils import _profiler

_ENABLE_PROFILING = _os.environ.get("ENABLE_PROFILING", False)

if _ENABLE_PROFILING:
    _sys.setprofile(_profiler)
