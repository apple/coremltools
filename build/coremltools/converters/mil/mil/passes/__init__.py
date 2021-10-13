#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# Import all passes in this dir
from os.path import dirname, basename, isfile, join
import glob

excluded_files = ["pass_registry.py", "apply_common_pass_pipeline.py", "__init__.py"]
modules = glob.glob(join(dirname(__file__), "*.py"))
pass_modules = [
    basename(f)[:-3]
    for f in modules
    if isfile(f)
    and basename(f)[:1] != "_"  # Follow python convention to hide _* files.
    and basename(f)[:4] != "test"
    and basename(f) not in excluded_files
]
__all__ = pass_modules

from . import *  # import everything in __all__

from coremltools.converters.mil.experimental.passes import (
    generic_gelu_tanh_approximation_fusion,
    generic_layernorm_instancenorm_pattern_fusion,
    generic_linear_bias_fusion,
    generic_conv_batchnorm_fusion,
    generic_conv_scale_fusion,
    generic_conv_bias_fusion
)