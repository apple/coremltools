#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.frontend.tensorflow.test import (
    testing_utils as tf_testing_utils,
)
from coremltools.converters.mil.frontend.tensorflow2.test.testing_utils import (
    make_tf2_graph as make_tf_graph,
    run_compare_tf2 as run_compare_tf,
)
from coremltools.converters.mil.testing_reqs import *

tf = pytest.importorskip("tensorflow", minversion="2.1.0")

backends = testing_reqs.backends

# -----------------------------------------------------------------------------
# Overwrite utilities to enable different conversion / compare method
tf_testing_utils.frontend = "TensorFlow2"
tf_testing_utils.make_tf_graph = make_tf_graph
tf_testing_utils.run_compare_tf = run_compare_tf

# -----------------------------------------------------------------------------
# Import TF 2.x-compatible TF 1.x test cases
from coremltools.converters.mil.frontend.tensorflow.test.test_composite_ops import (
    TestCompositeOp,
)
