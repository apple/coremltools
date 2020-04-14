from coremltools.converters.nnv2 import testing_reqs
from coremltools.converters.nnv2.frontend.tensorflow.test import testing_utils as tf_testing_utils
from coremltools.converters.nnv2.frontend.tensorflow2.test.testing_utils import (
    make_tf2_graph, run_compare_tf2,
)
from coremltools.converters.nnv2.testing_reqs import *

tf = pytest.importorskip('tensorflow', minversion='2.1.0')

backends = testing_reqs.backends

# -----------------------------------------------------------------------------
# Overwrite utilities to enable different conversion / compare method
tf_testing_utils.frontend = 'tensorflow2'
tf_testing_utils.make_tf_graph = make_tf2_graph
tf_testing_utils.run_compare_tf = run_compare_tf2

# -----------------------------------------------------------------------------
# Import TF 2.x-compatible TF 1.x test cases
from coremltools.converters.nnv2.frontend.tensorflow.test.test_ops import (
    TestActivationReLU,
    TestConv,
    TestConv3d,
)
