from .op_registry import register_tf_op

# TF 2.x now imports and registers all TF 1.x op against the new registry
# (separated from TF 1.x registry). Overwrite might needed in case the op
# semantics are different between TF 1.x and TF 2.x.
from coremltools.converters.nnv2.frontend.tensorflow.ops import *
