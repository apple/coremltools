from coremltools._deps import HAS_TF, HAS_TF_1
register_tf_op = None

# suppress TensorFlow stdout prints
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

if HAS_TF:
    from .load import load
if HAS_TF_1:
    from .dialect_ops import * # register tf extension ops
    from .tf_op_registry import register_tf_op
