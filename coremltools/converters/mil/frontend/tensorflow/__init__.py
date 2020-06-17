from coremltools._deps import _HAS_TF_1

# suppress TensorFlow stdout prints
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

register_tf_op = None

if _HAS_TF_1:
    from .ops import *  # register all
    from .dialect_ops import *  # register tf extension ops
    from .tf_op_registry import register_tf_op
