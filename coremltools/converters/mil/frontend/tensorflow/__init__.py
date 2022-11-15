#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging
# suppress TensorFlow stdout prints
import os

from coremltools._deps import _HAS_TF

if os.getenv("TF_SUPPRESS_LOGS", "1") == "1":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
    logging.getLogger("tensorflow").setLevel(logging.FATAL)

register_tf_op = None

if _HAS_TF:
    # Importing these causes them to register their ops
    from . import ops
    from .dialect_ops import (TfLSTMBase, tf_lstm_block, tf_lstm_block_cell,
                              tf_make_list)
    from .tf_op_registry import register_tf_op
