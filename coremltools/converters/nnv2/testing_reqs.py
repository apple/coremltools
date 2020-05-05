import os
import itertools
import numpy as np
import pytest

from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from coremltools.converters.nnv2.nnv2_program.ops.registry import SSAOpRegistry

from coremltools.converters.nnv2.builtin_types import builtins
from coremltools._deps import (
    HAS_TF_1, HAS_TF_2, HAS_TORCH,
    MSG_TF1_NOT_FOUND, MSG_TF2_NOT_FOUND
)
from .testing_utils import (
    ssa_fn, is_close, random_gen, converter, _converter
)

backends = _converter.ConverterRegistry.backends.keys()

np.random.seed(1984)

if HAS_TF_1 or HAS_TF_2:
    import tensorflow as tf

    tf.set_random_seed(1234) if HAS_TF_1 else tf.random.set_seed(1234)

if HAS_TORCH:
    import torch
