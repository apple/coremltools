import os
import itertools
import numpy as np
import pytest

from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from coremltools.converters.nnv2.nnv2_program.ops.registry import SSAOpRegistry

from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2._deps import (
    HAS_TF1, HAS_TF2, HAS_PYTORCH,
    MSG_TF1_NOT_FOUND, MSG_TF2_NOT_FOUND
)
import coremltools.converters.nnv2.converter as converter

from . import _test_utils as _utils
from ._test_utils import (
    run_compare_builder,
    run_compare_tf1,
    run_compare_tf2,
    run_compare_tf_keras,
    ssa_fn,
    is_close,
    _random_gen as random_gen)

_utils.converter = converter
backends = converter.ConverterRegistry.backends.keys()

if HAS_TF1 or HAS_TF2:
    import tensorflow as tf

    tf.set_random_seed(1234) if HAS_TF1 else tf.random.set_seed(1234)

if HAS_PYTORCH:
    import torch
