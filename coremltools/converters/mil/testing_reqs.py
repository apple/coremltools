#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os
import itertools
import numpy as np
from numpy import linalg as la
import pytest

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.ops.registry import SSAOpRegistry

from coremltools.converters.mil.mil import types
from coremltools._deps import (
    _HAS_TF_1,
    _HAS_TF_2,
    _HAS_TORCH,
    MSG_TF1_NOT_FOUND,
    MSG_TF2_NOT_FOUND,
)
from .testing_utils import ssa_fn, random_gen
import coremltools as ct

backends = []
if 'PYMIL_TEST_TARGETS' in os.environ:
    targets = os.environ['PYMIL_TEST_TARGETS'].split(',')
    for i in range(len(targets)):
        targets[i] = targets[i].strip()

    if 'mlprogram' in targets:
        backends.append(('mlprogram', 'fp16'))
        if os.getenv('INCLUDE_MIL_FP32_UNIT_TESTS') == '1':
            backends.append(('mlprogram', 'fp32'))
    if 'neuralnetwork' in targets:
        backends.append(('neuralnetwork', 'fp32'))

    if not backends:
        raise ValueError("PYMIL_TEST_TARGETS can be set to one or more of: neuralnetwork, mlprogram")
else:
    backends = [('mlprogram', "fp16"), ('neuralnetwork', "fp32")]
    if os.getenv('INCLUDE_MIL_FP32_UNIT_TESTS') == '1':
            backends.append(('mlprogram', 'fp32'))

np.random.seed(1984)

if _HAS_TF_1:
    tf = pytest.importorskip("tensorflow")
    tf.compat.v1.set_random_seed(1234)

if _HAS_TF_2:
    tf = pytest.importorskip("tensorflow")
    tf.random.set_seed(1234)

if _HAS_TORCH:
    import torch
