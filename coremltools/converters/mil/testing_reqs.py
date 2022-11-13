#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import os

import numpy as np
import pytest

import coremltools as ct
from coremltools._deps import (_HAS_TF_1, _HAS_TF_2, _HAS_TORCH)


# Setting up backend / precision
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
        
# Setting up compute unit
compute_units = []
if 'COMPUTE_UNITS' in os.environ:
    for i, cur_str_val in enumerate(os.environ['COMPUTE_UNITS'].split(',')):
        cur_str_val = cur_str_val.strip().upper()
        if cur_str_val not in ct.ComputeUnit.__members__:
            raise ValueError("Compute unit \"{}\" not supported in coremltools.".format(cur_str_val))
        compute_units.append(ct.ComputeUnit[cur_str_val])
else:
    compute_units = [ct.ComputeUnit.CPU_ONLY]

np.random.seed(1984)

if _HAS_TF_1:
    tf = pytest.importorskip("tensorflow")
    tf.compat.v1.set_random_seed(1234)

if _HAS_TF_2:
    tf = pytest.importorskip("tensorflow")
    tf.random.set_seed(1234)
