#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

from coremltools.converters.mil.frontend.tensorflow.test.testing_utils import (
    TensorFlowBaseTest, make_tf_graph)
from coremltools.converters.mil.testing_reqs import backends, compute_units

tf = pytest.importorskip("tensorflow")


class TestTFGraphs(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_masked_input(self, compute_unit, backend):

        input_shape = [4, 10, 8]
        val = np.random.rand(*input_shape).astype(np.float32)

        @make_tf_graph([input_shape])
        def build_model(input):
            sliced_input = input[..., 4]
            mask = tf.where(sliced_input > 0)
            masked_input = tf.gather_nd(input, mask)
            return masked_input

        model, inputs, outputs = build_model

        input_values = [val]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend,
        )
