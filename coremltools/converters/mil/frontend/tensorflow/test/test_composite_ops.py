#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

from coremltools.converters.mil.frontend.tensorflow.test.testing_utils import (
    TensorFlowBaseTest, make_tf_graph)
# Importing _TF_OPS_REGISTRY to ensure `overriding` existing TF op does not break
# testing of default op
# pytest imports all the tests and hence overriding op invokes custom op which is not expected
# In real usecase, importing following is not recommended!!
from coremltools.converters.mil.frontend.tensorflow.tf_op_registry import (
    _TF_OPS_REGISTRY, register_tf_op)
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_reqs import backends, compute_units
from coremltools.converters.mil.testing_utils import random_gen

tf = pytest.importorskip("tensorflow")

class TestCompositeOp(TensorFlowBaseTest):
    @pytest.fixture(scope="class")
    def create_custom_selu(self):
        default_selu = _TF_OPS_REGISTRY.get("Selu", None)

        @register_tf_op(tf_alias=[], override=True)
        def Selu(context, node):
            x = context[node.inputs[0]]
            alpha = 1.6732631921768188
            lamda = 1.0507010221481323
            out_elu = mb.elu(x=x, alpha=alpha)
            out = mb.mul(x=out_elu, y=lamda, name=node.name)
            context.add(node.name, out)

        yield

        _TF_OPS_REGISTRY["Selu"] = default_selu

    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            backends,
            list(range(1, 5))
        ),
    )
    @pytest.mark.usefixtures("create_custom_selu")
    def test_selu(self, compute_unit, backend, rank):
        input_shape = np.random.randint(low=1, high=6, size=rank)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.keras.activations.selu(x)

        model, inputs, outputs = build_model

        input_values = [random_gen(input_shape, -10.0, 10.0)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend,
        )
