#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import itertools
import numpy as np
import pytest

from coremltools._deps import MSG_TF1_NOT_FOUND
from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.testing_reqs import backends
from coremltools.converters.mil.frontend.tensorflow.test.testing_utils import (
    tf_graph_to_mlmodel
)

if testing_reqs._HAS_TF_1:
    from coremltools.converters.mil.testing_reqs import tf

# Custom Op imports
from coremltools.converters.mil.frontend.tensorflow.tf_op_registry import register_tf_op

# Importing _TF_OPS_REGISTRY to ensure `overriding` existing TF op does not break
# testing of default op
# pytest imports all the tests and hence overriding op invokes custom op which is not expected
# In real usecase, importing following is not recommended!!
from coremltools.converters.mil.frontend.tensorflow.tf_op_registry import (
    _TF_OPS_REGISTRY,
)
from coremltools.converters.mil.testing_utils import random_gen
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil import (
    Builder as mb,
    Operation,
    types,
)
from coremltools.converters.mil.mil.input_type import (
    BoolInputType,
    DefaultInputs,
    InputSpec,
    IntInputType,
    TensorInputType,
)
from coremltools.converters.mil.mil.types.symbolic import is_symbolic


class TestCustomMatMul:
    # Define SSA Custom Op for Sparse MatMul
    # This will map to `custom_op` in SSA with binding information
    # to bind input spec to the custom implementation
    @register_op(doc_str="Sparse MatMul Layer", is_custom_op=True)
    class custom_sparse_matmul(Operation):
        # Defining input spec for current op
        input_spec = InputSpec(
            x=TensorInputType(),
            y=TensorInputType(),
            transpose_x=BoolInputType(const=True, optional=True),
            transpose_y=BoolInputType(const=True, optional=True),
            x_is_sparse=BoolInputType(const=True, optional=True),
            y_is_sparse=BoolInputType(const=True, optional=True),
        )

        # Specifying binding for custom op for specifying inputs,
        # parameters required for creating custom op to be synced with Swift API
        bindings = {
            "class_name": "SparseMatMul",
            "input_order": ["x", "y"],
            "parameters": ["transpose_x", "transpose_y", "x_is_sparse", "y_is_sparse"],
            "description": "Custom Sparse MatMul Layer",
        }

        def default_inputs(self):
            return DefaultInputs(
                transpose_x=False,
                transpose_y=False,
                x_is_sparse=False,
                y_is_sparse=False,
                )

        def __init__(self, **kwargs):
            super(TestCustomMatMul.custom_sparse_matmul, self).__init__(**kwargs)

        def type_inference(self):
            x_type = self.x.dtype
            x_shape = self.x.shape
            y_shape = self.y.shape
            # For illustration purpose, assumming getting valid shape
            # Ideally, should consider transpose_?, ?_is_sparse parameters into consideration
            # for computing output shape
            ret_shape = [x_shape[0], y_shape[1]]
            return types.tensor(x_type, [x_shape[0], y_shape[1]])

    # TensorFlow Sparse Matmul Op
    @register_tf_op()
    def SparseMatMul(context, node):
        a = context[node.inputs[0]]
        b = context[node.inputs[1]]
        transpose_a = node.attr.get("transpose_a", False)
        transpose_b = node.attr.get("transpose_b", False)
        a_is_sparse = node.attr.get("a_is_sparse", False)
        b_is_sparse = node.attr.get("b_is_sparse", False)

        x = mb.custom_sparse_matmul(
            x=a,
            y=b,
            transpose_x=transpose_a,
            transpose_y=transpose_b,
            x_is_sparse=a_is_sparse,
            y_is_sparse=b_is_sparse,
            name=node.name,
        )
        context.add(node.name, x)

    @pytest.mark.skipif(not testing_reqs._HAS_TF_1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize(
        "use_cpu_only, backend, transpose_a, transpose_b," "a_is_sparse, b_is_sparse, b_is_const",
        itertools.product(
            [True], backends, [True, False], [True, False], [True, False], [True, False], [True, False],
        ),
    )
    def test_tf(
        self, use_cpu_only, backend, transpose_a, transpose_b, a_is_sparse, b_is_sparse, b_is_const,
    ):
        if backend[0] == 'mlprogram':
            pytest.xfail("Custom layer not supported with ML Program backend")

        rank = 2
        shape = list(np.random.randint(low=3, high=100, size=1)) * rank
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)

            if b_is_const:
                y_value = random_gen(shape, rand_min=-100, rand_max=100)
                y = tf.constant(y_value, dtype=tf.float32)
            else:
                y = tf.placeholder(tf.float32, shape=shape)

            ref = tf.sparse_matmul(
                x,
                y,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                a_is_sparse=a_is_sparse,
                b_is_sparse=b_is_sparse,
            )
            mlmodel, _, _, _ = tf_graph_to_mlmodel(
                graph,
                {
                    x: random_gen(shape, rand_min=-100, rand_max=100),
                } if b_is_const else {
                    x: random_gen(shape, rand_min=-100, rand_max=100),
                    y: random_gen(shape, rand_min=-100, rand_max=100),
                },
                ref,
                backend=backend,
            )
            layers = mlmodel.get_spec().neuralNetwork.layers
            assert layers[-1].custom is not None, "Expecting a custom layer"
            assert (
                "SparseMatMul" == layers[-1].custom.className
            ), "Custom Layer class name mis-match"
            assert (
                transpose_a == layers[-1].custom.parameters["transpose_x"].boolValue
            ), "Incorrect parameter value k"
            assert (
                transpose_b == layers[-1].custom.parameters["transpose_y"].boolValue
            ), "Incorrect parameter value k"
            assert (
                a_is_sparse == layers[-1].custom.parameters["x_is_sparse"].boolValue
            ), "Incorrect parameter value k"
            assert (
                b_is_sparse == layers[-1].custom.parameters["y_is_sparse"].boolValue
            ), "Incorrect parameter value k"

            assert len(layers) == 2 if b_is_const else len(layers) == 1


class TestCustomTopK:
    @pytest.fixture(scope="class")
    def create_custom_TopK(self):
        # Defining SSA TopK Op
        @register_op(doc_str="Custom TopK Layer", is_custom_op=True)
        class custom_topk(Operation):
            input_spec = InputSpec(
                x=TensorInputType(),
                k=IntInputType(const=True, optional=True),
                axis=IntInputType(const=True, optional=True),
                sorted=BoolInputType(const=True, optional=True),
            )

            bindings = {
                "class_name": "TopK",
                "input_order": ["x"],
                "parameters": ["k", "axis", "sorted"],
                "description": "Top K Custom layer",
            }

            def default_inputs(self):
                return DefaultInputs(
                    k=1,
                    axis=-1,
                    sorted=False,
                    )

            def __init__(self, **kwargs):
                super(custom_topk, self).__init__(**kwargs)

            def type_inference(self):
                x_type = self.x.dtype
                x_shape = self.x.shape
                k = self.k.val
                axis = self.axis.val

                if not is_symbolic(x_shape[axis]) and k > x_shape[axis]:
                    msg = "K={} is greater than size of the given axis={}"
                    raise ValueError(msg.format(k, axis))

                ret_shape = list(x_shape)
                ret_shape[axis] = k
                return types.tensor(x_type, ret_shape), types.tensor(types.int32, ret_shape)

        # Following logging is to ensure testing of TopK implemented in tf converter
        # default path is testing with appropriate conversion function
        # Log default tf topk
        default_tf_topk = _TF_OPS_REGISTRY.get("TopKV2", None)

        # Override TopK op with override=True flag
        @register_tf_op(tf_alias=["TopKV2"], override=True)
        def CustomTopK(context, node):
            x = context[node.inputs[0]]
            k = context[node.inputs[1]]
            sorted = node.attr.get("sorted", False)
            x = mb.custom_topk(x=x, k=k.val, axis=-1, sorted=sorted, name=node.name)
            context.add(node.name, x)

        yield

        _TF_OPS_REGISTRY["TopKV2"] = default_tf_topk


    @pytest.mark.skipif(not testing_reqs._HAS_TF_1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank, k",
        itertools.product([True], backends, [rank for rank in range(1, 4)], [1, 2],),
    )
    @pytest.mark.usefixtures("create_custom_TopK")
    def test_tf(self, use_cpu_only, backend, rank, k):
        if backend[0] == 'mlprogram':
            pytest.xfail("Custom layer not supported with ML Program backend")

        shape = np.random.randint(low=3, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.math.top_k(x, k=k, sorted=True)
            ref = (ref[1], ref[0])
            mlmodel, _, _, _ = tf_graph_to_mlmodel(
                graph,
                {x: random_gen(shape, rand_min=-100, rand_max=100)},
                ref,
                backend=backend,
            )
            layers = mlmodel.get_spec().neuralNetwork.layers
            assert layers[-1].custom is not None, "Expecting a custom layer"
            assert (
                "TopK" == layers[-1].custom.className
            ), "Custom Layer class name mis-match"
            assert (
                k == layers[-1].custom.parameters["k"].intValue
            ), "Incorrect parameter value k"
            assert (
                True == layers[-1].custom.parameters["sorted"].boolValue
            ), "Incorrect parameter value for Sorted"
