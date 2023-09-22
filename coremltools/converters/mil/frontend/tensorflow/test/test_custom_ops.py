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
from coremltools.converters.mil.mil import Operation, types
from coremltools.converters.mil.mil.input_type import (DefaultInputs,
                                                       InputSpec,
                                                       TensorInputType)
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from coremltools.converters.mil.testing_reqs import backends, compute_units
from coremltools.converters.mil.testing_utils import random_gen

tf = pytest.importorskip("tensorflow")


class TestCustomMatMul:
    # Define SSA Custom Op for Sparse MatMul
    # This will map to `custom_op` in SSA with binding information
    # to bind input spec to the custom implementation
    @register_op(is_custom_op=True)
    class custom_sparse_matmul(Operation):
        # Defining input spec for current op
        input_spec = InputSpec(
            x=TensorInputType(type_domain="T"),
            y=TensorInputType(type_domain="T"),
            transpose_x=TensorInputType(const=True, optional=True, type_domain=types.bool),
            transpose_y=TensorInputType(const=True, optional=True, type_domain=types.bool),
            x_is_sparse=TensorInputType(const=True, optional=True, type_domain=types.bool),
            y_is_sparse=TensorInputType(const=True, optional=True, type_domain=types.bool),
        )
        
        type_domains = {
            "T": (types.fp16, types.fp32),
        }

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

        def type_inference(self):
            x_type = self.x.dtype
            x_shape = self.x.shape
            y_shape = self.y.shape
            # For illustration purpose, assumming getting valid shape
            # Ideally, should consider transpose_?, ?_is_sparse parameters into consideration
            # for computing output shape
            return types.tensor(x_type, [x_shape[0], y_shape[1]])

    # TensorFlow Sparse Matmul Op
    @register_tf_op
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


    @pytest.mark.parametrize(
        "compute_unit, backend, transpose_a, transpose_b," "a_is_sparse, b_is_sparse, b_is_const",
        itertools.product(
            compute_units,
            backends,
            [True, False],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
        ),
    )
    def test_tf(
        self, compute_unit, backend, transpose_a, transpose_b, a_is_sparse, b_is_sparse, b_is_const,
    ):
        if backend[0] == 'mlprogram':
            pytest.skip("Custom layer not supported with ML Program backend")

        rank = 2
        input_shape = list(np.random.randint(low=3, high=100, size=1)) * rank
        if b_is_const:
            @make_tf_graph([input_shape])
            def build_model(x):
                ref = tf.compat.v1.sparse_matmul(
                    x,
                    random_gen(input_shape),
                    transpose_a=transpose_a,
                    transpose_b=transpose_b,
                    a_is_sparse=a_is_sparse,
                    b_is_sparse=b_is_sparse,
                )
                return ref
            input_values = [random_gen(input_shape, -1.0, 1.0)]
        else:
            @make_tf_graph([input_shape, input_shape])
            def build_model(x, y):
                ref = tf.compat.v1.sparse_matmul(
                    x,
                    y,
                    transpose_a=transpose_a,
                    transpose_b=transpose_b,
                    a_is_sparse=a_is_sparse,
                    b_is_sparse=b_is_sparse,
                )
                return ref
            input_values = [random_gen(input_shape, -1.0, 1.0), random_gen(input_shape, -1.0, 1.0)]
        
        model, inputs, outputs = build_model
        input_dict = dict(zip(inputs, input_values))
        spec, _, _, _, _, _ = TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            frontend_only=True,
            backend=backend,
        )
        
        layers = spec.neuralNetwork.layers
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
        @register_op(is_custom_op=True)
        class custom_topk(Operation):
            input_spec = InputSpec(
                x=TensorInputType(type_domain="T"),
                k=TensorInputType(const=True, optional=True, type_domain=types.int32),
                axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
                sorted=TensorInputType(const=True, optional=True, type_domain=types.bool),
            )
            
            type_domains = {
                "T": (types.fp16, types.fp32),
            }
            
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

    @pytest.mark.parametrize(
        "compute_unit, backend, rank, k",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(1, 4)],
            [1, 2],
        ),
    )
    @pytest.mark.usefixtures("create_custom_TopK")
    def test_tf(self, compute_unit, backend, rank, k):
        if backend[0] == 'mlprogram':
            pytest.skip("Custom layer not supported with ML Program backend")

        input_shape = np.random.randint(low=3, high=6, size=rank)
        
        @make_tf_graph([input_shape])
        def build_model(x):
            ref = tf.math.top_k(x, k=k, sorted=True)
            return ref[1], ref[0]
            
        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape, -1.0, 1.0)]
        input_dict = dict(zip(inputs, input_values))
        spec, _, _, _, _, _ = TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            frontend_only=True,
            backend=backend,
        )
        
        layers = spec.neuralNetwork.layers
        assert layers[-1].custom is not None, "Expecting a custom layer"
        assert (
            "TopK" == layers[-1].custom.className
        ), "Custom Layer class name mis-match"
        assert (
            k == layers[-1].custom.parameters["k"].intValue
        ), "Incorrect parameter value k"
        assert (
            layers[-1].custom.parameters["sorted"].boolValue is True
        ), "Incorrect parameter value for Sorted"
