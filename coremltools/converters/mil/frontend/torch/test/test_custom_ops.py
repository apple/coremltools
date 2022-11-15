#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch
import torch.nn as nn

from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.ops import \
    cosine_similarity as cosine_similarity_main
from coremltools.converters.mil.frontend.torch.torch_op_registry import \
    _TORCH_OPS_REGISTRY as _TORCH_OPS_REG
from coremltools.converters.mil.frontend.torch.torch_op_registry import \
    register_torch_op
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, types
from coremltools.converters.mil.mil.input_type import (DefaultInputs,
                                                       InputSpec,
                                                       TensorInputType)
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op

from .testing_utils import TorchBaseTest, convert_to_mlmodel

# Custom layer imports



# Log Converter supported Cosine Similarity conversion function
default_cosine_similarity = _TORCH_OPS_REG.get("cosine_similarity", None)


@register_torch_op(override=True)
def cosine_similarity(context, node):
    cosine_similarity_main(context, node)


# Log custom Cosine Similarity conversion function
custom_cosine_similarity = _TORCH_OPS_REG["cosine_similarity"]


def _set_torch_reg_op(op_type, op_func):
    _TORCH_OPS_REG[op_type] = op_func


class TestCompositeOp(TorchBaseTest):

    @pytest.mark.parametrize("input_shape", [(100, 180), (56, 123)])
    def test_composite_op(self, input_shape):
        _set_torch_reg_op("cosine_similarity", custom_cosine_similarity)
        model = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.run_compare_torch([input_shape, input_shape], model)
        _set_torch_reg_op("cosine_similarity", default_cosine_similarity)


class TestCustomOp:
    # Define SSA Custom Op for Sparse MatMul
    # This will map to `custom_op` in SSA with binding information
    # to bind input spec to the custom implementation
    @register_op(is_custom_op=True)
    class custom_torch_sparse_matmul(Operation):
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

        def default_inputs(self):
            return DefaultInputs(
                transpose_x=False,
                transpose_y=False,
                x_is_sparse=False,
                y_is_sparse=False,
                )

        # Specifying binding for custom op for specifying inputs,
        # parameters required for creating custom op to be synced with Swift API
        bindings = {
            "class_name": "SparseMatMul",
            "input_order": ["x", "y"],
            "parameters": ["transpose_x", "transpose_y", "x_is_sparse", "y_is_sparse"],
            "description": "Custom Sparse MatMul Layer",
        }

        def type_inference(self):
            x_type = self.x.dtype
            x_shape = self.x.shape
            y_shape = self.y.shape
            # For illustration purpose, assumming getting valid shape
            # Ideally, should consider transpose_?, ?_is_sparse parameters into consideration
            # for computing output shape
            return types.tensor(x_type, [x_shape[0], y_shape[1]])

    @register_torch_op()
    def _sparse_mm(context, node):
        inputs = _get_inputs(context, node, expected=2)
        x = mb.custom_torch_sparse_matmul(
            x=inputs[0], y=inputs[1], x_is_sparse=True, y_is_sparse=True, name=node.name
        )
        context.add(x)

    def test_custom_sparse_mm_op(self, input_shape=(4, 4)):
        class TestLayer(nn.Module):
            def __init__(self):
                super(TestLayer, self).__init__()

            def forward(self, x, y):
                x = torch.sparse.mm(x, y)
                return x

        model = TestLayer()
        input_data_x = torch.ones(input_shape)
        input_data_y = torch.ones(input_shape)
        input_data = [input_data_x, input_data_y]
        model.eval()
        torch_model = torch.jit.trace(model, (input_data_x, input_data_y))
        mlmodel = convert_to_mlmodel(torch_model, input_data)

        layers = mlmodel.get_spec().neuralNetwork.layers
        assert layers[-1].custom is not None, "Expecting a custom layer"
        assert (
            "SparseMatMul" == layers[-1].custom.className
        ), "Custom Layer class name mis-match"
        assert (
            not layers[-1].custom.parameters["transpose_x"].boolValue
        ), "Incorrect parameter value k"
        assert (
            not layers[-1].custom.parameters["transpose_y"].boolValue
        ), "Incorrect parameter value k"
        assert (
            layers[-1].custom.parameters["x_is_sparse"].boolValue
        ), "Incorrect parameter value k"
        assert (
            layers[-1].custom.parameters["y_is_sparse"].boolValue
        ), "Incorrect parameter value k"
