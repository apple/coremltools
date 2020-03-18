import itertools
import unittest

import numpy as np
import pytest
import torch
import torch.nn as nn

from coremltools.converters.nnv2.nnv2_program.program import SsaFunction
from coremltools.converters.nnv2.nnv2_program.program.var import Var

from ..converter import TorchConverter, TranscriptionContext
from ..internal_graph import InternalTorchIRNode

from .. import ops


class TestTorchConversion:
    """Class containing tests for the backend of torch -> CoreML conversion.

    Test Modules, (i.e @ConstantModule) must define an @_input attribute
    as a tensor or iterable of input tensors. This is used to convert the module.

    NOTE: Confused where @context is coming from? Its from the pytest fixture defined below.
    """

    @pytest.fixture
    def context(self):
        return TranscriptionContext()

    @pytest.fixture
    def set_random_seeds(self):
        torch.manual_seed(1)
        np.random.seed(1)

    @pytest.mark.parametrize("dtype", [torch.bool, torch.float, torch.int])
    def test_constant(self, context, dtype):
        test_data = torch.ones(1, dtype=dtype)
        node = InternalTorchIRNode(
            val=test_data, kind="Constant", inputs=[], outputs=["1"]
        )
        with SsaFunction(inputs={}) as ssa_func:
            ops.constant(context, node)
        ssa = context._current_graph["1"]
        assert np.allclose(test_data, ssa.val)
        assert test_data.shape == ssa.shape

    @staticmethod
    def _gen_constants(size, vals):
        """Helper function. Generates a list of internal constant nodes.
        
        Arguments:
            size: number of constants to generate
            vals: Either a list of values for each constant or one value used for all constants."""
        is_list = isinstance(vals, list)
        if is_list:
            if len(vals) != size:
                raise ValueError("len(@vals : {}): != size: {}".format(vals, size))
        constants = []
        for index in range(size):
            if is_list:
                val = vals[index]
            else:
                val = vals
            constants.append(
                InternalTorchIRNode(
                    val=val, kind="Constant", inputs=[], outputs=[str(index)]
                )
            )
        input_list = [str(i) for i in range(size)]
        output_name = str(len(input_list))
        return constants, input_list, output_name

    def test_add(self, context):
        test_input_1 = np.random.rand(2, 3)
        test_input_2 = np.random.rand(2, 3)
        scale_factor = 1
        self._test_elementwise_binary(
            context,
            "Add",
            ops.add,
            [test_input_1, test_input_2, scale_factor],
            3,
            test_input_1 + test_input_2,
        )

    def test_sub(self, context):
        test_input_1 = np.random.rand(3, 2)
        test_input_2 = np.random.rand(3, 2)
        self._test_elementwise_binary(
            context,
            "Sub",
            ops.sub,
            [test_input_1, test_input_2],
            2,
            test_input_1 - test_input_2,
        )

    def test_mul(self, context):
        test_input_1 = np.random.rand(3, 2)
        test_input_2 = np.random.rand(3, 2)
        self._test_elementwise_binary(
            context,
            "Sub",
            ops.mul,
            [test_input_1, test_input_2],
            2,
            test_input_1 * test_input_2,
        )

    def test_div(self, context):
        test_input_1 = np.random.rand(3, 2)
        test_input_2 = np.random.rand(3, 2)
        self._test_elementwise_binary(
            context,
            "Div",
            ops.div,
            [test_input_1, test_input_2],
            2,
            numpy.divide(test_input_1, test_input_2),
        )

    def test_div(self, context):
        test_input_1 = np.random.rand(3, 2)
        test_input_2 = np.random.rand(3, 2)
        self._test_elementwise_binary(
            context,
            "Pow",
            ops.pow,
            [test_input_1, test_input_2],
            2,
            np.power(test_input_1, test_input_2),
        )

    def _test_elementwise_binary(
        self, context, op_name, op, test_input, num_constants, expected_result
    ):
        """Runs op on test input and compares against expected result"""
        constants, input_list, output_name = self._gen_constants(
            num_constants, test_input
        )
        eb_node = InternalTorchIRNode(
            val=None, kind=op_name, inputs=input_list, outputs=[output_name]
        )
        with SsaFunction(inputs={}) as ssa_func:
            for node in constants:
                ops.constant(context, node)
            op(context, eb_node)

        ssa = context._current_graph[output_name]
        assert np.allclose(expected_result, ssa.val)

    @pytest.mark.parametrize("size", [1, 5, 7])
    def test_listconstruct(self, context, size):
        constant_vals = list(np.arange(size))
        constants, input_list, output_name = self._gen_constants(size, constant_vals)
        lc_node = InternalTorchIRNode(
            kind="ListConstruct", inputs=input_list, outputs=[output_name],
        )
        with SsaFunction(inputs={}) as ssa_func:
            for node in constants:
                ops.constant(context, node)
            ops.listconstruct(context, lc_node)

        ssa = context._current_graph[output_name]
        expected_val = np.arange(size)
        assert ssa.shape == (size,)
        assert np.array_equal(ssa.val, expected_val)

    @pytest.mark.parametrize(
        "input_shape, start, end",
        [
            ((2, 1, 1, 2), 1, 3),
            ((2, 2, 1, 1), 1, -2),
            ((1, 1, 1), 0, 2),
            ((1, 2), 0, 1),
            ((1, 2), 1, 1),
            ((1, 1), 1, -1),
            ((1,), 0, 0),
        ],
    )
    def test_flatten(self, context, input_shape, start, end):
        test_data = torch.rand(input_shape)
        constants, input_list, output_name = self._gen_constants(
            3, [test_data, start, end]
        )
        flatten_node = InternalTorchIRNode(
            kind="Flatten", inputs=input_list, outputs=[output_name]
        )
        with SsaFunction(inputs={}) as ssa_func:
            for node in constants:
                ops.constant(context, node)
            ops.flatten(context, flatten_node)

        ssa = context._current_graph[output_name]
        expected_result = torch.flatten(test_data, start, end)
        assert np.allclose(expected_result, ssa.val)

    @pytest.mark.parametrize(
        "start, end", [(0, -5), (100, 2), (2, 100), (-3, -4),],
    )
    def test_flatten_exception(self, context, start, end):
        test_data = torch.rand(1, 1, 1, 1)
        constants, input_list, output_name = self._gen_constants(
            3, [test_data, start, end]
        )
        flatten_node = InternalTorchIRNode(
            kind="Flatten", inputs=input_list, outputs=[output_name]
        )
        with SsaFunction(inputs={}) as ssa_func:
            flag = True
            for node in constants:
                ops.constant(context, node)
            try:
                ops.flatten(context, flatten_node)
            except:
                flag = False
            if flag:
                pytest.fail("Conversion should have thrown an exception.")

    @pytest.mark.parametrize(
        "in_features, out_features, scaling",
        itertools.product([10, 25, 100], [3, 6], [1.0, 0.5]),
    )
    def test_addmm(self, context, in_features, out_features, scaling):
        input_data = torch.rand((1, in_features))
        weight_data = torch.rand((in_features, out_features))
        bias_data = torch.rand((out_features))
        constant_vals = [
            scaling,
            input_data,
            weight_data,
            bias_data,
        ]
        constants, _, output_name = self._gen_constants(4, constant_vals)

        addmm_node = InternalTorchIRNode(
            kind="addmm", inputs=["3", "1", "2", "0", "0"], outputs=["output"],
        )

        with SsaFunction(inputs={}) as ssa_func:
            for node in constants:
                ops.constant(context, node)
            ops.addmm(context, addmm_node)

        ssa = context._current_graph["output"]
        torch_linear = nn.Linear(in_features=in_features, out_features=out_features,)
        expected_shape = tuple(torch_linear(input_data).shape)
        assert expected_shape == ssa.shape

    @pytest.mark.parametrize(
        "height, width, kernel_size, stride, padding, dilation",
        itertools.product([5, 6], [5, 7], [1, 3], [1, 3], [1, 3], [1, 3]),
    )
    def test_convolution2d(
        self,
        context,
        height,
        width,
        kernel_size,
        stride,
        padding,
        dilation,
        groups=1,
        in_channels=1,
        out_channels=2,
    ):
        test_input = torch.rand(1, in_channels, height, width)
        constant_vals = [
            1,
            test_input,
            np.random.rand(
                out_channels, in_channels, kernel_size, kernel_size
            ),  # weights
            np.random.rand(out_channels),  # bias
            np.array([stride, stride]),
            np.array([padding, padding]),
            np.array([dilation, dilation]),
            groups,
        ]
        constants, _, output_name = self._gen_constants(8, constant_vals)
        conv_node = InternalTorchIRNode(
            kind="_convolution",
            inputs=["1", "2", "3", "4", "5", "6", "0", "0", "7", "0", "0", "0"],
            outputs=[output_name],
        )

        with SsaFunction(inputs={}) as ssa_func:
            for node in constants:
                ops.constant(context, node)
            ops._convolution(context, conv_node)

        ssa = context._current_graph[output_name]
        torch_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        expected_shape = tuple(torch_conv(test_input).shape)
        assert expected_shape == ssa.shape

    @pytest.mark.parametrize(
        "input_shape, dim, keepdim",
        itertools.product([(3, 20, 20), (1, 50, 50)], [0, 1, 2, [0, 2]], [True, False]),
    )
    def test_mean(self, context, input_shape, dim, keepdim):
        test_input = torch.rand(*input_shape)

        constants, input_list, output_name = self._gen_constants(
            4, [test_input, dim, keepdim, None]
        )
        mean_node = InternalTorchIRNode(
            kind="mean", inputs=input_list, outputs=[output_name]
        )
        with SsaFunction(inputs={}) as ssa_func:
            for node in constants:
                ops.constant(context, node)
            ops.mean(context, mean_node)

        ssa = context._current_graph[output_name]
        expected_result = torch.mean(test_input, dim, keepdim)
        assert np.allclose(expected_result, ssa.val)
