import itertools

import numpy as np
import pytest
import torch
import torch.nn as nn

from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from coremltools.converters.nnv2.nnv2_program.program import SsaFunction
from coremltools.converters.nnv2.nnv2_program.program.var import Var

from ..converter import TorchConverter, TranscriptionContext
from ..internal_graph import InternalTorchIRNode

from .. import ops


class TestTorchOps:
    """Class containing tests for converting TorchIR -> CoreML ops.

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
                raise ValueError("len(@vals): {}): != size: {}".format(len(vals), size))
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
            np.divide(test_input_1, test_input_2),
        )

    def test_pow(self, context):
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
        "dim1, dim2, dim3", itertools.product([1, 2, 5], [2, 5, 10], [1, 2, 5]),
    )
    def test_matmul(self, context, dim1, dim2, dim3):
        mat1 = torch.rand((dim1, dim2))
        mat2 = torch.rand((dim2, dim3))
        constant_vals = [
            mat1,
            mat2,
        ]
        constants, input_list, output_name = self._gen_constants(2, constant_vals)

        matmul_node = InternalTorchIRNode(
            kind="matmul", inputs=input_list, outputs=[output_name],
        )

        with SsaFunction(inputs={}) as ssa_func:
            for node in constants:
                ops.constant(context, node)
            ops.matmul(context, matmul_node)

        expected_result = torch.matmul(mat1, mat2).detach().numpy()
        ssa = context._current_graph[output_name]
        assert np.allclose(expected_result, ssa.val)

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
            kind="addmm", inputs=["3", "1", "2", "0", "0"], outputs=[output_name],
        )

        with SsaFunction(inputs={}) as ssa_func:
            for node in constants:
                ops.constant(context, node)
            ops.addmm(context, addmm_node)

        ssa = context._current_graph[output_name]
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
            1, # None argument
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
        assert ssa.val == None
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

    @pytest.mark.parametrize(
        "dim", [0, 1, 2, 3, 4],
    )
    def test_size(self, context, dim):
        test_input = torch.rand(1, 2, 3, 4, 5)

        dim_node = InternalTorchIRNode(
            val=dim, kind="Constant", inputs=[], outputs=["0"],
        )
        size_node = InternalTorchIRNode(
            kind="size", inputs=["input", "0"], outputs=["output"]
        )
        graph_inputs = {"input": cb.placeholder(test_input.shape, dtype=builtins.float)}

        with SsaFunction(inputs=graph_inputs) as ssa_func:
            context.add(ssa_func.inputs["input"])
            ops.constant(context, dim_node)
            ops.size(context, size_node)

        ssa = context._current_graph["output"]
        expected_result = test_input.shape[dim]
        assert expected_result == ssa.val

    @pytest.mark.parametrize(
        "input_size, shape",
        itertools.product([(5, 12), (1, 4, 15), (3, 5, 4)], [(3, 20), (-1, 6), (60,)],),
    )
    def test_view(self, context, input_size, shape):
        test_input = torch.rand(input_size)

        constants, input_list, output_name = self._gen_constants(2, [test_input, shape])
        view_node = InternalTorchIRNode(
            kind="view", inputs=input_list, outputs=[output_name]
        )
        with SsaFunction(inputs={}) as ssa_func:
            for node in constants:
                ops.constant(context, node)
            ops.view(context, view_node)

        ssa = context._current_graph[output_name]
        expected_result = test_input.view(shape)
        assert np.allclose(expected_result, ssa.val)

    @pytest.mark.parametrize(
        "input_shape", [(3, 15, 15), (1, 1, 1)],
    )
    def test_adaptive_avg_pool2d(self, context, input_shape):
        test_input = torch.rand(input_shape)

        constants, input_list, output_name = self._gen_constants(
            2, [test_input, (1, 1)]
        )

        adaptive_avg_pool2d_node = InternalTorchIRNode(
            kind="adaptive_avg_pool2d", inputs=input_list, outputs=[output_name]
        )
        with SsaFunction(inputs={}) as ssa_func:
            for node in constants:
                ops.constant(context, node)
            ops.adaptive_avg_pool2d(context, adaptive_avg_pool2d_node)

        ssa = context._current_graph[output_name]
        expected_result = torch._adaptive_avg_pool2d(test_input, (1, 1))
        assert np.allclose(expected_result, ssa.val)

    def test_adaptive_avg_pool2d_exception(self, context):
        test_input = torch.rand(3, 2, 2)

        constants, input_list, output_name = self._gen_constants(
            2, [test_input, (2, 1)]
        )
        adaptive_avg_pool2d_node = InternalTorchIRNode(
            kind="adaptive_avg_pool2d", inputs=input_list, outputs=[output_name]
        )
        with pytest.raises(Exception):
            with SsaFunction(inputs={}) as ssa_func:
                for node in constants:
                    ops.constant(context, node)
                ops.adaptive_avg_pool2d(context, adaptive_avg_pool2d_node)

    @pytest.mark.parametrize(
        "input_shape", [(1, 3, 15, 15), (1, 1, 1, 1)],
    )
    def test_batch_norm(self, context, input_shape):
        test_input = torch.rand(input_shape)
        channels = input_shape[1]
        constants, input_list, output_name = self._gen_constants(
            9,
            [
                torch.rand(input_shape),  # input
                torch.rand(channels),  # weight
                torch.rand(channels),  # bias
                torch.rand(channels),  # running mean
                torch.rand(channels),  # running var
                0,  # training
                0.1,  # momentum
                1e-6,  # eps
                1,  # cudnn_enabled
            ],
        )

        batch_norm_node = InternalTorchIRNode(
            kind="batch_norm", inputs=input_list, outputs=[output_name]
        )
        with SsaFunction(inputs={}) as ssa_func:
            for node in constants:
                ops.constant(context, node)
            ops.batch_norm(context, batch_norm_node)

        ssa = context._current_graph[output_name]
        assert ssa.val == None
        assert ssa.shape == tuple(test_input.shape)

    def test_dropout(self, context):
        test_input = torch.rand(3, 4, 5)
        constants, input_list, output_name = self._gen_constants(
            3, [test_input, 0.5, False]
        )
        dropout_node = InternalTorchIRNode(
            kind="dropout", inputs=input_list, outputs=[output_name]
        )
        with SsaFunction(inputs={}) as ssa_func:
            for node in constants:
                ops.constant(context, node)
            ops.dropout(context, dropout_node)
        ssa = context._current_graph[output_name]
        assert np.allclose(test_input.numpy(), ssa.val)

    @pytest.mark.parametrize(
        "min_val, max_val", [(-1.0, 1.0), (0, 0.1), (1.0, 3.0), (-1.0, 6.0),]
    )
    def test_hardtanh(self, context, min_val, max_val):
        test_input = torch.rand(3, 4, 5)
        constants, input_list, output_name = self._gen_constants(
            3, [test_input, min_val, max_val]
        )
        hardtanh_node = InternalTorchIRNode(
            kind="hardtanh_", inputs=input_list, outputs=[output_name]
        )
        with SsaFunction(inputs={}) as ssa_func:
            for node in constants:
                ops.constant(context, node)
            ops.hardtanh_(context, hardtanh_node)

        ssa = context._current_graph[output_name]
        torch_hardtan = nn.Hardtanh(min_val, max_val)
        expected_result = torch_hardtan(test_input).numpy()
        # Tolerance needs to be higher because of float errors with sigmoid representation
        assert np.allclose(expected_result, ssa.val, atol=1e-6)
