import itertools

import numpy as np
import pytest
import torch
import torch.nn as nn

from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from coremltools.converters.nnv2.nnv2_program.program import SsaFunction, get_new_symbol
from coremltools.converters.nnv2.nnv2_program.program.var import Var


from .. import ops
from ..converter import TorchConverter, TranscriptionContext
from ..internal_graph import InternalTorchIRNode


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
            val=test_data, kind="constant", inputs=[], outputs=["1"]
        )
        ssa = self._construct_test_graph(context, ops.constant, node, "1")
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
                    val=val, kind="constant", inputs=[], outputs=[str(index)]
                )
            )
        input_list = [str(i) for i in range(size)]
        output_name = str(len(input_list))
        return constants, input_list, output_name

    @staticmethod
    def _construct_test_graph(
        context, test_op, test_node, output_name, graph_inputs=None, constants=None
    ):
        """ Construct an SsaFunction for the given @graph_inputs, @constants,
            and @test_node. Returns the output of the graph, which is the ssa
            Var of the given @output_name.
        """
        if graph_inputs is None:
            graph_inputs = {}
        if constants is None:
            constants = []

        with SsaFunction(inputs=graph_inputs) as ssa_func:
            for name in ssa_func.inputs.keys():
                context.add(ssa_func.inputs[name])
            for node in constants:
                ops.constant(context, node)
            test_op(context, test_node)

        ssa = context[output_name]
        return ssa

    def _test_elementwise_binary(
        self, context, op_name, op, test_input, num_constants, expected_result
    ):
        """Helper function, runs op on test input and compares against expected result"""
        constants, input_list, output_name = self._gen_constants(
            num_constants, test_input
        )
        eb_node = InternalTorchIRNode(
            val=None, kind=op_name, inputs=input_list, outputs=[output_name]
        )
        ssa = self._construct_test_graph(
            context, op, eb_node, output_name, constants=constants
        )
        assert np.allclose(expected_result, ssa.val)

    def _test_cast(self, context, test_val, op_kind, op_func, python_type):
        constants, input_list, output_name = self._gen_constants(1, [test_val])
        node = InternalTorchIRNode(
            kind=op_kind, inputs=input_list, outputs=[output_name]
        )
        ssa = self._construct_test_graph(
            context, op_func, node, output_name, constants=constants
        )
        assert ssa == python_type(test_val)

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
        scale_factor = 1
        self._test_elementwise_binary(
            context,
            "Sub",
            ops.sub,
            [test_input_1, test_input_2, scale_factor],
            3,
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

    def test_eq(self, context):
        test_input_1 = torch.zeros([2, 3, 4, 5, 6]).float()
        test_input_2 = torch.ones([2, 3, 4, 5, 6]).float()
        test_input_2[0][0][0][0][0] = 0
        expected_output = (test_input_1 == test_input_2).float()

        self._test_elementwise_binary(
            context, "Eq", ops.eq, [test_input_1, test_input_2], 2, expected_output
        )

    def test_le(self, context):
        test_input_1 = torch.zeros([2, 3, 4, 5, 6]).float()
        test_input_2 = torch.ones([2, 3, 4, 5, 6]).float()
        test_input_2[0][0][0][0][0] = 0
        expected_output = (test_input_1 <= test_input_2).float()

        self._test_elementwise_binary(
            context, "Le", ops.le, [test_input_1, test_input_2], 2, expected_output
        )

    def test_lt(self, context):
        test_input_1 = torch.zeros([2, 3, 4, 5, 6]).float()
        test_input_2 = torch.ones([2, 3, 4, 5, 6]).float()
        test_input_2[0][0][0][0][0] = 0
        expected_output = (test_input_1 < test_input_2).float()

        self._test_elementwise_binary(
            context, "Lt", ops.lt, [test_input_1, test_input_2], 2, expected_output
        )

    def test_ge(self, context):
        test_input_1 = torch.zeros([2, 3, 4, 5, 6]).float()
        test_input_2 = torch.ones([2, 3, 4, 5, 6]).float()
        test_input_2[0][0][0][0][0] = 0
        expected_output = (test_input_1 >= test_input_2).float()

        self._test_elementwise_binary(
            context, "Ge", ops.ge, [test_input_1, test_input_2], 2, expected_output
        )

    def test_gt(self, context):
        test_input_1 = torch.zeros([2, 3, 4, 5, 6]).float()
        test_input_2 = torch.ones([2, 3, 4, 5, 6]).float()
        test_input_2[0][0][0][0][0] = 0
        expected_output = (test_input_1 > test_input_2).float()

        self._test_elementwise_binary(
            context, "Gt", ops.gt, [test_input_1, test_input_2], 2, expected_output
        )

    @pytest.mark.parametrize(
        "size, array_type",
        itertools.product(
            [1, 5, 7],
            [
                ("ListConstruct", ops.listconstruct),
                ("TupleConstruct", ops.tupleconstruct),
            ],
        ),
    )
    def test_arrayconstruct(self, context, size, array_type):
        constant_vals = list(np.arange(size))
        array_kind = array_type[0]
        array_op = array_type[1]
        constants, input_list, output_name = self._gen_constants(size, constant_vals)
        ac_node = InternalTorchIRNode(
            kind=array_kind, inputs=input_list, outputs=[output_name],
        )
        ssa = self._construct_test_graph(
            context, array_op, ac_node, output_name, constants=constants
        )
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

        ssa = self._construct_test_graph(
            context, ops.matmul, matmul_node, output_name, constants=constants
        )
        expected_result = torch.matmul(mat1, mat2).detach().numpy()
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
        ssa = self._construct_test_graph(
            context, ops.flatten, flatten_node, output_name, constants=constants
        )
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
        with pytest.raises(ValueError):
            self._construct_test_graph(
                context, ops.flatten, flatten_node, output_name, constants=constants,
            )

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

        ssa = self._construct_test_graph(
            context, ops.addmm, addmm_node, output_name, constants=constants
        )
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
            1,  # None argument
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

        ssa = self._construct_test_graph(
            context, ops._convolution, conv_node, output_name, constants=constants
        )
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
        ssa = self._construct_test_graph(
            context, ops.mean, mean_node, output_name, constants=constants
        )
        expected_result = torch.mean(test_input, dim, keepdim)
        assert np.allclose(expected_result, ssa.val)

    def test_mean_no_dims(self, context):
        test_input = torch.rand((3, 20, 20))

        constants, input_list, output_name = self._gen_constants(2, [test_input, None])
        mean_node = InternalTorchIRNode(
            kind="mean", inputs=input_list, outputs=[output_name]
        )
        ssa = self._construct_test_graph(
            context, ops.mean, mean_node, output_name, constants=constants
        )
        expected_result = torch.mean(test_input)
        assert np.allclose(expected_result, ssa.val)

    def test_embedding(self, context):
        EMBEDDING_DIMENSION = 10
        NUM_EMBEDDINGS = 20
        input_shape = (NUM_EMBEDDINGS, EMBEDDING_DIMENSION)
        # size is arbitrary for indices
        indices = np.random.randint(NUM_EMBEDDINGS, size=100)
        test_input = torch.rand(input_shape)
        constants, input_list, output_name = self._gen_constants(
            2, [test_input, indices]
        )
        gather_node = InternalTorchIRNode(
            kind="embedding", inputs=input_list, outputs=[output_name]
        )
        ssa = self._construct_test_graph(
            context, ops.embedding, gather_node, output_name, constants=constants
        )
        torch_embedding = nn.Embedding.from_pretrained(test_input)
        expected_result = torch_embedding(torch.LongTensor(indices))
        assert np.allclose(expected_result, ssa.val)

    @pytest.mark.parametrize(
        "dim", [0, 1, 2, 3, 4],
    )
    def test_size(self, context, dim):
        test_input = torch.rand(1, 2, 3, 4, 5)

        graph_inputs = {"input": cb.placeholder(test_input.shape, dtype=builtins.float)}
        constants, input_list, output_name = self._gen_constants(1, [dim])
        size_node = InternalTorchIRNode(
            kind="size", inputs=["input"] + input_list, outputs=[output_name]
        )
        ssa = self._construct_test_graph(
            context,
            ops.size,
            size_node,
            output_name,
            constants=constants,
            graph_inputs=graph_inputs,
        )
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
        ssa = self._construct_test_graph(
            context, ops.view, view_node, output_name, constants=constants
        )
        expected_result = test_input.view(shape)
        assert np.allclose(expected_result, ssa.val)

    @pytest.mark.parametrize(
        "input_shape, output_shape",
        itertools.product(
            [(1, 3, 15, 15), (1, 1, 2, 2), (1, 3, 10, 10)], [(1, 1), (2, 2), (2, 1)],
        ),
    )
    def test_adaptive_avg_pool2d(self, context, input_shape, output_shape):
        test_input = torch.rand(input_shape)

        constants, input_list, output_name = self._gen_constants(
            2, [test_input, output_shape]
        )

        adaptive_avg_pool2d_node = InternalTorchIRNode(
            kind="adaptive_avg_pool2d", inputs=input_list, outputs=[output_name]
        )
        ssa = self._construct_test_graph(
            context,
            ops.adaptive_avg_pool2d,
            adaptive_avg_pool2d_node,
            output_name,
            constants=constants,
        )
        expected_result = torch._adaptive_avg_pool2d(test_input, output_shape)
        expected_shape = tuple(expected_result.shape)
        assert expected_shape == ssa.shape
        # We only expect numerical output when reducing to global average.
        if output_shape == (1, 1):
            assert np.allclose(expected_result, ssa.val)

    def test_adaptive_avg_pool2d_exception(self, context):
        # For this test, the input tensor HW channels are dynamic.
        input_shape = [1, 3, get_new_symbol(), get_new_symbol()]
        graph_inputs = {"input": cb.placeholder(input_shape, dtype=builtins.float)}
        constants, input_list, output_name = self._gen_constants(1, [(2, 1)])
        adaptive_avg_pool2d_node = InternalTorchIRNode(
            kind="adaptive_avg_pool2d",
            inputs=["input"] + input_list,
            outputs=[output_name],
        )
        with pytest.raises(ValueError):
            ssa = self._construct_test_graph(
                context,
                ops.adaptive_avg_pool2d,
                adaptive_avg_pool2d_node,
                output_name,
                constants=constants,
                graph_inputs=graph_inputs,
            )

    @pytest.mark.parametrize("input_shape", [(1, 3, 15, 15), (1, 1, 1, 1)])
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
        ssa = self._construct_test_graph(
            context, ops.batch_norm, batch_norm_node, output_name, constants=constants
        )
        assert ssa.val == None
        assert ssa.shape == tuple(test_input.shape)

    @pytest.mark.parametrize(
        "dropout_type",
        [("dropout", ops.dropout), ("feature_dropout", ops.feature_dropout)],
    )
    def test_dropout(self, context, dropout_type):
        test_input = torch.rand(3, 4, 5)
        dropout_kind = dropout_type[0]
        dropout_op = dropout_type[1]
        constants, input_list, output_name = self._gen_constants(
            3, [test_input, 0.5, False]
        )
        dropout_node = InternalTorchIRNode(
            kind=dropout_kind, inputs=input_list, outputs=[output_name]
        )
        ssa = self._construct_test_graph(
            context, dropout_op, dropout_node, output_name, constants=constants
        )
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
        ssa = self._construct_test_graph(
            context, ops.hardtanh_, hardtanh_node, output_name, constants=constants
        )
        torch_hardtanh = nn.Hardtanh(min_val, max_val)
        expected_result = torch_hardtanh(test_input).numpy()
        # Tolerance needs to be higher because of float errors with sigmoid representation
        assert np.allclose(expected_result, ssa.val, atol=1e-6)

    @pytest.mark.parametrize("axis", [1, 2, 3])
    def test_cat(self, context, axis):
        input_shape = (1, 3, 240, 320)

        test_input1 = torch.rand(input_shape)
        test_input2 = torch.rand(input_shape)
        const_input = torch.rand(input_shape)

        graph_inputs = {
            "input1": cb.placeholder(input_shape, dtype=builtins.float),
            "input2": cb.placeholder(input_shape, dtype=builtins.float),
        }
        dim_node = InternalTorchIRNode(
            val=axis, kind="constant", inputs=[], outputs=["0"],
        )
        const_tensor_node = InternalTorchIRNode(
            val=const_input.numpy(), kind="constant", inputs=[], outputs=["1"],
        )
        listconstruct_node = InternalTorchIRNode(
            kind="listconstruct", inputs=["1", "input1", "input2"], outputs=["2"]
        )
        cat_node = InternalTorchIRNode(
            kind="cat", inputs=["2", "0"], outputs=["output"]
        )

        with SsaFunction(inputs=graph_inputs) as ssa_func:
            context.add(ssa_func.inputs["input1"])
            context.add(ssa_func.inputs["input2"])
            ops.constant(context, dim_node)
            ops.constant(context, const_tensor_node)
            ops.listconstruct(context, listconstruct_node)
            ops.cat(context, cat_node)

        ssa = context["output"]
        expected_result = torch.cat(
            (const_input, test_input1, test_input2), dim=axis
        ).numpy()
        assert np.allclose(expected_result.shape, ssa.shape)

    def test_item(self, context):
        const_val = 0
        constants, input_list, output_name = self._gen_constants(1, [const_val])
        item_node = InternalTorchIRNode(
            kind="item", inputs=input_list, outputs=[output_name]
        )
        ssa = self._construct_test_graph(
            context, ops.item, item_node, output_name, constants=constants
        )
        assert ssa.val == const_val

    def test_item_exception(self, context):
        const_val = [0]
        constants, input_list, output_name = self._gen_constants(1, [const_val])
        item_node = InternalTorchIRNode(
            kind="item", inputs=input_list, outputs=[output_name]
        )
        with pytest.raises(ValueError):
            ssa = self._construct_test_graph(
                context, ops.item, item_node, output_name, constants=constants,
            )

    @pytest.mark.parametrize("input_shape", [(1, 3, 15, 15), (1, 1, 1, 1)])
    def test_layer_norm(self, context, input_shape):
        graph_inputs = {"input": cb.placeholder(input_shape, dtype=builtins.float)}
        channels = input_shape[1]
        constants, input_list, output_name = self._gen_constants(
            5,
            [
                input_shape,  # normalized shape
                torch.rand(channels),  # weight
                torch.rand(channels),  # running bias
                1e-6,
                1,  # cudnn enabled
            ],
        )

        layer_norm_node = InternalTorchIRNode(
            kind="layer_norm", inputs=["input"] + input_list, outputs=[output_name]
        )
        ssa = self._construct_test_graph(
            context,
            ops.layer_norm,
            layer_norm_node,
            output_name,
            graph_inputs=graph_inputs,
            constants=constants,
        )
        assert ssa.val == None
        assert ssa.shape == input_shape

    @pytest.mark.parametrize(
        "scales_h, scales_w, output_size, align_corners",
        [
            x
            for x in itertools.product(
                [None], [None], [(10, 10), (1, 1), (20, 20)], [1, 0]
            )
        ]
        + [x for x in itertools.product([2, 3], [4, 5], [None], [1, 0])],
    )
    def test_upsample_bilinear2d(
        self, context, scales_h, scales_w, output_size, align_corners
    ):
        # Does not have value inference so we don't need to a separate constant test
        input_shape = (1, 3, 10, 10)
        graph_inputs = {"input": cb.placeholder(input_shape, dtype=builtins.float)}
        constants, constant_input_list, output_name = self._gen_constants(
            4, [output_size, align_corners, scales_h, scales_w]
        )
        upsample_node = InternalTorchIRNode(
            kind="upsample_bilinear2d",
            inputs=["input"] + constant_input_list,
            outputs=[output_name],
        )
        ssa = self._construct_test_graph(
            context,
            ops.upsample_bilinear2d,
            upsample_node,
            output_name,
            graph_inputs=graph_inputs,
            constants=constants,
        )
        if output_size is None:
            expected_shape = list(input_shape)
            expected_shape[-2] *= scales_h
            expected_shape[-1] *= scales_w
        else:
            expected_shape = list(input_shape)
            expected_shape[-2] = output_size[0]
            expected_shape[-1] = output_size[1]

        assert ssa.shape == tuple(expected_shape)

    def test_ones(self, context):
        input_shape = (get_new_symbol(),)
        graph_inputs = {"input": cb.placeholder(input_shape, dtype=builtins.float)}
        constants, constant_input_list, output_name = self._gen_constants(4, 1)
        ones_node = InternalTorchIRNode(
            kind="ones", inputs=["input"] + constant_input_list, outputs=[output_name],
        )
        ssa = self._construct_test_graph(
            context,
            ops.ones,
            ones_node,
            output_name,
            graph_inputs=graph_inputs,
            constants=constants,
        )
        assert ssa.shape == input_shape

    @pytest.mark.parametrize(
        "input_size, dim, index",
        itertools.product([(13, 43, 10), (39, 14, 11, 9)], [0, 1, 2], [0, 1, 3, 8],),
    )
    def test_select(self, context, input_size, dim, index):
        graph_inputs = {"input1": cb.placeholder(input_size, dtype=builtins.float)}
        constants, constant_input_list, output_name = self._gen_constants(
            2, [dim, index]
        )
        select_node = InternalTorchIRNode(
            kind="select",
            inputs=["input1"] + constant_input_list,
            outputs=[output_name],
        )
        ssa = self._construct_test_graph(
            context,
            ops.select,
            select_node,
            output_name,
            graph_inputs=graph_inputs,
            constants=constants,
        )
        expected_shape = tuple(
            torch.rand(input_size)
            .index_select(dim, torch.tensor([index]))
            .squeeze(dim)
            .shape
        )
        assert np.allclose(ssa.shape, expected_shape)

    @pytest.mark.parametrize("dynamic", [True, False])
    def test_tupleunpack(self, context, dynamic):
        """ if @dynamic is True then packs up a dynamic input """
        input_shape = (1, 2, 3)
        constant_vals = [str(i) for i in range(1, 6)]
        constants_unpacked = [str(i) for i in range(6, 11)]
        constants, input_list, _ = self._gen_constants(5, constant_vals)
        output_list = constants_unpacked[:]
        graph_inputs = {}
        if dynamic:
            graph_input_name = "input1"
            graph_inputs = {
                graph_input_name: cb.placeholder(input_shape, dtype=builtins.float)
            }
            input_list += [graph_input_name]
            output_list += [graph_input_name + "_out"]

        tupleconstruct_node = InternalTorchIRNode(
            kind="TupleConstruct", inputs=input_list, outputs=["construct"],
        )
        tupleunpack_node = InternalTorchIRNode(
            kind="TupleUnpack", inputs=["construct"], outputs=output_list
        )
        with SsaFunction(inputs=graph_inputs) as ssa_func:
            if dynamic:
                context.add(ssa_func.inputs["input1"])
            for node in constants:
                ops.constant(context, node)
            ops.tupleconstruct(context, tupleconstruct_node)
            ops.tupleunpack(context, tupleunpack_node)

        ssa_constants = []
        for name in constants_unpacked:
            ssa_constants.append(context[name].val)
        assert ssa_constants == constant_vals

        if dynamic:
            ssa_dyanmic = context[graph_input_name + "_out"]
            assert ssa_dyanmic.val is None
            assert ssa_dyanmic.shape == input_shape
