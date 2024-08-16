#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
from contextlib import nullcontext
from typing import Optional

import numpy as np
import pytest
import torch
import torchvision

import coremltools as ct
import coremltools.optimize as cto
from coremltools import TensorType
from coremltools._deps import (
    _HAS_TORCH,
    _HAS_TORCH_VISION,
    MSG_TORCH_NOT_FOUND,
    MSG_TORCH_VISION_NOT_FOUND,
)
from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.testing_utils import get_op_types_in_program
from coremltools.optimize.coreml import _quantization_passes
from coremltools.test.ml_program.test_compression import get_test_model_and_data
from coremltools.test.optimize.coreml.test_post_training_quantization import (
    create_quantize_friendly_weight,
    create_sparse_weight,
    create_unique_weight,
)

from .testing_utils import TorchBaseTest

pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)

torch.manual_seed(30)
np.random.seed(30)
torch.backends.quantized.engine = "qnnpack"
compute_units = testing_reqs.compute_units


def _force_quantize_model(
    model: torch.nn.Module,
    q_dtype: torch.dtype,
    low: Optional[int] = None,
    high: Optional[int] = None,
    scale: Optional[float] = None,
    zero_point: Optional[int] = None,
    channel_axis: Optional[int] = None,
):
    """
    In torch, the quantized model can only be obtained from PTQ.
    This utility allows us to produce an int8 quantized model.

    If channel_axis is set, it will do per-channel quantization instead of per-tensor, for the param
    that channel_axis is valid for.
    """
    if scale is None:
        scale = 1.0
    if zero_point is None:
        zero_point = 0

    # modify the parameter to force the quantization within a specific range.
    with torch.no_grad():
        for name, param in model.named_parameters():
            shape = param.shape
            input_data = (
                torch.rand(*shape) if low is None else torch.randint(low, high, shape).float()
            )
            input_data = (input_data - zero_point) * scale

            if channel_axis is not None and -len(shape) <= channel_axis < len(shape):
                scale = torch.Tensor([scale] * shape[channel_axis])
                zero_point = torch.Tensor([zero_point] * shape[channel_axis])
                new_value = torch.quantize_per_channel(
                    input_data,
                    scales=scale,
                    zero_points=zero_point,
                    axis=channel_axis,
                    dtype=q_dtype,
                )
            else:
                new_value = torch.quantize_per_tensor(
                    input_data, scale=scale, zero_point=zero_point, dtype=q_dtype
                )

            param_cls = type(param)
            new_value = param_cls(new_value, requires_grad=False).to(torch.device("cpu"))
            model._parameters[name] = new_value
    return model


class TorchQuantizationBaseTest(TorchBaseTest):
    @staticmethod
    def run_compare_torch(
        input_data,
        model,
        atol=1e-04,
        rtol=1e-05,
        input_as_shape=True,
        minimum_deployment_target=ct.target.iOS17,
        compute_unit=ct.ComputeUnit.CPU_ONLY,
        converter=ct.convert,
    ):
        # TODO(rdar://108472419): properly design a random input
        if input_as_shape:
            input_data = [torch.ones(*shape) for shape in input_data]

        return TorchBaseTest.run_compare_torch(
            input_data,
            model,
            atol=atol,
            rtol=rtol,
            input_as_shape=False,
            backend=("mlprogram", "fp32"),
            use_scripting=False,
            compute_unit=compute_unit,
            minimum_deployment_target=minimum_deployment_target,
            converter=converter,
        )


# TODO(rdar://107430678): test stand-alone quantize and dequantize when cast is ready
class TestPyTorchQuantizationOps(TorchQuantizationBaseTest):
    @pytest.mark.parametrize(
        "quant_dtype, input_rank, is_zp_present, zp_dtype, are_params_tensors",
        itertools.product(
            (torch.qint8, torch.quint8, torch.qint32),
            (1, 3, 5),
            (True, False),
            (np.int8, np.uint8, np.int32),
            (True, False),
        ),
    )
    def test_quantize_dequantize_per_tensor(
        self,
        quant_dtype,
        input_rank,
        is_zp_present,
        zp_dtype,
        are_params_tensors,
    ):
        input_shape = [*np.random.randint(low=1, high=5, size=(input_rank,))]
        scale = np.random.rand()
        zero_point = 0
        if is_zp_present:
            low = 0 if quant_dtype == torch.quint8 or zp_dtype == np.uint8 else -128
            high = 128 if quant_dtype == torch.qint8 or zp_dtype == np.int8 else 256
            zero_point = np.random.randint(low, high, dtype=zp_dtype)
        if are_params_tensors:
            scale = torch.tensor([scale])
            zero_point = torch.tensor([zero_point])

        class Model(torch.nn.Module):
            def forward(self, x):
                quantized = torch.quantize_per_tensor(x, scale, zero_point, quant_dtype)
                dequantized = torch.dequantize(quantized)
                return dequantized

        model = Model()
        if quant_dtype == torch.qint32:
            with pytest.raises(
                ValueError,
                match=r"MIL quantization dtype must be int8 or uint8",
            ):
                self.run_compare_torch([input_shape], model)
        else:
            self.run_compare_torch([input_shape], model, atol=5e-4, rtol=5e-4)

    @pytest.mark.parametrize(
        "quant_dtype, input_rank, is_zp_present, zp_dtype",
        itertools.product(
            (torch.qint8, torch.quint8, torch.qint32),
            (1, 4, 5),
            (True, False),
            (torch.int8, torch.uint8, torch.int32),
        ),
    )
    def test_quantize_dequantize_per_channel(
        self, quant_dtype, input_rank, is_zp_present, zp_dtype
    ):
        input_shape = [*np.random.randint(low=1, high=5, size=(input_rank,))]
        axis = np.random.randint(low=0, high=input_rank)
        scale = torch.rand(input_shape[axis])
        zero_point = torch.zeros(input_shape[axis], dtype=zp_dtype)
        if is_zp_present:
            low = 0 if quant_dtype == torch.quint8 or zp_dtype == torch.uint8 else -128
            high = 128 if quant_dtype == torch.qint8 or zp_dtype == torch.int8 else 256
            zero_point = torch.randint(low, high, (input_shape[axis],), dtype=zp_dtype)

        class Model(torch.nn.Module):
            def forward(self, x):
                quantized = torch.quantize_per_channel(x, scale, zero_point, axis, quant_dtype)
                dequantized = torch.dequantize(quantized)
                return dequantized

        model = Model()
        if quant_dtype == torch.qint32:
            with pytest.raises(
                ValueError,
                match=r"MIL quantization dtype must be int8 or uint8",
            ):
                self.run_compare_torch([input_shape], model)
        else:
            self.run_compare_torch([input_shape], model, atol=5e-4, rtol=5e-4)


# TODO(rdar://108463675): refactor torch op tests later to parametrize quantized vs standard ops
class TestPytorchQuantizedOps(TorchQuantizationBaseTest):
    # PyTorch quantized_linear kernel only supports rank >= 2
    @pytest.mark.parametrize(
        "use_relu, input_rank, quant_dtype",
        itertools.product([True, False], [2, 3, 4], [torch.quint8, torch.qint8]),
    )
    def test_quantized_linear(self, use_relu, input_rank, quant_dtype):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if use_relu:
                    linear = torch.nn.intrinsic.quantized.LinearReLU
                else:
                    linear = torch.nn.quantized.Linear
                self.quant_linear = linear(5, 4)

            def forward(self, x):
                x = torch.quantize_per_tensor(x, scale=1.0, zero_point=0, dtype=quant_dtype)
                x = self.quant_linear(x)
                return torch.dequantize(x)

        model = Model()

        if input_rank == 2:
            input_shape = (3, 5)
        elif input_rank == 3:
            input_shape = (1, 3, 5)
        elif input_rank == 4:
            input_shape = (1, 2, 3, 5)
        self.run_compare_torch([input_shape], model)

    @pytest.mark.parametrize(
        ",".join(
            [
                "use_relu",
                "quant_dtype",
                "padding",
                "stride",
                "height",
                "width",
                "in_channels",
                "out_channels",
                "kernel_size",
                "dilation",
                "bias",
            ]
        ),
        [
            (use_relu, quant_dtype, padding, stride, *param)
            for use_relu, quant_dtype, padding, stride, param in itertools.product(
                [True, False],
                [torch.quint8, torch.qint8],
                [1, 0],
                [1, 2, 3],
                [
                    (5, 3, 1, 1, 1, 1, True),
                    (3, 3, 1, 1, 1, 3, False),
                    (4, 3, 3, 3, 2, 1, True),
                    (7, 3, 3, 3, 1, 1, False),
                ],
            )
        ],
    )
    def test_quantized_conv2d(
        self,
        use_relu,
        quant_dtype,
        padding,
        stride,
        height,
        width,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        bias,
    ):
        if padding == "same" and stride != 1:
            return

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if use_relu:
                    conv = torch.nn.intrinsic.quantized.ConvReLU2d
                else:
                    conv = torch.nn.quantized.Conv2d
                self.quant_conv = conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                    dtype=quant_dtype,
                )

            def forward(self, x):
                x = torch.quantize_per_tensor(x, scale=1.0, zero_point=0, dtype=quant_dtype)
                x = self.quant_conv(x)
                return torch.dequantize(x)

        model = Model()

        self.run_compare_torch(
            [(1, in_channels, height, width)],
            model,
        )

    @pytest.mark.parametrize(
        "input_dtype",
        (np.int32, np.float32),
    )
    def test_quantized_embedding(self, input_dtype):
        pytest.xfail("rdar://106152706 gather: Required param 'validate_indices' is missing")

        num_embeddings = 4
        embedding_size = 10
        B = 2
        dim = 5
        converter_input_type = TensorType(shape=(B, dim), dtype=input_dtype)

        # input shape: (B, dim)
        # output shape : (B, dim, embedding_size)
        # shape of weights : (num_embeddings, embedding_size)
        class EmbeddingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.quantized.Embedding(num_embeddings, embedding_size)

            def forward(self, x):
                return self.embedding(x)

        input_data = np.random.randint(low=0, high=num_embeddings, size=(B, dim))
        input_data = torch.from_numpy(input_data)
        model = EmbeddingModel()
        self.run_compare_torch(
            [input_data], model, input_as_shape=False, converter_input_type=converter_input_type
        )

    # Tests for add, add_relu, mul
    # See: https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.QFunctional.html
    @pytest.mark.parametrize(
        "quant_dtype, qfunc_name",
        itertools.product(
            [torch.quint8, torch.qint8],
            ["add", "add_relu", "mul"],
        ),
    )
    def test_qfunc_binary_ops(self, quant_dtype, qfunc_name):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.qfunc = torch.nn.quantized.QFunctional()

            def forward(self, x):
                x = torch.quantize_per_tensor(x, scale=1.0, zero_point=0, dtype=quant_dtype)
                x = getattr(self.qfunc, qfunc_name)(x, x)
                return torch.dequantize(x)

        model = Model()
        self.run_compare_torch([(2, 3)], model)

    @pytest.mark.xfail(
        reason="torch.ops.quantized.matmul is not supporting mixed precision computation.",
        strict=True,
    )
    @pytest.mark.parametrize(
        "quant_dtype",
        [torch.quint8, torch.qint8],
    )
    def test_quantized_matmul(self, quant_dtype):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.parameter.Parameter(torch.rand(5, 4))

            def forward(self, x):
                return torch.ops.quantized.matmul(x, self.weight, 0, 0)

        model = Model()
        model = _force_quantize_model(model, q_dtype=quant_dtype)
        input_shape = [(3, 5)]
        self.run_compare_torch(input_shape, model)

    @pytest.mark.parametrize(
        "compute_unit, quant_dtype, channel_axis, minimum_deployment_target",
        itertools.product(
            compute_units,
            [torch.quint8, torch.qint8],
            [0, 1, None],
            [ct.target.iOS16, ct.target.iOS17, ct.target.iOS18],
        ),
    )
    def test_quantized_params(
        self, compute_unit, quant_dtype, channel_axis, minimum_deployment_target
    ):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.parameter.Parameter(torch.rand(5, 4))

            def forward(self, x):
                dequanitized_weight = torch.dequantize(self.weight)
                return torch.matmul(x, dequanitized_weight)

        model = Model()
        model = _force_quantize_model(model, q_dtype=quant_dtype, channel_axis=channel_axis)
        input_shape = [(3, 5)]
        res = self.run_compare_torch(
            input_shape,
            model,
            minimum_deployment_target=minimum_deployment_target,
            compute_unit=compute_unit,
        )
        prog = res[1]._mil_program
        if minimum_deployment_target < ct.target.iOS18:
            assert get_op_types_in_program(prog) == ["constexpr_affine_dequantize", "linear"]
        else:
            assert get_op_types_in_program(prog) == ["constexpr_blockwise_shift_scale", "matmul"]


@pytest.mark.skipif(not _HAS_TORCH_VISION, reason=MSG_TORCH_VISION_NOT_FOUND)
class TestTorchvisionQuantizedModels(TorchQuantizationBaseTest):
    # TODO (rdar://107444188): add other torchvision quantized models
    # As of torchvision 0.13.1, there are 5 quantized models:
    #     googlenet, inception, mobilenet, resnet, shufflenet
    # Unfortunately, only mobilenet is working. Others would have
    #     RuntimeError: Quantized backend not supported
    # Presumably because they need `fbgemm`, which does not support macOS
    # We should add them to our end-to-end test once torchvision fix their macOS

    def test_quantized_mobilenetv2(self):
        model = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)
        self.run_compare_torch([(1, 3, 224, 224)], model, atol=1.0)


class TestPytorchCarryCompressionInfo(TorchQuantizationBaseTest):
    """Test compressed PyTorch models which use register_buffer to carry compression info."""

    @pytest.mark.parametrize(
        "compute_unit, n_bits, signed, minimum_deployment_target",
        itertools.product(
            compute_units,
            [4, 8],
            [True, False],
            [ct.target.iOS16, ct.target.iOS18],
        ),
    )
    def test_quantization(self, compute_unit, n_bits, signed, minimum_deployment_target):
        if n_bits == 4 and minimum_deployment_target < ct.target.iOS18:
            pytest.skip("Sub-byte quantization is only supported since iOS18.")

        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data(
            quantize_config=cto.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=types.get_nbits_int_builtin_type(n_bits, signed),
                granularity="per_tensor",
            )
        )

        scale = np.array([2.0], dtype=np.float32).reshape(1, 1, 1, 1)
        zero_point = np.array(
            [0 if signed else 2 ** (n_bits - 1)], dtype=np.int8 if signed else np.uint8
        ).reshape(1, 1, 1, 1)

        model.register_buffer("_COREML_/metadata_version", torch.tensor(2))
        model.register_buffer("_COREML_/weight/compression_type", torch.tensor([3]))
        model.register_buffer("_COREML_/weight/quantization_n_bits", torch.tensor(n_bits))
        model.register_buffer("_COREML_/weight/quantization_scale", torch.from_numpy(scale))
        model.register_buffer("_COREML_/weight/zero_point", torch.from_numpy(zero_point))

        traced_model = torch.jit.trace(model, torch_input_values)
        input_shape = [input.shape.to_list() for input in inputs]
        res = self.run_compare_torch(
            input_shape,
            traced_model,
            minimum_deployment_target=minimum_deployment_target,
            compute_unit=compute_unit,
            converter=ct.convert,
            rtol=1e-04,
            atol=1e-03,
        )
        main_func = res[1]._mil_program.functions["main"]

        target_dtype_str = ("int" if signed else "uint") + str(n_bits)
        if minimum_deployment_target >= ct.target.iOS18:
            quantize_ops = main_func.find_ops(op_type="constexpr_blockwise_shift_scale")
            assert len(quantize_ops) > 0
            for quantize_op in quantize_ops:
                assert types.builtin_to_string(quantize_op.data.dtype) == target_dtype_str
                if not signed:
                    assert types.builtin_to_string(quantize_op.offset.dtype) == target_dtype_str
        else:
            quantize_ops = main_func.find_ops(op_type="constexpr_affine_dequantize")
            assert len(quantize_ops) > 0
            for quantize_op in quantize_ops:
                assert types.builtin_to_string(quantize_op.quantized_data.dtype) == target_dtype_str
                assert types.builtin_to_string(quantize_op.zero_point.dtype) == target_dtype_str

    @pytest.mark.parametrize(
        "compute_unit, n_bits, minimum_deployment_target",
        itertools.product(compute_units, [4, 8], [ct.target.iOS16, ct.target.iOS18]),
    )
    def test_multiple_parameters_in_same_layer(
        self, compute_unit, n_bits, minimum_deployment_target
    ):
        """Test one layer has multiple parameters (such as weight and bias in a linear layer)"""
        if n_bits == 4 and minimum_deployment_target < ct.target.iOS18:
            pytest.skip("Sub-byte quantization is only supported since iOS18.")

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.linear_1 = torch.nn.Linear(16, 32)
                self.linear_2 = torch.nn.Linear(32, 64)

            def forward(self, x):
                return self.linear_2(self.linear_1(x))

        model = Model().eval()
        with torch.no_grad():
            fake_weight_scale = 2 if n_bits == 4 else 40
            model.linear_2.weight = torch.nn.Parameter(
                torch.from_numpy(
                    np.ones_like(model.linear_2.weight.detach().numpy()) * fake_weight_scale
                ).float()
            )
            model.linear_2.bias = torch.nn.Parameter(
                torch.from_numpy(
                    np.ones_like(model.linear_2.bias.detach().numpy()) * fake_weight_scale
                ).float()
            )

        # Register buffers for both weight and bias for linear_2 layer.
        weight_scale = np.array([2.0], dtype=np.float32).reshape(1, 1)
        bias_scale = np.array([2.0], dtype=np.float32)
        model.linear_2.register_buffer("_COREML_/weight/compression_type", torch.tensor([3]))
        model.linear_2.register_buffer("_COREML_/weight/quantization_n_bits", torch.tensor(n_bits))
        model.linear_2.register_buffer(
            "_COREML_/weight/quantization_scale", torch.from_numpy(weight_scale)
        )
        model.linear_2.register_buffer("_COREML_/bias/compression_type", torch.tensor([3]))
        model.linear_2.register_buffer("_COREML_/bias/quantization_n_bits", torch.tensor(n_bits))
        model.linear_2.register_buffer(
            "_COREML_/bias/quantization_scale", torch.from_numpy(bias_scale)
        )
        model.register_buffer("_COREML_/metadata_version", torch.tensor(2))

        torch_input_values = torch.rand((8, 16))
        traced_model = torch.jit.trace(model, torch_input_values)
        res = self.run_compare_torch(
            [(8, 16)],
            traced_model,
            minimum_deployment_target=minimum_deployment_target,
            compute_unit=compute_unit,
            converter=ct.convert,
        )
        main_func = res[1]._mil_program.functions["main"]

        quantize_op_type = (
            "constexpr_blockwise_shift_scale"
            if minimum_deployment_target >= ct.target.iOS18
            else "constexpr_affine_dequantize"
        )
        # Only the linear_2 layer got quantized based on registered buffers.
        linear_ops = main_func.find_ops(op_type="linear")
        assert linear_ops[0].weight.op.op_type == "const"
        assert linear_ops[0].bias.op.op_type == "const"
        assert linear_ops[1].weight.op.op_type == quantize_op_type
        assert linear_ops[1].bias.op.op_type == quantize_op_type

        quantize_ops = main_func.find_ops(op_type=quantize_op_type)
        assert len(quantize_ops) == 2
        for quantize_op in quantize_ops:
            if minimum_deployment_target >= ct.target.iOS18:
                assert types.builtin_to_string(quantize_op.data.dtype) == f"uint{n_bits}"
            else:
                assert types.builtin_to_string(quantize_op.quantized_data.dtype) == f"uint{n_bits}"

    def test_invalid_compression_info(self):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()

        # Invalid key combination (didn't specify compression schema)
        model.register_buffer("_COREML_/weight/quantization_n_bits", torch.tensor(4))
        with pytest.raises(
            ValueError,
            match="There are coreml compression related buffers registered in the torch .* but "
            "the 'compression_type' is not set",
        ):
            self.run_compare_torch(
                [input.shape.to_list() for input in inputs],
                torch.jit.trace(model, torch_input_values),
                minimum_deployment_target=ct.target.iOS18,
                compute_unit=ct.ComputeUnit.CPU_ONLY,
                converter=ct.convert,
            )

        # Invalid key names.
        model.register_buffer("_COREML_/weight/invalid_key", torch.tensor(4))
        with pytest.raises(AttributeError, match="has no attribute 'invalid_key'"):
            self.run_compare_torch(
                [input.shape.to_list() for input in inputs],
                torch.jit.trace(model, torch_input_values),
                minimum_deployment_target=ct.target.iOS18,
                compute_unit=ct.ComputeUnit.CPU_ONLY,
                converter=ct.convert,
            )

        # The lut must be specified for palettization.
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data()
        model.register_buffer("_COREML_/weight/compression_type", torch.tensor([2]))
        with pytest.raises(
            ValueError, match="Missing lut in compression info. Please register a buffer for lut."
        ):
            self.run_compare_torch(
                [input.shape.to_list() for input in inputs],
                torch.jit.trace(model, torch_input_values),
                minimum_deployment_target=ct.target.iOS18,
                compute_unit=ct.ComputeUnit.CPU_ONLY,
                converter=ct.convert,
            )

    @pytest.mark.parametrize(
        "compute_unit, n_bits, group_size, channel_axis, cluster_dim, minimum_deployment_target",
        itertools.product(
            compute_units,
            [4, 8],
            [0, 1, 2],
            [0, 1],
            [1, 2],
            [ct.target.iOS16, ct.target.iOS18],
        ),
    )
    def test_palettization(
        self, compute_unit, n_bits, group_size, channel_axis, cluster_dim, minimum_deployment_target
    ):
        if (
            group_size in (0, 2)
            and cluster_dim == 2
            and minimum_deployment_target == ct.target.iOS18
        ):
            pytest.xfail("rdar://131964912 [Quantization] Test Should not Overflow FP16")

        if cluster_dim > 1:
            if minimum_deployment_target < ct.target.iOS18:
                pytest.skip("Vector palettization is only supported in iOS18+")
            if group_size != 0 and group_size < cluster_dim:
                pytest.skip("Cluster dim must <= group size.")

        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data(
            multi_layer=True
        )

        # per-channel scales for the [32, 64, 2, 2] and [64, 32, 2, 2] weight.
        scale_1 = np.array([2.0] * 32, dtype=np.float32).reshape(32, 1, 1, 1)
        scale_2 = np.array([3.0] * 64, dtype=np.float32).reshape(64, 1, 1, 1)

        unique_weight_1 = create_unique_weight(
            model.conv_1.weight, nbits=n_bits, vector_size=cluster_dim, vector_axis=channel_axis
        )
        unique_weight_2 = create_unique_weight(
            model.conv_2.weight, nbits=n_bits, vector_size=cluster_dim, vector_axis=channel_axis
        )

        # Use grouped-channel-wise lut for conv1 for iOS18+.
        block_sizes = [0] * len(unique_weight_1.shape)
        if minimum_deployment_target >= ct.target.iOS18:
            block_sizes[channel_axis] = group_size
        lut_1_params = _quantization_passes.palettize_weights.blockwise_compress(
            unique_weight_1,
            "UNIQUE",
            nbits=n_bits,
            block_sizes=block_sizes,
            cluster_dim=cluster_dim,
            channel_axis=channel_axis,
        )

        # Use per-tensor lut for conv2.
        lut_2_params = _quantization_passes.palettize_weights.blockwise_compress(
            unique_weight_2,
            "UNIQUE",
            nbits=n_bits,
            block_sizes=[0] * len(unique_weight_2.shape),
            cluster_dim=cluster_dim,
            channel_axis=channel_axis,
        )

        if minimum_deployment_target >= ct.target.iOS18:
            # Only do per-channel-scale for iOS18+.
            unique_weight_1 *= scale_1
            unique_weight_2 *= scale_2

        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(torch.Tensor(unique_weight_1))
            model.conv_2.weight = torch.nn.Parameter(torch.Tensor(unique_weight_2))

        model.register_buffer("_COREML_/metadata_version", torch.tensor(1))
        if minimum_deployment_target >= ct.target.iOS18:
            model.conv_1.register_buffer("_COREML_/weight/compression_type", torch.tensor([2]))
            model.conv_1.register_buffer("_COREML_/weight/lut", torch.tensor(lut_1_params.lut))
            model.conv_1.register_buffer(
                "_COREML_/weight/palettization_scale", torch.from_numpy(scale_1)
            )
        model.conv_2.register_buffer("_COREML_/weight/compression_type", torch.tensor([2]))
        model.conv_2.register_buffer("_COREML_/weight/lut", torch.tensor(lut_2_params.lut))
        if minimum_deployment_target >= ct.target.iOS18:
            model.conv_2.register_buffer(
                "_COREML_/weight/palettization_scale", torch.from_numpy(scale_2)
            )

        traced_model = torch.jit.trace(model, torch_input_values)
        input_shape = [input.shape.to_list() for input in inputs]
        res = self.run_compare_torch(
            input_shape,
            traced_model,
            minimum_deployment_target=minimum_deployment_target,
            compute_unit=compute_unit,
            converter=ct.convert,
            rtol=0.2 if cluster_dim > 1 else 1e-5,  # Vector palettization has larger info loss.
        )
        main_func = res[1]._mil_program.functions["main"]

        if minimum_deployment_target >= ct.target.iOS18:
            expected_dtype = f"uint{n_bits}"
            expected_quantize_ops_num = 2
            expected_palettize_ops_num = 2
            palettize_op_child_op_type = "constexpr_blockwise_shift_scale"
        else:
            expected_dtype = "uint8"
            expected_quantize_ops_num = 0
            expected_palettize_ops_num = 1
            # The iOS16 doesn't have per-channel-scale, so lut output is directly fed into conv.
            palettize_op_child_op_type = "conv"

        quantize_ops = main_func.find_ops(op_type="constexpr_blockwise_shift_scale")
        assert len(quantize_ops) == expected_quantize_ops_num
        palettize_ops = main_func.find_ops(op_type="constexpr_lut_to_dense")
        assert len(palettize_ops) == expected_palettize_ops_num
        for palettize_op in palettize_ops:
            assert types.builtin_to_string(palettize_op.indices.dtype) == expected_dtype
            assert palettize_op.outputs[0].child_ops[0].op_type == palettize_op_child_op_type
            if minimum_deployment_target >= ct.target.iOS18:
                assert palettize_op.lut.shape[-1] == cluster_dim

    @pytest.mark.parametrize(
        "compute_unit, minimum_deployment_target",
        itertools.product(compute_units, [ct.target.iOS16, ct.target.iOS18]),
    )
    def test_palettization_8bit_lut(self, compute_unit, minimum_deployment_target):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data(
            multi_layer=True
        )
        unique_weight_1 = create_unique_weight(model.conv_1.weight, nbits=4)
        unique_weight_2 = create_unique_weight(model.conv_2.weight, nbits=6)

        lut_1_params = _quantization_passes.palettize_weights.grouped_channelwise_compress(
            unique_weight_1,
            "UNIQUE",
            nbits=4,
            channel_axis=0,
            channel_group_size=0,
        )
        quant_1_params = _quantization_passes.linear_quantize_weights.blockwise_compress(
            lut_1_params.lut,
            nbits=8,
            mode="LINEAR",
            signed=True,
            block_sizes=[0] * len(lut_1_params.lut.shape),
        )
        lut_2_params = _quantization_passes.palettize_weights.grouped_channelwise_compress(
            unique_weight_2,
            "UNIQUE",
            nbits=6,
            channel_axis=1,
            channel_group_size=0,
        )
        quant_2_params = _quantization_passes.linear_quantize_weights.blockwise_compress(
            lut_2_params.lut,
            nbits=8,
            mode="LINEAR_SYMMETRIC",
            signed=False,
            block_sizes=[0] * len(lut_2_params.lut.shape),
        )

        # Reconstruct the weight in torch model for numerical comparison.
        dequantized_lut_1 = _quantization_passes.linear_quantize_weights.decompress(quant_1_params)
        reconstruct_weight_1 = _quantization_passes.palettize_weights.decompress(
            lut_1_params._replace(lut=dequantized_lut_1)
        )
        dequantized_lut_2 = _quantization_passes.linear_quantize_weights.decompress(quant_2_params)
        reconstruct_weight_2 = _quantization_passes.palettize_weights.decompress(
            lut_2_params._replace(lut=dequantized_lut_2)
        )
        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(torch.Tensor(reconstruct_weight_1))
            model.conv_2.weight = torch.nn.Parameter(torch.Tensor(reconstruct_weight_2))

        # Register buffers for compression metadata.
        model.register_buffer("_COREML_/metadata_version", torch.tensor(1))
        model.conv_1.register_buffer("_COREML_/weight/compression_type", torch.tensor([2, 3]))
        model.conv_1.register_buffer("_COREML_/weight/lut", torch.tensor(dequantized_lut_1))
        model.conv_1.register_buffer("_COREML_/weight/quantization_n_bits", torch.tensor(8))
        model.conv_1.register_buffer(
            "_COREML_/weight/quantization_scale", torch.from_numpy(quant_1_params.scale)
        )
        model.conv_1.register_buffer(
            "_COREML_/weight/zero_point", torch.from_numpy(quant_1_params.offset)
        )
        model.conv_2.register_buffer("_COREML_/weight/compression_type", torch.tensor([2, 3]))
        model.conv_2.register_buffer("_COREML_/weight/lut", torch.tensor(dequantized_lut_2))
        model.conv_2.register_buffer("_COREML_/weight/quantization_n_bits", torch.tensor(8))
        model.conv_2.register_buffer(
            "_COREML_/weight/quantization_scale", torch.from_numpy(quant_2_params.scale)
        )
        model.conv_2.register_buffer(
            "_COREML_/weight/zero_point", torch.from_numpy(quant_2_params.offset)
        )

        traced_model = torch.jit.trace(model, torch_input_values)
        input_shape = [input.shape.to_list() for input in inputs]

        pytest_context_manager = nullcontext()
        if minimum_deployment_target < ct.target.iOS18:
            pytest_context_manager = pytest.raises(
                ValueError, match="Please set minimum_deployment_target to iOS18 or later"
            )

        with pytest_context_manager:
            res = self.run_compare_torch(
                input_shape,
                traced_model,
                minimum_deployment_target=minimum_deployment_target,
                compute_unit=compute_unit,
                converter=ct.convert,
            )

        if minimum_deployment_target < ct.target.iOS18:
            return

        main_func = res[1]._mil_program.functions["main"]
        quantize_ops = main_func.find_ops(op_type="constexpr_blockwise_shift_scale")
        assert len(quantize_ops) == 2
        palettize_ops = main_func.find_ops(op_type="constexpr_lut_to_dense")
        assert len(palettize_ops) == 2
        assert types.builtin_to_string(palettize_ops[0].indices.dtype) == "uint4"
        assert types.builtin_to_string(palettize_ops[1].indices.dtype) == "uint6"
        for palettize_op in palettize_ops:
            assert palettize_op.outputs[0].child_ops[0].op_type == "constexpr_blockwise_shift_scale"

    @pytest.mark.parametrize(
        "compute_unit, sparse_ratio, minimum_deployment_target",
        itertools.product(
            compute_units,
            [0.01, 0.5, 0.99],
            [ct.target.iOS16, ct.target.iOS18],
        ),
    )
    def test_pruning(self, compute_unit, sparse_ratio, minimum_deployment_target):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data(
            multi_layer=True
        )
        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(
                torch.Tensor(
                    create_sparse_weight(model.conv_1.weight, target_sparsity=sparse_ratio)
                )
            )
            model.conv_2.weight = torch.nn.Parameter(
                torch.Tensor(
                    create_sparse_weight(model.conv_2.weight, target_sparsity=sparse_ratio)
                )
            )

        model.register_buffer("_COREML_/metadata_version", torch.tensor(1))
        model.conv_1.register_buffer("_COREML_/weight/compression_type", torch.tensor([1]))
        model.conv_2.register_buffer("_COREML_/weight/compression_type", torch.tensor([1]))

        traced_model = torch.jit.trace(model, torch_input_values)
        input_shape = [input.shape.to_list() for input in inputs]
        res = self.run_compare_torch(
            input_shape,
            traced_model,
            minimum_deployment_target=minimum_deployment_target,
            compute_unit=compute_unit,
            converter=ct.convert,
        )
        main_func = res[1]._mil_program.functions["main"]
        sparse_ops = main_func.find_ops(op_type="constexpr_sparse_to_dense")
        assert len(sparse_ops) == 2

        for sparse_op in sparse_ops:
            assert sparse_op.outputs[0].child_ops[0].op_type == "conv"
            assert types.builtin_to_string(sparse_op.nonzero_data.dtype) == "fp32"
            if minimum_deployment_target >= ct.target.iOS18:
                assert types.builtin_to_string(sparse_op.mask.dtype) == "uint1"
            else:
                assert types.builtin_to_string(sparse_op.mask.dtype) == "uint8"
                assert types.builtin_to_string(sparse_op.shape.dtype) == "uint32"

    @pytest.mark.parametrize(
        "compute_unit, n_bits, signed",
        itertools.product(
            compute_units,
            [4, 8],
            [True, False],
        ),
    )
    def test_joint_pruning_quantization(self, compute_unit, n_bits, signed):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data(
            multi_layer=True,
        )

        # Make the weight sparse and also quantization-friendly.
        weight_1, scale_1, zero_point_1 = create_quantize_friendly_weight(
            model.conv_1.weight.detach().numpy(), nbits=n_bits, signed=signed
        )
        weight_1 *= np.random.randint(low=0, high=2, size=model.conv_1.weight.shape)
        weight_2, scale_2, zero_point_2 = create_quantize_friendly_weight(
            model.conv_2.weight.detach().numpy(), nbits=n_bits, signed=signed
        )
        weight_2 *= np.random.randint(low=0, high=2, size=model.conv_2.weight.shape)
        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(torch.Tensor(weight_1))
            model.conv_2.weight = torch.nn.Parameter(torch.Tensor(weight_2))

        model.register_buffer("_COREML_/metadata_version", torch.tensor(2))
        model.conv_1.register_buffer("_COREML_/weight/compression_type", torch.tensor([1, 3]))
        model.conv_1.register_buffer("_COREML_/weight/quantization_n_bits", torch.tensor(n_bits))
        model.conv_1.register_buffer(
            "_COREML_/weight/quantization_scale", torch.from_numpy(scale_1)
        )
        model.conv_1.register_buffer("_COREML_/weight/zero_point", torch.from_numpy(zero_point_1))
        model.conv_2.register_buffer("_COREML_/weight/compression_type", torch.tensor([1, 3]))
        model.conv_2.register_buffer("_COREML_/weight/quantization_n_bits", torch.tensor(n_bits))
        model.conv_2.register_buffer(
            "_COREML_/weight/quantization_scale", torch.from_numpy(scale_2)
        )
        model.conv_2.register_buffer("_COREML_/weight/zero_point", torch.from_numpy(zero_point_2))

        traced_model = torch.jit.trace(model, torch_input_values)
        input_shape = [input.shape.to_list() for input in inputs]
        res = self.run_compare_torch(
            input_shape,
            traced_model,
            minimum_deployment_target=ct.target.iOS18,
            compute_unit=compute_unit,
            converter=ct.convert,
            atol=1e-2,
        )
        main_func = res[1]._mil_program.functions["main"]

        sparse_quantize_ops = main_func.find_ops(op_type="constexpr_sparse_blockwise_shift_scale")
        assert len(sparse_quantize_ops) == 2
        for sparse_quantize_op in sparse_quantize_ops:
            expected_dtype = f"int{n_bits}" if signed else f"uint{n_bits}"
            assert types.builtin_to_string(sparse_quantize_op.nonzero_data.dtype) == expected_dtype
            assert types.builtin_to_string(sparse_quantize_op.data_mask.dtype) == "uint1"
            assert types.builtin_to_string(sparse_quantize_op.scale.dtype) == "fp32"
            assert sparse_quantize_op.outputs[1].child_ops[0].op_type == "constexpr_sparse_to_dense"

        sparse_ops = main_func.find_ops(op_type="constexpr_sparse_to_dense")
        assert len(sparse_ops) == 2
        for sparse_op in sparse_ops:
            assert types.builtin_to_string(sparse_op.mask.dtype) == "uint1"
            assert types.builtin_to_string(sparse_op.nonzero_data.dtype) == "fp32"
            assert sparse_op.outputs[0].child_ops[0].op_type == "conv"

    @pytest.mark.parametrize(
        "compute_unit, n_bits, group_size",
        itertools.product(
            compute_units,
            [4, 8],
            [0, 1, 2],
        ),
    )
    def test_joint_pruning_palettization(self, compute_unit, n_bits, group_size):
        model, inputs, torch_input_values, coreml_input_values = get_test_model_and_data(
            multi_layer=True
        )

        # Make the weight sparse and also can be represented by lut.
        weight_1 = create_unique_weight(model.conv_1.weight, nbits=n_bits) * np.random.randint(
            low=0, high=2, size=model.conv_1.weight.shape
        )
        weight_2 = create_unique_weight(model.conv_2.weight, nbits=n_bits) * np.random.randint(
            low=0, high=2, size=model.conv_2.weight.shape
        )

        with torch.no_grad():
            model.conv_1.weight = torch.nn.Parameter(torch.Tensor(weight_1))
            model.conv_2.weight = torch.nn.Parameter(torch.Tensor(weight_2))

        lut_1_params = _quantization_passes.palettize_weights.blockwise_compress(
            weight_1,
            "UNIQUE",
            nbits=n_bits,
            block_sizes=[group_size] + [0] * (len(weight_1.shape) - 1),
        )
        lut_2_params = _quantization_passes.palettize_weights.blockwise_compress(
            weight_2,
            "UNIQUE",
            nbits=n_bits,
            block_sizes=[group_size] + [0] * (len(weight_2.shape) - 1),
        )

        model.register_buffer("_COREML_/metadata_version", torch.tensor(1))
        model.conv_1.register_buffer("_COREML_/weight/compression_type", torch.tensor([1, 2]))
        model.conv_1.register_buffer("_COREML_/weight/lut", torch.tensor(lut_1_params.lut))
        model.conv_2.register_buffer("_COREML_/weight/compression_type", torch.tensor([1, 2]))
        model.conv_2.register_buffer("_COREML_/weight/lut", torch.tensor(lut_2_params.lut))

        traced_model = torch.jit.trace(model, torch_input_values)
        input_shape = [input.shape.to_list() for input in inputs]
        res = self.run_compare_torch(
            input_shape,
            traced_model,
            minimum_deployment_target=ct.target.iOS18,
            compute_unit=compute_unit,
            converter=ct.convert,
        )
        main_func = res[1]._mil_program.functions["main"]

        sparse_palettize_ops = main_func.find_ops(op_type="constexpr_lut_to_sparse")
        assert len(sparse_palettize_ops) == 2
        for sparse_palettize_op in sparse_palettize_ops:
            assert (
                types.builtin_to_string(sparse_palettize_op.indices_nonzero_data.dtype)
                == f"uint{n_bits}"
            )
            assert types.builtin_to_string(sparse_palettize_op.indices_mask.dtype) == "uint1"
            assert types.builtin_to_string(sparse_palettize_op.lut.dtype) == "fp32"
            assert (
                sparse_palettize_op.outputs[1].child_ops[0].op_type == "constexpr_sparse_to_dense"
            )
            # As both palettization and pruning is on the original weight, the shape of lut should
            # match the original weight's shape except on the output channel.
            weight_shape = sparse_palettize_op.outputs[1].child_ops[0].outputs[0].shape
            expected_lut_shape = [1] * len(weight_shape) + [2**n_bits] + [1]
            if group_size > 0:
                expected_lut_shape[0] = weight_shape[0] // group_size
            assert sparse_palettize_op.lut.shape == tuple(expected_lut_shape)

        sparse_ops = main_func.find_ops(op_type="constexpr_sparse_to_dense")
        assert len(sparse_ops) == 2
        for sparse_op in sparse_ops:
            assert types.builtin_to_string(sparse_op.mask.dtype) == "uint1"
            assert types.builtin_to_string(sparse_op.nonzero_data.dtype) == "fp32"
            assert sparse_op.outputs[0].child_ops[0].op_type == "conv"
