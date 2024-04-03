#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
from typing import Optional

import numpy as np
import pytest
import torch
import torchvision

import coremltools as ct
from coremltools import TensorType
from coremltools._deps import (
    _HAS_TORCH,
    _HAS_TORCH_VISION,
    MSG_TORCH_NOT_FOUND,
    MSG_TORCH_VISION_NOT_FOUND,
)
from coremltools.converters.mil.testing_utils import get_op_types_in_program

from .testing_utils import TorchBaseTest

pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)

torch.manual_seed(30)
np.random.seed(30)
torch.backends.quantized.engine = "qnnpack"


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
        "quant_dtype, channel_axis",
        itertools.product([torch.quint8, torch.qint8], [0, 1, None]),
    )
    def test_quantized_params(self, quant_dtype, channel_axis):
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
        res = self.run_compare_torch(input_shape, model)
        prog = res[1]._mil_program
        assert get_op_types_in_program(prog) == ["constexpr_affine_dequantize", "linear"]


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
