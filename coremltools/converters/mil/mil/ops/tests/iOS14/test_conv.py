#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools._deps import _HAS_TORCH, MSG_TORCH_NOT_FOUND
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.mil.ops.tests.iOS14 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.testing_reqs import compute_units
from coremltools.converters.mil.testing_utils import random_gen

if _HAS_TORCH:
    import torch
    import torch.nn as nn


class TestConvTranspose:
    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "conv_dim",
                "config",
                "x_weight_dtype",
            ]
        ),
        itertools.product(
            compute_units,
            backends,
            ["conv1d", "conv2d", "conv3d"],
            [
                {
                    "padding": (1, 2, 3),
                    "DHWKdKhKw": (10, 12, 14, 3, 2, 4),
                    "stride": (2, 1, 1),
                    "dilation": (1, 1, 1),
                    "has_bias": False,
                    "groups": 1,
                    "test_symbolic": False,
                    "test_output_shape": True,
                },
                {
                    "padding": (2, 2, 2),
                    "DHWKdKhKw": (10, 12, 14, 3, 2, 4),
                    "stride": (2, 2, 2),
                    "dilation": (2, 1, 1),
                    "has_bias": False,
                    "groups": 2,
                    "test_symbolic": True,
                    "test_output_shape": False,
                },
                {
                    "padding": (1, 2, 3),
                    "DHWKdKhKw": (7, 7, 7, 2, 2, 2),
                    "stride": (2, 2, 2),
                    "dilation": (2, 1, 1),
                    "has_bias": True,
                    "groups": 1,
                    "test_symbolic": True,
                    "test_output_shape": False,
                },
                {
                    "padding": (2, 2, 2),
                    "DHWKdKhKw": (7, 7, 7, 2, 2, 2),
                    "stride": (2, 1, 1),
                    "dilation": (1, 1, 1),
                    "has_bias": True,
                    "groups": 2,
                    "test_symbolic": False,
                    "test_output_shape": False,
                },
            ],
            [(np.float32, np.float32), (np.float16, np.float16)],
        ),
    )
    def test_builder_to_backend_stress(
        self,
        compute_unit,
        backend,
        conv_dim,
        config,
        x_weight_dtype,
    ):
        padding = config["padding"]
        DHWKdKhKw = config["DHWKdKhKw"]
        stride = config["stride"]
        dilation = config["dilation"]
        has_bias = config["has_bias"]
        groups = config["groups"]
        test_symbolic = config["test_symbolic"]
        test_output_shape = config["test_output_shape"]

        D, H, W, Kd, Kh, Kw = DHWKdKhKw
        N, C_in, C_out = 1, 1 * groups, 2 * groups

        x_dtype, weight_bias_dtype = x_weight_dtype
        x_builtin_dtype = types.numpy_type_to_builtin_type(x_dtype)

        isDeconv1d = conv_dim == "conv1d"
        isDeconv2d = conv_dim == "conv2d"

        if isDeconv1d:
            strides = [stride[0]]
            dilations = [dilation[0]]
            kernels = [Kh]
            m = nn.ConvTranspose1d(
                C_in,
                C_out,
                kernels,
                stride=strides,
                dilation=dilations,
                bias=has_bias,
                groups=groups,
                padding=padding[0],
            )
            input_shape = [N, C_in, H]
            paddings = [padding[0], padding[0]]

        elif isDeconv2d:
            strides = [stride[0], stride[1]]
            dilations = [dilation[0], dilation[1]]
            kernels = [Kh, Kw]
            m = nn.ConvTranspose2d(
                C_in,
                C_out,
                kernels,
                stride=strides,
                dilation=dilations,
                bias=has_bias,
                groups=groups,
                padding=(padding[0], padding[1]),
            )
            input_shape = [N, C_in, H, W]
            paddings = [padding[0], padding[0], padding[1], padding[1]]
        else:
            strides = [stride[0], stride[1], stride[2]]
            dilations = [dilation[0], dilation[1], dilation[2]]
            kernels = [Kd, Kh, Kw]
            m = nn.ConvTranspose3d(
                C_in,
                C_out,
                kernels,
                stride=strides,
                dilation=dilations,
                bias=has_bias,
                groups=groups,
                padding=padding,
            )
            input_shape = [N, C_in, D, H, W]
            paddings = [
                padding[0],
                padding[0],
                padding[1],
                padding[1],
                padding[2],
                padding[2],
            ]

        wts = m.state_dict()
        weight = wts["weight"].detach().numpy().astype(weight_bias_dtype)
        bias = wts["bias"].detach().numpy().astype(weight_bias_dtype) if has_bias else None

        input = torch.randn(*input_shape)
        output = m(input)
        output = output.detach().numpy().astype(x_dtype)
        input = input.detach().numpy().astype(x_dtype)

        output_shape = list(output.shape)
        if test_symbolic:
            # For symbolic input test
            # Make Batch Size and input channel as symbolic
            symbolic_batch_size = get_new_symbol()
            input_shape[0] = symbolic_batch_size
            output_shape[0] = symbolic_batch_size

        expected_output_types = tuple(output_shape[:]) + (types.fp32,)
        expected_outputs = [output]

        input_placeholders = {"x": mb.placeholder(shape=input_shape, dtype=x_builtin_dtype)}
        input_values = {"x": input}

        def build(x):
            arguments = {
                "x": x,
                "weight": weight,
                "pad": paddings,
                "pad_type": "custom",
                "strides": strides,
                "dilations": dilations,
                "groups": groups,
            }
            if has_bias:
                arguments["bias"] = bias
            if test_output_shape:
                arguments["output_shape"] = output.shape
            return mb.conv_transpose(**arguments)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            atol=1e-3 if x_dtype == np.float16 and backend.backend == "neuralnetwork" else 1e-4,
        )


class TestConv:
    @pytest.mark.parametrize(
        "backend, pad_type",
        itertools.product(
            backends,
            ["valid", "same", "same_lower", "custom"],
        ),
    )
    def test_type_inference_cache_no_pad(self, backend, pad_type):
        # Test the type inference has the caching mechanism to ensure
        # same symbolic input shapes results in the same output shape
        if pad_type == "same_lower" and backend.opset_version == ct.target.iOS15:
            return

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 3, get_new_symbol(), get_new_symbol()), dtype=types.fp32)
            ],
            opset_version=backend.opset_version,
        )
        def prog(x):
            weight = np.random.rand(2, 3, 2, 2)

            # Basic conv
            conv_1 = mb.conv(x=x, weight=weight)
            conv_2 = mb.conv(x=x, weight=weight)
            assert conv_1.shape == conv_2.shape

            # With strides / dialations
            conv_1 = mb.conv(x=x, weight=weight, strides=[1, 2], dilations=[3, 4])
            conv_2 = mb.conv(x=x, weight=weight, strides=[1, 2], dilations=[3, 4])
            assert conv_1.shape == conv_2.shape

            # With padding
            conv_1 = mb.conv(x=x, weight=weight, pad_type=pad_type, pad=[2, 3, 4, 5])
            conv_2 = mb.conv(x=x, weight=weight, pad_type=pad_type, pad=[2, 3, 4, 5])
            assert conv_1.shape == conv_2.shape
            return conv_1

    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        "compute_unit, backend, padding_mode, conv_dim",
        itertools.product(
            compute_units,
            backends,
            ["same_lower", "same", "valid"],
            ["conv1d", "conv2d", "conv3d"],
        ),
    )
    def test_padding_mode_stress(self, compute_unit, backend, padding_mode, conv_dim):
        def rotation_tensor(tensor):
            assert tensor.shape[0] == tensor.shape[1] == 1
            tensor = tensor[0][0]
            rank = len(tensor.shape)
            new_tensor = np.copy(np.flip(tensor, axis=tuple(range(rank))))
            return np.expand_dims(new_tensor, axis=(0, 1))

        if conv_dim == "conv3d" and padding_mode == "same_lower":
            if backend.backend == "neuralnetwork":
                pytest.skip("same_lower mode not supported for conv3d in neuralnetwork backend")

        if padding_mode == "same_lower" and backend.opset_version == ct.target.iOS15:
            pytest.skip("same_lower pad_type not supported iOS15 opset")

        batch, in_channels, out_channels = 1, 1, 1
        input_shape = (batch, in_channels, 4, 5, 6)  # batch, channel, height, width
        kernel_size = (2, 4, 3)
        torch_padding_mode = padding_mode if padding_mode != "same_lower" else "same"

        # Get the right shape for each conv_dim
        if conv_dim == "conv1d":
            input_shape = input_shape[:3]
            kernel_size = kernel_size[:1]
        elif conv_dim == "conv2d":
            input_shape = input_shape[:4]
            kernel_size = kernel_size[:2]

        # Get the ground truth answer from torch
        if conv_dim == "conv1d":
            m = torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=torch_padding_mode,
                bias=False,
            )
        elif conv_dim == "conv2d":
            m = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=torch_padding_mode,
                bias=False,
            )
        elif conv_dim == "conv3d":
            m = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=torch_padding_mode,
                bias=False,
            )

        # Original weight / inputs for the torch model
        weight = torch.clone(m.state_dict()["weight"])
        input = torch.randn(*input_shape, dtype=torch.float32)

        # Coreml weights / inputs values
        coreml_weight = weight.detach().numpy()
        coreml_input = input.detach().numpy()

        if padding_mode == "same_lower":
            # For the same_lower padding mode, we get the ground truth output by doing the following steps
            # (1) Rotate the input value
            # (2) Rotate the kernel value
            # (3) Rotate the torch out
            rotated_input = torch.tensor(
                rotation_tensor(input.detach().numpy()), dtype=torch.float32
            )
            rotated_weight = torch.tensor(
                rotation_tensor(weight.detach().numpy()), dtype=torch.float32
            )
            m.load_state_dict({"weight": rotated_weight}, strict=False)
            output = m(rotated_input).detach().numpy()
            output = rotation_tensor(output)
        else:
            output = m(input).detach().numpy()

        output_shape = list(output.shape)
        expected_output_types = tuple(output_shape[:]) + (types.fp32,)
        expected_outputs = [output]
        input_placeholders = {"x": mb.placeholder(shape=input_shape)}
        input_values = {"x": coreml_input}

        def build(x):
            arguments = {
                "x": x,
                "weight": coreml_weight,
                "pad_type": padding_mode,
            }
            return mb.conv(**arguments)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "conv_dim",
                "config",
                "x_weight_dtype",
            ]
        ),
        itertools.product(
            compute_units,
            backends,
            ["conv1d", "conv2d", "conv3d"],
            [
                {
                    "padding": (1, 1, 1),
                    "DHWKdKhKw": (10, 12, 14, 3, 2, 4),
                    "stride": (2, 1, 1),
                    "dilation": (1, 1, 1),
                    "has_bias": False,
                    "groups": 1,
                    "symbolic": False,
                },
                {
                    "padding": (2, 2, 2),
                    "DHWKdKhKw": (10, 12, 14, 3, 2, 4),
                    "stride": (2, 2, 2),
                    "dilation": (2, 1, 1),
                    "has_bias": False,
                    "groups": 2,
                    "symbolic": True,
                },
                {
                    "padding": (1, 1, 1),
                    "DHWKdKhKw": (5, 5, 5, 2, 2, 2),
                    "stride": (2, 2, 2),
                    "dilation": (2, 1, 1),
                    "has_bias": True,
                    "groups": 1,
                    "symbolic": True,
                },
                {
                    "padding": (2, 2, 2),
                    "DHWKdKhKw": (5, 5, 5, 2, 2, 2),
                    "stride": (2, 1, 1),
                    "dilation": (1, 1, 1),
                    "has_bias": True,
                    "groups": 2,
                    "symbolic": False,
                },
            ],
            [(np.float32, np.float32), (np.float16, np.float16)],
        ),
    )
    def test_builder_to_backend_stress(
        self,
        compute_unit,
        backend,
        conv_dim,
        config,
        x_weight_dtype,
    ):
        if (
            backend.backend == 'neuralnetwork' and
            conv_dim == "conv2d" and
            config == {
                "padding": (1, 1, 1),
                "DHWKdKhKw": (5, 5, 5, 2, 2, 2),
                "stride": (2, 2, 2),
                "dilation": (2, 1, 1),
                "has_bias": True,
                "groups": 1,
                "symbolic": True,
            }
        ):
            pytest.xfail("rdar://121954894: Conv2d starts to fail")

        padding = config["padding"]
        DHWKdKhKw = config["DHWKdKhKw"]
        stride = config["stride"]
        dilation = config["dilation"]
        has_bias = config["has_bias"]
        groups = config["groups"]
        symbolic = config["symbolic"]

        D, H, W, Kd, Kh, Kw = DHWKdKhKw
        N, C_in, C_out = 1, 1 * groups, 2 * groups

        x_dtype, weight_bias_dtype = x_weight_dtype
        x_builtin_dtype = types.numpy_type_to_builtin_type(x_dtype)

        isConv1d = conv_dim == "conv1d"
        isConv2d = conv_dim == "conv2d"

        if isConv1d:
            strides = [stride[0]]
            dilations = [dilation[0]]
            kernels = [Kh]
            m = nn.Conv1d(
                C_in,
                C_out,
                kernels,
                stride=strides,
                dilation=dilations,
                bias=has_bias,
                groups=groups,
                padding=padding[0],
            )
            input_shape = [N, C_in, H]
            paddings = [padding[0], padding[0]]
        elif isConv2d:
            strides = [stride[0], stride[1]]
            dilations = [dilation[0], dilation[1]]
            kernels = [Kh, Kw]
            m = nn.Conv2d(
                C_in,
                C_out,
                kernels,
                stride=strides,
                dilation=dilations,
                bias=has_bias,
                groups=groups,
                padding=(padding[0], padding[1]),
            )
            input_shape = [N, C_in, H, W]
            paddings = [padding[0], padding[0], padding[1], padding[1]]
        else:
            strides = [stride[0], stride[1], stride[2]]
            dilations = [dilation[0], dilation[1], dilation[2]]
            kernels = [Kd, Kh, Kw]
            m = nn.Conv3d(
                C_in,
                C_out,
                kernels,
                stride=strides,
                dilation=dilations,
                bias=has_bias,
                groups=groups,
                padding=padding,
            )
            input_shape = [N, C_in, D, H, W]
            paddings = [
                padding[0],
                padding[0],
                padding[1],
                padding[1],
                padding[2],
                padding[2],
            ]

        wts = m.state_dict()
        weight = wts["weight"].detach().numpy().astype(weight_bias_dtype)
        bias = wts["bias"].detach().numpy().astype(weight_bias_dtype) if has_bias else None

        # PyTorch and CoreML weight format is same
        # PyTorch weight format: C_out, C_in, H, W
        # MIL weight format: C_out, C_in, H, W

        input = random_gen(input_shape)
        input = torch.Tensor(input)
        output = m(input)
        output = output.detach().numpy().astype(x_dtype)
        input = input.detach().numpy().astype(x_dtype)

        output_shape = list(output.shape)
        if symbolic:
            # For symbolic input test
            # Make Batch Size and input channel as symbolic
            symbolic_batch_size = get_new_symbol()
            input_shape[0] = symbolic_batch_size
            output_shape[0] = symbolic_batch_size

        expected_output_types = tuple(output_shape[:]) + (x_builtin_dtype,)
        expected_outputs = [output]

        input_placeholders = {"x": mb.placeholder(shape=input_shape, dtype=x_builtin_dtype)}
        input_values = {"x": input}

        def build(x):
            arguments = {
                "x": x,
                "weight": weight,
                "pad": paddings,
                "pad_type": "custom",
                "strides": strides,
                "dilations": dilations,
                "groups": groups,
            }
            if has_bias:
                arguments["bias"] = bias
            return mb.conv(**arguments)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            atol=1e-3 if x_dtype == np.float16 and backend.backend == "neuralnetwork" else 1e-4,
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "conv_dim",
                "config",
            ]
        ),
        itertools.product(
            compute_units,
            backends,
            ["conv1d", "conv2d"],
            [
                {
                    "padding": (1, 1, 1),
                    "DHWKdKhKw": (10, 12, 14, 3, 2, 4),
                    "stride": (2, 1, 1),
                    "dilation": (1, 1, 1),
                    "has_bias": False,
                    "groups": 1,
                    "symbolic": False,
                },
                {
                    "padding": (2, 2, 2),
                    "DHWKdKhKw": (10, 12, 14, 3, 2, 4),
                    "stride": (2, 2, 2),
                    "dilation": (2, 1, 1),
                    "has_bias": False,
                    "groups": 2,
                    "symbolic": True,
                },
                {
                    "padding": (1, 1, 1),
                    "DHWKdKhKw": (5, 5, 5, 2, 2, 2),
                    "stride": (2, 2, 2),
                    "dilation": (2, 1, 1),
                    "has_bias": True,
                    "groups": 1,
                    "symbolic": True,
                },
                {
                    "padding": (2, 2, 2),
                    "DHWKdKhKw": (5, 5, 5, 2, 2, 2),
                    "stride": (2, 1, 1),
                    "dilation": (1, 1, 1),
                    "has_bias": True,
                    "groups": 2,
                    "symbolic": False,
                },
            ],
        ),
    )
    def test_builder_to_backend_stress_weights_input(
        self,
        compute_unit,
        backend,
        conv_dim,
        config,
    ):
        if (
            conv_dim == "conv2d" and
            config == {
                'padding': (1, 1, 1),
                'DHWKdKhKw': (5, 5, 5, 2, 2, 2),
                'stride': (2, 2, 2),
                'dilation': (2, 1, 1),
                'has_bias': True,
                'groups': 1,
                'symbolic': True,
            }
        ):
            pytest.xfail("rdar://121954894: Conv2d starts to fail")

        padding = config["padding"]
        DHWKdKhKw = config["DHWKdKhKw"]
        stride = config["stride"]
        has_bias = config["has_bias"]
        groups = config["groups"]
        symbolic = config["symbolic"]

        if backend.backend == "neuralnetwork" and groups > 1:
            pytest.skip(
                "dynamic conv with groups > 1 is not supported on the neuralnetwork backend"
            )

        if backend.backend == "mlprogram" and compute_unit != ct.ComputeUnit.CPU_ONLY:
            pytest.xfail(
                "rdar://97398343 (test_builder_to_backend_stress_weights_input is failing on mlprogram + GPU)"
            )

        D, H, W, Kd, Kh, Kw = DHWKdKhKw
        N, C_in, C_out = 1, 1 * groups, 2 * groups

        isConv1d = conv_dim == "conv1d"
        isConv2d = conv_dim == "conv2d"

        if isConv1d:
            strides = [stride[0]]
            kernels = [Kh]
            m = nn.Conv1d(
                C_in,
                C_out,
                kernels,
                stride=strides,
                bias=has_bias,
                groups=groups,
                padding=padding[0],
            )
            input_shape = [N, C_in, H]
            paddings = [padding[0], padding[0]]
        elif isConv2d:
            strides = [stride[0], stride[1]]
            kernels = [Kh, Kw]
            m = nn.Conv2d(
                C_in,
                C_out,
                kernels,
                stride=strides,
                groups=groups,
                padding=(padding[0], padding[1]),
                bias=has_bias,
            )
            input_shape = [N, C_in, H, W]
            paddings = [padding[0], padding[0], padding[1], padding[1]]

        wts = m.state_dict()
        weight = wts["weight"].detach().numpy()
        bias = wts["bias"].detach().numpy() if has_bias else None

        # PyTorch and CoreML weight format is same
        # PyTorch weight format: C_out, C_in, H, W
        # MIL weight format: C_out, C_in, H, W

        input = torch.randn(*input_shape)
        output = m(input)
        output = output.detach().numpy()
        input = input.detach().numpy()

        output_shape = list(output.shape)
        if symbolic:
            # For symbolic input test
            # Make Batch Size and input channel as symbolic
            symbolic_batch_size = get_new_symbol()
            input_shape[0] = symbolic_batch_size
            output_shape[0] = symbolic_batch_size

        expected_output_types = tuple(output_shape[:]) + (types.fp32,)
        expected_outputs = [output]

        input_placeholders = {
            "x": mb.placeholder(shape=input_shape),
            "input_weight": mb.placeholder(shape=weight.shape),
        }
        input_values = {"x": input, "input_weight": weight}

        def build(x, input_weight):
            arguments = {
                "x": x,
                "weight": input_weight,
                "pad": paddings,
                "pad_type": "custom",
                "strides": strides,
                "groups": groups,
            }
            if has_bias:
                arguments["bias"] = bias
            return mb.conv(**arguments)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_conv_bias_fusion(self, compute_unit, backend):
        """
        Test conv bias fusion when const input.


        Input graph:
                                        Const
                                          |
                                          V
        input -----> convolution -----> add/sub  ---> out

        Output graph:
        input -----> convolution -----> out
        """
        weight = np.array([2.5], dtype=np.float32).reshape([1, 1, 1, 1])

        def build(x):
            x = mb.conv(x=x, weight=weight)
            bias = mb.const(val=[10.0])
            return mb.add(x=x, y=bias)

        input = np.array([1, 2, 3, 4], dtype=np.float32).reshape((1, 1, 2, 2))
        output = np.array([12.5, 15.0, 17.5, 20.0], dtype=np.float32).reshape((1, 1, 2, 2))
        expected_output_types = output.shape + (types.fp32,)
        expected_outputs = [output]
        input_placeholders = {"x": mb.placeholder(shape=input.shape)}
        input_values = {"x": input}

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestInvalidConvConfig:
    @pytest.mark.parametrize(
        "compute_unit, backend, conv_dim",
        itertools.product(
            compute_units,
            backends,
            (1, 2, 3),
        ),
    )
    def test_invalid_weight(self, compute_unit, backend, conv_dim):
        N, C_in, C_out = tuple(np.random.randint(low=16, high=32, size=3))
        D = tuple(np.random.randint(low=8, high=16, size=conv_dim))
        K = tuple(np.random.randint(low=1, high=4, size=conv_dim))

        input_shape = (N, C_in) + D
        x = np.random.rand(*input_shape)

        groups = np.random.randint(low=1, high=C_in + 1)
        while C_in % groups != 0:
            groups = np.random.randint(low=1, high=C_in + 1)

        weight = (
            np.random.rand(C_out, C_in // groups + +np.random.randint(low=1, high=8), *K) * 2.0
            - 1.0
        )

        def build(x):
            return mb.conv(x=x, weight=weight, groups=groups)

        with pytest.raises(
            ValueError, match=r"C_in / groups = [0-9]+/[0-9]+ != weight\[1\] \([0-9]+\)"
        ):
            run_compare_builder(
                build,
                {"x": mb.placeholder(shape=input_shape)},
                {"x": x},
                compute_unit=compute_unit,
                backend=backend,
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, conv_dim",
        itertools.product(
            compute_units,
            backends,
            (1, 2, 3),
        ),
    )
    def test_invalid_bias(self, compute_unit, backend, conv_dim):
        N, C_in, C_out = tuple(np.random.randint(low=1, high=10, size=3))
        D = tuple(np.random.randint(low=8, high=16, size=conv_dim))
        K = tuple(np.random.randint(low=1, high=4, size=conv_dim))

        input_shape = (N, C_in) + D
        x = np.random.rand(*input_shape)

        weight = np.random.rand(C_out, C_in, *K) * 2.0 - 1.0

        wrong_bias_size = C_out + np.random.randint(low=1, high=8)
        bias = np.random.rand(wrong_bias_size) * 2.0 - 1.0

        def build(x):
            return mb.conv(x=x, weight=weight, bias=bias)

        with pytest.raises(
            ValueError, match=r"# of bias values [0-9]+ not equal to # output channels [0-9]+"
        ):
            run_compare_builder(
                build,
                {"x": mb.placeholder(shape=input_shape)},
                {"x": x},
                compute_unit=compute_unit,
                backend=backend,
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, conv_dim",
        itertools.product(
            compute_units,
            backends,
            (1, 2, 3),
        ),
    )
    def test_invalid_kernel(self, compute_unit, backend, conv_dim):
        N, C_in, C_out = tuple(np.random.randint(low=1, high=10, size=3))
        D = tuple(np.random.randint(low=8, high=16, size=conv_dim))
        K = tuple(np.random.randint(low=16, high=32, size=conv_dim))

        input_shape = (N, C_in) + D
        x = np.random.rand(*input_shape)

        weight = np.random.rand(C_out, C_in, *K) * 2.0 - 1.0

        def build(x):
            return mb.conv(x=x, weight=weight)

        with pytest.raises(
            ValueError, match=r"spatial dimension [0-9]+ has invalid output size -?[0-9]+"
        ):
            run_compare_builder(
                build,
                {"x": mb.placeholder(shape=input_shape)},
                {"x": x},
                compute_unit=compute_unit,
                backend=backend,
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, conv_dim",
        itertools.product(
            compute_units,
            backends,
            (1, 2, 3),
        ),
    )
    def test_invalid_dilation(self, compute_unit, backend, conv_dim):
        N, C_in, C_out = tuple(np.random.randint(low=1, high=10, size=3))
        D = tuple(np.random.randint(low=8, high=16, size=conv_dim))
        K = tuple(np.random.randint(low=2, high=4, size=conv_dim))
        dilations = tuple(np.random.randint(low=16, high=32, size=conv_dim))

        input_shape = (N, C_in) + D
        x = np.random.rand(*input_shape)

        weight = np.random.rand(C_out, C_in, *K) * 2.0 - 1.0

        def build(x):
            return mb.conv(x=x, weight=weight, dilations=dilations)

        with pytest.raises(
            ValueError, match=r"spatial dimension [0-9]+ has invalid output size -?[0-9]+"
        ):
            run_compare_builder(
                build,
                {"x": mb.placeholder(shape=input_shape)},
                {"x": x},
                compute_unit=compute_unit,
                backend=backend,
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, conv_dim",
        itertools.product(
            compute_units,
            backends,
            (1, 2, 3),
        ),
    )
    def test_invalid_groups(self, compute_unit, backend, conv_dim):
        N, C_in, C_out = tuple(np.random.randint(low=16, high=32, size=3))
        D = tuple(np.random.randint(low=8, high=16, size=conv_dim))
        K = tuple(np.random.randint(low=1, high=4, size=conv_dim))

        input_shape = (N, C_in) + D
        x = np.random.rand(*input_shape)

        groups = np.random.randint(low=1, high=C_in)
        while C_in % groups == 0:
            groups = np.random.randint(low=1, high=C_in)

        weight = np.random.rand(C_out, C_in // groups, *K) * 2.0 - 1.0

        def build(x):
            return mb.conv(x=x, weight=weight, groups=groups)

        with pytest.raises(
            ValueError, match=r"# of input channels [0-9]+ not divisible by groups [0-9]+"
        ):
            run_compare_builder(
                build,
                {"x": mb.placeholder(shape=input_shape)},
                {"x": x},
                compute_unit=compute_unit,
                backend=backend,
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, conv_dim",
        itertools.product(
            compute_units,
            backends,
            (1, 2, 3),
        ),
    )
    def test_invalid_rank(self, compute_unit, backend, conv_dim):
        N, C_in, C_out = tuple(np.random.randint(low=16, high=32, size=3))
        D = tuple(np.random.randint(low=8, high=16, size=conv_dim))

        input_shape = (N, C_in) + D
        x = np.random.rand(*input_shape)

        wrong_K = tuple(np.random.randint(low=1, high=4, size=conv_dim - 1))

        weight = np.random.rand(C_out, C_in, *wrong_K) * 2.0 - 1.0
        strides = tuple(np.random.randint(low=1, high=4, size=conv_dim + 1))
        dilations = tuple(np.random.randint(low=1, high=4, size=conv_dim + 2))
        pad = tuple(np.random.randint(low=1, high=4, size=2 * conv_dim + 3))

        def build(x):
            return mb.conv(
                x=x, weight=weight, strides=strides, dilations=dilations, pad_type="custom", pad=pad
            )

        with pytest.raises(
            ValueError,
            match=r"input_shape \(length [0-9]+\), "
            r"kernel_shape \(length [0-9]+\), "
            r"strides \(length [0-9]+\), "
            r"dilations \(length [0-9]+\), "
            r"and custom_pad \(length [0-9]+\) divided by two "
            r"must all be the same length",
        ):
            run_compare_builder(
                build,
                {"x": mb.placeholder(shape=input_shape)},
                {"x": x},
                compute_unit=compute_unit,
                backend=backend,
            )
