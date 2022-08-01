#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import numpy as np
import pytest

from .testing_utils import run_compare_builder
from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.mil import (
    Builder as mb,
    get_new_symbol,
    types
)
from coremltools.converters.mil.testing_reqs import backends


class TestConvTranspose:

    @pytest.mark.skipif(not testing_reqs._HAS_TORCH, reason="PyTorch not installed.")
    @pytest.mark.parametrize(
        ",".join(
            [
                "use_cpu_only",
                "backend",
                "conv_dim",
                "config",
            ]
        ),
        itertools.product(
            [True, False],
            backends,
            ["conv1d", "conv2d", "conv3d"],
            [{
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
        ),
    )
    def test_builder_to_backend_stress(
        self,
        use_cpu_only,
        backend,
        conv_dim,
        config,
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

        import torch
        import torch.nn as nn

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
        weight = wts["weight"].detach().numpy()
        bias = wts["bias"].detach().numpy() if has_bias else None

        input = torch.randn(*input_shape)
        output = m(input)
        output = output.detach().numpy()
        input = input.detach().numpy()

        output_shape = list(output.shape)
        if test_symbolic:
            # For symbolic input test
            # Make Batch Size and input channel as symbolic
            symbolic_batch_size = get_new_symbol()
            input_shape[0] = symbolic_batch_size
            output_shape[0] = symbolic_batch_size

        expected_output_types = tuple(output_shape[:]) + (types.fp32,)
        expected_outputs = [output]

        input_placeholders = {"x": mb.placeholder(shape=input_shape)}
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
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestConv:
    @pytest.mark.skipif(not testing_reqs._HAS_TORCH, reason="PyTorch not installed.")
    @pytest.mark.parametrize(
        ",".join(
            [
                "use_cpu_only",
                "backend",
                "conv_dim",
                "config",
            ]
        ),
        itertools.product(
            [True, False],
            backends,
            ["conv1d", "conv2d", "conv3d"],
            [{
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
    def test_builder_to_backend_stress(
        self,
        use_cpu_only,
        backend,
        conv_dim,
        config,
    ):
        padding = config["padding"]
        DHWKdKhKw = config["DHWKdKhKw"]
        stride = config["stride"]
        dilation = config["dilation"]
        has_bias = config["has_bias"]
        groups = config["groups"]
        symbolic = config["symbolic"]

        D, H, W, Kd, Kh, Kw = DHWKdKhKw
        N, C_in, C_out = 1, 1 * groups, 2 * groups

        import torch
        import torch.nn as nn

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

        input_placeholders = {"x": mb.placeholder(shape=input_shape)}
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
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

    @pytest.mark.skipif(not testing_reqs._HAS_TORCH, reason="PyTorch not installed.")
    @pytest.mark.parametrize(
        ",".join(
            [
                "use_cpu_only",
                "backend",
                "conv_dim",
                "config",
            ]
        ),
        itertools.product(
            [True, False],
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
        use_cpu_only,
        backend,
        conv_dim,
        config,
    ):
        padding = config["padding"]
        DHWKdKhKw = config["DHWKdKhKw"]
        stride = config["stride"]
        has_bias = config["has_bias"]
        groups = config["groups"]
        symbolic = config["symbolic"]

        if backend[0] == "neuralnetwork" and groups > 1:
            pytest.skip("dynamic conv with groups > 1 is not supported on the neuralnetwork backend")
            
        if backend[0] == "mlprogram" and not use_cpu_only:
            pytest.xfail("rdar://97398343 (test_builder_to_backend_stress_weights_input is failing on mlprogram + GPU)")

        D, H, W, Kd, Kh, Kw = DHWKdKhKw
        N, C_in, C_out = 1, 1 * groups, 2 * groups

        import torch
        import torch.nn as nn

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

        input_placeholders = {"x": mb.placeholder(shape=input_shape), "input_weight":mb.placeholder(shape=weight.shape)}
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
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True], backends, )
    )
    def test_conv_int_bias_fusion(self, use_cpu_only, backend):
        """
        Test conv bias fusion when const input is of type int.
        Expected behavior is that the bias const will be cast to the same dtype as the
        weight during the fuse_conv_bias pass, otherwise mb.conv() will raise an error.


        Input graph:
                                        Const(int type)
                                          |
                                          V
        input -----> convolution -----> add/sub  ---> out

        Output graph:
        input -----> convolution -----> out
        """
        weight = np.array([2.5], dtype=np.float32).reshape([1, 1, 1, 1])

        def build(x):
            x = mb.conv(x=x, weight=weight)
            bias = mb.const(val=[10])
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
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )
