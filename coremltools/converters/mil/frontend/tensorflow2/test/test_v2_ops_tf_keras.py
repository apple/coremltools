#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import platform
import random

import numpy as np
import pytest
from packaging.version import Version

import coremltools as ct
from coremltools._deps import _get_version
from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.frontend._utils import is_symbolic_dim_in_prog
from coremltools.converters.mil.frontend.tensorflow.test.testing_utils import (
    TensorFlowBaseTest,
)
from coremltools.converters.mil.frontend.tensorflow2.test.testing_utils import (
    TensorFlow2BaseTest,
)
from coremltools.converters.mil.testing_utils import get_op_types_in_program, random_gen
from coremltools.models.utils import _macos_version

TensorFlowBaseTest.run_compare_tf_keras = TensorFlow2BaseTest.run_compare_tf_keras
backends = testing_reqs.backends
compute_units = testing_reqs.compute_units

tf = pytest.importorskip("tensorflow", minversion="2.1.0")

import tensorflow as _tf  # should be after pytest.importorskip checks
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model


class TestActivation(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, op",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(1, 6)],
            [
                tf.keras.layers.ELU,
                tf.keras.layers.LeakyReLU,
                tf.keras.layers.ReLU,
                tf.keras.layers.PReLU,
                tf.keras.layers.Softmax,
                tf.keras.layers.ThresholdedReLU,
            ],
        ),
    )
    def test_layer(self, compute_unit, backend, rank, op):
        shape = np.random.randint(low=2, high=4, size=rank)
        model = tf.keras.Sequential([op(batch_input_shape=shape)])
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, -10, 10)],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, rank, op",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(1, 6)],
            [
                tf.keras.activations.elu,
                tf.keras.activations.exponential,
                tf.keras.activations.hard_sigmoid,
                tf.keras.activations.linear,
                tf.keras.activations.relu,
                tf.keras.activations.selu,
                tf.keras.activations.sigmoid,
                tf.keras.activations.softmax,
                tf.keras.activations.softplus,
                tf.keras.activations.softsign,
                tf.keras.activations.tanh,
            ],
        ),
    )
    def test_activation(self, compute_unit, backend, rank, op):
        kwargs = (
            {"atol": 1e-3, "rtol": 1e-4}
            if op == tf.keras.activations.exponential and compute_unit != ct.ComputeUnit.CPU_ONLY
            else {}
        )
        if op == tf.keras.activations.softmax and rank == 1:
            return  # skip apply softmax to a tensor that is 1D
        shape = np.random.randint(low=2, high=4, size=rank)
        model = tf.keras.Sequential(
            [tf.keras.layers.Activation(op, batch_input_shape=shape)]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, -10, 10)],
            compute_unit=compute_unit,
            backend=backend,
            **kwargs
        )

    @pytest.mark.parametrize("backend", backends)
    def test_conv2d_prelu_fusion(self, backend):
        x_shape = (1, 10, 10, 32)
        x = tf.keras.Input(batch_input_shape=x_shape)  # (B, H, W, C)
        x1 = tf.keras.layers.Conv2D(16, kernel_size=1)(x)
        x1 =  tf.keras.layers.PReLU(alpha_initializer='glorot_uniform', shared_axes=[1, 2])(x1)
        x1 = tf.keras.layers.Conv2D(16, kernel_size=1)(x1)
        x1 = tf.keras.layers.PReLU(alpha_initializer='glorot_uniform', shared_axes=[1, 2])(x1)
        keras_model = tf.keras.Model(inputs=x, outputs=x1)

        res = TensorFlowBaseTest.run_compare_tf_keras(
            keras_model,
            [random_gen(x_shape, -1, 1)],
            compute_unit=ct.ComputeUnit.CPU_ONLY,
            backend=backend,
        )
        coreml_model = res[1]
        mil_prog = coreml_model._get_mil_internal()
        # assert that "prelu" ops are present in the mil program,
        # which should be if "fuse_prelu" pass worked correctly
        assert len(mil_prog.find_ops(op_type="prelu")) == 2
        assert "relu" not in get_op_types_in_program(mil_prog)


class TestBinary(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, op",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(2, 6)],
            [
                tf.keras.layers.Add,
                tf.keras.layers.Average,
                tf.keras.layers.Subtract,
                tf.keras.layers.Maximum,
                tf.keras.layers.Minimum,
            ],
        ),
    )
    def test(self, compute_unit, backend, rank, op):
        shape = np.random.randint(low=1, high=4, size=rank)
        input_x = tf.keras.layers.Input(batch_input_shape=tuple(shape))
        input_y = tf.keras.layers.Input(batch_input_shape=tuple(shape))
        out = op()([input_x, input_y])
        model = tf.keras.Model(inputs=[input_x, input_y], outputs=out)
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, -10, 10), random_gen(shape, -10, 10)],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, rank, axes, normalize",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(2, 3)],
            [-1,],
            [True, False],
        ),
    )
    def test_dot(self, compute_unit, rank, backend, axes, normalize):
        shape = np.random.randint(low=2, high=4, size=rank)
        input_x = tf.keras.layers.Input(batch_input_shape=tuple(shape))
        input_y = tf.keras.layers.Input(batch_input_shape=tuple(shape))
        out = tf.keras.layers.Dot(axes=axes, normalize=normalize)([input_x, input_y])
        model = tf.keras.Model(inputs=[input_x, input_y], outputs=out)
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, -10, 10), random_gen(shape, -10, 10)],
            compute_unit=compute_unit,
            backend=backend,
        )


class TestConcatenate(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, axis",
        itertools.product(
            compute_units, backends, [rank for rank in range(5, 6)], [-1, -2],
        ),
    )
    def test(self, compute_unit, backend, rank, axis):
        shape = np.random.randint(low=2, high=4, size=rank)
        inputs = []
        for _ in range(2):
            inputs.append(tf.keras.layers.Input(batch_input_shape=tuple(shape)))
        out = tf.keras.layers.Concatenate(axis=axis)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=out)
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape), random_gen(shape)],
            compute_unit=compute_unit,
            backend=backend,
        )


class TestConvolution(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "op",
                "padding",
                "data_format",
                "spatial_dim_and_ks",
                "strides",
                "dilations",
                "batch_size",
                "groups",
            ]
        ),
        itertools.product(
            compute_units,
            backends,
            [
                tf.keras.layers.Conv1D,
                tf.keras.layers.Conv2D,
                tf.keras.layers.Conv3D,
            ],
            ["same", "valid"],
            ["channels_last"],
            [
                (2, 4, 4, 2, 2, 2),
                (3, 7, 5, 1, 3, 2)
            ],
            [
                (1, 1, 1),
                (1, 2, 3),
                (1, 3, 2)
            ],
            [
                (1, 1, 1), (2, 2, 2),
            ],
            [1, 3],
            [1, 2],
        ),
    )
    def test_conv(
        self,
        compute_unit,
        backend,
        op,
        padding,
        data_format,
        spatial_dim_and_ks,
        strides,
        dilations,
        batch_size,
        groups,
    ):
        if _get_version(_tf.__version__) < Version("2.5.0") and groups != 1:
            pytest.skip("TF supports groupwise convolution only for version > tf.2.5.0-rc3")

        if _get_version(_tf.__version__) > Version("2.8.0") and groups != 1:
            pytest.xfail("rdar://100814590 ([TF] [Infra] TF 2.10.0 Uses Unimplemented "
                         "PartitionedCall op for Groupwise Convolution)")

        if op == tf.keras.layers.Conv3D and groups != 1:
            pytest.xfail("rdar://81629932 (Conv3d with group > 1 tests failing in TF2.0 converter)")

        for i, stride in enumerate(strides):
            if stride > 1 and dilations[i] > 1:
                pytest.skip("TF does not support strides > 1 in conjunction with dilation_rate > 1")

        for d in dilations:
            if d > 1 and op == tf.keras.layers.Conv3D:
                pytest.skip("Dilations with Conv3D not supported yet, since SpaceToBatchND is "
                            "only supported for ranks 3 or 4")

        s1, s2, s3, k1, k2, k3 = spatial_dim_and_ks
        c_in, c_out = 2, 4
        input_shape = None
        kernel_size = None
        if op == tf.keras.layers.Conv1D:
            input_shape = (batch_size, s3, c_in)
            kernel_size = k3
            strides = strides[2]
            dilations = dilations[2]
        elif op == tf.keras.layers.Conv2D:
            input_shape = (batch_size, s2, s3, c_in)
            kernel_size = (k2, k3)
            strides = (strides[1], strides[2])
            dilations = dilations[1:]
        elif op == tf.keras.layers.Conv3D:
            input_shape = (batch_size, s1, s2, s3, c_in)
            kernel_size = (k1, k2, k3)

        model = tf.keras.Sequential(
            [
                op(
                    batch_input_shape=input_shape,
                    filters=c_out,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding.upper(),
                    data_format=data_format,
                    dilation_rate=dilations,
                    groups=groups,
                )
            ]
        )

        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(input_shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "op",
                "padding",
                "data_format",
                "spatial_dim_and_ks",
                "strides",
                "dilations",
                "batch_size",
            ]
        ),
        itertools.product(
            compute_units,
            backends,
            [
                tf.keras.layers.LocallyConnected1D,
                tf.keras.layers.LocallyConnected2D,
            ],
            ["same", "valid"],
            ["channels_last"],
            [
                (2, 4, 4, 2, 2, 2),
                (3, 7, 5, 1, 3, 2)
            ],
            [
                (1, 1, 1),
                (1, 2, 3),
                (1, 3, 2)
            ],
            [
                (1, 1, 1), (2, 2, 2),
            ],
            [1, 3],
        ),
    )
    def test_conv_locally_connected(
        self,
        compute_unit,
        backend,
        op,
        padding,
        data_format,
        spatial_dim_and_ks,
        strides,
        dilations,
        batch_size,
    ):
        s1, s2, s3, k1, k2, k3 = spatial_dim_and_ks
        c_in, c_out = 2, 3
        input_shape = None
        kernel_size = None
        if op in {tf.keras.layers.Conv1D, tf.keras.layers.LocallyConnected1D}:
            input_shape = (batch_size, s3, c_in)
            kernel_size = k3
            strides = strides[2]
            dilations = dilations[2]
        elif op in {tf.keras.layers.Conv2D, tf.keras.layers.LocallyConnected2D}:
            input_shape = (batch_size, s2, s3, c_in)
            kernel_size = (k2, k3)
            strides = (strides[1], strides[2])
            dilations = dilations[1:]
        elif op == tf.keras.layers.Conv3D:
            input_shape = (batch_size, s1, s2, s3, c_in)
            kernel_size = (k1, k2, k3)

        if op in {
            tf.keras.layers.LocallyConnected1D,
            tf.keras.layers.LocallyConnected2D,
        }:
            if padding != "valid":
                return  # tf.keras only supports "valid"
            model = tf.keras.Sequential(
                [
                    op(
                        batch_input_shape=input_shape,
                        filters=c_out,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding.upper(),
                        data_format=data_format,
                    )
                ]
            )
        else:
            model = tf.keras.Sequential(
                [
                    op(
                        batch_input_shape=input_shape,
                        filters=c_out,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding.upper(),
                        data_format=data_format,
                        dilation_rate=dilations,
                    )
                ]
            )

        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(input_shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "op",
                "padding",
                "data_format",
                "spatial_dim_and_ks",
                "strides",
                "dilations",
                "batch_size",
            ]
        ),
        itertools.product(
            compute_units,
            backends,
            [tf.keras.layers.DepthwiseConv2D],
            ["same", "valid"],
            ["channels_last"],
            [(11, 12, 3, 2), (12, 11, 2, 3)],
            [(1, 1), (2, 2)],
            [(1, 1), (2, 2)],
            [1, 3],
        ),
    )
    def test_depth_wise_conv(
        self,
        compute_unit,
        backend,
        op,
        padding,
        data_format,
        spatial_dim_and_ks,
        strides,
        dilations,
        batch_size,
    ):
        s1, s2, k1, k2 = spatial_dim_and_ks
        c_in = 2

        if len(strides) != np.sum(strides) and len(dilations) != np.sum(dilations):
            # TF produces incorrect output for non-one strides + dilations
            return

        input_shape = (batch_size, s1, s2, c_in)
        model = tf.keras.Sequential(
            [
                op(
                    batch_input_shape=input_shape,
                    kernel_size=(k1, k2),
                    strides=strides,
                    padding=padding.upper(),
                    data_format=data_format,
                    dilation_rate=dilations,
                )
            ]
        )

        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(input_shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "padding",
            ]
        ),
        itertools.product(
            compute_units,
            backends,
            ["same", "valid"],
        ),
    )
    def test_conv2d_padding_dynamic_input(
        self,
        compute_unit,
        backend,
        padding,
    ):
        if backend[0] == "mlprogram" and _macos_version() < (13, 0):
            pytest.skip("Error in declaring network.")

        # Test same padding
        input_layer = Input(batch_size=1, shape=(None, None, 1))
        layer = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            padding=padding,
            activation="relu"
        )(input_layer)
        output_layer = GlobalMaxPooling2D()(layer)
        model = Model(inputs=[input_layer], outputs=[output_layer])
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen((1, 80, 40, 1), rand_min=-10, rand_max=10)],
            inputs_for_conversion=[
                ct.TensorType(
                    shape=(1, ct.RangeDim(upper_bound=80), ct.RangeDim(upper_bound=80), 1)
                )
            ],
            compute_unit=compute_unit,
            backend=backend,
        )


    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "op",
                "padding",
                "data_format",
                "spatial_dim_and_ks",
                "strides",
                "dilations",
                "batch_size",
            ]
        ),
        itertools.product(
            compute_units,
            backends,
            [tf.keras.layers.SeparableConv1D, tf.keras.layers.SeparableConv2D],
            ["same", "valid"],
            ["channels_last"],
            [
                (14, 14, 2, 2),
                (11, 9, 3, 2),
                (12, 11, 2, 3)
            ],
            [
                (1, 1), (2, 2), (3, 3)
            ],
            [(1, 1)],
            [1, 3],
        ),
    )
    def test_separable_conv(
        self,
        compute_unit,
        backend,
        op,
        padding,
        data_format,
        spatial_dim_and_ks,
        strides,
        dilations,
        batch_size,
    ):
        s1, s2, k1, k2 = spatial_dim_and_ks
        c_in, c_out = 2, 3
        input_shape = None
        kernel_size = None
        if op == tf.keras.layers.SeparableConv1D:
            input_shape = (batch_size, s2, c_in)
            kernel_size = k2
            strides = strides[1]
            dilations = dilations[1]
        elif op == tf.keras.layers.SeparableConv2D:
            input_shape = (batch_size, s1, s2, c_in)
            kernel_size = (k1, k2)

        model = tf.keras.Sequential(
            [
                op(
                    batch_input_shape=input_shape,
                    filters=c_out,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding.upper(),
                    data_format=data_format,
                    dilation_rate=dilations,
                )
            ]
        )

        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(input_shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )

class TestConvTranspose(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "op",
                "padding",
                "data_format",
                "spatial_dim_and_ks",
                "output_padding",
                "strides",
                "dilations",
                "batch_size",
            ]
        ),
        itertools.product(
            compute_units,
            backends,
            [tf.keras.layers.Conv2DTranspose, tf.keras.layers.Conv3DTranspose],
            ["same", "valid"],
            ["channels_last"],
            [(7, 11, 12, 1, 2, 2), (9, 5, 7, 3, 3, 3)],
            [(1, 1, 1)],
            [(2, 2, 2), (2, 3, 3)],
            [(1, 1, 1)], # Dilation > 1 not supported by TF
            [1, 3],
        ),
    )
    def test_conv_transpose(
        self,
        compute_unit,
        backend,
        op,
        padding,
        data_format,
        spatial_dim_and_ks,
        output_padding,
        strides,
        dilations,
        batch_size,
    ):
        if (
            platform.machine() == "arm64"
            and backend == ("mlprogram", "fp16")
            and op == tf.keras.layers.Conv3DTranspose
            and padding == "valid"
            and spatial_dim_and_ks == (7, 11, 12, 1, 2, 2)
            and strides == (2, 3, 3)
            and batch_size == 3
        ):
           pytest.xfail("rdar://98015195 ([M1 native tests] Some MIL unittests are failing M1 native)")

        s1, s2, s3, k1, k2, k3 = spatial_dim_and_ks
        c_in, c_out = 2, 3
        input_shape = None
        kernel_size = None
        if op == tf.keras.layers.Conv2DTranspose:
            input_shape = (batch_size, s2, s3, c_in)
            kernel_size = (k2, k3)
            strides = (strides[1], strides[2])
            dilations = dilations[1:]
            output_padding = (output_padding[1], output_padding[2])
        elif op == tf.keras.layers.Conv3DTranspose:
            input_shape = (batch_size, s1, s2, s3, c_in)
            kernel_size = (k1, k2, k3)

        model = tf.keras.Sequential(
            [
                op(
                    batch_input_shape=input_shape,
                    filters=c_out,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding.upper(),
                    output_padding=output_padding,
                    data_format=data_format,
                    dilation_rate=dilations,
                )
            ]
        )

        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(input_shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )

class TestCropping(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, begin_end",
        itertools.product(
            compute_units, backends, [(0, 0), (1, 1), (1, 2), (2, 1), (2, 4), (3, 2)],
        ),
    )
    def test_cropping_1d(self, compute_unit, backend, begin_end):
        shape = (1, 10, 3)
        model = tf.keras.Sequential(
            [tf.keras.layers.Cropping1D(batch_input_shape=shape, cropping=begin_end)]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-1, rand_max=1)],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, begin_end1, begin_end2",
        itertools.product(
            compute_units,
            backends,
            [(0, 0), (1, 1), (2, 1)],
            [(0, 0), (1, 2), (4, 2)],
        ),
    )
    def test_cropping_2d(self, compute_unit, backend, begin_end1, begin_end2):
        shape = (1, 10, 10, 3)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Cropping2D(
                    batch_input_shape=shape, cropping=(begin_end1, begin_end2)
                )
            ]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-1, rand_max=1)],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, begin_end1, begin_end2, begin_end3",
        itertools.product(
            compute_units,
            backends,
            [(0, 0), (1, 2), (2, 1)],
            [(1, 1), (1, 2), (4, 2)],
            [(0, 0), (1, 1), (2, 4)],
        ),
    )
    def test_cropping_3d(
        self, compute_unit, backend, begin_end1, begin_end2, begin_end3
    ):
        shape = (1, 10, 10, 10, 3)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Cropping3D(
                    batch_input_shape=shape,
                    cropping=(begin_end1, begin_end2, begin_end3),
                )
            ]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-1, rand_max=1)],
            compute_unit=compute_unit,
            backend=backend,
        )


class TestDense(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, units, activation, use_bias",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(2, 6)],
            [2, 4, 8],
            [tf.nn.relu, tf.nn.softmax, tf.nn.swish],
            [True, False],
        ),
    )
    def test(self, compute_unit, backend, rank, units, activation, use_bias):
        shape = np.random.randint(low=2, high=4, size=rank)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    batch_input_shape=shape,
                    units=units,
                    activation=activation,
                    use_bias=use_bias,
                )
            ]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )


class TestEmbedding(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, dims, batch_size, input_length",
        itertools.product(
            compute_units,
            backends,
            [(4, 1), (8, 3), (16, 5), (32, 7), (64, 9)],
            [1, 3, 5],
            [2, 4, 10],
        ),
    )
    def test(self, compute_unit, backend, dims, batch_size, input_length):
        # input shape: 2D tensor (batch_size, input_length)
        # output shape: 3D tensor (batch_size, input_length, output_dim)
        shape = (batch_size, input_length)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    batch_input_shape=shape,
                    input_dim=dims[0],
                    output_dim=dims[1],
                    input_length=input_length,
                )
            ]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=0, rand_max=dims[0])],
            compute_unit=compute_unit,
            backend=backend,
            atol=1e-3,
            rtol=1e-4,
        )


class TestFlatten(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, data_format",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(1, 6)],
            ["channels_last", "channels_first"],
        ),
    )
    def test(self, compute_unit, backend, rank, data_format):
        shape = np.random.randint(low=2, high=4, size=rank)
        model = tf.keras.Sequential(
            [tf.keras.layers.Flatten(batch_input_shape=shape, data_format=data_format,)]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )

class TestMasking(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, mask_value, is_masked",
        itertools.product(
            compute_units,
            backends,
            [2, 5],
            [0, 0.4],
            [False, True],
        ),
    )
    def test(self, compute_unit, backend, rank, mask_value, is_masked):
        shape = np.random.randint(low=2, high=4, size=rank)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Masking(
                    batch_input_shape=shape,
                    mask_value=mask_value,
                )
            ]
        )
        input_value = random_gen(shape, -10, 10)
        if is_masked:
            input_value[:, 1] = mask_value
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [input_value],
            compute_unit=compute_unit,
            backend=backend,
        )

class TestLambda(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, function",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(1, 6)],
            [
                lambda x: x + x,
                lambda x: x * 3.14 - 1.0,
                lambda x: np.sqrt(4) + x,
                lambda x: tf.math.abs(x),
            ],
        ),
    )
    def test_unary(self, compute_unit, backend, rank, function):
        shape = np.random.randint(low=2, high=4, size=rank)
        model = tf.keras.Sequential(
            [tf.keras.layers.Lambda(batch_input_shape=shape, function=function,)]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-5, rand_max=5)],
            compute_unit=compute_unit,
            backend=backend,
        )

class TestBatchNormalization(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, axis, momentum, epsilon, mixed_precision",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(1, 6)],
            [0, -1],
            [0.99, 0.85],
            [1e-2, 1e-5],
            [True, False],
        ),
    )
    def test_batch_normalization(
        self, compute_unit, backend, rank, axis, momentum, epsilon, mixed_precision
    ):
        if backend[0] != "mlprogram" and mixed_precision:
            pytest.skip("neuralnetwork backend doesn't support fp16 computation.")

        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        shape = np.random.randint(low=2, high=4, size=rank)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.BatchNormalization(
                    batch_input_shape=shape,
                    axis=axis,
                    momentum=momentum,
                    epsilon=epsilon,
                )
            ]
        )
        random_weights = np.random.rand(4, shape[axis])
        model.layers[0].set_weights(random_weights)

        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )

        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy(tf.keras.backend.floatx())

    @pytest.mark.parametrize(
        "compute_unit, backend, rank_and_axis, momentum, epsilon, mixed_precision",
        itertools.product(
            compute_units,
            backends,
            [(4, 1), (4, -3)],
            [0.99, 0.85],
            [1e-2, 1e-5],
            [True, False],
        ),
    )
    def test_fused_batch_norm_v3(
        self, compute_unit, backend, rank_and_axis, momentum, epsilon, mixed_precision
    ):
        if backend[0] != "mlprogram" and mixed_precision:
            pytest.skip("neuralnetwork backend doesn't support fp16 computation.")

        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        rank, axis = rank_and_axis
        shape = np.random.randint(low=2, high=4, size=rank)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.BatchNormalization(
                    batch_input_shape=shape,
                    axis=axis,
                    momentum=momentum,
                    epsilon=epsilon,
                )
            ]
        )
        random_weights = np.random.rand(4, shape[axis])
        model.layers[0].set_weights(random_weights)

        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )

        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy(tf.keras.backend.floatx())


class TestInstanceNormalization(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, axis, epsilon, center, scale",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(4, 5)],
            [-1],
            [1e-3, 1e-5],
            [True, False],
            [True, False],
        ),
    )
    def test_instance_normalization(
        self, compute_unit, backend, rank, axis, epsilon, center, scale
    ):
        tensorflow_addons = pytest.importorskip("tensorflow_addons")
        from tensorflow_addons.layers import InstanceNormalization

        shape = np.random.randint(low=2, high=4, size=rank)
        model = tf.keras.Sequential(
            [
                InstanceNormalization(
                    batch_input_shape=shape,
                    axis=axis,
                    epsilon=epsilon,
                    center=center,
                    scale=scale,
                )
            ]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-1, rand_max=1)],
            compute_unit=compute_unit,
            backend=backend,
            atol=1e-2,
            rtol=1e-3,
        )


class TestNormalization(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, axis, epsilon, dynamic",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(3, 4)],
            [-1,],
            [1e-2, 1e-10],
            [True, False],
        ),
    )
    def test_layer_normalization(self, compute_unit, backend, rank, axis, epsilon, dynamic):
        shape = np.random.randint(low=2, high=4, size=rank)
        keras_shape = shape.tolist()
        inputs_for_conversion = None

        if dynamic:
            keras_shape[0] = None
            if backend[0] == "mlprogram":
                inputs_for_conversion = [
                    ct.TensorType(shape=[ct.RangeDim(upper_bound=4)] + keras_shape[1:])
                ]

        model = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(
                    batch_input_shape=keras_shape, axis=axis, epsilon=epsilon, trainable=False
                )
            ]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-100, rand_max=100)],
            inputs_for_conversion=inputs_for_conversion,
            compute_unit=compute_unit,
            backend=backend,
        )


    @pytest.mark.parametrize(
        "compute_unit, backend, rank, groups, axis, epsilon, center, scale",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(4, 5)],
            [1, 2, 3],
            [-1],
            [1e-3, 1e-5],
            [True, False],
            [True, False],
        ),
    )
    def test_group_normalization(
        self, compute_unit, backend, rank, groups, axis, epsilon, center, scale
    ):
        tensorflow_addons = pytest.importorskip("tensorflow_addons")
        from tensorflow_addons.layers import GroupNormalization

        shape = np.random.randint(low=2, high=4, size=rank)
        shape[-1] = shape[-1] * groups  # groups must be a multiple of channels
        model = tf.keras.Sequential(
            [
                GroupNormalization(
                    batch_input_shape=shape,
                    groups=groups,
                    axis=axis,
                    epsilon=epsilon,
                    center=center,
                    scale=scale,
                )
            ]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-1, rand_max=1)],
            compute_unit=compute_unit,
            backend=backend,
            atol=1e-3,
            rtol=1e-4,
        )


class TestPadding(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, op, data_format, padding, mixed_precision",
        itertools.product(
            compute_units,
            backends,
            [
                tf.keras.layers.ZeroPadding1D,
                tf.keras.layers.ZeroPadding2D,
                tf.keras.layers.ZeroPadding3D,
            ],
            ["channels_first", "channels_last"],
            [(1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 3, 4), (2, 3, 5)],
            [True, False],
        ),
    )
    def test(self, compute_unit, backend, op, data_format, padding, mixed_precision):
        if backend[0] != "mlprogram" and mixed_precision:
            pytest.skip("neuralnetwork backend doesn't support fp16 computation.")

        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        shape = None
        kwargs = {}
        if op == tf.keras.layers.ZeroPadding1D:
            padding = padding[-1]
            shape = np.random.randint(low=2, high=4, size=3)
        elif op == tf.keras.layers.ZeroPadding2D:
            padding = padding[1:]
            kwargs = {"data_format": data_format}
            shape = np.random.randint(low=2, high=4, size=4)
        elif op == tf.keras.layers.ZeroPadding3D:
            kwargs = {"data_format": data_format}
            shape = np.random.randint(low=2, high=4, size=5)
        model = tf.keras.Sequential(
            [op(batch_input_shape=shape, padding=padding, **kwargs)]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )

        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy(tf.keras.backend.floatx())


class TestPermute(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank_and_perm",
        itertools.product(
            compute_units,
            backends,
            [
                (rank, perm)
                for rank in range(3, 6)
                for perm in list(itertools.permutations(range(rank)[1:]))
            ],
        ),
    )
    def test(self, compute_unit, backend, rank_and_perm):
        rank, perm = rank_and_perm
        shape = np.random.randint(low=2, high=4, size=rank)
        model = tf.keras.Sequential(
            [tf.keras.layers.Permute(batch_input_shape=shape, dims=perm)]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )


class TestGlobalPooling(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, op, data_format",
        itertools.product(
            compute_units,
            backends,
            [
                tf.keras.layers.GlobalAveragePooling1D,
                tf.keras.layers.GlobalAveragePooling2D,
                tf.keras.layers.GlobalAveragePooling3D,
                tf.keras.layers.GlobalMaxPool1D,
                tf.keras.layers.GlobalMaxPool2D,
                tf.keras.layers.GlobalMaxPool3D,
            ],
            ["channels_first", "channels_last"],
        ),
    )
    def test_global_pooling(self, compute_unit, backend, op, data_format):
        shape = None
        if op in {
            tf.keras.layers.GlobalAveragePooling1D,
            tf.keras.layers.GlobalMaxPool1D,
        }:
            shape = np.random.randint(low=2, high=4, size=3)
        elif op in {
            tf.keras.layers.GlobalAveragePooling2D,
            tf.keras.layers.GlobalMaxPool2D,
        }:
            shape = np.random.randint(low=2, high=4, size=4)
        elif op in {
            tf.keras.layers.GlobalAveragePooling3D,
            tf.keras.layers.GlobalMaxPool3D,
        }:
            shape = np.random.randint(low=2, high=4, size=5)
        model = tf.keras.Sequential(
            [op(batch_input_shape=shape, data_format=data_format)]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )


class TestPooling(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, op, data_format, pool_size",
        itertools.product(
            compute_units,
            backends,
            [
                tf.keras.layers.AveragePooling1D,
                tf.keras.layers.AveragePooling2D,
                tf.keras.layers.AveragePooling3D,
                tf.keras.layers.MaxPool1D,
                tf.keras.layers.MaxPool2D,
                tf.keras.layers.MaxPool3D,
            ],
            ["channels_first", "channels_last"],
            [(2, 2, 1), (2, 3, 2), (1, 2, 3)],
        ),
    )
    def test_pooling(self, compute_unit, backend, op, data_format, pool_size):
        shape = None
        if op in {tf.keras.layers.AveragePooling1D, tf.keras.layers.MaxPool1D}:
            shape = np.random.randint(low=3, high=9, size=3)
            pool_size = pool_size[2]
        elif op in {tf.keras.layers.AveragePooling2D, tf.keras.layers.MaxPool2D}:
            if data_format == "channels_first":
                return  # AvgPoolingOp only supports NHWC on CPU
            shape = np.random.randint(low=3, high=9, size=4)
            pool_size = pool_size[1:]
        elif op in {tf.keras.layers.AveragePooling3D, tf.keras.layers.MaxPool3D}:
            shape = np.random.randint(low=3, high=9, size=5)
        model = tf.keras.Sequential(
            [op(batch_input_shape=shape, pool_size=pool_size, data_format=data_format)]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )


class TestRecurrent(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, units, activation, "
        "recurrent_activation, use_bias, return_sequences",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(3, 4)],
            [1, 3],
            [None, tf.nn.tanh],
            [None, tf.nn.relu],
            [True, False],
            [True, False],
        ),
    )
    def test_lstm(
        self,
        compute_unit,
        backend,
        rank,
        units,
        activation,
        recurrent_activation,
        use_bias,
        return_sequences,
    ):
        shape = np.random.randint(low=1, high=4, size=rank)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(
                    batch_input_shape=shape,
                    units=units,
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    use_bias=use_bias,
                    return_sequences=return_sequences,
                ),
            ]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-1, rand_max=1)],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_lstmcell(self, compute_unit, backend):
        shape = np.random.randint(low=1, high=4, size=3)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.RNN(
                    batch_input_shape=shape, cell=tf.keras.layers.LSTMCell(units=3)
                )
            ]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-1, rand_max=1)],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_lstm_time_distributed_dense(self, compute_unit, backend):
        shape = list(np.random.randint(low=1, high=4, size=3))
        k_in = tf.keras.layers.Input(batch_size=shape[0], shape=shape[1:])
        lstm = tf.keras.layers.LSTM(units=32, return_sequences=True)(k_in)
        k_out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(lstm)
        model = tf.keras.Model(inputs=k_in, outputs=k_out)

        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-1, rand_max=1)],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_lstm_dynamic_batch(self, compute_unit, backend):
        if backend == ("mlprogram", "fp16"):
            pytest.xfail(
                "rdar://116060011: re-activate coremltools tests blocked by Core ML regressions"
            )

        input_shape = (1, 1280)
        inp = tf.keras.layers.Input(shape=input_shape)
        out, hn, cn = tf.keras.layers.LSTM(512,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_activation='sigmoid')(inp)
        model = tf.keras.models.Model(inputs=[inp], outputs=[out, hn, cn])
        batch_size = 2
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen((batch_size, 1, 1280), -1, 1),],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_lstm_conversion_static_shapes(self, compute_unit, backend):
        '''
        Test that intermediate tensor shapes are populated correctly by the converter.
        That is, there are no symbolic dimensions in the shapes, when conversion is
        performed with a fixed input shape, irrespective of the shape used in the source model definition.
        '''
        def _get_keras_simple_lstm_model(input_shape):
            input = tf.keras.Input(batch_input_shape=input_shape)
            output = tf.keras.layers.LSTM(5)(input)
            keras_model = tf.keras.Model(inputs=input, outputs=output)
            return keras_model

        def _test_for_symbolic_shapes(keras_input_shape, input_shape_for_conversion, are_symbols_expected):
            keras_model = _get_keras_simple_lstm_model(keras_input_shape)
            res = TensorFlowBaseTest.run_compare_tf_keras(
                keras_model,
                [random_gen((1, 32, 10), -1, 1)],
                inputs_for_conversion=[ct.TensorType(shape=input_shape_for_conversion)],
                compute_unit=compute_unit,
                backend=backend,
            )
            coreml_model = res[1]
            mil_prog = coreml_model._get_mil_internal()
            assert is_symbolic_dim_in_prog(mil_prog) == are_symbols_expected

        _test_for_symbolic_shapes(keras_input_shape=(1, 32, 10),
                                  input_shape_for_conversion=(1, 32, 10),
                                  are_symbols_expected=False)

        _test_for_symbolic_shapes(keras_input_shape=(None, 32, 10),
                                  input_shape_for_conversion=(1, 32, 10),
                                  are_symbols_expected=False)

        _test_for_symbolic_shapes(keras_input_shape=(None, None, 10),
                                  input_shape_for_conversion=(1, 32, 10),
                                  are_symbols_expected=False)

        _test_for_symbolic_shapes(keras_input_shape=(None, 32, 10),
                                  input_shape_for_conversion=(ct.RangeDim(1, 10), 32, 10),
                                  are_symbols_expected=True)

        if backend[0] != "mlprogram":
            # FIX ME: model load fails if backend is "mlprogram". rdar://84862138
            _test_for_symbolic_shapes(keras_input_shape=(None, None, 10),
                                      input_shape_for_conversion=(ct.RangeDim(1, 10), ct.RangeDim(16, 64), 10),
                                      are_symbols_expected=True)

    @pytest.mark.parametrize(
        "compute_unit, tf_raw_lstm_op, is_flexible_input, batch_size, backend",
        itertools.product(
            compute_units,
            [
                tf.raw_ops.BlockLSTMV2,
                tf.raw_ops.BlockLSTM,
            ],
            [False, True],
            [1, 2],
            backends,
        ),
    )
    def test_lstm_block_fused_op(
        self, compute_unit, tf_raw_lstm_op, is_flexible_input, batch_size, backend
    ):
        """
        Define a model with custom LSTM ops that uses tf.raw_ops.BlockLSTM / tf.raw_ops.BlockLSTMV2
        and verify that it converts to a fused lstm op.

        %x (shape: (Seq, Batch, idim) == (seq_len, batch, 4))
        %x1 = LSTM(h=10) (%input) # shape = (seq_len, batch, 10)
        %x2 = LSTM(h=20) (%x1) # shape = (seq_len, batch, 20)
        %x3 = slice()(%x2) # shape = (1, batch, 20), to get the final seq value
        %x4 = reshape((1, -1)) (%x3) # shape = (1, batch * 20)
        %x5 = Dense(h=3)(%x4) # shape = (1, 3)
        """

        class CustomLSTM(tf.keras.layers.Layer):
            def __init__(self, num_units, max_seq_length, batch_size):
                super(CustomLSTM, self).__init__()
                self.hidden_dim = num_units
                self.seq_length = max_seq_length
                self.batch_size = batch_size

            def build(self, input_shape):
                input_dim = input_shape[-1]
                self.w = self.add_weight(
                    shape=(input_dim + self.hidden_dim, 4 * self.hidden_dim),
                    initializer="random_normal",
                    trainable=True,
                )
                self.b = self.add_weight(shape=(4 * self.hidden_dim,), initializer="random_normal", trainable=True)
                self.init_h = tf.constant(np.zeros((self.batch_size, self.hidden_dim)).astype(np.float32))
                self.init_c = tf.constant(np.zeros((self.batch_size, self.hidden_dim)).astype(np.float32))

            def call(self, inputs):
                _, output_state, _, _, _, _, output = tf_raw_lstm_op(
                    seq_len_max=self.seq_length,
                    x=inputs,
                    cs_prev=self.init_c,
                    h_prev=self.init_h,
                    w=self.w,
                    wci=tf.constant(np.zeros((self.hidden_dim)).astype(np.float32)),
                    wcf=tf.constant(np.zeros((self.hidden_dim)).astype(np.float32)),
                    wco=tf.constant(np.zeros((self.hidden_dim)).astype(np.float32)),
                    b=self.b,
                )
                return output, output_state

        input_dim = 4
        seq_length = 5
        batch_size = batch_size
        x_shape = (seq_length, batch_size, input_dim)
        hidden_dim_1 = 10
        hidden_dim_2 = 20

        x = tf.keras.Input(batch_input_shape=x_shape)  # (seq_len, batch, 4)
        x1, output_states_1 = CustomLSTM(num_units=hidden_dim_1, max_seq_length=seq_length, batch_size=batch_size)(x)  # (seq_len, batch, 10), (seq_len, batch, 10)
        x2, output_states_2 = CustomLSTM(num_units=hidden_dim_2, max_seq_length=seq_length, batch_size=batch_size)(x1)  # (seq_len, batch, 20), (seq_len, batch 10)
        x3 = tf.slice(x2, begin=[4, 0, 0], size=[1, batch_size, 20])  # (1, batch, 20)
        x4 = tf.reshape(x3, shape=(1, -1))  # (1, batch * 20)
        x5 = tf.keras.layers.Dense(3)(x4)  # (1, 3)

        # Test that we can fuse the lstm op if we have an output that only extract the information from the last cell state
        x6 = tf.keras.layers.ReLU()(output_states_1[4, :, :])
        x7 = output_states_2[4:5, :, :]
        x8 = output_states_1[-1, :, :]
        x9 = tf.keras.layers.ReLU()(output_states_2[-1:, :, :])
        outputs = [x5, x8, x9] if is_flexible_input else [x5, x6, x7, x8, x9]

        keras_model = tf.keras.Model(inputs=x, outputs=outputs)

        inputs = None
        if is_flexible_input:
            inputs = [
                ct.TensorType(
                    shape=(ct.RangeDim(seq_length, 20), batch_size, input_dim)
                )
            ]

        res = TensorFlowBaseTest.run_compare_tf_keras(
            keras_model,
            [random_gen(x_shape, -1, 1)],
            compute_unit=compute_unit,
            backend=backend,
            inputs_for_conversion=inputs,
        )
        coreml_model = res[1]
        mil_prog = coreml_model._get_mil_internal()
        # assert that "lstm" ops are present in the mil program
        assert len(mil_prog.find_ops(op_type="lstm")) == 2


class TestRepeatVector(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, n",
        itertools.product(
            compute_units,
            backends,
            [2, 3, 5, 7],
        ),
    )
    def test(self, compute_unit, backend, n):
        # input shape 2D tensor (batch size, features)
        # output shape 3D tensor (batch size, n, features)
        shape = np.random.randint(low=1, high=4, size=2)
        model = tf.keras.Sequential(
            [tf.keras.layers.RepeatVector(batch_input_shape=shape, n=n)]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )


class TestReshape(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, infer_shape",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(1, 6)],
            [True, False],
        ),
    )
    def test(self, compute_unit, backend, rank, infer_shape):
        shape = np.random.randint(low=2, high=4, size=rank)
        # target shape does not include the batch dimension
        target_shape = random.sample(list(shape[1:]), len(shape[1:]))
        if len(target_shape) > 0 and infer_shape:
            target_shape[-1] = -1
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Reshape(
                    batch_input_shape=shape, target_shape=target_shape
                )
            ]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )


class TestSkips(TensorFlowBaseTest):
    # ops in this class should be ignored / pass-through during conversion

    @pytest.mark.parametrize(
        "compute_unit, backend, skip_op",
        itertools.product(
            compute_units,
            backends,
            [
                tf.keras.layers.Dropout,
                tf.keras.layers.AlphaDropout,
                tf.keras.layers.GaussianDropout,
                tf.keras.layers.SpatialDropout1D,
                tf.keras.layers.SpatialDropout2D,
                tf.keras.layers.SpatialDropout3D,
            ],
        ),
    )
    def test_skip_dropout(self, compute_unit, backend, skip_op):
        shape = np.random.randint(low=1, high=4, size=5)
        if skip_op == tf.keras.layers.SpatialDropout1D:
            shape = shape[:3]
        elif skip_op == tf.keras.layers.SpatialDropout2D:
            shape = shape[:4]
        model = tf.keras.Sequential([skip_op(batch_input_shape=shape, rate=0.5)])
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends,)
    )
    def test_skip_noise(self, compute_unit, backend):
        shape = np.random.randint(low=1, high=4, size=5)
        model = tf.keras.Sequential(
            [
                # GaussianNoise should do nothing in inference mode
                tf.keras.layers.GaussianNoise(batch_input_shape=shape, stddev=0.5)
            ]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, rank, l1, l2",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(5, 6)],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ),
    )
    def test_skip_regularization(self, compute_unit, backend, rank, l1, l2):
        shape = np.random.randint(low=2, high=4, size=rank)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.ActivityRegularization(
                    batch_input_shape=shape, l1=l1, l2=l2
                )
            ]
        )
        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-10, rand_max=10)],
            compute_unit=compute_unit,
            backend=backend,
        )


class TestUpSampling(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, op, upsample_factor, data_format, interpolation, dynamic",
        itertools.product(
            compute_units,
            backends,
            [
                tf.keras.layers.UpSampling1D,
                tf.keras.layers.UpSampling2D,
                tf.keras.layers.UpSampling3D,
            ],
            [(2, 2, 1), (4, 3, 2), (1, 2, 3)],
            ["channels_first", "channels_last"],
            ["nearest", "bilinear"],
            [True, False],
        ),
    )
    def test(
        self, compute_unit, backend, op, upsample_factor, data_format, interpolation, dynamic
    ):
        kwargs = {}
        shape = None
        keras_shape = None

        if op == tf.keras.layers.UpSampling1D:
            shape = np.random.randint(low=2, high=4, size=3)
            keras_shape = np.copy(shape).tolist()
            if dynamic:
                keras_shape[1] = None
            upsample_factor = upsample_factor[2]
        elif op == tf.keras.layers.UpSampling2D:
            kwargs = {"data_format": data_format, "interpolation": interpolation}
            shape = np.random.randint(low=2, high=4, size=4)
            keras_shape = np.copy(shape).tolist()
            if dynamic:
                keras_shape[1] = keras_shape[2] = None
            upsample_factor = (upsample_factor[1], upsample_factor[2])
        elif op == tf.keras.layers.UpSampling3D:
            kwargs = {"data_format": data_format}
            shape = np.random.randint(low=2, high=4, size=5)
            keras_shape = np.copy(shape).tolist()
            if dynamic:
                pytest.skip(
                    "upsampling3D with dynamic input shape is not supported, since 6D tensors are produced in that case"
                )

        inputs_for_conversion = None
        if backend[0] == "mlprogram" and dynamic:
            inputs_for_conversion = [
                ct.TensorType(shape=[dim or ct.RangeDim(upper_bound=10) for dim in keras_shape])
            ]

        model = tf.keras.Sequential(
            [op(batch_input_shape=keras_shape, size=upsample_factor, **kwargs)]
        )
        spec = TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, rand_min=-10, rand_max=10)],
            inputs_for_conversion=inputs_for_conversion,
            compute_unit=compute_unit,
            backend=backend,
        )[0]
        # also check if the scale factor are integers
        if backend[0] == 'neuralnetwork':
            for layer in spec.neuralNetwork.layers:
                if layer.WhichOneof('layer') == "upsample":
                    assert len(layer.upsample.fractionalScalingFactor) == 0

class TestGelu(TensorFlowBaseTest):
    @pytest.mark.skipif(
        _get_version(_tf.__version__) < Version("2.4.0"),
        reason="Gelu is a new layer for tf 2.4.0 and above."
    )
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, approximate",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(1, 6)],
            [True, False],
        ),
    )
    def test(
        self, compute_unit, backend, rank, approximate
    ):
        shape = np.random.randint(low=2, high=4, size=rank)
        input = tf.keras.layers.Input(batch_input_shape=tuple(shape))
        out = tf.keras.activations.gelu(input, approximate=approximate)
        model = tf.keras.Model(inputs=[input], outputs=out)

        TensorFlowBaseTest.run_compare_tf_keras(
            model,
            [random_gen(shape, -10, 10)],
            compute_unit=compute_unit,
            backend=backend,
        )
