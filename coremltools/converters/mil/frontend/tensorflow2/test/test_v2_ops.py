#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.frontend.tensorflow2.test.testing_utils import \
    TensorFlow2BaseTest
from coremltools.converters.mil.frontend.tensorflow2.test.testing_utils import \
    make_tf2_graph as make_tf_graph
from coremltools.converters.mil.frontend.tensorflow.test.testing_utils import \
    TensorFlowBaseTest
from coremltools.converters.mil.testing_utils import random_gen

TensorFlowBaseTest.run_compare_tf = TensorFlow2BaseTest.run_compare_tf2

tf = pytest.importorskip("tensorflow", minversion="2.1.0")

backends = testing_reqs.backends
compute_units = testing_reqs.compute_units

class TestImageResample(TensorFlowBaseTest):
    @pytest.mark.skip(
        "TODO: rdar://100812753 ([TF] [Infra] TensorFlow Addons dylib issues in TF 2.10.0)"
    )
    @pytest.mark.parametrize(
        "compute_unit, backend, data_warp_shapes",
        itertools.product(
            compute_units,
            backends,
            [
                # Data shape format: (Batch, Hin, Win, C)
                # Warp shape format: (Batch, Hout, Wout, 2)
                [(1, 3, 3, 1), (1, 3, 3, 2)],  # no size change
                [(2, 5, 5, 3), (2, 3, 3, 2)],  # down-sampling
                [(3, 6, 6, 1), (3, 8, 8, 2)],  # up-sampling
            ],
        ),
    )
    def test_resample(
        self, compute_unit, backend, data_warp_shapes,
    ):
        if backend[0] == "neuralnetwork":
            pytest.skip("nn backend not supported")

        tfa = pytest.importorskip("tensorflow_addons")

        data_shape, warp_shape = data_warp_shapes

        @make_tf_graph([data_shape, warp_shape])
        def build_model(x, warp):
            return tfa.image.resampler(data=x, warp=warp)

        model, inputs, outputs = build_model
        # warp exceeding input sizes in order to test more padding modes
        input_values = [
            random_gen(data_shape, -100, 100),
            random_gen(warp_shape, -15, 15),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestImageTransform(TensorFlowBaseTest):
    @pytest.mark.skip(
        "TODO: rdar://73165549 (Add other mode in 'affine' to coremltools when backend is ready)"
    )
    @pytest.mark.parametrize(
        "compute_unit, backend, transforms, interpolation, shapes",
        itertools.product(
            [True],
            backends,
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, -250, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.25, -1.75, 25.0, -25.0, 1.5, -1.5, 0.0, 0.0],
            ],
            ["BILINEAR"],
            [
                ((1, 2, 2, 1), None),
                ((2, 2, 2, 1), (2, 3)),
                ((3, 5, 5, 2), (4, 4)),
                ((1, 3, 3, 2), (6, 6)),
                ((3, 50, 50, 2), (20, 20)),
            ],
        ),
    )
    def test(self, compute_unit, backend, transforms, interpolation, shapes):
        x_shape, output_shape = shapes
        if backend[0] == "neuralnetwork":
            pytest.skip("nn backend not supported")

        tfa = pytest.importorskip("tensorflow_addons")

        @make_tf_graph([x_shape])
        def build_model(x):
            return tfa.image.transform(
                x,
                transforms=transforms,
                interpolation=interpolation,
                output_shape=output_shape,
            )

        model, inputs, outputs = build_model
        input_values = [
            random_gen(x_shape, -100, 100),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, compute_unit=compute_unit, backend=backend,
        )
        
    @pytest.mark.parametrize(
        "compute_unit, backend, InputShape_OutputShape, op",
        itertools.product(
            compute_units,
            backends,
            [
                [(2, 5, 15, 3), (2, 5, 15, 3)],
                [(2, 4, 8, 5), (2, 2, 4, 5)],
                [(2, 4, 8, 3), (2, 9, 13, 3)],
            ],
            ["V2", "V3"],
        ),
    )
    def test_affine_transform(self, compute_unit, backend, InputShape_OutputShape, op):
        if backend[0] == "neuralnetwork":
            pytest.skip("Affine op not available in the neuralnetwork backend")
            
        input_shape, output_shape = InputShape_OutputShape
        batch_size = input_shape[0]
        transforms = np.random.rand(batch_size, 8) - 0.05
        transforms[:, 6:8] = 0

        @make_tf_graph([input_shape])
        def build_model(x):
            if op == "V2":
                return tf.raw_ops.ImageProjectiveTransformV2(
                    images=x,
                    transforms=transforms,
                    fill_mode="CONSTANT",
                    output_shape=(output_shape[0], output_shape[1]),
                    interpolation="BILINEAR",
                )
            elif op == "V3":
                return tf.raw_ops.ImageProjectiveTransformV3(
                    images=x,
                    transforms=transforms,
                    fill_mode="CONSTANT",
                    output_shape=(output_shape[0], output_shape[1]),
                    interpolation="BILINEAR",
                    fill_value=0.0,
                )
            else:
                raise ValueError("tensorflow op {} not supported".format(op))
                
        model, inputs, outputs = build_model
        input_values = [np.random.rand(*input_shape).astype(np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestActivationSiLU(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, tf_op",
        itertools.product(
            compute_units,
            backends,
            list(range(1, 6)),
            [
                tf.nn.swish,  # TODO(yuduo): in TF 2.4.0+, it's renamed to tf.nn.silu,
                tf.keras.activations.swish,
            ],
        ),
    )
    def test(self, compute_unit, backend, rank, tf_op):
        if backend[0] == "neuralnetwork":
            pytest.skip("nn backend not supported")

        x_shape = tuple(np.random.randint(low=1, high=4, size=rank))

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf_op(x)

        model, inputs, outputs = build_model
        input_values = [
            random_gen(x_shape, -100, 100),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestResizeNearestNeighbor(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, input_shape, target_shape, align_corners, half_pixel_centers",
        itertools.product(
            compute_units,
            backends,
            [(1, 10, 20, 1), (2, 5, 1, 3)],
            [(25, 30), (2, 20)],
            [False],
            [True, False],
        ),
    )
    def test_raw_ops(
        self,
        compute_unit,
        backend,
        input_shape,
        target_shape,
        align_corners,
        half_pixel_centers,
    ):
        if align_corners is True and half_pixel_centers is True:
            return

        if backend[0] == "neuralnetwork":
            # neural network backend does not support fractional scale factors for nearest neighbor upsample op
            if target_shape[-1] % input_shape[-1] != 0:
                return
            if target_shape[-2] % input_shape[-2] != 0:
                return
                
        if backend[0] == "mlprogram" and compute_unit != ct.ComputeUnit.CPU_ONLY and not half_pixel_centers:
            pytest.xfail("rdar://97399545 (TestResizeNearestNeighbor failing on mlprogram + GPU + half_pixel_centers=False)")

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.raw_ops.ResizeNearestNeighbor(
                images=x,
                size=target_shape,
                align_corners=align_corners,
                half_pixel_centers=half_pixel_centers,
            )

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape, -100, 100)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, size",
        itertools.product(compute_units, backends, [(1, 1), (2, 3), (4, 1)]),
    )
    def test_keras_layer(self, compute_unit, backend, size):
        if backend[0] == "neuralnetwork":
            pytest.skip("nn backend not supported")

        x_shape = tuple(np.random.randint(low=1, high=4, size=4))

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.keras.layers.UpSampling2D(
                size=size, interpolation="nearest",
            )(x)

        model, inputs, outputs = build_model
        input_values = [random_gen(x_shape, -100, 100)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, size, method",
        itertools.product(
            compute_units,
            backends,
            [(1, 1), (2, 3)],
            [tf.image.ResizeMethod.NEAREST_NEIGHBOR],
        ),
    )
    def test_tf_image_resize(self, compute_unit, backend, size, method):
        if backend[0] == "mlprogram" and size == (1, 1):
            pytest.xfail("rdar://79699954 (Nearest neighbor resize numerical mismatch when output size is (1,1))")

        if backend[0] == "neuralnetwork":
            pytest.skip("nn backend not supported")

        x_shape = tuple(np.random.randint(low=1, high=3, size=4))

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.image.resize(x, size=size, method=method)

        model, inputs, outputs = build_model
        input_values = [
            random_gen(x_shape, -100, 100),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestNormalizationTF2(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, func, backend, epsilon",
        itertools.product(
            compute_units,
            [tf.raw_ops.FusedBatchNorm, tf.raw_ops.FusedBatchNormV3],
            backends,
            [1e-1, 1e-10]
        ),
    )
    def test_fused_batch_norm(self, compute_unit, func, backend, epsilon):
        input_shape = np.random.randint(low=1, high=4, size=4)
        attr_shape = [list(input_shape)[-1]]

        m = random_gen(shape=attr_shape, rand_min=-1.0, rand_max=1.0)
        v = random_gen(shape=attr_shape, rand_min=0.0, rand_max=10.0)
        o = random_gen(shape=attr_shape, rand_min=1.0, rand_max=10.0)
        s = random_gen(shape=attr_shape, rand_min=-1.0, rand_max=1.0)

        @make_tf_graph([input_shape])
        def build_model(x):
            return func(
                x=x,
                scale=s,
                offset=o,
                mean=m,
                variance=v,
                epsilon=epsilon,
                is_training=False,
            )[0]

        model, inputs, outputs = build_model
        input_values = [random_gen(shape=input_shape)]
        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend,
            atol=1e-2,
            rtol=1e-3,
        )


class TestElementWiseBinaryTF2(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(compute_units, backends, [rank for rank in range(1, 4)]),  # False
    )
    def test_add_v2(self, compute_unit, backend, rank):
        x_shape = list(np.random.randint(low=2, high=5, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        @make_tf_graph([x_shape, y_shape])
        def build_model(x, y):
            return tf.raw_ops.AddV2(x=x, y=y)

        model, inputs, outputs = build_model

        input_values = [
            np.random.randint(low=-1, high=1, size=x_shape).astype(np.float32),
            np.random.randint(low=-1, high=1, size=y_shape).astype(np.float32),
        ]

        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend
        )


class TestControlFlowFromAutoGraph(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_if_unary_const(self, compute_unit, backend):
        @make_tf_graph([(1,)])
        def build_model(x):
            if x > 0.5:
                y = x - 0.5
            else:
                y = x + 0.5
            return y

        model, inputs, outputs = build_model
        input_values = [np.array([0.7], dtype=np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend
        )

    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_if_unary_double_if_positive_else_square(self, compute_unit, backend):
        @make_tf_graph([(1,)])
        def build_model(x):
            if x >= 0:
                out = x + x
            else:
                out = x * x
            return out

        model, inputs, outputs = build_model
        input_values = [np.array([2], dtype=np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend
        )

    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_if_binary_add_if_else_mul(self, compute_unit, backend):
        @make_tf_graph([(1,), (1,)])
        def build_model(x, y):
            if x > y:
                out = x + x
            else:
                out = x * x
            return out

        model, inputs, outputs = build_model
        input_values = [
            np.array([3], dtype=np.float32),
            np.array([7], dtype=np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend
        )

    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_while_loop_square(self, compute_unit, backend):
        @make_tf_graph([(1,)])
        def build_model(x):
            i = 0
            while i < 10:
                x *= 2
                i += 1
            return x

        model, inputs, outputs = build_model
        input_values = [np.array([2.0], dtype=np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend
        )

    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_while_loop_power(self, compute_unit, backend):
        @make_tf_graph([(1,)])
        def build_model(x):
            i = 0
            while i < 3:
                x *= x
                i += 1
            return x

        model, inputs, outputs = build_model
        input_values = [np.array([2.0], dtype=np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend
        )

    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_while_loop_nested_body(self, compute_unit, backend):
        @make_tf_graph([(1,)])
        def build_model(x):
            i, j = 0, 10
            while i < j:
                while 2 * i < i + 2:
                    i += 1
                    x -= 1
                i += 2
                x *= 2
            return x

        model, inputs, outputs = build_model
        input_values = [np.array([9.0], dtype=np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend
        )

@pytest.mark.xfail(reason="rdar://76293949 (TF2 unit test InvalidArgumentError)", run=False)
class TestTensorList(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, size_dynamic_shape",
        itertools.product(
            compute_units,
            backends,
            [
                (1, True, None),
                (1, True, (1,)),
                (2, False, (1,))
            ],
        ),
    )
    def test_write_read_and_stack(self, compute_unit, backend, size_dynamic_shape):
        size, dynamic_size, element_shape = size_dynamic_shape

        @make_tf_graph([(1,), (1,)])
        def build_model(x, y):
            ta = tf.TensorArray(
                tf.float32,
                size=size,
                dynamic_size=dynamic_size,
                element_shape=element_shape,
            )
            ta = ta.write(0, x)
            ta = ta.write(1, y)
            return ta.read(0), ta.read(1), ta.stack()

        model, inputs, outputs = build_model
        input_values = [
            np.array([3.14], dtype=np.float32),
            np.array([6.17], dtype=np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, size_dynamic_shape",
        itertools.product(
            compute_units,
            backends,
            [
                (0, True, None),
                (1, True, (1,)),
                (3, False, (1,))
            ],
        ),
    )
    def test_unstack_and_read(self, compute_unit, backend, size_dynamic_shape):
        size, dynamic_size, element_shape = size_dynamic_shape

        @make_tf_graph([(3, 1)])
        def build_model(x):
            ta = tf.TensorArray(
                tf.float32,
                size=size,
                dynamic_size=dynamic_size,
                element_shape=element_shape,
            )
            ta = ta.unstack(x)
            return ta.read(0), ta.read(1), ta.read(2)

        model, inputs, outputs = build_model
        input_values = [np.array([[3.14], [6.17], [12.14]], dtype=np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, size_dynamic_shape",
        itertools.product(
            compute_units,
            backends,
            [
                (2, True, None),
                (1, True, (1,)),
                (3, False, (1,))
            ],
        ),
    )
    def test_write_and_gather(self, compute_unit, backend, size_dynamic_shape):
        size, dynamic_size, element_shape = size_dynamic_shape

        @make_tf_graph([(1,), (1,)])
        def build_model(x, y):
            ta = tf.TensorArray(
                tf.float32,
                size=size,
                dynamic_size=dynamic_size,
                element_shape=element_shape,
            )
            ta = ta.write(0, x)
            ta = ta.write(1, y)
            return ta.gather(indices=[0, 1])

        model, inputs, outputs = build_model
        input_values = [
            np.array([3.14], dtype=np.float32),
            np.array([6.17], dtype=np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, size_dynamic_shape",
        itertools.product(
            compute_units,
            backends,
            [
                (2, True, None),
                (1, True, (1,)),
                (3, False, (1,))
            ],
        ),
    )
    def test_scatter_and_read(self, compute_unit, backend, size_dynamic_shape):
        size, dynamic_size, element_shape = size_dynamic_shape

        @make_tf_graph([(3, 1)])
        def build_model(x):
            ta = tf.TensorArray(
                tf.float32,
                size=size,
                dynamic_size=dynamic_size,
                element_shape=element_shape,
            )
            ta = ta.scatter(indices=[0, 1, 2], value=x)
            return ta.read(0), ta.read(1), ta.read(2)

        model, inputs, outputs = build_model
        input_values = [np.array([[3.14], [6.17], [12.14]], dtype=np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, size_dynamic_shape",
        itertools.product(compute_units, backends, [(2, False, (None, 8))]),
    )
    def test_partial_element_shape(self, compute_unit, backend, size_dynamic_shape):
        size, dynamic_size, element_shape = size_dynamic_shape

        @make_tf_graph([(3, 1, 8)])
        def build_model(x):
            ta = tf.TensorArray(
                tf.float32,
                size=size,
                dynamic_size=dynamic_size,
                element_shape=element_shape,
            )
            ta = ta.scatter(indices=[0, 1, 2], value=x)
            return ta.read(0), ta.read(1), ta.read(2)

        model, inputs, outputs = build_model
        input_values = [np.random.rand(3, 1, 8).astype(np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend
        )


class TestPartitionedCall(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_partitioned_call_optimized_to_add_op(self, compute_unit, backend):
        """
        The PartitionedCall will be optimized to V2Add op in TF's internal optimization pass (see
        `_run_inline_graph_optimization`), so this test passes even when we haven't implemented
        the `PartitionedCall` op).
        """
        x_shape = [2, 3]
        y_shape = [2, 3]

        @tf.function
        def simple_func(*args):
            output = [args[0] + args[1]]
            return output

        @make_tf_graph([x_shape, y_shape])
        def build_model(x, y):
            return tf.raw_ops.PartitionedCall(
                args=[x, y],
                f=simple_func.get_concrete_function(tf.zeros(x_shape), tf.zeros(y_shape)),
                Tout=[tf.float32]
            )

        model, inputs, outputs = build_model

        input_values = [
            np.zeros(x_shape).astype(np.float32),
            np.zeros(y_shape).astype(np.float32),
        ]

        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=compute_unit,
            backend=backend
        )
