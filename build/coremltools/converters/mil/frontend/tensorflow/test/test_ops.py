#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.testing_reqs import *
from coremltools.converters.mil.frontend.tensorflow.test.testing_utils import (
    make_tf_graph,
    layer_counts,
    load_tf_pb,
    freeze_g,
    TensorFlowBaseTest,
    tf_graph_to_mlmodel
)
from coremltools.models.utils import _macos_version

import math
import tempfile
import shutil

backends = testing_reqs.backends

tf = pytest.importorskip("tensorflow")


class TestContribResampler(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, data_warp_shapes",
        itertools.product(
            [True, False],
            backends,
            [
                # Data shape format: (Batch, Hin, Win, C)
                # Warp shape format: (Batch, Hout, Wout, 2)
                [(1, 3, 3, 1), (1, 3, 3, 2)],  # no size change
                [(2, 5, 5, 3), (2, 3, 3, 2)],  # down-sampling
                [(3, 6, 6, 1), (3, 8, 8, 2)],  # up-sampling
                [(1, 3, 9, 1), (1, 19, 2)],  # rank-3 warp tensor
            ],
        ),
    )
    def test(
        self, use_cpu_only, backend, data_warp_shapes,
    ):
        if backend[0] == "neuralnetwork":
            pytest.xfail("nn backend not supported")

        data_shape, warp_shape = data_warp_shapes

        @make_tf_graph([data_shape, warp_shape])
        def build_model(x, warp):
            return tf.contrib.resampler.resampler(data=x, warp=warp)

        model, inputs, outputs = build_model
        # warp exceeding input sizes in order to test more padding modes
        input_values = [
            random_gen(data_shape, -100, 100),
            random_gen(warp_shape, -15, 15),
        ]
        input_dict = dict(zip(inputs, input_values))
        self.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend,
        )


class TestDebugging(TensorFlowBaseTest):
    """
    TF converter does not handling debugging nodes, they are
    expected to be deleted by graph pass before op conversions
    in Grappler graph pass: debug_stripper.
    """

    @pytest.mark.parametrize(
        "use_cpu_only, backend",
        itertools.product([True, False], backends),
    )
    def test_assert(self, use_cpu_only, backend):
        input_shape = (1,)

        @make_tf_graph([input_shape])
        def build_model(x):
            tf.debugging.Assert(True, [x])
            return tf.nn.relu(x)

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape, 0, 1)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend",
        itertools.product([True, False], backends),
    )
    def test_check_numerics(self, use_cpu_only, backend):
        input_shape = (1,)

        @make_tf_graph([input_shape])
        def build_model(x):
            tf.debugging.check_numerics(x, 'check')
            return tf.nn.relu(x)

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape, 0, 1)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend",
        itertools.product([True, False], backends),
    )
    def test_print(self, use_cpu_only, backend):
        input_shape = (1,)

        @make_tf_graph([input_shape])
        def build_model(x):
            tf.raw_ops.Print(input=x, data=[x], message='[x]')
            return tf.nn.relu(x)

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape, 0, 1)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )


class TestPlaceholderAsOutput(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [rank for rank in range(6)]),
    )
    def test(self, use_cpu_only, backend, rank):
        if rank == 0:
            pytest.xfail('Rank 0 not supported by CoreML runtime')
        input_shape = np.random.randint(low=1, high=4, size=rank)

        @make_tf_graph([input_shape, input_shape])
        def build_model(x, y):
            return x, y, x + 1, x + y

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape, -1, 1), random_gen(input_shape, -1, 1)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestDuplicateOutputs(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [rank for rank in range(6)]),
    )
    def test(self, use_cpu_only, backend, rank):
        if rank == 0:
            pytest.xfail('Rank 0 not supported by CoreML runtime')
        input_shape = np.random.randint(low=1, high=4, size=rank)

        @make_tf_graph([input_shape])
        def build_model(x):
            b = tf.identity(x)
            c = tf.identity(x)
            d = b + c
            return b, c, d

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape, -1, 1), random_gen(input_shape, -1, 1)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestIdentity(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [rank for rank in range(6)]),
    )
    def test(self, use_cpu_only, backend, rank):
        if rank == 0:
            pytest.xfail('Rank 0 not supported by CoreML runtime')

        input_shape = np.random.randint(low=1, high=4, size=rank)
        @make_tf_graph([input_shape])
        def build_model(x):
            return x

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape, -1, 1)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestActivationElu(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [rank for rank in range(1, 6)]),
    )
    def test(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=1, high=4, size=rank)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.nn.elu(x)

        model, inputs, outputs = build_model

        input_values = [random_gen(input_shape, -1, 1)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestAddN(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank, num_inputs",
        itertools.product([True, False], backends, list(range(6)), [1, 3, 9],),
    )
    def test(self, use_cpu_only, backend, rank, num_inputs):
        if rank == 0:
            pytest.xfail('Rank 0 not supported by CoreML runtime')

        input_shape = np.random.randint(low=1, high=4, size=rank)
        input_shapes = [input_shape[:] for _ in range(num_inputs)]

        @make_tf_graph(input_shapes)
        def build_model(*inputs):
            return tf.raw_ops.AddN(inputs=inputs)

        model, inputs, outputs = build_model
        input_values = [random_gen(shape, -1, 1) for shape in input_shapes]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestAddOrdering(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend",
        itertools.product([True, False], backends),
    )
    def test(self, use_cpu_only, backend):
        if backend[0] == "mlprogram":
            pytest.xfail("Not supported on ML Program backend")

        @make_tf_graph([(2, 3, 4), (2, 3, 4)])
        def build_model(x, y):
            return tf.math.add(x, y)

        model, inputs, outputs = build_model
        input_values = [random_gen((2, 3, 4), -1, 1)] * 2
        input_dict = dict(zip(inputs, input_values))

        mlmodel, input_key_values, output_names, output_nodes = tf_graph_to_mlmodel(
            model, input_dict, outputs, "tensorflow", backend 
        )
        nn_spec = mlmodel.get_spec().neuralNetwork
        assert nn_spec.layers[0].input[0] == "Placeholder"
        assert nn_spec.layers[0].input[1] == "Placeholder_1"


class TestActivationLeakyReLU(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [rank for rank in range(1, 6)]),
    )
    def test(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=1, high=4, size=rank)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.nn.leaky_relu(x, 0.2)

        model, inputs, outputs = build_model

        input_values = [random_gen(input_shape, -1, 1)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestActivationReLU(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [rank for rank in range(1, 6)]),
    )
    def test(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=1, high=4, size=rank)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.nn.relu(x)

        model, inputs, outputs = build_model

        input_values = [random_gen(input_shape, -10.0, 10)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestActivationReLU6(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [rank for rank in range(1, 6)]),
    )
    def test(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=1, high=4, size=rank)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.nn.relu6(x)

        model, inputs, outputs = build_model

        input_values = [random_gen(input_shape, -1, 1)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestGeluTanhApproximation(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True], backends, [rank for rank in range(2, 3)]),
    )
    def test(self, use_cpu_only, backend, rank):
        if backend[0] == 'mlprogram':
            pytest.xfail("Not supported with ML Program backend")

        input_shape = np.random.randint(low=1, high=4, size=rank)

        @make_tf_graph([input_shape])
        def build_model(x):
            a = 0.5 * (
                1.0 + tf.tanh((math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))
            )
            return a * x

        model, inputs, outputs = build_model

        input_values = [random_gen(input_shape, -5, 5)]
        input_dict = dict(zip(inputs, input_values))
        spec, _, _, _, _, _ = TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )
        assert len(spec.neuralNetwork.layers) == 1
        assert spec.neuralNetwork.layers[0].WhichOneof("layer") == "gelu"


class TestActivationSigmoid(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [rank for rank in range(1, 6)]),
    )
    def test(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=1, high=4, size=rank)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.math.sigmoid(x)

        model, inputs, outputs = build_model

        input_values = [random_gen(input_shape, -1, 1)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestActivationSoftPlus(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [rank for rank in range(1, 6)]),
    )
    def test(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=1, high=4, size=rank)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.math.softplus(x)

        model, inputs, outputs = build_model

        input_values = [random_gen(input_shape, -1, 1)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestActivationSoftmax(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank_and_axes",
        itertools.product(
            [True, False],
            backends,
            [(rank, axis) for rank in range(1, 6) for axis in range(-1, rank)],
        ),
    )
    def test(self, use_cpu_only, backend, rank_and_axes):
        rank, axis = rank_and_axes
        input_shape = np.random.randint(low=1, high=4, size=rank)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.nn.softmax(x, axis=axis)

        model, inputs, outputs = build_model

        input_values = [random_gen(input_shape, -1, 1)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestActivationSoftSign(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [rank for rank in range(1, 6)]),
    )
    def test(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=1, high=4, size=rank)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.math.softsign(x)

        model, inputs, outputs = build_model

        input_values = [random_gen(input_shape, -1, 1)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestActivationSelu(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [rank for rank in range(1, 6)]),
    )
    def test(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=1, high=4, size=rank)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.nn.selu(x)

        model, inputs, outputs = build_model

        input_values = [random_gen(input_shape, -1.0, 1.0)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestSelect(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend, rank, broadcast, dynamic",
        itertools.product(
            [True, False],
            backends,
            [rank for rank in range(1, 6)],
            [True, False],
            [True, False],
        ),
    )
    def test_select(self, use_cpu_for_conversion, backend, rank, broadcast, dynamic):
        if backend[0] == "mlprogram" and broadcast and not use_cpu_for_conversion:
            # use_cpu_for_conversion == False and broadcast == False fails with MIL
            pytest.xfail("rdar://77441455")

        shape = np.random.randint(low=1, high=4, size=rank)
        cond_shape = np.array([shape[0]]) if broadcast else shape

        cond_val = np.random.randint(low=0, high=2, size=cond_shape).astype(np.bool)
        a_val = random_gen(shape=shape, rand_min=-1962.0, rand_max=0.0)
        b_val = random_gen(shape=shape, rand_min=0.0, rand_max=1964.0)

        if dynamic:
            cond_shape = [None] * len(cond_shape) + [tf.bool]
            a_shape = [None] * len(shape) + [tf.float32]
            b_shape = [None] * len(shape) + [tf.float32]
        else:
            cond_shape = cond_shape.tolist() + [tf.bool]
            a_shape = shape.tolist() + [tf.float32]
            b_shape = shape.tolist() + [tf.float32]

        @make_tf_graph([cond_shape, a_shape, b_shape])
        def build_model_select(cond, a, b):
            return tf.raw_ops.Select(condition=cond, x=a, y=b)

        model, inputs, outputs = build_model_select
        inputs_dic = dict(zip(inputs, [cond_val, a_val, b_val]))
        TensorFlowBaseTest.run_compare_tf(
            model, inputs_dic, outputs, use_cpu_only=use_cpu_for_conversion, backend=backend,
            use_cpu_for_conversion=use_cpu_for_conversion,
        )


class TestWhere(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [rank for rank in range(1, 6)]),
    )
    def test_where_1_input(self, use_cpu_only, backend, rank):
        with tf.Graph().as_default() as graph:
            shape = np.random.randint(low=1, high=4, size=rank)
            x_val = np.random.randint(low=-1, high=2, size=shape).astype(np.float32)
            x = tf.placeholder(tf.float32, shape=shape)
            TensorFlowBaseTest.run_compare_tf(
                graph,
                {x: x_val},
                tf.where(x),
                use_cpu_only=use_cpu_only,
                backend=backend,
            )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [rank for rank in range(1, 6)]),
    )
    def test_where(self, use_cpu_only, backend, rank):
        shape = np.random.randint(low=1, high=4, size=rank)
        cond_val = np.random.randint(low=0, high=2, size=shape).astype(np.int32)
        a_val = random_gen(shape=shape, rand_min=-1962.0, rand_max=0.0)
        b_val = random_gen(shape=shape, rand_min=0.0, rand_max=1964.0)
        with tf.Graph().as_default() as graph:
            cond = tf.placeholder(tf.bool, shape=shape)
            a = tf.placeholder(tf.float32, shape=shape)
            b = tf.placeholder(tf.float32, shape=shape)
            ref = tf.where(cond, a, b)
            TensorFlowBaseTest.run_compare_tf(
                graph,
                {cond: cond_val, a: a_val, b: b_val},
                ref,
                use_cpu_only=use_cpu_only,
                backend=backend,
            )


class TestCast(TensorFlowBaseTest):
    @pytest.mark.parametrize('use_cpu_only, backend, rank, dtype',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 list(range(1, 6)),
                                 ['int32', 'float64']
                             ))
    def test(self, use_cpu_only, backend, rank, dtype):
        shape = np.random.randint(low=1, high=3, size=rank)

        if backend[0] == "mlprogram" and dtype == "int32":
            pytest.xfail("rdar://78630549")

        @make_tf_graph([shape])
        def build_model(x):
            y = tf.cast(x, dtype=dtype)
            y = tf.square(y)
            return y

        model, inputs, outputs = build_model
        input_values = [random_gen(shape, -100, 100)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(model, input_dict, outputs,
                       use_cpu_only=use_cpu_only,
                       backend=backend)


class TestCond(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends,)
    )
    def test_cond_naive(self, use_cpu_only, backend):
        @make_tf_graph([(1,), (1,)])
        def build_model(x, y):
            return tf.cond(tf.constant(True), lambda: x + y, lambda: x * y)

        model, inputs, outputs = build_model
        input_values = [
            np.array([1], dtype=np.float32),
            np.array([6], dtype=np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )

    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend", itertools.product([True, False], backends,)
    )
    def test_cond(self, use_cpu_for_conversion, backend):
        if backend[0] == "mlprogram" and not use_cpu_for_conversion:
            pytest.xfail("rdar://77441664")

        @make_tf_graph([(1,), (1,)])
        def build_model(x, y):
            z = tf.multiply(x, y)
            pred = tf.less(tf.math.reduce_mean(x), tf.math.reduce_mean(y))
            return tf.cond(pred, lambda: tf.add(x, z), lambda: tf.square(y))

        model, inputs, outputs = build_model
        input_values = [
            np.array([1], dtype=np.float32),
            np.array([2], dtype=np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_for_conversion, backend=backend,
            use_cpu_for_conversion=use_cpu_for_conversion,
        )

    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend", itertools.product([True, False], backends,)
    )
    def test_cond_multi_returns(self, use_cpu_for_conversion, backend):
        if backend[0] == "mlprogram" and not use_cpu_for_conversion:
            pytest.xfail("rdar://77441664")

        @make_tf_graph([(1,), (1,)])
        def build_model(x, y):
            z = tf.multiply(x, y)
            pred = tf.less(tf.math.reduce_mean(x), tf.math.reduce_mean(y))

            def true_fn():
                return tf.add(x, z), tf.math.multiply(x, z)

            def false_fn():
                return tf.square(y), tf.sqrt(z)

            return tf.cond(pred, true_fn, false_fn)

        model, inputs, outputs = build_model
        input_values = [
            np.array([1], dtype=np.float32),
            np.array([2], dtype=np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_for_conversion, backend=backend,
            use_cpu_for_conversion=use_cpu_for_conversion,
        )

    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend", itertools.product([True, False], backends,)
    )
    def test_cond_with_identity(self, use_cpu_for_conversion, backend):
        if backend[0] == "mlprogram" and not use_cpu_for_conversion:
            pytest.xfail("rdar://77441664")

        @make_tf_graph([(1,), (1,)])
        def build_model(x, y):
            z = tf.multiply(x, y)
            pred = tf.less(tf.math.reduce_mean(x), tf.math.reduce_mean(y))
            return tf.cond(pred, lambda: z, lambda: tf.square(y))

        model, inputs, outputs = build_model
        input_values = [
            np.array([1], dtype=np.float32),
            np.array([2], dtype=np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_for_conversion, backend=backend,
            use_cpu_for_conversion=use_cpu_for_conversion,
        )

    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend", itertools.product([True, False], backends,)
    )
    def test_cond_multi_returns_with_identity(self, use_cpu_for_conversion, backend):
        if backend[0] == "mlprogram" and not use_cpu_for_conversion:
            pytest.xfail("rdar://77441664")

        @make_tf_graph([(1,), (1,)])
        def build_model(x, y):
            z = tf.multiply(x, y)
            pred = tf.less(tf.math.reduce_mean(x), tf.math.reduce_mean(y))

            def true_fn():
                return tf.add(x, z), x

            def false_fn():
                return tf.square(y), z

            return tf.cond(pred, true_fn, false_fn)

        model, inputs, outputs = build_model
        input_values = [
            np.array([1], dtype=np.float32),
            np.array([2], dtype=np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_for_conversion, backend=backend,
            use_cpu_for_conversion=use_cpu_for_conversion,
        )

    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend", itertools.product([True, False], backends,)
    )
    def test_cond_nested_0(self, use_cpu_for_conversion, backend):
        if backend[0] == "mlprogram" and not use_cpu_for_conversion:
            pytest.xfail("rdar://77441664")

        if backend == ("mlprogram", "fp16"):
            pytest.xfail("rdar://80660074 (Cond mlprogram FP16 tests falling in TF1 converter with numerical errors)")

        @make_tf_graph([(1,), (1,)])
        def build_model(x, y):
            z = tf.multiply(x, y)
            t = tf.less(tf.math.reduce_mean(x), tf.math.reduce_mean(y))
            f = tf.less(tf.math.reduce_mean(z), tf.math.reduce_mean(y))
            inner_cond = tf.cond(
                f, lambda: tf.pow(x, y), lambda: tf.math.subtract(x, y)
            )
            return tf.cond(t, lambda: inner_cond, lambda: tf.square(y))

        model, inputs, outputs = build_model

        input_values = [
            np.array([2], dtype=np.float32),
            np.array([3], dtype=np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_for_conversion, backend=backend,
            use_cpu_for_conversion=use_cpu_for_conversion,
        )

    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend", itertools.product([True, False], backends,)
    )
    def test_cond_nested_1(self, use_cpu_for_conversion, backend):
        if backend[0] == "mlprogram" and not use_cpu_for_conversion:
            pytest.xfail("rdar://77441664")

        if backend == ("mlprogram", "fp16"):
            pytest.xfail("rdar://80660074 (Cond mlprogram FP16 tests falling in TF1 converter with numerical errors)")

        @make_tf_graph([(1,), (1,)])
        def build_model(x, y):
            z = tf.multiply(x, y)
            t = tf.less(tf.math.reduce_mean(x), tf.math.reduce_mean(y))
            f = tf.less(tf.math.reduce_mean(z), tf.math.reduce_mean(y))
            cond_1 = tf.cond(f, lambda: tf.pow(x, y), lambda: tf.math.subtract(x, y))
            cond_2 = tf.cond(t, lambda: tf.multiply(x, y), lambda: tf.math.mod(x, y))
            cond_3 = tf.cond(f, lambda: tf.math.divide(x, y), lambda: cond_2)
            return tf.cond(t, lambda: cond_1, lambda: cond_3)

        model, inputs, outputs = build_model

        input_values = [
            np.array([2], dtype=np.float32),
            np.array([3], dtype=np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_for_conversion, backend=backend,
            use_cpu_for_conversion=use_cpu_for_conversion,
        )


class TestWhileLoop(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends))
    def test_while_loop_with_changing_shape(self, use_cpu_only, backend):
        @make_tf_graph([(2,1),(2,1)])
        def build_model(x,y):
            c = lambda i,j: tf.less(tf.shape(j)[1], 5)
            b = lambda i,j: (i, tf.concat([i,j], axis=1))
            return tf.while_loop(c, b, [x,y], shape_invariants=[x.get_shape(), tf.TensorShape([2, None])])

        model, inputs, outputs = build_model
        input_values = [np.array([[1],[2]], dtype=np.float32),np.array([[1],[2]], dtype=np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(model, input_dict, outputs,
                       use_cpu_only=use_cpu_only,
                       backend=backend)

    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends)
    )
    def test_while_loop_no_entry(self, use_cpu_only, backend):
        @make_tf_graph([(1,)])
        def build_model(x):
            c = lambda i: tf.greater(tf.math.reduce_mean(i), 5)
            b = lambda i: i - 1
            return tf.while_loop(c, b, [x])

        model, inputs, outputs = build_model
        input_values = [np.array([5], dtype=np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends)
    )
    def test_while_loop_0(self, use_cpu_only, backend):
        @make_tf_graph([(1,)])
        def build_model(x):
            c = lambda i: tf.greater(tf.math.reduce_mean(i), 5)
            b = lambda i: i - 1
            return tf.while_loop(c, b, [x])

        model, inputs, outputs = build_model
        input_values = [np.array([10], dtype=np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends)
    )
    def test_while_loop_1(self, use_cpu_only, backend):
        @make_tf_graph([(1,), (1,)])
        def build_model(x, y):
            c = lambda i, j: tf.greater(tf.math.reduce_mean(i), tf.math.reduce_mean(j))
            b = lambda i, j: (tf.add(i, 1), tf.square(j))
            return tf.while_loop(c, b, [x, y])

        model, inputs, outputs = build_model
        input_values = [
            np.array([1], dtype=np.float32),
            np.array([2], dtype=np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends)
    )
    def test_while_loop_2(self, use_cpu_only, backend):
        @make_tf_graph([(1,), (1, 2)])
        def build_model(x, y):
            c = lambda i, j: tf.greater(tf.math.reduce_mean(i), 5)
            b = lambda i, j: (i - 3, j * 2)
            return tf.while_loop(c, b, [x, y])

        model, inputs, outputs = build_model
        input_values = [
            np.array([10], dtype=np.float32),
            np.array([[2, 3]], dtype=np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends)
    )
    def test_while_loop_3(self, use_cpu_only, backend):
        @make_tf_graph([(1,), (1, 2), (1,)])
        def build_model(x, y, z):
            c = lambda i, j, k: tf.greater(
                tf.math.reduce_mean(i), tf.math.reduce_mean(j)
            )
            b = lambda i, j, k: (i / 3, j ** 2, k - 2)
            return tf.while_loop(c, b, [x, y, z])

        model, inputs, outputs = build_model
        input_values = [
            np.array([10], dtype=np.float32),
            np.array([[2, 3]], dtype=np.float32),
            np.array([5], dtype=np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends)
    )
    def test_while_loop_4(self, use_cpu_only, backend):
        @make_tf_graph([(1,), (1, 2), (1,), (2, 1)])
        def build_model(x, y, z, m):
            c = lambda i, j, k, l: tf.greater(
                tf.math.reduce_mean(i), tf.math.reduce_mean(j)
            )
            b = lambda i, j, k, l: (i / 3, j ** 2, k - 2, l % 2)
            return tf.while_loop(c, b, [x, y, z, m])

        model, inputs, outputs = build_model
        input_values = [
            np.array([10], dtype=np.float32),
            np.array([[2, 3]], dtype=np.float32),
            np.array([5], dtype=np.float32),
            np.array([[2], [3]], dtype=np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends,)
    )
    def test_nested_while_body(self, use_cpu_only, backend):
        @make_tf_graph([(1,), (1,)])
        def build_model(x, y):
            # The following while loop:
            #
            # i, j = 0, 10
            # while i < j:
            #   while 2*i < i+2:
            #     i += 1
            #   i += 2

            def cond2(i):
                return tf.less(2 * tf.math.reduce_mean(i), tf.math.reduce_mean(i + 2))

            def body2(i):
                return i + 1

            def cond1(i, j):
                return tf.less(tf.math.reduce_mean(i), tf.math.reduce_mean(j))

            def body1(i, j):
                new_i = tf.while_loop(cond2, body2, [i])
                return new_i + 2, j

            return tf.while_loop(cond1, body1, [x, y])

        model, inputs, outputs = build_model
        input_values = [
            np.array([0], dtype=np.float32),
            np.array([10], dtype=np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends,)
    )
    def test_nested_while_cond(self, use_cpu_only, backend):
        @make_tf_graph([(1,), (1,)])
        def build_model(x, y):
            # The following while loop:
            #
            # def cond(i, j):
            #  while 2*i < i+2:
            #    i += 1
            #  return i < j
            #
            # i, j = 0, 10
            # while cond(i, j):
            #   i += 2
            #   j += 1

            def cond2(i):
                return tf.less(2 * tf.math.reduce_mean(i), tf.math.reduce_mean(i + 2))

            def body2(i):
                return i + 1

            def cond1(i, j):
                new_i = tf.while_loop(cond2, body2, [i])
                return tf.less(tf.squeeze(new_i), tf.squeeze(j))

            def body1(i, j):
                return i + 2, j + 1

            return tf.while_loop(cond1, body1, [x, y])

        model, inputs, outputs = build_model
        input_values = [
            np.array([0], dtype=np.float32),
            np.array([10], dtype=np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )


class TestConv(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        ",".join(
            [
                "use_cpu_only",
                "backend",
                "conv_dim",  # 1d or 2d conv
                "padding",
                "data_format",
                "HWkHkW",
                "strides",
                "dilations",
                "dynamic_weights",
                "batch_size",
            ]
        ),
        itertools.product(
            [True, False],
            backends,
            ["conv1d", "conv2d"],
            ["SAME", "VALID", [[2, 3], [3, 2]]],
            ["NHWC"],  # NCHW not supported by TF.
            [(11, 12, 3, 2), (12, 11, 2, 3)],
            [(1, 1), (2, 3)],
            [(1, 1), (2, 3)],
            [True, False],
            [1, 3],
        ),
    )
    def test(
        self,
        use_cpu_only,
        backend,
        conv_dim,
        padding,
        data_format,
        HWkHkW,
        strides,
        dilations,
        dynamic_weights,
        batch_size,
    ):
        H, W, kH, kW = HWkHkW
        N, C_in, C_out = batch_size, 2, 3
        if data_format == "NHWC":
            input_shape = (N, W, C_in) if conv_dim == "conv1d" else (N, H, W, C_in)
            if isinstance(padding, list):
                padding = [[0, 0]] + padding + [[0, 0]]
            if conv_dim == "conv1d":
                data_format = "NWC"
                if isinstance(padding, list):
                    # No explicit padding for conv1d in TF
                    return
        else:  # 'NCHW'
            input_shape = (N, C_in, W) if conv_dim == "conv1d" else (N, C_in, H, W)
            if isinstance(padding, list):
                padding = [[0, 0], [0, 0]] + padding
            if conv_dim == "conv1d":
                data_format = "NCW"
                if isinstance(padding, list):
                    # No explicit padding for conv1d in TF
                    return
        W_shape = (kW, C_in, C_out) if conv_dim == "conv1d" else (kH, kW, C_in, C_out)
        dilations = dilations[1] if conv_dim == "conv1d" else dilations
        strides = strides[1] if conv_dim == "conv1d" else strides

        # We do not support dynamic weight when dilations != 1.
        if dynamic_weights and dilations == (1, 1):

            @make_tf_graph([input_shape, W_shape])
            def build_model_dynamic_weights(x, W):
                if conv_dim == "conv1d":
                    conv = tf.nn.conv1d(
                        x,
                        W,
                        stride=strides,
                        padding=padding,
                        dilations=dilations,
                        data_format=data_format,
                    )
                else:
                    conv = tf.nn.conv2d(
                        x,
                        W,
                        strides=strides,
                        padding=padding,
                        dilations=dilations,
                        data_format=data_format,
                    )
                return conv

            model, inputs, outputs = build_model_dynamic_weights
            input_values = [
                random_gen(input_shape, -10.0, 10.0),
                random_gen(W_shape, -1.0, 1.0),
            ]
            input_dict = dict(zip(inputs, input_values))

        else:

            @make_tf_graph([input_shape])
            def build_model_static_weights(x):
                W = tf.constant(np.random.rand(*W_shape), tf.float32)
                if conv_dim == "conv1d":
                    conv = tf.nn.conv1d(
                        x,
                        W,
                        stride=strides,
                        padding=padding,
                        dilations=dilations,
                        data_format=data_format,
                    )
                else:
                    conv = tf.nn.conv2d(
                        x,
                        W,
                        strides=strides,
                        padding=padding,
                        dilations=dilations,
                        data_format=data_format,
                    )
                return conv

            model, inputs, outputs = build_model_static_weights
            input_values = [random_gen(input_shape, -10.0, 10.0)]
            input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestConv3d(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        ",".join(
            [
                "use_cpu_only",
                "backend",
                "data_format",
                "input_size",
                "kernel_size",
                "strides",
                "dilations",
                "padding_type",
                "batch_size",
            ]
        ),
        itertools.product(
            [True, False],  # use_cpu_only
            backends,
            ["NDHWC"],  # NCDHW not supported by TF.
            [(7, 11, 13), (32, 16, 8)],  # input_size
            [(1, 1, 1), (3, 3, 3), (1, 2, 3)],  # kernel_size
            [(1, 1, 1), (2, 2, 2), (3, 2, 1)],  # strides
            [
                (1, 1, 1)
            ], # , (2, 2, 2), (2, 3, 1)],  # dilations: dilations greater than 1 not supported on CPU
            ["SAME", "VALID"],  # padding_type
            [1, 3],  # batch_size
        ),
    )
    def test_tf(
        self,
        use_cpu_only,
        backend,
        data_format,
        input_size,
        kernel_size,
        strides,
        dilations,
        padding_type,
        batch_size,
    ):
        C_in = np.random.randint(low=1, high=4)
        C_out = np.random.randint(low=1, high=(C_in + 1))
        input_shape = [batch_size] + list(input_size) + [C_in]
        weights_shape = list(kernel_size) + [C_in, C_out]
        # TF1 and TF2 tf.nn.conv3d require dilations and strides to have length 5 or greater, with values of 1 for
        # indices 0 and 4 (batch and channel in NDHWC format)
        tf_strides = [1] + list(strides) + [1]
        tf_dilations = [1] + list(dilations) + [1]

        @make_tf_graph([input_shape])
        def build_model_static_weights(x):
            W = tf.constant(np.random.rand(*weights_shape), tf.float32)
            return tf.nn.conv3d(
                x,
                W,
                strides=tf_strides,
                padding=padding_type,
                data_format=data_format,
                dilations=tf_dilations,
            )

        model, inputs, outputs = build_model_static_weights
        input_values = [random_gen(input_shape, -10.0, 10.0)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
            frontend_only=False,
            atol=1e-03,  # default 1e-04
            rtol=2e-03,  # default 1e-05
        )


class TestDepthwiseConv(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        ",".join(
            [
                "use_cpu_for_conversion",
                "backend",
                "padding",
                "HWkHkW",
                "strides",
                "dilations",
                "dynamic_weights",
                "batch_size",
            ]
        ),
        itertools.product(
            [True, False],
            backends,
            ["SAME", "VALID"],
            [(11, 12, 3, 2), (12, 11, 2, 3)],
            # TF doesn't support non-square strides for depthwise
            # https://github.com/tensorflow/tensorflow/issues/33005
            [(1, 1, 1, 1), (1, 2, 2, 1)],
            [
                (1, 1),
                (2, 2),
            ],
            [True, False],
            [1, 3],
        ),
    )
    def test_depthwise_conv(
        self,
        use_cpu_for_conversion,
        backend,
        padding,
        HWkHkW,
        strides,
        dilations,
        dynamic_weights,
        batch_size,
    ):
        use_cpu_only = use_cpu_for_conversion
        if backend[0] == "mlprogram" and dilations == (1,1) and dynamic_weights and not use_cpu_for_conversion:
            # in this case, there is a numerical mismatch on the GPU MIL backend. The GPU runtime tests are
            # tracked seprately.
            use_cpu_only = True

        if np.sum(strides) != len(strides) and np.sum(dilations) != len(dilations):
            # TF doesn't compute correct output for non-one stride+dilation
            return
        H, W, kH, kW = HWkHkW
        N, C_in, C_out = batch_size, 2, 6
        input_shape = (N, H, W, C_in)
        data_format = "NHWC"
        assert C_out % C_in == 0
        multiplier = int(C_out / C_in)
        W_shape = (kH, kW, C_in, multiplier)

        def test_static_W():
            W = np.random.rand(*W_shape).astype(np.float32)

            @make_tf_graph([input_shape])
            def build_model_static_weights(x):
                return tf.nn.depthwise_conv2d(
                    x,
                    W,
                    strides=strides,
                    padding=padding,
                    dilations=dilations,
                    data_format=data_format,
                )

            model, inputs, outputs = build_model_static_weights

            input_values = [(np.random.rand(*input_shape).astype(np.float32))]
            input_dict = dict(zip(inputs, input_values))

            proto,_,_,_,_,_ = TensorFlowBaseTest.run_compare_tf(
                model,
                input_dict,
                outputs,
                use_cpu_only=use_cpu_only,
                backend=backend,
                frontend_only=False,
                use_cpu_for_conversion=use_cpu_for_conversion,
            )

            if backend[0] == 'nnv1_proto':
                assert layer_counts(proto, "reorganizeData") == 0

        def test_dynamic_W():
            @make_tf_graph([input_shape, W_shape])
            def build_model_dynamic_weights(x, W):
                return tf.nn.depthwise_conv2d(
                    x,
                    W,
                    strides=strides,
                    padding=padding,
                    dilations=dilations,
                    data_format=data_format,
                )

            model, inputs, outputs = build_model_dynamic_weights

            input_values = [
                (np.random.rand(*input_shape).astype(np.float32)),
                (np.random.rand(*W_shape).astype(np.float32)),
            ]
            input_dict = dict(zip(inputs, input_values))

            TensorFlowBaseTest.run_compare_tf(
                model,
                input_dict,
                outputs,
                use_cpu_only=use_cpu_only,
                backend=backend,
                frontend_only=False,
                use_cpu_for_conversion=use_cpu_for_conversion,
            )

        # We do not support dynamic weight when dilations != 1.
        test_dynamic_W() if dynamic_weights and dilations == (1, 1) else test_static_W()


class TestSeparableConv(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        ",".join(
            [
                "use_cpu_for_conversion",
                "backend",
                "padding",
                "HWkHkW",
                "strides",
                "dilations",
                "dynamic_weights",
                "batch_size",
            ]
        ),
        itertools.product(
            [True, False],
            backends,
            ["SAME", "VALID"],
            [(11, 12, 3, 2), (12, 11, 2, 3)],
            [(1, 1, 1, 1), (1, 2, 2, 1)],
            [(1, 1), (2, 2)],
            [True, False],
            [1, 3],
        ),
    )
    def test_separable_conv(
        self,
        use_cpu_for_conversion,
        backend,
        padding,
        HWkHkW,
        strides,
        dilations,
        dynamic_weights,
        batch_size,
    ):
        use_cpu_only = use_cpu_for_conversion
        if backend[0] == "mlprogram" and dilations == (1,1):
            # in this case, there is a numerical mismatch on the GPU MIL backend. The GPU runtime tests are
            # tracked seprately.
            use_cpu_only = True

        H, depthwise_filter, kH, kW = HWkHkW
        N, C_in, C_out = batch_size, 2, 6
        input_shape = (N, H, depthwise_filter, C_in)
        data_format = "NHWC"
        assert C_out % C_in == 0
        multiplier = int(C_out / C_in)
        depthwise_filter_shape = (kH, kW, C_in, multiplier)
        pointwise_filter_shape = [1, 1, multiplier * C_in, C_out]
        if dilations != (1, 1):
            strides = (1, 1, 1, 1)

        def test_dynamic_W():
            @make_tf_graph(
                [input_shape, depthwise_filter_shape, pointwise_filter_shape]
            )
            def build_model_dynamic_weights(x, depthwise_filter, pointwise_filter):
                return tf.nn.separable_conv2d(
                    x,
                    depthwise_filter,
                    pointwise_filter,
                    strides=strides,
                    padding=padding,
                    dilations=dilations,
                    data_format=data_format,
                )

            model, inputs, outputs = build_model_dynamic_weights

            input_values = [
                (np.random.rand(*input_shape).astype(np.float32)),
                (np.random.rand(*depthwise_filter_shape).astype(np.float32)),
                (np.random.rand(*pointwise_filter_shape).astype(np.float32)),
            ]
            input_dict = dict(zip(inputs, input_values))

            TensorFlowBaseTest.run_compare_tf(
                model,
                input_dict,
                outputs,
                use_cpu_only=use_cpu_only,
                backend=backend,
                frontend_only=False,
                use_cpu_for_conversion=use_cpu_for_conversion,
            )

        def test_static_W():
            depthwise_filter = np.random.rand(*depthwise_filter_shape).astype(
                np.float32
            )
            pointwise_filter = np.random.rand(*pointwise_filter_shape).astype(
                np.float32
            )

            @make_tf_graph([input_shape])
            def build_model_static_weights(x):
                return tf.nn.separable_conv2d(
                    x,
                    depthwise_filter,
                    pointwise_filter,
                    strides=strides,
                    padding=padding,
                    dilations=dilations,
                    data_format=data_format,
                )

            model, inputs, outputs = build_model_static_weights

            input_values = [(np.random.rand(*input_shape).astype(np.float32))]
            input_dict = dict(zip(inputs, input_values))

            TensorFlowBaseTest.run_compare_tf(
                model,
                input_dict,
                outputs,
                use_cpu_only=use_cpu_only,
                backend=backend,
                frontend_only=False,
                use_cpu_for_conversion=use_cpu_for_conversion,
            )

        test_static_W()
        if not any([True if d > 1 else False for d in dilations]):
            test_dynamic_W()

class TestConvTranspose(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        ",".join(
            [
                "use_cpu_only",
                "backend",
                "conv_dim",  # 1d or 2d conv
                "padding",
                "data_format",
                "HWkHkW",
                "strides",
                "dilations",
                "dynamic",
            ]
        ),
        itertools.product(
            [True, False],
            backends,
            ["conv1d", "conv2d"],
            ["SAME", "VALID"],
            ["NHWC"],  # NCHW not supported by TF
            [(12, 10, 2, 2), (4, 2, 2, 3), (7, 5, 3, 3)],
            [(1, 1), (1, 2)],
            [(1, 1)],  # Dilation > 1 not supported by TF
            [True, False],
        ),
    )
    def test_conv_transpose(
        self,
        use_cpu_only,
        backend,
        conv_dim,
        padding,
        data_format,
        HWkHkW,
        strides,
        dilations,
        dynamic,
    ):
        H, W, kH, kW = HWkHkW
        N, C_in, C_out = 1, 1, 2

        if data_format == "NHWC":
            input_shape = (N, W, C_in) if conv_dim == "conv1d" else (N, H, W, C_in)
            if conv_dim == "conv1d":
                data_format = "NWC"
        else:  # 'NCHW'
            pass

        w_shape = (kW, C_out, C_in) if conv_dim == "conv1d" else (kH, kW, C_out, C_in)

        # dynamic input shape
        tf_input_shape = list(input_shape)
        if dynamic:
            if data_format == "NHWC":
                tf_input_shape[1] = None
                tf_input_shape[2] = None
            elif data_format == "NWC":
                tf_input_shape[1] = None

        @make_tf_graph([tf_input_shape])
        def build_model(x):
            Weight = tf.constant(np.random.rand(*w_shape), tf.float32)

            # get the dynamic height and width
            if dynamic:
                shape = tf.shape(x)
                if data_format == "NHWC":
                    H, W = shape[1], shape[2]
                elif data_format == "NWC":
                    W = shape[1]
            else:
                H, W = HWkHkW[:2]

            kH, kW = HWkHkW[2:]

            is_conv_2d = conv_dim == "conv2d"

            # compute the output shape, in both static / dynamic cases
            if padding == "SAME":
                oW = W * strides[1]
                if is_conv_2d:
                    oH = H * strides[0]
            elif padding == "VALID":
                oW = (W - 1) * strides[1] + (kW - 1) * dilations[1] + 1
                if is_conv_2d:
                    oH = (H - 1) * strides[0] + (kH - 1) * dilations[0] + 1

            if data_format == "NHWC":
                output_shape = [N, oH, oW, C_out]
            elif data_format == "NWC":
                output_shape = [N, oW, C_out]

            if conv_dim == "conv1d":
                return tf.nn.conv1d_transpose(
                    x,
                    Weight,
                    output_shape=output_shape,
                    strides=strides[1],
                    padding=padding,
                    dilations=dilations[1],
                    data_format=data_format,
                )
            elif conv_dim == "conv2d":
                return tf.nn.conv2d_transpose(
                        x,
                        Weight,
                        output_shape=output_shape,
                        strides=strides,
                        padding=padding,
                        dilations=dilations,
                        data_format=data_format,
                    )
        model, inputs, outputs = build_model

        input_values = [(np.random.rand(*input_shape).astype(np.float32))]
        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
            frontend_only=False,
        )

    @pytest.mark.parametrize(
        ",".join(
            [
                "use_cpu_only",
                "backend",
                "padding",
                "data_format",
                "DHWkDkHkW",
                "strides",
                "dilations",
                "dynamic",
            ]
        ),
        itertools.product(
            [True, False],
            backends,
            [
                "SAME", "VALID"
            ],
            ["NDHWC"],
            [
                (10, 12, 14, 2, 3, 5),
                (4, 6, 8, 2, 3, 1),
                (6, 8, 10, 3, 3, 3),
                (5, 7, 9, 2, 4, 2),
            ],
            [(1, 1, 1), (1, 2, 3)],
            [(1, 1, 1)],  # Dilation > 1 not supported by TF
            [True, False],
        ),
    )
    def test_conv3d_transpose(
        self, use_cpu_only, backend, padding, data_format, DHWkDkHkW, strides, dilations, dynamic,
    ):
        if _macos_version() < (12, 0) and strides == (1, 2, 3) and padding == "VALID":
            # Behavior changed in macOS 12
            return

        D, H, W, kD, kH, kW = DHWkDkHkW
        N, C_in, C_out = 2, 1, 2

        if data_format == "NDHWC":
            input_shape = (N, D, H, W, C_in)
        else:  # 'NCDHW'
            pass

        tf_input_shape = list(input_shape)
        if dynamic:
            if data_format == "NDHWC":
                tf_input_shape[1] = None
                tf_input_shape[2] = None
                tf_input_shape[3] = None
            else:
                pass

        w_shape = (kD, kH, kW, C_out, C_in)
        x_input = np.random.randn(*input_shape)

        @make_tf_graph([tf_input_shape])
        def build_model(x):
            weight = tf.constant(np.random.rand(*w_shape), tf.float32)

            # get the depth, height and width
            if dynamic:
                shape = tf.shape(x)
                if data_format == "NDHWC":
                    D, H, W = shape[1], shape[2], shape[3]
                else:
                    pass
            else:
                D, H, W = DHWkDkHkW[:3]

            kD, kH, kW = DHWkDkHkW[3:]

            # compute the output shape
            if padding == "SAME":
                oD = D * strides[0]
                oH = H * strides[1]
                oW = W * strides[2]
            else:
                oD = (D - 1) * strides[0] + (kD - 1) * dilations[0] + 1
                oH = (H - 1) * strides[1] + (kH - 1) * dilations[1] + 1
                oW = (W - 1) * strides[2] + (kW - 1) * dilations[2] + 1

            if data_format == "NDHWC":
                output_shape = [N, oD, oH, oW, C_out]
            else:
                pass

            return tf.nn.conv3d_transpose(
                    x,
                    weight,
                    output_shape=output_shape,
                    strides=strides,
                    padding=padding,
                    dilations=dilations,
                    data_format=data_format,
                )
        model, inputs, outputs = build_model

        input_values = [(np.random.rand(*input_shape).astype(np.float32))]
        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
            frontend_only=False,
        )


class TestElementWiseBinary(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank, tf_op, broadcast_case",
        itertools.product(
            [True, False],
            backends,
            [0, 1, 2, 3, 4],
            [
                tf.math.add,
                tf.math.floordiv,
                tf.math.floormod,
                tf.math.maximum,
                tf.math.minimum,
                tf.math.mod,
                tf.math.multiply,
                tf.math.pow,
                tf.math.truediv,
                tf.math.subtract,
                tf.math.squared_difference,
            ],
            [0, 1, 2, 3]
        ),
    )
    def test_binary_math(self, use_cpu_only, backend, rank, tf_op,
            broadcast_case):
        if rank == 0 or broadcast_case == 0:
            pytest.xfail("Rank-0 input is not supported")

        if backend[0] == "mlprogram" and not use_cpu_only and tf_op == tf.math.floormod:
            pytest.xfail("rdar://78343225 ((MIL GPU) Core ML Tools Unit Test failures [numerical error])")

        if backend[0] == "mlprogram" and not use_cpu_only and tf_op == tf.math.floordiv:
            pytest.xfail("rdar://82743379")

        x_shape = y_shape = list(np.random.randint(low=2, high=4, size=rank))

        # test broadcasting
        # 0 -> broadcast with one of the inputs is a 0-D tensor (scalar)
        # 1 -> broadcast with same rank, some of dimensions are size 1
        # 2 -> broadcast with different rank, extra dimension with size 1
        # 3 -> no broadcast, same type for both inputs
        if broadcast_case == 0:
            y_shape = []
        elif broadcast_case == 1:
            y_shape = [1 if np.random.randint(2) == 0 else d for d in y_shape]
        elif broadcast_case == 2:
            y_shape = [1] + y_shape

        # randomly swap x and y
        if np.random.randint(2) == 0:
            x_shape, y_shape = y_shape, x_shape

        # lower precision input data for non-CPU tests
        dtype = np.float32 if use_cpu_only else np.float16

        if tf_op in {tf.math.add, tf.math.subtract, tf.math.multiply}:
            x_val = random_gen(x_shape, -100, 100, dtype=dtype).astype(np.float32)
            y_val = random_gen(y_shape, -100, 100, dtype=dtype).astype(np.float32)
        elif tf_op in {tf.math.truediv, tf.math.floordiv, tf.math.floormod, tf.math.mod}:
            x_val = random_gen(x_shape, -100, 100, dtype=dtype).astype(np.float32)
            y_val = random_gen(y_shape, 1, 20, dtype=dtype).astype(np.float32)
        elif tf_op in {tf.math.maximum, tf.math.minimum}:
            x_val = random_gen(x_shape, -10, 10, dtype=dtype).astype(np.float32)
            y_val = random_gen(y_shape, -10, 10, dtype=dtype).astype(np.float32)
        elif tf_op in {tf.math.pow, tf.math.squared_difference}:
            x_val = random_gen(x_shape, -5, 5, dtype=np.int).astype(np.float32)
            y_val = random_gen(y_shape, -5, 5, dtype=np.int).astype(np.float32)
        else:
            raise NotImplementedError("input values needs to be defined")

        @make_tf_graph([x_shape, y_shape])
        def build_model(x, y):
            return tf_op(x, y)

        model, inputs, outputs = build_model
        input_values = [x_val, y_val]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )

    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend, rank, tf_op, broadcast_case",
        itertools.product(
            [True, False],
            backends,
            [0, 1, 2, 3, 4],
            [
                tf.equal,
                tf.not_equal,
                tf.greater,
                tf.greater_equal,
                tf.less,
                tf.less_equal,
            ],
            [0, 1, 2, 3],
        ),
    )
    def test_binary_compare(self, use_cpu_for_conversion, backend, rank, tf_op,
            broadcast_case):
        if rank == 0 or broadcast_case == 0:
            pytest.xfail("Rank-0 input is not supported")

        use_cpu_only = use_cpu_for_conversion
        if backend[0] == "mlprogram" and not use_cpu_for_conversion:
            pytest.xfail("Error in building plan : MIL GPU backend failure. rdar://78218824")

        x_shape = y_shape = list(np.random.randint(low=2, high=4, size=rank))

        # test broadcasting
        # 0 -> broadcast with one of the inputs is a 0-D tensor (scalar)
        # 1 -> broadcast with same rank, some of dimensions are size 1
        # 2 -> broadcast with different rank, extra dimension with size 1
        # 3 -> no broadcast, same type for both inputs
        if broadcast_case == 0:
            y_shape = []
        elif broadcast_case == 1:
            y_shape = [1 if np.random.randint(2) == 0 else d for d in y_shape]
        elif broadcast_case == 2:
            y_shape = [1] + y_shape

        # randomly swap x and y
        if np.random.randint(2) == 0:
            x_shape, y_shape = y_shape, x_shape

        # lower precision input data for non-CPU tests
        dtype = np.float32 if use_cpu_only else np.float16

        @make_tf_graph([x_shape, y_shape])
        def build_model(x, y):
            return tf_op(x, y)

        model, inputs, outputs = build_model
        input_values = [
            random_gen(x_shape, -5, 3, dtype=dtype).astype(np.float32),
            random_gen(y_shape, -5, 3, dtype=dtype).astype(np.float32),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend,
            use_cpu_for_conversion=use_cpu_for_conversion,
        )

    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend, rank, tf_op, broadcast_case",
        itertools.product(
            [True, False],
            backends,
            [0, 1, 2, 3, 4],
            [
                tf.math.logical_and,
                tf.math.logical_or,
                tf.math.logical_xor,
            ],
            [0, 1, 2, 3],
        ),
    )
    def test_binary_logical(self, use_cpu_for_conversion, backend, rank, tf_op,
            broadcast_case):
        if rank == 0 or broadcast_case == 0:
            pytest.xfail("Rank-0 input is not supported")

        use_cpu_only = use_cpu_for_conversion
        if backend[0] == "mlprogram" and not use_cpu_for_conversion:
            pytest.xfail("Error in building plan : MIL GPU backend failure. rdar://78218824")

        x_shape = y_shape = list(np.random.randint(low=2, high=4, size=rank))

        # test broadcasting
        # 0 -> broadcast with one of the inputs is a 0-D tensor (scalar)
        # 1 -> broadcast with same rank, some of dimensions are size 1
        # 2 -> broadcast with different rank, extra dimension with size 1
        # 3 -> no broadcast, same type for both inputs
        if broadcast_case == 0:
            y_shape = []
        elif broadcast_case == 1:
            y_shape = [1 if np.random.randint(2) == 0 else d for d in y_shape]
        elif broadcast_case == 2:
            y_shape = [1] + y_shape

        # randomly swap x and y
        if np.random.randint(2) == 0:
            x_shape, y_shape = y_shape, x_shape

        @make_tf_graph([x_shape + [tf.bool], y_shape + [tf.bool]])
        def build_model(x, y):
            return tf_op(x, y)

        model, inputs, outputs = build_model
        input_values = [
            random_gen(x_shape, 0, 2, dtype=np.int).astype(np.bool),
            random_gen(y_shape, 0, 2, dtype=np.int).astype(np.bool),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend,
            use_cpu_for_conversion=use_cpu_for_conversion,
        )

class TestEinsum(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend, equation, reverse_input_order",
        itertools.product(
            [True, False],
            backends,
            ["abcd,adce->abce",
             "abc,cbd->abd",
             "bnqd,bnkd->bnqk",
             "abc,cd->abd",
             "abc,cde->abde",
             "btnh,bfnh->bnft",
             "bnft,btnh->bfnh",
             "abcd,cde->abe"],
            [False, True],
        )
    )
    def test(self, use_cpu_for_conversion, backend, equation, reverse_input_order):
        if backend[0] == "mlprogram" and equation == "abcd,adce->abce" and not use_cpu_for_conversion:
            pytest.xfail("rdar://80397986")

        if equation == "abcd,adce->abce":
            input_shapes = [[3, 4, 2, 6], [3, 6, 2, 2]]
        elif equation == "abc,cbd->abd":
            input_shapes = [[4, 2, 6], [6, 2, 2]]
        elif equation == "bnqd,bnkd->bnqk":
            input_shapes = [[1, 2, 3, 4], [1, 2, 4, 4]]
        elif equation == "abc,cd->abd":
            input_shapes = [[2, 3, 4], [4, 5]]
        elif equation == "abc,cde->abde":
            input_shapes = [[2, 3, 4], [4, 5, 6]]
        elif equation == "btnh,bfnh->bnft":
            input_shapes = [[1, 2, 3, 4], [1, 5, 3, 4]]
        elif equation == "bnft,btnh->bfnh":
            input_shapes = [[1, 2, 3, 4], [1, 4, 2, 6]]
        elif equation == "abcd,cde->abe":
            input_shapes = [[1, 2, 3, 4], [3, 4, 6]]
        else:
            raise ValueError("unrecognized equation")

        if reverse_input_order:
            input_output_strings = equation.split('->')
            input_strings = input_output_strings[0].split(',')
            equation = input_strings[1] + ',' + input_strings[0] + '->' + input_output_strings[1]
            input_shapes = [input_shapes[1], input_shapes[0]]

        @make_tf_graph(input_shapes)
        def build_model(x, y):
            return tf.einsum(equation, x, y)

        model, inputs, outputs = build_model

        input_values = [
            random_gen(input_shapes[0], -1, 1),
            random_gen(input_shapes[1], -1, 1),
        ]

        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_for_conversion, backend=backend,
            use_cpu_for_conversion=use_cpu_for_conversion,
        )


class TestElementWiseUnary(TensorFlowBaseTest):
    _FP16_UNSUPPORTED = {'acos', 'asin', 'atan', 'atanh', 'cosh', 'sinh'}

    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend, rank, mode",
        itertools.product(
            [True, False],
            backends,
            [1, 2, 5],
            [
                "abs",
                "acos",
                "asin",
                "atan",
                "atanh",
                "cast",
                "ceil",
                "clip",
                "cos",
                "cosh",
                "erf",
                "exp",
                "floor",
                "inverse",
                "log",
                "negative",
                "round",
                "rsqrt",
                "sign",
                "sin",
                "sinh",
                "sqrt",
                "square",
                "tan",
                "tanh",
            ],
        ),
    )
    def test_unary(self, use_cpu_for_conversion, backend, rank, mode):
        use_cpu_only = use_cpu_for_conversion # use the same context for prediction as used for loading the model

        if not use_cpu_only and mode in self._FP16_UNSUPPORTED:
            return

        if not use_cpu_for_conversion:
            pytest.xfail("rdar://78358015 (TestElementWiseUnary failing numerically on the GPU (both NNv1 and MIL))")

        atol, rtol = 1e-4, 1e-5
        input_shape = np.random.randint(low=2, high=4, size=rank)
        if use_cpu_only:
            dtype = np.float32
            tf_dtype = tf.float32
        else:
            dtype = np.float16
            tf_dtype = tf.float16

        def cast_func(x):
            return tf.cast(x, dtype=tf.int32)

        def clip_func(x):
            return tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=5.0)

        def _get_test(test_mode):
            if test_mode == "abs":
                res = tf.abs
                val = random_gen(input_shape, rand_min=-1, rand_max=1)
            elif test_mode == "acos":
                res = tf.acos
                val = random_gen(input_shape, rand_min=-1, rand_max=1)
            elif test_mode == "asin":
                res = tf.asin
                val = random_gen(input_shape, rand_min=-1, rand_max=1)
            elif test_mode == "atan":
                res = tf.atan
                val = random_gen(input_shape, rand_min=-100, rand_max=100)
            elif test_mode == "atanh":
                res = tf.atanh
                val = random_gen(input_shape, rand_min=-0.9, rand_max=0.9)
            elif test_mode == "cast":
                eps_from_int = 0.0
                if not use_cpu_only:
                    eps_from_int = 0.1
                res = cast_func
                val = random_gen(
                    input_shape,
                    rand_min=-10,
                    rand_max=10,
                    eps_from_int=eps_from_int,
                    dtype=dtype,
                )
            elif test_mode == "ceil":
                res = tf.math.ceil
                eps_from_int = 0.0
                if not use_cpu_only:
                    eps_from_int = 0.1
                val = random_gen(
                    input_shape,
                    rand_min=-100,
                    rand_max=100,
                    eps_from_int=eps_from_int,
                    dtype=dtype,
                )
            elif test_mode == "clip":
                if use_cpu_only is False:
                    return None, None  # clip does not support float16
                res = clip_func
                val = random_gen(input_shape, rand_min=-5, rand_max=10)
            elif test_mode == "cos":
                res = tf.cos
                rand_range = 1000
                if not use_cpu_only:
                    rand_range = 10
                val = random_gen(input_shape, rand_min=-rand_range, rand_max=rand_range)
            elif test_mode == "cosh":
                res = tf.cosh
                val = random_gen(input_shape, rand_min=-4, rand_max=4)
            elif test_mode == "erf":
                res = tf.math.erf
                val = random_gen(input_shape, rand_min=1, rand_max=6)
            elif test_mode == "exp":
                if not use_cpu_only:
                    # We skip GPU here, since exp(1) already differs in backend.
                    return None, None
                res = tf.exp
                val = random_gen(input_shape, rand_min=-4, rand_max=20)
            elif test_mode == "floor":
                res = tf.floor
                eps_from_int = 0.0
                if not use_cpu_only:
                    eps_from_int = 0.1
                val = random_gen(
                    input_shape,
                    rand_min=-100,
                    rand_max=100,
                    eps_from_int=eps_from_int,
                    dtype=dtype,
                )
            elif test_mode == "inverse":
                res = tf.math.reciprocal
                val = random_gen(input_shape, rand_min=0.1, rand_max=10)
            elif test_mode == "log":
                res = tf.math.log
                val = random_gen(input_shape, rand_min=0.2, rand_max=1000)
            elif test_mode == "negative":
                res = tf.math.negative
                val = random_gen(input_shape, rand_min=-100.0, rand_max=100.0)
            elif test_mode == "round":
                res = tf.round
                val = random_gen(
                    input_shape, rand_min=-1000, rand_max=1000, dtype=dtype
                )
            elif test_mode == "rsqrt":
                res = tf.math.rsqrt
                val = random_gen(input_shape, rand_min=0.5, rand_max=1000)
            elif test_mode == "sign":
                res = tf.sign
                val = random_gen(input_shape, rand_min=-5, rand_max=5)
            elif test_mode == "sin":
                res = tf.sin
                rand_range = 1000
                if not use_cpu_only:
                    rand_range = 10
                val = random_gen(input_shape, rand_min=-rand_range, rand_max=rand_range)
            elif test_mode == "sinh":
                res = tf.sinh
                val = random_gen(input_shape, rand_min=-10, rand_max=10)
            elif test_mode == "sqrt":
                res = tf.sqrt
                val = random_gen(input_shape, rand_min=0.5, rand_max=1000)
            elif test_mode == "square":
                res = tf.math.square
                val = random_gen(input_shape, rand_min=-5, rand_max=5)
            elif test_mode == "tan":
                res = tf.tan
                val = random_gen(input_shape, rand_min=-1000, rand_max=1000)
            elif test_mode == "tanh":
                res = tf.tanh
                val = random_gen(input_shape, rand_min=-1000, rand_max=1000)

            return res, val

        func, input_val = _get_test(mode)
        if func is None:
            return

        input_type = list(input_shape) + [tf_dtype]
        @make_tf_graph([input_type])
        def build_model(x):
            return func(x)

        model, inputs, outputs = build_model

        input_dict = dict(zip(inputs, [input_val.astype(dtype)]))

        if mode == "inverse" or mode == "rsqrt":
            atol, rtol = 1e-2, 1e-3

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=True,
            backend=backend,
            atol=atol,
            rtol=rtol,
            use_cpu_for_conversion=use_cpu_for_conversion,
        )


class TestImageResizing(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, input_shape, target_shape, align_corners, half_pixel_centers",
        itertools.product(
            [True, False],
            backends,
            [(1, 10, 20, 1), (2, 5, 1, 3)],
            [(25, 30), (2, 20)],
            [True, False],
            [True, False],
        ),
    )
    def test_resize_bilinear(
        self,
        use_cpu_only,
        backend,
        input_shape,
        target_shape,
        align_corners,
        half_pixel_centers,
    ):
        if half_pixel_centers and align_corners:
            return

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.raw_ops.ResizeBilinear(
                    images=x,
                    size=target_shape,
                    half_pixel_centers=half_pixel_centers,
                    align_corners=align_corners,
                )

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape, -100, 100)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, input_shape, upsample_factor, data_format",
        itertools.product(
            [True, False],
            backends,
            [(1, 1, 1, 3), (1, 10, 5, 3)],
            [(1, 2), (4, 3)],
            ["channels_last", "channels_first"],
        ),
    )
    def test_upsampling_2d(
        self, use_cpu_only, backend, input_shape, upsample_factor, data_format
    ):
        if data_format == "channels_last":
            input_shape = (
                input_shape[0],
                input_shape[2],
                input_shape[3],
                input_shape[1],
            )

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.keras.layers.UpSampling2D(
                    size=upsample_factor, data_format=data_format, interpolation="nearest"
                )(x)

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape, -100, 100)]
        input_dict = dict(zip(inputs, input_values))
        spec = TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )[0]
        # also check if the scale factor are integers
        if backend[0] == 'neuralnetwork':
            for layer in spec.neuralNetwork.layers:
                if layer.WhichOneof('layer') == "upsample":
                    assert len(layer.upsample.fractionalScalingFactor) == 0

    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend, input_shape, num_of_crops, crop_size, method, dynamic",
        itertools.product(
            [True, False],
            backends,
            [(1, 64, 64, 1)],
            [1, 3, 5],
            [(2, 2), (1, 1), (4, 4), (128, 128)],
            ["bilinear"],
            [False, True],
        ),
    )
    def test_crop_and_resize(
        self,
        use_cpu_for_conversion,
        backend,
        input_shape,
        num_of_crops,
        crop_size,
        method,
        dynamic,
    ):

        if backend[0] == "mlprogram" and not use_cpu_for_conversion and dynamic:
            pytest.xfail("Seg fault. rdar://77444115")

        use_cpu_only = use_cpu_for_conversion
        if backend[0] == "mlprogram" and not use_cpu_for_conversion and crop_size == (1, 1):
            # in this case, there is a numerical mismatch on the GPU MIL backend. The GPU runtime tests are
            # tracked seprately.
            use_cpu_only = True

        input = np.random.randn(*input_shape).astype(np.float32)
        boxes = np.random.uniform(size=(num_of_crops, 4)).astype(np.float32)
        box_indices = np.random.randint(
            size=(num_of_crops,), low=0, high=input_shape[0]
        ).astype(np.int32)

        def test_static():
            @make_tf_graph([input_shape])
            def build_model(x):
                return tf.raw_ops.CropAndResize(
                    image=x,
                    boxes=boxes,
                    box_ind=box_indices,
                    crop_size=crop_size,
                    method=method,
                )

            model, inputs, outputs = build_model
            input_values = [input]
            input_dict = dict(zip(inputs, input_values))
            TensorFlowBaseTest.run_compare_tf(
                model,
                input_dict,
                outputs,
                use_cpu_only=use_cpu_only,
                backend=backend,
                use_cpu_for_conversion=use_cpu_for_conversion,
            )

        def test_dynamic():
            @make_tf_graph([input_shape, boxes.shape, list(box_indices.shape) + [tf.int32]])
            def build_model(x, boxes_pl, box_indices_pl):
                return tf.raw_ops.CropAndResize(
                    image=x,
                    boxes=boxes_pl,
                    box_ind=box_indices_pl,
                    crop_size=crop_size,
                    method=method,
                )
            model, inputs, outputs = build_model
            input_values = [input, boxes, box_indices]
            input_dict = dict(zip(inputs, input_values))
            TensorFlowBaseTest.run_compare_tf(
                model,
                input_dict,
                outputs,
                use_cpu_only=use_cpu_only,
                backend=backend,
                use_cpu_for_conversion=use_cpu_for_conversion,
            )

        test_dynamic() if dynamic else test_static()

    @pytest.mark.parametrize(
        "use_cpu_only, backend, width, height, strides, sizes, padding,",
        list(
            itertools.product(
                [True, False],
                backends,
                [1, 3, 5],
                [2, 7, 12],
                [(1, 1), (2, 1), (3, 5)],
                [(1, 1), (1, 2), (5, 4)],
                ["VALID", "SAME"],
            )
        ),
    )
    def test_extract_patches(
        self, use_cpu_only, backend, width, height, strides, sizes, padding
    ):
        # TODO: theoritically, the current extractpatches code handle batch size rather than 1,
        # but there seems to have a bug in crop_resize when using GPU and batch_size > 1.
        # We should test batch_size > 1 after the issue is fixed.
        # <rdar://problem/61602238>
        input = np.random.rand(1, height, width, 128).astype(np.float32)
        if padding == "VALID":
            size_h = min(sizes[0], height)
            size_w = min(sizes[1], width)
        else:
            size_h = sizes[0]
            size_w = sizes[1]

        @make_tf_graph([input.shape])
        def build_model(x):
            return tf.compat.v1.image.extract_image_patches(
                images=x,
                ksizes=[1, size_h, size_w, 1],
                strides=[1, strides[0], strides[1], 1],
                rates=[1, 1, 1, 1],
                padding=padding,
            )
        model, inputs, outputs = build_model
        input_values = [input]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )


class TestLinear(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, dim, transpose_a, transpose_b, use_constant",
        itertools.product(
            [True, False],
            backends,
            [2, 4, 8],
            [True, False],
            [True, False],
            [True, False],
        ),
    )
    def test_matmul(
        self, use_cpu_only, backend, dim, transpose_a, transpose_b, use_constant
    ):
        shape_x = np.array([dim, dim * 2, dim * 4])
        shape_y = np.array([dim * 4, dim * 2])

        flip = (not transpose_a and transpose_b) or (transpose_a and not transpose_b)
        shape_y = np.flip(shape_y) if flip else shape_y

        if not use_constant:

            @make_tf_graph([shape_x, shape_y])
            def build_model(x, y):
                return tf.linalg.matmul(
                    x, y, transpose_a=transpose_a, transpose_b=transpose_b
                )

            input_values = [
                random_gen(shape=shape_x, rand_min=-100, rand_max=100),
                random_gen(shape=shape_y, rand_min=-1.0, rand_max=1.0),
            ]
        else:
            y = random_gen(shape=shape_y, rand_min=-1.0, rand_max=1.0)

            @make_tf_graph([shape_x])
            def build_model(x):
                return tf.linalg.matmul(
                    x, y, transpose_a=transpose_a, transpose_b=transpose_b
                )

            input_values = [random_gen(shape=shape_x, rand_min=-100, rand_max=100)]

        model, inputs, outputs = build_model

        input_dict = dict(zip(inputs, input_values))

        proto,_,_,_,_,_ = TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )

        for layer in proto.neuralNetwork.layers:
            if layer.WhichOneof("layer") == "batchedMatmul":
                wp = layer.batchedMatmul.weights
                if use_constant:
                    assert len(wp.floatValue) != 0
                else:
                    assert len(wp.floatValue) == 0


class TestBatchNormalization(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank, shape_mode, epsilon",
        itertools.product(
            [True, False],
            backends,
            [rank for rank in range(3, 6)],
            [True, False],
            [1e-1, 1e-10],
        ),
    )
    def test_batch_norm(self, use_cpu_only, backend, rank, shape_mode, epsilon):
        input_shape = np.random.randint(low=1, high=4, size=rank)
        if shape_mode:
            # same shape with 1 for being normalized over
            attr_shape = list(input_shape)
            attr_shape[1] = 1
            attr_shape[2] = 1
        else:
            # 1D tensor of the same size as channel dimension
            attr_shape = [list(input_shape)[-1]]

        @make_tf_graph([input_shape, attr_shape, attr_shape, attr_shape, attr_shape])
        def build_model(x, m, v, o, s):
            return tf.nn.batch_normalization(
                x, mean=m, variance=v, offset=o, scale=s, variance_epsilon=epsilon
            )

        model, inputs, outputs = build_model

        input_values = [
            random_gen(shape=input_shape, rand_min=-100.0, rand_max=100.0),
            random_gen(shape=attr_shape, rand_min=-1.0, rand_max=1.0),
            random_gen(shape=attr_shape, rand_min=0.0, rand_max=10.0),
            random_gen(shape=attr_shape, rand_min=1.0, rand_max=10.0),
            random_gen(shape=attr_shape, rand_min=-1.0, rand_max=1.0),
        ]
        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
            atol=.2,
            rtol=1e-4,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank, shape_mode, epsilon, scale_after_normalization",
        itertools.product(
            [True, False],
            backends,
            [rank for rank in range(3, 6)],
            [True, False],
            [1e-1, 1e-10],
            [True, False],
        ),
    )
    def test_batch_norm_with_global_normalization(
        self,
        use_cpu_only,
        backend,
        rank,
        shape_mode,
        epsilon,
        scale_after_normalization,
    ):
        input_shape = np.random.randint(low=1, high=4, size=rank)
        if shape_mode:
            # same shape with 1 for being normalized over
            attr_shape = list(input_shape)
            attr_shape[1] = 1
            attr_shape[2] = 1
        else:
            # 1D tensor of the same size as channel dimension
            attr_shape = [list(input_shape)[-1]]

        if scale_after_normalization:

            @make_tf_graph(
                [input_shape, attr_shape, attr_shape, attr_shape, attr_shape]
            )
            def build_model(x, m, v, b, g):
                return tf.nn.batch_norm_with_global_normalization(
                    x,
                    mean=m,
                    variance=v,
                    beta=b,
                    gamma=g,
                    variance_epsilon=epsilon,
                    scale_after_normalization=scale_after_normalization,
                )

        else:

            @make_tf_graph([input_shape, attr_shape, attr_shape, attr_shape])
            def build_model(x, m, v, b):
                return tf.nn.batch_norm_with_global_normalization(
                    x,
                    mean=m,
                    variance=v,
                    beta=b,
                    gamma=None,
                    variance_epsilon=epsilon,
                    scale_after_normalization=scale_after_normalization,
                )

        model, inputs, outputs = build_model

        input_values = [
            random_gen(shape=input_shape, rand_min=-100.0, rand_max=100.0),
            random_gen(shape=attr_shape, rand_min=-1.0, rand_max=1.0),
            random_gen(shape=attr_shape, rand_min=0.0, rand_max=10.0),
            random_gen(shape=attr_shape, rand_min=1.0, rand_max=10.0),
        ]
        if scale_after_normalization:
            input_values.append(
                random_gen(shape=attr_shape, rand_min=-1.0, rand_max=1.0)
            )

        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
            atol=0.2,
            rtol=1e-4,
        )


class TestNormalization(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, epsilon",
        itertools.product([True, False], backends, [1e-1, 1e-10]),
    )
    def test_fused_batch_norm(self, use_cpu_only, backend, epsilon):
        # TensorFlow's FusedBatchNorm is only for 4D inputs
        input_shape = np.random.randint(low=1, high=4, size=4)
        attr_shape = [list(input_shape)[-1]]

        m = random_gen(shape=attr_shape, rand_min=-1.0, rand_max=1.0)
        v = random_gen(shape=attr_shape, rand_min=0.0, rand_max=10.0)
        o = random_gen(shape=attr_shape, rand_min=1.0, rand_max=10.0)
        s = random_gen(shape=attr_shape, rand_min=-1.0, rand_max=1.0)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.compat.v1.nn.fused_batch_norm(
                x,
                mean=m,
                variance=v,
                offset=o,
                scale=s,
                epsilon=epsilon,
                is_training=False,
            )[0]

        model, inputs, outputs = build_model

        input_values = [random_gen(shape=input_shape, rand_min=-100.0, rand_max=100.0)]

        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
            atol=1e-2,
            rtol=1e-3,
        )

class TestL2Normalization(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank, axes, epsilon",
        itertools.product(
            [True, False],
            backends,
            [rank for rank in range(3, 6)],
            [(-1,), (-2,), (0, 1)],
            [1e-5, 1e-10],
        ),
    )
    def test_l2_normalize(self, use_cpu_only, backend, rank, axes, epsilon):
        input_shape = np.random.randint(low=1, high=4, size=rank)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.math.l2_normalize(x, axis=axes, epsilon=epsilon)

        model, inputs, outputs = build_model

        input_values = [random_gen(input_shape, rand_min=-10, rand_max=10)]

        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
            atol=0.05,
            rtol=1e-4,
        )

class TestLocalResponseNormalization(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, size, alpha, beta, k",
        itertools.product(
            [True, False], backends, [1, 2, 3], [0.0001, 0.01], [0.75, 1.0], [1.0, 2.0],
        ),
    )
    def test_local_response_normalization(
        self, use_cpu_only, backend, size, alpha, beta, k
    ):
        # TensorFlow's local_response_normalization only supports rank 4
        input_shape = np.random.randint(low=3, high=4, size=4)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.nn.local_response_normalization(
                x, depth_radius=size, bias=k, alpha=alpha, beta=beta
            )

        model, inputs, outputs = build_model

        input_values = [random_gen(shape=input_shape, rand_min=-100, rand_max=100)]

        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
            atol=1e-2,
            rtol=1e-3,
        )


class TestPool1d(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, kernel_sizes, strides, pad_type",
        itertools.product(
            [True, False], backends, [(1,)], [(1,), (2,)], ["same", "valid"]
        ),
    )
    def test_avg_pool_1d(self, use_cpu_only, backend, kernel_sizes, strides, pad_type):
        input_shape = np.random.randint(low=2, high=4, size=3)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.nn.avg_pool1d(
                x, ksize=kernel_sizes[:], strides=strides[:], padding=pad_type.upper()
            )

        model, inputs, outputs = build_model
        input_values = [random_gen(shape=input_shape, rand_min=-100, rand_max=100)]
        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, kernel_sizes, strides, pad_type",
        itertools.product(
            [True, False], backends, [(1,)], [(1,), (2,)], ["same", "valid"]
        ),
    )
    def test_max_pool_1d(self, use_cpu_only, backend, kernel_sizes, strides, pad_type):
        input_shape = np.random.randint(low=2, high=4, size=3)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.nn.max_pool1d(
                x, ksize=kernel_sizes[:], strides=strides[:], padding=pad_type.upper()
            )

        model, inputs, outputs = build_model
        input_values = [random_gen(shape=input_shape, rand_min=-100, rand_max=100)]
        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )


class TestPool2d(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, kernel_sizes, strides, pad_type",
        itertools.product(
            [True, False],
            backends,
            [(1,), (2,), (1, 1), (1, 2), (2, 2)],
            [(1,), (2,), (1, 1), (1, 2), (2, 2)],
            ["same", "valid"],
        ),
    )
    def test_avg_pool_2d(self, use_cpu_only, backend, kernel_sizes, strides, pad_type):
        input_shape = np.random.randint(low=2, high=4, size=4)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.nn.avg_pool(
                x, ksize=kernel_sizes[:], strides=strides[:], padding=pad_type.upper()
            )

        model, inputs, outputs = build_model
        input_values = [random_gen(shape=input_shape, rand_min=-100, rand_max=100)]
        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, kernel_sizes, strides, pad_type",
        itertools.product(
            [True, False],
            backends,
            [(1,), (2,), (1, 1), (1, 2), (2, 2)],
            [(1,), (2,), (1, 1), (1, 2), (2, 2)],
            ["same", "valid"],
        ),
    )
    def test_max_pool_2d(self, use_cpu_only, backend, kernel_sizes, strides, pad_type):
        input_shape = np.random.randint(low=2, high=4, size=4)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.nn.max_pool(
                x, ksize=kernel_sizes[:], strides=strides[:], padding=pad_type.upper()
            )

        model, inputs, outputs = build_model
        input_values = [random_gen(shape=input_shape, rand_min=-100, rand_max=100)]
        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )


class TestPool3d(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, kernel_sizes, strides, pad_type",
        itertools.product(
            [True, False],
            backends,
            [(1,), (2,), (1, 1, 1), (1, 2, 3), (2, 2, 3), (3, 3, 3)],
            [(1,), (2,), (1, 1, 1), (1, 2, 3), (2, 2, 3), (3, 3, 3)],
            ["same", "valid"],
        ),
    )
    def test_avg_pool_3d(self, use_cpu_only, backend, kernel_sizes, strides, pad_type):
        input_shape = np.random.randint(low=3, high=4, size=5)

        if kernel_sizes[0] == 1 and pad_type == "same":
            pytest.xfail("rdar://81630684 (Pool3d with pad type == same fails from TF2.5 onwards)")

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.nn.avg_pool3d(
                x, ksize=kernel_sizes[:], strides=strides[:], padding=pad_type.upper()
            )

        model, inputs, outputs = build_model
        input_values = [random_gen(shape=input_shape, rand_min=-100, rand_max=100)]
        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, kernel_sizes, strides, pad_type",
        itertools.product(
            [True, False],
            backends,
            [(1,), (2,), (1, 1, 1), (1, 2, 3), (2, 2, 3), (3, 3, 3)],
            [(1,), (2,), (1, 1, 1), (1, 2, 3), (2, 2, 3), (3, 3, 3)],
            ["same", "valid"],
        ),
    )
    def test_max_pool_3d(self, use_cpu_only, backend, kernel_sizes, strides, pad_type):
        input_shape = np.random.randint(low=3, high=4, size=5)

        if kernel_sizes[0] == 1 and pad_type == "same":
            pytest.xfail("rdar://81630684 (Pool3d with pad type == same fails from TF2.5 onwards)")

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.nn.max_pool3d(
                x, ksize=kernel_sizes[:], strides=strides[:], padding=pad_type.upper()
            )

        model, inputs, outputs = build_model
        input_values = [random_gen(shape=input_shape, rand_min=-100, rand_max=100)]
        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )


class TestPrint(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [size for size in range(1, 5)],),
    )
    def test_print(self, use_cpu_only, backend, rank):
        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)

        @make_tf_graph([shape])
        def build_model(x):
            print_layer = tf.raw_ops.Print(input=x, data=[])
            res = print_layer + 1
            return res

        model, inputs, outputs = build_model
        input_value = [random_gen(shape=shape, rand_min=-100, rand_max=100)]
        input_dict = dict(zip(inputs, input_value))

        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )


class TestRandom(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, size, rank, constant",
        itertools.product(
            [True, False],
            backends,
            [size for size in range(1, 5)],
            [rank for rank in range(1, 6)],
            [True, False],
        ),
    )
    def test_random_binomial(self, use_cpu_only, backend, size, rank, constant):
        if not constant and backend[0] != "neuralnetwork":
            return  # dynamic input is only support in neuralnetwork backend

        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            if constant:
                ref = tf.add(x, tf.keras.backend.random_binomial(shape=shape, p=1.0))
            else:
                ref = tf.add(
                    x,
                    tf.keras.backend.random_binomial(
                        shape=tf.raw_ops.Shape(input=x), p=1.0
                    ),
                )
            TensorFlowBaseTest.run_compare_tf(
                graph,
                {x: np.random.rand(*shape)},
                ref,
                use_cpu_only=use_cpu_only,
                backend=backend,
            )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, size",
        itertools.product([True, False], backends, [size for size in range(1, 5)]),
    )
    def test_random_categorical(self, use_cpu_only, backend, size):
        # TensorFlow's input is 2-D tensor with shape [batch_size, num_classes].
        shape = np.random.randint(low=1, high=4, size=2)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.random.categorical(x, size)
            TensorFlowBaseTest.run_compare_tf(
                graph,
                {x: np.random.rand(*shape)},
                ref,
                use_cpu_only=use_cpu_only,
                validate_shapes_only=True,
                backend=backend,
            )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, mean, rank, constant",
        itertools.product(
            [True, False],
            backends,
            [0.0],
            [rank for rank in range(1, 6)],
            [True, False],
        ),
    )
    def test_random_normal(self, use_cpu_only, backend, mean, rank, constant):
        if not constant and backend[0] != "neuralnetwork":
            return  # dynamic input is only support in neuralnetwork backend

        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            if constant:
                ref = tf.add(x, tf.random.normal(shape=shape, mean=mean, stddev=0.0))
            else:
                ref = tf.add(
                    x,
                    tf.random.normal(
                        shape=tf.raw_ops.Shape(input=x), mean=mean, stddev=0.0
                    ),
                )
            TensorFlowBaseTest.run_compare_tf(
                graph,
                {x: np.random.rand(*shape)},
                ref,
                use_cpu_only=use_cpu_only,
                backend=backend,
            )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, mean, rank, constant",
        itertools.product(
            [True, False],
            backends,
            [0.0],
            [rank for rank in range(1, 6)],
            [True, False],
        ),
    )
    def test_keras_random_normal(self, use_cpu_only, backend, mean, rank, constant):
        if not constant and backend[0] != "neuralnetwork":
            return  # dynamic input is only support in neuralnetwork backend

        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            if constant:
                ref = tf.add(
                    x,
                    tf.keras.backend.random_normal(shape=shape, mean=mean, stddev=0.0),
                )
            else:
                ref = tf.add(
                    x,
                    tf.keras.backend.random_normal(
                        shape=tf.raw_ops.Shape(input=x), mean=mean, stddev=0.0
                    ),
                )
            TensorFlowBaseTest.run_compare_tf(
                graph,
                {x: np.random.rand(*shape)},
                ref,
                use_cpu_only=use_cpu_only,
                backend=backend,
            )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, low, high, rank, constant",
        itertools.product(
            [True, False],
            backends,
            [0.0],
            [0.0],
            [rank for rank in range(1, 2)],
            [True, False],
        ),
    )
    def test_random_uniform(self, use_cpu_only, backend, low, high, rank, constant):
        if not constant and backend[0] != "neuralnetwork":
            return  # dynamic input is only support in neuralnetwork backend

        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            if constant:
                ref = tf.add(x, tf.random.uniform(shape=shape, minval=low, maxval=high))
            else:
                ref = tf.add(
                    x,
                    tf.random.uniform(
                        shape=tf.raw_ops.Shape(input=x), minval=low, maxval=high
                    ),
                )
            TensorFlowBaseTest.run_compare_tf(
                graph,
                {x: np.random.rand(*shape)},
                ref,
                use_cpu_only=use_cpu_only,
                backend=backend,
            )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, low, high, rank, constant",
        itertools.product(
            [True, False],
            backends,
            [1.0],
            [1.0],
            [rank for rank in range(1, 6)],
            [True, False],
        ),
    )
    def test_keras_random_uniform(
        self, use_cpu_only, backend, low, high, rank, constant
    ):
        if not constant and backend[0] != "neuralnetwork":
            return  # dynamic input is only support in neuralnetwork backend
        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            if constant:
                ref = tf.add(
                    x,
                    tf.keras.backend.random_uniform(
                        shape=shape, minval=low, maxval=high
                    ),
                )
            else:
                ref = tf.add(
                    x,
                    tf.keras.backend.random_uniform(
                        shape=tf.raw_ops.Shape(input=x), minval=low, maxval=high
                    ),
                )
            TensorFlowBaseTest.run_compare_tf(
                graph,
                {x: np.random.rand(*shape)},
                ref,
                use_cpu_only=use_cpu_only,
                backend=backend,
            )


@pytest.mark.skipif(_macos_version() < (10, 16),
                    reason="This only works for 'neuralnetwork' on macOS 11")
class TestReduction(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend, rank_and_axes, keep_dims, tf_op",
        itertools.product(
            [True, False],
            backends,
            [
                (1, (-1,)),
                (2, (0,)),
                (2, (-1, 0)),
                (3, (1, -3)),
                (3, (-2,)),
                (3, (-3, -2, -1)),
                (4, (0, 1, 2)),
                (4, (-2, -1, 0)),
                (4, (1, -2)),
                (5, (-3, -1)),
                (5, (-2, -1)),
                (5, (-3, -2, -1)),
                (5, (0, -1, 1, -2)),
                (3, None),
                (5, None),
                (3, 1),
            ],
            [True, False],
            [
                tf.reduce_all,
                tf.math.reduce_euclidean_norm,
                tf.reduce_max,
                tf.reduce_mean,
                tf.reduce_min,
                tf.reduce_prod,
                tf.reduce_sum,
                tf.reduce_any,
                tf.reduce_logsumexp,
                tf.math.argmax,
                tf.math.argmin,
            ],
        ),
    )
    def test_reduction(self, use_cpu_for_conversion, backend, rank_and_axes, keep_dims, tf_op):
        rank, axes = rank_and_axes
        shape = np.random.randint(low=1, high=3, size=rank)

        if backend[0] == 'mlprogram' and not use_cpu_for_conversion:
            if rank_and_axes == (5, None) or tf_op in {tf.reduce_logsumexp}:
                pytest.xfail("Seg fault. rdar://77443572")

        def parse_axes(axes):
            if axes is None:
                axes = 0
            elif isinstance(axes, (tuple, list)):
                axes = axes[0]
            return axes

        def test_tf_argmax():
            @make_tf_graph([shape])
            def build_model(x):
                return tf.math.argmax(x, axis=parse_axes(axes))

            model, inputs, outputs = build_model
            input_values = [random_gen(shape, rand_min=-5.0, rand_max=5.0)]
            input_dict = dict(zip(inputs, input_values))
            TensorFlowBaseTest.run_compare_tf(
                model,
                input_dict,
                outputs,
                use_cpu_only=use_cpu_for_conversion,
                backend=backend,
                use_cpu_for_conversion=use_cpu_for_conversion,
            )

        def test_tf_argmin():
            @make_tf_graph([shape])
            def build_model(x):
                return tf.math.argmin(x, axis=parse_axes(axes))

            model, inputs, outputs = build_model
            input_values = [random_gen(shape, rand_min=-5.0, rand_max=5.0)]
            input_dict = dict(zip(inputs, input_values))
            TensorFlowBaseTest.run_compare_tf(
                model,
                input_dict,
                outputs,
                use_cpu_only=use_cpu_for_conversion,
                backend=backend,
                use_cpu_for_conversion=use_cpu_for_conversion,
            )

        def test_tf_reduction():
            if isinstance(axes, list) and axes and len(axes) == rank and not keep_dims:
                return

            input_type = list(shape)
            x_val = random_gen(shape=shape, rand_min=-5.0, rand_max=5.0)
            if tf_op in {tf.reduce_all, tf.reduce_any}:
                input_type += [tf.bool]
                x_val = np.random.randint(low=0, high=2, size=shape).astype(
                    np.bool
                )
            elif tf_op in {tf.math.reduce_euclidean_norm}:
                x_val = random_gen(shape=shape, rand_min=0.0, rand_max=10.0)
            elif tf_op in {tf.reduce_prod}:
                x_val = random_gen(shape=shape, rand_min=1.0, rand_max=1.3)
            elif tf_op in {tf.reduce_logsumexp}:
                x_val = random_gen(shape=shape, rand_min=-5, rand_max=5)

            @make_tf_graph([input_type])
            def build_model(x):
                ref = tf_op(x, axis=axes, keepdims=keep_dims)
                if tf_op == tf.reduce_any:
                    ref = tf.cast(ref, tf.float32)
                return ref

            model, inputs, outputs = build_model
            input_dict = dict(zip(inputs, [x_val]))
            TensorFlowBaseTest.run_compare_tf(
                model,
                input_dict,
                outputs,
                use_cpu_only=use_cpu_for_conversion,
                backend=backend,
                use_cpu_for_conversion=use_cpu_for_conversion,
            )

        if tf_op in {tf.math.argmax}:
            test_tf_argmax()
        elif tf_op in {tf.math.argmin}:
            test_tf_argmin()
        else:
            test_tf_reduction()

class TestGather(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rankX_rankIndices_axis, mode",
        itertools.product(
            [True, False],
            backends,
            [
                (1, 2, -1),
                (2, 1, 0),
                (3, 2, -2),
                (2, 3, 1),
                (2, 2, 1),
                (1, 1, 0),
                (3, 3, -2),
                (3, 3, 2),
                (3, 3, 0),
                (1, 3, -1),
                (3, 1, 2),
                (3, 1, -1),
            ],
            ["Gather", "GatherV2", "gather"],
        ),
    )
    def test_gather_function(self, use_cpu_only, backend, rankX_rankIndices_axis, mode):
        x_rank, indices_rank, axis = rankX_rankIndices_axis
        x_shape = np.random.randint(low=2, high=4, size=x_rank)
        indices_shape = np.random.randint(low=2, high=4, size=indices_rank)

        @make_tf_graph([x_shape, list(indices_shape) + [tf.int32]])
        def build_model(x, indices):
            if mode == "Gather":
                res = tf.raw_ops.Gather(params=x, indices=indices)
            elif mode == "GatherV2":
                res = tf.raw_ops.GatherV2(params=x, indices=indices, axis=axis)
            elif mode == "gather":
                res = tf.gather(x, indices, axis=axis)

            return res

        model, inputs, outputs = build_model

        axis = 0 if mode == "Gather" else axis
        input_dict = {inputs[0]: np.random.rand(*x_shape).astype(np.float32),
                       inputs[1]: np.random.randint(0, x_shape[axis], size=indices_shape, dtype=np.int32)}

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, rankX_rankIndices",
        itertools.product(
            [True, False],
            backends,
            [
                (1, 2),
                (2, 2),
                (3, 2),
                (2, 3),
                (1, 4),
                (5, 2),
                (2, 5),
                (4, 3),
                (3, 4),
                (2, 4),
                (4, 2),
                (1, 5),
            ],
        ),
    )
    def test_gather_nd(self, use_cpu_only, backend, rankX_rankIndices):
        x_rank, indices_rank = rankX_rankIndices
        x_shape = np.random.randint(low=2, high=4, size=x_rank)
        indices_shape = np.random.randint(low=2, high=4, size=indices_rank)
        indices_shape[-1] = np.random.randint(low=1, high=x_rank + 1)

        @make_tf_graph([x_shape, list(indices_shape) +[tf.int32]])
        def build_model(x, indices):
            return tf.gather_nd(x, indices)

        model, inputs, outputs = build_model

        a = np.random.rand(*x_shape).astype(np.float32)
        indices_list = []
        for i in range(indices_shape[-1]):
            indices_list.append(
                np.random.randint(0, x_shape[i], size=indices_shape[:-1])
            )

        input_dict = {
            inputs[0]: a,
            inputs[1]: np.stack(indices_list, axis=-1).astype(np.int32),
        }

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

class TestScatter(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, data_rank, indices_rank",
        itertools.product(
            [True, False], backends, list(range(1, 4)), list(range(2, 4)),
        ),
    )
    def test_scatter_nd_with_zeros(
        self, use_cpu_only, backend, data_rank, indices_rank
    ):

        shape = np.random.randint(low=2, high=4, size=data_rank).astype(np.int32)
        indices_shape = np.random.randint(low=2, high=4, size=indices_rank)
        indices_shape[-1] = np.random.randint(low=1, high=data_rank + 1)
        updates_shape = list(indices_shape[:-1]) + list(shape[indices_shape[-1] :])

        updates = np.random.rand(*updates_shape).astype(np.int32)
        indices_list = []
        for i in range(indices_shape[-1]):
            indices_list.append(np.random.randint(0, shape[i], size=indices_shape[:-1]))

        indices = np.stack(indices_list, axis=-1).astype(np.int32)

        @make_tf_graph(
            [list(indices.shape) + [tf.int32], updates_shape + [tf.int32], [data_rank, tf.int32]]
        )
        def build_model(indices, updates, shape):
            return tf.raw_ops.ScatterNd(indices=indices, updates=updates, shape=shape)

        model, inputs, outputs = build_model
        input_values = [indices, updates, shape]

        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestSliceByIndex(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank, masking",
        itertools.product(
            [True, False],
            backends,
            [rank for rank in range(1, 5)],
            [True, False]
        ),
    )
    def test_slice_by_index(self, use_cpu_only, backend, rank, masking):
        input_shape = np.random.randint(low=2, high=4, size=rank)
        begin_val = np.array(
            [
                np.random.randint(low=-input_shape[i], high=input_shape[i])
                for i in range(rank)
            ]
        ).astype(np.int32)
        end_val = np.array(
            [
                np.random.randint(low=-input_shape[i], high=input_shape[i])
                for i in range(rank)
            ]
        ).astype(np.int32)
        stride_val = np.array(
            [
                np.random.randint(low=-input_shape[i], high=input_shape[i])
                for i in range(rank)
            ]
        ).astype(np.int32)
        if not masking:
            begin_mask = [False] * rank
            end_mask = [False] * rank
            squeeze_mask = [False] * rank
        else:
            begin_mask = np.array(
                [np.random.choice([True, False, False]) for i in range(rank)]
            ).astype(np.bool)
            end_mask = np.array(
                [np.random.choice([True, False, False]) for i in range(rank)]
            ).astype(np.bool)
            squeeze_flag = True
            # We do not squeeze to scalar in nn
            while squeeze_flag:
                squeeze_mask = np.array(
                    [np.random.choice([True, False]) for i in range(rank)]
                ).astype(np.bool)
                for i in range(rank):
                    if begin_mask[i] or end_mask[i]:
                        squeeze_mask[i] = False
                for s in squeeze_mask:
                    if not s:
                        squeeze_flag = False

        for i in range(rank):
            if begin_mask[i] or end_mask[i]:
                stride = 0
                while stride == 0:
                    stride = np.random.randint(low=-input_shape[i], high=input_shape[i])
                stride_val[i] = stride

                if not end_mask[i]:
                    while True:
                        end = np.random.randint(
                            low=-input_shape[i], high=input_shape[i]
                        )
                        normalized_end = input_shape[i] + end if end < 0 else end
                        if normalized_end == 0 and stride_val[i] > 0:
                            continue
                        elif normalized_end == input_shape[i] - 1 and stride_val[i] < 0:
                            continue
                        else:
                            end_val[i] = end
                            break
                continue
            if squeeze_mask[i]:
                stride_val[i] = 1
            while True:
                end = np.random.randint(low=-input_shape[i], high=input_shape[i])
                normalized_end = input_shape[i] + end if end < 0 else end
                normalized_begin = (
                    input_shape[i] + begin_val[i] if begin_val[i] < 0 else begin_val[i]
                )
                if normalized_end == normalized_begin:
                    continue
                if begin_mask[i] or end_mask[i] or squeeze_mask[i]:
                    stride = 1
                elif normalized_end < normalized_begin:
                    stride = -np.random.randint(low=1, high=input_shape[i])
                else:
                    stride = np.random.randint(low=1, high=input_shape[i])
                end_val[i] = end
                stride_val[i] = stride
                break

        def _mask_to_bit(mask):
            ret = 0
            for x in mask[::-1]:
                ret <<= 1
                if x:
                    ret += 1
            return ret

        @make_tf_graph(
            [
                input_shape,
                list(begin_val.shape) + [tf.int32],
                list(end_val.shape) + [tf.int32],
            ]
        )
        def build_model(x, begin, end):
            return tf.strided_slice(
                x,
                begin,
                end,
                stride_val,
                begin_mask=_mask_to_bit(begin_mask),
                end_mask=_mask_to_bit(end_mask),
                shrink_axis_mask=_mask_to_bit(squeeze_mask),
            )

        model, inputs, outputs = build_model

        input_values = [
            np.array(list(range(np.prod(input_shape))))
            .reshape(input_shape)
            .astype(np.float32),
            begin_val,
            end_val,
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, testcase",
        itertools.product(
            [True, False],
            backends,
            # Change to slice representation for allowing iteration with a non-constant input
            [
                (
                    slice(1, 2),
                    slice(1, 2),
                    slice(1, 2),
                ),  # equivalent to [1:2, 1:2, 1:2]
                (slice(-3, -2), slice(-4, -3), slice(-5, -4)),
                (slice(0, -2), slice(0, -1), slice(-3, -2)),
                (slice(-1, 0, -2), slice(-1, 1, -1), slice(-1, -3, -3)),
                (slice(1, 2), slice(1, 3), slice(1, 4, 2)),
                (slice(None, 2), slice(1, 3), slice(None, 4, 2)),
                (
                    slice(None),
                    slice(1, None),
                    slice(None, 4, 2),
                ),  # equivalent to [:,1:,:4:2]
                (slice(1, None, 1), 1, slice(None, 3, 2)),
                (slice(None), slice(None), slice(None)),
                (slice(1, 2), slice(1, 2), 1),
                (slice(1, 2), slice(None), slice(None)),
                (slice(None), slice(None), slice(None)),
                (slice(1, 2), slice(None), slice(1, 2)),
                (slice(None), slice(None), 1),
                (0, 0, slice(None)),
                (slice(1, 2)),
                (slice(1, 2), slice(1, 2)),
                (1),
                (slice(0, 3)),
                (slice(None)),
                (slice(None), slice(None), slice(None, None, -1)),
            ],
        ),
    )
    def test_slice_by_index_from_scratch(self, use_cpu_only, backend, testcase):
        input_shape = np.array([3, 4, 5])

        if backend == ("mlprogram", "fp16") and testcase == (slice(None, None, None), slice(None, None, None), slice(None, None, -1)):
            pytest.xfail("rdar://80661727 (SliceByIndex FP16 TF converter unit test failing on one particular configuration)")

        @make_tf_graph([input_shape])
        def build_model(x):
            return x[testcase]

        model, inputs, outputs = build_model

        input_values = [
            np.array(list(range(np.prod(input_shape))))
            .reshape(input_shape)
            .astype(np.float32)
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, shape_and_slice",
        itertools.product(
            [True, False],
            backends,
            [
                [[3], (slice(1, 2))],
                [[2,10], (slice(0, 2), slice(None, 8, 2))],
                [[2,3,4,5], (slice(None), slice(1, None, 3), slice(None), slice(0, 5))],
                [[2,3,4,5], (slice(0, None), slice(None), slice(2, None, 1), slice(None))],
            ],
        ),
    )
    def test_slice_by_index_one_dimension(self, use_cpu_only, backend, shape_and_slice):
        input_shape, testcase = shape_and_slice

        @make_tf_graph([input_shape])
        def build_model(x):
            return x[testcase]

        model, inputs, outputs = build_model

        input_values = [
            np.array(list(range(np.prod(input_shape))))
            .reshape(input_shape)
            .astype(np.float32)
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends)
    )
    def test_slice_by_index_smoke(self, use_cpu_only, backend):
        input_shape = [1, 64, 2]
        x_val = np.random.rand(*input_shape).astype(np.float32)
        y_val = np.random.rand(*input_shape).astype(np.float32)

        @make_tf_graph([input_shape, input_shape])
        def build_model(x, y):
            x_slice = x[:, :, 0]
            y_slice = y[:, :, 0]
            return (x_slice, y_slice)

        model, inputs, outputs = build_model

        input_values = [x_val, y_val]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

    @pytest.mark.xfail(reason="ExpandDims exist mismatch", run=False)
    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends)
    )
    def test_slice_by_index_with_new_axes(self, use_cpu_only, backend):
        input_shape = [4, 5, 64]
        val = np.random.rand(*input_shape).astype(np.float32)
        num_cases = 8

        @make_tf_graph([input_shape] * num_cases)
        def build_model(*args):
            a, b, c, d, e, f, g, h = args
            slice_0 = a[:1, tf.newaxis, :3, :]
            slice_1 = b[:, tf.newaxis]
            slice_2 = c[..., tf.newaxis]
            slice_3 = d[..., tf.newaxis, :, 10]
            slice_4 = e[:, 2, tf.newaxis, ...]
            slice_5 = f[2, ..., :, tf.newaxis]
            slice_6 = g[tf.newaxis, ..., tf.newaxis]
            slice_7 = h[tf.newaxis, 2, tf.newaxis, ...]

            return (
                slice_0,
                slice_1,
                slice_2,
                slice_3,
                slice_4,
                slice_5,
                slice_6,
                slice_7,
            )

        model, inputs, outputs = build_model

        input_values = [val] * num_cases
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestSliceBySize(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank, single_size, dynamic_size",
        itertools.product(
            [True, False],
            backends,
            [rank for rank in range(1, 5)],
            [True, False],
            [True, False],
        ),
    )
    def test_dynamic_slice_by_size(
        self, use_cpu_only, backend, rank, single_size, dynamic_size
    ):
        # Test for when either begin or size are runtime determines
        input_shape = np.random.randint(low=2, high=4, size=rank)
        begin_val = np.array(
            [np.random.randint(input_shape[i]) for i in range(rank)]
        ).astype(np.int32)
        size_val = np.array(
            [np.random.randint(input_shape[i] - begin_val[i]) + 1 for i in range(rank)]
        )
        if single_size:
            for r in range(rank):
                size_val_r = np.array(
                    [s if i == r else -1 for i, s in enumerate(size_val)]
                ).astype(np.int32)

                @make_tf_graph([input_shape, list(begin_val.shape) + [tf.int32]])
                def build_model(x, begin):
                    return tf.slice(x, begin, size_val_r)

                @make_tf_graph(
                    [
                        input_shape,
                        list(begin_val.shape) + [tf.int32],
                        list(size_val_r.shape) + [tf.int32],
                    ]
                )
                def build_model_dynamic_size(x, begin, size):
                    return tf.slice(x, begin, size)

                if dynamic_size:
                    model, inputs, outputs = build_model_dynamic_size
                    input_values = [
                        random_gen(input_shape, rand_min=-100, rand_max=100),
                        begin_val,
                        size_val_r,
                    ]
                else:
                    model, inputs, outputs = build_model
                    input_values = [
                        random_gen(input_shape, rand_min=-100, rand_max=100),
                        begin_val,
                    ]

                input_dict = dict(zip(inputs, input_values))
                TensorFlowBaseTest.run_compare_tf(
                    model,
                    input_dict,
                    outputs,
                    use_cpu_only=use_cpu_only,
                    frontend_only=False,
                    backend=backend,
                )
        else:
            size_val = np.array(
                [s if np.random.randint(2) == 0 else -1 for s in size_val]
            ).astype(np.int32)

            @make_tf_graph([input_shape, list(begin_val.shape) + [tf.int32]])
            def build_model(x, begin):
                return tf.slice(x, begin, size_val)

            @make_tf_graph(
                [
                    input_shape,
                    list(begin_val.shape) + [tf.int32],
                    list(size_val.shape) + [tf.int32],
                ]
            )
            def build_model_dynamic_size(x, begin, size):
                return tf.slice(x, begin, size)

            if dynamic_size:
                model, inputs, outputs = build_model_dynamic_size
                input_values = [
                    random_gen(input_shape, rand_min=-100, rand_max=100),
                    begin_val,
                    size_val,
                ]
            else:
                model, inputs, outputs = build_model
                input_values = [
                    random_gen(input_shape, rand_min=-100, rand_max=100),
                    begin_val,
                ]

            input_dict = dict(zip(inputs, input_values))
            TensorFlowBaseTest.run_compare_tf(
                model,
                input_dict,
                outputs,
                use_cpu_only=use_cpu_only,
                frontend_only=False,
                backend=backend,
            )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, begin_size",
        itertools.product(
            [True, False],
            backends,
            [
                [[0, 1, 2], [1, 1, 1]],
                [[0, 0, 0], [-1, -1, -1]],
                [[0, 0, 1], [1, 2, -1]],
                [[0, 1, 2], [-1, -1, -1]],
            ]
        ),
    )
    def test_static_slice_by_size(
        self, use_cpu_only, backend, begin_size
    ):
        # Test for when begin and size are both constant
        input_shape = [1, 2, 3]
        begin, size = begin_size
        tf_input_shape = input_shape.copy()

        for i in range(3):
            if np.random.randint(2) == 0:
                tf_input_shape[i] = None
                # We set the begin to 0 for the symbolic dimension,
                # since the default input shape will be 1 in this case,
                # we need to make sure that begin = 0 and size = 1 (unless size == -1)
                begin[i] = 0
                if size[i] != -1:
                    size[i] = 1

        @make_tf_graph([tf_input_shape])
        def build_model(x):
            return tf.slice(x, begin, size)

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape, rand_min=-2, rand_max=2)]

        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestMatrixBandPart(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank, lower_and_upper",
        itertools.product(
            [True, False],
            backends,
            [rank for rank in range(2, 6)],
            [(0, -1), (-1, 0), (0, 0)],
        ),
    )
    def test_matrix_band_part(self, use_cpu_only, backend, rank, lower_and_upper):

        lower, upper = lower_and_upper
        shape = np.random.randint(low=3, high=4, size=rank)

        @make_tf_graph([shape])
        def build_model(x):
            return tf.raw_ops.MatrixBandPart(input=x, num_lower=lower, num_upper=upper)

        model, inputs, outputs = build_model
        TensorFlowBaseTest.run_compare_tf(
            model,
            {inputs[0]: random_gen(shape, rand_min=-100, rand_max=100)},
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )


class TestCumSum(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank, reverse, exclusive",
        itertools.product(
            [True, False],
            backends,
            [rank for rank in range(1, 6)],
            [True, False],
            [True, False],
        ),
    )
    def test_cumsum(self, use_cpu_only, backend, rank, reverse, exclusive):
        input_shape = np.random.randint(low=1, high=4, size=rank)
        for axis in range(-1, rank, 3):
            @make_tf_graph([input_shape])
            def build_model(x):
                return tf.math.cumsum(x, axis=axis, reverse=reverse, exclusive=exclusive)

            model, inputs, outputs = build_model
            input_values = [random_gen(input_shape, rand_min=-10, rand_max=10)]
            input_dict = dict(zip(inputs, input_values))

            TensorFlowBaseTest.run_compare_tf(model,
                           input_dict,
                           outputs,
                           use_cpu_only=use_cpu_only,
                           frontend_only=False,
                           backend=backend)


class TestFakeQuant(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "num_bits, weight_boundaries, use_cpu_only, backend",
        itertools.product(
            [bits for bits in range(2, 9)],  # TensorFlow does not support 1-bit quantization
            [(-10, 0), (0, 10), (-0.01, 0.02), (-0.001, 0.003), (-101, 100)],
            [True, False],
            backends,
        ),
    )
    def test_fake_quant_weight_quantization_with_conv(self, num_bits, weight_boundaries, use_cpu_only, backend):
        if backend[0] == 'mlprogram':
            pytest.xfail("Not supported with ML Program backend")

        tf.reset_default_graph()
        filter_width = 1
        filter_height = 1
        spatial_size = 2
        input_channels = 3
        output_channels = 1
        input_tensor = tf.placeholder(tf.float32, [1, spatial_size, spatial_size, input_channels], name='input')
        output_tensor = tf.placeholder(tf.float32, [1, spatial_size, spatial_size, output_channels], name='output')
        kernel_in = random_gen((filter_width, filter_height), weight_boundaries[0], weight_boundaries[1])
        init = tf.constant_initializer(kernel_in)

        def model(x):
            with tf.compat.v1.variable_scope('quantized_model'):
                x = tf.layers.conv2d(x, filters=3, kernel_size=1, strides=1, kernel_initializer=init)
                return x

        with tf.compat.v1.variable_scope('quantize'):
            output = model(x=input_tensor)
        tf.contrib.quantize.experimental_create_training_graph(quant_delay=0, weight_bits=num_bits,
                                                               activation_bits=num_bits)
        loss = tf.losses.mean_squared_error(labels=input_tensor, predictions=output)
        saver = tf.train.Saver()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer().minimize(loss)

        checkpoint_dir = tempfile.mkdtemp()
        # Run training pass to retrieve the correct min and max in FakeQuant op (to avoid using default values) and
        # save dummy checkpoint.
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for iter in range(1):
                image = np.random.rand(spatial_size, spatial_size, input_channels).astype(np.float32) * 255
                label = np.random.rand(spatial_size, spatial_size, output_channels).astype(np.float32) * 255
                training_loss, _ = sess.run([loss, optimizer], feed_dict={input_tensor: image[None, ...],
                                                                          output_tensor: label[None, ...]})

            saver.save(sess=sess, save_path=os.path.join(checkpoint_dir, 'quantization'))

        with tf.Graph().as_default() as g:
            input_tensor = tf.placeholder(tf.float32, [1, spatial_size, spatial_size, input_channels], name='input')
            with tf.variable_scope('quantize'):
                output = model(x=input_tensor)

            # define eval graph, by quantizing the weights of the model with learned min/max values for each layer
            tf.contrib.quantize.experimental_create_eval_graph(input_graph=g, weight_bits=num_bits,
                                                               activation_bits=num_bits)
            tmpdir = tempfile.mkdtemp()
            tf_graph_path = os.path.join(str(tmpdir), "tf_graph.pb")
            tf_graph_path_quantized = os.path.join(str(tmpdir), "frozen_graph_quantized.pb")
            with open(tf_graph_path, 'wb') as f:
                f.write(g.as_graph_def().SerializeToString())
            freeze_g(input_graph=tf_graph_path,
                     input_saver="",
                     input_binary=True,
                     input_checkpoint=os.path.join(checkpoint_dir, 'quantization'),
                     output_node_names="quantize/quantized_model/conv2d/Conv2D",
                     restore_op_name="save/restore_all",
                     filename_tensor_name="save/Const:0",
                     output_graph=tf_graph_path_quantized,
                     clear_devices=True,
                     initializer_nodes="")
            shutil.rmtree(checkpoint_dir)

        graph = load_tf_pb(tf_graph_path_quantized)

        tf.reset_default_graph()
        graphdef = tf.GraphDef()
        input_dict = {}
        with open(tf_graph_path_quantized, "rb") as f:
            graphdef.ParseFromString(f.read())
        shutil.rmtree(tmpdir)

        with tf.Graph().as_default(), tf.Session(config=None) as sess:
            tf.graph_util.import_graph_def(graphdef, name='')
            input_dict[sess.graph.get_tensor_by_name('input:0')] = (np.random.rand(1, spatial_size, spatial_size,
                                                                                   input_channels).astype(np.float32))
            outputs = []
            outputs.append(sess.graph.get_tensor_by_name('quantize/quantized_model/conv2d/Conv2D:0'))
            tf_outs = sess.run(outputs, feed_dict=input_dict)

        TensorFlowBaseTest.run_compare_tf(
            graph,
            input_dict,
            ["quantize/quantized_model/conv2d/Conv2D"],
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
            tf_outputs=tf_outs,
            rtol=0.005,
        )


class TestFill(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank, value",
        itertools.product(
            [True, False], backends, [rank for rank in range(1, 6)], [-19.0, 0.0, 37.0]
        ),
    )
    def test_fill(self, use_cpu_only, backend, rank, value):
        def test_tf_static():
            shape = np.random.randint(low=1, high=3, size=rank)

            @make_tf_graph([shape])
            def build_model(x):
                return tf.add(
                    x, tf.fill(dims=np.array(shape, dtype=np.float32), value=value)
                )

            model, inputs, outputs = build_model
            input_values = [np.random.rand(*shape).astype(np.float32)]
            input_dict = dict(zip(inputs, input_values))

            TensorFlowBaseTest.run_compare_tf(model,
                           input_dict,
                           outputs,
                           use_cpu_only=use_cpu_only,
                           frontend_only=False,
                           backend=backend)


        def test_tf_dynamic():
            shape = np.random.randint(low=1, high=3, size=rank)
            @make_tf_graph([(len(shape), tf.int32)])
            def build_model(x):
                return tf.fill(dims=x, value=value)

            model, inputs, outputs = build_model
            input_values = [np.array(shape, dtype=np.int32)]
            input_dict = dict(zip(inputs, input_values))

            TensorFlowBaseTest.run_compare_tf(model,
                           input_dict,
                           outputs,
                           use_cpu_only=use_cpu_only,
                           frontend_only=False,
                           backend=backend)

        test_tf_static()
        test_tf_dynamic()

class TestNonMaximumSuppression(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        ",".join(
            [
                "use_cpu_only",
                "backend",
                "num_boxes",
                "max_boxes",
                "iou_threshold",
                "score_threshold",
            ]
        ),
        itertools.product(
            [True, False],
            backends,
            [20, 30, 80],
            [5, 20, 100],
            [1.0, 0.99],
            [float("-inf"), -200.0],
        ),
    )
    def test_non_max_suppression(
        self,
        use_cpu_only,
        backend,
        num_boxes,
        max_boxes,
        iou_threshold,
        score_threshold,
    ):
        if backend == ("mlprogram", "fp16") and not use_cpu_only:
            pytest.xfail("rdar://80661262 ([GPU failures ] NonMaximumSuppression FP16 coremltools unit tests)")

        boxes_val = random_gen(shape=(num_boxes, 4), rand_min=0, rand_max=32)
        scores_val = random_gen(shape=(num_boxes,), rand_min=-100, rand_max=100)

        @make_tf_graph([boxes_val.shape, scores_val.shape])
        def build_model(boxes, scores):
            ret = tf.image.non_max_suppression(
                boxes=boxes,
                scores=scores,
                max_output_size=max_boxes,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
            )
            return ret

        model, inputs, outputs = build_model
        input_dict = dict(zip(inputs, [boxes_val, scores_val]))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )


class TestOneHot(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank_and_axis, dynamic",
        itertools.product(
            [True, False],
            backends,
            [
                (2, 0),
                (2, -1),
                (3, 3),
                (3, 0),
                (3, -2),
                (4, -4),
                (4, 1),
                (4, -1),
                (4, -2),
                (4, 3),
            ],
            [True, False],
        ),
    )
    def test_one_hot(self, use_cpu_only, backend, rank_and_axis, dynamic):
        rank, axis = rank_and_axis
        depth, on_value, off_value = 30, 28.0, -4.0
        x_shape = np.random.randint(low=2, high=4, size=rank)
        axis = (axis if axis >= -1 else axis + rank + 1)

        if not dynamic:
            @make_tf_graph([list(x_shape)+[tf.int32]])
            def build_model(x):
                return tf.one_hot(x, axis=axis, depth=depth, on_value=on_value, off_value=off_value)

            model, inputs, outputs = build_model
            input_values = [np.random.randint(0, depth, size=x_shape).astype(np.int32)]
            input_dict = dict(zip(inputs, input_values))

        else:  # Dynamic Case with depth being an input
            @make_tf_graph([list(x_shape)+[tf.int32], [1, tf.int32]])
            def build_model(x, depth_input):
                # tf.squeeze since CoreML input has to be rank 1~5.
                return tf.one_hot(x, axis=axis, depth=tf.squeeze(depth_input),
                        on_value=on_value, off_value=off_value)

            model, inputs, outputs = build_model
            input_values = [np.random.randint(0, depth, size=x_shape).astype(np.int32),
                    np.array([depth]).astype(np.int32)]
            input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(model, input_dict, outputs,
               use_cpu_only=use_cpu_only,
               frontend_only=False, backend=backend)


class TestPad(TensorFlowBaseTest):
    @pytest.mark.parametrize("use_cpu_only, backend, rank, mode, dynamic, trial",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [2, 3, 4],
                                 ['constant', 'reflect'],
                                 [True, False],
                                 list(range(10)),
                             )
                             )
    def test(self, use_cpu_only, backend, rank, mode, dynamic, trial):
        input_shape = np.random.randint(low=2, high=10, size=rank)
        min_input_dim_size = input_shape.min()
        padding_val = np.random.randint(low=0, high=min_input_dim_size, size=(rank, 2), dtype=np.int32)

        # Only constant mode supports padding across all dimensions
        # All other padding modes are only applied on two dimensions.
        perm = list(range(rank))
        import random
        random.shuffle(perm)
        if mode != "constant":
            padding_val[perm[:-2]] = 0
        tf_mode = mode.upper()

        if dynamic:
            if mode != "constant":
                return
            padding_shape = padding_val.shape
            @make_tf_graph([input_shape, list(padding_shape)+[tf.int32]])
            def build_model(x, paddings):
                return tf.pad(x, paddings=paddings, mode=tf_mode)

            model, inputs, outputs = build_model
            input_values = [random_gen(input_shape, rand_min=0.2, rand_max=1000), padding_val]
            input_dict = dict(zip(inputs, input_values))

        else:
            @make_tf_graph([input_shape])
            def build_model(x):
                return tf.pad(x, paddings=padding_val, mode=tf_mode)

            model, inputs, outputs = build_model
            input_values = [random_gen(input_shape, rand_min=0.2, rand_max=1000)]
            input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(model, input_dict, outputs,
                       use_cpu_only=use_cpu_only,
                       frontend_only=False, backend=backend)


class TestPadV2(TensorFlowBaseTest):
    @pytest.mark.parametrize("use_cpu_only, backend, rank, constant_values, dynamic, trial",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 list(range(1, 6)),
                                 [0., 10, -1],
                                 [True],
                                 list(range(10))
                             )
                             )
    def test(self, use_cpu_only, backend, rank, constant_values, dynamic, trial):
        input_shape = np.random.randint(low=2, high=10, size=rank)
        paddings = np.random.randint(low=2, high=5, size=2*rank).astype(np.int32)
        padding_val = paddings.reshape(-1,2)
        if dynamic:
            padding_shape = padding_val.shape
            @make_tf_graph([input_shape, list(padding_shape)+[tf.int32]])
            def build_model(x, paddings):
                return tf.raw_ops.PadV2(input=x, paddings=paddings, constant_values=constant_values)

            model, inputs, outputs = build_model

            input_values = [random_gen(input_shape, rand_min=0.2, rand_max=1000), padding_val]
            input_dict = dict(zip(inputs, input_values))

        else:
            @make_tf_graph([input_shape])
            def build_model(x):
                return tf.raw_ops.PadV2(input=x, paddings=padding_val, constant_values=constant_values)

            model, inputs, outputs = build_model

            input_values = [random_gen(input_shape, rand_min=0.2, rand_max=1000)]
            input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(model, input_dict, outputs,
                       use_cpu_only=use_cpu_only,
                       frontend_only=False, backend=backend)


class TestRange(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, params",
        itertools.product(
            [True, False],
            backends,
            [
                (-10.4, 23, 12.2),
                (0, 10, 1),
                (50.5, 90.5, 1.5),
                (5, 8, 2),
                (5, 8, 98),
                (5, 8, 1.5),
                (10, 5, -0.6),
                (24, -65, -2),
            ],
        ),
    )
    def test_range(self, use_cpu_only, backend, params):
        start, end, step = np.array(params).astype(np.float32)

        # CoreML requires rank-1~5 input.
        @make_tf_graph([[1, tf.float32]])
        def build_model(limit):
            return tf.range(start=start, limit=tf.squeeze(limit), delta=step)

        model, inputs, outputs = build_model
        input_values = [np.array([end])]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )

        # CoreML requires rank-1~5 input.
        @make_tf_graph([[1, tf.float32]])
        def build_model(delta):
            return tf.range(start=start, limit=end, delta=tf.squeeze(delta))

        model, inputs, outputs = build_model
        input_values = [np.array([step])]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )

        # CoreML requires rank-1~5 input.
        @make_tf_graph([[1, tf.float32]])
        def build_model(begin):
            return tf.range(start=tf.squeeze(begin), limit=end, delta=step)

        model, inputs, outputs = build_model
        input_values = [np.array([start])]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )


class TestTile(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank_and_reps",
        itertools.product(
            [True, False],
            backends,
            [
                (1, (2,)),
                (2, (1, 2)),
                (2, (2, 2)),
                (3, (3, 2, 1)),
                (3, (2, 1, 3)),
                (3, (2, 1, 1)),
                (4, (1, 3, 2, 1)),
                (4, (2, 1, 1, 2)),
                (5, (2, 1, 1, 3, 2)),
                (5, (1, 1, 2, 3, 2)),
            ],
        ),
    )
    def test_tile(self, use_cpu_only, backend, rank_and_reps):
        rank, reps = rank_and_reps
        x_shape = np.random.randint(low=2, high=4, size=rank)

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.tile(x, multiples=reps)

        model, inputs, outputs = build_model
        input_values = [random_gen(x_shape)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )

class TestDynamicTile(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [1, 2, 3, 4, 5]),
    )
    def test_tile(self, use_cpu_only, backend, rank):
        x_shape = np.random.randint(low=2, high=4, size=rank)
        reps_val = np.random.randint(low=1, high=3, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            reps = tf.placeholder(tf.int32, shape=reps_val.shape)
            res = tf.tile(x, multiples=reps)
            TensorFlowBaseTest.run_compare_tf(
                graph,
                {x: np.random.rand(*x_shape), reps: reps_val},
                res,
                use_cpu_only=use_cpu_only,
                frontend_only=False,
                backend=backend,
            )


class TestTopK(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank, k",
        itertools.product(
            [True, False], backends, [rank for rank in range(1, 6)], [1, 2, 3],
        ),
    )
    def test_top_k(self, use_cpu_only, backend, rank, k):
        # TensorFlow only supports last dimension (axis = -1).
        shape = np.random.randint(low=3, high=4, size=rank)

        if backend == ("mlprogram", "fp16") and not use_cpu_only:
            pytest.xfail("operation is ill-conditioned on FP16")

        @make_tf_graph([shape])
        def build_model(x):
            ref = tf.math.top_k(x, k=k, sorted=True)
            return (ref[1], ref[0])

        model, inputs, outputs = build_model
        input_values = [random_gen(shape, rand_min=-100, rand_max=100)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )


class TestConcat(TensorFlowBaseTest):
    @pytest.mark.parametrize("use_cpu_only, backend, op_version, rank, num_inputs",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 ['v1', 'v2'],
                                 list(range(6)),
                                 list(range(1, 4)),
                             ))
    def test_concat(self, use_cpu_only, backend, op_version, rank, num_inputs):

        if backend == ("mlprogram", "fp16"):
            pytest.xfail("rdar://80658663 (Concat FP16 tests are failing in self.fp16_overflow method)")

        import random
        for axis in range(-rank, rank):
            input_shape = np.random.randint(low=1, high=4, size=rank)
            input_shapes = [input_shape.copy() for _ in range(num_inputs)]
            concat_axis_value = np.random.randint(low=1, high=3, size=num_inputs)
            for i, v in enumerate(concat_axis_value):
                input_shapes[i][axis] = concat_axis_value[i]

            @make_tf_graph(input_shapes)
            def build_model(*inputs):
                # add 3 additional tensor contains dimension size of 0
                zero_shape = input_shape.copy()
                zero_shape[axis] = 0
                const = [tf.constant([], shape=zero_shape) for _ in range(3)]
                values = inputs + tuple(const)
                values = list(values)
                random.shuffle(values)
                values = tuple(values)
                if op_version == 'v1':
                    # Seems like now the tf functions are using concatV2, so create as raw_ops here
                    res = tf.raw_ops.Concat(concat_dim=axis, values=values)
                elif op_version == 'v2':
                    res = tf.raw_ops.ConcatV2(values=values, axis=axis)
                return res

            model, inputs, outputs = build_model
            input_values = [random_gen(shape) for shape in input_shapes]
            input_dict = dict(zip(inputs, input_values))
            TensorFlowBaseTest.run_compare_tf(model, input_dict, outputs,
                           use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestSplit(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend, rank, dynamic",
        itertools.product([True, False], backends, [1, 2, 3, 4], [True, False]),
    )
    def test_split(self, use_cpu_for_conversion, backend, rank, dynamic):
        if backend[0] == "mlprogram" and not use_cpu_for_conversion:
            pytest.xfail("rdar://80397986")

        input_shape1 = np.random.randint(low=1, high=3, size=rank)
        for axis in range(-rank, rank, 2):
            for split_num in range(2, input_shape1[axis] + 1, 2):
                if input_shape1[axis] % split_num != 0:
                    continue
                tf_input_shape = list(input_shape1)
                if dynamic:
                    axis1 = np.random.randint(low=0, high=rank)
                    tf_input_shape[axis1] = None

                @make_tf_graph([tf_input_shape])
                def build_model(x):
                    res = tf.split(x, split_num, axis=axis)
                    # Comment: If tf.split output is returned, there's no
                    # get_tuple nodes. Some graph pass is needed. Example:
                    #
                    #    x = tf.placeholder(tf.float32, shape=input_shape1)
                    #    res = tf.split(x, 3, axis=0)
                    #
                    # res are ['split:0', 'split:1', 'split']
                    #
                    # but node.outputs == ['gto_1', 'gto_2', 'gto_3']
                    import random

                    random.shuffle(res)
                    return tuple(res)

                model, inputs, outputs = build_model
                input_values = [random_gen(input_shape1)]
                input_dict = dict(zip(inputs, input_values))
                TensorFlowBaseTest.run_compare_tf(
                    model,
                    input_dict,
                    outputs,
                    use_cpu_only=use_cpu_for_conversion,
                    backend=backend,
                    use_cpu_for_conversion=use_cpu_for_conversion,
                )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, sizes",
        itertools.product([True, False], backends,
                          [[1, 1, 2], [0, 2, 2], [1, 0, 3], [2, 0, 1, 1, 0]]),
    )
    def test_split_with_sizes(self, use_cpu_only, backend, sizes):
        input_shape = (4, 2)

        @make_tf_graph([input_shape])
        def build_model(x):
            res = tf.split(x, sizes, axis=0)
            # split sizes can contain 0s, and we skip those in outputs
            return tuple([res[i] for i in range(len(sizes)) if sizes[i] != 0])

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends)
    )
    def test_splitv(self, use_cpu_only, backend):
        input_shape = [3, 2, 1]

        @make_tf_graph([input_shape])
        def build_model(x):
            res = tf.split(x, [1, 2], axis=0)
            return res[0], res[1]

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )


class TestStack(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends, )
    )
    def test_stack(self, use_cpu_only, backend):
        input_shape1 = [3, 1, 1]
        input_shape2 = [3, 1, 1]

        @make_tf_graph([input_shape1, input_shape2])
        def build_model(x, y):
            return [tf.stack((x, y), axis=0), tf.stack((y, x), axis=-1)]

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape1), random_gen(input_shape2)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )

class TestUnstack(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, shape", itertools.product([True, False], backends, [[3, 1], [4, 3]],)
    )
    def test_unstack(self, use_cpu_only, backend, shape):
        @make_tf_graph([shape])
        def build_model(x):
            return tf.unstack(x, axis=1)

        model, inputs, outputs = build_model
        input_values = [random_gen(shape)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, shape", itertools.product([True, False], backends, [[3, 1], [4, 3]])
    )

    def test_unstack_and_stack(self, use_cpu_only, backend, shape):
        @make_tf_graph([shape])
        def build_model(x):
            x = tf.unstack(x, axis=1)
            return tf.stack(x)

        model, inputs, outputs = build_model
        input_values = [random_gen(shape)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )


class TestPack(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank, num_inputs",
        itertools.product([True, False], backends, list(range(5)), list(range(1, 5))),
    )
    def test_pack(self, use_cpu_only, backend, rank, num_inputs):
        if rank == 0:
            pytest.xfail('Rank 0 not supported by CoreML runtime')

        shape = np.random.randint(low=1, high=4, size=rank)
        input_shapes = [shape[:] for _ in range(num_inputs)]

        @make_tf_graph(input_shapes)
        def build_model(*inputs):
            return tf.raw_ops.Pack(values=inputs, axis=0)

        model, inputs, outputs = build_model
        input_values = [
            random_gen(shape, rand_min=-1, rand_max=1) for shape in input_shapes
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestArgSort(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank, axis, direction",
        itertools.product(
            [True, False],
            backends,
            [rank for rank in range(1, 6)],
            [-1, 0],
            ["ascending", "descending"],
        ),
    )
    def test_argsort(self, use_cpu_only, backend, rank, axis, direction):
        shape = np.random.randint(low=1, high=4, size=rank)
        if use_cpu_only:
            dtype = np.float32
            tf_dtype = tf.float32
        else:
            dtype = np.float16
            tf_dtype = tf.float16

        @make_tf_graph([list(shape) + [tf_dtype]])
        def build_model(x):
            return tf.argsort(x, axis=axis, direction=direction.upper())

        model, inputs, outputs = build_model
        input_values = [random_gen(shape, rand_min=-100, rand_max=100, allow_duplicate=False, dtype=dtype)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend
        )


class TestDepthToSpace(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, input_shape, block_size",
        itertools.product(
            [True, False],
            backends,
            [(1, 1, 1, 16), (1, 1, 1, 32), (1, 3, 3, 16)],
            [2, 4],
        ),
    )
    def test_depth_to_space(self, use_cpu_only, backend, input_shape, block_size):

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.nn.depth_to_space(x, block_size)

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape)]
        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )


class TestExpandDims(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank_and_axis",
        itertools.product(
            [True, False],
            backends,
            [
                (rank, axis)
                for rank in range(1, 5)
                for axis in range(-rank - 1, rank + 1)
            ],
        ),
    )
    def test_expand_dims(self, use_cpu_only, backend, rank_and_axis):
        rank, axis = rank_and_axis
        input_shape = np.random.randint(low=2, high=4, size=rank)

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.expand_dims(x, axis=axis)

        model, inputs, outputs = build_model

        input_values = [np.random.rand(*input_shape).astype(np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

class TestReshape(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends,)
    )
    def test_flatten(self, use_cpu_only, backend):
        shapes = [[2, 2], [3, 2, 1, 2], [2, 1, 4, 3]]

        for input_shape in shapes:

            @make_tf_graph([input_shape])
            def build_model(x):
                return tf.keras.backend.flatten(x)

            model, inputs, outputs = build_model

            input_values = [np.random.rand(*input_shape).astype(np.float32)]
            input_dict = dict(zip(inputs, input_values))
            TensorFlowBaseTest.run_compare_tf(
                model,
                input_dict,
                outputs,
                use_cpu_only=use_cpu_only,
                frontend_only=False,
                backend=backend,
            )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, input_shape",
        itertools.product(
            [False],
            backends,
            [
                ([10, 10], [5, 20]),
                ([3, 4, 5, 6], [4, 5, 3, 6]),
                ([4, 4, 5, 6], [2, 2, -1]),
            ],
        ),
    )
    def test_reshape_static(self, use_cpu_only, backend, input_shape):
        @make_tf_graph([input_shape[0]])
        def build_model(x):
            return tf.reshape(x, shape=input_shape[1])

        model, inputs, outputs = build_model

        input_values = [np.random.rand(*input_shape[0]).astype(np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, input_shape",
        itertools.product(
            [False],
            backends,
            [
                ([10, 10], [5, 20]),
                ([3, 4, 5, 6], [4, 5, 3, 6]),
                ([4, 4, 5, 6], [2, 2, -1]),
                ([2, 3, 5, 3], [2, -1]),
            ],
        ),
    )
    def test_reshape_dynamic(self, use_cpu_only, backend, input_shape):
        @make_tf_graph([input_shape[0], (len(input_shape[1]), tf.int32)])
        def build_model(x, y):
            return tf.reshape(x, shape=y)

        model, inputs, outputs = build_model

        input_values = [
            np.random.rand(*input_shape[0]).astype(np.float32),
            np.array(input_shape[1], dtype=np.int32),
        ]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, shape",
        itertools.product([False], backends, [[1], [1, 1], [1, 1, -1], []],),
    )
    def test_reshape_scalar(self, use_cpu_only, backend, shape):
        pytest.xfail('Rank 0 not supported by CoreML runtime')

        input_shape = ()

        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.raw_ops.Reshape(tensor=x, shape=shape)

        model, inputs, outputs = build_model

        input_values = [np.random.rand(*input_shape)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

class TestShape(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [rank for rank in range(1, 6)],),
    )
    def test_shape(self, use_cpu_only, backend, rank):
        shape = np.random.randint(low=3, high=4, size=rank)
        shape_holder = [None] * rank

        @make_tf_graph([shape_holder])
        def build_model(x):
            return tf.shape(x)

        model, inputs, outputs = build_model

        input_values = [random_gen(shape, rand_min=-100, rand_max=100)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

class TestMatrixDiag(TensorFlowBaseTest):
    @pytest.mark.parametrize("use_cpu_only, backend, length, dynamic",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [length for length in range(1, 5)],
                                 [True, False]))
    def test(self, use_cpu_only, backend, length, dynamic):

        if dynamic:
            input_shape = np.random.randint(low=1, high=4, size=length)
            a, b = np.prod(input_shape[:2]), np.prod(input_shape[2:])
            size = np.array([a,b]).astype(np.int32)
            reshape_shape = [2]
            @make_tf_graph([input_shape, reshape_shape+[tf.int32]])
            def build_model(x, reshape):
                x = tf.reshape(x, reshape)
                x = tf.reshape(x, [-1])
                return tf.raw_ops.MatrixDiag(diagonal=x)
            model, inputs, outputs = build_model
            input_values = [random_gen(input_shape, -1, 1), size]
        else:
            input_shape = [length]
            @make_tf_graph([input_shape])
            def build_model(x):
                return tf.raw_ops.MatrixDiag(diagonal=x)
            model, inputs, outputs = build_model
            input_values = [random_gen(input_shape, -1, 1)]

        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(model, input_dict, outputs,
                       use_cpu_only=use_cpu_only,
                       frontend_only=False, backend=backend)


class TestReverse(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank_and_axes",
        itertools.product(
            [True, False],
            backends,
            [
                (1, (-1,)),
                (2, (0,)),
                (2, (-1, 0)),
                (3, (1, -3)),
                (3, (-2,)),
                (3, (0, 1, 2)),
                (4, (-2, -1, 0)),
                (4, (-1, -2)),
                (4, []),
                (5, (-3, -1, 3)),
                (5, (0, -1, 1, -2)),
            ],
        ),
    )
    def test_reverse(self, use_cpu_only, backend, rank_and_axes):
        rank, axes = rank_and_axes
        shape = np.random.randint(low=1, high=4, size=rank)

        if backend == ("mlprogram", "fp16") and not use_cpu_only:
            pytest.xfail("rdar://80710884 (Crash in Reverse op unit test on MIL GPU context)")

        @make_tf_graph([shape])
        def build_model(x):
            return tf.reverse(x, axis=axes)

        model, inputs, outputs = build_model
        input_values = [random_gen(shape)]
        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )


class TestReverseSequence(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [rank for rank in range(2, 6)]),
    )
    def test_reverse_sequence(self, use_cpu_only, backend, rank):
        shape = np.random.randint(low=1, high=4, size=rank)
        seq_axis = np.random.randint(low=1, high=rank)
        batch_axis = np.random.randint(low=0, high=seq_axis)
        lengths = np.random.randint(low=0, high=shape[seq_axis], size=shape[batch_axis])

        @make_tf_graph([shape])
        def build_model(x):
            return tf.reverse_sequence(
                x, seq_lengths=lengths, seq_axis=seq_axis, batch_axis=batch_axis
            )

        model, inputs, outputs = build_model
        input_values = [random_gen(shape)]
        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )


class TestSpaceToDepth(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, input_shape, block_size",
        itertools.product(
            [True, False],
            backends,
            [(1, 6, 6, 1), (1, 12, 12, 1), (1, 6, 6, 3)],
            [2, 3],
        ),
    )
    def test_space_to_depth(self, use_cpu_only, backend, input_shape, block_size):
        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.nn.space_to_depth(x, block_size)

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape)]
        input_dict = dict(zip(inputs, input_values))

        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )


class TestSqueeze(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank_and_axes",
        itertools.product(
            [True, False],
            backends,
            [
                (2, (1,)),
                (2, (0,)),
                (3, (1,)),
                (3, (0, -1)),
                (3, []),
                (4, (-1, 2, 1)),
                (4, (0, 1)),
                (5, (3, 1, 2)),
                (5, (-1,)),
            ],
        ),
    )
    def test_squeeze(self, use_cpu_only, backend, rank_and_axes):
        rank, axes = rank_and_axes
        x_shape = np.random.randint(low=2, high=4, size=rank)
        for axis in axes:
            x_shape[axis] = 1

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.squeeze(x, axis=axes)

        model, inputs, outputs = build_model

        input_values = [np.random.rand(*x_shape).astype(np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestTranspose(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank_and_perm",
        itertools.product(
            [True, False],
            backends,
            [
                (1, (0,)),
                (2, (1, 0)),
                (2, (0, 1)),
                (3, (0, 2, 1)),
                (3, (2, 1, 0)),
                (3, (2, 0, 1)),
                (4, (0, 3, 2, 1)),
                (4, (3, 0, 1, 2)),
                (5, (2, 3, 1, 0, 4)),
                (5, (3, 1, 0, 4, 2)),
            ],
        ),
    )
    def test_transpose_1(self, use_cpu_only, backend, rank_and_perm):

        rank, perm = rank_and_perm
        x_shape = np.random.randint(low=1, high=4, size=rank)

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.transpose(x, perm=perm)

        model, inputs, outputs = build_model
        input_values = [random_gen(x_shape)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product([True, False], backends, [1, 2, 3, 4], ),
    )
    def test_transpose_2(self, use_cpu_only, backend, rank):

        input_shape = np.random.randint(low=1, high=4, size=rank)
        perm = np.random.permutation(rank)

        def static_perm():
            @make_tf_graph([input_shape])
            def build_model(x):
                return tf.transpose(x, perm=perm)

            model, inputs, outputs = build_model
            input_values = [random_gen(input_shape)]
            input_dict = dict(zip(inputs, input_values))
            TensorFlowBaseTest.run_compare_tf(
                model,
                input_dict,
                outputs,
                use_cpu_only=use_cpu_only,
                backend=backend,
            )

        def dynamic_perm():
            @make_tf_graph([input_shape, list(perm.shape) + [tf.int32]])
            def build_model(x, tf_perm):
                return tf.transpose(x, perm=tf_perm)

            model, inputs, outputs = build_model
            input_values = [random_gen(input_shape), perm.astype(np.int32)]
            input_dict = dict(zip(inputs, input_values))
            TensorFlowBaseTest.run_compare_tf(
                model,
                input_dict,
                outputs,
                use_cpu_only=use_cpu_only,
                backend=backend,
            )

        static_perm()
        # Note that TF supports dynamic perm in tf.transpose.
        with pytest.raises(ValueError, match=r".*must be const at compile time.*"):
            dynamic_perm()

    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank",
        itertools.product(
            [True, False],
            backends,
            [1,3],
        ),
    )
    def test_redundant_transpose(self, use_cpu_only, backend, rank):
        import random
        random.seed(10)
        input_shape = np.random.randint(low=1, high=4, size=rank)
        num_layers = 30
        perms = []
        for _ in range(num_layers):
            perm = list(range(rank))
            random.shuffle(perm)
            perms.append(perm)

        @make_tf_graph([input_shape])
        def build_model(x):
            net = x
            for perm in perms:
                net = tf.transpose(net, perm=perm)
            return net

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestSpaceToBatchND(TensorFlowBaseTest):
    # No direct mil smoke test since it's a TF op which is a composite of several ops.
    @pytest.mark.parametrize(
        "use_cpu_only, backend, input_shape, block_shape, paddings",
        itertools.product(
            [True, False],
            backends,
            [(1, 4, 4, 1), (1, 4, 4, 3), (2, 4, 6, 1)],
            [[2, 2]],
            [[[0, 0], [0, 0]], [[1, 1], [0, 2]], [[4, 2], [4, 2]]],
        ),
    )
    def test_smoke(self, use_cpu_only, backend, input_shape, block_shape, paddings):
        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.raw_ops.SpaceToBatchND(
                input=x, block_shape=block_shape, paddings=paddings
            )

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, input_block_rank, dynamic",
        itertools.product(
            [True, False],
            backends,
            [(3, 1), (3, 2), (4, 1)],
            [True, False],
        ),
    )
    def test_programmatic(
        self, use_cpu_only, backend, input_block_rank, dynamic
    ):

        input_rank, block_rank = input_block_rank

        # generate data
        input_shape = np.random.randint(low=1, high=4, size=input_rank)
        block_shape = np.random.randint(low=1, high=3, size=block_rank)
        paddings = []
        for i in range(block_rank):
            while True:
                temp = np.random.randint(low=0, high=10, size=2)
                if (np.sum(temp) + input_shape[i + 1]) % block_shape[i] == 0:
                    paddings.append(temp)
                    break
        paddings = np.array(paddings)

        if not dynamic:

            @make_tf_graph([input_shape])
            def build_model(x):
                return tf.raw_ops.SpaceToBatchND(
                    input=x, block_shape=block_shape, paddings=paddings
                )

        else:

            @make_tf_graph([[None] * input_rank])
            def build_model(x):
                return tf.raw_ops.SpaceToBatchND(
                    input=x, block_shape=block_shape, paddings=paddings
                )

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestBatchToSpaceND(TensorFlowBaseTest):
    # No direct mil smoke test since it's a TF op which is a composite of several ops.
    @pytest.mark.parametrize(
        "use_cpu_only, backend, input_shape, block_size, crops",
        itertools.product(
            [True, False],
            backends,
            [(4, 4, 4, 1), (4, 4, 4, 3), (4, 4, 6, 1)],
            [[2, 2]],
            [[[0, 0], [0, 0]], [[1, 1], [0, 2]], [[4, 2], [4, 2]]],
        ),
    )
    def test_smoke(self, use_cpu_only, backend, input_shape, block_size, crops):
        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.raw_ops.BatchToSpaceND(
                input=x, block_shape=block_size, crops=crops
            )

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend, input_block_rank, dynamic",
        itertools.product(
            [True, False],
            backends,
            [(3, 1), (3, 2), (4, 1)],
            [True, False]
        ),
    )
    def test_programmatic(
        self, use_cpu_only, backend, input_block_rank, dynamic):

        input_rank, block_rank = input_block_rank

        # generate data
        input_shape = np.random.randint(low=1, high=4, size=input_rank)
        block_shape = np.random.randint(low=1, high=3, size=block_rank)
        input_shape[0] = input_shape[0] * np.prod(block_shape)
        crops = []
        for i in range(block_rank):
            while True:
                temp = np.random.randint(low=0, high=4, size=2)
                if np.sum(temp) < input_shape[i + 1] * block_shape[i]:
                    crops.append(temp)
                    break
        crops = np.array(crops)

        if not dynamic:

            @make_tf_graph([input_shape])
            def build_model(x):
                return tf.raw_ops.BatchToSpaceND(
                    input=x, block_shape=block_shape, crops=crops
                )

        else:

            @make_tf_graph([[None] * input_rank])
            def build_model(x):
                return tf.raw_ops.BatchToSpaceND(
                    input=x, block_shape=block_shape, crops=crops
                )

        model, inputs, outputs = build_model
        input_values = [random_gen(input_shape)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestTensorArray(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends,)
    )
    def test_tf_basic(self, use_cpu_only, backend):
        # TF1: TensorArrayV3, TensorArrayWriteV3, TensorArrayScatterV3,
        #      TensorArraySizeV3, TensorArrayGatherV3
        # TF2: TensorListReserve, TensorListLength, TensorListSetItem,
        #      TensorListScatterIntoExistingList, TensorListStack,
        #      TensorListResize

        elem_shape = (3, 2)

        @make_tf_graph([elem_shape])
        def build_model(x):
            ta = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

            ta = ta.write(2, x)

            # TensorArray has write-once semantics, and thus we write to a new
            # index
            # (https://www.tensorflow.org/api_docs/python/tf/TensorArray)
            # writing to out of bound index
            ta = ta.scatter([3], tf.expand_dims(x, 0))

            # writing to in-bound index
            ta = ta.scatter([0], tf.expand_dims(x, 0))

            return ta.stack()

        model, inputs, outputs = build_model
        input_values = [random_gen(elem_shape)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends)
    )
    def test_tf_dynamic_elem_shape(self, use_cpu_only, backend):

        # TF1: TensorArrayV3, TensorArrayWriteV3, TensorArrayScatterV3,
        #      TensorArraySizeV3, TensorArrayGatherV3
        # TF2: TensorListReserve, TensorListLength, TensorListSetItem,
        #      TensorListScatterIntoExistingList, TensorListStack,
        #      TensorListResize
        elem_shape = (None, None)

        @make_tf_graph([elem_shape])
        def build_model(x):
            ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            ta = ta.write(10, x)
            ta = ta.write(9, x)
            ta = ta.scatter([3], tf.expand_dims(x, 0))
            ta = ta.scatter([8], tf.expand_dims(x, 0))

            return ta.stack()

        model, inputs, outputs = build_model
        input_values = [random_gen((2,3))]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict, outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False, backend=backend)

    @pytest.mark.skip(
        reason="[NNv2 TensorArray scatter returns wrong result](rdar://63345281)"
    )
    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True, False], backends,)
    )
    def test_tf_while_loop(self, use_cpu_only, backend):
        @make_tf_graph([(3, 2)])
        def build_model(x):
            def body(i, num_iters, array, update):
                return i + 1, num_iters, array.write(i, update), update

            def cond(i, num_iters, array, update):
                return i < num_iters

            i = 0
            max_iters = 3
            ta = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
            _, _, new_ta, _ = tf.while_loop(cond, body, [i, max_iters, ta, x])
            new_ta = new_ta.scatter([max_iters], tf.expand_dims(x, 0))

            return new_ta.stack()

        model, inputs, outputs = build_model
        input_values = [random_gen(shape=(3, 2))]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )


class TestBroadcastTo(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, shapes, is_dynamic",
        itertools.product(
            [True, False],
            backends,
            [
                ((2,), (2,)),
                ((1,), (10,)),
                ((3,), (3, 3)),
                ((1, 1), (1, 4)),
                ((1, 1, 5), (3, 4, 4, 4, 5)),
                ((3,), (1, 3, 2, 1, 3)),
                ((3, 5), (2, 3, 5)),
                ((1, 2), (2, 3, 1, 2)),
                ((1, 3, 1, 4), (8, 3, 32, 4)),
                ((2, 16), (3, 1, 4, 2, 16)),
            ],
            [False],
        ),
    )
    def test(self, use_cpu_only, backend, shapes, is_dynamic):
        input_shape, output_shape = shapes

        if is_dynamic is False:

            @make_tf_graph([input_shape])
            def build_model(x):
                return tf.broadcast_to(x, output_shape)

        else:  # output / target shape is an input (placeholder)

            @make_tf_graph([input_shape, (len(output_shape), tf.int32)])
            def build_model(x, shape):
                return tf.broadcast_to(x, shape)

        model, inputs, outputs = build_model
        if is_dynamic is False:
            input_values = [random_gen(input_shape)]
        else:
            input_values = [
                random_gen(input_shape),
                np.array(output_shape, dtype=np.int32),
            ]

        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model, input_dict, outputs, use_cpu_only=use_cpu_only, backend=backend
        )


class TestLSTMBlockCell(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, batch, return_hc_only, has_peephole, has_clip",
        itertools.product(
            [True, False],
            backends,
            [1, 2],
            [True, False],
            [True, False],
            [True, False],
        ),
    )
    def test_tf_no_variable(
        self, use_cpu_only, batch, backend, return_hc_only, has_peephole, has_clip
    ):
        """
        If return_hc_only == True, the op can be mapped to mb.lstm.
        Otherwise it has to be expanded.
        """
        # _lstm_block_cell allows fine-grained control of W, peephole etc
        from tensorflow.contrib.rnn.python.ops.lstm_ops import _lstm_block_cell

        actual_len, padded_len = 3, 4
        input_dim, hidden_dim = 2, 3
        x_shape = (batch, input_dim)
        init_h = np.random.rand(batch, hidden_dim).astype(np.float32)
        init_c = np.random.rand(batch, hidden_dim).astype(np.float32)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            res = _lstm_block_cell(
                x,
                tf.constant(init_c),
                tf.constant(init_h),
                w=tf.constant(
                    np.random.rand(input_dim + hidden_dim, 4 * hidden_dim).astype(
                        np.float32
                    )
                ),
                b=tf.constant(np.random.rand(4 * hidden_dim).astype(np.float32)),
                use_peephole=has_peephole,
                wci=tf.constant(np.random.rand(hidden_dim).astype(np.float32)),
                wcf=tf.constant(np.random.rand(hidden_dim).astype(np.float32)),
                wco=tf.constant(np.random.rand(hidden_dim).astype(np.float32)),
                forget_bias=np.random.rand(),
                cell_clip=np.random.rand() if has_clip else -1,
            )
            if return_hc_only:
                # All other outputs aren't supported by mb.lstm.
                res = res[1], res[6]

            TensorFlowBaseTest.run_compare_tf(
                graph,
                {x: np.random.rand(*x_shape).astype(np.float32),},
                res,
                use_cpu_only=use_cpu_only,
                frontend_only=False,
                backend=backend,
            )

    @pytest.mark.xfail(
        reason="Revert the assumption of invoking set_global before get_global: <rdar://problem/63326545>",
        run=False,
    )
    @pytest.mark.parametrize(
        "use_cpu_only, backend, batch",
        itertools.product([True, False], backends, [1, 2],),
    )
    def test_tf_lstm_block_cell(self, use_cpu_only, backend, batch):
        actual_len, padded_len = 3, 4
        input_dim, hidden_dim = 2, 3
        # [timelen, batch_size, num_inputs]
        x_shape = (batch, input_dim)
        init_h = np.random.rand(batch, hidden_dim).astype(np.float32)
        init_c = np.random.rand(batch, hidden_dim).astype(np.float32)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            rnn_cell = tf.contrib.rnn.LSTMBlockCell(
                hidden_dim, use_peephole=True, forget_bias=np.random.rand()
            )
            res = rnn_cell(x, (init_h, init_c))
            cs_new, h_new = res[1][0], res[1][1]
            res = [h_new, cs_new]

            TensorFlowBaseTest.run_compare_tf(
                graph,
                {x: np.random.rand(*x_shape).astype(np.float32),},
                res,
                use_cpu_only=use_cpu_only,
                frontend_only=False,
                backend=backend,
                # variable needs to be frozen
                freeze_graph=True,
            )


class TestVariable(TensorFlowBaseTest):
    @pytest.mark.xfail(reason="Investigate get_global <rdar://79621723>", run=False)
    @pytest.mark.parametrize(
        "use_cpu_only, backend", itertools.product([True], backends,)
    )
    def test_tf_no_variable(self, use_cpu_only, backend):
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=[1,], name="input")
            y = tf.Variable([1.0], dtype=tf.float32, name="y")

            # We set our assign op
            assign_op = tf.assign(y, y + 10)

            with tf.control_dependencies([assign_op]):
                res = tf.multiply(x, y, name="output")

            TensorFlowBaseTest.run_compare_tf(
                graph,
                {x: np.random.rand(1).astype(np.float32),},
                res,
                use_cpu_only=use_cpu_only,
                frontend_only=False,
                backend=backend,
            )


class TestZerosLike(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, rank, dynamic",
        itertools.product(
            [True, False], backends, [rank for rank in range(5)], [True, False],
        ),
    )
    def test(self, use_cpu_only, backend, rank, dynamic):
        if rank == 0:
            pytest.xfail('Rank 0 not supported by CoreML runtime')
        input_shape = np.random.randint(low=2, high=4, size=rank)
        input_value = random_gen(input_shape, rand_min=-1, rand_max=1)
        if dynamic:
            a, b = np.prod(input_shape[:2]), np.prod(input_shape[2:])
            reshape_vals = np.array([a, b], dtype=np.int32)
            reshape_input_shape = np.array([2], dtype=np.int32)

            @make_tf_graph([input_shape, list(reshape_input_shape) + [tf.int32]])
            def build_model(x, reshape):
                x = tf.reshape(x, shape=reshape)
                return tf.raw_ops.ZerosLike(x=x)

            model, inputs, outputs = build_model
            input_values = [input_value, reshape_vals]
        else:

            @make_tf_graph([input_shape])
            def build_model(x):
                return tf.raw_ops.ZerosLike(x=x)

            model, inputs, outputs = build_model
            input_values = [input_value]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestIsFinite(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend, rank, dynamic",
        itertools.product(
            [True, False], backends, [rank for rank in range(5)], [True, False]
        ),
    )
    def test(self, use_cpu_for_conversion, backend, rank, dynamic):
        if rank == 0:
            pytest.xfail('Rank 0 not supported by CoreML runtime')

        if backend[0] == "mlprogram" and not use_cpu_for_conversion:
            pytest.xfail("rdar://78343191")

        def _generate_num_with_inf(input_shape):
            res = random_gen(input_shape, rand_min=-1, rand_max=1)
            random_map = np.random.choice([np.inf, -np.inf, 0], size=input_shape)
            if len(input_shape) == 0:
                return random_map.astype(np.float32)
            res[np.where(random_map == np.inf)] = np.inf
            res[np.where(random_map == -np.inf)] = -np.inf
            return res.astype(np.float32)

        input_shape = np.random.randint(low=2, high=4, size=rank)
        input_value = _generate_num_with_inf(input_shape)
        if dynamic:
            reshape_shape = [2, tf.int32]

            if len(input_shape) == 0:
                reshape_value = np.array([1, 1], dtype=np.int32)
            else:
                reshape_value = np.array(
                    [input_shape[0], np.prod(input_shape[1:])], dtype=np.int32
                )

            @make_tf_graph([input_shape, reshape_shape])
            def build_model(x, reshape):
                x = tf.reshape(x, reshape)
                x = tf.raw_ops.IsFinite(x=x)
                return tf.raw_ops.Cast(x=x, DstT=tf.float32)

            model, inputs, outputs = build_model
            input_values = [input_value, reshape_value]

        else:

            @make_tf_graph([input_shape])
            def build_model(x):
                x = tf.raw_ops.IsFinite(x=x)
                return tf.raw_ops.Cast(x=x, DstT=tf.float32)

            model, inputs, outputs = build_model
            input_values = [input_value]

        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            frontend_only=False,
            backend=backend,
            use_cpu_for_conversion=use_cpu_for_conversion,
            use_cpu_only=use_cpu_for_conversion,
        )

class TestLogSoftMax(TensorFlowBaseTest):
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test(self, use_cpu_only, backend):
        if backend == ("mlprogram", "fp16") and not use_cpu_only:
            pytest.xfail("operation is ill-conditioned on FP16")
        input_shape = (5, 20)
        input_value = random_gen(input_shape, rand_min=-10, rand_max=10)
        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.math.log_softmax(x)

        model, inputs, outputs = build_model
        input_values = [input_value]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(model, input_dict, outputs,
                       use_cpu_only=use_cpu_only,
                       frontend_only=False, backend=backend)


class TestClipByValue(TensorFlowBaseTest):
    @pytest.mark.parametrize('use_cpu_only, backend, rank, min_and_max',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(5)],
                                 [(-1,1),(-1,-1),(1,2),(-3,-2)],
                             ))
    def test(self, use_cpu_only, backend, rank, min_and_max):
        if rank == 0:
            pytest.xfail('Rank 0 not supported by CoreML runtime')

        input_shape = np.random.randint(low=2, high=4, size=rank)
        min_val, max_val = min_and_max
        input_value = random_gen(input_shape, rand_min=min_val-1, rand_max=max_val+1)
        @make_tf_graph([input_shape])
        def build_model(x):
            return tf.raw_ops.ClipByValue(t=x, clip_value_min=min_val, clip_value_max=max_val)

        model, inputs, outputs = build_model
        input_values = [input_value]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(model, input_dict, outputs,
                       use_cpu_only=use_cpu_only,
                       frontend_only=False, backend=backend)


class TestSize(TensorFlowBaseTest):
    @pytest.mark.parametrize('use_cpu_only, backend, rank, dynamic',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(5)],
                                 [True, False],
                             ))
    def test(self, use_cpu_only, backend, rank, dynamic):
        if rank == 0:
            pytest.xfail('Rank 0 not supported by CoreML runtime')

        input_shape = np.random.randint(low=2, high=4, size=rank)
        input_value = random_gen(input_shape, rand_min=-1, rand_max=1)
        if dynamic:
            a, b = np.prod(input_shape[:2]), np.prod(input_shape[2:])
            reshape_vals = np.array([a,b], dtype=np.int32)
            reshape_input_shape = np.array([2], dtype=np.int32)

            @make_tf_graph([input_shape, list(reshape_input_shape)+[tf.int32]])
            def build_model(x, reshape):
                x = tf.reshape(x, shape=reshape)
                return tf.raw_ops.Size(input=x)

            model, inputs, outputs = build_model
            input_values = [input_value, reshape_vals]
        else:
            @make_tf_graph([input_shape])
            def build_model(x):
                return tf.raw_ops.Size(input=x)

            model, inputs, outputs = build_model
            input_values = [input_value]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(model, input_dict, outputs,
                       use_cpu_only=use_cpu_only,
                       frontend_only=False, backend=backend)

class TestAudioSpectrogram(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, params, magnitude_squared",
        itertools.product(
            [True, False],
            backends,
            [((100, 2), 5, 10),
             ((50, 1), 18, 2),
             ((512, 1), 512, 320),
            ],
            [True, False],
        ),
    )
    def test_audio_spectrogram(self, use_cpu_only, backend, params, magnitude_squared):
        input_shape = params[0]
        window_size = params[1]
        stride = params[2]

        @make_tf_graph([input_shape])
        def build_model(x):
            y = tf.raw_ops.AudioSpectrogram(input=x,
                                            window_size=window_size,
                                            stride=stride,
                                            magnitude_squared=magnitude_squared)
            return y

        model, inputs, outputs = build_model

        input_values = [(2 * np.random.rand(*input_shape) - 1).astype(np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )

class TestMfcc(TensorFlowBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_only, backend, params",
        itertools.product(
            [True, False],
            backends,
            [((100, 2), 5, 10, 8000, (40, 4000), 20, 13),
             ((50, 1), 18, 2, 4000, (20, 1500), 40, 26),
             ((512, 1), 512, 320, 16000, (20, 8000), 40, 26),
            ],
        ),
    )
    def test_mfcc(self, use_cpu_only, backend, params):
        if backend == ("mlprogram", "fp16"):
            pytest.xfail("rdar://80660411 (MFCC FP16 unit tests failing in TF1 converter with numerical errors)")

        input_shape = params[0]
        window_size = params[1]
        stride = params[2]
        sample_rate = params[3]
        lower_frequency_limit, upper_frequency_limit = params[4]
        filterbank_channel_count = params[5]
        dct_coefficient_count = params[6]


        @make_tf_graph([input_shape])
        def build_model(x):
            y = tf.raw_ops.AudioSpectrogram(input=x,
                                            window_size=window_size,
                                            stride=stride,
                                            magnitude_squared=True)
            y_out = tf.raw_ops.Mfcc(spectrogram=y,
                                    sample_rate=sample_rate,
                                    upper_frequency_limit=upper_frequency_limit,
                                    lower_frequency_limit=lower_frequency_limit,
                                    filterbank_channel_count=filterbank_channel_count,
                                    dct_coefficient_count=dct_coefficient_count)
            return y_out

        model, inputs, outputs = build_model

        input_values = [(2 * np.random.rand(*input_shape) - 1).astype(np.float32)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(
            model,
            input_dict,
            outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )
