from ._test_utils import get_core_ml_prediction
from . import _test_reqs
from ._test_reqs import *
backends = _test_reqs.backends

class TestRandomBernoulli:
    @pytest.mark.parametrize('use_cpu_only, backend',
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):

        x_val = np.array([0.], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.add(x=x, y=x),
                cb.random_bernoulli(shape=np.array([2, 1, 3], np.int32), prob=1.0),
                cb.random_bernoulli(shape=np.array([3, 1, 2], np.int32), prob=0.0),
            ]

        expected_outputs = [
            np.array(np.zeros(shape=(1,)), np.float32),
            np.array(np.ones(shape=(2, 1, 3)), np.float32),
            np.array(np.zeros(shape=(3, 1, 2)), np.float32),
        ]

        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values,
                expected_output_types,
                expected_outputs=expected_outputs,
                use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.parametrize('use_cpu_only, backend, rank, prob',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 6)],
                                 [1.0, 0.0]))
    def test_builder_to_backend_stress(self, use_cpu_only, backend, rank, prob):
        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)
        x_val = np.array([0.], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.add(x=x, y=x),
                cb.random_bernoulli(shape=shape, prob=prob)
            ]

        expected_outputs = [
            np.array(np.zeros(shape=(1,)), np.float32),
            np.random.binomial(1, prob, shape)
        ]

        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values, expected_output_types,
                            expected_outputs=expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, size, rank',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [size for size in range(1, 5)],
                                 [rank for rank in range(1, 6)],
                             ))
    def test_tf_keras(self, use_cpu_only, backend, size, rank):
        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.add(x, tf.keras.backend.random_binomial(shape=shape, p=1.0))
            run_compare_tf(graph, {x: np.random.rand(*shape)},
                           ref, use_cpu_only=use_cpu_only, backend=backend)


class TestRandomCategorical:
    def softmax(self, data):
        e_data = np.exp(data - np.max(data))
        return e_data / e_data.sum()

    @pytest.mark.parametrize('use_cpu_only, backend',
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([1], dtype=np.int32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.random_categorical(x=x, seed=1),
                cb.random_categorical(x=x, seed=1, size=4),
            ]

        expected_outputs = [
            np.array(np.zeros(shape=(1,)), dtype=np.float32),
            np.array(np.zeros(shape=(4,)), dtype=np.float32),
        ]

        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs=expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.parametrize('use_cpu_only, backend, n_sample, n_class',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [50000],
                                 [2, 10, 20]))
    def test_builder_to_backend_stress(self, use_cpu_only, backend,
            n_sample, n_class):
        output_name = 'random_categorical'
        logits = np.random.rand(2, n_class)
        probs = [self.softmax(logits[0]), self.softmax(logits[1])]

        # Test logits input
        input_placeholders = {'x': cb.placeholder(shape=(2, n_class))}
        input_values = {'x': logits}

        def build(x):
            return [cb.random_categorical(
                x=x, size=n_sample, mode='logits', name=output_name)
            ]

        prediction = get_core_ml_prediction(build, input_placeholders,
                input_values, backend=backend)

        ref0 = np.random.multinomial(n_sample, probs[0])
        ref1 = np.random.multinomial(n_sample, probs[1])

        pred0 = prediction[output_name].reshape(2, n_sample)[0]
        pred1 = prediction[output_name].reshape(2, n_sample)[1]

        # convert to bincount and validate probabilities
        pred0 = np.bincount(np.array(pred0).astype(np.int), minlength=n_class)
        pred1 = np.bincount(np.array(pred1).astype(np.int), minlength=n_class)

        assert np.allclose(np.true_divide(pred0, n_sample), probs[0], atol=1e-2)
        assert np.allclose(np.true_divide(pred0, n_sample),
                           np.true_divide(ref0, n_sample), atol=1e-2)

        assert np.allclose(np.true_divide(pred1, n_sample), probs[1], atol=1e-2)
        assert np.allclose(np.true_divide(pred1, n_sample),
                           np.true_divide(ref1, n_sample), atol=1e-2)

        # Test probs input
        input_placeholders = {'x': cb.placeholder(shape=(2, n_class))}
        input_values = {'x': np.array(probs)}

        def build(x):
            return [cb.random_categorical(
                x=x, size=n_sample, mode='probs', name=output_name)
            ]

        prediction = get_core_ml_prediction(build, input_placeholders,
                input_values, backend=backend)

        pred0 = prediction[output_name].reshape(2, n_sample)[0]
        pred1 = prediction[output_name].reshape(2, n_sample)[1]

        # convert to bincount and validate probabilities
        pred0 = np.bincount(np.array(pred0).astype(np.int), minlength=n_class)
        pred1 = np.bincount(np.array(pred1).astype(np.int), minlength=n_class)

        assert np.allclose(np.true_divide(pred0, n_sample), probs[0], atol=1e-2)
        assert np.allclose(np.true_divide(pred0, n_sample),
                           np.true_divide(ref0, n_sample), atol=1e-2)

        assert np.allclose(np.true_divide(pred1, n_sample), probs[1], atol=1e-2)
        assert np.allclose(np.true_divide(pred1, n_sample),
                           np.true_divide(ref1, n_sample), atol=1e-2)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, size',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [size for size in range(1, 10)]))
    def test_tf(self, use_cpu_only, backend, size):
        # TensorFlow's input is 2-D tensor with shape [batch_size, num_classes].
        shape = np.random.randint(low=1, high=6, size=2)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.random.categorical(x, size)
            run_compare_tf(graph, {x: np.random.rand(*shape)},
                           ref, use_cpu_only=use_cpu_only,
                           validate_shapes_only=True, backend=backend)


class TestRandomNormal:
    @pytest.mark.parametrize('use_cpu_only, backend',
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([0.], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.add(x=x, y=x),
                cb.random_normal(shape=np.array([2, 1, 3], np.int32), mean=1., stddev=0.),
                cb.random_normal(shape=np.array([3, 1, 2], np.int32), mean=0., stddev=0.),
            ]

        expected_outputs = [
            np.array(np.zeros(shape=(1,)), np.float32),
            np.array(np.ones(shape=(2, 1, 3)), np.float32),
            np.array(np.zeros(shape=(3, 1, 2)), np.float32),
        ]

        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values, expected_output_types,
                            expected_outputs=expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.parametrize('use_cpu_only, backend, rank, mean',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 6)],
                                 [1.0, 0.0]))
    def test_builder_to_backend_stress(self, use_cpu_only, backend, rank, mean):
        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)
        x_val = np.array([0.], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.add(x=x, y=x),
                cb.random_normal(shape=shape, mean=mean, stddev=0.)
            ]

        expected_outputs = [
            np.array(np.zeros(shape=(1,)), np.float32),
            np.random.normal(loc=mean, scale=0., size=shape)
        ]

        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values, expected_output_types,
                            expected_outputs=expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, mean, rank',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [0.],
                                 [rank for rank in range(1, 6)]
                             ))
    def test_tf(self, use_cpu_only, backend, mean, rank):
        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.add(x, tf.random.normal(shape=shape, mean=mean, stddev=0.))
            run_compare_tf(graph, {x: np.random.rand(*shape)},
                           ref, use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, mean, rank',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [0.],
                                 [rank for rank in range(1, 6)]
                             ))
    def test_tf_keras(self, use_cpu_only, backend, mean, rank):
        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.add(x, tf.keras.backend.random_normal(shape=shape, mean=mean, stddev=0.))
            run_compare_tf(graph, {x: np.random.rand(*shape)},
                           ref, use_cpu_only=use_cpu_only, backend=backend)


class TestRandomUniform:
    @pytest.mark.parametrize('use_cpu_only, backend',
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([0.], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.add(x=x, y=x),
                cb.random_uniform(shape=np.array([2, 1, 3], np.int32), low=0., high=0.),
                cb.random_uniform(shape=np.array([3, 1, 2], np.int32), low=1., high=1.),
            ]

        expected_outputs = [
            np.array(np.zeros(shape=(1,)), np.float32),
            np.array(np.zeros(shape=(2, 1, 3)), np.float32),
            np.array(np.ones(shape=(3, 1, 2)), np.float32),
        ]

        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values, expected_output_types,
                            expected_outputs=expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.parametrize('use_cpu_only, backend, rank, low, high',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 6)],
                                 [0.0], [0.0]))
    def test_builder_to_backend_stress(self, use_cpu_only, backend, rank, low, high):
        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)
        x_val = np.array([0.], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.add(x=x, y=x),
                cb.random_uniform(shape=shape, low=low, high=high)
            ]

        expected_outputs = [
            np.array(np.zeros(shape=(1,)), np.float32),
            np.random.uniform(low=low, high=high, size=shape)
        ]

        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values, expected_output_types,
                            expected_outputs=expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, low, high, rank',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [0.], [0.], [rank for rank in range(1, 2)]))
    def test_tf(self, use_cpu_only, backend, low, high, rank):
        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.add(x, tf.random.uniform(shape=shape, minval=low, maxval=high))
            run_compare_tf(graph, {x: np.random.rand(*shape)},
                           ref, use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, low, high, rank',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [1.], [1.], [rank for rank in range(1, 6)]))
    def test_tf_keras(self, use_cpu_only, backend, low, high, rank):
        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.add(x, tf.keras.backend.random_uniform(shape=shape, minval=low, maxval=high))
            run_compare_tf(graph, {x: np.random.rand(*shape)},
                           ref, use_cpu_only=use_cpu_only, backend=backend)
