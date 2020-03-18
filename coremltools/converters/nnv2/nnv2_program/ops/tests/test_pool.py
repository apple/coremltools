from . import _test_reqs
from ._test_reqs import *
backends = _test_reqs.backends

class TestAvgPool:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([
            [[[-10.80291205, -6.42076184],
              [-7.07910997, 9.1913279]],
             [[-3.18181497, 0.9132147],
              [11.9785544, 7.92449539]]]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.avg_pool(x=x, kernel_sizes=[1, 2], strides=[2, 1], pad_type='valid'),
            ]

        expected_output_types = [(1, 2, 1, 1, builtins.fp32)]
        expected_outputs = [np.array([[[[-8.611837]], [[-1.1343001]]]], dtype=np.float32)]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, 
                            frontend_only=False, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, kernel_sizes, strides, pad_type',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(1,)],
                                 [(1,), (2,)],
                                 ['same', 'valid']))
    def test_tf_avg_pool_1d(self, use_cpu_only, backend, kernel_sizes, strides, pad_type):
        input_shape = np.random.randint(low=2, high=6, size=3)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.nn.avg_pool1d(x, ksize=kernel_sizes[:], strides=strides[:], padding=pad_type.upper())
            run_compare_tf(graph, {x: random_gen(input_shape, rand_min=-100, rand_max=100)},
                           res, use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, kernel_sizes, strides, pad_type',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(1,), (2,), (1, 1), (1, 2), (2, 2)],
                                 [(1,), (2,), (1, 1), (1, 2), (2, 2)],
                                 ['same', 'valid']))
    def test_tf_avg_pool_2d(self, use_cpu_only, backend, kernel_sizes, strides, pad_type):
        shape = np.random.randint(low=2, high=6, size=4)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            res = tf.nn.avg_pool(x, ksize=kernel_sizes[:], strides=strides[:], padding=pad_type.upper())
            run_compare_tf(graph, {x: random_gen(shape, rand_min=-100, rand_max=100)},
                           res, use_cpu_only=use_cpu_only, backend=backend)


class TestL2Pool:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([
            [[[-10., -6.], [-7., 9.]], [[-3., 0.], [11., 7.]]]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.l2_pool(x=x, kernel_sizes=[1, 2], strides=[2, 1], pad_type='valid'),
            ]

        expected_output_types = [(1, 2, 1, 1, builtins.fp32)]
        expected_outputs = [np.array([[[[11.66190338]], [[3.]]]], dtype=np.float32)]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False, backend=backend)


class TestMaxPool:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([
            [[[-10.80291205, -6.42076184],
              [-7.07910997, 9.1913279]],
             [[-3.18181497, 0.9132147],
              [11.9785544, 7.92449539]]]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [cb.max_pool(x=x, kernel_sizes=[1, 2], strides=[2, 1], pad_type='valid')]

        expected_output_types = [(1, 2, 1, 1, builtins.fp32)]
        expected_outputs = [np.array([[[[-6.42076184]], [[0.9132147]]]], dtype=np.float32)]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not installed.')
    @pytest.mark.parametrize('use_cpu_only, backend, kernel_sizes, strides, pad_type',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(1,)],
                                 [(1,), (2,)],
                                 ['same', 'valid']))
    def test_tf_max_pool_1d(self, use_cpu_only, backend, kernel_sizes, strides, pad_type):
        input_shape = np.random.randint(low=2, high=6, size=3)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.nn.max_pool1d(x, ksize=kernel_sizes[:], strides=strides[:], padding=pad_type.upper())
            run_compare_tf(graph, {x: random_gen(input_shape, rand_min=-100, rand_max=100)},
                           res, use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not installed.')
    @pytest.mark.parametrize('use_cpu_only, backend, kernel_sizes, strides, pad_type',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(1,), (2,), (1, 1), (1, 2), (2, 2)],
                                 [(1,), (2,), (1, 1), (1, 2), (2, 2)],
                                 ['same', 'valid']))
    def test_tf_max_pool_2d(self, use_cpu_only, backend, kernel_sizes, strides, pad_type):
        shape = np.random.randint(low=2, high=6, size=4)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            res = tf.nn.max_pool(x, ksize=kernel_sizes[:], strides=strides[:], padding=pad_type.upper())
            run_compare_tf(graph, {x: random_gen(shape, rand_min=-100, rand_max=100)},
                           res, use_cpu_only=use_cpu_only, backend=backend)
