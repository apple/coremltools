from . import _test_reqs
from ._test_reqs import *
backends = _test_reqs.backends

class TestElementwiseBinary:
    # All in this test share the same backends
    @pytest.mark.parametrize('use_cpu_only, backend, mode',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 ['add', 'floor_div', 'maximum', 'minimum',
                                  'mod', 'mul', 'pow', 'real_div', 'sub']
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend, mode):
        if mode == 'add':
            x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[0, 4, 0], [8, 0, 12]], dtype=np.float32)

            build = lambda x, y: cb.add(x=x, y=y)
        elif mode == 'floor_div':
            x = np.array([[10, 20, 30],[40, 50, 60]], dtype=np.float32)
            y = np.array([[11, 12, 13],[14, 15, 16]], dtype=np.float32)
            expected_outputs = np.array([[0, 1, 2], [2, 3, 3]], dtype=np.float32)

            build = lambda x, y: cb.floor_div(x=x, y=y)
        elif mode == 'maximum':
            x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

            build = lambda x, y: cb.maximum(x=x, y=y)
        elif mode == 'minimum':
            x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)

            build = lambda x, y: cb.minimum(x=x, y=y)
        elif mode == 'mod':
            x = np.array([[10, 20, 30],[40, 50, 60]], dtype=np.float32)
            y = np.array([[11, 12, 13],[14, 15, 16]], dtype=np.float32)
            expected_outputs = np.array([[10, 8, 4], [12, 5, 12]], dtype=np.float32)

            build = lambda x, y: cb.mod(x=x, y=y)
        elif mode == 'mul':
            x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[-1, 4, -9], [16, -25, 36]], dtype=np.float32)

            build = lambda x, y: cb.mul(x=x, y=y)
        elif mode == 'pow':
            x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[1, 4, 0.037], [256, 0.00032, 46656]], dtype=np.float32)

            build = lambda x, y: cb.pow(x=x, y=y)
        elif mode == 'real_div':
            x = np.array([[10, 20, 30],[40, 50, 60]], dtype=np.float32)
            y = np.array([[11, 12, 13],[14, 15, 16]], dtype=np.float32)
            expected_outputs = np.array([[0.90909091, 1.66666667, 2.30769231],
                                         [2.85714286, 3.33333333, 3.75]],
                                         dtype=np.float32)

            build = lambda x, y: cb.real_div(x=x, y=y)
        elif mode == 'sub':
            x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[2, 0, 6], [0, 10, 0]], dtype=np.float32)

            build = lambda x, y: cb.sub(x=x, y=y)

        expected_output_types = (2, 3, builtins.fp32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_add(self):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[0, 4, 0], [8, 0, 12]], dtype=np.float32)
        v = cb.add(x=x, y=y)
        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_floor_div(self):
        x = np.array([[10, 20, 30],[40, 50, 60]], dtype=np.float32)
        y = np.array([[11, 12, 13],[14, 15, 16]], dtype=np.float32)
        expected_outputs = np.array([[0, 1, 2], [2, 3, 3]], dtype=np.float32)
        v = cb.floor_div(x=x, y=y)
        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_maximum(self):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        v = cb.maximum(x=x, y=y)
        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_minimum(self):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        v = cb.minimum(x=x, y=y)
        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_mod(self):
        x = np.array([[10, 20, 30],[40, 50, 60]], dtype=np.float32)
        y = np.array([[11, 12, 13],[14, 15, 16]], dtype=np.float32)
        expected_outputs = np.array([[10, 8, 4], [12, 5, 12]], dtype=np.float32)
        v = cb.mod(x=x, y=y)
        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_mul(self):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[-1, 4, -9], [16, -25, 36]], dtype=np.float32)
        v = cb.mul(x=x, y=y)
        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_pow(self):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[1, 4, 0.037], [256, 0.00032, 46656]], dtype=np.float32)
        v = cb.pow(x=x, y=y)
        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_real_div(self):
        x = np.array([[10, 20, 30],[40, 50, 60]], dtype=np.float32)
        y = np.array([[11, 12, 13],[14, 15, 16]], dtype=np.float32)
        expected_outputs = np.array([[0.90909091, 1.66666667, 2.30769231],
                                     [2.85714286, 3.33333333, 3.75]],
                                     dtype=np.float32)
        v = cb.real_div(x=x, y=y)
        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_sub(self):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[2, 0, 6], [0, 10, 0]], dtype=np.float32)
        v = cb.sub(x=x, y=y)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank, mode",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 4)],
                                 ['add', 'floor_div', 'maximum', 'minimum',
                                  'mod', 'mul', 'pow', 'real_div', 'sub',
                                  'squared_difference']
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank, mode):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            if mode == 'add':
                res = tf.add(x, y)
                x_val = np.random.randint(low=-1000, high=1000, size=x_shape).astype(np.float32)
                y_val = np.random.randint(low=-1000, high=1000, size=y_shape).astype(np.float32)
            elif mode == 'floor_div':
                res = tf.floor_div(x, y)
                x_val = np.random.randint(low=0, high=1000, size=x_shape).astype(np.float32)
                y_val = np.random.randint(low=1, high=20, size=y_shape).astype(np.float32)
            elif mode == 'maximum':
                res = tf.maximum(x, y)
                x_val = np.random.randint(low=-10, high=10, size=x_shape).astype(np.float32)
                y_val = np.random.randint(low=-10, high=10, size=y_shape).astype(np.float32)
            elif mode == 'minimum':
                res = tf.minimum(x, y)
                x_val = np.random.randint(low=-10, high=10, size=x_shape).astype(np.float32)
                y_val = np.random.randint(low=-10, high=10, size=y_shape).astype(np.float32)
            elif mode == 'mod':
                res = tf.mod(x, y)
                x_val = np.random.randint(low=0, high=1000, size=x_shape).astype(np.float32)
                y_val = np.random.randint(low=1, high=20, size=y_shape).astype(np.float32)
            elif mode == 'mul':
                res = tf.multiply(x, y)
                x_val = np.random.randint(low=-100, high=100, size=x_shape).astype(np.float32)
                y_val = np.random.randint(low=-100, high=100, size=y_shape).astype(np.float32)
            elif mode == 'pow':
                res = tf.pow(x, y)
                x_val = np.random.randint(low=-5, high=5, size=x_shape).astype(np.float32)
                y_val = np.random.randint(low=-5, high=5, size=y_shape).astype(np.float32)
            elif mode == 'real_div':
                res = tf.truediv(x, y)
                x_val = np.random.randint(low=0, high=1000, size=x_shape).astype(np.float32)
                y_val = np.random.randint(low=1, high=20, size=y_shape).astype(np.float32)
            elif mode == 'sub':
                res = tf.subtract(x, y)
                x_val = np.random.randint(low=-1000, high=1000, size=x_shape).astype(np.float32)
                y_val = np.random.randint(low=-1000, high=1000, size=y_shape).astype(np.float32)
            elif mode == 'squared_difference':
                if backend == 'nnv2_proto': return  # TODO
                res = tf.math.squared_difference(x, y)
                x_val = np.random.randint(low=-5, high=5, size=x_shape).astype(np.float32)
                y_val = np.random.randint(low=-5, high=5, size=y_shape).astype(np.float32)

            run_compare_tf(graph, {x: x_val, y: y_val},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestEqual:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.equal(x=x, y=y), cb.equal(x=-3, y=y)
        expected_output_types = [
                (2, 3, builtins.bool),
                (2, 3, builtins.bool),
                ]
        expected_outputs = [
                np.array([[0, 1, 0], [1, 0, 1]], dtype=np.bool),
                np.array([[0, 0, 1], [0, 0, 0]], dtype=np.bool),
                ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.bool)
        v = cb.equal(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.equal(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=3, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=3, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestGreater:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.greater(x=x, y=y), cb.greater(x=x, y=3.5)
        expected_output_types = [
                (2, 3, builtins.bool),
                (2, 3, builtins.bool),
                ]
        expected_outputs = [
                np.array([[1, 0, 1], [0, 1, 0]], dtype=np.bool),
                np.array([[0, 0, 0], [1, 1, 1]], dtype=np.bool),
                ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.bool)
        v = cb.greater(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.greater(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=3, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=3, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestGreaterEqual:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.greater_equal(x=x, y=y), cb.greater_equal(x=x, y=3.5)
        expected_output_types = [
                (2, 3, builtins.bool),
                (2, 3, builtins.bool),
                ]
        expected_outputs = [
                np.array([[1, 1, 1], [1, 1, 1]], dtype=np.bool),
                np.array([[0, 0, 0], [1, 1, 1]], dtype=np.bool),
                ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.bool)
        v = cb.greater_equal(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.greater_equal(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=3, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=3, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestLess:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.less(x=x, y=y)
        expected_output_types = (2, 3, builtins.bool)
        expected_outputs = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.bool)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke2(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape)}
        input_values = {"x": x}

        def build(x):
            # y is const
            y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            return cb.less(x=x, y=y)
        expected_output_types = (2, 3, builtins.bool)
        expected_outputs = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.bool)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_broadcast(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape)}
        input_values = {"x": x}

        def build(x):
            # y is const
            return cb.less(x=x, y=3.5)
        expected_output_types = (2, 3, builtins.bool)
        expected_outputs = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.bool)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.bool)
        v = cb.less(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.less(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=3, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=3, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestLessEqual:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.less_equal(x=x, y=y)
        expected_output_types = (2, 3, builtins.bool)
        expected_outputs = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.bool)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.bool)
        v = cb.less_equal(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.less_equal(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=3, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=3, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestNotEqual:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.not_equal(x=x, y=y)
        expected_output_types = (2, 3, builtins.bool)
        expected_outputs = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.bool)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.bool)
        v = cb.not_equal(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.not_equal(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=3, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=3, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

