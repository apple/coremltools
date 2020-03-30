from . import _test_reqs
from ._test_reqs import *
backends = _test_reqs.backends

class TestSelect:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        cond_val = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]], dtype=np.float32)
        a_val = np.array([[3, 1, 1], [1, 4, 1], [5, 6, 1]], dtype=np.float32)
        b_val = np.array([[3, 2, 2], [2, 4, 2], [5, 6, 2]], dtype=np.float32)
        input_placeholders = {
            'cond': cb.placeholder(shape=cond_val.shape),
            'a': cb.placeholder(shape=a_val.shape),
            'b': cb.placeholder(shape=b_val.shape),
        }
        input_values = {'cond': cond_val, 'a': a_val, 'b': b_val}

        def build(cond, a, b):
            return [cb.select(cond=cond, a=a, b=b)]

        expected_output_types = [(3, 3, builtins.fp32)]
        expected_outputs = [
            np.array([[3., 2., 2.], [2., 4., 2.], [5., 6., 2.]], dtype=np.float32)
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        cond = np.random.randint(low=0, high=2, size=(6, 1, 7))
        a = random_gen(shape=(6, 1, 7), rand_min=-1962., rand_max=0.)
        b = random_gen(shape=(6, 1, 7), rand_min=0., rand_max=1964.)
        res = cb.select(cond=cond, a=a, b=b)
        assert is_close(np.where(cond, a, b), res.val)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize('use_cpu_only, backend, rank',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 6)]
                             ))
    def test_tf1(self, use_cpu_only, backend, rank):
        shape = np.random.randint(low=1, high=4, size=rank)
        cond_val = np.random.randint(low=0, high=2, size=shape).astype(np.int32)
        a_val = random_gen(shape=shape, rand_min=-1962., rand_max=0.)
        b_val = random_gen(shape=shape, rand_min=0., rand_max=1964.)
        with tf.Graph().as_default() as graph:
            cond = tf.placeholder(tf.bool, shape=shape)
            a = tf.placeholder(tf.float32, shape=shape)
            b = tf.placeholder(tf.float32, shape=shape)
            ref = tf.where(cond, a, b)
            run_compare_tf1(graph, {cond: cond_val, a: a_val, b: b_val}, ref,
                            use_cpu_only=use_cpu_only, backend=backend)


class TestCond:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):

        input_placeholders = {
                "a": cb.placeholder(shape=(1,), dtype=builtins.bool),
                "b": cb.placeholder(shape=(1,)),
                }
        def build(a, b):
            def true_fn():
                return cb.add(x=b, y=1), cb.mul(x=b, y=2)

            def false_fn():
                return cb.add(x=b, y=-1), cb.mul(x=b, y=-2)

            pred = cb.squeeze(x=a)
            return cb.cond(pred=pred, _true_fn=true_fn, _false_fn=false_fn)

        input_values = {
                "a": np.array([0], dtype=np.float32),
                "b": np.array([2], dtype=np.float32),
                }

        expected_output_types = [
                (1, builtins.fp32),
                (1, builtins.fp32),
                ]

        expected_outputs = [
                np.array([1], dtype=np.float32),
                np.array([-4], dtype=np.float32),
                ]
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize("use_cpu_only, backend",
                             itertools.product(
                                 [True, False],
                                 backends,
                             )
                             )
    def test_tf1(self, use_cpu_only, backend):
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=(1,))
            y = tf.placeholder(tf.float32, shape=(1,))
            z = tf.multiply(x, y)
            pred = tf.less(tf.math.reduce_mean(x), tf.math.reduce_mean(y))
            res = tf.cond(pred, lambda: tf.add(x, z), lambda: tf.square(y))
            run_compare_tf1(graph,
                            {x: np.array([1], dtype=np.float32),
                            y: np.array([2], dtype=np.float32),
                            },
                            res, use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize("use_cpu_only, backend",
                             itertools.product(
                                 [True, False],
                                 backends,
                             )
                             )
    def test_tf1_multi_returns(self, use_cpu_only, backend):
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=(1,))
            y = tf.placeholder(tf.float32, shape=(1,))
            z = tf.multiply(x, y)
            pred = tf.less(tf.math.reduce_mean(x), tf.math.reduce_mean(y))
            def true_fn(): return tf.add(x, z), x
            def false_fn(): return tf.square(y), z
            res = tf.cond(pred, true_fn, false_fn)
            run_compare_tf1(graph,
                            {x: np.array([1], dtype=np.float32),
                            y: np.array([2], dtype=np.float32),
                            },
                            res, use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)

class TestWhileLoop:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        def body(a, b):
            return cb.add(x=a, y=np.float32(1)), b

        def cond(a, b):
            return cb.less(x=a, y=b)

        input_placeholders = {
                "a": cb.placeholder(shape=(1,)),
                "b": cb.placeholder(shape=(1,)),
                }
        def build(a, b):
            return cb.while_loop(_cond=cond, _body=body, loop_vars=(a, b))

        input_values = {
                "a": np.array([1], dtype=np.float32),
                "b": np.array([2], dtype=np.float32),
                }

        expected_output_types = [
                (1, builtins.fp32),
                (1, builtins.fp32),
                ]

        expected_outputs = [
                np.array([2], dtype=np.float32),
                np.array([2], dtype=np.float32),
                ]
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize("use_cpu_only, backend",
                             itertools.product(
                                 [True, False],
                                 ['nnv1_proto'],
                             )
                             )
    def test_tf1(self, use_cpu_only, backend):
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=(1,))
            y = tf.placeholder(tf.float32, shape=(1,))
            c = lambda i, j: \
                    tf.less(tf.math.reduce_mean(i), tf.math.reduce_mean(j))
            b = lambda i, j: (tf.add(i, 1), j)
            res = tf.while_loop(c, b, [x, y])
            run_compare_tf1(graph,
                            {x: np.array([1], dtype=np.float32),
                            y: np.array([2], dtype=np.float32),
                               },
                            res, use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize("use_cpu_only, backend",
                             itertools.product(
                                 [True, False],
                                 backends,
                             )
                             )
    def test_tf1_nested_while_body(self, use_cpu_only, backend):
        with tf.Graph().as_default() as graph:
            # The following while loop:
            #
            # i, j = 0, 10
            # while i < j:
            #   while 2*i < i+2:
            #     i += 1
            #   i += 2
            x = tf.placeholder(tf.float32, shape=(1,))
            y = tf.placeholder(tf.float32, shape=(1,))
            def cond2(i):
                return tf.less(2*tf.math.reduce_mean(i), tf.math.reduce_mean(i+2))
            def body2(i):
                return i+1
            def cond1(i, j):
                return tf.less(tf.math.reduce_mean(i), tf.math.reduce_mean(j))
            def body1(i, j):
                new_i = tf.while_loop(cond2, body2, [i])
                return new_i + 2, j
            res = tf.while_loop(cond1, body1, [x, y])
            run_compare_tf1(graph,
                            {x: np.array([0], dtype=np.float32),
                            y: np.array([10], dtype=np.float32),
                               },
                            res, use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize("use_cpu_only, backend",
                             itertools.product(
                                 [True, False],
                                 backends,
                             )
                             )
    def test_tf1_nested_while_cond(self, use_cpu_only, backend):
        with tf.Graph().as_default() as graph:
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
            x = tf.placeholder(tf.float32, shape=(1,))
            y = tf.placeholder(tf.float32, shape=(1,))
            def cond2(i):
                return tf.less(2*tf.math.reduce_mean(i), tf.math.reduce_mean(i+2))
            def body2(i):
                return i+1
            def cond1(i, j):
                new_i = tf.while_loop(cond2, body2, [i])
                return tf.less(tf.squeeze(new_i), tf.squeeze(j))
            def body1(i, j):
                return i + 2, j + 1
            res = tf.while_loop(cond1, body1, [x, y])
            run_compare_tf1(graph,
                            {x: np.array([0], dtype=np.float32),
                            y: np.array([10], dtype=np.float32),
                               },
                            res, use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)
