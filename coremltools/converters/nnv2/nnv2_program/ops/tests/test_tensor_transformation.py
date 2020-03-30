from coremltools.converters.nnv2.nnv2_program.program import get_new_symbol, get_new_variadic_symbol
from ._test_utils import UNK_SYM, UNK_VARIADIC
from . import _test_reqs
from ._test_reqs import *
backends = _test_reqs.backends

class TestDepthToSpace:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        # original input type is (1, 4, 1, 1, fp32)
        val = np.array([[[[9.]], [[5.]], [[1.]], [[3.]]]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=val.shape)}
        input_values = {'x': val}

        def build(x):
            return [cb.depth_to_space(x=x, block_size=2)]

        expected_output_types = (1, 1, 2, 2, builtins.fp32)
        expected_outputs = np.array([[[[9., 5.], [1., 3.]]]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize('use_cpu_only, backend, shape, block_size',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(1, 1, 1, 16), (1, 1, 1, 32), (1, 3, 3, 16)],
                                 [2, 4]
                             ))
    def test_tf1(self, use_cpu_only, backend, shape, block_size):
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.depth_to_space(x, block_size)
            run_compare_tf1(graph, {x: np.random.rand(*shape)}, ref,
                            use_cpu_only=use_cpu_only, backend=backend)


class TestExpandDims:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return [cb.expand_dims(x=x, axis=0),
                    cb.expand_dims(x=x, axis=1),
                    cb.expand_dims(x=x, axis=2),
                    cb.expand_dims(x=x, axis=-1)]
        expected_output_types = [
                (1, 2, 3, builtins.fp32),
                (2, 1, 3, builtins.fp32),
                (2, 3, 1, builtins.fp32),
                (2, 3, 1, builtins.fp32),
                ]
        expected_outputs = [
                np.array([[[1, 2, 3],
                           [4, 5, 6]]], dtype=np.float32),
                np.array([[[1, 2, 3]],
                          [[4, 5, 6]]], dtype=np.float32),
                np.array([[[1],
                           [2],
                           [3]],
                          [[4],
                           [5],
                           [6]]], dtype=np.float32),
                np.array([[[1],
                           [2],
                           [3]],
                          [[4],
                           [5],
                           [6]]], dtype=np.float32),
                          ]


        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_symbolic(self, use_cpu_only, backend):
        s0 = get_new_symbol()

        input_placeholders = {
                "x": cb.placeholder(shape=(2, s0)),
                }

        def build(x):
            return [cb.expand_dims(x=x, axis=-1),
                    cb.expand_dims(x=x, axis=1),
                    ]

        expected_output_types = [
                (2, s0, 1, builtins.fp32),
                (2, 1, s0, builtins.fp32),
                ]
        expected_outputs = [
                np.array([[[1], [2], [3]],
                          [[4], [5], [6]]], dtype=np.float32),
                np.array([[[1, 2, 3]],
                          [[4, 5, 6]]], dtype=np.float32),
                ]

        input_values = {
                "x": np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32),
                }
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.random.rand(1, 1, 6, 6)
        v = cb.expand_dims(x=x_val, axis=2)
        assert is_close(np.expand_dims(x_val, 2), v.val)

    @pytest.mark.parametrize("use_cpu_only, backend, rank_and_axis",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(rank, axis) for rank in range(1, 5) for
                                     axis in range(-rank - 1, rank + 1)],
                             )
                             )
    def test_builder_to_backend_programmatic(self, use_cpu_only, backend,
            rank_and_axis):
        rank, axis = rank_and_axis
        x_shape = np.random.randint(low=2, high=6, size=rank)
        input_placeholders = {"x": cb.placeholder(shape=x_shape)}
        input_values = {"x": np.random.sample(x_shape).astype(np.float32)}
        def build(x): return cb.expand_dims(x=x, axis=axis)
        adjusted_axis = axis if axis >= 0 else rank + axis + 1
        x_shape = list(x_shape)
        out_shape = x_shape[:adjusted_axis] + [1] + x_shape[adjusted_axis:]
        expected_output_types = tuple(out_shape[:]) + (builtins.fp32,)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types,
                            np.expand_dims(input_values['x'], axis),
                            use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize("use_cpu_only, backend, rank_and_axis",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(rank, axis) for rank in range(1, 5) for
                                     axis in range(-rank - 1, rank + 1)],
                             )
                             )
    def test_tf1(self, use_cpu_only, backend, rank_and_axis):
        rank, axis = rank_and_axis
        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.expand_dims(x, axis=axis)
            run_compare_tf1(graph, {x: np.random.rand(*input_shape)},
                            res, use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)


class TestReshape:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}
        def build(x):
            return [cb.reshape(x=x, shape=[3, 2]),
                    cb.reshape(x=x, shape=[2, -1]),
                    cb.reshape(x=x, shape=[2, 1, 1, 3])]

        expected_output_types = [
                (3, 2, builtins.fp32),
                (2, 3, builtins.fp32),
                (2, 1, 1, 3, builtins.fp32),
                ]
        expected_outputs = [
                np.array([[1, 2],
                          [3, 4],
                          [5, 6]], dtype=np.float32),
                np.array([[1, 2, 3],
                          [4, 5, 6]], dtype=np.float32),
                np.array([[[[1., 2., 3.]]],
                          [[[4., 5., 6.]]]], dtype=np.float32)
                ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        r = cb.reshape(x=t, shape=[3, 2])
        expected_r = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        assert is_close(expected_r, r.val)
        r2 = cb.reshape(x=t, shape=[2, -1])
        expected_r2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        assert is_close(expected_r2, r2.val)

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_symbolic(self, use_cpu_only, backend):
        s0 = get_new_symbol()
        s_len = get_new_symbol()
        s1 = get_new_variadic_symbol()

        # Test variadic (rdar://59559656)
        input_placeholders = {
                "x": cb.placeholder(shape=(2, s0)),
                # TODO: variadic (rdar://59559656)
                #"x2": cb.placeholder(shape=(s1, 2)),
                "shape": cb.placeholder(shape=(3,), dtype=builtins.int32),
                "shape2": cb.placeholder(shape=(s_len,), dtype=builtins.int32),
                }

        def build(x, shape, shape2):
            return [cb.reshape(x=x, shape=[2, -1]),
                    cb.reshape(x=x, shape=[1, -1]),
                    cb.reshape(x=x, shape=[2, 1, 1, -1]),
                    # TODO: variadic (rdar://59559656)
                    #cb.reshape(x=x2, shape=[2, 1, 1]),
                    cb.reshape(x=x, shape=shape),
                    cb.reshape(x=x, shape=shape2),
                    ]

        expected_output_types = [
                (2, s0, builtins.fp32),
                (1, 2*s0, builtins.fp32),
                (2, 1, 1, s0, builtins.fp32),
                # TODO: variadic (rdar://59559656)
                #(2, 1, 1, builtins.fp32),
                (UNK_SYM, UNK_SYM, UNK_SYM, builtins.fp32),
                (UNK_VARIADIC, builtins.fp32),
                ]
        expected_outputs = [
                np.array([[1, 2, 3],
                          [4, 5, 6]], dtype=np.float32),
                np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32),
                np.array([[[[1., 2., 3.]]],
                          [[[4., 5., 6.]]]], dtype=np.float32),
                # TODO: variadic (rdar://59559656)
                #np.array([[1, 2, 3],
                #          [4, 5, 6]], dtype=np.float32),
                np.array([[[1, 2, 3]],
                          [[4, 5, 6]]], dtype=np.float32),
                np.array([[[1, 2, 3]],
                          [[4, 5, 6]]], dtype=np.float32),
                ]

        input_values = {
                "x": np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32),
                # TODO: variadic (rdar://59559656)
                #"x2": np.array([[[1, 2, 3],[4, 5, 6]]], dtype=np.float32),
                "shape": np.array([2, 1, 3], dtype=np.float32),
                "shape2": np.array([2, 1, 3], dtype=np.float32),
                }
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize("use_cpu_only, backend",
                             itertools.product(
                                 [False],
                                 backends,
                             ))
    def test_tf1(self, use_cpu_only, backend):
        def test_tf_static():
            # (input_shape, output_shape_as_a_parameter)
            shapes = [([10, 10], [5, 20]),
                      ([3, 4, 5, 6], [4, 5, 3, 6]),
                      ([4, 4, 5, 6], [2, 2, -1])]

            for _shape in shapes:
                with tf.Graph().as_default() as graph:
                    x = tf.placeholder(tf.float32, shape=_shape[0])
                    res = tf.reshape(x, shape=_shape[1])
                    run_compare_tf1(graph, {x: np.random.rand(*_shape[0])},
                                    res, use_cpu_only=use_cpu_only,
                                    frontend_only=False, backend=backend)
        def test_tf_dynamic():
            shapes = [([10, 10], [5, 20]),
                      ([3, 4, 5, 6], [4, 5, 3, 6]),
                      ([4, 4, 5, 6], [2, 2, -1]),
                      ([2, 3, 5, 3], [2, -1])]

            for _shape in shapes:
                with tf.Graph().as_default() as graph:
                    x = tf.placeholder(tf.float32, shape=_shape[0])
                    y = tf.placeholder(tf.int32, shape=[len(_shape[1])])
                    res = tf.reshape(x, shape=y)
                    run_compare_tf1(graph, {x: np.random.rand(*_shape[0]),
                                            # numpy
                                            y:np.array(_shape[1], dtype=np.float32)},
                                    res, use_cpu_only=use_cpu_only,
                                    frontend_only=False, backend=backend)
        test_tf_static()
        test_tf_dynamic()


class TestReverse:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        val = np.array([[-1., 2., -3.], [4., -5., 6.]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=val.shape)}
        input_values = {'x': val}

        def build(x):
            return [
                cb.reverse(x=x),
                cb.reverse(x=x, axes=[0])
            ]

        expected_output_types = [
            (2, 3, builtins.fp32),
            (2, 3, builtins.fp32)
        ]
        expected_outputs = [
            np.array([[6., -5., 4.], [-3., 2., -1.]], dtype=np.float32),
            np.array([[4., -5., 6.], [-1., 2., -3.]], dtype=np.float32)
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        val = np.array([[-1., 7., -3.], [4., -5., 8.]], dtype=np.float32)
        res = cb.reverse(x=val, axes=[0])
        assert is_close(np.flip(val, axis=0), res.val)

    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_symbolic(self, use_cpu_only, backend):
        s0 = get_new_symbol()

        val = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=(s0, 3))}
        input_values = {'x': val}

        def build(x):
            return [
                cb.reverse(x=x, axes=[1]),
                cb.reverse(x=x, axes=[0]),
            ]

        expected_output_types = [
            (s0, 3, builtins.fp32),
            (s0, 3, builtins.fp32),
        ]
        expected_outputs = [
            np.array([[3., 2., 1.], [6., 5., 4.]], dtype=np.float32),
            np.array([[4., 5., 6.], [1., 2., 3.]], dtype=np.float32),
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize('use_cpu_only, backend, rank_and_axes',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(1, (-1,)), (2, (0,)), (2, (-1, 0)),
                                  (3, (1, -3)), (3, (-2,)), (3, (0, 1, 2)),
                                  (4, (-2, -1, 0)), (4, (-1, -2)), (4, []),
                                  (5, (-3, -1, 3)), (5, (0, -1, 1, -2))],
                             ))
    def test_tf1(self, use_cpu_only, backend, rank_and_axes):
        rank, axes = rank_and_axes
        shape = np.random.randint(low=1, high=4, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            res = tf.reverse(x, axis=axes)
            run_compare_tf1(graph, {x: np.random.rand(*shape)}, res,
                            use_cpu_only=use_cpu_only, backend=backend)


class TestReverseSequence:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([[1, 2, 3, 4, 5, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0],
                          [1, 2, 3, 4, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8]],
                         dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.reverse_sequence(x=x, lengths=[7, 2, 3, 5], seq_axis=1, batch_axis=0),
            ]

        expected_output_types = [
            (4, 8, builtins.fp32),
        ]
        expected_outputs = [
            np.array([
                [0, 0, 5, 4, 3, 2, 1, 0], [2, 1, 0, 0, 0, 0, 0, 0],
                [3, 2, 1, 4, 0, 0, 0, 0], [5, 4, 3, 2, 1, 6, 7, 8]
            ], dtype=np.float32)
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_symbolic(self, use_cpu_only, backend):
        s0 = get_new_symbol()

        x_val = np.array([[1, 2, 3, 4, 5, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0],
                          [1, 2, 3, 4, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8]],
                         dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=(4, s0))}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.reverse_sequence(x=x, lengths=[7, 2, 3, 5], seq_axis=1, batch_axis=0),
            ]

        expected_output_types = [
            (4, s0, builtins.fp32),
        ]
        expected_outputs = [
            np.array([
                [0, 0, 5, 4, 3, 2, 1, 0], [2, 1, 0, 0, 0, 0, 0, 0],
                [3, 2, 1, 4, 0, 0, 0, 0], [5, 4, 3, 2, 1, 6, 7, 8]
            ], dtype=np.float32)
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize('use_cpu_only, backend, rank',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(2, 6)]
                             ))
    def test_tf1(self, use_cpu_only, backend, rank):
        shape = np.random.randint(low=1, high=6, size=rank)
        seq_axis = np.random.randint(low=1, high=rank)
        batch_axis = np.random.randint(low=0, high=seq_axis)
        lengths = np.random.randint(
            low=0, high=shape[seq_axis], size=shape[batch_axis])

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            res = tf.reverse_sequence(x, seq_lengths=lengths,
                                      seq_axis=seq_axis, batch_axis=batch_axis)
            run_compare_tf1(graph, {x: np.random.rand(*shape)}, res,
                            use_cpu_only=use_cpu_only, backend=backend)


class TestSliceBySize:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array(list(range(24))).reshape((2, 3, 4)).astype(np.float32)
        begin_val = np.array([1, 1, 1], dtype=np.int32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape), 'begin': cb.placeholder(shape=begin_val.shape, dtype=builtins.int32)}
        input_values = {'x': x_val, 'begin': begin_val}

        def build_non_single(x, begin):
            return [
                cb.slice_by_size(x=x, begin=begin, size=[1, 2, 3]),
            ]

        def build_single(x, begin):
            return [
                cb.slice_by_size(x=x, begin=begin, size=[-1, 2, -1]),
            ]

        expected_output_types = [(1, 2, 3, builtins.fp32)]
        expected_outputs = [np.array([[[17, 18, 19], [21, 22, 23]]], dtype=np.float32)]
        run_compare_builder(build_non_single, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False, backend=backend)

        expected_output_types = [(UNK_SYM, 2, UNK_SYM, builtins.fp32)]
        run_compare_builder(build_single, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False, backend=backend)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize('use_cpu_only, backend, rank, single_size',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 5)],
                                 [True, False]
                                 ))
    def test_tf1(self, use_cpu_only, backend, rank, single_size):
        shape = np.random.randint(low=2, high=6, size=rank)
        begin_val = np.array([np.random.randint(shape[i]) for i in range(rank)]).astype(np.int32)
        size_val = np.array([np.random.randint(shape[i]-begin_val[i])+1 for i in range(rank)])
        if single_size:
            for r in range(rank):
                size_val_r = np.array([s if i == r else -1 for i, s in enumerate(size_val)])
                with tf.Graph().as_default() as graph:
                    x = tf.placeholder(tf.float32, shape=shape)
                    begin = tf.placeholder(tf.int32, shape=begin_val.shape)

                    x_val = random_gen(shape, rand_min=-100, rand_max=100)
                    res = tf.slice(x, begin, size_val_r)
                    run_compare_tf1(graph, {x: x_val, begin: begin_val},
                                    res, use_cpu_only=use_cpu_only, backend=backend)
        else:
            size_val = np.array([s if np.random.randint(2) == 0 else -1 for s in size_val])
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=shape)
                begin = tf.placeholder(tf.int32, shape=begin_val.shape)

                x_val = random_gen(shape, rand_min=-100, rand_max=100)
                res = tf.slice(x, begin, size_val)
                run_compare_tf1(graph, {x: x_val, begin: begin_val},
                                res, use_cpu_only=use_cpu_only, backend=backend)


class TestSpaceToDepth:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        # original input type is (1, 1, 2, 2, fp32)
        val = np.array([[[[7., 9.], [4., 6.]]]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=val.shape)}
        input_values = {'x': val}

        def build(x):
            return [cb.space_to_depth(x=x, block_size=2)]

        expected_output_types = (1, 4, 1, 1, builtins.fp32)
        expected_outputs = np.array([[[[7.]], [[9.]], [[4.]], [[6.]]]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize('use_cpu_only, backend, shape, block_size',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(1, 6, 6, 1), (1, 12, 12, 1), (1, 6, 6, 3)],
                                 [2, 3]
                             ))
    def test_tf1(self, use_cpu_only, backend, shape, block_size):
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.space_to_depth(x, block_size)
            run_compare_tf1(graph, {x: np.random.rand(*shape)}, ref,
                            use_cpu_only=use_cpu_only, backend=backend)


class TestSqueeze:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[[[1], [2], [3]]]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape)}

        input_values = {"x": x}
        def build(x):
            return [
                    cb.squeeze(x=x, axes = (-1,)),
                    cb.squeeze(x=x, axes = (-3,0)),
                    cb.squeeze(x=x, axes = (0,1,3)),
                    cb.squeeze(x=x),
                    ]

        expected_output_types = [
                (1,1,3, builtins.fp32),
                (3,1, builtins.fp32),
                (3, builtins.fp32),
                (3, builtins.fp32)
                ]

        expected_outputs = [
                np.array([[[1, 2, 3]]], dtype=np.float32),
                np.array([[1], [2], [3]], dtype=np.float32),
                np.array([1, 2, 3], dtype=np.float32),
                np.array([1, 2, 3], dtype=np.float32),
                ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)
    @ssa_fn
    def test_builder_eval(self):
        x = np.array([[[[1], [2], [3]],[[4], [5], [6]]]], dtype=np.float32)
        v = cb.squeeze(x=x, axes = (-4,3))
        assert is_close(np.squeeze(x, axis=(-4,3)), v.val)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize("use_cpu_only, backend, rank_and_axes",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(2, (1,)), (2, (0,)), (3, (1,)), (3, (0,-1)), (3, []),
                                  (4, (-1,2,1)), (4, (0,1)), (5, (3,1,2)), (5, (-1,))]
                             )
                             )
    def test_tf1(self, use_cpu_only, backend, rank_and_axes):
        rank, axes = rank_and_axes
        x_shape = np.random.randint(low=2, high=5, size=rank)
        for axis in axes:
            x_shape[axis] = 1

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            res = tf.squeeze(x, axis = axes)
            run_compare_tf1(graph,
                            {x: np.random.rand(*x_shape)},
                            res, use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            res = tf.squeeze(x, axis = None)
            run_compare_tf1(graph,
                            {x: np.random.rand(*x_shape)},
                            res, use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)


class TestTranspose:
    @pytest.mark.parametrize(argnames=["use_cpu_only", "backend", "is_symbolic"],
                             argvalues=itertools.product([True, False],
                                                         backends,
                                                         [True, False],
                                                         )
                            )
    def test_builder_to_backend_smoke(self, use_cpu_only, backend, is_symbolic):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)

        input_shape = x.shape
        # is_symbolic = False
        if is_symbolic:
            input_shape = [get_new_symbol(), get_new_symbol()]

        input_placeholders = {"x": cb.placeholder(shape=input_shape)}

        input_values = {"x": x}
        def build(x):
            return [cb.transpose(x=x, perm = (0,1)),
                    cb.transpose(x=x, perm = (1,0)),
                    cb.transpose(x=x, perm = (-1,0)),
                    cb.transpose(x=x, perm = (-2,-1))
                    ]

        d0 = input_shape[0]
        d1 = input_shape[1]
        expected_output_types = [
                (d0, d1, builtins.fp32),
                (d1, d0, builtins.fp32),
                (d1, d0, builtins.fp32),
                (d0, d1, builtins.fp32),
                ]

        expected_outputs = [
                x,
                x.T,
                x.T,
                x
                ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)


    @ssa_fn
    def test_builder_eval(self):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        v = cb.transpose(x=x, perm = (1,0))
        assert is_close(x.T, v.val)

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_symbolic(self, use_cpu_only, backend):
        s0 = get_new_symbol()

        # Test variadic (rdar://59559656)
        input_placeholders = {
                "x": cb.placeholder(shape=(2, s0)),
                }

        def build(x):
            return [cb.transpose(x=x, perm=[1, 0]),
                    ]

        expected_output_types = [
                (s0, 2, builtins.fp32),
                ]
        expected_outputs = [
                np.array([[1, 4],
                          [2, 5],
                          [3, 6]], dtype=np.float32),
                ]

        input_values = {
                "x": np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32),
                }
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize("use_cpu_only, backend, rank_and_perm",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(1,(0,)), (2,(1,0)), (2,(0,1)), (3,(0,2,1)), (3, (2,1,0)), (3,(2,0,1)),
                                  (4, (0,3,2,1)), (4,(3,0,1,2)), (5,(2,3,1,0,4)), (5, (3,1,0,4,2))]
                             )
                             )
    def test_tf1(self, use_cpu_only, backend, rank_and_perm):
        rank, perm = rank_and_perm
        x_shape = np.random.randint(low=2, high=5, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            res = tf.transpose(x, perm = perm)
            run_compare_tf1(graph,
                            {x: np.random.rand(*x_shape)},
                            res, use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [1, 2, 3, 4],
                             )
                             )
    def test_tf2(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        perm = np.random.permutation(rank).astype(np.float32)
        def static_perm():
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=input_shape)
                res = tf.transpose(x, perm)
                run_compare_tf1(graph, {x: np.random.rand(*input_shape)},
                                res, use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)
        def dynamic_perm():
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=input_shape)
                tf_perm = tf.placeholder(tf.int32, shape=[None])
                res = tf.transpose(x, tf_perm)
                run_compare_tf1(graph, {x: np.random.rand(*input_shape),
                                        tf_perm: perm}, res, use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)

        static_perm()
        # Note that TF supports dynamic perm in tf.transpose.
        with pytest.raises(ValueError, \
                match=r'.*must be const at compile time.*'):
            dynamic_perm()


class TestPixelShuffle:

    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        # original input type is (1, 4, 1, 1, fp32)
        val = np.array([[[[9.]], [[5.]], [[1.]], [[3.]]]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=val.shape)}
        input_values = {'x': val}

        def build(x):
            return [cb.pixel_shuffle(x=x, upscale_factor=2)]

        expected_output_types = (1, 1, 2, 2, builtins.fp32)
        expected_outputs = np.array([[[[9., 5.], [1., 3.]]]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_PYTORCH, reason='PyTorch not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, shape, upscale_factor',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(1, 16, 1, 1), (2, 16, 3, 3), (1, 32, 1, 1)],
                                 [2, 4],
                             ))
    def test_builder_to_backend_stress(self, use_cpu_only, backend, shape, upscale_factor):
        val = np.random.rand(*shape)
        input_placeholders = {'x': cb.placeholder(shape=val.shape)}
        input_values = {'x': val}

        def build(x):
            return [cb.pixel_shuffle(x=x, upscale_factor=upscale_factor)]

        torch_pixel_shuffle = torch.nn.PixelShuffle(upscale_factor)
        expected_outputs = [torch_pixel_shuffle(torch.Tensor(val)).numpy()]
        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)


class TestSlidingWindows:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        # original input type is (1, 4, 1, 1, fp32)
        val = np.array([[[[9.]], [[5.]], [[1.]], [[3.]]]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=val.shape)}
        input_values = {'x': val}

        def build(x):
            return [cb.sliding_windows(x=x, axis=1, size=2)]

        expected_output_types = (1, 3, 2, 1, 1, builtins.fp32)
        expected_outputs = np.array(
            [[[[[9.]], [[5.]]], [[[5.]], [[1.]]],
              [[[1.]], [[3.]]]]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.parametrize('use_cpu_only, backend, rank_and_axis, size, stride',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(rank, axis) for rank in range(1, 5) for axis in range(-rank, rank)],
                                 [1, 2],
                                 [1, 2],
                             ))
    def test_builder_to_backend_stress(self, use_cpu_only, backend, rank_and_axis, size, stride):
        def np_sliding_windows(a, np_axis, np_size, np_stride):
            n = (a.shape[np_axis] - np_size) // np_stride + 1
            x_shape = list(a.shape)
            x_shape[np_axis] = n
            if np_axis < 0:
                np_axis += len(x_shape)
            x_shape.insert(np_axis + 1, np_size)
            strides = list(a.strides)
            eff_stride = strides[np_axis] * np_stride
            strides.insert(np_axis, eff_stride)
            return np.lib.stride_tricks.as_strided(a, x_shape, strides)

        rank, axis = rank_and_axis
        shape = np.random.randint(low=2, high=5, size=rank)
        val = np.random.rand(*shape)
        input_placeholders = {'x': cb.placeholder(shape=val.shape)}
        input_values = {'x': val}

        def build(x):
            return [cb.sliding_windows(x=x, axis=axis, size=size, stride=stride)]

        expected_outputs = [np_sliding_windows(val, np_axis=axis, np_size=size, np_stride=stride)]
        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)
