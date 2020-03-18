from . import _test_reqs
from ._test_reqs import *
backends = _test_reqs.backends

class TestScatter:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        data = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        indices = np.array([1,0], dtype=np.int32)
        updates = np.array([[5,6,7], [8,9,10]], dtype=np.float32)
        input_placeholders = {"data": cb.placeholder(shape=data.shape),
                              "indices": cb.placeholder(shape=indices.shape, dtype=builtins.int32),
                              "updates": cb.placeholder(shape=updates.shape)}

        input_values = {"data": data, "indices": indices, "updates": updates}
        def build(data, indices, updates):
            return cb.scatter(data=data, indices=indices, updates = updates),


        expected_output_types = (2, 3, builtins.fp32)

        expected_outputs = np.array([[9, 11, 13],
                                     [9, 11, 13]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rankData_rankIndices_axis, accumulate_mode",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(1,2,-1), (2,1,0), (3,2,-2), (2,3,1), (2,2,1), (1,1,0),
                                  (3,3,-2), (3,3,2), (3,3,0), (1,3,-1), (3,1,2), (3,1,-1)],
                                 ["update", "add", "sub", "mul", "div", "max", "min"]
                             )
                             )
    def test_builder_to_backend_programmatic(self, use_cpu_only, backend,
                                             rankData_rankIndices_axis, accumulate_mode):
        data_rank, indices_rank, axis = rankData_rankIndices_axis
        data_shape = np.random.randint(low=2, high=5, size=data_rank)
        indices_shape = np.random.randint(low=2, high=5, size=indices_rank)
        updates_shape = list(indices_shape) + list(data_shape[1:])

        data = np.random.rand(*data_shape).astype(np.float32)
        updates = np.random.rand(*updates_shape).astype(np.float32)
        indices = np.random.randint(0, data_shape[0], size=indices_shape).astype(np.int32)

        def build(data, indices, updates):
            return cb.scatter(data=data, indices=indices, updates=updates, mode=accumulate_mode)

        with tf.Graph().as_default(), tf.Session() as sess:
            tf_output = tf.Variable(data)
            sess.run(tf.global_variables_initializer())
            if accumulate_mode == "update":
                sess.run(tf.scatter_update(tf_output, indices, updates))
            if accumulate_mode == "add":
                sess.run(tf.scatter_add(tf_output, indices, updates))
            if accumulate_mode == "sub":
                sess.run(tf.scatter_sub(tf_output, indices, updates))
            if accumulate_mode == "mul":
                sess.run(tf.scatter_mul(tf_output, indices, updates))
            if accumulate_mode == "div":
                sess.run(tf.scatter_div(tf_output, indices, updates))
            if accumulate_mode == "max":
                sess.run(tf.scatter_max(tf_output, indices, updates))
            if accumulate_mode == "min":
                sess.run(tf.scatter_min(tf_output, indices, updates))
            expected_output = sess.run(tf_output)

        input_placeholders = {"data": cb.placeholder(shape=data.shape),
                              "indices": cb.placeholder(shape=indices.shape, dtype=builtins.int32),
                              "updates": cb.placeholder(shape=updates.shape)}

        input_values = {"data": data, "indices": indices, "updates": updates}

        expected_output_types = tuple(data_shape[:]) + (builtins.fp32,)
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types,
                            expected_output,
                            use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)


class TestScatterAlongAxis:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        data = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        indices = np.array([[1,0,1],[1,1,0]], dtype=np.int32)
        updates = np.array([[5,6,7], [8,9,10]], dtype=np.float32)
        input_placeholders = {"data": cb.placeholder(shape=data.shape),
                              "indices": cb.placeholder(shape=indices.shape, dtype=builtins.int32),
                              "updates": cb.placeholder(shape=updates.shape)}

        input_values = {"data": data, "indices": indices, "updates": updates}
        def build(data, indices, updates):
            return cb.scatter_along_axis(data=data, indices=indices, updates=updates, axis=0, mode="update")


        expected_output_types = (2, 3, builtins.fp32)


        expected_outputs = np.array([[1, 6, 10],
                                     [8, 9, 7]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)


    @ssa_fn
    def test_builder_eval(self):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        indices = np.array([[1,0,1], [1, 1, 0]], dtype=np.int32)
        updates = np.array([[5,6,7], [8,9,10]], dtype=np.float32)
        v = cb.scatter_along_axis(data=x, indices=indices, updates=updates, axis=0, mode="update")
        assert is_close(np.array([[1, 6, 10],[8, 9, 7]], dtype=np.float32), v.val)


    @pytest.mark.parametrize("use_cpu_only, backend, rank_axis",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(rank, axis) for rank in range(1, 5) for
                                  axis in range(-rank, rank)]
                             )
                             )
    def test_builder_to_backend_programmatic(self, use_cpu_only, backend, rank_axis):
        rank, axis = rank_axis
        data_shape = np.random.randint(low=2, high=8, size=rank)
        indices_shape = np.copy(data_shape)
        indices_shape[axis] = np.random.randint(low=1, high=8)
        updates_shape = indices_shape

        data = np.random.rand(*data_shape).astype(np.float32)
        updates = np.random.rand(*updates_shape).astype(np.float32)
        indices = np.random.randint(-data_shape[axis], data_shape[axis], size=indices_shape).astype(np.int32)

        def build(data, indices, updates):
            return cb.scatter_along_axis(data=data, indices=indices, updates=updates, axis=axis, mode="update")

        input_placeholders = {"data": cb.placeholder(shape=data.shape),
                              "indices": cb.placeholder(shape=indices.shape, dtype=builtins.int32),
                              "updates": cb.placeholder(shape=updates.shape)}

        input_values = {'data': data, 'indices': indices, 'updates': updates}

        expected_output_types = tuple(data_shape[:]) + (builtins.fp32,)

        np_output = np.copy(data)
        np.put_along_axis(np_output, indices, updates, axis=axis)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types,
                            np_output,
                            use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)


class TestScatterNd:
    # TODO: <rdar://problem/59737282> [NNv2] Scatter and ScatterNd in tensoflow
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        data = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        indices = np.array([[1,0], [0,2]], dtype=np.int32)
        updates = np.array([5, 10], dtype=np.float32)
        input_placeholders = {"data": cb.placeholder(shape=data.shape),
                              "indices": cb.placeholder(shape=indices.shape, dtype=builtins.int32),
                              "updates": cb.placeholder(shape=updates.shape)}

        input_values = {"data": data, "indices": indices, "updates": updates}
        def build(data, indices, updates):
            return cb.scatter_nd(data=data, indices=indices, updates = updates),


        expected_output_types = (2, 3, builtins.fp32)

        expected_outputs = np.array([[1, 2, 13],
                                     [9, 5, 6]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rankData_rankIndices, accumulate_mode",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(1,2), (2,2), (3,2), (2,3), (1,4), (5,2),
                                  (2,5), (4,3), (3,4), (2,4), (4,2), (1,5)],
                                 ["update", "add", "sub"]
                             )
                             )
    def test_builder_to_backend_programmatic(self, use_cpu_only, backend,
                                             rankData_rankIndices, accumulate_mode):
        data_rank, indices_rank = rankData_rankIndices
        data_shape = np.random.randint(low=2, high=5, size=data_rank)
        indices_shape = np.random.randint(low=2, high=5, size=indices_rank)
        indices_shape[-1] = np.random.randint(low=1, high=data_rank + 1)
        updates_shape = list(indices_shape[:-1]) + list(data_shape[indices_shape[-1]:])

        data = np.random.rand(*data_shape).astype(np.float32)
        updates = np.random.rand(*updates_shape).astype(np.float32)
        indices_list = []
        for i in range(indices_shape[-1]):
            indices_list.append(np.random.randint(0, data_shape[i], size=indices_shape[:-1]))

        indices = np.stack(indices_list, axis=-1).astype(np.int32)

        def build(data, indices, updates):
            return cb.scatter_nd(data=data, indices=indices, updates=updates, mode=accumulate_mode)

        with tf.Graph().as_default(), tf.Session() as sess:
            tf_output = tf.Variable(data)
            sess.run(tf.global_variables_initializer())
            if accumulate_mode == "update":
                sess.run(tf.scatter_nd_update(tf_output, indices, updates))
            if accumulate_mode == "add":
                sess.run(tf.scatter_nd_add(tf_output, indices, updates))
            if accumulate_mode == "sub":
                sess.run(tf.scatter_nd_sub(tf_output, indices, updates))
            expected_output = sess.run(tf_output)

        input_placeholders = {"data": cb.placeholder(shape=data.shape),
                              "indices": cb.placeholder(shape=indices.shape, dtype=builtins.int32),
                              "updates": cb.placeholder(shape=updates.shape)}

        input_values = {"data": data, "indices": indices, "updates": updates}

        expected_output_types = tuple(data_shape[:]) + (builtins.fp32,)
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types,
                            expected_output,
                            use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)


class TestGather:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        indices = np.array([1,0], dtype=np.int32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape),
                              "indices": cb.placeholder(shape=indices.shape, dtype=builtins.int32)}

        input_values = {"x": x, "indices": indices}
        def build(x, indices):
            return [cb.gather(x=x, indices=indices, axis=0),
                    cb.gather(x=x, indices=indices, axis=1),
                    cb.gather(x=x, indices=indices, axis=-2),
                    cb.gather(x=x, indices=indices, axis=-1),
                    cb.gather(x=x, indices=indices)
                    ]

        expected_output_types = [
                (2, 3, builtins.fp32),
                (2, 2, builtins.fp32),
                (2, 3, builtins.fp32),
                (2, 2, builtins.fp32),
                (2, 3, builtins.fp32),
                ]

        expected_outputs = [
                np.array([[4, 5, 6],
                          [1, 2, 3]], dtype=np.float32),
                np.array([[2, 1],
                          [5, 4]], dtype=np.float32),
                np.array([[4, 5, 6],
                          [1, 2, 3]], dtype=np.float32),
                np.array([[2, 1],
                          [5, 4]], dtype=np.float32),
                np.array([[4, 5, 6],
                        [1, 2, 3]], dtype=np.float32),
                ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        indices = np.array([1,0], dtype=np.int32)
        v = cb.gather(x=x, indices = indices, axis=-1)
        assert is_close(np.array([[2, 1],[5, 4]], dtype=np.float32), v.val)


    # TODO: <rdar://problem/59738824> [NNv2] Gather layer with 0-d indices leads to input shape mismatch
    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rankX_rankIndices_axis",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(1,2,-1), (2,1,0), (3,2,-2), (2,3,1), (2,2,1), (1,1,0),
                                  (3,3,-2), (3,3,2), (3,3,0), (1,3,-1), (3,1,2), (3,1,-1)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rankX_rankIndices_axis):
        x_rank, indices_rank, axis = rankX_rankIndices_axis
        x_shape = np.random.randint(low=2, high=5, size=x_rank)
        indices_shape = np.random.randint(low=2, high=5, size=indices_rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            indices = tf.placeholder(tf.int32, shape=indices_shape)
            res = tf.gather(x, indices, axis=axis)
            run_compare_tf(graph,
                           {x: np.random.rand(*x_shape),
                            indices:  np.random.randint(0, x_shape[axis], size=indices_shape,dtype=np.int32)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestGatherAlongAxis:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        indices = np.array([[1,0,1],[1,1,0]], dtype=np.int32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape),
                              "indices": cb.placeholder(shape=indices.shape, dtype=builtins.int32)}

        input_values = {"x": x, "indices": indices}
        def build(x, indices):
            return [cb.gather_along_axis(x=x, indices=indices, axis=0),
                    cb.gather_along_axis(x=x, indices=indices, axis=1),
                    cb.gather_along_axis(x=x, indices=indices, axis=-2),
                    cb.gather_along_axis(x=x, indices=indices, axis=-1),
                    cb.gather_along_axis(x=x, indices=indices)
                    ]

        expected_output_types = [
                (2, 3, builtins.fp32),
                (2, 3, builtins.fp32),
                (2, 3, builtins.fp32),
                (2, 3, builtins.fp32),
                (2, 3, builtins.fp32),
                ]

        expected_outputs = [
                np.array([[4, 2, 6],
                          [4, 5, 3]], dtype=np.float32),
                np.array([[2, 1, 2],
                          [5, 5, 4]], dtype=np.float32),
                np.array([[4, 2, 6],
                          [4, 5, 3]], dtype=np.float32),
                np.array([[2, 1, 2],
                          [5, 5, 4]], dtype=np.float32),
                np.array([[4, 2, 6],
                          [4, 5, 3]], dtype=np.float32),
                ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)


    @ssa_fn
    def test_builder_eval(self):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        indices = np.array([[1,0,1], [0, 0, 1]], dtype=np.int32)
        v = cb.gather_along_axis(x=x, indices = indices, axis=0)
        assert is_close(np.array([[4, 2, 6],[1, 2, 6]], dtype=np.float32), v.val)


    @pytest.mark.parametrize("use_cpu_only, backend, rank_axis",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(rank, axis) for rank in range(1, 5) for
                                  axis in range(-rank, rank)]
                             )
                             )
    def test_builder_to_backend_programmatic(self, use_cpu_only, backend, rank_axis):
        rank, axis = rank_axis
        x_shape = np.random.randint(low=2, high=8, size=rank)
        indices_shape = np.copy(x_shape)
        indices_shape[axis] = np.random.randint(low=1, high=8)

        x = np.random.rand(*x_shape).astype(np.float32)
        indices = np.random.randint(-x_shape[axis], x_shape[axis], size=indices_shape).astype(np.int32)

        def build(x, indices):
            return cb.gather_along_axis(x=x, indices=indices, axis=axis)

        input_placeholders = {"x": cb.placeholder(shape=x.shape),
                              "indices": cb.placeholder(shape=indices.shape, dtype=builtins.int32)}

        input_values = {"x": x, "indices": indices}

        expected_output_types = tuple(indices_shape[:]) + (builtins.fp32,)
        expected_output = np.take_along_axis(x, indices, axis=axis)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types,
                            expected_output,
                            use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)


class TestGatherNd:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        indices = np.array([[1,0], [0,2]], dtype=np.int32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape),
                              "indices": cb.placeholder(shape=indices.shape, dtype=builtins.int32)}

        input_values = {"x": x, "indices": indices}
        def build(x, indices):
            return cb.gather_nd(x=x, indices=indices),

        expected_output_types = (2, builtins.fp32)
        expected_outputs = np.array([4, 3], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rankX_rankIndices",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(1,2), (2,2), (3,2), (2,3), (1,4), (5,2),
                                  (2,5), (4,3), (3,4), (2,4), (4,2), (1,5)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rankX_rankIndices):
        x_rank, indices_rank = rankX_rankIndices
        x_shape = np.random.randint(low=2, high=8, size=x_rank)
        indices_shape = np.random.randint(low=2, high=8, size=indices_rank)
        indices_shape[-1] = np.random.randint(low=1, high=x_rank + 1)

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            indices = tf.placeholder(tf.int32, shape=indices_shape)
            res = tf.gather_nd(x, indices)

            a = np.random.rand(*x_shape)
            indices_list = []
            for i in range(indices_shape[-1]):
                indices_list.append(np.random.randint(0, x_shape[i], size=indices_shape[:-1]))

            input_values = {x: a, indices: np.stack(indices_list, axis=-1).astype(np.float)}

            run_compare_tf(graph, input_values,
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)
