from coremltools.converters.nnv2.nnv2_program.program import get_new_symbol
from ._test_utils import UNK_SYM, UNK_VARIADIC
from . import _test_reqs
from ._test_reqs import *
backends = _test_reqs.backends

class TestBandPart:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array(
            [[3., 3., 5., 1.],
             [5., 6., 3., 8.],
             [7., 2., 7., 2.],
             [6., 7., 7., 1.]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.band_part(x=x),
                cb.band_part(x=x, lower=0, upper=-1),
                cb.band_part(x=x, lower=-1, upper=0),
                cb.band_part(x=x, lower=0, upper=0)
            ]

        expected_output_types = [
            (4, 4, builtins.fp32), (4, 4, builtins.fp32),
            (4, 4, builtins.fp32), (4, 4, builtins.fp32),
        ]

        expected_outputs = [
            np.array(
                [[3., 3., 5., 1.],
                 [5., 6., 3., 8.],
                 [7., 2., 7., 2.],
                 [6., 7., 7., 1.]], dtype=np.float32),
            np.array(
                [[3., 3., 5., 1.],
                 [0., 6., 3., 8.],
                 [0., 0., 7., 2.],
                 [0., 0., 0., 1.]], dtype=np.float32),
            np.array(
                [[3., 0., 0., 0.],
                 [5., 6., 0., 0.],
                 [7., 2., 7., 0.],
                 [6., 7., 7., 1.]], dtype=np.float32),
            np.array(
                [[3., 0., 0., 0.],
                 [0., 6., 0., 0.],
                 [0., 0., 7., 0.],
                 [0., 0., 0., 1.]], dtype=np.float32),
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, rank, lower_and_upper',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(2, 6)],
                                 [(0, -1), (-1, 0), (0, 0)]))
    def test_tf(self, use_cpu_only, backend, rank, lower_and_upper):
        lower, upper = lower_and_upper
        shape = np.random.randint(low=3, high=4, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            res = tf.matrix_band_part(x, num_lower=lower, num_upper=upper)
            run_compare_tf(graph, {x: random_gen(shape, rand_min=-100, rand_max=100)},
                           res, use_cpu_only=use_cpu_only, backend=backend)


class TestCumsum():
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        if backend == 'nnv1_proto':
            pytest.skip("CumSum not implemented for NNv1 Backend")

        #Need to be removed
        frontend_only = True

        t = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.cumsum(x=x, axis=0, reverse=True, exclusive=False)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[5,7,9],[4,5,6]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only,
                            frontend_only=frontend_only, backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        v = cb.cumsum(x=x_val)
        assert is_close(np.cumsum(x_val, axis=0), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank, reverse, exclusive",
                             itertools.product(
                                [True, False],
                                backends,
                                [rank for rank in range(1,6)],
                                [True, False],
                                [True, False]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank, reverse, exclusive):
        if backend == 'nnv1_proto':
            pytest.skip("CumSum not implemented for NNv1 Backend")

        #Need to be removed
        frontend_only = True

        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            for axis in range(-1, rank):
                res = tf.math.cumsum(x, axis=axis, reverse=reverse, exclusive=exclusive)
                run_compare_tf(graph, {x: random_gen(input_shape, rand_min=-100, rand_max=100)},
                               res, use_cpu_only=use_cpu_only,
                               frontend_only=frontend_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_empty_input_tf(self, use_cpu_only, backend):
        if backend == 'nnv1_proto':
            pytest.skip("CumSum not implemented for NNv1 Backend")

        #Corner cases
        #Need to be removed
        frontend_only = True
        empty_inputs = [[], [[]], [[[]]], [[],[]], [[[]],[[]]]]
        for input_x in empty_inputs:
            input_x = np.array(input_x)
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=input_x.shape)
                for axis in range(-1, len(input_x.shape)):
                    res = tf.math.cumsum(x, axis=axis)
                    run_compare_tf(graph, {x: input_x},
                                   res, use_cpu_only=use_cpu_only,
                                   frontend_only=frontend_only,
                                   backend=backend)

    @ssa_fn
    def test_invalid_axis1(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(ValueError):
            cb.cumsum(x=x_val, axis=-2)

    @ssa_fn
    def test_invalid_axis2(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(ValueError):
            cb.cumsum(x=x_val, axis=len(x_val.shape))

    @ssa_fn
    def test_invalid_axis3(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(TypeError):
            cb.cumsum(x=x_val, axis='')

    @ssa_fn
    def test_invalid_reverse1(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(TypeError):
            cb.cumsum(x=x_val, reverse='')

    @ssa_fn
    def test_invalid_reverse2(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(TypeError):
            pred = cb.cumsum(x=x_val, reverse=0)

    @ssa_fn
    def test_invalid_reverse3(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(TypeError):
            pred = cb.cumsum(x=x_val, reverse=1)

    @ssa_fn
    def test_invalid_exclusive1(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(TypeError):
            pred = cb.cumsum(x=x_val, exclusive='')

    @ssa_fn
    def test_invalid_exclusive2(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(TypeError):
            pred = cb.cumsum(x=x_val, exclusive=0)

    @ssa_fn
    def test_invalid_exclusive3(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(TypeError):
            pred = cb.cumsum(x=x_val, exclusive=1)

    @ssa_fn
    def test_invalid_input1(self):
        x_val = 1
        with pytest.raises(TypeError):
            pred = cb.cumsum(x=x_val)

    @ssa_fn
    def test_invalid_input2(self):
        x_val = ['1']
        with pytest.raises(TypeError):
            pred = cb.cumsum(x=x_val)

class TestFill:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        shape = (2, 1, 3)
        x_val = np.zeros(shape=shape, dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}

        input_values = {'x': x_val}

        def build(x):
            return cb.add(x=x, y=cb.fill(shape=shape, value=1.))

        expected_output_types = [(2, 1, 3, builtins.fp32)]
        expected_outputs = [np.full(shape=shape, fill_value=1.)]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        shape = np.random.randint(low=1, high=3, size=5).astype(np.int32)
        res = cb.fill(shape=shape, value=1991.).val
        assert is_close(np.full(shape, fill_value=1991.), res)

    @pytest.mark.parametrize('use_cpu_only, backend, rank, value',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 6)],
                                 [-1917., 0., 2048.]
                             ))
    def test_builder_to_backend_stress(self, use_cpu_only, backend, rank, value):
        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)
        x_val = np.zeros(shape=shape, dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return cb.add(x=x, y=cb.fill(shape=shape, value=value))

        expected_outputs = [np.full(shape=shape, fill_value=value)]
        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_symbolic(self, use_cpu_only, backend):
        # Test variadic (rdar://59559656)

        s_len = get_new_symbol()
        input_placeholders = {
            'shape': cb.placeholder(shape=(s_len,), dtype=builtins.int32),
        }

        def build(shape):
            return [cb.fill(shape=shape)]

        expected_output_types = [(UNK_VARIADIC, builtins.fp32)]
        expected_outputs = [np.zeros(shape=(2, 1, 3), dtype=np.float32)]
        input_values = {'shape': np.array([2, 1, 3], dtype=np.float32)}

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, rank, value',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 6)],
                                 [-19., 0., 37.]))
    def test_tf(self, use_cpu_only, backend, rank, value):

        def test_tf_static():
            shape = np.random.randint(low=1, high=3, size=rank)
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=shape)
                ref = tf.add(x, tf.fill(dims=np.array(shape, dtype=np.float32), value=value))
                run_compare_tf(graph, {x: np.random.rand(*shape)},
                               ref, use_cpu_only=use_cpu_only, backend=backend)

        def test_tf_dynamic():
            shape = np.random.randint(low=1, high=3, size=rank)
            with tf.Graph().as_default() as graph:
                s = tf.placeholder(tf.int32, shape=(len(shape),))
                ref = tf.fill(dims=s, value=value)
                run_compare_tf(graph, {s: np.array(shape, dtype=np.float32)},
                               ref, use_cpu_only=use_cpu_only, backend=backend)

        test_tf_static()
        test_tf_dynamic()


class TestNonMaximumSuppression:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        boxes_val = np.array([
            [[0., 0., 0., 0.], [1., 1., 1., 1.],
             [2., 2., 2., 2.], [3., 3., 3., 3.]]], dtype=np.float32)
        scores_val = np.array(
            [[[-3.5], [9.4], [2.3], [0.7]]], dtype=np.float32)
        input_placeholders = {
            'boxes': cb.placeholder(shape=(1, 4, 4)),
            'scores': cb.placeholder(shape=(1, 4, 1)),
        }
        input_values = {'boxes': boxes_val, 'scores': scores_val}

        expected_output_types = [
            (1, 2, 4, builtins.fp32), (1, 2, 1, builtins.fp32),
            (1, 2, builtins.int32), (1, builtins.int32),
        ]
        expected_outputs = [
            np.array([[[1., 1., 1., 1.],
                       [2., 2., 2., 2.]]], dtype=np.float32),
            np.array([[[9.4], [2.3]]], dtype=np.float32),
            np.array([[1, 2]], dtype=np.int32),
            np.array([2], dtype=np.int32)
        ]

        def build(boxes, scores):
            return cb.non_maximum_suppression(
                boxes=boxes, scores=scores, iou_threshold=0.2,
                score_threshold=0.4, max_boxes=2, per_class_suppression=True)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @staticmethod
    def _compute_iou_matrix(boxes):
        # input is (N, 4), in order [center_w, center_h, width, height]
        boxes = boxes.astype(np.float)
        center_w, center_h, width, height = np.split(boxes, 4, axis=1)
        top = center_h + 0.5 * height
        bottom = center_h - 0.5 * height
        left = center_w - 0.5 * width
        right = center_w + 0.5 * width
        area = width * height

        h_b = np.minimum(top, np.transpose(top))
        w_b = np.minimum(right, np.transpose(right))
        h_a = np.maximum(bottom, np.transpose(bottom))
        w_a = np.maximum(left, np.transpose(left))

        intersection_area = np.maximum(0, h_b - h_a) * np.maximum(0, w_b - w_a)
        union_area = area + np.transpose(area) - intersection_area
        return intersection_area / union_area

    @staticmethod
    def _ref_non_maximum_suppression(
            boxes, scores, iou_threshold, score_threshold,
            max_boxes, per_class_suppression):
        """
        Reference implementation of Core ML's NMS op using TensorFlow.
        boxes of shape (n_batch, n_box, 4), [center_w, center_h, width, height]
        scores of shape (n_batch, n_box, n_score)
        output shapes [
           (n_batch, max_boxes, 4),
           (n_batch, max_boxes, n_score),
           (n_batch, max_boxes),
           (n_batch,)
        ]
        """
        n_batch, n_box, n_score = scores.shape

        iou_threshold = iou_threshold.astype(np.float32)
        score_threshold = score_threshold.astype(np.float32)

        # convert box ids to TF style
        center_w, center_h, width, height = np.split(boxes, 4, axis=-1)  # (n_batch,n_box,1)
        y1 = center_h - 0.5 * height
        y2 = center_h + 0.5 * height
        x1 = center_w - 0.5 * width
        x2 = center_w + 0.5 * width
        boxes_tf = np.concatenate((y1, x1, y2, x2), axis=-1)  # (n_batch,n_box,4)

        out1 = np.zeros((n_batch, max_boxes, 4))
        out2 = np.zeros((n_batch, max_boxes, n_score))
        out3 = -1 * np.ones((n_batch, max_boxes))
        out4 = np.zeros((n_batch,))

        for b in range(n_batch):
            box_coord_matrix = boxes_tf[b, :, :]  # (n_box,4)
            score_vector = np.max(scores[b, :, :], axis=-1)  # (n_box,)
            if not per_class_suppression:
                # this is the simple case as TF directly supports it
                with tf.Graph().as_default(), tf.Session() as sess:
                    box_coord_matrix_pl = tf.placeholder(
                        tf.float32, shape=box_coord_matrix.shape)
                    score_vector_pl = tf.placeholder(
                        tf.float32, shape=score_vector.shape)
                    ids_g = tf.image.non_max_suppression(
                        box_coord_matrix_pl,
                        score_vector_pl,
                        max_output_size=max_boxes,
                        iou_threshold=iou_threshold,
                        score_threshold=score_threshold)
                    ids = sess.run(ids_g, feed_dict={
                        box_coord_matrix_pl: box_coord_matrix,
                        score_vector_pl: score_vector})
            else:
                # this is slightly complicated as TF does not directly support it
                class_ids = np.argmax(scores[b, :, :], axis=-1)  # (n_box,)
                sorted_score_ids = np.argsort(-score_vector)
                box_coord_matrix2 = np.take(box_coord_matrix, sorted_score_ids, axis=0)
                score_vector2 = np.take(score_vector, sorted_score_ids)
                class_ids = np.take(class_ids, sorted_score_ids)
                classes_seen = dict()
                ids_intermediate = np.array([], dtype=np.int)
                for n in range(n_box):
                    if class_ids[n] in classes_seen:
                        continue
                    c = class_ids[n]
                    classes_seen[c] = True
                    current_class_ids = np.where(class_ids == c)[0]
                    if len(current_class_ids) > 0:
                        feed_in1 = np.take(box_coord_matrix2, current_class_ids, axis=0)
                        feed_in2 = np.take(score_vector2, current_class_ids)

                        with tf.Graph().as_default(), tf.Session() as sess:
                            box_coord_matrix_pl = tf.placeholder(tf.float32, shape=feed_in1.shape)
                            score_vector_pl = tf.placeholder(tf.float32, shape=feed_in2.shape)
                            cur_ids_g = tf.image.non_max_suppression(
                                box_coord_matrix_pl,
                                score_vector_pl,
                                max_output_size=max_boxes,
                                iou_threshold=iou_threshold,
                                score_threshold=score_threshold)
                            cur_ids = sess.run(cur_ids_g, feed_dict={
                                box_coord_matrix_pl: feed_in1,
                                score_vector_pl: feed_in2})

                        from_sort_ids = np.take(current_class_ids, cur_ids)
                        ids_intermediate = np.append(ids_intermediate, from_sort_ids)
                ids_intermediate.sort()
                ids = np.take(sorted_score_ids, ids_intermediate)

            xx = len(ids)
            if xx == 0:
                ids = np.array([np.argmax(score_vector)])
                xx = 1
            if xx > max_boxes:
                ids = ids[:max_boxes]
                xx = len(ids)
            out1[b, :xx, :] = np.take(boxes[b, :, :], ids, axis=0)
            out2[b, :xx, :] = np.take(scores[b, :, :], ids, axis=0)
            out3[b, :xx] = ids
            out4[b] = xx

        return out1, out2, out3, out4

    @pytest.mark.xfail(reason='rdar://60390856', run=False)
    @pytest.mark.parametrize(','.join([
        'use_cpu_only',
        'backend',
        'iou_threshold_percentile',
        'score_threshold_percentile',
        'n_boxes',
        'n_batch',
        'n_score',
        'per_class_suppression',
    ]), itertools.product(
        [True, False],
        backends,
        [0, 30, 80, 100],
        [0, 40, 100],
        [(10, 7), (30, 37), (100, 64)],
        [1],
        [1, 4, 7],
        [True, False],
    ))
    def test_builder_to_backend_stress(
            self, use_cpu_only, backend, iou_threshold_percentile,
            score_threshold_percentile, n_boxes, n_batch, n_score,
            per_class_suppression):
        n_boxes_in, n_boxes_out = n_boxes
        boxes_val = random_gen((n_batch, n_boxes_in, 4), 0, 100)
        scores_val = random_gen((n_batch, n_boxes_in, n_score), -100, 100)

        iou_matrix = self._compute_iou_matrix(boxes_val[0, :, :])
        iou_matrix = iou_matrix[~np.eye(
            iou_matrix.shape[0], dtype=bool)].reshape(iou_matrix.shape[0], -1)

        if score_threshold_percentile == 0:
            score_threshold = np.min(scores_val) - 1
        elif score_threshold_percentile == 100:
            score_threshold = np.max(scores_val) + 1
        else:
            score_threshold = np.percentile(scores_val, score_threshold_percentile) + .01

        if iou_threshold_percentile == 0:
            iou_threshold = np.maximum(np.min(iou_matrix) - .01, 0.0)
        else:
            iou_threshold = np.percentile(iou_matrix, iou_threshold_percentile) + .01

        tf_boxes, tf_scores, tf_indices, tf_num_boxes = self._ref_non_maximum_suppression(
            boxes_val, scores_val, iou_threshold, score_threshold,
            n_boxes_out, per_class_suppression)
        expected_outputs = [tf_boxes, tf_scores, tf_indices, tf_num_boxes]
        expected_output_types = [
            tf_boxes.shape[:] + (builtins.fp32,),
            tf_scores.shape[:] + (builtins.fp32,),
            tf_indices.shape[:] + (builtins.int32,),
            tf_num_boxes.shape[:] + (builtins.int32,),
        ]

        input_placeholders = {
            'boxes': cb.placeholder(shape=(n_batch, n_boxes_in, 4)),
            'scores': cb.placeholder(shape=(n_batch, n_boxes_in, n_score)),
        }
        input_values = {'boxes': boxes_val, 'scores': scores_val}

        def build(boxes, scores):
            return cb.non_maximum_suppression(
                boxes=boxes, scores=scores, iou_threshold=iou_threshold,
                score_threshold=score_threshold, max_boxes=n_boxes_out,
                per_class_suppression=per_class_suppression)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    # TODO (rdar://60390856) TF may output fewer than max_boxes, but
    # current TF frontend will always output max_boxes. Need to apply
    # slice.
    @pytest.mark.xfail
    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize(
        ','.join([
            'use_cpu_only',
            'backend',
            'num_boxes',
            'max_boxes',
            'iou_threshold',
            'score_threshold']),
        itertools.product(
            [True, False],
            backends,
            [20, 30, 100],
            [5, 20],
            [1.0, 0.99],
            [float('-inf'), -200.],
        ))
    def test_tf(self, use_cpu_only, backend, num_boxes,
                max_boxes, iou_threshold, score_threshold):
        """
        Note: TensorFlow and Core ML does not have exact same implementation,
        Core ML pad -1s to the results while TensorFlow does not. Threshold
        values are carefully choose to make test success as it fails when:
        1) input num_boxes < max_boxes because of shape mis-match.
        2) output num_boxes < max_boxes because of shape mis-match.
        """
        boxes_val = random_gen(shape=(num_boxes, 4), rand_min=0, rand_max=32)
        scores_val = random_gen(shape=(num_boxes,), rand_min=-100, rand_max=100)

        with tf.Graph().as_default() as graph:
            boxes = tf.placeholder(tf.float32, shape=boxes_val.shape)
            scores = tf.placeholder(tf.float32, shape=scores_val.shape)
            ref = tf.image.non_max_suppression(
                boxes=boxes, scores=scores, max_output_size=max_boxes,
                iou_threshold=iou_threshold, score_threshold=score_threshold)
            run_compare_tf(graph, {boxes: boxes_val, scores: scores_val},
                           ref, use_cpu_only=use_cpu_only, backend=backend)


class TestNonZero:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 ['nnv1_proto']
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [cb.non_zero(x=x)]

        expected_output_types = [(UNK_SYM, 2, builtins.fp32)]
        expected_outputs = [
            np.array(np.transpose(np.nonzero(x_val)))
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.random.randint(low=-1, high=2, size=(6, 1, 7))
        res = cb.non_zero(x=x_val)
        assert is_close(np.transpose(np.nonzero(x_val)), res.val)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, rank',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 6)]
                             ))
    def test_tf(self, use_cpu_only, backend, rank):
        with tf.Graph().as_default() as graph:
            shape = np.random.randint(low=1, high=4, size=rank)
            x_val = np.random.randint(low=-1, high=2, size=shape).astype(np.float32)
            x = tf.placeholder(tf.float32, shape=shape)
            run_compare_tf(graph, {x: x_val}, tf.where(x),
                           use_cpu_only=use_cpu_only, backend=backend)


class TestOneHot:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([1,0], dtype=np.int32)
        depth = 4

        input_placeholders = {"x": cb.placeholder(shape=x.shape,dtype=builtins.int32),
                              "y": cb.placeholder(shape=(),dtype=builtins.int32)}

        input_values = {"x": x, "y": depth}
        def build(x, y):
            return [
                    cb.one_hot(indices=x,one_hot_vector_size=4),
                    cb.one_hot(indices=x, one_hot_vector_size=4, axis=0),
                    cb.one_hot(indices=x, one_hot_vector_size=4, on_value = 1.0, off_value = 0.0),
                    cb.one_hot(indices=x, one_hot_vector_size=y, on_value=1.0, off_value=0.0)
            ]

        expected_output_types = [
                (2,4, builtins.int32),
                (4,2, builtins.int32),
                (2,4, builtins.fp32),
                (2,UNK_SYM, builtins.fp32),
        ]

        expected_outputs = [
                np.array([[0,1,0,0],[1,0,0,0]], dtype=np.float32),
                np.array([[0,1],[1,0], [0,0], [0,0]], dtype=np.float32),
                np.array([[0,1,0,0], [1,0,0,0]], dtype=np.float32),
                np.array([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank_and_axis, dynamic",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(2, 0), (2, -1), (3, 3), (3, 0), (3,-2),
                                  (4, -4), (4, 1), (4,-1), (4,-2), (4,3)],
                                 [True, False]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank_and_axis, dynamic):
        rank, axis = rank_and_axis
        depth, on_value, off_value = 30, 28.0, -4.0
        x_shape = np.random.randint(low=2, high=5, size=rank)

        if not dynamic:
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.int32, shape=x_shape)
                axis = axis if axis >= -1 else axis + rank + 1 ## TF limitation: Doesn't support axis < -1
                res = tf.one_hot(x, axis = axis, depth=depth, on_value=on_value, off_value=off_value)
                run_compare_tf(graph,
                               {x: np.random.randint(0, depth, size=x_shape)},
                               res, use_cpu_only=use_cpu_only,
                               frontend_only=False, backend=backend)
        else: # Dynamic Case with depth being an input
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.int32, shape=x_shape)
                depth_input = tf.placeholder(tf.int32)
                axis = axis if axis >= -1 else axis + rank + 1 ## TF limitation: Doesn't support axis < -1
                res = tf.one_hot(x, axis = axis, depth=depth_input, on_value=on_value, off_value=off_value)
                run_compare_tf(graph,
                               {x: np.random.randint(0, depth, size=x_shape),
                               depth_input:depth},
                               res, use_cpu_only=use_cpu_only,
                               frontend_only=False, backend=backend)


class TestPad:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True , False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        def test_constant_mode():
            t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            pad = np.array([1, 1, 2, 2], dtype=np.int32)
            input_placeholders = {"x": cb.placeholder(shape=t.shape)}
            input_values = {"x": t}

            def build(x):
                return cb.pad(x=x, pad=pad, mode="constant", constant_val=0.0)
            expected_output_types = (4, 7, builtins.fp32)
            expected_outputs = np.array([[0., 0., 0., 0., 0., 0., 0.],
                                         [0., 0., 1., 2., 3., 0., 0.],
                                         [0., 0., 4., 5., 6., 0., 0.],
                                         [0., 0., 0., 0., 0., 0., 0.]],
                                         dtype=np.float32)

            run_compare_builder(build, input_placeholders, input_values,
                                expected_output_types, expected_outputs,
                                use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)

        def test_constant_mode_constant_val():
            t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            pad = np.array([1, 1, 2, 2], dtype=np.int32)
            input_placeholders = {"x": cb.placeholder(shape=t.shape)}
            input_values = {"x": t}

            def build(x):
                return cb.pad(x=x, pad=pad, mode="constant", constant_val=0.5)
            expected_output_types = (4, 7, builtins.fp32)
            expected_outputs = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                         [0.5, 0.5, 1.,  2.,  3.,  0.5, 0.5],
                                         [0.5, 0.5, 4.,  5.,  6.,  0.5, 0.5],
                                         [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
                                         dtype=np.float32)

            run_compare_builder(build, input_placeholders, input_values,
                                expected_output_types, expected_outputs,
                                use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)

        def test_reflect_mode():
            t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            pad = np.array([1, 1, 2, 2], dtype=np.int32)
            input_placeholders = {"x": cb.placeholder(shape=t.shape)}
            input_values = {"x": t}

            def build(x):
                return cb.pad(x=x, pad=pad, mode="reflect")
            expected_output_types = (4, 7, builtins.fp32)
            expected_outputs = np.array([[6., 5., 4., 5., 6., 5., 4.],
                                         [3., 2., 1., 2., 3., 2., 1.],
                                         [6., 5., 4., 5., 6., 5., 4.],
                                         [3., 2., 1., 2., 3., 2., 1.]],
                                         dtype=np.float32)

            run_compare_builder(build, input_placeholders, input_values,
                                expected_output_types, expected_outputs,
                                use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)

        def test_replicate_mode():
            t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            pad = np.array([1, 1, 2, 2], dtype=np.int32)
            input_placeholders = {"x": cb.placeholder(shape=t.shape)}
            input_values = {"x": t}

            def build(x):
                return cb.pad(x=x, pad=pad, mode="replicate")
            expected_output_types = (4, 7, builtins.fp32)
            expected_outputs = np.array([[1., 1., 1., 2., 3., 3., 3.],
                                         [1., 1., 1., 2., 3., 3., 3.],
                                         [4., 4., 4., 5., 6., 6., 6.],
                                         [4., 4., 4., 5., 6., 6., 6.]],
                                         dtype=np.float32)

            run_compare_builder(build, input_placeholders, input_values,
                                expected_output_types, expected_outputs,
                                use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)

        def test_constant_general():
            t = np.arange(12, dtype=np.float32).reshape([2, 2, 3])
            pad = np.array([[1, 1], [2, 2], [1, 1]], dtype=np.int32)
            input_placeholders = {"x": cb.placeholder(shape=t.shape)}
            input_values = {"x": t}

            def build(x):
                return cb.pad(x=x, pad=pad.reshape(-1), mode="constant", constant_val=0.0)
            expected_output_types = (4, 6, 5, builtins.fp32)
            expected_outputs = np.pad(t, pad, mode="constant")

            run_compare_builder(build, input_placeholders, input_values,
                                expected_output_types, expected_outputs,
                                use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)
        # Test different modes
        test_constant_mode()
        test_constant_mode_constant_val()
        test_reflect_mode()
        test_replicate_mode()
        test_constant_general()

    @ssa_fn
    def test_builder_eval(self):
        def test_constant_mode():
            x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            v = cb.pad(x=x_val, pad=np.array([1, 1, 2, 2], dtype=np.int32), mode="constant", constant_val=0.0)
            expected_outputs = np.array([[0., 0., 0., 0., 0., 0., 0.],
                                         [0., 0., 1., 2., 3., 0., 0.],
                                         [0., 0., 4., 5., 6., 0., 0.],
                                         [0., 0., 0., 0., 0., 0., 0.]],
                                         dtype=np.float32)
            assert is_close(expected_outputs, v.val)

        def test_reflect_mode():
            x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            v = cb.pad(x=x_val, pad=np.array([1, 1, 2, 2], dtype=np.int32), mode="reflect")
            expected_outputs = np.array([[6., 5., 4., 5., 6., 5., 4.],
                                         [3., 2., 1., 2., 3., 2., 1.],
                                         [6., 5., 4., 5., 6., 5., 4.],
                                         [3., 2., 1., 2., 3., 2., 1.]],
                                         dtype=np.float32)
            assert is_close(expected_outputs, v.val)

        def test_replicate_mode():
            x_val = np.array([[[1, 2, 3],[4, 5, 6]]], dtype=np.float32)
            v = cb.pad(x=x_val, pad=np.array([1, 1, 2, 2], dtype=np.int32), mode="replicate")
            expected_outputs = np.array([[1., 1., 1., 2., 3., 3., 3.],
                                         [1., 1., 1., 2., 3., 3., 3.],
                                         [4., 4., 4., 5., 6., 6., 6.],
                                         [4., 4., 4., 5., 6., 6., 6.]],
                                         dtype=np.float32)
            assert is_close(expected_outputs, v.val)

        def test_constant_general():
            x_val = np.arange(12, dtype=np.float32).reshape([2, 2, 3])
            pad = np.array([[1, 1], [2, 2], [1, 1]], dtype=np.int32)
            v = cb.pad(x=x_val, pad=pad.reshape(-1), mode="constant", constant_val=0.0)
            expected_outputs = np.pad(x_val, pad, mode="constant")
            assert is_close(expected_outputs, v.val)

        # Test different modes
        test_constant_mode()
        test_reflect_mode()
        test_replicate_mode()
        test_constant_general()

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank, mode",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [2, 3, 4],
                                 # rdar://59854962 ([Pad Precision issue] Rank 5 Pad precision dropped on GPU comparing to CPU)
                                 ['reflect', 'constant']
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank, mode):
        input_shape = np.random.randint(low=2, high=10, size=rank)
        min_input_dim_size = input_shape.min()
        padding_val = np.random.randint(low=0, high=min_input_dim_size, size=(rank, 2), dtype=np.int32)

        # Only constant mode supports padding across all dimensions
        # All other padding modes are only applied on last two dimensions.
        if mode != "constant":
            padding_val[:-2] = 0

        tf_mode = mode.upper()
        tf.reset_default_graph()
        input = random_gen(input_shape, rand_min=0.2, rand_max=1000)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            paddings = tf.constant(padding_val)
            res = tf.pad(x, paddings=paddings, mode=tf_mode)
            run_compare_tf(graph, {x: input},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestRange1d:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([15.], dtype=np.float32)
        y = 5.
        z = 2.
        input_placeholders = {'x': cb.placeholder(shape=x.shape),
                              'y': cb.placeholder(shape=()),
                              'z': cb.placeholder(shape=())}
        input_values = {'x': x, 'y': y, 'z': z}

        def build(x,y,z):
            return [
                cb.mul(x=x, y=x),
                cb.range_1d(start=y, end=15.0, step=2.0),
                cb.range_1d(start=y, end=15.0, step=z),
                cb.range_1d(start=y, end=x, step=2.0),
                cb.range_1d(start=y, end=x, step=z),
                cb.range_1d(start=5.0, end=15.0, step=z),
                cb.range_1d(start=5.0, end=x, step=2.0),
                cb.range_1d(start=5.0, end=x, step=z)
            ]

        expected_output_types = [
                (1, builtins.fp32),
                (UNK_SYM, builtins.fp32),
                (UNK_SYM, builtins.fp32),
                (UNK_SYM, builtins.fp32),
                (UNK_SYM, builtins.fp32),
                (UNK_SYM, builtins.fp32),
                (UNK_SYM, builtins.fp32),
                (UNK_SYM, builtins.fp32)
        ]

        expected_outputs = [
                np.array([225.], dtype=np.float32),
                np.array([5, 7, 9, 11, 13], dtype=np.float32),
                np.array([5, 7, 9, 11, 13], dtype=np.float32),
                np.array([5, 7, 9, 11, 13], dtype=np.float32),
                np.array([5, 7, 9, 11, 13], dtype=np.float32),
                np.array([5, 7, 9, 11, 13], dtype=np.float32),
                np.array([5, 7, 9, 11, 13], dtype=np.float32),
                np.array([5, 7, 9, 11, 13], dtype=np.float32),
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        v = cb.range_1d(start=5, end=15, step=2)
        assert is_close(np.arange(5,15,2), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, params",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(-10.4, 23, 12.2), (0, 1000, 1), (50.5, 90.5, 1.5), (5, 8, 2),
                                  (5, 8, 98), (5, 8, 1.5), (10, 5, -0.6), (24, -65, -2)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, params):
        start, end, step = params
        with tf.Graph().as_default() as graph:
            limit = tf.placeholder(tf.float32)
            res = tf.range(start=start, limit= limit, delta=step)
            run_compare_tf(graph,
                           {limit: end},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

        with tf.Graph().as_default() as graph:
            delta = tf.placeholder(tf.float32)
            res = tf.range(start=start, limit= end, delta=delta)
            run_compare_tf(graph,
                           {delta: step},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

        with tf.Graph().as_default() as graph:
            begin = tf.placeholder(tf.float32)
            res = tf.range(start=begin, limit= end, delta=step)
            run_compare_tf(graph,
                           {begin: start},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestTile:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape)}

        input_values = {"x": x}
        def build(x):
            return [cb.tile(x=x, reps = (1,1)),
                    cb.tile(x=x, reps = (2,)),
                    cb.tile(x=x, reps = (2,1)),
                    ]

        expected_output_types = [
                (2, 3, builtins.fp32),
                (2, 6, builtins.fp32),
                (4, 3, builtins.fp32),
                ]

        expected_outputs = [
                x,
                np.array([[1, 2, 3, 1, 2, 3],[4, 5, 6, 4, 5, 6]], dtype=np.float32),
                np.array([[1, 2, 3],[4, 5, 6], [1, 2, 3],[4, 5, 6]], dtype=np.float32)
                ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        v = cb.tile(x=x, reps = (2,))
        assert is_close(np.tile(x, reps = (2,)), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank_and_reps",
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [(1, (2,)), (2, (1,2)), (2, (2,2)), (3, (3,2,1)), (3, (2,1,3)), (3, (2,1,1)),
                                  (4, (1,3,2,1)), (4, (2,1,1,2)), (5, (2,1,1,3,2)), (5, (1,1,2,3,2))]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank_and_reps):
        rank, reps = rank_and_reps
        x_shape = np.random.randint(low=2, high=5, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            res = tf.tile(x, multiples = reps)
            run_compare_tf(graph,
                           {x: np.random.rand(*x_shape)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestTopK:
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
            return cb.topk(x=x, k=2, axis=1)

        expected_output_types = [
            (2, 2, builtins.fp32), (2, 2, builtins.int32),
        ]
        expected_outputs = [
            np.array([[2., -1.], [6., 4.]], dtype=np.float32),
            np.array([[1, 0], [2, 0]], dtype=np.float32),
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        def np_topk(x, k, axis, ascending=False):
            indices = np.argsort(x, axis=axis)
            if not ascending:
                indices = np.argsort(-x, axis=axis)
            slc = [slice(None)] * len(x.shape)
            slc[axis] = slice(0, k)
            indices = indices[tuple(slc)]
            values = np.take_along_axis(x, indices, axis=axis)
            return values, indices

        val = np.array([[-1., 7., -3.], [4., -5., 8.]], dtype=np.float32)
        res_values, res_indices = cb.topk(x=val, k=1, axis=0)
        ref_values, ref_indices = np_topk(x=val, k=1, axis=0)
        assert is_close(ref_values, res_values.val)
        assert is_close(ref_indices, res_indices.val)
        res_values, res_indices = cb.topk(x=val, k=2, axis=-1, ascending=True)
        ref_values, ref_indices = np_topk(x=val, k=2, axis=-1, ascending=True)
        assert is_close(ref_values, res_values.val)
        assert is_close(ref_indices, res_indices.val)

    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_symbolic(self, use_cpu_only, backend):
        s0 = get_new_symbol()

        val = np.array([[1., 2., -3.], [4., -5., 6.]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=(s0, 3))}
        input_values = {'x': val}

        def build(x):
            return cb.topk(x=x, k=2, axis=-1, ascending=True)

        expected_output_types = [
            (s0, 2, builtins.fp32),
            (s0, 2, builtins.int32),
        ]
        expected_outputs = [
            np.array([[-3., 1.], [-5., 4.]], dtype=np.float32),
            np.array([[2, 0], [1, 0]], dtype=np.float32)
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, rank, k',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 6)],
                                 [1, 2, 3],
                             ))
    def test_tf(self, use_cpu_only, backend, rank, k):
        # TensorFlow only supports last dimension (axis = -1).
        shape = np.random.randint(low=3, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.math.top_k(x, k=k, sorted=True)
            ref = (ref[0] + 1, ref[1] + 2)
            run_compare_tf(graph, {x: random_gen(shape, rand_min=-100, rand_max=100)},
                           ref, use_cpu_only=use_cpu_only, backend=backend)


class TestFlatten:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
            ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[[1, 2, 3], [4, 5, 6]], [[-1, -2, -3], [-4, -5, -6]]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}
        def build(x):
            return [cb.flatten(x=x)]

        expected_output_types = [
                (2, 6, builtins.fp32),
                ]
        expected_outputs = [
                np.array([[1, 2, 3, 4, 5, 6],
                          [-1, -2, -3, -4, -5, -6]], dtype=np.float32),
                ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        t = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
        f = cb.flatten(x=t)
        expected_f = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
        assert is_close(expected_f, f.val)

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
            ))
    def test_builder_to_backend_symbolic(self, use_cpu_only, backend):
        s0 = get_new_symbol()

        # Test variadic (rdar://59559656)
        input_placeholders = {
                "x": cb.placeholder(shape=(s0, 4, 5, 6)),
                }

        def build(x):
            return [cb.flatten(x=x)]

        input = np.random.rand(10, 4, 5, 6)
        output = input.reshape(10, -1)

        expected_output_types = (s0, 120, builtins.fp32)
        expected_outputs = [ output ]

        input_values = { "x": input }
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend",
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_tf(self, use_cpu_only, backend):
        shapes = [[10, 10],
                  [3, 4, 5, 6],
                  [4, 4, 5, 6]]

        for _shape in shapes:
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=_shape)
                # NOTE: Currently, this gets map to reshape
                res = tf.keras.backend.flatten(x)
                run_compare_tf(graph, {x: np.random.rand(*_shape)},
                                res, use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)


class TestShape:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
            ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        pad = np.array([1, 1, 2, 2], dtype=np.int32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            x = cb.pad(x=x, pad=pad, mode="constant", constant_val=0.0)
            return cb.shape(x=x)

        expected_output_types = (2, builtins.int32)
        expected_outputs = [
                np.array([4, 7], dtype=np.int32),
                ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        t = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
        f = cb.shape(x=t)
        expected_f = np.array([1, 2, 3], dtype=np.float32)
        assert is_close(expected_f, f.val)

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_symbolic(self, use_cpu_only, backend):
        s0 = get_new_symbol()

        # Test variadic (rdar://59559656)
        input_placeholders = {
                "x": cb.placeholder(shape=(s0, 4, 5, 6)),
                }

        def build(x):
            return [cb.shape(x=x)]

        input = np.random.rand(10, 4, 5, 6)
        output = np.array([10, 4, 5, 6], dtype=np.float32)

        expected_output_types = (4, builtins.int32)
        expected_outputs = [ output ]

        input_values = { "x": input }
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, rank',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 6)],
                             ))
    def test_tf(self, use_cpu_only, backend, rank):
        shape = np.random.randint(low=3, high=6, size=rank)
        shape_holder = [None] * rank
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape_holder)
            values = tf.shape(x)
            run_compare_tf(graph, {x: random_gen(shape, rand_min=-100, rand_max=100)},
                           values, use_cpu_only=use_cpu_only, backend=backend)


class TestConcat:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
            ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t1 = np.array([[1, 2],[4, 5]], dtype=np.float32)
        t2 = np.array([[7, 8]], dtype=np.float32)

        input_placeholders = {
                "x": cb.placeholder(shape=t1.shape),
                "y": cb.placeholder(shape=t2.shape),
                }
        input_values = {"x": t1, "y": t2}

        def build(x, y):
            return cb.concat(values=(x, y), axis=0),

        expected_output_types = [
                (3, 2, builtins.fp32),
                ]
        expected_outputs = [
                np.array([[1, 2],
                          [4, 5],
                          [7, 8]], dtype=np.float32),
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
    def test_builder_to_backend_type_promotion(self, use_cpu_only, backend):
        t1 = np.array([[1, 2],[4, 5]], dtype=np.float32)
        t2 = np.array([[7, 8]], dtype=np.float32)

        input_placeholders = {
                "x": cb.placeholder(shape=t1.shape),
                }
        input_values = {"x": t1}

        def build(x):
            t2 = np.array([[7, 8]], dtype=np.int32)
            return cb.concat(values=(x, t2), axis=0),

        expected_output_types = [
                # np.int32 should be promoted to fp32
                (3, 2, builtins.fp32),
                ]
        expected_outputs = [
                np.array([[1, 2],
                          [4, 5],
                          [7, 8]], dtype=np.float32),
                          ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        values = [np.random.rand(1, 1, 6, 2),
                  np.random.rand(1, 1, 3, 2),]
        v = cb.concat(values=values, axis=2)
        assert is_close(np.concatenate(values, 2), v.val)

    @ssa_fn
    def test_builder_eval_failure(self):
        values = [np.random.rand(1, 1, 6, 2),
                  np.random.rand(1, 1, 3, 1),]
        with pytest.raises(ValueError):
            v = cb.concat(values=values, axis=2)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend",
                             itertools.product(
                                 [True, False],
                                 ['nnv1_proto'],
                             )
                             )
    def test_tf(self, use_cpu_only, backend):
        input_shape1 = [3, 2, 1]
        input_shape2 = [3, 1, 1]
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape1)
            y = tf.placeholder(tf.float32, shape=input_shape2)
            res = tf.concat((x, y), axis=-2)
            inputs = {x: np.random.rand(*input_shape1),
                      y: np.random.rand(*input_shape2)
                    }
            run_compare_tf(graph, inputs,
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestSplit:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
            ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[1, 2],
                      [3, 4],
                      [5, 6]], dtype=np.float32)

        input_placeholders = {
                "x": cb.placeholder(shape=t.shape),
                }
        input_values = {"x": t}

        def build(x):
            return cb.split(x=x, num_splits=3, axis=0) + \
                    cb.split(x=x, split_sizes=[1,2], axis=0)

        expected_output_types = [
                (1, 2, builtins.fp32),
                (1, 2, builtins.fp32),
                (1, 2, builtins.fp32),
                (1, 2, builtins.fp32),
                (2, 2, builtins.fp32),
                ]
        expected_outputs = [
                np.array([[1, 2]], dtype=np.float32),
                np.array([[3, 4]], dtype=np.float32),
                np.array([[5, 6]], dtype=np.float32),
                np.array([[1, 2]], dtype=np.float32),
                np.array([[3, 4],
                          [5, 6]], dtype=np.float32),
                          ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        t = np.array([[1, 2],
                      [3, 4],
                      [5, 6]], dtype=np.float32)
        vs = cb.split(x=t, num_splits=3, axis=0)
        es = np.split(t, [1,2,3], axis=0)
        for v, e in zip(vs, es):
            assert is_close(e, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend",
                             itertools.product(
                                 [True, False],
                                 backends,
                             )
                             )
    def test_tf(self, use_cpu_only, backend):
        input_shape1 = [3, 2, 1]
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape1)
            res = tf.split(x, 3, axis=0)
            # TODO (rdar://60358242) If tf.split output is returned, there's no
            # get_tuple nodes. Some graph pass is needed. Example:
            #
            #    x = tf.placeholder(tf.float32, shape=input_shape1)
            #    res = tf.split(x, 3, axis=0)
            #
            # res are ['split:0', 'split:1', 'split']
            #
            # but node.outputs == ['gto_1', 'gto_2', 'gto_3']
            res = (res[0]+1, res[1]+2, res[2]+3)
            inputs = {x: np.random.rand(*input_shape1),
                    }
            run_compare_tf(graph, inputs,
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend",
                             itertools.product(
                                 [True, False],
                                 backends,
                             )
                             )
    def test_tf_splitv(self, use_cpu_only, backend):
        input_shape1 = [3, 2, 1]
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape1)
            res = tf.split(x, [1, 2], axis=0)
            res = (res[0]+1, res[1]+2)
            inputs = {x: np.random.rand(*input_shape1),
                    }
            run_compare_tf(graph, inputs,
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestStack:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
            ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t1 = np.array([1, 2, 3], dtype=np.float32)
        t2 = np.array([7, 8, 9], dtype=np.float32)

        input_placeholders = {
                "x": cb.placeholder(shape=t1.shape),
                "y": cb.placeholder(shape=t2.shape),
                }
        input_values = {"x": t1, "y": t2}

        def build(x, y):
            return [cb.stack(values=(x, y), axis=0),
                    cb.stack(values=(x, y), axis=1)]

        expected_output_types = [
                (2, 3, builtins.fp32),
                (3, 2, builtins.fp32),
                ]
        expected_outputs = [
                np.array([[1, 2, 3],
                          [7, 8, 9]], dtype=np.float32),
                np.array([[1, 7],
                          [2, 8],
                          [3, 9]], dtype=np.float32),
                          ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        values = [np.random.rand(1, 1, 3, 2).astype(np.float32),
                  np.random.rand(1, 1, 3, 2).astype(np.float32),]
        v = cb.stack(values=values, axis=2)
        assert is_close(np.stack(values, 2), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend",
                             itertools.product(
                                 [True, False],
                                 backends,
                             )
                             )
    def test_tf(self, use_cpu_only, backend):
        input_shape1 = [3, 1, 1]
        input_shape2 = [3, 1, 1]
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape1)
            y = tf.placeholder(tf.float32, shape=input_shape2)
            res = [tf.stack((x, y), axis=0), tf.stack((x, y), axis=-1)]
            inputs = {x: np.random.rand(*input_shape1),
                      y: np.random.rand(*input_shape2)
                    }
            run_compare_tf(graph, inputs,
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestArgSort:

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
                cb.argsort(x=x),
                cb.argsort(x=x, axis=0, ascending=True)
            ]

        expected_output_types = [
            (2, 3, builtins.int32),
            (2, 3, builtins.int32),
        ]
        expected_outputs = [
            np.array([[1, 0, 2], [2, 0, 1]], dtype=np.int32),
            np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32),
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = random_gen(shape=(1, 3, 2, 2), rand_min=-100, rand_max=100)
        res = cb.argsort(x=x_val, axis=-3)
        assert is_close(np.argsort(x_val, axis=-3), res.val)

    @pytest.mark.skipif(True, reason='rdar://60498397 (Re-enable NNv2->NNv1 TF tests for ArgSort)')
    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, rank, axis, direction',
                             itertools.product(
                                 [True],
                                 backends,
                                 [rank for rank in range(1, 6)],
                                 [-1, 0],
                                 ['ascending', 'descending'],
                             ))
    def test_tf(self, use_cpu_only, backend, rank, axis, direction):
        shape = np.random.randint(low=1, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.argsort(x, axis=axis, direction=direction.upper())
            run_compare_tf(graph, {x: random_gen(shape, rand_min=-100, rand_max=100)},
                           ref, use_cpu_only=use_cpu_only, backend=backend)
