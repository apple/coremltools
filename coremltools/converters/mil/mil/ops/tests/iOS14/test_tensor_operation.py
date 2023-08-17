#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import platform

import numpy as np
import pytest

import coremltools as ct
from coremltools._deps import _HAS_TF_2, MSG_TF2_NOT_FOUND
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, get_new_symbol, types
from coremltools.converters.mil.mil.ops.tests.iOS14 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import (
    UNK_SYM,
    UNK_VARIADIC,
    construct_inputs_from_placeholders,
    mark_api_breaking,
    run_compare_builder,
)
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from coremltools.converters.mil.mil.types.type_mapping import nptype_from_builtin
from coremltools.converters.mil.testing_reqs import compute_units
from coremltools.converters.mil.testing_utils import get_op_types_in_program, random_gen, ssa_fn

if _HAS_TF_2:
    import tensorflow as tf


class TestBandPart:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x_val = np.array(
            [
                [3.0, 3.0, 5.0, 1.0],
                [5.0, 6.0, 3.0, 8.0],
                [7.0, 2.0, 7.0, 2.0],
                [6.0, 7.0, 7.0, 1.0],
            ],
            dtype=np.float32,
        )
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        input_values = {"x": x_val}

        def build(x):
            return [
                mb.band_part(x=x),
                mb.band_part(x=x, lower=0, upper=-1),
                mb.band_part(x=x, lower=-1, upper=0),
                mb.band_part(x=x, lower=0, upper=0),
            ]

        expected_output_types = [
            (4, 4, types.fp32),
            (4, 4, types.fp32),
            (4, 4, types.fp32),
            (4, 4, types.fp32),
        ]

        expected_outputs = [
            np.array(
                [
                    [3.0, 3.0, 5.0, 1.0],
                    [5.0, 6.0, 3.0, 8.0],
                    [7.0, 2.0, 7.0, 2.0],
                    [6.0, 7.0, 7.0, 1.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [3.0, 3.0, 5.0, 1.0],
                    [0.0, 6.0, 3.0, 8.0],
                    [0.0, 0.0, 7.0, 2.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [3.0, 0.0, 0.0, 0.0],
                    [5.0, 6.0, 0.0, 0.0],
                    [7.0, 2.0, 7.0, 0.0],
                    [6.0, 7.0, 7.0, 1.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [3.0, 0.0, 0.0, 0.0],
                    [0.0, 6.0, 0.0, 0.0],
                    [0.0, 0.0, 7.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    def get_output_from_mlmodel(
        self,
        x_val: np.ndarray,
        num_lower: int,
        num_upper: int,
        dtype: type,
    ) -> np.ndarray:
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 4), dtype=dtype)])
        def prog(x):
            return mb.band_part(x=x, lower=num_lower, upper=num_upper, name="out")

        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT32,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        out = mlmodel.predict({"x": x_val})["out"]
        return out

    def get_value_inference_output(
        self,
        x_val: np.ndarray,
        num_lower: int,
        num_upper: int,
        dtype: type,
    ) -> np.ndarray:
        func_inputs = {"x": mb.placeholder(shape=[3, 4], dtype=dtype)}
        with Function(func_inputs) as ssa_fun:
            x = ssa_fun.inputs["x"]
            v = mb.band_part(x=x_val, lower=num_lower, upper=num_upper)
        return v.val

    @pytest.mark.skipif(
        ct.utils._macos_version() < (10, 15), reason="needs mlprogram, skip on macos < 10.15"
    )
    @pytest.mark.parametrize(
        "lower_upper, dtype",
        itertools.product(
            [(0, -1), (-1, 0), (0, 0), (1, 1), (1, 2), (2, 1)],
            [types.int32, types.fp32],
        ),
    )
    def test_value_inference(self, lower_upper, dtype):
        num_lower, num_upper = lower_upper
        np_type = nptype_from_builtin(dtype)
        test_input = np.random.rand(3, 4).astype(np_type)
        out_value_inference = self.get_value_inference_output(
            test_input, num_lower, num_upper, dtype
        )
        out_from_model_prediction = self.get_output_from_mlmodel(
            test_input, num_lower, num_upper, dtype
        )
        assert out_value_inference.dtype == test_input.dtype
        np.testing.assert_allclose(
            out_value_inference, out_from_model_prediction, atol=1e-3, rtol=1e-3
        )


class TestCumSum:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return mb.cumsum(x=x, axis=0, reverse=True, exclusive=False)

        expected_output_types = (2, 3, types.fp32)
        expected_outputs = np.array([[5, 7, 9], [4, 5, 6]], dtype=np.float32)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        v = mb.cumsum(x=x_val)
        np.testing.assert_allclose(np.cumsum(x_val, axis=0), v.val, atol=1e-04, rtol=1e-05)

    @ssa_fn
    def test_invalid_arg(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(ValueError):
            mb.cumsum(x=x_val, axis=0, invalid_arg=3)

    @ssa_fn
    def test_invalid_axis1(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(ValueError):
            mb.cumsum(x=x_val, axis=-2)

    @ssa_fn
    def test_invalid_axis2(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(ValueError):
            mb.cumsum(x=x_val, axis=len(x_val.shape))

    @ssa_fn
    def test_invalid_axis3(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(ValueError):
            mb.cumsum(x=x_val, axis="")

    @ssa_fn
    def test_invalid_reverse1(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(ValueError):
            mb.cumsum(x=x_val, reverse="")

    @ssa_fn
    def test_invalid_reverse2(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(ValueError):
            mb.cumsum(x=x_val, reverse=0)

    @ssa_fn
    def test_invalid_reverse3(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(ValueError):
            mb.cumsum(x=x_val, reverse=1)

    @ssa_fn
    def test_invalid_exclusive1(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(ValueError):
            mb.cumsum(x=x_val, exclusive="")

    @ssa_fn
    def test_invalid_exclusive2(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(ValueError):
            mb.cumsum(x=x_val, exclusive=0)

    @ssa_fn
    def test_invalid_exclusive3(self):
        x_val = random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(ValueError):
            mb.cumsum(x=x_val, exclusive=1)

    @ssa_fn
    def test_invalid_input1(self):
        x_val = 1
        with pytest.raises(ValueError):
            mb.cumsum(x=x_val)

    @ssa_fn
    def test_invalid_input2(self):
        x_val = ["1"]
        with pytest.raises(ValueError):
            mb.cumsum(x=x_val)


class TestFill:
    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        shape = (2, 1, 3)
        x_val = np.zeros(shape=shape, dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}

        input_values = {"x": x_val}

        def build(x):
            return mb.add(x=x, y=mb.fill(shape=shape, value=1.0))

        expected_output_types = [(2, 1, 3, types.fp32)]
        expected_outputs = [np.full(shape=shape, fill_value=1.0)]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        shape = np.random.randint(low=1, high=3, size=5).astype(np.int32)
        res = mb.fill(shape=shape, value=1991.0).val
        np.testing.assert_allclose(np.full(shape, fill_value=1991.0), res, atol=1e-04, rtol=1e-05)

    @pytest.mark.parametrize(
        "compute_unit, backend, rank, value",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(1, 6)],
            [-1917.0, 0.0, 2048.0],
        ),
    )
    def test_builder_to_backend_stress(self, compute_unit, backend, rank, value):
        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)
        x_val = np.zeros(shape=shape, dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        input_values = {"x": x_val}

        def build(x):
            return mb.add(x=x, y=mb.fill(shape=shape, value=value))

        expected_outputs = [np.full(shape=shape, fill_value=value)]
        expected_output_types = [o.shape[:] + (types.fp32,) for o in expected_outputs]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_symbolic(self, compute_unit, backend):
        s_len = get_new_symbol()
        input_placeholders = {
            "shape": mb.placeholder(shape=(s_len,), dtype=types.int32),
        }

        def build(shape):
            return [mb.fill(shape=shape)]

        expected_output_types = [(UNK_VARIADIC, types.fp32)]
        expected_outputs = [np.zeros(shape=(2, 1, 3), dtype=np.float32)]
        input_values = {"shape": np.array([2, 1, 3], dtype=np.float32)}

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            inputs=construct_inputs_from_placeholders(input_placeholders, 3)
            if backend.backend == "mlprogram"
            else None,
            compute_unit=compute_unit,
            backend=backend,
        )


@pytest.mark.skipif(not _HAS_TF_2, reason=MSG_TF2_NOT_FOUND)
class TestNonMaximumSuppression:
    @mark_api_breaking(breaking_opset_version=ct.target.iOS17)
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        boxes_val = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 3.0],
                ]
            ],
            dtype=np.float32,
        )
        scores_val = np.array([[[-3.5], [9.4], [2.3], [0.7]]], dtype=np.float32)
        input_placeholders = {
            "boxes": mb.placeholder(shape=(1, 4, 4)),
            "scores": mb.placeholder(shape=(1, 4, 1)),
        }
        input_values = {"boxes": boxes_val, "scores": scores_val}

        expected_output_types = [
            (1, 2, 4, types.fp32),
            (1, 2, 1, types.fp32),
            (1, 2, types.int32),
            (1, types.int32),
        ]
        expected_outputs = [
            np.array([[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]], dtype=np.float32),
            np.array([[[9.4], [2.3]]], dtype=np.float32),
            np.array([[1, 2]], dtype=np.int32),
            np.array([2], dtype=np.int32),
        ]

        def build(boxes, scores):
            return mb.non_maximum_suppression(
                boxes=boxes,
                scores=scores,
                iou_threshold=0.2,
                score_threshold=0.4,
                max_boxes=2,
                per_class_suppression=True,
            )

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @staticmethod
    def _compute_iou_matrix(boxes):
        # input is (N, 4), in order [center_w, center_h, width, height]
        boxes = boxes.astype(np.float32)
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
        boxes, scores, iou_threshold, score_threshold, max_boxes, per_class_suppression
    ):
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
                ids_g = tf.image.non_max_suppression(
                    box_coord_matrix,
                    score_vector,
                    max_output_size=max_boxes,
                    iou_threshold=iou_threshold,
                    score_threshold=score_threshold,
                )
                ids = ids_g.numpy()
            else:
                # this is slightly complicated as TF does not directly support it
                class_ids = np.argmax(scores[b, :, :], axis=-1)  # (n_box,)
                sorted_score_ids = np.argsort(-score_vector)
                box_coord_matrix2 = np.take(box_coord_matrix, sorted_score_ids, axis=0)
                score_vector2 = np.take(score_vector, sorted_score_ids)
                class_ids = np.take(class_ids, sorted_score_ids)
                classes_seen = dict()
                ids_intermediate = np.array([], dtype=np.int32)
                for n in range(n_box):
                    if class_ids[n] in classes_seen:
                        continue
                    c = class_ids[n]
                    classes_seen[c] = True
                    current_class_ids = np.where(class_ids == c)[0]
                    if len(current_class_ids) > 0:
                        feed_in1 = np.take(box_coord_matrix2, current_class_ids, axis=0)
                        feed_in2 = np.take(score_vector2, current_class_ids)
                        cur_ids_g = tf.image.non_max_suppression(
                            feed_in1,
                            feed_in2,
                            max_output_size=max_boxes,
                            iou_threshold=iou_threshold,
                            score_threshold=score_threshold,
                        )
                        cur_ids = cur_ids_g.numpy()

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

    @mark_api_breaking(breaking_opset_version=ct.target.iOS17)
    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "iou_threshold_percentile",
                "score_threshold_percentile",
                "n_boxes",
                "n_batch",
                "n_score",
                "per_class_suppression",
            ]
        ),
        itertools.product(
            compute_units,
            backends,
            [0, 30, 80, 100],
            [0, 40, 100],
            [(10, 7), (30, 37), (100, 64)],
            [1],
            [1, 4, 7],
            [True, False],
        ),
    )
    def test_builder_to_backend_stress(
        self,
        compute_unit,
        backend,
        iou_threshold_percentile,
        score_threshold_percentile,
        n_boxes,
        n_batch,
        n_score,
        per_class_suppression,
    ):
        if backend.backend == "mlprogram" and iou_threshold_percentile == 0:
            pytest.xfail("rdar://78080118")

        if (
            backend.backend == "neuralnetwork"
            and n_boxes == (10, 7)
            and platform.machine() == "x86_64"
        ):
            pytest.xfail("rdar://78080118 (Investigate failing tests for NMS in coremltools)")

        if backend.backend == "mlprogram" and backend.precision == "fp16":
            pytest.xfail("CPU: rdar://80662705 and GPU: rdar://80661262")

        n_boxes_in, n_boxes_out = n_boxes
        boxes_val = random_gen((n_batch, n_boxes_in, 4), 0, 100)
        scores_val = random_gen((n_batch, n_boxes_in, n_score), -100, 100, allow_duplicate=False)

        iou_matrix = self._compute_iou_matrix(boxes_val[0, :, :])
        iou_matrix = iou_matrix[~np.eye(iou_matrix.shape[0], dtype=bool)].reshape(
            iou_matrix.shape[0], -1
        )

        if score_threshold_percentile == 0:
            score_threshold = np.min(scores_val) - 1
        elif score_threshold_percentile == 100:
            score_threshold = np.max(scores_val) + 1
        else:
            score_threshold = np.percentile(scores_val, score_threshold_percentile) + 0.01

        if iou_threshold_percentile == 0:
            iou_threshold = np.maximum(np.min(iou_matrix) - 0.01, 0.0)
        else:
            iou_threshold = np.percentile(iou_matrix, iou_threshold_percentile) + 0.01
        iou_threshold = np.maximum(iou_threshold, 1e-8)

        (tf_boxes, tf_scores, tf_indices, tf_num_boxes,) = self._ref_non_maximum_suppression(
            boxes_val,
            scores_val,
            iou_threshold,
            score_threshold,
            n_boxes_out,
            per_class_suppression,
        )
        expected_outputs = [tf_boxes, tf_scores, tf_indices, tf_num_boxes]
        expected_output_types = [
            tf_boxes.shape[:] + (types.fp32,),
            tf_scores.shape[:] + (types.fp32,),
            tf_indices.shape[:] + (types.int32,),
            tf_num_boxes.shape[:] + (types.int32,),
        ]

        input_placeholders = {
            "boxes": mb.placeholder(shape=(n_batch, n_boxes_in, 4)),
            "scores": mb.placeholder(shape=(n_batch, n_boxes_in, n_score)),
        }
        input_values = {"boxes": boxes_val, "scores": scores_val}

        def build(boxes, scores):
            return mb.non_maximum_suppression(
                boxes=boxes,
                scores=scores,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                max_boxes=n_boxes_out,
                per_class_suppression=per_class_suppression,
            )

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestNonZero:
    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x_val = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        input_values = {"x": x_val}

        def build(x):
            return [mb.non_zero(x=x)]

        expected_output_types = [(UNK_SYM, 2, types.int32)]
        expected_outputs = [np.array(np.transpose(np.nonzero(x_val)))]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.random.randint(low=-1, high=2, size=(6, 1, 7))
        res = mb.non_zero(x=x_val)
        np.testing.assert_allclose(np.transpose(np.nonzero(x_val)), res.val, atol=1e-04, rtol=1e-05)

    @ssa_fn
    def test_shape_inference_for_deterministic_input(self):
        # If the input is compile time known, the builder should be able to infer the shape from value
        x_val = np.array([[0, 2], [1, 1]])
        res = mb.non_zero(x=x_val)
        assert res.shape == (3, 2)


class TestOneHot:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([1, 0], dtype=np.int32)
        depth = 4

        input_placeholders = {
            "x": mb.placeholder(shape=x.shape, dtype=types.int32),
            "y": mb.placeholder(shape=(1,), dtype=types.int32),
        }

        input_values = {"x": x, "y": depth}

        def build(x, y):
            return [
                mb.one_hot(indices=x, one_hot_vector_size=4),
                mb.one_hot(indices=x, one_hot_vector_size=4, axis=0),
                mb.one_hot(indices=x, one_hot_vector_size=4, on_value=1.0, off_value=0.1),
                mb.one_hot(indices=x, one_hot_vector_size=mb.squeeze(x=y), on_value=1, off_value=9),
            ]

        expected_output_types = [
            (2, 4, types.int32),
            (4, 2, types.int32),
            (2, 4, types.fp32),
            (2, UNK_SYM, types.int32),
        ]

        expected_outputs = [
            np.array([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
            np.array([[0, 1], [1, 0], [0, 0], [0, 0]], dtype=np.float32),
            np.array([[0.1, 1, 0.1, 0.1], [1, 0.1, 0.1, 0.1]], dtype=np.float32),
            np.array([[9, 1, 9, 9], [1, 9, 9, 9]], dtype=np.float32),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestPad:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        def test_constant_mode():
            t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            pad = np.array([1, 1, 2, 2], dtype=np.int32)
            input_placeholders = {"x": mb.placeholder(shape=t.shape)}
            input_values = {"x": t}

            def build(x):
                return mb.pad(x=x, pad=pad, mode="constant", constant_val=0.0)

            expected_output_types = (4, 7, types.fp32)
            expected_outputs = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )

            run_compare_builder(
                build,
                input_placeholders,
                input_values,
                expected_output_types,
                expected_outputs,
                compute_unit=compute_unit,
                backend=backend,
            )

        def test_constant_mode_constant_val():
            t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            pad = np.array([1, 1, 2, 2], dtype=np.int32)
            input_placeholders = {"x": mb.placeholder(shape=t.shape)}
            input_values = {"x": t}

            def build(x):
                return mb.pad(x=x, pad=pad, mode="constant", constant_val=0.5)

            expected_output_types = (4, 7, types.fp32)
            expected_outputs = np.array(
                [
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    [0.5, 0.5, 1.0, 2.0, 3.0, 0.5, 0.5],
                    [0.5, 0.5, 4.0, 5.0, 6.0, 0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                ],
                dtype=np.float32,
            )

            run_compare_builder(
                build,
                input_placeholders,
                input_values,
                expected_output_types,
                expected_outputs,
                compute_unit=compute_unit,
                backend=backend,
            )

        def test_reflect_mode():
            t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            pad = np.array([1, 1, 2, 2], dtype=np.int32)
            input_placeholders = {"x": mb.placeholder(shape=t.shape)}
            input_values = {"x": t}

            def build(x):
                return mb.pad(x=x, pad=pad, mode="reflect")

            expected_output_types = (4, 7, types.fp32)
            expected_outputs = np.array(
                [
                    [6.0, 5.0, 4.0, 5.0, 6.0, 5.0, 4.0],
                    [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
                    [6.0, 5.0, 4.0, 5.0, 6.0, 5.0, 4.0],
                    [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
                ],
                dtype=np.float32,
            )

            run_compare_builder(
                build,
                input_placeholders,
                input_values,
                expected_output_types,
                expected_outputs,
                compute_unit=compute_unit,
                backend=backend,
            )

        def test_replicate_mode():
            t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            pad = np.array([1, 1, 2, 2], dtype=np.int32)
            input_placeholders = {"x": mb.placeholder(shape=t.shape)}
            input_values = {"x": t}

            def build(x):
                return mb.pad(x=x, pad=pad, mode="replicate")

            expected_output_types = (4, 7, types.fp32)
            expected_outputs = np.array(
                [
                    [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0],
                    [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0],
                    [4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0],
                ],
                dtype=np.float32,
            )

            run_compare_builder(
                build,
                input_placeholders,
                input_values,
                expected_output_types,
                expected_outputs,
                compute_unit=compute_unit,
                backend=backend,
            )

        def test_constant_general():
            t = np.arange(12, dtype=np.float32).reshape([2, 2, 3])
            pad = np.array([[1, 1], [2, 2], [1, 1]], dtype=np.int32)
            input_placeholders = {"x": mb.placeholder(shape=t.shape)}
            input_values = {"x": t}

            def build(x):
                return mb.pad(x=x, pad=pad.reshape(-1), mode="constant", constant_val=0.0)

            expected_output_types = (4, 6, 5, types.fp32)
            expected_outputs = np.pad(t, pad, mode="constant")

            run_compare_builder(
                build,
                input_placeholders,
                input_values,
                expected_output_types,
                expected_outputs,
                compute_unit=compute_unit,
                backend=backend,
            )

        # Test different modes
        test_constant_mode()
        test_constant_mode_constant_val()
        test_reflect_mode()
        test_replicate_mode()
        test_constant_general()

    @ssa_fn
    def test_builder_eval(self):
        def test_constant_mode():
            x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            v = mb.pad(
                x=x_val,
                pad=np.array([1, 1, 2, 2], dtype=np.int32),
                mode="constant",
                constant_val=0.0,
            )
            expected_outputs = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)

        def test_reflect_mode():
            x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            v = mb.pad(x=x_val, pad=np.array([1, 1, 2, 2], dtype=np.int32), mode="reflect")
            expected_outputs = np.array(
                [
                    [6.0, 5.0, 4.0, 5.0, 6.0, 5.0, 4.0],
                    [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
                    [6.0, 5.0, 4.0, 5.0, 6.0, 5.0, 4.0],
                    [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)

        def test_replicate_mode():
            x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            v = mb.pad(x=x_val, pad=np.array([1, 1, 2, 2], dtype=np.int32), mode="replicate")
            expected_outputs = np.array(
                [
                    [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0],
                    [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0],
                    [4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0],
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)

        def test_constant_general():
            x_val = np.arange(12, dtype=np.float32).reshape([2, 2, 3])
            pad = np.array([[1, 1], [2, 2], [1, 1]], dtype=np.int32)
            v = mb.pad(x=x_val, pad=pad.reshape(-1), mode="constant", constant_val=0.0)
            expected_outputs = np.pad(x_val, pad, mode="constant")
            np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)

        # Test different modes
        test_constant_mode()
        test_reflect_mode()
        test_replicate_mode()
        test_constant_general()

    @staticmethod
    def test_value_inference_with_symbolic_padding():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(get_new_symbol(), get_new_symbol(), 1, 3), dtype=types.fp32)
            ]
        )
        def prog(x):
            paddings = mb.shape(x=x)
            res = mb.pad(x=np.random.rand(1, 1), pad=paddings)
            shape = res.shape
            assert is_symbolic(shape[0])
            assert shape[1] == 5
            return res

    @staticmethod
    def test_error_out_with_dynamic_paddings_with_invaid_shape():
        with pytest.raises(
            ValueError, match="Non-constant 'pad' must have shape \(8,\). Got \(4,\)"
        ):

            @mb.program(
                input_specs=[
                    mb.TensorSpec(shape=(1, 1, 3, 4)),
                    mb.TensorSpec(shape=(2, 2), dtype=types.int32),
                ]
            )
            def prog(x, y):
                pad = mb.reshape(x=y, shape=[-1])
                res = mb.pad(x=x, pad=pad)


class TestRange1d:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = 15.0
        y = 5.0
        z = 2.0
        # Model inputs must have rank at least 1
        input_placeholders = {
            "x": mb.placeholder(shape=(1,)),
            "y": mb.placeholder(shape=(1,)),
            "z": mb.placeholder(shape=(1,)),
        }
        input_values = {"x": x, "y": y, "z": z}

        def build(x, y, z):
            return [
                mb.range_1d(start=mb.squeeze(x=y), end=15.0, step=2.0),
                mb.range_1d(start=mb.squeeze(x=y), end=15.0, step=mb.squeeze(x=z)),
                mb.range_1d(start=mb.squeeze(x=y), end=mb.squeeze(x=x), step=2.0),
                mb.range_1d(start=mb.squeeze(x=y), end=mb.squeeze(x=x), step=mb.squeeze(x=z)),
                mb.range_1d(start=5.0, end=15.0, step=mb.squeeze(x=z)),
                mb.range_1d(start=5.0, end=mb.squeeze(x=x), step=2.0),
                mb.range_1d(start=5.0, end=mb.squeeze(x=x), step=mb.squeeze(x=z)),
            ]

        expected_output_types = [
            (UNK_SYM, types.fp32),
            (UNK_SYM, types.fp32),
            (UNK_SYM, types.fp32),
            (UNK_SYM, types.fp32),
            (UNK_SYM, types.fp32),
            (UNK_SYM, types.fp32),
            (UNK_SYM, types.fp32),
        ]

        expected_outputs = [
            np.array([5, 7, 9, 11, 13], dtype=np.float32),
            np.array([5, 7, 9, 11, 13], dtype=np.float32),
            np.array([5, 7, 9, 11, 13], dtype=np.float32),
            np.array([5, 7, 9, 11, 13], dtype=np.float32),
            np.array([5, 7, 9, 11, 13], dtype=np.float32),
            np.array([5, 7, 9, 11, 13], dtype=np.float32),
            np.array([5, 7, 9, 11, 13], dtype=np.float32),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_large_array(self, compute_unit, backend):
        input_placeholders = {
            "x": mb.placeholder(shape=(1,)),  # dummpy input
        }
        input_values = {"x": 0.5}

        def build(x):
            return [mb.range_1d(start=0.0, end=2000000.0, step=1.0)]

        expected_output_types = [(2000000, types.fp32)]

        expected_outputs = [
            np.arange(0.0, 2000000.0, 1.0),
        ]

        mlmodel = run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

        # verify that the range_1d op is not const folded
        prog = mlmodel._mil_program
        ops = get_op_types_in_program(prog)
        assert ops == ["range_1d", "identity"]

    @ssa_fn
    def test_builder_eval(self):
        v = mb.range_1d(start=5, end=15, step=2)
        np.testing.assert_allclose(np.arange(5, 15, 2), v.val, atol=1e-04, rtol=1e-05)


class TestTile:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x.shape)}

        input_values = {"x": x}

        def build(x):
            return [
                mb.tile(x=x, reps=(1, 1)),
                mb.tile(x=x, reps=(2, 1)),
            ]

        expected_output_types = [
            (2, 3, types.fp32),
            (4, 3, types.fp32),
        ]

        expected_outputs = [
            x,
            np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]], dtype=np.float32),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        v = mb.tile(x=x, reps=(1, 2))
        np.testing.assert_allclose(np.tile(x, reps=(1, 2)), v.val, atol=1e-04, rtol=1e-05)


class TestDynamicTile:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        rep1 = np.array([1, 1]).astype(np.int32)
        rep2 = np.array([2, 1]).astype(np.int32)
        rep3 = np.array([2, 3]).astype(np.int32)
        input_placeholders = {
            "x": mb.placeholder(shape=x.shape),
            "reps1": mb.placeholder(shape=rep1.shape, dtype=types.int32),
            "reps2": mb.placeholder(shape=rep2.shape, dtype=types.int32),
            "reps3": mb.placeholder(shape=rep3.shape, dtype=types.int32),
        }

        input_values = {"x": x, "reps1": rep1, "reps2": rep2, "reps3": rep3}

        def build(x, reps1, reps2, reps3):
            return [
                mb.tile(x=x, reps=reps1),
                mb.tile(x=x, reps=reps2),
                mb.tile(x=x, reps=reps3),
            ]

        expected_output_types = [
            (UNK_SYM, UNK_SYM, types.fp32),
            (UNK_SYM, UNK_SYM, types.fp32),
            (UNK_SYM, UNK_SYM, types.fp32),
        ]

        expected_outputs = [
            x,
            np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]], dtype=np.float32),
            np.array(
                [
                    [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    [4, 5, 6, 4, 5, 6, 4, 5, 6],
                    [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    [4, 5, 6, 4, 5, 6, 4, 5, 6],
                ],
                dtype=np.float32,
            ),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestTopK:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        val = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=val.shape)}
        input_values = {"x": val}

        def build(x):
            return mb.topk(x=x, k=2, axis=1)

        expected_output_types = [
            (2, 2, types.fp32),
            (2, 2, types.int32),
        ]
        expected_outputs = [
            np.array([[2.0, -1.0], [6.0, 4.0]], dtype=np.float32),
            np.array([[1, 0], [2, 0]], dtype=np.float32),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

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

        val = np.array([[-1.0, 7.0, -3.0], [4.0, -5.0, 8.0]], dtype=np.float32)
        res_values, res_indices = mb.topk(x=val, k=1, axis=0)
        ref_values, ref_indices = np_topk(x=val, k=1, axis=0)
        np.testing.assert_allclose(ref_values, res_values.val, atol=1e-04, rtol=1e-05)
        np.testing.assert_allclose(ref_indices, res_indices.val, atol=1e-04, rtol=1e-05)
        res_values, res_indices = mb.topk(x=val, k=2, axis=-1, ascending=True)
        ref_values, ref_indices = np_topk(x=val, k=2, axis=-1, ascending=True)
        np.testing.assert_allclose(ref_values, res_values.val, atol=1e-04, rtol=1e-05)
        np.testing.assert_allclose(ref_indices, res_indices.val, atol=1e-04, rtol=1e-05)

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_symbolic(self, compute_unit, backend):
        s0 = get_new_symbol()

        val = np.array([[1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=(s0, 3))}
        input_values = {"x": val}

        def build(x):
            return mb.topk(x=x, k=2, axis=-1, ascending=True)

        expected_output_types = [
            (s0, 2, types.fp32),
            (s0, 2, types.int32),
        ]
        expected_outputs = [
            np.array([[-3.0, 1.0], [-5.0, 4.0]], dtype=np.float32),
            np.array([[2, 0], [1, 0]], dtype=np.float32),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestFlatten2d:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        t = np.array([[[1, 2, 3], [4, 5, 6]], [[-1, -2, -3], [-4, -5, -6]]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return [mb.flatten2d(x=x)]

        expected_output_types = [
            (2, 6, types.fp32),
        ]
        expected_outputs = [
            np.array([[1, 2, 3, 4, 5, 6], [-1, -2, -3, -4, -5, -6]], dtype=np.float32),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, rank, axis, backend",
        itertools.product(
            compute_units,
            range(1, 6),
            range(-5, 6),
            backends,
        ),
    )
    def test_builder_to_backend_stress(self, compute_unit, rank, axis, backend):
        if axis < -rank or axis >= rank + 1:
            return

        shape = np.random.randint(low=2, high=6, size=rank)
        t = np.random.random(shape)

        input_placeholders = {"x": mb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return [mb.flatten2d(x=x, axis=axis)]

        np_axis = axis + rank if axis < 0 else axis
        pl, pr = 1, 1
        for i in range(0, np_axis):
            pl *= shape[i]
        for i in range(np_axis, len(shape)):
            pr *= shape[i]

        new_shape = [pl, pr]
        ref = t.reshape(new_shape)

        expected_outputs = [ref]
        expected_output_types = [
            tuple(list(ref.shape) + [types.fp32]),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        t = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
        f = mb.flatten2d(x=t)
        expected_f = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
        np.testing.assert_allclose(expected_f, f.val, atol=1e-04, rtol=1e-05)

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_symbolic(self, compute_unit, backend):
        s0 = get_new_symbol()

        input_placeholders = {
            "x": mb.placeholder(shape=(s0, 4, 5, 6)),
        }

        def build(x):
            return [mb.flatten2d(x=x)]

        input = np.random.rand(10, 4, 5, 6)
        output = input.reshape(10, -1)

        expected_output_types = (s0, 120, types.fp32)
        expected_outputs = [output]

        input_values = {"x": input}
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            inputs=construct_inputs_from_placeholders(input_placeholders, 10)
            if backend.backend == "mlprogram"
            else None,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestShape:
    @pytest.mark.parametrize(
        "compute_unit, backend, input_type",
        itertools.product(compute_units, backends, ["int32", "float32"]),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, input_type):
        np_type = np.int32 if input_type == "int32" else np.float32
        mb_type = types.int32 if input_type == "int32" else types.fp32

        t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np_type)
        input_placeholders = {"x": mb.placeholder(shape=t.shape, dtype=mb_type)}
        input_values = {"x": t}

        def build(x):
            return mb.shape(x=x)

        expected_output_types = (2, types.int32)
        expected_outputs = [
            np.array([2, 3], dtype=np.int32),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        t = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
        f = mb.shape(x=t)
        expected_f = np.array([1, 2, 3], dtype=np.float32)
        np.testing.assert_allclose(expected_f, f.val, atol=1e-04, rtol=1e-05)

    @pytest.mark.parametrize(
        "compute_unit, backend, input_type",
        itertools.product(compute_units, backends, ["int32", "float32"]),
    )
    def test_builder_to_backend_symbolic(self, compute_unit, backend, input_type):
        np_type = np.int32 if input_type == "int32" else np.float32
        mb_type = types.int32 if input_type == "int32" else types.fp32

        s0 = get_new_symbol()

        input_placeholders = {
            "x": mb.placeholder(shape=(s0, 4, 5, 6), dtype=mb_type),
        }

        def build(x):
            return [mb.shape(x=x)]

        input = np.random.rand(10, 4, 5, 6)
        input = input.astype(np_type)
        output = np.array([10, 4, 5, 6], dtype=np.int32)

        expected_output_types = (4, types.int32)
        expected_outputs = [output]

        input_values = {"x": input}
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            inputs=construct_inputs_from_placeholders(input_placeholders, 10)
            if backend.backend == "mlprogram"
            else None,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestIdentity:
    @pytest.mark.parametrize(
        "compute_unit, backend, input_type",
        itertools.product(compute_units, backends, ["int32", "float32"]),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, input_type):
        np_type = np.int32 if input_type == "int32" else np.float32
        mb_type = types.int32 if input_type == "int32" else types.fp32

        t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np_type)
        input_placeholders = {"x": mb.placeholder(shape=t.shape, dtype=mb_type)}
        input_values = {"x": t}

        def build(x):
            return mb.identity(x=x)

        expected_output_types = [(2, 3, mb_type)]
        expected_outputs = [
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np_type),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        t = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
        f = mb.identity(x=t)
        expected_f = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
        np.testing.assert_allclose(expected_f, f.val, atol=1e-04, rtol=1e-05)

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_symbolic(self, compute_unit, backend):
        input_placeholders = {
            "x": mb.placeholder(shape=(10, 4, 5, 6)),
        }

        def build(x):
            return [mb.identity(x=x)]

        input = np.random.rand(10, 4, 5, 6)
        output = input

        expected_output_types = [(10, 4, 5, 6, types.fp32)]
        expected_outputs = [output]

        input_values = {"x": input}
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestArgSort:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        val = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=val.shape)}
        input_values = {"x": val}

        def build(x):
            return [mb.argsort(x=x), mb.argsort(x=x, axis=0, ascending=True)]

        expected_output_types = [
            (2, 3, types.int32),
            (2, 3, types.int32),
        ]
        expected_outputs = [
            np.array([[1, 0, 2], [2, 0, 1]], dtype=np.int32),
            np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        x_val = random_gen(shape=(1, 3, 2, 2), rand_min=-100, rand_max=100)
        res = mb.argsort(x=x_val, axis=-3)
        # The default np argsort mode is ascending, which is opposite to MIL's argsort op.
        np.testing.assert_allclose(np.argsort(-x_val, axis=-3), res.val, atol=1e-04, rtol=1e-05)


class TestConcat:
    @pytest.mark.parametrize(
        "compute_unit, backend, axis",
        itertools.product(
            compute_units,
            backends,
            [0, 1],
        ),
    )
    def test_builder_to_backend_numerical(self, compute_unit, backend, axis):
        def build(x1, x2):
            return mb.concat(values=[x1, x2], axis=axis)

        val1 = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
        val2 = -val1
        input_placeholders = {
            "x1": mb.placeholder(shape=val1.shape),
            "x2": mb.placeholder(shape=val2.shape),
        }
        input_values = {"x1": val1, "x2": val2}
        expected_res = np.concatenate([val1, val2], axis=axis)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types=[expected_res.shape + (types.fp32,)],
            expected_outputs=expected_res,
            compute_unit=compute_unit,
            backend=backend,
        )

    def test_builder_eval_different_dtypes_error_out(self):
        """If the input to the concat op has different dtypes, it will error out."""
        with pytest.raises(
            ValueError,
            match="Tensors in 'values' of the concat op \(concat_0\) should share the same data type",
        ):

            @mb.program(
                input_specs=[
                    mb.TensorSpec(shape=(2, 3), dtype=types.fp32),
                    mb.TensorSpec(shape=(2, 3), dtype=types.int32),
                ]
            )
            def prog(x1, x2):
                return mb.concat(values=[x1, x2], axis=0)
