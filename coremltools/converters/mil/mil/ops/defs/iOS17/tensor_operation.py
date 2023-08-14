#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import DefaultInputs, InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS15.tensor_operation import (
    non_maximum_suppression as _nms_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS16.tensor_operation import topk as _topk_iOS16
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET


@register_op(opset_version=_IOS17_TARGET)
class non_maximum_suppression(_nms_iOS15):
    """
    Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU).

    NMS iteratively removes lower-scoring boxes which have an IoU greater than ``iou_threshold`` with
    another (higher-scoring) box.

    The major differences between this version and the iOS 15 :py:class:`~.iOS15.tensor_operation.non_maximum_suppression`
    are as follows:

       - The input parameter ``score_threshold`` has been removed.
       - The inputs ``boxes`` and ``scores`` are ordered with number of boxes in the last dimension.
       - The fourth output containing number of boxes for each batch has been removed.

    Parameters
    ----------

    boxes: tensor<[n, 4, B], T> (Required)
        * Box coordinates on which to perform NMS. The coordinates are expected in
          ``CENTER_SIZE_WIDTH_FIRST`` format ``(x, y, width, height)``, in which ``(x, y)`` is the center.
    scores: tensor<[n, K, B], T> (Required)
        * Scores for each one of the boxes. ``K`` is the number of classes.
    iou_threshold: const<T> (Required)
        * The intersection over union (IoU) threshold over which boxes are
          suppressed. NMS remove all overlapping boxes with ``IoU > iou_threshold``.
    max_boxes: const<i32> (Required)
        * Maximum number of boxes to select. If the number of surviving boxes are
          less, the output is padded up to this number.
    per_class_suppression: const<bool> (Optional)
        * Defaults to ``False``.
        * If ``True``, suppression is performed independently within boxes of each class.

    Returns
    -------
    tensor<[n, 4, max_boxes], T>
        * Coordinates of selected boxes.
    tensor<[n, K, max_boxes], T>
        * Scores of selected boxes.
    tensor<[n, max_boxes], i32>
        * Indices of selected boxes.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        boxes=TensorInputType(type_domain="T"),
        scores=TensorInputType(type_domain="T"),
        iou_threshold=TensorInputType(const=True, type_domain="T"),
        max_boxes=TensorInputType(const=True, type_domain=types.int32),
        per_class_suppression=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    def type_inference(self):
        boxes_dtype = self.boxes.dtype
        scores_dtype = self.scores.dtype
        n_batch, n_score_class, _ = self.scores.shape
        max_boxes = self.max_boxes.val

        return (
            types.tensor(boxes_dtype, (n_batch, 4, max_boxes)),
            types.tensor(scores_dtype, (n_batch, n_score_class, max_boxes)),
            types.tensor(types.int32, (n_batch, max_boxes)),
        )


@register_op(opset_version=_IOS17_TARGET)
class topk(_topk_iOS16):
    """
    A version of ``topk`` for iOS 17+. The differences between this version and the
    iOS 16 :py:class:`~.iOS16.tensor_operation.topk` are:
    - New data type support. The newly added data type is:
        - int8, uint8, int16, unint16 for ``x`` and output.
        - int8, int16 for ``k``.
    - Validation restrictions on the optional ``indices`` output: must be either uint16 or int32. Also
      a new input parameter ``output_indices_dtype`` is added to set the dtype of output ``indices``.

    Parameters
    ----------
    x: <\*?, T> (Required)
        * Input tensor.
    k: const<K> (Optional)
        * Defaults to ``1``.
        * Number of values/indices to be computed along each axis.
        * Set to ``-1`` to select all elements.
    axis: const<i32> (Optional)
        * Defaults to ``-1`` (last dimension).
        * Axis to perform the operation.
    ascending: const<bool> (Optional)
        * Defaults to ``False``, sort in descending order.
        * ``True`` to sort in ascending order.
    sort: const<bool> (Optional)
        * Defaults to ``True``.
        * If ``True``, ``top-k`` elements are themselves sorted.
          Otherwise, no particular ordering is guaranteed.
    return_indices: const<bool> (Optional)
        * Defaults to ``True``.
        * If ``True``, returns both values and indices. Otherwise, returns only the ``top-k`` values.
    output_indices_dtype: const<str> (Optional, default="int32")
        * It can only be set when ``return_indices`` is ``True``.
        * This parameter can take ``"int32"`` or ``"uint16"`` as values.

    Returns
    -------
    tensor<\*?, T>
        * Values of top/bottom ``k`` elements.

    tensor<\*?, U>
        * Only returned when ``return_indices = True``
        * Indices of the top/bottom ``k`` elements along axis.
        * U is int32 or uint16 determined by ``output_indices_dtype`` (int32 by default).

    Attributes
    ----------
    T: fp16, fp32, int8, int16, int32, uint8, uint16
    K: int8, int16, int32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        k=TensorInputType(const=True, optional=True, type_domain="K"),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
        ascending=TensorInputType(const=True, optional=True, type_domain=types.bool),
        sort=TensorInputType(const=True, optional=True, type_domain=types.bool),
        return_indices=TensorInputType(const=True, optional=True, type_domain=types.bool),
        output_indices_dtype=TensorInputType(const=True, optional=True, type_domain=types.str),
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int8,
            types.int16,
            types.int32,
            types.uint8,
            types.uint16,
        ),
        "K": (types.int8, types.int16, types.int32),
    }

    _ALLOWED_OUTPUT_INDICES_DTYPES = {"int32", "uint16"}

    def default_inputs(self):
        parent_default_inputs = super().default_inputs()
        # If return_indices is not set, it is default to True.
        # output_indices_dtype can only be set when return_indices = True
        if self.return_indices is None or self.return_indices.val:
            return parent_default_inputs + DefaultInputs(output_indices_dtype="int32")
        return parent_default_inputs

    def type_inference(self):
        if not self.return_indices.val and self.output_indices_dtype is not None:
            raise ValueError(
                'In iOS17 topk op, "output_indices_dtype" can only be set when "return_indices=True".'
            )

        if self.return_indices.val:
            if self.output_indices_dtype.val not in self._ALLOWED_OUTPUT_INDICES_DTYPES:
                raise ValueError(
                    f'"topk" op invalid output_indices_dtype: "{self.output_indices_dtype.val}". '
                    f"Valid options are: {self._ALLOWED_OUTPUT_INDICES_DTYPES}"
                )

            value_type, indices_type = super().type_inference()
            indices_type = types.tensor(
                types.string_to_builtin(self.output_indices_dtype.val), indices_type.get_shape()
            )
            return value_type, indices_type
        else:
            return super().type_inference()
