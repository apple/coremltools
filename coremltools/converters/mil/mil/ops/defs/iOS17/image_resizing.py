#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import DefaultInputs, InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS16.image_resizing import (
    crop_resize as _crop_resize_iOS16,
)
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET


@register_op(opset_version=_IOS17_TARGET)
class crop_resize(_crop_resize_iOS16):
    """
    The major differences between this version and the iOS 16 :py:class:`~.iOS16.image_resizing.crop_resize`
    are as follows:

    - The input ``ROI`` is replaced by ``boxes`` and ``box_indices``.
    - The dtype domain of input ``x``, ``boxes``, and ``box_indices`` are independent.
    - The output no longer has the ``B`` dim. The output is ``[N, C, target_height, target_width]``
      rather than the ``[N, B, C, target_height, target_width]`` in iOS 16.

    Parameters
    ----------
    x: tensor<[B, C, H, W], T> (Required)
        * The input, from which patches (regions of interest) are extracted
          and resized using bilinear interpolation.
        * Rank ``4``.

    boxes: tensor<[N, 4], BOX_T> (Required)
        * Coordinates of ``N`` boxes.
        * The convention to express coordinates depends on the value of ``box_coordinate_mode``.
        * If ``normalized_coordinates`` is True, only fp16 and fp32 dtypes are allowed.

    box_indices: tensor<[N], BOX_INDEX_T> (Optional)
        * Default is ``arange(N)``, or ``[0, 1, ..., N-1]``.
        * If ``box_indices[i]=j``, this means that ``boxes[i]`` will be applied to the ``j``-th image.
          Therefore, it is invalid for ``box_indices[i]`` to be greater than ``B``.

    target_height: const<i32> (Optional, Default=1)
        * Target height for resizing each patch.

    target_width: const<i32> (Optional, Default=1)
        * Target width for resizing each patch.

    normalized_coordinates : const<bool> (Optional, default=False)
        * If ``True``, the bounding box coordinates must be in the
          interval ``[0, 1]``. Scaling is based on the input spatial
          dimensions: ``(H_in - 1)`` for height and ``(W_in - 1)`` for width.
        * If ``False``, the bounding box coordinates must be in the interval
          ``[0, H_in - 1]`` for height dimensions and ``[0, W_in - 1]`` for
          width dimensions.

    spatial_scale : const<fp32> (Optional, default=1.0)
        * Additional spatial scale that multiplies the bounding box coordinates.
        * You would use this to implement the RoI Align layer, which typically
          uses unnormalized RoI coordinates along with a spatial scale that is
          less than or equal to ``1``.

    box_coordinate_mode: const<str> (Optional, default="CORNERS_HEIGHT_FIRST")
        * Specifies the convention for specifying the four bounding box
          coordinates for an image of size ``(Height, Width)``. The ``(0,0)``
          coordinate corresponds to the top-left corner of the image.
        * This parameter can take one of four values:

          ``"CORNERS_HEIGHT_FIRST"``: ``[h_start, w_start, h_end, w_end]``

          ``"CORNERS_WIDTH_FIRST"``: ``[w_start, h_start, w_end, h_end]``

          ``"CENTER_SIZE_HEIGHT_FIRST"``: ``[h_center, w_center, box_height, box_width]``

          ``"CENTER_SIZE_WIDTH_FIRST"``: ``[w_center, h_center, box_width, box_height]``

    sampling_mode : const<str> (Optional, default="DEFAULT")
        * This parameter can take ``"STRICT_ALIGN_CORNERS"``,
          ``"ALIGN_CORNERS"``, ``"DEFAULT"``, ``"OFFSET_CORNERS"`` or
          ``UNALIGN_CORNERS`` as values.
        * This is the same convention used by the :py:class:`~.iOS15.image_resizing.resize_bilinear` op.

    pad_value : const<T> (Optional, default=0.0)
        * If the box indexes go beyond the input boundary, the input image is padded with ``pad_value``.
        * Defaults to ``0``.
        * It is the same as ``extrapolation_value`` in `tf.image.crop_and_resize <https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize>`_.

    Returns
    -------
    tensor<[N, C, target_height, target_width], T>

    Attributes
    ----------
    T: fp16, fp32
    BOX_T: fp16, fp32, uint16
    BOX_INDEX_T: uint16, int32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        boxes=TensorInputType(type_domain="BOX_T"),
        box_indices=TensorInputType(optional=True, type_domain="BOX_INDEX_T"),
        target_height=TensorInputType(const=True, optional=True, type_domain=types.int32),
        target_width=TensorInputType(const=True, optional=True, type_domain=types.int32),
        normalized_coordinates=TensorInputType(const=True, optional=True, type_domain=types.bool),
        spatial_scale=TensorInputType(const=True, optional=True, type_domain=types.fp32),
        box_coordinate_mode=TensorInputType(const=True, optional=True, type_domain=types.str),
        sampling_mode=TensorInputType(const=True, optional=True, type_domain=types.str),
        pad_value=TensorInputType(const=True, optional=True, type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "BOX_T": (types.fp16, types.fp32, types.uint16),
        "BOX_INDEX_T": (types.uint16, types.int32),
    }

    def default_inputs(self):
        if self.box_indices is None and self.boxes.shape[0] > self.x.shape[0]:
            # The default box indices is [0, 1, ..., N-1], which is out-of-range for N>B.
            raise ValueError(
                f'"crop_resize" op: N dimension of "boxes" ({self.boxes.shape[0]}) '
                f'should not be greater than the B dimension of "x" ({self.x.shape[0]}) '
                f'when "box_indices" is not specified, otherwise "box_indices" would '
                f'point outside of "x" bounds.'
            )

        return DefaultInputs(
            box_indices=list(range(self.boxes.shape[0])),
            target_height=1,
            target_width=1,
            normalized_coordinates=False,
            spatial_scale=1.0,
            box_coordinate_mode="CONRNERS_HEIGHT_FIRST",
            sampling_mode="DEFAULT",
            pad_value=0.0,
        )

    def _validate_input(self):
        if self.x.rank != 4:
            raise ValueError(
                f'input to the "crop_resize" op must be of rank 4, but got {self.x.rank}'
            )
        if self.boxes.rank != 2 or self.boxes.shape[1] != 4:
            raise ValueError(
                f'"crop_resize" op: input "boxes" must has shape [N, 4], but got {self.boxes.shape}'
            )
        if self.box_indices.rank != 1 or self.box_indices.shape[0] != self.boxes.shape[0]:
            raise ValueError(
                f'"crop_resize" op: input "box_indices" must has shape [{self.boxes.shape[0]}], '
                f"but got {self.box_indices.shape}"
            )
        if self.box_indices.val is not None and np.any(self.box_indices.val >= self.x.shape[0]):
            raise ValueError(
                f'"crop_resize" op: input "box_indices" should not have values >= B dimension of x '
                f"({self.x.shape[0]}), but got {self.box_indices.val}"
            )
        if self.box_coordinate_mode.val not in self._VALID_BOX_COORDINATE_MODES:
            raise ValueError(
                f'"crop_resize" op: unrecognized box_coordinate_mode "{self.box_coordinate_mode.val}"'
            )
        if self.sampling_mode.val not in self._VALID_SAMPLING_MODES:
            raise ValueError(
                f'"crop_resize" op: unrecognized sampling mode "{self.sampling_mode.val}"'
            )
        if self.normalized_coordinates.val:
            if self.boxes.dtype not in {types.fp16, types.fp32}:
                raise ValueError(
                    f'"crop_resize" op: When normalized_coordinates is set, the '
                    f'"boxes" must have fp16 or fp32 dtype, but got '
                    f"{types.builtin_to_string(self.sampling_mode.val)}"
                )

    def type_inference(self):
        self._validate_input()
        # Output shape is [N, C, h_out, w_out].
        ret_shape = [
            self.boxes.shape[0],
            self.x.shape[1],
            self.target_height.val,
            self.target_width.val,
        ]
        return types.tensor(self.x.dtype, ret_shape)
