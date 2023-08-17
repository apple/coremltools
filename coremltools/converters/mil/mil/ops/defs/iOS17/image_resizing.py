#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import Operation, get_new_symbol, types
from coremltools.converters.mil.mil.input_type import DefaultInputs, InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS16.image_resizing import (
    crop_resize as _crop_resize_iOS16,
)
from coremltools.converters.mil.mil.ops.defs.iOS16.image_resizing import resample as _resample_iOS16
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


@register_op(opset_version=_IOS17_TARGET)
class resample(_resample_iOS16):
    """
    Resample the input image tensor ``x`` at the ``coordinates``.

    The major difference between this version and the iOS 16 :py:class:`~.iOS16.image_resizing.resample`
    is that `coordinates` supports int8, uint8, int16, uint16.
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        coordinates=TensorInputType(type_domain="U"),
        sampling_mode=TensorInputType(const=True, type_domain=types.str),
        padding_mode=TensorInputType(const=True, type_domain=types.str),
        padding_value=TensorInputType(const=True, type_domain="T"),
        coordinates_mode=TensorInputType(const=True, type_domain=types.str),
        align_corners=TensorInputType(const=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (
            types.int8,
            types.uint8,
            types.int16,
            types.uint16,
            types.int32,
            types.fp16,
            types.fp32,
        ),
    }


@register_op(opset_version=_IOS17_TARGET)
class resize(Operation):
    """
    Resizes the input tensor ``x`` by choosing the right-most ``resized_dims`` dimensions from
    the input shape ``shape``, and by choosing the rest from ``x`` 's shape.

    This iOS17 ``resize`` is a superset of iOS 15 :py:class:`~.iOS15.image_resizing.resize_bilinear`
    and :py:class:`~.iOS15.image_resizing.resize_nearest_neighbor`.
    The main benefit is that this resize op allows a use-case in dynamic tensor shapes where
    a tensor needs to be resized to a dynamic shape as specified by another tensor.

    To illustrate how output shape is inferred, the following are two examples:

    - Example #1::

        x.shape: [1, 2, 3, 4]
        shape: [1, 6, 8]
        resized_dims: 2
        The output's shape will be [1, 2, 6, 8]

    - Example #2::

        x.shape: [1, 2, 3, is0]
        shape: [1, 0, 0]
        resized_dims: 2
        The output's shape will be [1, 2, 3, is0]

    Parameters
    ----------
    x: tensor<[...], T> (Required)

    shape: tensor<[K], U> (Required)
        * Restriction: ``size(shape)`` <= ``rank(x)``.
        * If ``shape[i]==0``, the dimension in the output tensor will instead be inferred from the
          corresponding element of ``x.shape()``.  Note this might not be ``x.shape()[i]``, as ``size(shape)``,
          ``resized_dims``, and ``size(x)`` may all be different sizes.

    resized_dims: const tensor<[], uint32> (Required)
        * Restriction: ``resized_dims`` <= ``size(shape)``.

    interpolation_mode: const<str> (Optional, default="LINEAR")
        * Available mode: ``LINEAR``, ``NEAREST_NEIGHBOR``.

    sampling_mode: const<str> (Optional, default="DEFAULT")
        * Available mode: ``DEFAULT``, ``STRICT_ALIGN_CORNERS``, ``ALIGN_CORNERS``,
        ``OFFSET_CORNERS``, ``UNALIGN_CORNERS``.
        * For details about different sampling modes, see iOS 15 :py:class:`~.iOS15.image_resizing.resize_bilinear`.

    Returns
    -------
    tensor<[...], T>

    Attributes
    ----------
    T: fp16, fp32, int32
    U: int32, int16, uint16, uint32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        shape=TensorInputType(type_domain="U"),
        resized_dims=TensorInputType(const=True, type_domain=types.uint32),
        interpolation_mode=TensorInputType(const=True, optional=True, type_domain=types.str),
        sampling_mode=TensorInputType(const=True, optional=True, type_domain=types.str),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
        "U": (types.int32, types.int16, types.uint16, types.uint32),
    }

    _VALID_INTERPOLATION_MODES = {"LINEAR", "NEAREST_NEIGHBOR"}
    _VALID_SAMPLING_MODE = {
        "DEFAULT",
        "STRICT_ALIGN_CORNERS",
        "ALIGN_CORNERS",
        "OFFSET_CORNERS",
        "UNALIGN_CORNERS",
    }

    def default_inputs(self):
        return DefaultInputs(
            interpolation_mode="LINEAR",
            sampling_mode="DEFAULT",
        )

    def _validate_input(self):
        if self.shape.val is not None:
            shape_element_num = self.shape.val.size
            if self.resized_dims.val > shape_element_num:
                raise ValueError(
                    f"The resized_dims ({self.resized_dims.val}) must <= shape's size ({shape_element_num})"
                )
            if shape_element_num > self.x.rank:
                raise ValueError(
                    f"The shape's size ({shape_element_num}) must <= x's rank ({self.x.rank})"
                )
        if self.shape.rank != 1:
            raise ValueError(f"The shape's rank must be 1, but got {self.shape.rank}")
        if self.interpolation_mode.val not in self._VALID_INTERPOLATION_MODES:
            raise ValueError(
                f"Invalid interpolation_mode {self.interpolation_mode.val}. Supported modes are: {self._VALID_INTERPOLATION_MODES}"
            )
        if self.sampling_mode.val not in self._VALID_SAMPLING_MODE:
            raise ValueError(
                f"Invalid sampling_mode {self.sampling_mode.val}. Supported modes are: {self._VALID_SAMPLING_MODE}"
            )

    def type_inference(self):
        self._validate_input()

        # The output tensor will have the same rank as the input tensor. The rightmost resized_dims
        # dimensions of the output_shape will be taken from the input "shape".
        ret_shape = list(self.x.shape)

        start_idx = self.shape.shape[0] - self.resized_dims.val
        for i in range(self.resized_dims.val):
            target_shape = (
                get_new_symbol() if self.shape.val is None else self.shape.val[start_idx + i]
            )
            if target_shape == 0:
                # The 0 in `shape` means inheriting from x's shape.
                target_shape = self.x.shape[self.x.rank - self.resized_dims.val + i]
            ret_shape[self.x.rank - self.resized_dims.val + i] = target_shape

        return types.tensor(self.x.dtype, ret_shape)
