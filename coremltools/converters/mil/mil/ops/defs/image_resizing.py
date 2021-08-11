# -*- coding: utf-8 -*-
#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import numpy as np

from ._op_reqs import *
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from coremltools.converters.mil.mil import get_new_symbol


@register_op(doc_str="")
class affine(Operation):
    """
    Apply a linear affine transform to the input 2D image tensor. Value at the
    (x, y), i.e., (w, h) coordinate of the output, is computed by first computing
    the coordinates x’ and y’ with the following equation and then compute the
    value at the coordinate (x’,y’) in the input image using either bilinear or
    nearest neighbor interpolation. If the (x’, y’) point falls outside the input
    image, then padding information is used to compute the value.
    * x’ = a0 * x + a1 * y + a2
    * y’ = b0 * x + b1 * y + b2

    Parameters
    ----------
    x: tensor<[B, C, H1, W1], T>
        * Must be rank ``4``.
    transform_matrix: tensor<[D, 6], T>
        * Must be rank ``2``
        * D can be either B or 1.
            when D == B, for each batch, there is a separate transform matrix
            when D == 1, the same matrix is used for all input batches
            for each batch: [a0, a1, a2, b0, b1, b2]
    output_height: const<i32>
        * Target output height
    output_width: const<i32>
        * Target output width
    sampling_mode: const<str>
        * Allowed values: "bilinear"
    padding_mode: const<str>
        * Allowed values: "constant"
        * Note that following illustration is 1D case for brevity, the op only support 2D image input.
        * if ``padding_mode == "constant"``:
            the input image is assumed to be padded with the padding_value
            E.g., |1, 2, 3| -> |0, 0, 0, 1, 2, 3, 0, 0, 0|
    padding_value: const<T>
        * Currently non-zero values are not supported.
        * To be used only when ``padding_mode == "constant"``, ignored in other cases.
    coordinates_mode: const<str>
        * allowed values: "normalized_minus_one_to_one",
        * if ``coordinates_mode == "normalized_minus_one_to_one"``, in-image values are [-1, 1]
        * E.g., if ``coordinates_mode == "normalized_minus_one_to_one"``,
            the in range values are [-1, 1]. That is:
            * (-1, -1), i.e., (w=-1, h=-1), corresponds to the top-left pixel
            * (1, -1), i.e., (w=1, h=-1), corresponds to the top-right pixel
            * (-1, 1), i.e., (w=-1, h=1), corresponds to the bottom-left pixel
            * (1, 1), i.e., (w=1, h=1), corresponds to the bottom-right pixel
    align_corners: const<bool>
        * Currently align_corners=False is not supported.
        * To be used only when ``coordinates_mode != unnormalized``, ignored otherwise.
        * if ``align_corners == True``, the extrema coordinates are corresponding
            to the center of the first and last corner pixels.
        * if ``align_corners == False``, the extrema coordinates are corresponding
            to the edge of the first and last corner pixels.

    Returns
    -------
    tensor<[B, C, output_height, output_width], T>

    Attributes
    ----------
    T: fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        transform_matrix=TensorInputType(),
        output_height=IntInputType(const=True),
        output_width=IntInputType(const=True),
        sampling_mode=StringInputType(const=True),
        padding_mode=StringInputType(const=True),
        padding_value=FloatInputType(const=True),
        coordinates_mode=StringInputType(const=True),
        align_corners=BoolInputType(const=True),
    )

    def __init__(self, **kwargs):
        super(affine, self).__init__(**kwargs)

    def type_inference(self):
        if self.x.rank != 4:
            raise ValueError(
                'input "x" to the "affine" op must be a rank 4 tensor. '
                "Got rank {} tensor of shape {}".format(
                    self.x.rank, self.x.shape
                )
            )
        if self.transform_matrix.rank != 2:
            raise ValueError(
                'input "transform_matrix" to the "affine" op must be a rank 2 tensor. '
                "Got rank {} tensor of shape {}".format(
                   self.transform_matrix.rank,  self.transform_matrix.shape
                )
            )
        if self.sampling_mode.val.lower() != "bilinear":
            raise NotImplementedError(
                'input "sampling_mode" to the "affine" not implemented. '
                'Got "{}"'.format(self.sampling_mode.val)
            )
        if self.coordinates_mode.val.lower() != "normalized_minus_one_to_one":
            raise NotImplementedError(
                'input "coordinates_mode" to the "affine" not implemented. '
                'Got "{}"'.format(self.coordinates_mode.val)
            )
        if self.padding_mode.val.lower() != "constant" or self.padding_value.val != 0.0:
            raise NotImplementedError(
                'input "padding_mode" to the "affine" not implemented. '
                'Got "{}" with "padding_value={}"'.format(
                    self.padding_mode.val, self.padding_value.val
                )
            )

        input_shape = self.x.shape
        transform_matrix_shape = self.transform_matrix.shape
        if (
            not is_symbolic(transform_matrix_shape[-1])
            and transform_matrix_shape[-1] != 6
        ):
            raise ValueError(
                'input "transform_matrix" to the "affine" op last dimension must be 6 '
                "[a0, a1, a2, b0, b1, b2], "
                "Got {} for last dimension".format(transform_matrix_shape[-1])
            )

        ret_shape = list(input_shape)
        ret_shape[2] = self.output_height.val
        ret_shape[3] = self.output_width.val
        return types.tensor(self.x.dtype, tuple(ret_shape))


@register_op(doc_str="TODO")
class upsample_nearest_neighbor(Operation):
    """
    Upsample the spatial dimensions (last two dimensions) of the input
    by integer scale factors using nearest-neighbor interpolation.

    Parameters
    ----------
    x: tensor<[\*D, H1, W1],T>  (Required)
        * Must be at least rank ``3``.
    scale_factor_height: const<i32> or const<fp32> (Optional, default=1)
        * Scale factor for the height dimension (``axis=-2``).
        * Can be either an integer or fractional.
    scale_factor_width: const<i32> or const<fp32> (Optional, default=1)
        * Scale factor for the width dimension (``axis=-1``).
        * Can be either an integer or fractional.

    Returns
    -------
    tensor<[\*D, H2, W2],T>
        * Tensor with same type as the input.
        * ``H2`` = floor(``H1`` * ``scale_factor_height``).
        * ``W2`` = floor(``W1`` * ``scale_factor_width``).

    Attributes
    ----------
    T: fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        scale_factor_height=IntOrFloatInputType(const=True, optional=True),
        scale_factor_width=IntOrFloatInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            scale_factor_height=1,
            scale_factor_width=1,
        )

    def __init__(self, **kwargs):
        super(upsample_nearest_neighbor, self).__init__(**kwargs)

    def type_inference(self):
        if self.x.rank < 3:
            raise ValueError(
                'input to the "upsample_nearest_neighbor" op must have rank at least 3'
            )

        ret_shape = list(self.x.shape)
        ret_shape[-1] = np.floor(self.scale_factor_width.val * ret_shape[-1]) if not is_symbolic(ret_shape[-1]) else get_new_symbol()
        ret_shape[-2] = np.floor(self.scale_factor_height.val * ret_shape[-2]) if not is_symbolic(ret_shape[-2]) else get_new_symbol()
        return types.tensor(self.x.dtype, ret_shape)


@register_op(doc_str="")
class resample(Operation):
    """
    Resample the input image tensor ``x``, at the ``coordinates``.
    input. Since the coordinates may not correspond to exact pixels in the
    input image, this would require "resampling". sampling_mode determines
    the algorithm used for resampling and computing the values.

    Parameters
    ----------
    x: tensor<[B, C, H1, W1], T>
        * Must be rank ``4``.
    coordinates: tensor<[B, H2, W2, 2], U>
        * Must be rank ``4``.
        * Coordinates are provided in the order (x, y), i.e., (w, h).
        * Value of each output location output[b, c, h, w] is calculated by
          sampling, from the input image x[b, c, :, :], the pixel at the (x, y)
          location corresponding to the length-2 vector: coordinates[b, h, w, :]
        * Coordinate (normalized or unnormalized) should be specified according
          to ``coordinates_mode``
    sampling_mode: const<str>
        * Allowed values: "bilinear" , "nearest"
    padding_mode: const<str>
        * Allowed values: "constant", "border", "reflection", "symmetric"
        * Note that following illustration is 1D case for brevity, the op only support 2D image input.
        * if ``padding_mode == "constant"``:
            the input image is assumed to be padded with the padding_value
            E.g., |1, 2, 3| -> |0, 0, 0, 1, 2, 3, 0, 0, 0|
        * if ``padding_mode == "border"``:
            the input image is assumed to be padded with the values replicated
            from the values at the edge. This is also referred to as the
            "clamped" or "replication" mode, since the padded values are
            clamped to the border values.
            E.g., |1, 2, 3| -> |1, 1, 1, 1, 2, 3, 3, 3, 3|
        * if ``padding_mode == "reflection"``:
            the border values are reflected, *not* including the values at the edge/border
            E.g., |1, 2, 3| -> |2, 3, 2, 1, 2, 3, 2, 1, 2|
        * if ``padding_mode == "symmetric"``:
            values are reflected, including the border/edge values
            E.g., |1, 2, 3| -> |3, 2, 1 , 1, 2, 3, 3, 2, 1|
    padding_value: const<T>
        * To be used only when ``padding_mode == "constant"``, ignored in other cases.
    coordinates_mode: const<str>
        * allowed values: "unnormalized", "normalized_minus_one_to_one",
                          "normalized_zero_to_one"
        * if ``coordinates_mode == "unnormalized"``, the coordinates input values
            are interpreted to be in range [0, W - 1] / [0, H - 1] corresponds to in-image point
        * if ``coordinates_mode == "normalized_minus_one_to_one"``, in-image values are [-1, 1]
        * if ``coordinates_mode == "normalized_zero_to_one"``, in-image values are [0, 1]
        * E.g., if ``coordinates_mode == "normalized_minus_one_to_one"``,
            the in range values are [-1, 1]. That is:
            * (-1, -1), i.e., (w=-1, h=-1), corresponds to the top-left pixel
            * (1, -1), i.e., (w=1, h=-1), corresponds to the top-right pixel
            * (-1, 1), i.e., (w=-1, h=1), corresponds to the bottom-left pixel
            * (1, 1), i.e., (w=1, h=1), corresponds to the bottom-right pixel
    align_corners: const<bool>
        * if ``align_corners == True``, the extrema coordinates are corresponding
            to the center of the first and last corner pixels.
        * if ``align_corners == False``, the extrema coordinates are corresponding
            to the edge of the first and last corner pixels.

    Returns
    -------
    tensor<[B, C, H2, W2], T>

    Attributes
    ----------
    T: fp32
    U: fp32, i32, i64
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        coordinates=TensorInputType(),
        sampling_mode=StringInputType(const=True),
        padding_mode=StringInputType(const=True),
        padding_value=FloatInputType(const=True),
        coordinates_mode=StringInputType(const=True),
        align_corners=BoolInputType(const=True),
    )

    def __init__(self, **kwargs):
        super(resample, self).__init__(**kwargs)

    def type_inference(self):
        if self.x.rank != 4:
            raise ValueError(
                'input "x" to the "resample" op must be a rank 4 tensor. '
                "Got rank {} tensor of shape {}".format(
                    self.x.rank, self.x.shape
                )
            )
        if self.coordinates.rank != 4:
            raise ValueError(
                'input "coordinates" to the "resample" op must be a rank 4 tensor. '
                "Got rank {} tensor of shape {}".format(
                    self.coordinates.rank, self.coordinates.shape
                )
            )

        input_shape = self.x.shape
        coord_shape = self.coordinates.shape
        if (
            not is_symbolic(input_shape[0])
            and not is_symbolic(coord_shape[0])
            and input_shape[0] != coord_shape[0]
        ):
            raise ValueError(
                'input "x" and "coordinates" to the "resample" must agree on '
                "dimension of batch size: {} vs. {}".format(
                    input_shape[0], coord_shape[0]
                )
            )
        if not is_symbolic(coord_shape[-1]) and coord_shape[-1] != 2:
            raise ValueError(
                'input "coordinates" to the "resample" op last dimension must be 2. '
                "Got {} for last dimension".format(
                    coord_shape[-1]
                )
            )

        ret_shape = list(input_shape)
        ret_shape[2] = coord_shape[1]  # Output height
        ret_shape[3] = coord_shape[2]  # Output width
        return types.tensor(self.x.dtype, tuple(ret_shape))


@register_op(doc_str="TODO")
class resize_nearest_neighbor(Operation):
    """
    Resize the spatial (last two) dimensions to the specified target size
    using nearest neighbor interpolation. Although this op is similar to
    ``upsample_nearest_neighbor``, ``resize_nearest_neighbor`` works with
    a target size rather than with scale factors.

    Parameters
    ----------
    x: tensor<[*D, H1, W1], T> (Required)
        * Must be at least rank ``3``.
    target_size_height: const<int32> (Required)
        * Target spatial size for the height dimension (``axis=-2``).
    target_size_width: const<int32> (Required)
        * Target spatial size for the width dimension (``axis=-1``).

    Notes
    -----
    See ``resize_bilinear`` for examples.

    Returns
    -------
    tensor<[*D, H2, W2], T>
        * Tensor with same type as the input.
        * ``H2`` = ``target_size_height``.
        * ``W2`` = ``target_size_width``.

    Attributes
    ----------
    T: fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        target_size_height=IntInputType(const=True),
        target_size_width=IntInputType(const=True),
    )

    def __init__(self, **kwargs):
        super(resize_nearest_neighbor, self).__init__(**kwargs)

    def type_inference(self):
        if self.x.rank < 3:
            raise ValueError(
                'input to the "resize_nearest_neighbor" op must have rank at least 3'
            )

        ret_shape = list(self.x.shape)
        ret_shape[-1] = int(self.target_size_width.val)
        ret_shape[-2] = int(self.target_size_height.val)
        return types.tensor(self.x.dtype, ret_shape)


@register_op(doc_str="TODO")
class upsample_bilinear(Operation):
    """
    Upsample the spatial dimensions (last two dimensions) of the input
    by scale factors using bilinear interpolation.
    The upsample_bilinear operation in MIL corresponds to the recompute_scale_factor=True
    mode in the pyorch bilinear interpolation op. That is,
    the scale factor is recomputed by the output size.
    Note that when the scale_factor_height and scale_factor_width are floating point, this
    could result in a different scale factor due to rounding.

    Parameters
    ----------
    x: tensor<[\*D, H1, W1],T>  (Required)
        * Must be at least rank ``3``.
    scale_factor_height: const<T2> (Optional, default=1)
        * Scale factor for the height dimension (``axis=-2``).
    scale_factor_width: const<T2> (Optional, default=1)
        * Scale factor for the width dimension (``axis=-1``).
    align_corners: const<bool> (Optional, default=True)
        * This parameter determines how samples are chosen for bilinear
          interpolation. For details, see the Notes section.

    Notes
    -----
    To understand the ``align_corners`` parameter, consider the 1-D case.
    You need to sample a grid of pixels whose values are computed using linear
    interpolation. This parameter controls how the grid is sampled. If the
    input grid is ``[0, Xin-1]`` (corresponding to an input size of ``Xin``),
    and if the output size is ``Xout``, then the grid points are sampled in
    the following manner:

    .. sourcecode:: python

        # If align_corners == True:
        spacing = (Xin - 1) / (Xout - 1)
        grid_point[i] = min(Xin - 1, max(0, i*spacing)), for i=0,1,...,Xout-1

        # If align_corners == False:
        spacing = Xin / Xout
        grid_point[i] = min(Xin - 1, max(0, i*spacing + 0.5*spacing - 0.5)),
        ...   for i=0,1,...,Xout-1

    For example:

    .. sourcecode:: python

        Xin = 2
        input_interval = [0,1]

    Grid points:

    .. sourcecode:: python

        [0., 0.1, 0.5, 0.9, 1.] (Xout = 5, align_corners=False)
        [0., 0.25, 0.5, 0.75, 1.] (Xout = 5, align_corners=True)
        [0., 0., 0.33, 0.67, 1., 1.] (Xout = 6, align_corners=False)
        [0., 0.2, 0.4, 0.6, 0.8, 1.] (Xout = 6, align_corners=True)

    Note the following similarities:

    * ``align_corners=False`` is the same as
      ``tf.raw_ops.ResizeBilinear(align_corners=False, half_pixel_centers=True)``.

    * ``align_corners=True`` is the same as
      ``tf.raw_ops.ResizeBilinear(align_corners=True, half_pixel_centers=False)``.

    Returns
    -------
    tensor<[\*D, H2, W2],T>
        * Tensor with same type as the input.
        * ``H2`` = floor(``H1`` * ``scale_factor_height``).
        * ``W2`` = floor(``W1`` * ``scale_factor_width``).

    Attributes
    ----------
    T: fp32
    T2 : fp32 or int32
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        scale_factor_height=IntOrFloatInputType(const=True,
          optional=True),
        scale_factor_width=IntOrFloatInputType(const=True,
          optional=True),
        align_corners=BoolInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            scale_factor_height=1,
            scale_factor_width=1,
            align_corners=True,
            )

    def __init__(self, **kwargs):
        super(upsample_bilinear, self).__init__(**kwargs)

    def type_inference(self):
        if self.x.rank < 3:
            raise ValueError(
                'input to the "upsample_bilinear" op must have rank at least 3'
            )

        ret_shape = list(self.x.shape)
        ret_shape[-1] = np.floor(self.scale_factor_width.val * ret_shape[-1]) if not is_symbolic(ret_shape[-1]) else get_new_symbol()
        ret_shape[-2] = np.floor(self.scale_factor_height.val * ret_shape[-2]) if not is_symbolic(ret_shape[-2]) else get_new_symbol()
        return types.tensor(self.x.dtype, ret_shape)


@register_op(doc_str="TODO")
class resize_bilinear(Operation):
    """
    Resize the spatial (last two) dimensions to the specified target size
    using bilinear interpolation. Although this op is similar to
    ``upsample_bilinear``, ``resize_bilinear`` works with a target size
    rather than with scale factors.

    Parameters
    ----------
    x: tensor<[\*D, H1, W1],T> (Required)
        * Must be at least rank ``3``.
    target_size_height: const<int32> (Optional, default=1)
        * Target spatial size for the height dimension (``axis=-2``).
    target_size_width: const<int32> (Optional, default=1)
        * Target spatial size for the width dimension (``axis=-1``).
    sampling_mode: const<str> (Optional, default="DEFAULT")
        * This parameter can take ``"STRICT_ALIGN_CORNERS”``, ``"ALIGN_CORNERS"``,
          ``"DEFAULT"``, ``"OFFSET_CORNERS"`` or ``UNALIGN_CORNERS`` as values.
          For details, see the Notes section.

    Notes
    -----
    To understand the ``sampling_mode`` parameter, consider the 1-D case.
    You need to sample a grid of pixels whose values are computed using
    linear interpolation. This parameter controls how the grid is sampled.
    If the input grid is ``[0, Xin-1]`` (corresponding to an input size of
    ``Xin``), and if the output size is ``Xout``, then the grid points are
    sampled in the following manner:

    .. sourcecode:: python

        # "STRICT_ALIGN_CORNERS":
        spacing = (Xin - 1) / (Xout - 1)
        grid_point[i] = min(Xin-1, max(0, i*spacing)), for i=0,1,...,Xout-1

        # "ALIGN_CORNERS": Same as "STRICT_ALIGN_CORNERS" unless Xout=1,
        # in which case:
        grid_point[0] = (Xin-1) / 2, if Xout==1

        # "DEFAULT":
        spacing = (Xin - Xin/Xout) / (Xout - 1)
        grid_point[i] = min(Xin-1, max(0, i*spacing)), for i=0,1,...,Xout-1

        # "OFFSET_CORNERS":
        delta = max(1, Xin - 1) / Xout
        spacing = ((Xout - 1) * delta) / (Xout - 1)
        grid_point[i] = min(Xin-1, max(0, 0.5*delta + i*spacing)), for
        ...   i=0,1,...,Xout-1

        # "UNALIGN_CORNERS":
        spacing = Xin / Xout
        grid_point[i] = min(Xin - 1, max(0, i*spacing + 0.5*spacing - 0.5)), for i=0,1,...,Xout-1

    For example:

    .. sourcecode:: python

        Xin = 2
        input_interval = [0,1]

    Grid points:

    .. sourcecode:: python

        [0., 0.1, 0.5, 0.9, 1.] (Xout = 5, UNALIGN_CORNERS)
        [0., 0.25, 0.5, 0.75, 1.] (Xout = 5, "STRICT_ALIGN_CORNERS" / "ALIGN_CORNERS")
        [0., 0.4, 0.8, 1., 1.] (Xout = 5, "DEFAULT")
        [0.1, 0.3, 0.5, 0.7, 0.9] (Xout = 5, "OFFSET_CORNERS")

        [0., 0., 0.33, 0.67, 1., 1.] (Xout = 6, UNALIGN_CORNERS)
        [0., 0.2, 0.4, 0.6, 0.8, 1.] (Xout = 6, "STRICT_ALIGN_CORNERS" / "ALIGN_CORNERS")
        [0., 0.33, 0.67, 1., 1., 1.] (Xout = 6, "DEFAULT")
        [0.08, 0.25, 0.42, 0.58, 0.75, 0.92] (Xout = 6, "OFFSET_CORNERS")

    Note the following similarities:

        * ``"DEFAULT"`` is same as
          ``tf.raw_ops.ResizeBilinear(align_corners=False,
          half_pixel_centers=False)``.
        * ``"STRICT_ALIGN_CORNERS"`` is same as
          ``tf.raw_ops.ResizeBilinear(align_corners=True,
          half_pixel_centers=False)``.

    Returns
    -------
    tensor<[\*D, H2, W2],T>
        * Tensor with same type as the input.
        * ``H2`` = ``target_size_height``.
        * ``W2`` = ``target_size_width``.

    Attributes
    ----------
    T: fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        target_size_height=IntInputType(const=True, optional=True),
        target_size_width=IntInputType(const=True, optional=True),
        sampling_mode=StringInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            target_size_height=1,
            target_size_width=1,
            sampling_mode="DEFAULT",
            )

    def __init__(self, **kwargs):
        super(resize_bilinear, self).__init__(**kwargs)

    def type_inference(self):
        if self.x.rank < 3:
            raise ValueError(
                'input to the "resize_bilinear" op must have rank at least 3'
            )

        if self.sampling_mode.val not in {
            "STRICT_ALIGN_CORNERS",
            "ALIGN_CORNERS",
            "UNALIGN_CORNERS",
            "DEFAULT",
            "OFFSET_CORNERS",
        }:
            raise ValueError(
                '"resize_bilinear" op: unrecognized sampling mode "{}"'.format(
                    self.sampling_mode.val
                )
            )

        ret_shape = list(self.x.shape)
        ret_shape[-1] = self.target_size_width.val
        ret_shape[-2] = self.target_size_height.val
        return types.tensor(self.x.dtype, ret_shape)


@register_op(doc_str="TODO")
class crop_resize(Operation):
    """
    Resize the spatial dimensions (last two dimensions) of the first input
    according to the bounding boxes specified in the second input, using
    bilinear interpolation.

    Parameters
    ----------

    x: tensor<[B, C, H, W],T> (Required)
        * The input, from which patches (regions of interest) are extracted
          and resized using bilinear interpolation.
        * Rank ``4``.

    roi: tensor<[N,1,4,1,1], T> or tensor<[N,1,5,1,1], T> (Required)
        * Regions of interest, or coordinates of the boxes. The above input
          represents coordinates of ``N`` boxes.
        * The convention to express coordinates  depends on the value of the
          input ``box_coordinate_mode``.
        * Rank ``5``.
        * If ``tensor<[N,1,4,1,1], T>``: Resized images are computed for all
          ``B`` input images.
        * If ``tensor<[N,1,5,1,1], T>``: The first element from ``axis=-3``
          to be resized is an index. It must be within range ``[0, B)``.

    target_height: const<i32> (Optional, Default=1)
        * Target height for resizing each patch.

    target_width: const<i32> (Optional, Default=1)
        * Target width for resizing each patch.

    normalized_coordinates : const<bool> (Optional, default=False)
        * If true, the bounding box coordinates must be in the
          interval ``[0, 1]``. Scaling is based on the input spatial
          dimensions: ``(H_in - 1)`` for height and ``(W_in - 1)`` for width.
        * If false, the bounding box coordinates must be in the interval
          ``[0, H_in - 1]`` for height dimensions and ``[0, W_in - 1]`` for
          width dimensions.

    spatial_scale : const<fp32> (Optional, default=1.0)
        * Additional spatial scale that multiplies the bounding box coordinates.
          You would use this to implement the RoI Align layer, which typically
          uses unnormalized RoI coordinates along with a spatial scale that is
          less than or equal to 1.

    box_coordinate_mode: const<str> (Optional, default="CORNERS_HEIGHT_FIRST")
        * Specifies the convention for specifying the four bounding box
          coordinates for an image of size ``(Height, Width)``. The ``(0,0)``
          coordinate corresponds to the top-left corner of the image.
        * This parameter can take one of four values:

          "CORNERS_HEIGHT_FIRST": ``[h_start, w_start, h_end, w_end]``

          "CORNERS_WIDTH_FIRST": ``[w_start, h_start, w_end, h_end]``

          "CENTER_SIZE_HEIGHT_FIRST": ``[h_center, w_center, box_height, box_width]``

          "CENTER_SIZE_WIDTH_FIRST": ``[w_center, h_center, box_width, box_height]``

    sampling_mode : const<str> (Optional, default="DEFAULT")
        * This parameter can take ``"STRICT_ALIGN_CORNERS"``,
          ``"ALIGN_CORNERS"``, ``"DEFAULT"``, ``"OFFSET_CORNERS"`` or
          ``UNALIGN_CORNERS`` as values.
        * This same convention is used by the ``resize_bilinear`` op (see
          that op for details).

    See Also
    --------
    resize_bilinear

    Returns
    -------
    tensor<[N, B, C, target_height, target_width],T> or tensor<[N, 1, C, target_height, target_width],T>
        * Tensor with same type as the input.
        * If ``roi : tensor<[N,1,4,1,1], T>``, the output is
          ``tensor<[N, B, C, target_height, target_width],T>``.
          Total crops = ``N*B``; that is, ``N`` crops for each input in the batch.
        * If ``roi : tensor<[N,1,5,1,1], T>``, the output is
          ``tensor<[N, 1, C, target_height, target_width],T>``.
          Total crops = ``N``; that is, 1 crop for given input image index
          in the batch.

    Attributes
    ----------
    T: fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        roi=TensorInputType(),
        target_height=IntInputType(const=True, optional=True),
        target_width=IntInputType(const=True, optional=True),
        normalized_coordinates=BoolInputType(const=True, optional=True),
        spatial_scale=FloatInputType(const=True, optional=True),
        box_coordinate_mode=StringInputType(const=True, optional=True),
        sampling_mode=StringInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            target_height=1,
            target_width=1,
            normalized_coordinates=False,
            spatial_scale=1.,
            box_coordinate_mode="CONRNERS_HEIGHT_FIRST",
            sampling_mode="DEFAULT",
            )

    def __init__(self, **kwargs):
        super(crop_resize, self).__init__(**kwargs)

    def type_inference(self):
        if self.x.rank != 4:
            raise ValueError(
                'input to the "crop_resize" op must be of rank 4. Provided {}'.format(
                    self.x.rank
                )
            )

        if self.roi.rank != 5:
            raise ValueError(
                'ROI input to the "crop_resize" op must be of rank 5, provided {}'.format(
                    self.roi.rank
                )
            )

        if self.sampling_mode.val not in {
            "STRICT_ALIGN_CORNERS",
            "ALIGN_CORNERS",
            "UNALIGN_CORNERS",
            "DEFAULT",
            "OFFSET_CORNERS",
        }:
            raise ValueError(
                '"crop_resize" op: unrecognized sampling mode "{}"'.format(
                    self.sampling_mode
                )
            )

        # ret_shape: [N] + [B, C, h_out, w_out]
        N, B, C = self.roi.shape[0], self.x.shape[0], self.x.shape[1]
        ret_shape = [N, B, C, self.target_height.val, self.target_width.val]
        return types.tensor(self.x.dtype, ret_shape)


@register_op(doc_str="TODO")
class crop(Operation):
    """
    Crop the spatial dimensions (last two dimensions) of the input by the
    specified amounts.

    Parameters
    ----------
    x: tensor<[\*D, H1, W1],T> (Required)
        * Must be at least rank ``3``.
    crop_height: const<2, i32> (Required)
        * Amount to be cropped from the top and bottom of the height dimension
          (``axis=-2``).
    crop_width: const<2, i32> (Required)
        * Amount to be cropped from the left and right sides of the width dimension (``axis=-1``).

    Returns
    -------
    tensor<[\*D, H2, W2],T>
        * Tensor with same type as the input.
        * ``H2`` = ``H1`` - crop_height[0] - crop_height[1].
        * ``W2`` = ``W1`` - crop_width[0] - crop_width[1].

    Attributes
    ----------
    T: fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        crop_height=IntTensorInputType(const=True),
        crop_width=IntTensorInputType(const=True),
    )

    def __init__(self, **kwargs):
        super(crop, self).__init__(**kwargs)

    def type_inference(self):
        if self.x.rank < 3:
            raise ValueError(
                'input to the "crop" op must at least be of rank 3. Provided {}'.format(
                    self.x.rank
                )
            )

        crop_height = self.crop_height.val
        crop_width = self.crop_width.val

        if len(crop_height.flatten()) != 2:
            raise ValueError(
                "crop_height must have 2 elements. Provided {}".format(
                    len(crop_height.flatten())
                )
            )

        if len(crop_width.flatten()) != 2:
            raise ValueError(
                "crop_width must have 2 elements. Provided {}".format(
                    len(crop_width.flatten())
                )
            )

        input_shape = list(self.x.shape)
        ret_shape = (
            input_shape[:-2]
            + [input_shape[-2] - crop_height[0] - crop_height[1]]
            + [input_shape[-1] - crop_width[0] - crop_width[1]]
        )
        return types.tensor(self.x.dtype, ret_shape)
