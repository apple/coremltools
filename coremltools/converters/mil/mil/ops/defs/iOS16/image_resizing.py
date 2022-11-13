# Copyright (c) 2022, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import (DefaultInputs,
                                                       InputSpec,
                                                       TensorInputType)
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS15.image_resizing import \
    crop_resize as _crop_resize_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS15.image_resizing import \
    resample as _resample_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS15.image_resizing import \
    upsample_bilinear as _upsample_bilinear_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS16 import _IOS16_TARGET


@register_op(opset_version=_IOS16_TARGET)
class resample(_resample_iOS15):
    """
    The iOS 16 version of ``resample`` supports float 16 coordinates.
    
    For the complete documentation, see the 
    `iOS 15 version <#module-coremltools.converters.mil.mil.ops.defs.iOS15.image_resizing>`_.
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
        "U": (types.int32, types.fp16, types.fp32),
    }

    def type_inference(self):
        return super().type_inference()

@register_op(opset_version=_IOS16_TARGET)
class upsample_bilinear(_upsample_bilinear_iOS15):
    """
    iOS16 version of upsample_bilinear supports half_pixel_centers

    Additional Parameters
    ----------
    half_pixel_centers: const<bool> (Optional)
        * Default to !align_corners if not provided
    """

    input_spec = _upsample_bilinear_iOS15.input_spec + InputSpec(
        half_pixel_centers=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    def default_inputs(self):
        return super().default_inputs() + DefaultInputs(half_pixel_centers=not self.align_corners.val)

@register_op(opset_version=_IOS16_TARGET)
class crop_resize(_crop_resize_iOS15):
    """
    iOS16 version of crop_resize, which supports ``pad_value``
    
    Additional Parameters
    ----------
    pad_value : const<T> (Optional, default=1.0)
        * If the box indexes go beyond the input boundary, the input image is padded with pad_value.
        * Defaults to 0.
        * It is the same as extrapolation_value in tf.image.crop_and_resize.

    Attributes
    ----------
    T: fp16, fp32
    """
    input_spec = _crop_resize_iOS15.input_spec + InputSpec(
        pad_value=TensorInputType(const=True, optional=True, type_domain="T"),
    )

    def default_inputs(self):
        return super().default_inputs() + DefaultInputs(pad_value=1.0)
