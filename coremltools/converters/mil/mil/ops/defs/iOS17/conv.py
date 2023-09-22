#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS15.conv import conv as _conv_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS15.conv import (
    conv_transpose as _conv_transpose_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET


@register_op(opset_version=_IOS17_TARGET)
class conv(_conv_iOS15):
    """
    Perform convolution over input. Supports 1-D, 2-D, and 3-D convolution.

    The difference between this version and the iOS 15 :py:class:`~.iOS15.conv.conv` is that the
    ``weight`` and ``bias`` may have a different dtype than the input/output.
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        weight=TensorInputType(type_domain="U"),
        bias=TensorInputType(optional=True, type_domain="U"),
        strides=TensorInputType(const=True, optional=True, type_domain=types.int32),
        pad_type=TensorInputType(const=True, optional=True, type_domain=types.str),
        pad=TensorInputType(const=True, optional=True, type_domain=types.int32),
        dilations=TensorInputType(const=True, optional=True, type_domain=types.int32),
        groups=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }


@register_op(opset_version=_IOS17_TARGET)
class conv_transpose(_conv_transpose_iOS15):
    """
    Perform transposed convolution (also known as deconvolution and fractionally
    stride convolution) over input. ``conv_transpose`` can also be used to compute
    the gradient of conv. Supports 1-D, 2-D, and 3-D convolution.

    The differences between this version and the iOS 15 :py:class:`~.iOS15.conv.conv_transpose` are:
    - ``weight`` and ``bias`` may have a different dtype than the input/output.
    - ``weight`` doesn't have to be const.
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        weight=TensorInputType(type_domain="U"),
        bias=TensorInputType(optional=True, type_domain="U"),
        pad=TensorInputType(const=True, optional=True, type_domain=types.int32),
        output_shape=TensorInputType(const=True, optional=True, type_domain=types.int32),
        pad_type=TensorInputType(const=True, optional=True, type_domain=types.str),
        strides=TensorInputType(const=True, optional=True, type_domain=types.int32),
        dilations=TensorInputType(const=True, optional=True, type_domain=types.int32),
        groups=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }
