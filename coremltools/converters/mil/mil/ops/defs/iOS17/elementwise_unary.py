#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_unary import cast as _cast_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_unary import clip as _clip_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET


@register_op(opset_version=_IOS17_TARGET)
class cast(_cast_iOS15):
    """
    Cast the input ``x`` to the new type ``dtype``.
    The only difference between this version and the iOS 15 :py:class:`~.iOS15.elementwise_unary.cast`
    is that it supports int16 and uint16.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)
    dtype: const str (Required)
        * Can be one of the following types: ``int16``, ``uint16``, ``int32``, ``int64``, ``fp16``,
        ``fp32``, ``fp64``, or ``bool``.

    Returns
    -------
    tensor<[\*d], dtype>
        * A tensor of the same shape as ``x``, with type ``dtype``.

    Attributes
    ----------
    T: i16, ui16, i32, i64, fp16, fp32, fp64, bool.
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"), dtype=TensorInputType(const=True, type_domain=types.str)
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.fp64,
            types.int16,
            types.uint16,
            types.int32,
            types.int64,
            types.bool,
        ),
    }

    str_to_types_map = {
        "int16": types.int16,
        "uint16": types.uint16,
        "int32": types.int32,
        "int64": types.int32,
        "fp16": types.fp16,
        "fp32": types.fp32,
        "fp64": types.fp32,
        "bool": types.bool,
    }

    str_to_numpy_type_map = {
        "int16": np.int16,
        "uint16": np.uint16,
        "int32": np.int32,
        "int64": np.int32,
        "fp16": np.float16,
        "fp32": np.float32,
        "fp64": np.float32,
        "bool": bool,
    }


@register_op(opset_version=_IOS17_TARGET)
class clip(_clip_iOS15):
    """
    Clip the values in the input ``x`` to ``[alpha, beta]``, element-wise.
    Any values less than ``alpha`` are set to ``alpha``, and any values greater
    than ``beta`` are set to ``beta``.

    The major difference between this version and the iOS 15 :py:class:`~.iOS15.elementwise_unary.clip`
    is that it uses strict validation to ensure that ``alpha < beta``.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)
    alpha: const T (Required)
    beta: const T (Required)

    Returns
    -------
    tensor<[\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def type_inference(self):
        if self.alpha.val >= self.beta.val:
            raise ValueError(
                f"The `alpha` value ({self.alpha.val}) should be smaller than `beta` value "
                f"({self.beta.val}) in `clip` op."
            )
        return self.x.sym_type
