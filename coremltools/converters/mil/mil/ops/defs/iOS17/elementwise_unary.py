#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_unary import cast as _cast_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_unary import clip as _clip_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_unary import (
    inverse as _inverse_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_unary import log as _log_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_unary import rsqrt as _rsqrt_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET


@register_op(opset_version=_IOS17_TARGET)
class cast(_cast_iOS15):
    """
    Cast the input ``x`` to the new type ``dtype``.
    The only difference between this version and the iOS 15 :py:class:`~.iOS15.elementwise_unary.cast`
    is that it supports int8, uint8, int16, and uint16.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)
    dtype: const str (Required)
        * Can be one of the following types: ``int8``, ``uint8``, ``int16``, ``uint16``, ``int32``, ``fp16``, ``fp32``, or ``bool``.

    Returns
    -------
    tensor<[\*d], dtype>
        * A tensor of the same shape as ``x``, with type ``dtype``.

    Attributes
    ----------
    T: i8, ui8, i16, ui16, i32, fp16, fp32, bool.
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"), dtype=TensorInputType(const=True, type_domain=types.str)
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int8,
            types.uint8,
            types.int16,
            types.uint16,
            types.int32,
            types.bool,
        ),
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


@register_op(opset_version=_IOS17_TARGET)
class inverse(_inverse_iOS15):
    """
    Return the reciprocal value of the input ``x``, element-wise.
    The only difference between this version and the iOS 15 :py:class:`~.iOS15.elementwise_unary.inverse`
    is ``epsilon`` may have different dtypes than the inputs/outputs.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)
    epsilon: const U (Optional, default=1e-4)
        * This is a small constant that is added to the input, before taking its
          inverse, for stability.
        * ``y = 1 / (x + epsilon)``.

    Returns
    -------
    tensor<[\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    U: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        epsilon=TensorInputType(const=True, optional=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }


@register_op(opset_version=_IOS17_TARGET)
class log(_log_iOS15):
    """
    Return the natural logarithm value of the input ``x``, element-wise.
    The only difference between this version and the iOS 15 :py:class:`~.iOS15.elementwise_unary.log`
    is ``epsilon`` may have different dtypes than the inputs/outputs.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)
    epsilon: const U (Optional, default=1e-45)
        * This is a small constant that is added to the input, before taking log.
        * ``y = log(x + epsilon)``.

    Returns
    -------
    tensor<[\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    U: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        epsilon=TensorInputType(const=True, optional=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }


@register_op(opset_version=_IOS17_TARGET)
class rsqrt(_rsqrt_iOS15):
    """
    Return the reciprocal value of the square root of the input ``x``, element-wise.
    The only difference between this version and the iOS 15 :py:class:`~.iOS15.elementwise_unary.rsqrt`
    is ``epsilon`` may have different dtypes than the inputs/outputs.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)
    epsilon: const U (Optional, default=1e-12)
        * This is a small constant that is added to the input, before applying the
          ``rsqrt`` function, for stability.
        * ``y = 1 / sqrt(x + epsilon)``.

    Returns
    -------
    tensor<[\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    U: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        epsilon=TensorInputType(const=True, optional=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }
