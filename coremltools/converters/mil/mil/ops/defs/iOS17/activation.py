#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS15.activation import (
    clamped_relu as _clamped_relu_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.activation import elu as _elu_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS15.activation import leaky_relu as _leaky_relu_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS15.activation import (
    linear_activation as _linear_activation_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.activation import prelu as _prelu_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS15.activation import (
    scaled_tanh as _scaled_tanh_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.activation import (
    sigmoid_hard as _sigmoid_hard_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.activation import (
    softplus_parametric as _softplus_parametric_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.activation import (
    thresholded_relu as _thresholded_relu_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET


@register_op(opset_version=_IOS17_TARGET)
class clamped_relu(_clamped_relu_iOS15):
    """
    If ``x >= 0`` return elementwise ``min(beta, x)``, otherwise return
    ``min(beta, alpha * x)``. 
    
    The major difference between this version and the iOS 15 :py:class:`~.iOS15.activation.clamped_relu`
    is that the ``alpha`` and ``beta`` may have a different dtype than the input/output.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
    alpha: const U (Required)
    beta: const U (Required)

    Returns
    -------
    tensor<\\*?, T>
        * A tensor of the same type and shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    U: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        alpha=TensorInputType(const=True, type_domain="U"),
        beta=TensorInputType(const=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }


@register_op(opset_version=_IOS17_TARGET)
class elu(_elu_iOS15):
    """
    If ``x > 0`` return elementwise ``x``, otherwise return ``alpha * (e^x - 1)``.

    The major difference between this version and the iOS 15 :py:class:`~.iOS15.activation.elu`
    is that the ``alpha`` may have a different dtype than the input/output.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
    alpha: const U (Required)

    Returns
    -------
    tensor<\\*?, T>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    U: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        alpha=TensorInputType(const=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }


@register_op(opset_version=_IOS17_TARGET)
class leaky_relu(_leaky_relu_iOS15):
    """
    If ``x >= 0`` apply ``x`` elementwise, otherwise apply ``alpha * x`` elementwise.

    The major difference between this version and the iOS 15 :py:class:`~.iOS15.activation.leaky_relu`
    is that the ``alpha`` may have a different dtype than the input/output.

    Parameters
    ----------
    x: <*?, T> (Required)
    alpha: const U (Required)

    Returns
    -------
    tensor<\\*?, T>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    U: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        alpha=TensorInputType(const=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }


@register_op(opset_version=_IOS17_TARGET)
class linear_activation(_linear_activation_iOS15):
    """
    Apply elementwise ``x * alpha + beta``.

    The major difference between this version and the iOS 15 :py:class:`~.iOS15.activation.linear_activation`
    is that the ``alpha`` and ``beta`` may have a different dtype than the input/output.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
    alpha: const U (Required)
    beta: const U (Required)

    Returns
    -------
    tensor<\\*?, T>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    U: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        alpha=TensorInputType(const=True, type_domain="U"),
        beta=TensorInputType(const=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }


@register_op(opset_version=_IOS17_TARGET)
class prelu(_prelu_iOS15):
    """
    Where ``i = 1 ... C``, if ``x_i > 0``, return ``x_i`` , otherwise return ``alpha_i * x_i``.

    The major difference between this version and the iOS 15 :py:class:`~.iOS15.activation.prelu`
    is that the ``alpha`` may have a different dtype than the input/output.

    Parameters
    ----------
    x: tensor<[B, C, 1..3], T> (Required)
        * ``x`` must have rank 4, rank 3, or rank 5; that is, a shape of
          ``(B,C,H)``, ``(B,C,H,W)``, or ``(B,C,D,H,W)``.
    alpha: const tensor<[C], U>, (Required)
        * The length of ``alpha`` must match the second dimension of ``x`` (channel dimension).

    Returns
    -------
    tensor<[B, C, 1..3], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp32, fp16
    U: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        alpha=TensorInputType(const=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }


@register_op(opset_version=_IOS17_TARGET)
class scaled_tanh(_scaled_tanh_iOS15):
    """
    Return ``alpha * tanh(beta * x)`` elementwise.

    The major difference between this version and the iOS 15 :py:class:`~.iOS15.activation.scaled_tanh`
    is that the ``alpha`` and ``beta`` may have a different dtype than the input/output.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
        * Input range is ``(-inf, inf)``.
    alpha: const U (Required)
    beta: const U (Required)

    Returns
    -------
    tensor<\\*?, T>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    U: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        alpha=TensorInputType(const=True, type_domain="U"),
        beta=TensorInputType(const=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }


@register_op(opset_version=_IOS17_TARGET)
class sigmoid_hard(_sigmoid_hard_iOS15):
    """
    Return ``min( max( alpha * x + beta, 0 ), 1 )`` elementwise.

    The major difference between this version and the iOS 15 :py:class:`~.iOS15.activation.sigmoid_hard`
    is that the ``alpha`` and ``beta`` may have a different dtype than the input/output.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
    alpha: const U (Required)
    beta: const U (Required)

    Returns
    -------
    tensor<\\*?, T>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    U: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        alpha=TensorInputType(const=True, type_domain="U"),
        beta=TensorInputType(const=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }


@register_op(opset_version=_IOS17_TARGET)
class softplus_parametric(_softplus_parametric_iOS15):
    """
    Return ``alpha_i * log( 1 + e^( beta_i * x_i ) )``, where ``i = 1 ... C``.

    Parameters
    ----------
    x: tensor<[b, C, n, m], T> (Required)
    alpha: const tensor<[C], U> (Required)
    beta: const tensor<[C], U> (Required)

    Returns
    -------
    tensor<[b, C, n, m], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    U: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        alpha=TensorInputType(const=True, type_domain="U"),
        beta=TensorInputType(const=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }


@register_op(opset_version=_IOS17_TARGET)
class thresholded_relu(_thresholded_relu_iOS15):
    """
    Return ``x`` if ``x >= alpha``, otherwise return ``0``.

    The major difference between this version and the iOS 15 :py:class:`~.iOS15.activation.thresholded_relu`
    is that the ``alpha`` may have a different dtype than the input/output.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
    alpha: const U (Required)

    Returns
    -------
    tensor<\\*, T>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    U: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        alpha=TensorInputType(const=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }
