#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS15.normalization import (
    batch_norm as _batch_norm_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.normalization import (
    instance_norm as _instance_norm_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.normalization import l2_norm as _l2_norm_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS15.normalization import (
    layer_norm as _layer_norm_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.normalization import (
    local_response_norm as _local_response_norm_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET


@register_op(opset_version=_IOS17_TARGET)
class batch_norm(_batch_norm_iOS15):
    """
    Normalize input tensor ``x`` by ``mean`` and ``variance``, and optionally apply a
    scale ``gamma`` and an offset ``beta``:

    .. math::
       y_i = \\gamma_i \\dfrac{ (x_i - mean_i)}{\\sqrt{variance_i + epsilon}} + beta_i \\;,\\;i=1,....,C

    The difference between this version and the iOS 15 :py:class:`~.iOS15.normalization.batch_norm`
    is that input/output can have different dtypes from other parameters.
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        mean=TensorInputType(const=True, type_domain="U"),
        variance=TensorInputType(const=True, type_domain="U"),
        gamma=TensorInputType(const=True, optional=True, type_domain="U"),
        beta=TensorInputType(const=True, optional=True, type_domain="U"),
        epsilon=TensorInputType(const=True, optional=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }


@register_op(opset_version=_IOS17_TARGET)
class instance_norm(_instance_norm_iOS15):
    """
    Apply instance normalization to the n-dimensional input tensor.

    The difference between this version and the iOS 15 :py:class:`~.iOS15.normalization.instance_norm`
    is that input/output can have different dtypes from other parameters.
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        gamma=TensorInputType(const=True, optional=True, type_domain="U"),
        beta=TensorInputType(const=True, optional=True, type_domain="U"),
        epsilon=TensorInputType(const=True, optional=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }


@register_op(opset_version=_IOS17_TARGET)
class l2_norm(_l2_norm_iOS15):
    """
    Apply L2 normalization to the n-dimensional input tensor. That is, divide the input
    tensor by the square root of the sum of squares of all elements of the input.

    .. math::
       x_i \\leftarrow \\dfrac{x_i}{\\sqrt{\\sum{x_i^2} + \\epsilon}}

    The difference between this version and the iOS 15 :py:class:`~.iOS15.normalization.l2_norm`
    is that input/output and ``epsilon`` can have different dtypes.
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
class layer_norm(_layer_norm_iOS15):
    """
    Apply layer normalization to the n-dimensional input tensor:

    .. math::
       out = gamma * (input - E[x]) / sqrt(Var[x] + epsilon) + beta

    The difference between this version and the iOS 15 :py:class:`~.iOS15.normalization.layer_norm`
    is that input/output can have different dtypes from other parameters.
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        axes=TensorInputType(const=True, optional=True, type_domain=types.int32),
        gamma=TensorInputType(const=True, optional=True, type_domain="U"),
        beta=TensorInputType(const=True, optional=True, type_domain="U"),
        epsilon=TensorInputType(const=True, optional=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }


@register_op(opset_version=_IOS17_TARGET)
class local_response_norm(_local_response_norm_iOS15):
    """
    Apply local response normalization to the n-dimensional input tensor:

    .. math::
       x_i \\leftarrow \\dfrac{x_i}{\\left ( k + \\dfrac{\\alpha}{\\text{size}} \\sum_j x_j^2 \\right )^\\beta}

    The difference between this version and the iOS 15 :py:class:`~.iOS15.normalization.local_response_norm`
    is that input/output can have different dtypes from other parameters.
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        size=TensorInputType(const=True, type_domain=types.int32),
        alpha=TensorInputType(const=True, optional=True, type_domain="U"),
        beta=TensorInputType(const=True, optional=True, type_domain="U"),
        k=TensorInputType(const=True, optional=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.fp16, types.fp32),
    }
