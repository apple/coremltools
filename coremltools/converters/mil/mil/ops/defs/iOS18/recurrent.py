# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS17.recurrent import gru as _gru_iOS17
from coremltools.converters.mil.mil.ops.defs.iOS18 import _IOS18_TARGET


@register_op(opset_version=_IOS18_TARGET)
class gru(_gru_iOS17):
    """
    Gated Recurrent Unit (GRU)

    The only difference between this version and the iOS 17 :py:class:`~.iOS17.recurrent.gru` is
    the reset_after parameter. This parameter is optional and defaults to False. When True, the
    reset gate is applied before the elementwise matrix multiplication.
    """
    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        initial_h=TensorInputType(type_domain="T"),
        weight_ih=TensorInputType(const=True, type_domain="T"),
        weight_hh=TensorInputType(const=True, type_domain="T"),
        bias=TensorInputType(const=True, optional=True, type_domain="T"),
        direction=TensorInputType(const=True, optional=True, type_domain=types.str),
        output_sequence=TensorInputType(const=True, optional=True, type_domain=types.bool),
        recurrent_activation=TensorInputType(const=True, optional=True, type_domain=types.str),
        activation=TensorInputType(const=True, optional=True, type_domain=types.str),
        reset_after=TensorInputType(const=True, optional=True, type_domain=types.bool),
        input_bias=TensorInputType(const=True, optional=True, type_domain="T"),
    )
