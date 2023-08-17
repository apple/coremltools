#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS15.recurrent import gru as _gru_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS15.recurrent import lstm as _lstm_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS15.recurrent import rnn as _rnn_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET


@register_op(opset_version=_IOS17_TARGET)
class gru(_gru_iOS15):
    """
    Gated Recurrent Unit (GRU)

    The only difference between this version and the iOS 15 :py:class:`~.iOS15.recurrent.gru` is
    adding the support for fp16.
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
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }


@register_op(opset_version=_IOS17_TARGET)
class lstm(_lstm_iOS15):
    """
    Long Short-Term Memory (LSTM)

    The only difference between this version and the iOS 15 :py:class:`~.iOS15.recurrent.lstm` is
    adding the support for fp16.
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        initial_h=TensorInputType(type_domain="T"),
        initial_c=TensorInputType(type_domain="T"),
        weight_ih=TensorInputType(const=True, type_domain="T"),  # ifoz layout,
        weight_hh=TensorInputType(const=True, type_domain="T"),  # ifoz layout
        bias=TensorInputType(const=True, optional=True, type_domain="T"),  # ifoz layout
        peephole=TensorInputType(const=True, optional=True, type_domain="T"),  # ifo layout
        weight_ih_back=TensorInputType(const=True, optional=True, type_domain="T"),  # ifoz layout,
        weight_hh_back=TensorInputType(const=True, optional=True, type_domain="T"),  # ifoz layout
        bias_back=TensorInputType(const=True, optional=True, type_domain="T"),  # ifoz layout
        peephole_back=TensorInputType(const=True, optional=True, type_domain="T"),  # ifo layout
        direction=TensorInputType(const=True, optional=True, type_domain=types.str),
        output_sequence=TensorInputType(const=True, optional=True, type_domain=types.bool),
        recurrent_activation=TensorInputType(const=True, optional=True, type_domain=types.str),
        cell_activation=TensorInputType(const=True, optional=True, type_domain=types.str),
        activation=TensorInputType(const=True, optional=True, type_domain=types.str),
        clip=TensorInputType(const=True, optional=True, type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }


@register_op(opset_version=_IOS17_TARGET)
class rnn(_rnn_iOS15):
    """
    Recurrent Neural Network (RNN)

    The only difference between this version and the iOS 15 :py:class:`~.iOS15.recurrent.rnn` is
    adding the support for fp16.
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        initial_h=TensorInputType(type_domain="T"),
        weight_ih=TensorInputType(const=True, type_domain="T"),
        weight_hh=TensorInputType(const=True, type_domain="T"),
        bias=TensorInputType(const=True, optional=True, type_domain="T"),
        direction=TensorInputType(const=True, optional=True, type_domain=types.str),
        output_sequence=TensorInputType(const=True, optional=True, type_domain=types.bool),
        activation=TensorInputType(const=True, optional=True, type_domain=types.str),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }
