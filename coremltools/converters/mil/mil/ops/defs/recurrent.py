#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import get_new_symbol
from ._op_reqs import *


@register_op(doc_str="")
class gru(Operation):
    r"""
    Gated recurrent unit (GRU).
    
    .. math::
       r_t = \rm{recurrent\_activation}(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr})
    
    .. math::
       z_t = \rm{recurrent\_activation}(W_{iz} x_t + b_{iz} + W_{hz} h_(t−1) + b_{hz})
    
    .. math::
       o_t = activation(W_{io} x_t + b_{io} + r_t * (W_{ho} h_(t−1) + b_{ho}))
    
    .. math::
       h_t = (1 − z_t) * o_t + z_t * h_{(t−1)}
    
    Where:

    * ``W_{ir}``, ``W_{iz}``, and `` W_{io}`` state input-hidden weight for reset, update
      and output gate, respectively.
    * Similar to the above, ``W_{h[r/z/o]}`` states hidden-hidden / recurrent weights.
    * ``h_t``  is the hidden state at time ``t``.
    * ``x_t`` is the input at time ``t``.
    * ``h_(t-1)`` is the hidden state of the layer at time ``t-1`` or the initial
      hidden state at time ``0``.
    * ``r_t``, ``z_t``, and ``o_t`` are the reset, update, and new gates, respectively.
    * ``*`` is elementwise product.
    
    Parameters
    ----------
    x: <s, b, I, T> (Required)
        * ``s`` is the sequence length, ``b`` is the batch size, and ``I`` is the
          input dimension.
    
    initial_h: <b, H, T> (Required)
        * ``H`` denotes hidden size.
    
    weight: const<I+H, 3*H, T> (Required) - Weight matrix
        * ``weight[:I] =  [W_{iz} | W_{ir} | W_{io}]`` where ``[a|b]`` denotes column
          concatenation and ``[a, b]`` denotes row concatenation. ``W_{iz}``,
          ``W_{ir}``, and ``W_{io}`` have shape ``(I, H)``.
        * ``weight[I:] =  [W_{hz} | W_{hr} | W_{hn}]``: ``W_{hz}``, ``W_{hr}``, and
          ``W_{hn}`` have shape ``(H, H)``.
    
    bias: const<2, 3*H, T> (Optional) [Default all 0s]
        * ``bias[0]`` and ``bias[1]`` are input-hidden and hidden-hidden
          bias, respectively.
        * ``3*H`` are biases for ``[b_{ir} + b_{hr}, b_{iz} + b_{hz}, b_{io} + b_{ho}]``.
    
    direction: const<str> (Optional) [Default=forward]
        * Either ``forward`` or ``reverse``.
    
    output_sequence: const<bool> (Optional) [Default=False]
        * Outputs every step if ``True``.
    
    recurrent_activation: const<str> (Optional) [Default=sigmoid]
        * Activation applied on update and reset gate.
    
    activation: const<str> (Optional) [Default=tanh]
        * Activation applied on output gate.
    
    Returns
    -------
    <s, b, H, T> or <1, b, H, T>
        * If ``output_sequence == True`` (hidden states from every step):
          ``<s, b, H, T>``.
        * Else ``<1, b, H, T>`` (hidden states of the final step).
    <b, H, T>
        * Hidden states of the final step.
    
    Attributes
    ----------
    T: fp32
    """
    
    input_spec = InputSpec(
        x=TensorInputType(),
        initial_h=TensorInputType(),
        weight=TensorInputType(const=True),
        bias=TensorInputType(const=True, optional=True),
        direction=StringInputType(const=True, optional=True),
        output_sequence=BoolInputType(const=True, optional=True),
        recurrent_activation=StringInputType(const=True, optional=True),
        activation=StringInputType(const=True, optional=True)
    )

    def default_inputs(self):
        return DefaultInputs(
            bias=None,
            direction="forward",
            output_sequence=False,
            recurrent_activation="sigmoid",
            activation="tanh",
            )

    def __init__(self, **kwargs):
        super(gru, self).__init__(**kwargs)

    def type_inference(self):
        if self.x.rank != 3:
            raise ValueError(
                "Invalid input shape. Expecting Rank 3 input, got {}".format(
                    len(self.x.shape)
                )
            )

        sequence_length, batch_size, input_size = self.x.shape

        if self.weight.rank != 2:
            raise ValueError(
                "Invalid weight shape. Expecting Rank 2 input, got {}".format(
                    len(self.weight.shape)
                )
            )

        input_hidden_size, hidden_dim = self.weight.shape
        hidden_size = input_hidden_size - input_size

        direction = self.direction.val
        valid_directions = {"forward", "reverse"}
        if direction not in valid_directions:
            raise ValueError(
                "Direction {} not supported. Supported directions: {}".format(
                    direction, valid_directions
                )
            )

        dim_factor = 3
        if hidden_size != (hidden_dim // dim_factor):
            raise ValueError(
                "Incorrect weight matrix: hidden dim size mismatch. \
                              Provided  {}. Expecting <b, 3*H>".format(
                    self.weight.shape
                )
            )

        out_seq_len = sequence_length if self.output_sequence.val else 1
        output_shape = [out_seq_len, batch_size, hidden_size]
        output_h_shape = [batch_size, hidden_size]
        return (
            types.tensor(self.x.dtype, tuple(output_shape)),
            types.tensor(self.x.dtype, tuple(output_h_shape)),
        )


@register_op(doc_str="")
class lstm(Operation):
    r"""
    Single long short-term memory (LSTM) sequence.
    
    .. math::
       i_t = \rm{recurrent\_activation}(W_{ii} x_t + B_{ii} + W_{hi} h_(t-1) + B_{hi})
    
    .. math::
       f_t = \rm{recurrent\_activation}(W_{if} x_t + B_{if} + W_{hf} h_(t-1) + B_{hf})
    
    .. math::
       z_t = cell_activation(W_{iz} x_t + B_{iz} + W_{hz} h_(t-1) + B_{hz})
    
    .. math::
       o_t = \rm{recurrent\_activation}(W_{io} x_t + B_{io} + W_{ho} h_(t-1) + B_{ho})
    
    .. math::
       c_t = f_t * c_(t-1) + i_t * z_t
    
    .. math::
       h_t = o_t * activation(c_t)
    
    Where:

    * ``i_t``, ``f_t``, ``o_t``, and ``z_t`` are input, forget, output, and cell gates,
      respectively, at time ``t``.
    * ``c_t`` is cell state at time ``t``.
    * ``h_t``  is the hidden state at time ``t``.
    * ``W_{ii}``, ``W_{if}``, ``W_{io}``, and ``W_{iz}`` are input weights for input,
      forget, output and cell gate, respectively.
    * ``W_{hi}``, ``W_{hf}``, ``W_{ho}``, and ``W_{hz}`` are recurrent weights for input,
      forget, output and cell gate, respectively.
    
    Parameters
    ----------
    x: <s, b, I, T> (Required)
        * ``s`` is the sequence length, ``b`` is the batch size, and ``I`` is the
          input dimension.
    
    initial_h: <b, DIRECTION*H, T> (Required)
        * Initial hidden state. ``DIRECTION = 1`` for uni-directional, ``2`` for
          bi-directional LSTM.
        * ``H`` denotes hidden size.
        * ``[b, :H]`` and ``[b, H:]`` represents forward and reverse direction
          values, respectively.
    
    initial_c: <b, DIRECTION*H, T> (Required)
        * Initial cell state.
        * Format is same as ``initial_h``.

    weight: const<I+H, 4*DIRECTION*H, T> (Required) - Weight matrix
        * Weight tensor should be in order of
          ``[input_gate, forget_gate, output_gate, cell_gate]``.
        * ``[I+H, :4*H]`` and ``[I+H, 4*H:]`` represent forward and reverse direction
          values, respectively.
    
    bias: const<2, 4*DIRECTION*H, T> (Optional) [Default all 0s]
        * ``bias[0]`` and ``bias[1]`` are input-hidden and hidden-hidden
          bias, respectively.
    
    direction: const<str> (Optional) [Default=forward]
        * One of the following: ``forward``, ``reverse``, or ``bidirectional``.
        * Must match ``DIRECTIONAL`` in initial states and weight parameters.
    
    output_sequence: const<bool> (Optional) [Default=False]
        * Outputs every step if ``True``.
    
    recurrent_activation: const<str> (Optional) [Default=sigmoid]
        * Activation applied on input, forget, and output gates.
    
    cell_activation: const<str> (Optional) [Default=tang]
        * Activation applied on cell gate.
    
    activation: const<str> (Optional) [Default=tanh]
        * Activation applied on output gate.
    
    peephole: const<3*DIRECTION*H, T> (Optional, default to 0)
        * Weight tensor for peephole.
        * Order is ``[input_gate, forget_gate, output_gate]``.
        * Shape of each peephole vector is ``(H,)`` (``H`` is hidden size).
    
    clip: const<fp32> (optional) [Default=None]
        * Cell gate is clipped to ``[-clip, +clip]``.
    
    Returns
    -------
    <s, b, DIRECTION*H, T> or <1, b, DIRECTION*H, T>
        * If ``output_sequence == True`` (hidden states from every step):
          ``<s, b, DIRECTION*H, T>``.
        * Else ``<1, b, DIRECTION*H, T>`` (hidden states of the final step).
    <b, DIRECTION*H, T>
        * Hidden states of the final step.
    <b, DIRECTION*H, T>
        * Memory state of the final step.
    
    Attributes
    ----------
    T: fp32
    """
    
    input_spec = InputSpec(
        x=TensorInputType(),
        initial_h=TensorInputType(),
        initial_c=TensorInputType(),
        weight=TensorInputType(const=True),  # ifoz layout
        bias=TensorInputType(const=True, optional=True),  # ifoz layout
        direction=StringInputType(const=True, optional=True),
        output_sequence=BoolInputType(const=True, optional=True),
        recurrent_activation=StringInputType(const=True, optional=True),
        cell_activation=StringInputType(const=True, optional=True),
        activation=StringInputType(const=True, optional=True),
        peephole=TensorInputType(const=True, optional=True),  # ifo layout
        clip=FloatInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            bias=None,
            direction="forward",
            output_sequence=False,
            recurrent_activation="sigmoid",
            cell_activation="tanh",
            activation="tanh",
            peephole=None,
            clip=None)

    def __init__(self, **kwargs):
        super(lstm, self).__init__(**kwargs)

    def type_inference(self):
        if self.x.rank != 3:
            raise ValueError(
                "Invalid input shape. Expecting Rank 3 input, got {}".format(
                    len(self.x.shape)
                )
            )

        sequence_length, batch_size, input_size = self.x.shape

        if self.weight.rank != 2:
            raise ValueError(
                "Invalid weight shape. Expecting Rank 2 input, got {}".format(
                    len(self.weight.shape)
                )
            )

        input_hidden_size, hidden_dim = self.weight.shape
        hidden_size = input_hidden_size - input_size

        direction = self.direction.val
        valid_directions = {"forward", "reverse", "bidirectional"}
        if direction not in valid_directions:
            raise ValueError(
                "Direction {} not supported. Supported directions: {}".format(
                    direction, valid_directions
                )
            )

        dim_factor = 8 if direction == "bidirectional" else 4
        if hidden_size != (hidden_dim // dim_factor):
            raise ValueError(
                "Incorrect weight matrix: hidden dim size mismatch. \
                              Provided  {}. Expecting <b, 4*DIRECTION*H>".format(
                    self.weight.shape
                )
            )

        out_seq_len = sequence_length if self.output_sequence.val else 1
        num_directions = dim_factor // 4
        output_shape = [out_seq_len, batch_size, num_directions * hidden_size]
        output_h_shape = [batch_size, num_directions * hidden_size]
        output_c_shape = [batch_size, num_directions * hidden_size]
        return (
            types.tensor(self.x.dtype, tuple(output_shape)),
            types.tensor(self.x.dtype, tuple(output_h_shape)),
            types.tensor(self.x.dtype, tuple(output_c_shape)),
        )


@register_op(doc_str="")
class rnn(Operation):
    """
    Recurrent neural network (RNN).
    
    .. math::
       h_t = activation(W_{ih} x_t + b_{ih} + W_{hh} h_(t−1) + b_{hh})
        
    Where:

    * ``W_{ih}`` is input weight.
    * ``W_{hh}`` is hidden/recurrent weight.
    * ``h_t``  is the hidden state at time ``t``.
    * ``x_t`` is the input at time ``t``.
    * ``h_(t-1)`` is the hidden state of the layer at time ``t-1`` or the initial
      hidden state at time ``0``.
    
    Parameters
    ----------
    x: <s, b, I, T> (Required)
        * ``s`` is the sequence length, ``b`` is the batch size, and ``I`` is the
          input dimension.
    
    initial_h: <b, H, T> (Required)
        * ``H`` denotes hidden size.
    
    weight: const<I+H, 3*H, T> (Required) - Weight matrix
    
    bias: const<2, H, T> (Optional) [Default all 0s]
        * ``bias[0]`` and ``bias[1]`` are input-hidden and hidden-hidden
          bias, respectively.
    
    direction: const<str> (Optional) [Default=forward]
        * Either ``forward`` or ``reverse``.
    
    output_sequence: const<bool> (Optional) [Default=False]
        * Outputs every step if ``True``.
    
    activation: const<str> (Optional) [Default=tanh]
        * Supported activation functions: ``relu``, ``tanh``, ``sigmoid``,
          ``sigmoid_hard``, ``scaled_tanh``, and ``linear``.
    
    Returns
    -------
    <s, b, H, T> or <1, b, H, T>
        * If ``output_sequence == True`` (hidden states from every step):
          ``<s, b, H, T>``.
        * Else ``<1, b, H, T>`` (hidden states of the final step).
    <b, H, T>
        * Hidden states of the final step.
    
    Attributes
    ----------
    T: fp32
    """
    
    input_spec = InputSpec(
        x=TensorInputType(),
        initial_h=TensorInputType(),
        weight=TensorInputType(const=True),
        bias=TensorInputType(const=True, optional=True),
        direction=StringInputType(const=True, optional=True),
        output_sequence=BoolInputType(const=True, optional=True),
        activation=StringInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            bias=None,
            direction="forward",
            output_sequence=False,
            activation="tanh")

    def __init__(self, **kwargs):
        super(rnn, self).__init__(**kwargs)

    def type_inference(self):
        if self.x.rank != 3:
            raise ValueError(
                "Invalid input shape. Expecting Rank 3 input, got {}".format(
                    len(self.x.shape)
                )
            )

        sequence_length, batch_size, input_size = self.x.shape

        if self.weight.rank != 2:
            raise ValueError(
                "Invalid weight shape. Expecting Rank 2 input, got {}".format(
                    len(self.weight.shape)
                )
            )

        _, hidden_size = self.weight.shape

        direction = self.direction.val
        valid_directions = {"forward", "reverse"}
        if direction not in valid_directions:
            raise ValueError(
                "Direction {} not supported. Supported directions: {}".format(
                    direction, valid_directions
                )
            )

        out_seq_len = sequence_length if self.output_sequence.val else 1
        output_shape = [out_seq_len, batch_size, hidden_size]
        output_h_shape = [batch_size, hidden_size]
        return (
            types.tensor(self.x.dtype, tuple(output_shape)),
            types.tensor(self.x.dtype, tuple(output_h_shape)),
        )
