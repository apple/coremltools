#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from ._op_reqs import register_op
from coremltools.converters.mil.mil import Operation, types
from coremltools.converters.mil.mil.input_type import (
    BoolInputType,
    DefaultInputs,
    FloatInputType,
    InputSpec,
    TensorInputType,
    StringInputType
)


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

    * ``W_{ir}``, ``W_{io}``, and ``W_{iz}`` state input-hidden weight for reset, output
      and update gate, respectively.
    * ``W_{h[r|o|z]}`` are recurrent weights on hidden state to reset, output, update gate.
    * ``h_t``  is the hidden state at time ``t``.
    * ``x_t`` is the input at time ``t``.
    * ``h_(t-1)`` is the hidden state of the layer at time ``t-1`` or the initial
      hidden state at time ``0``.
    * ``r_t``, ``o_t``, and ``z_t`` are the reset, new, and update gates, respectively.
    * ``*`` is elementwise product.

    Parameters
    ----------
    x: <s, b, I, T> (Required)
        * ``s`` is the sequence length, ``b`` is the batch size, and ``I`` is the
          input dimension.

    initial_h: <b, H, T> (Required)
        * ``H`` denotes hidden size.

    weight_ih: const<3*H, I, T> (Required) - Weight matrix
        * ``weigh_ih = [W_{ir} | W_{io} | W_{iz}]`` where ``[a|b]`` denotes column
          concatenation and ``[a, b]`` denotes row concatenation. ``W_{ir}``,
          ``W_{io}``, and ``W_{iz}`` have shape ``(H, I)``.
        * This is used when direction="forward" or "reverse".

    weight_hh: const<3*H, H, T> (Required) - Weight matrix
        * ``weight_hh =  [W_{hr} | W_{ho} | W_{hz}]``: ``W_{hr}``, ``W_{ho}``, and
          ``W_{hz}`` have shape ``(H, H)``.
        * This is used when direction="forward" or "reverse".

    bias: const<3*H, T> (Optional) [Default all 0s]
        * ``bias[0]`` are input-hidden and hidden-hidden bias.
        * ``3*H`` are biases for ``[b_{ir} + b_{hz}, b_{io}]``.
        * This is used when direction="forward" or "reverse".

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
        weight_ih=TensorInputType(const=True),
        weight_hh=TensorInputType(const=True),
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
                    len(self.x.rank)
                )
            )

        sequence_length, batch_size, input_size = self.x.shape

        if self.weight_ih.rank != 2:
            raise ValueError(
                "Invalid weight shape. Expecting Rank 2 input, got {}".format(
                    len(self.weight_ih.rank)
                )
            )
        if self.weight_hh.rank != 2:
            raise ValueError(
                "Invalid weight shape. Expecting Rank 2 input, got {}".format(
                    len(self.weight_hh.rank)
                )
            )

        hidden_dim, hidden_size = self.weight_hh.shape

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
                Provided weight_ih {}, weight_hh {}. Expecting <b, 3*H>").format(
                    self.weight_ih.shape, self.weight_hh.shape
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

    weight_ih: const<4*H, I, T> (Required)
        * Input-hidden weight matrix
        * Weight tensor should be in order of
          ``[input_gate, forget_gate, output_gate, cell_gate]``.
        * If direction=="bidirectional", this is applied in forward direction.
        * If direction=="forward" or "backward" these weights are used.

    weight_hh: const<4*H, H, T> (Required)
        * Hidden-hidden weight matrix.
        * Weight tensor should be in order of
          ``[input_gate, forget_gate, output_gate, cell_gate]``.
        * If direction=="bidirectional", this is applied in forward direction.
        * If direction=="forward" or "backward" these weights are used.

    bias: const<4*H, T> (Optional) [Default all 0s]
        * bias = input-hidden bias + hidden-hidden bias
        * If direction=="bidirectional", this is applied in forward direction.
        * If direction=="forward" or "backward" this bias are used.

    peephole: const<3*H, T> (Optional, default to 0)
        * Weight tensor for peephole.
        * Order is ``[input_gate, forget_gate, output_gate]``.
        * Shape of each peephole vector is ``(H,)`` (``H`` is hidden size).
        * If direction=="bidirectional", this is applied in forward direction.
        * If direction=="forward" or "backward" these weights are used.

    weight_ih_back: const<4*H, I, T> (Optional) -
        * Input-hidden weight matrix for backward direction for `bidirectinal LSTM`.
        * Weight tensor should be in order of
          ``[input_gate, forget_gate, output_gate, cell_gate]``.
        * Must be provided for `bidirectional LSTM`.
        * This is only used when `direction` is "bidirectional".
        * For direction="reverse" use `weight_ih` instead.

    weight_hh_back: const<4*H, H, T> (Optional) - Hidden-hidden weight matrix
        * Hidden-hidden weight matrix for backward direction for `bidirectinal LSTM`.
        * Weight tensor should be in order of
          ``[input_gate, forget_gate, output_gate, cell_gate]``.
        * Must be provided for `bidirectional LSTM`.
        * This is only used when `direction` is "bidirectional".
        * For direction="reverse" use `weight_hh` instead.

    bias_back: const<4*H, T> (Optional) [Default all 0s]
        * bias = input-hidden bias + hidden-hidden bias.
        * Bias of backward direction for `bidirectional lstm`
        * This is only used when `direction` is "bidirectional".
        * For direction="reverse" use `bias` instead.

    peephole_back: const<3*H, T> (Optional, default to 0)
        * Weight tensor for peephole in backward direction for `bidirectional LSTM`.
        * Order is ``[input_gate, forget_gate, output_gate]``.
        * Shape of each peephole vector is ``(H,)`` (``H`` is hidden size).
        * Peephole of backward direction for `bidirectional lstm`
        * Bias of backward direction for `bidirectional lstm`
        * This is only used when `direction` is "bidirectional".
        * For direction="reverse" use `peephole` instead.

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
        weight_ih=TensorInputType(const=True),  # ifoz layout,
        weight_hh=TensorInputType(const=True),  # ifoz layout
        bias=TensorInputType(const=True, optional=True),  # ifoz layout
        peephole=TensorInputType(const=True, optional=True),  # ifo layout
        weight_ih_back=TensorInputType(const=True, optional=True),  # ifoz layout,
        weight_hh_back=TensorInputType(const=True, optional=True),  # ifoz layout
        bias_back=TensorInputType(const=True, optional=True),  # ifoz layout
        peephole_back=TensorInputType(const=True, optional=True),  # ifo layout
        direction=StringInputType(const=True, optional=True),
        output_sequence=BoolInputType(const=True, optional=True),
        recurrent_activation=StringInputType(const=True, optional=True),
        cell_activation=StringInputType(const=True, optional=True),
        activation=StringInputType(const=True, optional=True),
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
                    len(self.x.rank)
                )
            )
        sequence_length, batch_size, input_size = self.x.shape

        def weight_shape_check(wt_ih, wt_hh):
            if wt_ih.rank != 2 or wt_hh.rank != 2:
                raise ValueError(
                    "Expecting Rank 2 input, got weight_ih rank: {}, weight_hh rank: {}").format(
                        wt_ih.rank, wt_hh.rank
                    )

            hidden_size = wt_hh.shape[1]
            input_size = wt_ih.shape[1]
            if wt_hh.shape[0] // hidden_size != 4 or wt_ih.shape[0] // hidden_size != 4:
                raise ValueError(
                    "Incorrect weight matrix: hidden dim size mismatch. \
                                Provided weight_ih {}, weight_hh {}. Expecting <4*H, H>").format(
                                    wt_ih.shape, wt_hh.shape
                                )

        direction = self.direction.val
        valid_directions = {"forward", "reverse", "bidirectional"}
        if direction not in valid_directions:
            raise ValueError(
                "Direction {} not supported. Supported directions: {}").format(
                    direction, valid_directions
                )

        weight_shape_check(self.weight_ih, self.weight_hh)
        if direction == "bidirectional":
            weight_shape_check(self.weight_ih_back, self.weight_hh_back)

        hidden_dim, hidden_size = self.weight_hh.shape

        dim_factor = 8 if direction == "bidirectional" else 4
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

    weight_ih: const<H, I, T> (Required) - Input-hidden weight matrix

    weight_hh: const<H, H, T> (Required) - Hidden-hidden weight matrix

    bias: const<H, T> (Optional) [Default all 0s]
        * bias for input-hidden and hidden-hidden

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
        weight_ih=TensorInputType(const=True),
        weight_hh=TensorInputType(const=True),
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
                    len(self.x.rank)
                )
            )

        sequence_length, batch_size, input_size = self.x.shape

        if self.weight_ih.rank != 2 or self.weight_hh.rank != 2:
            raise ValueError(
                "Invalid weight shape. Expecting Rank 2 input, got weight_ih {}, weight_hh {}").format(
                    self.weight_ih.rank, self.weight_hh.rank
                )

        hidden_size, _ = self.weight_ih.shape

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
