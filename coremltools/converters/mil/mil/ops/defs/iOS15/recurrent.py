#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Operation, Var, types
from coremltools.converters.mil.mil.input_type import DefaultInputs, InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op


@register_op
class gru(Operation):
    r"""
    Gated Recurrent Unit (GRU)

    .. math::
       r_t = \rm{recurrent\_activation}(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr})

    .. math::
       z_t = \rm{recurrent\_activation}(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz})

    .. math::
       o_t = \rm{activation}(W_{io} x_t + b_{io} + r_t * W_{ho} h_{t-1} + b_{ho})

    .. math::
       h_t = (1 − z_t) * o_t + z_t * h_{t−1}

    Where:

    * :math:`W_{i[r|o|z]}` are state input weights for reset, output and update gate, respectively.
    * :math:`b_{i[r|o|z]}` are input biases for reset, output and update gate, respectively.
    * :math:`W_{h[r|o|z]}` are recurrent/hidden weights on hidden state to reset, output, and update gates, respectively.
    * :math:`b_{h[r|o|z]}` are recurrent/hidden biases on hidden state to reset, output, and update gates, respectively.
    * :math:`h_t`  is the hidden state at time ``t``.
    * :math:`x_t` is the input at time ``t``.
    * :math:`h_{t-1}` is the hidden state of the layer at time ``t-1`` or the initial
      hidden state at time ``0``.
    * :math:`r_t`, :math:`o_t`, and :math:`z_t` are the reset, new, and update gates, respectively.
    * :math:`*` is elementwise product.

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

    weight_hh: const<3*H, H, T> (Required) - Weight matrix
        * ``weight_hh =  [W_{hr} | W_{ho} | W_{hz}]``: ``W_{hr}``, ``W_{ho}``, and
          ``W_{hz}`` have shape ``(H, H)``.

    bias: const<3*H, T> (Optional) [Default all 0s]
        * ``bias[0]`` are input-hidden and hidden-hidden bias.
        * ``3*H`` are biases for ``[b_{ir} | b_{io} | b_{hz}]``.

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
        x=TensorInputType(type_domain="T"),
        initial_h=TensorInputType(type_domain="T"),
        weight_ih=TensorInputType(const=True, type_domain="T"),
        weight_hh=TensorInputType(const=True, type_domain="T"),
        bias=TensorInputType(const=True, optional=True, type_domain="T"),
        direction=TensorInputType(const=True, optional=True, type_domain=types.str),
        output_sequence=TensorInputType(const=True, optional=True, type_domain=types.bool),
        recurrent_activation=TensorInputType(const=True, optional=True, type_domain=types.str),
        activation=TensorInputType(const=True, optional=True, type_domain=types.str)
    )

    type_domains = {
        "T": (types.fp32,),
    }

    def default_inputs(self):
        return DefaultInputs(
            bias=None,
            direction="forward",
            output_sequence=False,
            recurrent_activation="sigmoid",
            activation="tanh",
        )

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
                Provided weight_ih {}, weight_hh {}. Expecting <b, 3*H>".format(
                    self.weight_ih.shape, self.weight_hh.shape
                )
            )

        out_seq_len = sequence_length if self.output_sequence.val else 1
        output_shape = [out_seq_len, batch_size, hidden_size]
        output_h_shape = [batch_size, hidden_size]
        return (
            types.tensor(self.x.dtype, tuple(output_shape)),
            types.tensor(self.x.dtype, tuple(output_h_shape)),
        )


@register_op
class lstm(Operation):
    r"""
    Long Short-Term Memory (LSTM)

    .. math::
       i_t = \rm{recurrent\_activation}(W_{ii} x_t + B_{ii} + W_{hi} h_{t-1} + B_{hi})

    .. math::
       f_t = \rm{recurrent\_activation}(W_{if} x_t + B_{if} + W_{hf} h_{t-1} + B_{hf})

    .. math::
       z_t = \rm{cell\_activation}(W_{iz} x_t + B_{iz} + W_{hz} h_{t-1} + B_{hz})

    .. math::
       o_t = \rm{recurrent\_activation}(W_{io} x_t + B_{io} + W_{ho} h_{t-1} + B_{ho})

    .. math::
       c_t = f_t * c_{t-1} + i_t * z_t

    .. math::
       h_t = o_t * \rm{activation(c_t)}

    Where:

    * :math:`i_t`, :math:`f_t`, :math:`o_t`, and :math:`z_t` are input, forget, output, and cell gates,
      respectively, at time ``t``.
    * :math:`c_t` is cell state at time ``t``.
    * :math:`h_t`  is the hidden state at time ``t``.
    * :math:`W_{ii}`, :math:`W_{if}`, :math:`W_{io}`, and :math:`W_{iz}` are input weights for input,
      forget, output, and cell gate, respectively.
    * :math:`B_{ii}`, :math:`B_{if}`, :math:`B_{io}`, and :math:`B_{iz}` are input biases for input,
      forget, output, and cell gate, respectively.
    * :math:`W_{hi}`, :math:`W_{hf}`, :math:`W_{ho}`, and :math:`W_{hz}` are recurrent weights for input,
      forget, output, and cell gate, respectively.
    * :math:`B_{hi}`, :math:`B_{hf}`, :math:`B_{ho}`, and :math:`B_{hz}` are recurrent weights for input,
      forget, output, and cell gate, respectively.

    Parameters
    ----------
    x: <s, b, I, T> (Required)
        * ``s`` is the sequence length, ``b`` is the batch size, and ``I`` is the
          input dimension.

    initial_h: <b, DIRECTIONS*H, T> (Required)
        * Initial hidden state. ``DIRECTIONS = 1`` for uni-directional.
          ``DIRECTIONS = 2`` for bi-directional LSTM.
        * ``H`` denotes hidden size.
        * ``[b, :H]`` and ``[b, H:]`` represents forward and reverse direction
          values, respectively.

    initial_c: <b, DIRECTIONS*H, T> (Required)
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

    bias: const<4*H, T> (Optional, default all 0s)
        * bias = input-hidden bias + hidden-hidden bias
        * If direction=="bidirectional", this is applied in forward direction.
        * If direction=="forward" or "backward" this bias are used.

    peephole: const<3*H, T> (Optional, default all 0s)
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

    bias_back: const<4*H, T> (Optional, default all 0s)
        * bias = input-hidden bias + hidden-hidden bias.
        * Bias of backward direction for `bidirectional lstm`
        * This is only used when `direction` is "bidirectional".
        * For direction="reverse" use `bias` instead.

    peephole_back: const<3*H, T> (Optional, default all 0s)
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
        * Supported values: ``hard_sigmoid``, ``linear``, ``relu``, ``scaled_tanh``, ``sigmoid``, ``tanh``

    cell_activation: const<str> (Optional) [Default=tanh]
        * Activation applied on cell gate.
        * Supported values: ``hard_sigmoid``, ``linear``, ``relu``, ``scaled_tanh``, ``sigmoid``, ``tanh``

    activation: const<str> (Optional) [Default=tanh]
        * Activation applied on output gate.
        * Supported values: ``hard_sigmoid``, ``linear``, ``relu``, ``scaled_tanh``, ``sigmoid``, ``tanh``

    clip: const<T> (optional) [Default=None]
        * Cell gate is clipped to ``[-clip, +clip]``.

    Returns
    -------
    <s, b, DIRECTIONS*H, T> or <1, b, DIRECTIONS*H, T>
        * If ``output_sequence == True`` (hidden states from every step):
          ``<s, b, DIRECTIONS*H, T>``.
        * Else ``<1, b, DIRECTIONS*H, T>`` (hidden states of the final step).
    <b, DIRECTIONS*H, T>
        * Hidden states of the final step.
    <b, DIRECTIONS*H, T>
        * Memory state of the final step.

    Attributes
    ----------
    T: fp32
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
        "T": (types.fp32,),
    }

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

    def type_inference(self):
        self._validate_inputs()

        sequence_length, batch_size, input_size = self.x.shape
        hidden_dim, hidden_size = self.weight_hh.shape
        dim_factor = 8 if self.direction.val == "bidirectional" else 4
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

    def _validate_inputs(self):
        _ALLOWED_DIRECTIONS = {"forward", "reverse", "bidirectional"}
        _ALLOWED_ACTIVATIONS = {"tanh", "scaled_tanh", "sigmoid", "hard_sigmoid", "relu", "linear"}

        def check_activation(activation: str):
            if activation.lower() not in _ALLOWED_ACTIVATIONS:
                raise ValueError(
                    f"Activation `{activation}` not supported. Supported activations: {_ALLOWED_ACTIVATIONS}"
                )

        if self.x.rank != 3:
            raise ValueError(f"Invalid input shape. Expecting Rank 3 input, got {len(self.x.rank)}")

        direction = self.direction.val
        if direction not in _ALLOWED_DIRECTIONS:
            raise ValueError(
                f"Direction {direction} not supported. Supported directions: {_ALLOWED_DIRECTIONS}"
            )

        self._weight_shape_check(self.weight_ih, self.weight_hh)
        if direction == "bidirectional":
            if self.weight_ih_back is None or self.weight_hh_back is None:
                raise ValueError(
                    "For bidirectional LSTM, the `weight_ih_back` and `weight_hh_back`"
                    " must be provided."
                )
            self._weight_shape_check(self.weight_ih_back, self.weight_hh_back)

        check_activation(self.recurrent_activation.val)
        check_activation(self.cell_activation.val)
        check_activation(self.activation.val)

    @staticmethod
    def _weight_shape_check(wt_ih: Var, wt_hh: Var):
        if wt_ih.rank != 2 or wt_hh.rank != 2:
            raise ValueError(
                f"Expecting Rank 2 input, got weight_ih rank: {wt_ih.rank}, "
                f"weight_hh rank: {wt_hh.rank}"
            )
        hidden_size = wt_hh.shape[1]
        if wt_hh.shape[0] // hidden_size != 4 or wt_ih.shape[0] // hidden_size != 4:
            raise ValueError(
                f"Incorrect weight matrix: hidden dim size mismatch. Provided "
                f"weight_ih {wt_ih.shape}, weight_hh {wt_hh.shape}. Expecting <4*H, H>"
            )


@register_op
class rnn(Operation):
    r"""
    Recurrent Neural Network (RNN)

    .. math::
       h_t = \rm{activation}(W_{ih} x_t + b_{ih} + W_{hh} h_{t−1} + b_{hh})

    Where:

    * :math:`W_{ih}` is the input weight.
    * :math:`W_{hh}` is the hidden/recurrent weight.
    * :math:`h_t`  is the hidden state at time ``t``.
    * :math:`x_t` is the input at time ``t``.
    * :math:`h_{t-1}` is the hidden state of the layer at time ``t-1`` or the initial
      hidden state at ``t = 0``.
    * :math:`b_{ih}` is the input bias.
    * :math:`b_{hh}` if the hidden/recurrent bias.

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
        "T": (types.fp32,),
    }

    def default_inputs(self):
        return DefaultInputs(
            bias=None,
            direction="forward",
            output_sequence=False,
            activation="tanh")

    def type_inference(self):
        if self.x.rank != 3:
            raise ValueError(
                f"Invalid input shape. Expecting Rank 3 input, got {len(self.x.rank)}"
            )

        sequence_length, batch_size, input_size = self.x.shape

        if self.weight_ih.rank != 2 or self.weight_hh.rank != 2:
            raise ValueError(
                f"Invalid weight shape. Expecting Rank 2 input, got weight_ih "
                f"{self.weight_ih.rank}, weight_hh {self.weight_hh.rank}"
            )

        hidden_size, _ = self.weight_ih.shape

        direction = self.direction.val
        valid_directions = {"forward", "reverse"}
        if direction not in valid_directions:
            raise ValueError(
                f"Direction {direction} not supported. Supported directions: {valid_directions}"
            )

        out_seq_len = sequence_length if self.output_sequence.val else 1
        output_shape = [out_seq_len, batch_size, hidden_size]
        output_h_shape = [batch_size, hidden_size]
        return (
            types.tensor(self.x.dtype, tuple(output_shape)),
            types.tensor(self.x.dtype, tuple(output_h_shape)),
        )
