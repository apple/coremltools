#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools._deps import _HAS_TORCH, MSG_TORCH_NOT_FOUND
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol
from coremltools.converters.mil.mil.ops.tests.iOS14 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import (
    construct_inputs_from_placeholders,
    run_compare_builder,
)
from coremltools.converters.mil.mil.types.type_mapping import numpy_type_to_builtin_type
from coremltools.converters.mil.testing_reqs import compute_units
from coremltools.converters.mil.testing_utils import ssa_fn

if _HAS_TORCH:
    import torch

new_backends = []
for v in backends:
    if v.opset_version <= ct.target.iOS15:
        new_backends.append(v)
backends = new_backends


class TestGRU:
    @pytest.mark.parametrize(
        argnames=[
            "compute_unit",
            "backend",
            "seq_len",
            "batch_size",
            "input_size",
            "hidden_size",
            "has_bias",
            "output_sequence",
            "direction",
            "activation_functions",
            "symbolic",
            "dtype",
        ],
        argvalues=itertools.product(
            compute_units,
            backends,
            [1, 3],
            [1],  # <rdar://problem/59644603> [MIL] GRU with batch size 1 produces incorrect
            # output(always 0) for second batch onwards
            [1, 2],
            [1, 2],
            [True, False],
            [True, False],
            ["forward", "reverse"],
            [
                ["tanh", "sigmoid"],
                ["sigmoid", "tanh"],
            ],
            [True, False],
            [np.float32],
        ),
    )
    def test_builder_to_backend_smoke(
        self,
        compute_unit,
        backend,
        seq_len,
        batch_size,
        input_size,
        hidden_size,
        has_bias,
        output_sequence,
        direction,
        activation_functions,
        symbolic,
        dtype,
    ):
        torch.manual_seed(5)

        R_z = 2 * np.random.rand(hidden_size, hidden_size) - 1
        R_r = 2 * np.random.rand(hidden_size, hidden_size) - 1
        R_o = 2 * np.random.rand(hidden_size, hidden_size) - 1
        W_z = 2 * np.random.rand(hidden_size, input_size) - 1
        W_r = 2 * np.random.rand(hidden_size, input_size) - 1
        W_o = 2 * np.random.rand(hidden_size, input_size) - 1
        b_z = 2 * np.random.rand(hidden_size) - 1 if has_bias else np.zeros((hidden_size))
        b_r = 2 * np.random.rand(hidden_size) - 1 if has_bias else np.zeros((hidden_size))
        b_o = 2 * np.random.rand(hidden_size) - 1 if has_bias else np.zeros((hidden_size))

        def apply_act(x, option):
            if option == "tanh":
                return np.tanh(x)
            elif option == "sigmoid":
                return 1.0 / (1 + np.exp(-x))
            else:
                raise ValueError("activation invalid")

        def get_numpy_prediction_gru(
            X,
            H,
            return_seq,
            direction,
            inner_activation_str="sigmoid",
            activation_str="tanh",
        ):
            """
            shape of X : (B, Seq, input_size)

            shape of H : (B, hidden_size)

            shape of return = (B, 1, hidden_size) if return_seq=False else (B, Seq, hidden_size)
            """
            assert X.shape == (batch_size, seq_len, input_size)
            assert H.shape == (batch_size, hidden_size)
            out = []
            for i in range(batch_size):
                numpy_input = X[i]
                hidden_state = H[i]
                out.append(
                    get_numpy_prediction_gru_single_batch(
                        numpy_input,
                        hidden_state,
                        return_seq,
                        direction,
                        inner_activation_str=inner_activation_str,
                        activation_str=activation_str,
                    )
                )
            output = np.stack(out, axis=0)
            output = np.transpose(output, (1, 0, 2))
            return output, output[-1, :, :]

        def get_numpy_prediction_gru_single_batch(
            X, h, return_seq, direction, inner_activation_str="sigmoid", activation_str="tanh"
        ):
            np_out = np.zeros((seq_len, hidden_size))
            batch_x = X if direction == "forward" else X[::-1, :]
            for k in range(seq_len):
                x = batch_x[k, :]
                z = apply_act(np.dot(W_z, x) + np.dot(R_z, h) + b_z, inner_activation_str)
                r = apply_act(np.dot(W_r, x) + np.dot(R_r, h) + b_r, inner_activation_str)
                c = h * r
                o = apply_act(np.dot(W_o, x) + np.dot(R_o, c) + b_o, activation_str)
                h = (1 - z) * o + z * h
                np_out[k, :] = h

            if return_seq:
                np_out_final = np_out
            else:
                np_out_final = np_out[-1:, :]

            return np_out_final

        x = np.random.rand(batch_size, seq_len, input_size).astype(dtype)
        h = np.random.rand(batch_size, hidden_size).astype(dtype)

        activation, inner_activation = activation_functions
        output, state = get_numpy_prediction_gru(
            x, h, output_sequence, direction, inner_activation, activation
        )
        expected_outputs = [output, state]

        if symbolic:
            batch_size = get_new_symbol()
            seq_len = get_new_symbol()

        hh_wt = np.concatenate([R_r, R_o, R_z], axis=0).astype(dtype)
        ih_wt = np.concatenate([W_r, W_o, W_z], axis=0).astype(dtype)
        b = np.concatenate([b_r, b_o, b_z], axis=0).astype(dtype)

        input_shape = [seq_len, batch_size, input_size]
        h_shape = [batch_size, hidden_size]

        builtin_dtype = numpy_type_to_builtin_type(dtype)
        input_placeholders = {
            "x": mb.placeholder(shape=input_shape, dtype=builtin_dtype),
            "initial_h": mb.placeholder(shape=h_shape, dtype=builtin_dtype),
        }

        coreml_x = np.transpose(x, (1, 0, 2))
        input_values = {"x": coreml_x, "initial_h": h}

        expected_output_types = [
            (seq_len if output_sequence else 1, batch_size, hidden_size, builtin_dtype),
            (batch_size, hidden_size, builtin_dtype),
        ]

        def build(x, initial_h):
            arguments = {
                "x": x,
                "initial_h": initial_h,
                "weight_ih": ih_wt,
                "weight_hh": hh_wt,
                "direction": direction,
                "output_sequence": output_sequence,
                "activation": activation,
                "recurrent_activation": inner_activation,
            }
            # If bias is provided, add in arguments
            if has_bias:
                arguments["bias"] = b
            return mb.gru(**arguments)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            inputs=construct_inputs_from_placeholders(input_placeholders, upper_bound=10)
            if symbolic and backend.backend == "mlprogram"
            else None,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestLSTM:
    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "input_dims",
                "output_dim",
                "activation",
                "inner_activation",
                "outer_activation",
                "return_seq",
                "has_bias",
                "forget_bias",
                "has_peephole",
                "coupled_input_forget",
                "clip",
                "dtype",
            ]
        ),
        itertools.product(
            compute_units,
            backends,
            [[8, 32, 32]],
            [1, 4],
            ["sigmoid"],
            ["tanh"],
            ["relu", "scaled_tanh", "hard_sigmoid", "linear"],
            [False, True],
            [False, True],
            [False, True],
            [True, False],
            [False],  # We have not exposed this option yet!
            [50.0, 0.2, 0.01],
            [np.float32],  # Only support fp32 before iOS17.
        ),
    )
    def test_numpy_numerical(
        self,
        compute_unit,
        backend,
        input_dims,
        output_dim,
        activation,
        inner_activation,
        outer_activation,
        return_seq,
        has_bias,
        forget_bias,
        has_peephole,
        coupled_input_forget,
        clip,
        dtype,
    ):
        def _apply_act(x, option):
            # All activation functions use their standard default values.
            # This makes `tanh` equivalent to `scaled_tanh`, and makes `linear` a pass through.
            if option == "tanh" or option == "scaled_tanh":
                return np.tanh(x)
            elif option == "relu":
                return np.maximum(0, x)
            elif option == "sigmoid":
                return 1.0 / (1 + np.exp(-x))
            elif option == "hard_sigmoid":
                return np.minimum(np.maximum(0.2 * x + 0.5, 0), 1)
            elif option == "linear":
                return x
            else:
                raise ValueError("activation invalid")

        def _clip(x, threshold=500.0):
            return np.maximum(np.minimum(x, threshold), -threshold)

        def _get_numpy_prediction_lstm(Weights, X):
            # X : (batch, seq_len, channel)
            batch, _, _ = X.shape
            out = []
            for i in range(batch):
                out.append(
                    _get_numpy_prediction_lstm_single_batch(
                        Weights, np.expand_dims(X[i, :, :], axis=0)
                    )
                )
            return np.stack(out, axis=0)

        def _get_numpy_prediction_lstm_single_batch(Weights, X):

            batch_size, seq_len, input_size = X.shape
            X = X[0, :, :]
            hidden_size = output_dim

            b = Weights["b"]
            Wx_i, Wx_f, Wx_o, Wx_g = np.split(Weights["W_x"], 4)
            Wh_i, Wh_f, Wh_o, Wh_g = np.split(Weights["W_h"], 4)
            b_i, b_f, b_o, b_g = np.split(b, 4)
            p_i, p_f, p_o = np.split(Weights["p"], 3)

            act1 = activation
            act2 = inner_activation
            act3 = outer_activation

            h = np.zeros((hidden_size))
            c = np.zeros((hidden_size))
            np_out = np.zeros((seq_len, hidden_size))
            for k in range(seq_len):
                x = X[k, :]
                i = _apply_act(np.dot(Wx_i, x) + np.dot(Wh_i, h) + b_i + c * p_i, act1)
                f = _apply_act(np.dot(Wx_f, x) + np.dot(Wh_f, h) + b_f + c * p_f, act1)
                g = _apply_act(np.dot(Wx_g, x) + np.dot(Wh_g, h) + b_g, act2)
                if coupled_input_forget:
                    c = c * (1 - i) + i * g
                else:
                    c = c * f + i * g
                c = _clip(c, clip)
                o = _apply_act(np.dot(Wx_o, x) + np.dot(Wh_o, h) + b_o + c * p_o, act1)
                h = o * _apply_act(c, act3)
                np_out[k, :] = h

            if return_seq:
                np_out_final = np_out
            else:
                np_out_final = np_out[-1:, :]
            return np_out_final

        batch, seq_len, input_size = input_dims
        hidden_size = output_dim

        # define random weights
        W_x = np.random.rand(4 * hidden_size, input_size).astype(dtype)
        W_h = np.random.rand(4 * hidden_size, hidden_size).astype(dtype)

        if has_bias:
            b = np.random.rand(4 * hidden_size) - 0.5
            if forget_bias:
                b = b + 1
        else:
            b = np.zeros((4 * hidden_size))
        b = b.astype(dtype)

        if has_peephole:
            p = np.random.rand(3 * hidden_size) - 0.5
        else:
            p = np.zeros((3 * hidden_size))
        p = p.astype(dtype)

        weights = {"W_x": W_x, "W_h": W_h, "b": b, "p": p}

        input_data = np.random.rand(batch, seq_len, input_size).astype(dtype)
        numpy_preds = _get_numpy_prediction_lstm(weights, input_data)
        numpy_preds = np.transpose(numpy_preds, [1, 0, 2])

        coreml_input_data = np.transpose(input_data, [1, 0, 2])
        builtin_dtype = numpy_type_to_builtin_type(dtype)
        input_placeholders = {
            "x": mb.placeholder(shape=coreml_input_data.shape, dtype=builtin_dtype)
        }
        input_values = {"x": coreml_input_data}

        def build(x):
            h_all, ht, ct = mb.lstm(
                x=x,
                initial_h=np.zeros((batch, hidden_size)).astype(dtype),
                initial_c=np.zeros((batch, hidden_size)).astype(dtype),
                weight_ih=W_x,
                weight_hh=W_h,
                peephole=p,
                direction="forward",
                bias=b,
                output_sequence=return_seq,
                recurrent_activation=activation,
                cell_activation=inner_activation,
                activation=outer_activation,
                clip=dtype(clip),
            )
            return h_all

        expected_output_types = (
            seq_len if return_seq else 1,
            batch,
            hidden_size,
            builtin_dtype,
        )
        expected_outputs = numpy_preds

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            atol=1e-3,
            rtol=1e-3,
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        argnames=[
            "compute_unit",
            "backend",
            "seq_len",
            "batch_size",
            "input_size",
            "hidden_size",
            "has_bias",
            "output_sequence",
            "direction",
            "symbolic",
            "dtype",
        ],
        argvalues=itertools.product(
            compute_units,
            backends,
            [1, 8],
            [1, 32],
            [1, 64],
            [1, 16],
            [True, False],
            [True, False],
            ["forward", "reverse"],
            [True, False],
            [np.float32],  # Only support fp32 before iOS17.
        ),
    )
    def test_builder_to_backend_smoke_unilstm(
        self,
        compute_unit,
        backend,
        seq_len,
        batch_size,
        input_size,
        hidden_size,
        has_bias,
        output_sequence,
        direction,
        symbolic,
        dtype,
    ):

        torch.manual_seed(50)
        rnn = torch.nn.LSTM(input_size, hidden_size, 1, bias=has_bias)
        state_dict = rnn.state_dict()

        ih_wt = state_dict["weight_ih_l0"].detach().numpy()
        hh_wt = state_dict["weight_hh_l0"].detach().numpy()

        # Make weight compatible to CoreML format
        def ifzo_to_ifoz(x):
            i, f, z, o = np.split(x, 4)
            return np.concatenate([i, f, o, z], axis=0).astype(dtype)

        w_x = ifzo_to_ifoz(ih_wt)
        w_h = ifzo_to_ifoz(hh_wt)

        b = None
        if has_bias:
            ih_b = state_dict["bias_ih_l0"].detach().numpy()
            hh_b = state_dict["bias_hh_l0"].detach().numpy()
            ih_b = ifzo_to_ifoz(ih_b)
            hh_b = ifzo_to_ifoz(hh_b)
            b = ih_b + hh_b

        t = torch.randn(seq_len, batch_size, input_size)
        h0 = torch.randn(1, batch_size, hidden_size)
        c0 = torch.randn(1, batch_size, hidden_size)

        n_t = t
        if direction == "reverse":
            n_t = torch.flip(n_t, [0])

        output, (hn, cn) = rnn(n_t, (h0, c0))
        if not output_sequence:
            output = output[-1].unsqueeze(0)

        output = output.detach().numpy()
        hn = hn.detach().numpy().squeeze(0)
        cn = cn.detach().numpy().squeeze(0)

        t = np.reshape(t.detach().numpy(), [seq_len, batch_size, input_size])
        h = np.reshape(h0.detach().numpy().squeeze(0), [batch_size, hidden_size])
        c = np.reshape(c0.detach().numpy().squeeze(0), [batch_size, hidden_size])

        if symbolic:
            batch_size = get_new_symbol()
            seq_len = get_new_symbol()

        input_shape = [seq_len, batch_size, input_size]
        h_shape = [batch_size, hidden_size]
        c_shape = [batch_size, hidden_size]

        builtin_dtype = numpy_type_to_builtin_type(dtype)
        expected_output_types = [
            (seq_len if output_sequence else 1, batch_size, hidden_size, builtin_dtype),
            (batch_size, hidden_size, builtin_dtype),
            (batch_size, hidden_size, builtin_dtype),
        ]
        expected_outputs = [output, hn, cn]

        input_placeholders = {
            "x": mb.placeholder(shape=input_shape, dtype=builtin_dtype),
            "initial_h": mb.placeholder(shape=h_shape, dtype=builtin_dtype),
            "initial_c": mb.placeholder(shape=c_shape, dtype=builtin_dtype),
        }
        input_values = {
            "x": t.astype(dtype),
            "initial_h": h.astype(dtype),
            "initial_c": c.astype(dtype),
        }

        def build(x, initial_h, initial_c):
            arguments = {
                "x": x,
                "initial_h": initial_h,
                "initial_c": initial_c,
                "weight_ih": w_x,
                "weight_hh": w_h,
                "direction": direction,
                "output_sequence": output_sequence,
            }
            # If bias is provided, add in arguments
            if b is not None:
                arguments["bias"] = b
            return mb.lstm(**arguments)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            inputs=construct_inputs_from_placeholders(input_placeholders, upper_bound=64)
            if symbolic and backend.backend == "mlprogram"
            else None,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        argnames=[
            "compute_unit",
            "backend",
            "seq_len",
            "batch_size",
            "input_size",
            "hidden_size",
            "has_bias",
            "output_sequence",
            "symbolic",
            "dtype",
        ],
        argvalues=itertools.product(
            compute_units,
            backends,
            [1, 8],
            [1, 32],
            [1, 64],
            [2, 16],
            [True, False],
            [True, False],
            [True, False],
            [np.float32],
        ),
    )
    def test_builder_to_backend_smoke_bidirlstm(
        self,
        compute_unit,
        backend,
        seq_len,
        batch_size,
        input_size,
        hidden_size,
        has_bias,
        output_sequence,
        symbolic,
        dtype,
    ):
        def _pytorch_hidden_to_coreml(x):
            x = x.detach().numpy()
            # Split of Direction axis
            f, b = np.split(x, 2, axis=0)
            # Concat on Hidden Size axis
            x = np.concatenate([f, b], axis=2)
            x = np.squeeze(x, axis=0)
            return x

        direction = "bidirectional"
        torch.manual_seed(20)
        rnn = torch.nn.LSTM(input_size, hidden_size, 1, bidirectional=True, bias=has_bias)
        state_dict = rnn.state_dict()

        ih_wt = state_dict["weight_ih_l0"].detach().numpy()
        hh_wt = state_dict["weight_hh_l0"].detach().numpy()
        ih_wt_r = state_dict["weight_ih_l0_reverse"].detach().numpy()
        hh_wt_r = state_dict["weight_hh_l0_reverse"].detach().numpy()

        def ifzo_to_ifoz(x):
            i, f, z, o = np.split(x, 4)
            return np.concatenate([i, f, o, z], axis=0).astype(dtype)

        wx = ifzo_to_ifoz(ih_wt)
        wh = ifzo_to_ifoz(hh_wt)
        r_wx = ifzo_to_ifoz(ih_wt_r)
        r_wh = ifzo_to_ifoz(hh_wt_r)

        b, r_b = None, None
        if has_bias:
            ih_b = state_dict["bias_ih_l0"].detach().numpy()
            hh_b = state_dict["bias_hh_l0"].detach().numpy()
            r_ih_b = state_dict["bias_ih_l0_reverse"].detach().numpy()
            r_hh_b = state_dict["bias_hh_l0_reverse"].detach().numpy()
            # Convert forward bias into [4*H]
            b = ih_b + hh_b
            b = ifzo_to_ifoz(b)
            # Convert reverse bias into [*H]
            r_b = r_ih_b + r_hh_b
            r_b = ifzo_to_ifoz(r_b)

        t = torch.randn(seq_len, batch_size, input_size)
        h0 = torch.randn(2, batch_size, hidden_size)
        c0 = torch.randn(2, batch_size, hidden_size)

        output, (hn, cn) = rnn(t, (h0, c0))
        if not output_sequence:
            output_f = output[-1].unsqueeze(0)[:, :, :hidden_size]
            output_r = output[0].unsqueeze(0)[:, :, hidden_size:]
            output = torch.cat([output_f, output_r], dim=2)

        output = output.detach().numpy().astype(dtype)
        hn = _pytorch_hidden_to_coreml(hn).astype(dtype)
        cn = _pytorch_hidden_to_coreml(cn).astype(dtype)

        if symbolic:
            batch_size = get_new_symbol()
            seq_len = get_new_symbol()

        input_shape = [seq_len, batch_size, input_size]
        h_shape = [batch_size, 2 * hidden_size]
        c_shape = [batch_size, 2 * hidden_size]

        builtin_dtype = numpy_type_to_builtin_type(dtype)
        expected_output_types = [
            (
                seq_len if output_sequence else 1,
                batch_size,
                2 * hidden_size,
                builtin_dtype,
            ),
            (batch_size, 2 * hidden_size, builtin_dtype),
            (batch_size, 2 * hidden_size, builtin_dtype),
        ]
        expected_outputs = [output, hn, cn]

        t = t.detach().numpy()
        h = _pytorch_hidden_to_coreml(h0)
        c = _pytorch_hidden_to_coreml(c0)

        input_placeholders = {
            "x": mb.placeholder(shape=input_shape, dtype=builtin_dtype),
            "initial_h": mb.placeholder(shape=h_shape, dtype=builtin_dtype),
            "initial_c": mb.placeholder(shape=c_shape, dtype=builtin_dtype),
        }
        input_values = {
            "x": t.astype(dtype),
            "initial_h": h.astype(dtype),
            "initial_c": c.astype(dtype),
        }

        def build(x, initial_h, initial_c):
            arguments = {
                "x": x,
                "initial_h": initial_h,
                "initial_c": initial_c,
                "weight_ih": wx,
                "weight_hh": wh,
                "weight_ih_back": r_wx,
                "weight_hh_back": r_wh,
                "direction": direction,
                "output_sequence": output_sequence,
            }
            # If bias is provided, add in arguments
            if b is not None:
                arguments["bias"] = b
                arguments["bias_back"] = r_b
            return mb.lstm(**arguments)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            inputs=construct_inputs_from_placeholders(input_placeholders, upper_bound=64)
            if symbolic and backend.backend == "mlprogram"
            else None,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_invalid_bidirectional_lstm(self):
        with pytest.raises(
            ValueError,
            match="For bidirectional LSTM, the `weight_ih_back` and "
            "`weight_hh_back` must be provided.",
        ):
            seq_len = 3
            batch = 2
            input_size = 4
            hidden_size = 5
            mb.lstm(
                x=np.random.rand(seq_len, batch, input_size),
                initial_h=np.zeros((batch, hidden_size)).astype(np.float32),
                initial_c=np.zeros((batch, hidden_size)).astype(np.float32),
                weight_ih=np.random.rand(4 * hidden_size, input_size),
                weight_hh=np.random.rand(4 * hidden_size, hidden_size),
                direction="bidirectional",
            )

    @ssa_fn
    def test_invalid_activation_lstm(self):
        seq_len = 3
        batch = 2
        input_size = 4
        hidden_size = 5
        arguments = {
            "x": np.random.rand(seq_len, batch, input_size),
            "initial_h": np.zeros((batch, hidden_size)).astype(np.float32),
            "initial_c": np.zeros((batch, hidden_size)).astype(np.float32),
            "weight_ih": np.random.rand(4 * hidden_size, input_size),
            "weight_hh": np.random.rand(4 * hidden_size, hidden_size),
            "direction": "forward",
        }

        with pytest.raises(ValueError, match="Activation `dummy` not supported."):
            mb.lstm(recurrent_activation="dummy", **arguments)

        with pytest.raises(ValueError, match="Activation `dummy` not supported."):
            mb.lstm(cell_activation="dummy", **arguments)

        with pytest.raises(ValueError, match="Activation `dummy` not supported."):
            mb.lstm(activation="dummy", **arguments)


class TestRNN:
    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        argnames=[
            "compute_unit",
            "backend",
            "seq_len",
            "batch_size",
            "input_size",
            "hidden_size",
            "has_bias",
            "output_sequence",
            "direction",
            "symbolic",
            "dtype",
        ],
        argvalues=itertools.product(
            compute_units,
            backends,
            [2, 8],
            [1, 32],
            [1, 64],
            [1, 16],
            [True, False],
            [True, False],
            ["forward", "reverse"],
            [True, False],
            [np.float32],
        ),
    )
    def test_builder_to_backend_smoke(
        self,
        compute_unit,
        backend,
        seq_len,
        batch_size,
        input_size,
        hidden_size,
        has_bias,
        output_sequence,
        direction,
        symbolic,
        dtype,
    ):
        torch.manual_seed(50)
        rnn = torch.nn.RNN(input_size, hidden_size, 1, bias=has_bias)
        state_dict = rnn.state_dict()

        ih_wt = state_dict["weight_ih_l0"].detach().numpy().astype(dtype)
        hh_wt = state_dict["weight_hh_l0"].detach().numpy().astype(dtype)

        b = None
        if has_bias:
            ih_b = state_dict["bias_ih_l0"].detach().numpy().astype(dtype)
            hh_b = state_dict["bias_hh_l0"].detach().numpy().astype(dtype)
            b = ih_b + hh_b

        t = torch.randn(seq_len, batch_size, input_size)
        h0 = torch.randn(1, batch_size, hidden_size)

        n_t = t
        if direction == "reverse":
            n_t = torch.flip(n_t, [0])

        output, hn = rnn(n_t, h0)
        if not output_sequence:
            output = output[-1].unsqueeze(0)

        output = output.detach().numpy()
        hn = hn.detach().numpy().squeeze(0)

        t = np.reshape(t.detach().numpy(), [seq_len, batch_size, input_size])
        h = np.reshape(h0.detach().numpy().squeeze(0), [batch_size, hidden_size])

        if symbolic:
            batch_size = get_new_symbol()
            seq_len = get_new_symbol()

        input_shape = [seq_len, batch_size, input_size]
        h_shape = [batch_size, hidden_size]

        builtin_dtype = numpy_type_to_builtin_type(dtype)
        expected_output_types = [
            (seq_len if output_sequence else 1, batch_size, hidden_size, builtin_dtype),
            (batch_size, hidden_size, builtin_dtype),
        ]
        expected_outputs = [output, hn]

        input_placeholders = {
            "x": mb.placeholder(shape=input_shape, dtype=builtin_dtype),
            "initial_h": mb.placeholder(shape=h_shape, dtype=builtin_dtype),
        }
        input_values = {"x": t.astype(dtype), "initial_h": h.astype(dtype)}

        def build(x, initial_h):
            arguments = {
                "x": x,
                "initial_h": initial_h,
                "weight_ih": ih_wt,
                "weight_hh": hh_wt,
                "direction": direction,
                "output_sequence": output_sequence,
            }
            # If bias is provided, add in arguments
            if b is not None:
                arguments["bias"] = b
            return mb.rnn(**arguments)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            inputs=construct_inputs_from_placeholders(input_placeholders, upper_bound=64)
            if symbolic and backend.backend == "mlprogram"
            else None,
            compute_unit=compute_unit,
            backend=backend,
        )
