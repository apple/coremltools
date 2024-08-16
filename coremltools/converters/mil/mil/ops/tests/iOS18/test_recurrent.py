# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest
import torch

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types
from coremltools.converters.mil.mil.ops.tests.iOS17.test_recurrent import TestGRU as _TestGRU_iOS17
from coremltools.converters.mil.mil.ops.tests.iOS18 import backends
from coremltools.converters.mil.testing_reqs import compute_units


class TestGRU(_TestGRU_iOS17):
    # Test functionality from previous opset version
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
            [1],
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
            [np.float16, np.float32],
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
        super().test_builder_to_backend_smoke(
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
        )


    @pytest.mark.xfail(reason="rdar://128479517")
    @pytest.mark.parametrize(
        argnames=[
            "compute_units",
            "backend",
            "sequence_length",
            "num_features",    # also called "input_size"
            "hidden_size",
            "batch_size",
        ],
        argvalues=itertools.product(
            compute_units,
            backends,
            [1, 3],
            [1, 2],
            [1],
            [1, 2],
        ),
    )
    def test_pytorch_parity(self, backend, compute_units, sequence_length, num_features, hidden_size, batch_size):

        def get_weight_i_tensor():
            return np.random.rand(hidden_size, num_features).astype('float32')

        def get_weight_h_tensor():
            return np.random.rand(hidden_size, hidden_size).astype('float32')

        def get_bias_tensor():
            return np.random.rand(hidden_size).astype('float32')

        W_ir, W_iz, W_in = get_weight_i_tensor(), get_weight_i_tensor(), get_weight_i_tensor()
        W_hr, W_hz, W_hn = get_weight_h_tensor(), get_weight_h_tensor(), get_weight_h_tensor()

        b_ir, b_iz, b_in = get_bias_tensor(), get_bias_tensor(), get_bias_tensor()
        b_hr, b_hz, b_hn = get_bias_tensor(), get_bias_tensor(), get_bias_tensor()

        # MIL op only supports single direction and single layer
        x = np.random.rand(sequence_length, batch_size, num_features).astype('float16')
        initial_h = np.random.rand(1, batch_size, hidden_size).astype('float16')

        # Set up PyTorch model
        m_t = torch.nn.GRU(num_features, hidden_size)
        t_state = m_t.state_dict()
        t_state['weight_ih_l0'] = torch.Tensor(np.concatenate((W_ir, W_iz, W_in)))
        t_state['weight_hh_l0'] = torch.Tensor(np.concatenate((W_hr, W_hz, W_hn)))
        t_state['bias_ih_l0'] = torch.Tensor(np.concatenate((b_ir, b_iz, b_in)))
        t_state['bias_hh_l0'] = torch.Tensor(np.concatenate((b_hr, b_hz, b_hn)))
        m_t.load_state_dict(t_state)

        # Get PyTorch results
        (out_t, h_t) = m_t(torch.Tensor(x), torch.Tensor(initial_h))
        out_t = out_t.detach().numpy()
        h_t = h_t.detach().numpy()

        # MIL op only support num_layers=1 and D=1, so hidden state only has rank 2
        initial_h = initial_h.squeeze(0)

        # MIL program
        @mb.program(
            [
                mb.TensorSpec(shape=x.shape, dtype=types.fp32),
                mb.TensorSpec(shape=initial_h.shape, dtype=types.fp32)
            ],
            opset_version=backend.opset_version
        )
        def prog(x, initial_h):
            return mb.gru(
                x=x,
                initial_h=initial_h,
                weight_ih=np.concatenate((W_ir, W_in, W_iz)),
                weight_hh=np.concatenate((W_hr, W_hn, W_hz)),
                input_bias=np.concatenate((b_ir, b_in, b_iz)),
                bias=np.concatenate((b_hr, b_hn, b_hz)),
                reset_after=True,
                output_sequence=True,
            )

        mlmodel = ct.convert(
            prog,
            source="milinternal",
            convert_to=backend.backend,
            minimum_deployment_target=backend.opset_version,
            compute_units=compute_units,
            pass_pipeline=ct.PassPipeline.EMPTY,
        )

        # Core ML ouput
        y_cm = mlmodel.predict({'x': x, 'initial_h': initial_h})
        out_cm, h_cm = y_cm['gru_0_0'], y_cm['gru_0_1']

        # Check outputs
        np.testing.assert_allclose(out_cm, out_t, atol=0.01, rtol=0.1)
        np.testing.assert_allclose([h_cm], h_t, atol=0.01, rtol=0.1)
