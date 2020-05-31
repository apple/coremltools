from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.mil import get_new_symbol
from coremltools.converters.mil.testing_reqs import *

from .testing_utils import run_compare_builder

backends = testing_reqs.backends


class TestGRU:
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed.")
    @pytest.mark.parametrize(argnames=["use_cpu_only", "backend", "seq_len",
                             "batch_size", "input_size", "hidden_size", "has_bias",
                             "output_sequence", "direction", "symbolic"],
                             argvalues=
                             itertools.product(
                                [True, False],
                                backends,
                                [1, 8],
                                [1], # <rdar://problem/59644603> [MIL] GRU with batch size 1 produces incorrect
                                     # output(always 0) for second batch onwards
                                [2, 32],
                                [1, 16],
                                [False, False],
                                [True, False],
                                ["forward", "reverse"],
                                [False, False]
                                )
                            )
    def test_builder_to_backend_smoke(self, use_cpu_only, backend, seq_len, batch_size, input_size,
                                              hidden_size, has_bias, output_sequence, direction, symbolic):
        torch.manual_seed(5)
        rnn = torch.nn.GRU(input_size, hidden_size, 1, bias=has_bias)
        state_dict = rnn.state_dict()

        state_dict['weight_ih_l0'] /= state_dict['weight_ih_l0']
        state_dict['weight_hh_l0'] /= state_dict['weight_hh_l0']

        ih_wt = state_dict['weight_ih_l0'].detach().numpy()
        hh_wt = state_dict['weight_hh_l0'].detach().numpy()

        # Make weight compatible to CoreML format
        def rzo_to_zro(x):
            r, z, o = np.split(x, 3)
            return np.concatenate([z, r, o], axis=0)

        w = np.concatenate([ih_wt, hh_wt], axis=1)
        w = rzo_to_zro(w).transpose()

        b = None
        if has_bias:
            ih_b = state_dict['bias_ih_l0'].detach().numpy()
            hh_b = state_dict['bias_hh_l0'].detach().numpy()
            ih_b = rzo_to_zro(ih_b)
            hh_b = rzo_to_zro(hh_b)
            b = np.stack([ih_b, hh_b], axis=0)

        t = torch.randn(seq_len, batch_size, input_size)
        h0 = torch.randn(1, batch_size, hidden_size)

        n_t = t
        if direction == "reverse":
            n_t = torch.flip(n_t, [0])

        output, hn = rnn(n_t, h0)
        if output_sequence == False:
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

        expected_output_types = [(seq_len if output_sequence else 1, batch_size, hidden_size, types.fp32),
                                 (batch_size, hidden_size, types.fp32)]
        expected_outputs = [output, hn]

        input_placeholders = {"x": mb.placeholder(shape=input_shape),
                              "initial_h": mb.placeholder(shape=h_shape)}
        input_values = {"x": t, "initial_h": h}

        def build(x, initial_h):
            arguments = {
                        "x":x,
                        "initial_h":initial_h,
                        "weight":w,
                        "direction":direction,
                        "output_sequence":output_sequence,
                        }
            # If bias is provided, add in arguments
            if b is not None:
                arguments["bias"] = b
            return mb.gru(**arguments)
        run_compare_builder(build, input_placeholders, input_values,
                    expected_output_types, expected_outputs,
                    use_cpu_only=use_cpu_only, frontend_only=False,
                    backend=backend)


class TestLSTM:
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed.")
    @pytest.mark.xfail(reason="rdar://62272632")
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_zeros(self, use_cpu_only, backend):
        seq_len, b, input_dim, hidden_dim = 1, 1, 2, 3
        t = np.zeros((1, b, input_dim)).astype(np.float32)
        input_placeholders = {"x": mb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            h_all, ht, ct = mb.lstm(x=x,
                    initial_h=np.zeros((b, hidden_dim)).astype(np.float32),
                    initial_c=np.zeros((b, hidden_dim)).astype(np.float32),
                    weight=np.zeros((input_dim+hidden_dim,
                        4*hidden_dim)).astype(np.float32),
                    direction='forward',
                    bias=np.zeros((2, 4*hidden_dim)).astype(np.float32),
                    activations=('sigmoid', 'tanh', 'sigmoid'),
                    )
            return ht
        expected_output_types = (b, hidden_dim, types.fp32)
        expected_outputs = np.zeros((b, hidden_dim)).astype(np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)


    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed.")
    @pytest.mark.parametrize(argnames=["use_cpu_only", "backend", "seq_len",
                             "batch_size", "input_size", "hidden_size", "has_bias",
                             "output_sequence", "direction", "symbolic"],
                             argvalues=
                             itertools.product(
                                [True, False],
                                backends,
                                [1, 8],
                                [1, 32],
                                [1, 64],
                                [1, 16],
                                [True, False],
                                [True, False],
                                ["forward", "reverse"],
                                [True, False]
                                )
                            )
    def test_builder_to_backend_smoke_unilstm(self, use_cpu_only, backend, seq_len, batch_size, input_size,
                                              hidden_size, has_bias, output_sequence, direction, symbolic):
        # TODO: <rdar://problem/59540160> [MIL] LSTM layer- Implement eval and tf register routine
        # Testing 1. peephole values
        #         2. clip values

        torch.manual_seed(50)
        rnn = torch.nn.LSTM(input_size, hidden_size, 1, bias=has_bias)
        state_dict = rnn.state_dict()

        ih_wt = state_dict['weight_ih_l0'].detach().numpy()
        hh_wt = state_dict['weight_hh_l0'].detach().numpy()

        # Make weight compatible to CoreML format
        def ifzo_to_ifoz(x):
            i, f, z, o = np.split(x, 4)
            return np.concatenate([i, f, o, z], axis=0)

        w = np.concatenate([ih_wt, hh_wt], axis=1)
        w = ifzo_to_ifoz(w).transpose()

        b = None
        if has_bias:
            ih_b = state_dict['bias_ih_l0'].detach().numpy()
            hh_b = state_dict['bias_hh_l0'].detach().numpy()
            ih_b = ifzo_to_ifoz(ih_b).transpose()
            hh_b = ifzo_to_ifoz(hh_b).transpose()
            b = np.stack([ih_b, hh_b], axis=0)

        t = torch.randn(seq_len, batch_size, input_size)
        h0 = torch.randn(1, batch_size, hidden_size)
        c0 = torch.randn(1, batch_size, hidden_size)

        n_t = t
        if direction == "reverse":
            n_t = torch.flip(n_t, [0])

        output, (hn, cn) = rnn(n_t, (h0, c0))
        if output_sequence == False:
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

        expected_output_types = [(seq_len if output_sequence else 1, batch_size, hidden_size, types.fp32),
                                 (batch_size, hidden_size, types.fp32),
                                 (batch_size, hidden_size, types.fp32)]
        expected_outputs = [output, hn, cn]

        input_placeholders = {"x": mb.placeholder(shape=input_shape),
                              "initial_h": mb.placeholder(shape=h_shape),
                              "initial_c": mb.placeholder(shape=c_shape),}
        input_values = {"x": t, "initial_h": h, "initial_c": c}

        def build(x, initial_h, initial_c):
            arguments = {
                        "x":x,
                        "initial_h":initial_h,
                        "initial_c":initial_c,
                        "weight":w,
                        "direction":direction,
                        "output_sequence":output_sequence,
                        }
            # If bias is provided, add in arguments
            if b is not None:
                arguments["bias"] = b
            return mb.lstm(**arguments)
        run_compare_builder(build, input_placeholders, input_values,
                    expected_output_types, expected_outputs,
                    use_cpu_only=use_cpu_only, frontend_only=False,
                    backend=backend)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed.")
    @pytest.mark.parametrize(argnames=["use_cpu_only", "backend", "seq_len", "batch_size", "input_size",
                                       "hidden_size", "has_bias", "output_sequence", "symbolic"],
                             argvalues=
                             itertools.product(
                                [True],
                                backends,
                                [1, 8],
                                [1, 32],
                                [1, 64],
                                [2, 16],
                                [True, False],
                                [True, False],
                                [True, False]
                                )
                            )
    def test_builder_to_backend_smoke_bidirlstm(self, use_cpu_only, backend, seq_len, batch_size, input_size,
                                                hidden_size, has_bias, output_sequence, symbolic):
        def _pytorch_hidden_to_coreml(x):
            x = x.detach().numpy()
            # Split of Direction axis
            f, b = np.split(x, 2, axis=0)
            # Concat on Hidden Size axis
            x    = np.concatenate([f, b], axis=2)
            x    = np.squeeze(x, axis=0)
            return x
        direction = "bidirectional"
        torch.manual_seed(20)
        rnn = torch.nn.LSTM(input_size, hidden_size, 1, bidirectional=True, bias=has_bias)
        state_dict = rnn.state_dict()

        ih_wt = state_dict['weight_ih_l0'].detach().numpy()
        hh_wt = state_dict['weight_hh_l0'].detach().numpy()
        ih_wt_r = state_dict['weight_ih_l0_reverse'].detach().numpy()
        hh_wt_r = state_dict['weight_hh_l0_reverse'].detach().numpy()

        f_wt = np.concatenate([ih_wt, hh_wt], axis=1)
        r_wt = np.concatenate([ih_wt_r, hh_wt_r], axis=1)

        def ifzo_to_ifoz(x):
            i, f, z, o = np.split(x, 4)
            return np.concatenate([i, f, o, z], axis=0)

        f_wt = ifzo_to_ifoz(f_wt).transpose()
        r_wt = ifzo_to_ifoz(r_wt).transpose()
        w = np.concatenate([f_wt, r_wt], axis=1)

        b = None
        if has_bias:
            ih_b = state_dict['bias_ih_l0'].detach().numpy()
            hh_b = state_dict['bias_hh_l0'].detach().numpy()
            ih_b_r = state_dict['bias_ih_l0_reverse'].detach().numpy()
            hh_b_r = state_dict['bias_hh_l0_reverse'].detach().numpy()
            # Convert forward bias into [2, 4*H]
            ih_b = ifzo_to_ifoz(ih_b)
            hh_b = ifzo_to_ifoz(hh_b)
            f_b = np.stack([ih_b, hh_b], axis=0)
            # Convert reverse bias into [2, 4*H]
            ih_b_r = ifzo_to_ifoz(ih_b_r)
            hh_b_r = ifzo_to_ifoz(hh_b_r)
            r_b = np.stack([ih_b_r, hh_b_r], axis=0)
            # Final bias of [2, 2*4*H]
            b = np.concatenate([f_b, r_b], axis=1)

        t = torch.randn(seq_len, batch_size, input_size)
        h0 = torch.randn(2, batch_size, hidden_size)
        c0 = torch.randn(2, batch_size, hidden_size)

        output, (hn, cn) = rnn(t, (h0, c0))
        if output_sequence == False:
            output_f = output[-1].unsqueeze(0)[:,:,:hidden_size]
            output_r = output[0].unsqueeze(0)[:,:,hidden_size:]
            output = torch.cat([output_f, output_r], dim=2)

        output = output.detach().numpy()
        hn = _pytorch_hidden_to_coreml(hn)
        cn = _pytorch_hidden_to_coreml(cn)

        if symbolic:
            batch_size = get_new_symbol()
            seq_len = get_new_symbol()

        input_shape = [seq_len, batch_size, input_size]
        h_shape = [batch_size, 2*hidden_size]
        c_shape = [batch_size, 2*hidden_size]

        expected_output_types = [(seq_len if output_sequence else 1, batch_size, 2*hidden_size, types.fp32),
                                 (batch_size, 2*hidden_size, types.fp32),
                                 (batch_size, 2*hidden_size, types.fp32)]
        expected_outputs = [output, hn, cn]

        t = t.detach().numpy()
        h = _pytorch_hidden_to_coreml(h0)
        c = _pytorch_hidden_to_coreml(c0)

        input_placeholders = {"x": mb.placeholder(shape=input_shape),
                              "initial_h": mb.placeholder(shape=h_shape),
                              "initial_c": mb.placeholder(shape=c_shape)}
        input_values = {"x": t, "initial_h": h, "initial_c": c}

        def build(x, initial_h, initial_c):
            arguments = {
                        "x":x,
                        "initial_h":initial_h,
                        "initial_c":initial_c,
                        "weight":w,
                        "direction":direction,
                        "output_sequence":output_sequence,
                        }
            # If bias is provided, add in arguments
            if b is not None:
                arguments["bias"] = b
            return mb.lstm(**arguments)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)


class TestRNN:
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed.")
    @pytest.mark.parametrize(argnames=["use_cpu_only", "backend", "seq_len",
                             "batch_size", "input_size", "hidden_size", "has_bias",
                             "output_sequence", "direction", "symbolic"],
                             argvalues=
                             itertools.product(
                                [True, False],
                                backends,
                                [2, 8],
                                [1, 32],
                                [1, 64],
                                [1, 16],
                                [True, False],
                                [True, False],
                                ["forward", "reverse"],
                                [True, False]
                                )
                            )
    def test_builder_to_backend_smoke(self, use_cpu_only, backend, seq_len, batch_size, input_size,
                                              hidden_size, has_bias, output_sequence, direction, symbolic):
        torch.manual_seed(50)
        rnn = torch.nn.RNN(input_size, hidden_size, 1, bias=has_bias)
        state_dict = rnn.state_dict()

        ih_wt = state_dict['weight_ih_l0'].detach().numpy()
        hh_wt = state_dict['weight_hh_l0'].detach().numpy()
        w = np.concatenate([ih_wt, hh_wt], axis=1).transpose()

        b = None
        if has_bias:
            ih_b = state_dict['bias_ih_l0'].detach().numpy()
            hh_b = state_dict['bias_hh_l0'].detach().numpy()
            b = np.stack([ih_b, hh_b], axis=0)

        t = torch.randn(seq_len, batch_size, input_size)
        h0 = torch.randn(1, batch_size, hidden_size)

        n_t = t
        if direction == "reverse":
            n_t = torch.flip(n_t, [0])

        output, hn = rnn(n_t, h0)
        if output_sequence == False:
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

        expected_output_types = [(seq_len if output_sequence else 1, batch_size, hidden_size, types.fp32),
                                 (batch_size, hidden_size, types.fp32)]
        expected_outputs = [output, hn]

        input_placeholders = {"x": mb.placeholder(shape=input_shape),
                              "initial_h": mb.placeholder(shape=h_shape)}
        input_values = {"x": t, "initial_h": h}

        def build(x, initial_h):
            arguments = {
                        "x":x,
                        "initial_h":initial_h,
                        "weight":w,
                        "direction":direction,
                        "output_sequence":output_sequence,
                        }
            # If bias is provided, add in arguments
            if b is not None:
                arguments["bias"] = b
            return mb.rnn(**arguments)

        run_compare_builder(build, input_placeholders, input_values,
                    expected_output_types, expected_outputs,
                    use_cpu_only=use_cpu_only, frontend_only=False,
                    backend=backend)
