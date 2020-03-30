from coremltools.converters.nnv2.nnv2_program.program import get_new_symbol
from . import _test_reqs
from ._test_reqs import *
backends = _test_reqs.backends


class TestConv:
    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize(
            ','.join([
                'use_cpu_only',
                'backend',
                'conv_dim', # 1d or 2d conv
                'padding',
                'data_format',
                'HWkHkW',
                'strides',
                'dilations',
                'dynamic_weights',
                'batch_size']),
             itertools.product(
                 [True, False],
                 backends,
                 ['conv1d', 'conv2d'],
                 ['SAME', 'VALID'],
                 ['NHWC'],  # NCHW not supported by TF.
                 [(11, 12, 3, 2), (12, 11, 2, 3)],
                 [(1, 1), (2, 3)],
                 [(1, 1), (2, 3)],
                 [True, False],
                 [1, 3],
             ))
    def test_tf1(
            self, use_cpu_only, backend, conv_dim, padding, data_format,
            HWkHkW, strides, dilations, dynamic_weights, batch_size):
        H, W, kH, kW = HWkHkW
        N, C_in, C_out = batch_size, 2, 3
        if data_format == 'NHWC':
            input_shape = (N, W, C_in) if conv_dim == 'conv1d' \
                    else (N, H, W, C_in)
            if conv_dim == 'conv1d':
                data_format = 'NWC'
        else:  # 'NCHW'
            input_shape = (N, C_in, W) if conv_dim == 'conv1d' \
                    else (N, C_in, H, W)
            if conv_dim == 'conv1d':
                data_format = 'NCW'
        W_shape = (kW, C_in, C_out) if conv_dim == 'conv1d' \
                else (kH, kW, C_in, C_out)
        dilations = dilations[1] if conv_dim == 'conv1d' else dilations
        def test_static_W():
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=input_shape)
                W = tf.constant(np.random.rand(*W_shape), tf.float32)
                if conv_dim == 'conv1d':
                    conv = tf.nn.conv1d(x, W, stride=strides[1], padding=padding,
                        dilations=dilations, data_format=data_format)
                else:
                    conv = tf.nn.conv2d(x, W, strides=strides, padding=padding,
                        dilations=dilations, data_format=data_format)
                run_compare_tf1(graph, {x: np.random.rand(*input_shape)},
                                conv, use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)

        def test_dynamic_W():
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=input_shape)
                W = tf.placeholder(tf.float32, shape=W_shape)
                if conv_dim == 'conv1d':
                    # Notice stride vs strides argument
                    conv = tf.nn.conv1d(x, W, stride=strides[1], padding=padding,
                        dilations=dilations, data_format=data_format)
                else:
                    conv = tf.nn.conv2d(x, W, strides=strides, padding=padding,
                        dilations=dilations, data_format=data_format)
                run_compare_tf1(graph, {x: np.random.rand(*input_shape),
                                        W: np.random.rand(*W_shape)},
                                conv, use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)

        # We do not support dynamic weight when dilations != 1.
        test_dynamic_W() if dynamic_weights and dilations == (1, 1) else test_static_W()

    @pytest.mark.skipif(not HAS_TF2, reason=MSG_TF2_NOT_FOUND)
    @pytest.mark.parametrize(
        ','.join([
            'use_cpu_only',
            'backend',
            'op',
            'padding',
            'data_format',
            'HWkHkW',
            'strides',
            'dilations',
            'dynamic_weights',
            'batch_size']),
        itertools.product(
            [True, False],
            backends,
            [tf.nn.conv1d, tf.nn.conv2d],
            ['same', 'valid'],
            ['NHWC'],
            [(11, 12, 3, 2), (12, 11, 2, 3)],
            [(1, 1), (2, 3)],
            [(1, 1), (2, 3)],
            [True, False],
            [1, 3],
        ))
    def test_tf2(
            self, use_cpu_only, backend, op, padding, data_format,
            HWkHkW, strides, dilations, dynamic_weights, batch_size):
        H, W, kH, kW = HWkHkW
        N, C_in, C_out = batch_size, 2, 3
        if data_format == 'NHWC':
            shape = (N, W, C_in) if op == tf.nn.conv1d else (N, H, W, C_in)
            data_format = 'NWC' if op == tf.nn.conv1d else data_format
        else:  # 'NCHW'
            shape = (N, C_in, W) if op == tf.nn.conv1d else (N, C_in, H, W)
            data_format = 'NCW' if op == tf.nn.conv1d else data_format

        W_shape = (kW, C_in, C_out) if op == tf.nn.conv1d else (kH, kW, C_in, C_out)

        def test_static_W():
            class model(tf.Module):
                @tf.function(input_signature=[
                    tf.TensorSpec(shape=shape, dtype=tf.float32),
                ])
                def __call__(self, x):

                    return op(
                        x, tf.constant(np.random.rand(*W_shape), tf.float32),
                        strides[1] if op == tf.nn.conv1d else strides,
                        padding=padding.upper(),
                        dilations=dilations[1] if op == tf.nn.conv1d else dilations,
                        data_format=data_format)

            run_compare_tf2(
                model, input_values=[random_gen(shape, rand_min=-10, rand_max=10)],
                use_cpu_only=use_cpu_only, backend=backend)

        def test_dynamic_W():

            class model(tf.Module):
                @tf.function(input_signature=[
                    tf.TensorSpec(shape=shape, dtype=tf.float32),
                    tf.TensorSpec(shape=W_shape, dtype=tf.float32)
                ])
                def __call__(self, x, weights):

                    return op(
                        x, weights,
                        strides[1] if op == tf.nn.conv1d else strides,
                        padding=padding.upper(),
                        dilations=dilations[1] if op == tf.nn.conv1d else dilations,
                        data_format=data_format)

            run_compare_tf2(
                model, input_values=[
                    random_gen(shape, rand_min=-10, rand_max=10),
                    random_gen(W_shape, rand_min=-10, rand_max=10)],
                use_cpu_only=use_cpu_only, backend=backend)

        # We do not support dynamic weight when dilations != 1.
        test_dynamic_W() if dynamic_weights and dilations == (1, 1) else test_static_W()

    @pytest.mark.skipif(not HAS_TF2, reason=MSG_TF2_NOT_FOUND)
    @pytest.mark.parametrize(
        ','.join([
            'use_cpu_only',
            'backend',
            'op',
            'padding',
            'data_format',
            'HWkHkW',
            'strides',
            'dilations',
            'batch_size']),
        itertools.product(
            [True, False],
            backends,
            [tf.keras.layers.Conv1D, tf.keras.layers.Conv2D],
            ['same', 'valid'],
            ['channels_last'],
            [(11, 12, 3, 2), (12, 11, 2, 3)],
            [(1, 1), (2, 3), (3, 2)],
            [(1, 1)],  # rdar://60668562 (NNv2: Conversion for TF op 'SpaceToBatchND' not implemented.)
            [1, 3]
        ))
    def test_tf_keras(
            self, use_cpu_only, backend, op, padding, data_format,
            HWkHkW, strides, dilations, batch_size):
        H, W, kH, kW = HWkHkW
        N, C_in, C_out = batch_size, 2, 3
        shape = (N, W, C_in) if op == tf.keras.layers.Conv1D else (N, H, W, C_in)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=shape[1:], batch_size=shape[0]),
            op(
                filters=C_out,
                kernel_size=kW if op == tf.keras.layers.Conv1D else (kH, kW),
                strides=strides[0] if op == tf.keras.layers.Conv1D else strides,
                padding=padding.upper(), data_format=data_format,
                dilation_rate=dilations[0] if op == tf.keras.layers.Conv1D else dilations)
        ])

        run_compare_tf_keras(
            model, input_values=[random_gen(shape, rand_min=-10, rand_max=10)],
            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize(
        ','.join([
            'use_cpu_only',
            'backend',
            'padding',
            'HWkHkW',
            'strides',
            'dilations',
            'dynamic_weights',
            'batch_size']),
        itertools.product(
            [True, False],
            backends,
            ['SAME', 'VALID'],
            [(11, 12, 3, 2), (12, 11, 2, 3)],
            # TF doesn't support non-square strides for depthwise
            # https://github.com/tensorflow/tensorflow/issues/33005
            [(1, 1, 1, 1), (1, 2, 2, 1)],
            [(1, 1)],  # rdar://60668562 (NNv2: Conversion for TF op 'SpaceToBatchND' not implemented.)
            [True, False],
            [1, 3]
        ))
    def test_tf1_depthwise_conv(
            self, use_cpu_only, backend, padding, HWkHkW, strides,
            dilations, dynamic_weights, batch_size):
        H, W, kH, kW = HWkHkW
        N, C_in, C_out = batch_size, 2, 6
        input_shape = (N, H, W, C_in)
        data_format = 'NHWC'
        assert C_out % C_in == 0
        multiplier = int(C_out / C_in)
        W_shape = (kH, kW, C_in, multiplier)
        def test_static_W():
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=input_shape)
                W = tf.constant(np.random.rand(*W_shape), tf.float32)
                conv = tf.nn.depthwise_conv2d(x, W, strides=strides, padding=padding,
                    dilations=dilations, data_format=data_format)
                run_compare_tf1(graph, {x: np.random.rand(*input_shape)},
                                conv, use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)

        def test_dynamic_W():
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=input_shape)
                W = tf.placeholder(tf.float32, shape=W_shape)
                conv = tf.nn.depthwise_conv2d(x, W, strides=strides, padding=padding,
                    dilations=dilations, data_format=data_format)
                run_compare_tf1(graph, {x: np.random.rand(*input_shape),
                                        W: np.random.rand(*W_shape)},
                                conv, use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)

        # We do not support dynamic weight when dilations != 1.
        test_dynamic_W() if dynamic_weights and dilations == (1, 1) else test_static_W()


class TestConvTranspose:
    @pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch not installed.")
    @pytest.mark.parametrize(
            ','.join([
                'use_cpu_only',
                'backend',
                'conv_dim', # 1d or 2d conv
                'padding',
                'HWK',
                'stride',
                'dilation',
                'has_bias',
                'groups',
                'symbolic']),
             itertools.product(
                 [True, False],
                 backends,
                 ['conv1d', 'conv2d'],
                 [(1, 1), (2, 3)],
                 [(12, 12, 2), (5, 5, 4)],
                 [1, 2],
                 [1, 2],
                 [True, False],
                 [1, 2],
                 [True, False]
             ))
    def test_builder_to_backend_stress(self, use_cpu_only, backend, conv_dim, padding,
                                       HWK, stride, dilation, has_bias, groups, symbolic):
        H, W, K = HWK
        N, C_in, C_out = 1, 1, 2

        import torch
        import torch.nn as nn

        isConv1d = True if conv_dim == 'conv1d' else False
        if isConv1d:
            m = nn.ConvTranspose1d(C_in, C_out, K, stride=stride, dilation=dilation, bias=has_bias, padding=padding[0])
            input_shape = [N, C_in, H]
            padding = [padding[0], padding[0]]
            strides = [stride]
            dilations = [dilation]
        else:
            m = nn.ConvTranspose2d(C_in, C_out, K, stride=stride, dilation=dilation, bias=has_bias, padding=padding)
            input_shape = [N, C_in, H, W]
            padding = [padding[0], padding[0], padding[1], padding[1]]
            strides = [stride] * 2
            dilations = [dilation] * 2

        wts = m.state_dict()
        weight = wts['weight'].detach().numpy()
        bias = wts['bias'].detach().numpy() if has_bias else None

        # Reshape to CoreML format
        # PyTorch weight format: Cin, Cout, H, W
        # CoreML weight format: H, W, Cout, Cin
        if isConv1d:
            weight = np.transpose(weight, [2, 1, 0])
        else:
            weight = np.transpose(weight, [2, 3, 1, 0])

        input = torch.randn(*input_shape)
        output = m(input)
        output = output.detach().numpy()
        input = input.detach().numpy()

        output_shape = list(output.shape)
        if symbolic:
            # For symbolic input test
            # Make Batch Size and input channel as symbolic
            symbolic_batch_size = get_new_symbol()
            input_shape[0] = symbolic_batch_size
            output_shape[0] = symbolic_batch_size

        expected_output_types = tuple(output_shape[:]) + (builtins.fp32,)
        expected_outputs = [output]

        input_placeholders = {"x": cb.placeholder(shape=input_shape)}
        input_values = {"x": input}

        def build(x):
            arguments = {
                        "x":x,
                        "weight":weight,
                        "pad":padding,
                        "strides":strides,
                        "dilations":dilations,
                        }
            if has_bias:
                arguments["bias"] = bias
            return cb.conv_transpose(**arguments)
        run_compare_builder(build, input_placeholders, input_values,
                    expected_output_types, expected_outputs,
                    use_cpu_only=use_cpu_only, frontend_only=False,
                    backend=backend)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    @pytest.mark.parametrize(
            ','.join([
                'use_cpu_only',
                'backend',
                'conv_dim', # 1d or 2d conv
                'padding',
                'data_format',
                'HWkHkW',
                'strides',
                'dilations']),
             itertools.product(
                 [True, False],
                 backends,
                 ['conv1d', 'conv2d'],
                 ['VALID', 'SAME'],
                 ['NHWC'],  # NCHW not supported by TF
                 [(12, 12, 2, 2), (5, 5, 3, 3)],
                 [(1, 1), (2, 2)],
                 [(1, 1)] # Dilation > 1 not supported by TF
             ))
    def test_tf1(self, use_cpu_only, backend, conv_dim,
                padding, data_format, HWkHkW, strides, dilations):
        H, W, kH, kW = HWkHkW
        N, C_in, C_out = 1, 1, 2

        if padding == 'SAME':
            oH = H * strides[0]
            oW = W * strides[1]
        else:
            oH = (H - 1) * strides[0] + (kH - 1) * dilations[0] + 1
            oW = (W - 1) * strides[1] + (kW - 1) * dilations[1] + 1

        if data_format == 'NHWC':
            input_shape = (N, W, C_in) if conv_dim == 'conv1d' \
                    else (N, H, W, C_in)
            if conv_dim == 'conv1d':
                data_format = 'NWC'
            output_shape = [N, oH, C_out] if conv_dim == 'conv1d' else [N, oH, oW, C_out]
        else: # 'NCHW'
            input_shape = (N, C_in, W) if conv_dim == 'conv1d' \
                    else (N, C_in, H, W)
            if conv_dim == 'conv1d':
                data_format = 'NCW'
            output_shape = [N, C_out, oH] if conv_dim == 'conv1d' else [N, C_out, oH, oW]

        w_shape = (kW, C_out, C_in) if conv_dim == 'conv1d' \
                else (kH, kW, C_out, C_in)

        def test_static_W():
            x_input = np.random.randn(*input_shape)
            w_val = np.random.randn(*w_shape)
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=input_shape)
                w = tf.constant(w_val, tf.float32)
                if conv_dim == 'conv1d':
                    conv = tf.nn.conv1d_transpose(x, w, output_shape=output_shape, strides=strides[0], padding=padding,
                                                  dilations=dilations[0], data_format=data_format)
                else:
                    conv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=strides, padding=padding,
                                                  dilations=dilations, data_format=data_format)
                run_compare_tf1(graph, {x:x_input},
                                conv, use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)
        test_static_W()

    @pytest.mark.skipif(not HAS_TF2, reason=MSG_TF2_NOT_FOUND)
    @pytest.mark.parametrize(
        ','.join([
            'use_cpu_only',
            'backend',
            'padding',
            'data_format',
            'HWkHkW',
            'output_padding',
            'strides',
            'dilations',
            'batch_size']),
        itertools.product(
            [True, False],
            backends,
            ['same', 'valid'],
            ['channels_last'],
            [(11, 12, 2, 2), (5, 7, 3, 3)],
            [(1, 1)],
            [(2, 2), (3, 3)],
            [(1, 1)],  # Dilation > 1 not supported by TF
            [1, 3]
        ))
    def test_tf_keras(
            self, use_cpu_only, backend, padding, data_format, HWkHkW,
            output_padding, strides, dilations, batch_size):
        H, W, kH, kW = HWkHkW
        N, C_in, C_out = batch_size, 1, 2

        shape = (N, H, W, C_in) if data_format == 'channels_last' else (N, C_in, H, W)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=shape[1:], batch_size=shape[0]),
            tf.keras.layers.Conv2DTranspose(
                filters=C_out, kernel_size=(kH, kW), strides=strides,
                padding=padding.upper(), output_padding=output_padding,
                data_format=data_format, dilation_rate=dilations)
        ])
        run_compare_tf_keras(
            model, input_values=[random_gen(shape, rand_min=-10, rand_max=10)],
            use_cpu_only=use_cpu_only, backend=backend)
