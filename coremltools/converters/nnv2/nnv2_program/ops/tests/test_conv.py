from coremltools.converters.nnv2.nnv2_program.program import get_new_symbol
from . import _test_reqs
from ._test_reqs import *
backends = _test_reqs.backends

class TestConv:
    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
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
                 ['SAME', 'VALID'],
                 ['NHWC'],  # NCHW not supoported.
                 [(11, 12, 3, 2), (12, 11, 2, 3)],
                 [(1, 1), (2, 3)],
                 [(1, 1), (2, 3)],
             ))
    def test_tf_conv(self, use_cpu_only, backend, conv_dim,
            padding, data_format, HWkHkW, strides, dilations):
        H, W, kH, kW = HWkHkW
        N, C_in, C_out = 1, 2, 3
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
                run_compare_tf(graph, {x: np.random.rand(*input_shape)},
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
                run_compare_tf(graph, {x: np.random.rand(*input_shape),
                    W: np.random.rand(*W_shape)},
                    conv, use_cpu_only=use_cpu_only,
                    frontend_only=False, backend=backend)

        test_static_W()
        if dilations == (1, 1):
            # We do not support dynamic weight when dilation != 1.
            test_dynamic_W()

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.skip(reason='rdar://60100504')
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
                 ['SAME', 'VALID'],
                 ['NHWC'],  # NCHW not supoported.
                 [(11, 12, 3, 2), (12, 11, 2, 3)],
                 [(1, 1), (2, 3)],
                 [(1, 1), (2, 3)],
             ))
    def test_tf_keras(self, use_cpu_only, backend, conv_dim,
            padding, data_format, HWkHkW, strides, dilations):
        H, W, kH, kW = HWkHkW
        N, C_in, C_out = 1, 2, 3
        keras_data_format = 'channels_last' if data_format == 'NHWC' \
                else 'channels_first'
        input_shape = (N, W, C_in) if conv_dim == 'conv1d' \
                else (N, H, W, C_in)
        conv = tf.keras.layers.Conv2D(C_out, (kH, kW), strides,
                padding=padding, data_format=keras_data_format,
                dilation_rate=dilations)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = conv(x)
            run_compare_tf(graph, {x: np.random.rand(*input_shape)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize(
            ','.join([
                'use_cpu_only',
                'backend',
                'padding',
                'HWkHkW',
                'strides',
                'dilations']),
             itertools.product(
                 [True, False],
                 backends,
                 ['SAME', 'VALID'],
                 [(11, 12, 3, 2), (12, 11, 2, 3)],
                 # TF doesn't support non-square strides for depthwise
                 # https://github.com/tensorflow/tensorflow/issues/33005
                 [(1, 1, 1, 1), (1, 2, 2, 1)],
                 [(1, 1)], # Other dilation requires SpaceToBatchND
             ))
    def test_tf_depthwise_conv(self, use_cpu_only, backend,
            padding, HWkHkW, strides, dilations):
        H, W, kH, kW = HWkHkW
        N, C_in, C_out = 1, 2, 6
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
                run_compare_tf(graph, {x: np.random.rand(*input_shape)},
                               conv, use_cpu_only=use_cpu_only,
                               frontend_only=False, backend=backend)

        def test_dynamic_W():
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=input_shape)
                W = tf.placeholder(tf.float32, shape=W_shape)
                conv = tf.nn.depthwise_conv2d(x, W, strides=strides, padding=padding,
                    dilations=dilations, data_format=data_format)
                run_compare_tf(graph, {x: np.random.rand(*input_shape),
                    W: np.random.rand(*W_shape)},
                    conv, use_cpu_only=use_cpu_only,
                    frontend_only=False, backend=backend)

        test_static_W()
        if dilations == (1, 1):
            # We do not support dynamic weight when dilation != 1.
            test_dynamic_W()


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

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
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
                 ['NHWC'],  # NCHW not supoported by TF
                 [(12, 12, 2, 2), (5, 5, 3, 3)],
                 [(1, 1), (2, 2)],
                 [(1, 1)] # Dilation > 1 not supported by TF
             ))
    def test_tf(self, use_cpu_only, backend, conv_dim,
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
                run_compare_tf(graph, { x:x_input },
                               conv, use_cpu_only=use_cpu_only,
                               frontend_only=False, backend=backend)
        test_static_W()

    # Skipping following test rdar://60100504 ([NNv2] SSAv2 to NNv1 support for tf.keras.layers.ConvTranspose2d)
    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.skip(reason='rdar://60100504')
    @pytest.mark.parametrize(
        ','.join([
            'use_cpu_only',
            'backend',
            'conv_dim', # 1d or 2d conv
            'padding',
            'data_format',
            'HWkHkW',
            'output_padding',
            'strides',
            'dilations']),
            itertools.product(
                [True, False],
                backends,
                ['conv2d'], #, 'conv2d'],
                ['VALID', 'SAME'],
                ['channels_last'],  # NCHW not supoported by TF
                [(12, 12, 2, 2), (5, 5, 3, 3)],
                [(1, 1)],
                [(1, 1), (2, 2)],
                [(1, 1)] # Dilation > 1 not supported by TF
            ))
    def test_keras(self, use_cpu_only, backend, conv_dim,
                padding, data_format, HWkHkW, output_padding, strides, dilations):
        H, W, kH, kW = HWkHkW
        N, C_in, C_out = 1, 1, 2

        if data_format == 'channels_last':
            input_shape = (N, W, C_in) if conv_dim == 'conv1d' \
                    else (N, H, W, C_in)
        else: # 'NCHW'
            input_shape = (N, C_in, W) if conv_dim == 'conv1d' \
                    else (N, C_in, H, W)

        w_shape = (kW, C_out, C_in) if conv_dim == 'conv1d' \
                else (kH, kW, C_out, C_in)

        def test_static_W():
            x_input = np.random.randn(*input_shape)
            w_val = np.random.randn(*w_shape)
            if conv_dim == 'conv1d':
                conv = tf.keras.layers.Conv1DTranspose(C_out, kH, strides=strides[0], padding=padding,
                                                output_padding=output_padding, dilation_rate=dilations[0], data_format=data_format)
            else:
                conv = tf.keras.layers.Conv2DTranspose(C_out, (kH, kW), strides=strides, padding=padding,
                                                output_padding=output_padding, dilation_rate=dilations, data_format=data_format)
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=input_shape)
                w = tf.constant(w_val, tf.float32)
                conv(x)
                run_compare_tf(graph, { x:x_input },
                               conv, use_cpu_only=use_cpu_only,
                               frontend_only=False, backend=backend)
        test_static_W()
