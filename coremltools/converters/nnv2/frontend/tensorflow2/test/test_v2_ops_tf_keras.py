from coremltools.converters.nnv2 import testing_reqs
from coremltools.converters.nnv2.frontend.tensorflow2.test.testing_utils import run_compare_tf_keras
from coremltools.converters.nnv2.testing_reqs import *

backends = testing_reqs.backends

tf = pytest.importorskip('tensorflow', minversion='2.1.0')


class TestActivationReLU:

    @pytest.mark.parametrize('use_cpu_only, backend, rank',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 6)]))
    def test(self, use_cpu_only, backend, rank):
        shape = np.random.randint(low=1, high=6, size=rank)

        model = tf.keras.Sequential([
            tf.keras.layers.ReLU(batch_input_shape=shape)
        ])

        run_compare_tf_keras(
            model, [random_gen(shape, -10., 10.)],
            use_cpu_only=use_cpu_only, backend=backend)


class TestConvolution:
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
    def test_conv(
            self, use_cpu_only, backend, op, padding, data_format,
            HWkHkW, strides, dilations, batch_size):
        H, W, kH, kW = HWkHkW
        N, C_in, C_out = batch_size, 2, 3
        shape = (N, W, C_in) if op == tf.keras.layers.Conv1D else (N, H, W, C_in)
        model = tf.keras.Sequential([
            op(
                batch_input_shape=shape,
                filters=C_out,
                kernel_size=kW if op == tf.keras.layers.Conv1D else (kH, kW),
                strides=strides[0] if op == tf.keras.layers.Conv1D else strides,
                padding=padding.upper(), data_format=data_format,
                dilation_rate=dilations[0] if op == tf.keras.layers.Conv1D else dilations)
        ])

        run_compare_tf_keras(
            model, [random_gen(shape, rand_min=-10, rand_max=10)],
            use_cpu_only=use_cpu_only, backend=backend)

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
    def test_conv_transpose(
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
            model, [random_gen(shape, rand_min=-10, rand_max=10)],
            use_cpu_only=use_cpu_only, backend=backend)


class TestNormalization:
    @pytest.mark.parametrize(
        'use_cpu_only, backend, rank, axis, momentum, epsilon',
        itertools.product(
            [True, False],
            backends,
            [rank for rank in range(1, 6)],
            [0, -1],
            [0.99, 0.85],
            [1e-2, 1e-5],
        ))
    def test_batch_normalization(
            self, use_cpu_only, backend, rank, axis, momentum, epsilon):
        shape = np.random.randint(low=2, high=5, size=rank)
        model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(
                batch_input_shape=shape, axis=axis, momentum=momentum, epsilon=epsilon)
        ])
        run_compare_tf_keras(
            model, [random_gen(shape, rand_min=-10, rand_max=10)],
            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.parametrize(
        'use_cpu_only, backend, rank_and_axis, momentum, epsilon',
        itertools.product(
            [True, False],
            backends,
            [(4, 1), (4, -3)],
            [0.99, 0.85],
            [1e-2, 1e-5],
        ))
    def test_fused_batch_norm_v3(
            self, use_cpu_only, backend, rank_and_axis, momentum, epsilon):
        rank, axis = rank_and_axis
        shape = np.random.randint(low=2, high=5, size=rank)
        model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(
                batch_input_shape=shape, axis=axis, momentum=momentum, epsilon=epsilon)
        ])
        run_compare_tf_keras(
            model, [random_gen(shape, rand_min=-10, rand_max=10)],
            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.parametrize('use_cpu_only, backend, rank, axis, epsilon',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(3, 4)],
                                 [-1, ],
                                 [1e-10],
                             ))
    def test_layer_normalization(self, use_cpu_only, backend, rank, axis, epsilon):
        shape = np.random.randint(low=2, high=6, size=rank)
        model = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(
                batch_input_shape=shape, axis=axis, epsilon=epsilon, trainable=False)
        ])
        run_compare_tf_keras(
            model, [random_gen(shape, rand_min=-100, rand_max=100)],
            use_cpu_only=use_cpu_only, backend=backend)
