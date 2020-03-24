from . import _test_reqs
from ._test_reqs import *
backends = _test_reqs.backends

class TestNormalizationBatchNorm:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([[[[-16., 13.], [11., -16.]],
                           [[13., -15.], [13., 9.]],
                           [[-9., -4.], [-6., 3.]]]], dtype=np.float32)
        mean_val = np.array([9., 6., 3.], dtype=np.float32)
        variance_val = np.array([6., 1., 7.], dtype=np.float32)
        gamma_val = np.array([1., 1., 1.], dtype=np.float32)
        beta_val = np.array([1., 3., 0.], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.batch_norm(
                    x=x, mean=mean_val, variance=variance_val),
                cb.batch_norm(
                    x=x, mean=mean_val, variance=variance_val,
                    gamma=gamma_val, beta=beta_val, epsilon=1e-4)
            ]

        expected_output_types = [
            (1, 3, 2, 2, builtins.fp32),
            (1, 3, 2, 2, builtins.fp32),
        ]
        expected_outputs = [
            np.array([[[[-10.206199, 1.6329918],
                        [0.8164959, -10.206199]],
                       [[6.999965, -20.999895],
                        [6.999965, 2.9999852]],
                       [[-4.53557, -2.6457493],
                        [-3.4016776, 0.]]]], dtype=np.float32),
            np.array([[[[-9.206122, 2.6329796],
                        [1.8164899, -9.206122]],
                       [[9.99965, -17.998951],
                        [9.99965, 5.9998503]],
                       [[-4.535541, -2.6457324],
                        [-3.4016557, 0.]]]], dtype=np.float32)
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, rank, shape_mode, epsilon',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(3, 6)],
                                 [True, False],
                                 [1e-1, 1e-10]
                             ))
    def test_tf(self, use_cpu_only, backend, rank, shape_mode, epsilon):
        shape = np.random.randint(low=1, high=6, size=rank)
        if shape_mode:
            # same shape with 1 for being normalized over
            attr_shape = list(shape)
            attr_shape[1] = 1
            attr_shape[2] = 1
        else:
            # 1D tensor of the same size as channel dimension
            attr_shape = list(shape)[-1]
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            m = tf.placeholder(tf.float32, shape=attr_shape)
            v = tf.placeholder(tf.float32, shape=attr_shape)
            o = tf.placeholder(tf.float32, shape=attr_shape)
            s = tf.placeholder(tf.float32, shape=attr_shape)
            ref = tf.nn.batch_normalization(
                x, mean=m, variance=v, offset=o, scale=s, variance_epsilon=epsilon)
            run_compare_tf(graph, {
                x: random_gen(shape=shape, rand_min=-100., rand_max=100.),
                m: random_gen(shape=attr_shape, rand_min=-1., rand_max=1.),
                v: random_gen(shape=attr_shape, rand_min=0., rand_max=10.),
                o: random_gen(shape=attr_shape, rand_min=1., rand_max=10.),
                s: random_gen(shape=attr_shape, rand_min=-1., rand_max=1.),
            }, ref, use_cpu_only=use_cpu_only, backend=backend, atol=1e-2, rtol=1e-3)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, rank, shape_mode, epsilon, scale_after_normalization',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(3, 6)],
                                 [True, False],
                                 [1e-1, 1e-10],
                                 [True, False],
                             ))
    def test_tf_batch_norm_with_global_normalization(
            self, use_cpu_only, backend, rank, shape_mode, epsilon, scale_after_normalization):
        shape = np.random.randint(low=1, high=6, size=rank)
        if shape_mode:
            # same shape with 1 for being normalized over
            attr_shape = list(shape)
            attr_shape[1] = 1
            attr_shape[2] = 1
        else:
            # 1D tensor of the same size as channel dimension
            attr_shape = list(shape)[-1]
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            m = tf.placeholder(tf.float32, shape=attr_shape)
            v = tf.placeholder(tf.float32, shape=attr_shape)
            ref = tf.nn.batch_norm_with_global_normalization(
                x, mean=m, variance=v, variance_epsilon=epsilon,
                scale_after_normalization=scale_after_normalization)
            run_compare_tf(graph, {
                x: random_gen(shape=shape, rand_min=-100., rand_max=100.),
                m: random_gen(shape=attr_shape, rand_min=-1., rand_max=1.),
                v: random_gen(shape=attr_shape, rand_min=0., rand_max=10.),
            }, ref, use_cpu_only=use_cpu_only, backend=backend, atol=1e-2, rtol=1e-3)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, epsilon',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [1e-1, 1e-10]
                             ))
    def test_tf_fused_batch_norm(self, use_cpu_only, backend, epsilon):
        # TensorFlow's FusedBatchNorm is only for 4D inputs
        shape = np.random.randint(low=1, high=6, size=4)
        attr_shape = list(shape)[-1]
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            m = tf.constant(random_gen(shape=attr_shape, rand_min=-1., rand_max=1.))
            v = tf.constant(random_gen(shape=attr_shape, rand_min=0., rand_max=10.))
            o = tf.constant(random_gen(shape=attr_shape, rand_min=1., rand_max=10.))
            s = tf.constant(random_gen(shape=attr_shape, rand_min=-1., rand_max=1.))
            ref = tf.nn.fused_batch_norm(x, mean=m, variance=v, offset=o, scale=s,
                                         epsilon=epsilon, is_training=False)[0]
            run_compare_tf(graph, {
                x: random_gen(shape=shape, rand_min=-100., rand_max=100.),
            }, ref, use_cpu_only=use_cpu_only, backend=backend, atol=1e-2, rtol=1e-3)


class TestNormalizationInstanceNorm:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([[[[-16., 13.], [11., 16.]],
                           [[13., 15.], [13., 9.]],
                           [[-9., 4.], [-6., 3.]]]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return cb.instance_norm(x=x, epsilon=1e-2)

        expected_output_types = [(1, 3, 2, 2, builtins.fp32)]
        expected_outputs = [
            np.array([[[[-1.71524656, 0.54576027],
                        [0.38982874, 0.77965748]],
                       [[0.22917463, 1.14587319],
                        [0.22917463, -1.60422242]],
                       [[-1.2470212, 1.06887531],
                        [-0.71258354, 0.89072943]]]], dtype=np.float32)
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_PYTORCH, reason='PyTorch not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, epsilon',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [1e-5, 1e-10],
                             ))
    def test_builder_to_backend_stress(self, use_cpu_only, backend, epsilon):
        shape = np.random.randint(low=1, high=6, size=4)
        x_val = random_gen(shape=shape, rand_min=-10., rand_max=10.)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return cb.instance_norm(x=x, epsilon=epsilon)

        torch_op = torch.nn.InstanceNorm2d(num_features=shape[1])
        expected_outputs = [torch_op(torch.as_tensor(x_val)).numpy()]
        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend,
                            atol=1e-2, rtol=1e-3)

class TestNormalizationL2Norm:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([[[1., -7.], [5., -6.], [-3., -5.]]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [cb.l2_norm(x=x, axes=[-1], epsilon=1e-10)]

        expected_output_types = [(1, 3, 2, builtins.fp32)]
        expected_outputs = [
            np.array([[[0.08304548, -0.58131838],
                       [0.41522741, -0.4982729],
                       [-0.24913645, -0.41522741]]], dtype=np.float32)
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, rank, axes, epsilon',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(3, 6)],
                                 [(-1,), (-2,), (0, 1)],
                                 [1e-5, 1e-10],
                             ))
    def test_tf(self, use_cpu_only, backend, rank, axes, epsilon):
        shape = np.random.randint(low=1, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.math.l2_normalize(x, axis=axes, epsilon=epsilon)
            run_compare_tf(graph, {x: random_gen(shape, rand_min=-10, rand_max=10)},
                           ref, use_cpu_only=use_cpu_only, backend=backend,
                           atol=1e-2, rtol=1e-3)


class TestNormalizationLayerNorm:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([[[1., -7.], [5., -6.], [-3., -5.]]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                # V2->V1 lowering (op_mappings.py): if branch
                cb.layer_norm(x=x, axes=[2], epsilon=1e-4),
                # V2->V1 lowering (op_mappings.py): else branch
                cb.layer_norm(x=x, axes=[-2, -1], epsilon=1e-4),
            ]

        expected_output_types = [(1, 3, 2, builtins.fp32), (1, 3, 2, builtins.fp32)]
        expected_outputs = [
            np.array([[[0.9999969, -0.9999969],
                       [0.99999839, -0.99999839],
                       [0.99995005, -0.99995005]]], dtype=np.float32),
            np.array([[[0.8268512, -1.0630943],
                       [1.771824, -0.8268511],
                       [-0.11812156, -0.590608]]], dtype=np.float32)
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        def np_layer_norm(x, axes, gamma, beta, epsilon=1e-5):
            normalized_shape = x.shape[-len(axes):]
            gamma = np.ones(shape=normalized_shape) if gamma is None else gamma
            beta = np.zeros(shape=normalized_shape) if beta is None else beta
            num = x - np.mean(x, axis=tuple(axes), keepdims=True)
            dem = np.sqrt(
                np.sum(np.square(num), axis=tuple(axes), keepdims=True) /
                np.prod(normalized_shape) + epsilon)
            return num / dem * gamma + beta

        x_val = random_gen(shape=(1, 3, 4, 4), rand_min=-100., rand_max=100.)
        g = random_gen(shape=(4, 4), rand_min=1., rand_max=2.)
        b = random_gen(shape=(4, 4), rand_min=0., rand_max=1.)
        res = cb.layer_norm(x=x_val, axes=[-2, -1], gamma=g, beta=b)
        ref = np_layer_norm(x=x_val, axes=[-2, -1], gamma=g, beta=b)
        assert is_close(ref, res.val)

    @pytest.mark.skip('VarHandleOp')
    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, rank, axis, epsilon',
                             itertools.product(
                                 [True],
                                 backends,
                                 [rank for rank in range(3, 4)],
                                 [-1, ],
                                 [1e-10],
                             ))
    def test_tf(self, use_cpu_only, backend, rank, axis, epsilon):
        shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, trainable=False)(x)
            run_compare_tf(graph, {x: random_gen(shape, rand_min=-100, rand_max=100)},
                           ref, use_cpu_only=use_cpu_only, backend=backend)


class TestNormalizationLocalResponseNorm:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([[[1., -7.], [5., -6.], [-3., -5.]]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.local_response_norm(x=x, size=2),
                cb.local_response_norm(x=x, size=3, alpha=0.0001, beta=0.75, k=1.0),
            ]

        expected_output_types = [(1, 3, 2, builtins.fp32), (1, 3, 2, builtins.fp32)]
        expected_outputs = [
            np.array([[[0.99996257, -6.98716545],
                       [4.99531746, -5.99191284],
                       [-2.99898791, -4.99531746]]], dtype=np.float32),
            np.array([[[0.99997497, -6.99143696],
                       [4.99687672, -5.99460602],
                       [-2.99932504, -4.99687672]]], dtype=np.float32)
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_PYTORCH, reason='PyTorch not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, rank, size, alpha, beta, k',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(3, 6)],
                                 [2, 3, 5],
                                 [0.0001, 0.01],
                                 [0.75, 1.0],
                                 [1.0, 2.0],
                             ))
    def test_builder_to_backend_stress(self, use_cpu_only, backend,
                                       rank, size, alpha, beta, k):
        shape = np.random.randint(low=2, high=5, size=rank)
        x_val = random_gen(shape=shape)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return cb.local_response_norm(x=x, size=size, alpha=alpha, beta=beta, k=k)

        torch_lrn = torch.nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)
        expected_outputs = [torch_lrn(torch.as_tensor(x_val)).numpy()]
        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend,
                            atol=1e-2, rtol=1e-3)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, size, alpha, beta, k',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [1, 2, 3],
                                 [0.0001, 0.01],
                                 [0.75, 1.0],
                                 [1.0, 2.0],
                             ))
    def test_tf(self, use_cpu_only, backend, size, alpha, beta, k):
        # TensorFlow's local_response_normalization only supports rank 4
        shape = np.random.randint(low=3, high=6, size=4)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.nn.local_response_normalization(
                x, depth_radius=size, bias=k, alpha=alpha, beta=beta)
            run_compare_tf(graph, {x: random_gen(shape, rand_min=-100, rand_max=100)},
                           ref, use_cpu_only=use_cpu_only, backend=backend,
                           atol=1e-2, rtol=1e-3)
