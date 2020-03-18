from ._test_reqs import *
import pytest

@pytest.mark.skip("Broken for nnv2 backend")
class TestResizeBilinear:
    def test_builder_to_backend_smoke(self, use_cpu_only=True, backend='nnv1_proto'):
        x = np.array([0, 1], dtype=np.float32).reshape(1, 1, 2)
        input_placeholder_dict = {"x": cb.placeholder(shape=x.shape)}
        input_value_dict = {"x" : x}

        def build_mode_0(x):
            return cb.resize_bilinear(x=x,
                                      target_size_height=1,
                                      target_size_width=5,
                                      sampling_mode="STRICT_ALIGN_CORNERS")

        expected_output_type = (1, 1, 5, builtins.fp32)
        expected_output = np.array([0, 0.25, 0.5, 0.75, 1], dtype=np.float32).reshape(1, 1, 5)

        run_compare_builder(build_mode_0, input_placeholder_dict, input_value_dict,
                            expected_output_type, expected_output,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

        def build_mode_2(x):
            return cb.resize_bilinear(x=x,
                                      target_size_height=1,
                                      target_size_width=5,
                                      sampling_mode="DEFAULT")

        expected_output = np.array([0, 0.4, 0.8, 1, 1], dtype=np.float32).reshape(1, 1, 5)

        run_compare_builder(build_mode_2, input_placeholder_dict, input_value_dict,
                            expected_output_type, expected_output,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

        def build_mode_3(x):
            return cb.resize_bilinear(x=x,
                                      target_size_height=1,
                                      target_size_width=5,
                                      sampling_mode="OFFSET_CORNERS")

        expected_output = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32).reshape(1, 1, 5)

        run_compare_builder(build_mode_3, input_placeholder_dict, input_value_dict,
                            expected_output_type, expected_output,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.parametrize("use_cpu_only, backend, input_shape, target_shape, align_corners, half_pixel_centers",
            itertools.product(
                [True, False],
                ['nnv1_proto'],
                [(1,10,20,1), (2,5,1,3)],
                [(25, 30), (2, 20)],
                [True, False],
                [True, False]
                ))
    def test_tf(self, use_cpu_only, backend, input_shape, target_shape, align_corners, half_pixel_centers):
        if half_pixel_centers and align_corners:
            return
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            ref = tf.raw_ops.ResizeBilinear(images=x, size=target_shape, half_pixel_centers=half_pixel_centers, align_corners=align_corners)
            run_compare_tf(graph, {x: random_gen(input_shape, rand_min=-100, rand_max=100)},
                           ref, use_cpu_only=use_cpu_only, backend=backend)


@pytest.mark.skip("Broken for nnv1 backend")
class TestUpsampleBilinear:
    def test_builder_to_backend_smoke(self, use_cpu_only=True, backend='nnv1_proto'):
        x = np.array([0, 1], dtype=np.float32).reshape(1, 1, 2)
        input_placeholder_dict = {"x": cb.placeholder(shape=x.shape)}
        input_value_dict = {"x" : x}

        def build_upsample_integer(x):
            return cb.upsample_bilinear(x=x,
                                        scale_factor_height=1,
                                        scale_factor_width=3)

        expected_output_type = (1, 1, 6, builtins.fp32)
        expected_output = np.array([0, 0.2, 0.4, 0.6, 0.8, 1], dtype=np.float32).reshape(1, 1, 6)

        run_compare_builder(build_upsample_integer, input_placeholder_dict, input_value_dict,
                            expected_output_type, expected_output,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

        def build_upsample_fractional(x):
            return cb.upsample_bilinear(x=x,
                                        scale_factor_height=1,
                                        scale_factor_width=2.6,
                                        align_corners=False)

        expected_output_type = (1, 1, 5, builtins.fp32)
        expected_output = np.array([0, 0.1, 0.5, 0.9, 1], dtype=np.float32).reshape(1,1,5)

        run_compare_builder(build_upsample_fractional, input_placeholder_dict, input_value_dict,
                            expected_output_type, expected_output,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    # TODO: enable GPU test: rdar://problem/60309338
    @pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, input_shape, scale_factor, align_corners",
            itertools.product(
                [True],
                ['nnv1_proto'],
                [(2,5,10,22)],
                [(3, 4), (2.5, 2), (0.5, 0.75)],
                [True, False]
                ))
    def test_builder_to_backend_stress(self, use_cpu_only, backend, input_shape, scale_factor, align_corners):

        def _get_torch_upsample_prediction(x, scale_factor=(2, 2), align_corners=False):
            x = torch.from_numpy(x)
            m = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=align_corners)
            out = m(x)
            return out.numpy()

        x = random_gen(input_shape, rand_min=-100, rand_max=100)
        torch_pred = _get_torch_upsample_prediction(x, scale_factor=scale_factor, align_corners=align_corners)

        input_placeholder_dict = {"x": cb.placeholder(shape=x.shape)}
        input_value_dict = {"x" : x}

        def build_upsample(x):
            return cb.upsample_bilinear(x=x,
                                        scale_factor_height=scale_factor[0],
                                        scale_factor_width=scale_factor[1],
                                        align_corners=align_corners)

        expected_output_type = torch_pred.shape + (builtins.fp32,)
        run_compare_builder(build_upsample, input_placeholder_dict, input_value_dict,
                            expected_output_type, torch_pred,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)


class TestUpsampleNearestNeighbor:
    def test_builder_to_backend_smoke(self, use_cpu_only=True, backend='nnv1_proto'):
        x = np.array([1.5, 2.5, 3.5], dtype=np.float32).reshape(1,1,1,3)
        input_placeholder_dict = {"x": cb.placeholder(shape=x.shape)}
        input_value_dict = {"x" : x}

        def build(x):
            return cb.upsample_nearest_neighbor(x=x,
                                                upscale_factor_height=1,
                                                upscale_factor_width=2)

        expected_output_type = (1, 1, 1, 6, builtins.fp32)
        expected_output = np.array([1.5, 1.5, 2.5, 2.5, 3.5, 3.5], dtype=np.float32).reshape(1,1,1,6)

        run_compare_builder(build, input_placeholder_dict, input_value_dict,
                            expected_output_type, expected_output,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)


    @pytest.mark.parametrize("use_cpu_only, backend, input_shape, upsample_factor, data_format",
            itertools.product(
                [True, False],
                ['nnv1_proto'],
                [(1,1,1,3), (1,10,5,3)],
                [(1,2), (4,3)],
                ['channels_last', 'channels_first']
                ))
    def test_tf(self, use_cpu_only, backend, input_shape, upsample_factor, data_format):
        if data_format == 'channels_last':
            input_shape = (input_shape[0], input_shape[2], input_shape[3], input_shape[1])
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            ref = tf.keras.layers.UpSampling2D(size=upsample_factor, data_format=data_format, interpolation="nearest")(x)
            run_compare_tf(graph, {x: random_gen(input_shape, rand_min=-100, rand_max=100)},
                           ref, use_cpu_only=use_cpu_only, backend=backend)
