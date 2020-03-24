from ._test_reqs import *
import pytest
from coremltools.converters.nnv2.nnv2_program.program import get_new_symbol

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
                backends,
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
                backends,
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
                backends,
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


class TestCropResize:

    @pytest.mark.parametrize("use_cpu_only, backend, is_symbolic",
                            itertools.product(
                                [True, False],
                                backends,
                                [True, False]
                                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend, is_symbolic):
        x = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12],
                      [13, 14, 15, 16]], dtype=np.float32).reshape(1, 1, 4, 4)

        input_shape = list(x.shape)
        placeholder_input_shape = input_shape
        if is_symbolic:
            # set batch and channel dimension symbolic
            placeholder_input_shape[0] = get_new_symbol()
            placeholder_input_shape[1] = get_new_symbol()

        input_placeholder_dict = {"x": cb.placeholder(shape=placeholder_input_shape)}
        input_value_dict = {"x" : x}
        N = 1
        roi = np.array([[1, 1, 2, 2]], dtype=np.float32).reshape(1, 1, 4, 1, 1)
        roi_normalized = np.array([[0, 0.0, 0.0, 1.0/3, 1.0/3]], dtype=np.float32).reshape(1, 1, 5, 1, 1)
        roi_invert = np.array([[2, 2, 1, 1]], dtype=np.float32).reshape(1, 1, 4, 1, 1)

        def build(x):
            return [cb.crop_resize(x=x,
                                  roi=roi,
                                  target_width=2,
                                  target_height=2,
                                  normalized_coordinates=False,
                                  box_coordinate_mode='CORNERS_HEIGHT_FIRST',
                                  sampling_mode='ALIGN_CORNERS'),
                    cb.crop_resize(x=x,
                                  roi=roi,
                                  target_width=4,
                                  target_height=4,
                                  normalized_coordinates=False,
                                  box_coordinate_mode='CORNERS_HEIGHT_FIRST',
                                  sampling_mode='ALIGN_CORNERS'),
                    cb.crop_resize(x=x,
                                  roi=roi,
                                  target_width=1,
                                  target_height=1,
                                  normalized_coordinates=False,
                                  box_coordinate_mode='CORNERS_HEIGHT_FIRST',
                                  sampling_mode='ALIGN_CORNERS'),
                    cb.crop_resize(x=x,
                                  roi=roi_normalized,
                                  target_width=2,
                                  target_height=2,
                                  normalized_coordinates=True,
                                  box_coordinate_mode='CORNERS_HEIGHT_FIRST',
                                  sampling_mode='ALIGN_CORNERS'),
                    cb.crop_resize(x=x,
                                  roi=roi_invert,
                                  target_width=2,
                                  target_height=2,
                                  normalized_coordinates=False,
                                  box_coordinate_mode='CORNERS_HEIGHT_FIRST',
                                  sampling_mode='ALIGN_CORNERS'),
        ]

        expected_output_type = [(N, placeholder_input_shape[0], placeholder_input_shape[1], 2, 2, builtins.fp32),
                                (N, placeholder_input_shape[0], placeholder_input_shape[1], 4, 4, builtins.fp32),
                                (N, placeholder_input_shape[0], placeholder_input_shape[1], 1, 1, builtins.fp32),
                                (N, placeholder_input_shape[0], placeholder_input_shape[1], 2, 2, builtins.fp32),
                                (N, placeholder_input_shape[0], placeholder_input_shape[1], 2, 2, builtins.fp32)
        ]
        expected_output = [np.array([6, 7, 10, 11], dtype=np.float32).reshape(1,1,1,2,2),
                           np.array([[6, 6.333333, 6.66666, 7],
                                      [7.333333, 7.666666, 8, 8.333333],
                                      [8.666666, 9, 9.3333333, 9.666666],
                                      [10, 10.333333, 10.666666, 11]
                                     ], dtype=np.float32).reshape(1,1,1,4,4),
                           np.array([8.5], dtype=np.float32).reshape(1,1,1,1,1),
                           np.array([1, 2, 5, 6], dtype=np.float32).reshape(1,1,1,2,2),
                           np.array([11, 10, 7, 6], dtype=np.float32).reshape(1,1,1,2,2),
        ]

        run_compare_builder(build, input_placeholder_dict, input_value_dict,
                            expected_output_type, expected_output,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)


    @pytest.mark.parametrize("use_cpu_only, backend, input_shape, num_of_crops, crop_size, method, dynamic",
            list(itertools.product(
                [True, False],
                backends,
                [(1, 64, 64, 1)],
                [1, 3, 5],
                [(2, 2), (1, 1), (4, 4), (128, 128)],
                ['bilinear'],
                [False])) +
                [pytest.param(True, 'nnv1_proto', (1, 64, 64, 1), 1, (2, 2), 'bilinear', True, marks=pytest.mark.xfail)]
                )
    def test_tf(self, use_cpu_only, backend, input_shape, num_of_crops, crop_size, method, dynamic):
        input = np.random.randn(*input_shape)
        boxes = np.random.uniform(size=(num_of_crops, 4))
        box_indices = np.random.randint(size=(num_of_crops,), low=0, high=input_shape[0])

        def test_static():
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=input_shape)
                output = tf.raw_ops.CropAndResize(image=x, boxes=boxes, box_ind=box_indices, crop_size=crop_size, method=method)
                run_compare_tf(graph, {x: input}, output, use_cpu_only=use_cpu_only, backend=backend)

        def test_dynamic():
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=input_shape)
                boxes_pl = tf.placeholder(tf.float32, shape=boxes.shape)
                box_indices_pl = tf.placeholder(tf.int32, shape=box_indices.shape)
                output = tf.raw_ops.CropAndResize(image=x, boxes=boxes_pl, box_ind=box_indices_pl, crop_size=crop_size, method=method)
                run_compare_tf(graph, {x: input, boxes_pl: boxes, box_indices_pl: box_indices}, output, use_cpu_only=use_cpu_only, backend=backend)

        if dynamic:
            test_dynamic()
        else:
            test_static()