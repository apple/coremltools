
from coremltools._deps import _HAS_ONNX, MSG_ONNX_NOT_FOUND, _IS_MACOS
import unittest
import numpy as np
import numpy.testing as npt  # type: ignore
import numpy.random as npr

from PIL import Image  # type: ignore

if _HAS_ONNX:
    import onnx
    from onnx.numpy_helper import from_array
    from coremltools.converters.onnx import convert
    from ._test_utils import _onnx_create_single_node_model


@unittest.skipUnless(_HAS_ONNX, MSG_ONNX_NOT_FOUND)
class ConvertTest(unittest.TestCase):
    def setUp(self):  # type: () -> None
        self.img_arr = np.uint8(npr.rand(224, 224, 3) * 255)  # type: ignore
        self.img = Image.fromarray(np.uint8(self.img_arr))  # type: ignore
        self.img_arr = np.float32(self.img_arr)  # type: ignore
        self.onnx_model = _onnx_create_single_node_model(
            "Relu", [(3, 224, 224)], [(3, 224, 224)]
        )
        self.input_names = [i.name for i in self.onnx_model.graph.input]
        self.output_names = [o.name for o in self.onnx_model.graph.output]

    def test_convert_image_input(self):  # type: () -> None
        coreml_model = convert(self.onnx_model, image_input_names=self.input_names)
        spec = coreml_model.get_spec()
        for input_ in spec.description.input:
            self.assertEqual(input_.type.WhichOneof("Type"), "imageType")

    def test_convert_image_output(self):  # type: () -> None
        coreml_model = convert(self.onnx_model, image_output_names=self.output_names)
        spec = coreml_model.get_spec()
        for output in spec.description.output:
            self.assertEqual(output.type.WhichOneof("Type"), "imageType")

    def test_convert_image_input_preprocess(self):  # type: () -> None
        bias = np.array([100, 90, 80])
        coreml_model = convert(
            self.onnx_model,
            image_input_names=self.input_names,
            preprocessing_args={
                "is_bgr": True,
                "blue_bias": bias[0],
                "green_bias": bias[1],
                "red_bias": bias[2],
            },
        )

        if _IS_MACOS:
            output = coreml_model.predict({self.input_names[0]: self.img})[
                self.output_names[0]
            ]

            expected_output = self.img_arr[:, :, ::-1].transpose((2, 0, 1))
            expected_output[0] = expected_output[0] + bias[0]
            expected_output[1] = expected_output[1] + bias[1]
            expected_output[2] = expected_output[2] + bias[2]
            npt.assert_equal(output.flatten(), expected_output.flatten())

    def test_convert_image_output_bgr(self):  # type: () -> None
        coreml_model = convert(
            self.onnx_model,
            image_input_names=self.input_names,
            image_output_names=self.output_names,
            deprocessing_args={"is_bgr": True},
        )

        if _IS_MACOS:
            output = coreml_model.predict({self.input_names[0]: self.img})[
                self.output_names[0]
            ]
            output = np.array(output)[:, :, :3].transpose((2, 0, 1))
            expected_output = self.img_arr[:, :, ::-1].transpose((2, 0, 1))
            npt.assert_equal(output, expected_output)


@unittest.skipUnless(_HAS_ONNX, MSG_ONNX_NOT_FOUND)
class NodeConversionTest(unittest.TestCase):
    def test_resize_node_without_scales(self):
        input_shape = (1, 3, 192, 78)
        output_shape = (1, 3, 384, 234)
        roi = from_array(np.array([], dtype=int), name="roi")
        scales = from_array(np.empty([], dtype=int), name="scales")
        sizes = from_array(np.empty([], dtype=int), name="sizes")
        onnx_model_to_test = _onnx_create_single_node_model(
            "Resize",
            [input_shape],
            [output_shape],
            initializer=[roi, scales, sizes],
            coordinate_transformation_mode="pytorch_half_pixel",
            cubic_coeff_a=-0.5,
            mode="linear",
            nearest_mode="floor"
        )

        coreml_model = convert(onnx_model_to_test,
                               minimum_ios_deployment_target="13")

        self.assertEqual(len(coreml_model.get_spec().neuralNetwork.layers), 1,
                         msg="Wrong number of layers in converted network")

        layer = coreml_model.get_spec().neuralNetwork.layers[0]
        self.assertTrue(hasattr(layer, "upsample"),
                        msg="Wrong resize conversion")
        self.assertEqual(len(layer.upsample.scalingFactor), 2,
                         msg="Wrong number of scaling factors")
        self.assertSequenceEqual(layer.upsample.scalingFactor,
                                 (output_shape[2] // input_shape[2],
                                  output_shape[3] // input_shape[3]),
                                  msg="Conversion produces wrong scaling factor")
