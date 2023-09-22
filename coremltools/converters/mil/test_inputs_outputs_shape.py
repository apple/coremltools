# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import os
import tempfile

import numpy as _np
import PIL.Image
import pytest

import coremltools as ct
from coremltools._deps import _HAS_TF_2, _HAS_TORCH, MSG_TF2_NOT_FOUND, MSG_TORCH_NOT_FOUND
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.testing_reqs import backends, compute_units

if _HAS_TORCH:
    import torch

    torch.manual_seed(10)

    class TestConvModule(torch.nn.Module):
        def __init__(self, in_channels=3, out_channels=10, kernel_size=3):
            super(TestConvModule, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)

        def forward(self, x):
            return self.conv(x)

    class TestSimpleModule(torch.nn.Module):
        def forward(self, x):
            x = x + 1.0
            y = x - 9.0
            z = torch.sum(x)
            return x, y, z


if _HAS_TF_2:
    import tensorflow as tf


def _numpy_array_to_pil_image(x):
    """
    convert x of shape (1, 3, H, W) to PIL image
    """
    assert len(x.shape) == 4
    assert list(x.shape[:2]) == [1, 3]
    x = x[0, :, :, :]  # (3, H, W)
    x = _np.transpose(x, [1, 2, 0])  # (H, W, 3)
    x = x.astype(_np.uint8)
    return PIL.Image.fromarray(x)


def _compute_snr(arr1, arr2):
    arr1 = arr1.flatten()
    arr2 = arr2.flatten()
    noise = arr1 - arr2
    noise_var = _np.sum(noise**2) / len(noise) + 1e-7
    signal_energy = _np.sum(arr2**2) / len(arr2)
    max_signal_energy = _np.amax(arr2**2)
    snr = 10 * _np.log10(signal_energy / noise_var)
    psnr = 10 * _np.log10(max_signal_energy / noise_var)
    return snr, psnr


def _assert_torch_coreml_output_shapes(
    coreml_model, spec, torch_model, torch_example_input, is_image_input=False
):
    torch_out = torch_model(torch_example_input)
    input_name = spec.description.input[0].name
    output_name = spec.description.output[0].name
    input_dict = {}
    if is_image_input:
        input_dict[input_name] = _numpy_array_to_pil_image(torch_example_input.numpy())
    else:
        input_dict[input_name] = torch_example_input.numpy()
    coreml_out = coreml_model.predict(input_dict)[output_name]
    assert torch_out.shape == coreml_out.shape
    snr, psnr = _compute_snr(torch_out.cpu().detach().numpy(), coreml_out)
    _np.testing.assert_array_less(20, snr)
    _np.testing.assert_array_less(30, psnr)


class TestOutputShapes:
    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_static_output_shapes(backend):
        @mb.program(
            input_specs=[
                mb.TensorSpec(
                    shape=(2, 3),
                )
            ]
        )
        def prog(x):
            x = mb.add(x=x, y=1.0)
            y = mb.sub(x=x, y=3.0)
            z = mb.reduce_sum(x=x, axes=[0, 1], keep_dims=False)
            return x, y, z

        model = ct.convert(prog, convert_to=backend[0])
        spec = model.get_spec()
        expected_output_shape = [2, 3] if backend[0] == "mlprogram" else []
        assert spec.description.output[0].type.multiArrayType.shape == expected_output_shape
        assert spec.description.output[1].type.multiArrayType.shape == expected_output_shape
        # scalar outputs have shape ()
        assert spec.description.output[2].type.multiArrayType.shape == []

        coreml_in = {"x": _np.random.rand(2, 3)}
        model.predict(coreml_in)

    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_dynamic_output_shapes(backend):

        example_input = torch.rand(2, 3)
        traced_model = torch.jit.trace(TestSimpleModule().eval(), example_input)

        input_shape = ct.Shape(shape=(2, ct.RangeDim(3, 5)))
        model = ct.convert(
            traced_model, inputs=[ct.TensorType(shape=input_shape)], convert_to=backend[0]
        )

        spec = model.get_spec()
        # We don't put the shape information for dynamic output shapes,
        # otherwise a runtime validation error would raise
        assert spec.description.output[0].type.multiArrayType.shape == []
        assert spec.description.output[1].type.multiArrayType.shape == []
        # scalar outputs have shape ()
        assert spec.description.output[2].type.multiArrayType.shape == []

        coreml_in = {"x_1": _np.random.rand(2, 3)}
        model.predict(coreml_in)


@pytest.mark.skipif(not _HAS_TORCH or not ct.utils._is_macos(), reason=MSG_TORCH_NOT_FOUND)
class TestFlexibleInputShapesTorch:
    @pytest.mark.parametrize(
        "backend, compute_unit",
        itertools.product(
            backends,
            compute_units,
        ),
    )
    def test_multiarray_input_rangedim(self, backend, compute_unit):
        convert_to = backend[0]
        if convert_to == "mlprogram" and ct.utils._macos_version() < (12, 0):
            return

        example_input = torch.rand(1, 3, 50, 50) * 100
        traced_model = torch.jit.trace(TestConvModule().eval(), example_input)

        input_shape = ct.Shape(
            shape=(1, 3, ct.RangeDim(25, 100, default=45), ct.RangeDim(25, 100, default=45))
        )
        model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=input_shape)],
            convert_to=convert_to,
            compute_units=compute_unit,
        )

        spec = model.get_spec()
        assert list(spec.description.input[0].type.multiArrayType.shape) == [1, 3, 45, 45]
        assert (
            spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[2].lowerBound == 25
        )
        assert (
            spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[2].upperBound == 100
        )
        _assert_torch_coreml_output_shapes(model, spec, traced_model, example_input)

    @pytest.mark.parametrize(
        "backend, compute_unit, explicitly_set",
        itertools.product(
            backends,
            compute_units,
            [True, False],
        ),
    )
    def test_multiarray_input_rangedim_infinite(self, backend, compute_unit, explicitly_set):
        convert_to = backend[0]
        example_input = torch.rand(1, 3, 50, 50) * 100
        traced_model = torch.jit.trace(TestConvModule().eval(), example_input)
        second_dim = ct.RangeDim()
        if explicitly_set:
            second_dim.upper_bound = -1
        input_shape = ct.Shape(shape=(1, 3, second_dim, ct.RangeDim(25, 100, default=45)))

        if convert_to == "mlprogram":
            with pytest.raises(
                ValueError,
                match="For mlprogram, inputs with infinite upper_bound is not allowed. Please set "
                'upper_bound to a positive value in "RangeDim\(\)" for the "inputs" param in '
                "ct.convert\(\).",
            ):
                ct.convert(
                    traced_model,
                    inputs=[ct.TensorType(shape=input_shape)],
                    convert_to=convert_to,
                    compute_units=compute_unit,
                )
        else:
            model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=input_shape)],
                convert_to=convert_to,
                compute_units=compute_unit,
            )
            spec = model.get_spec()
            assert list(spec.description.input[0].type.multiArrayType.shape) == [1, 3, 1, 45]
            assert (
                spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[2].lowerBound
                == 1
            )
            assert (
                spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[2].upperBound
                == -1
            )
            _assert_torch_coreml_output_shapes(model, spec, traced_model, example_input)

    @pytest.mark.parametrize(
        "backend, compute_unit",
        itertools.product(
            backends,
            compute_units,
        ),
    )
    def test_multiarray_input_enumerated(self, backend, compute_unit):
        convert_to = backend[0]
        if convert_to == "mlprogram" and ct.utils._macos_version() < (12, 0):
            return

        example_input = torch.rand(1, 3, 50, 50) * 100
        traced_model = torch.jit.trace(TestConvModule().eval(), example_input)

        input_shape = ct.EnumeratedShapes(
            shapes=[[1, 3, 25, 25], [1, 3, 50, 50], [1, 3, 67, 67]], default=[1, 3, 67, 67]
        )
        model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=input_shape)],
            convert_to=convert_to,
            compute_units=compute_unit,
        )

        spec = model.get_spec()
        assert list(spec.description.input[0].type.multiArrayType.shape) == [1, 3, 67, 67]
        assert list(
            spec.description.input[0].type.multiArrayType.enumeratedShapes.shapes[0].shape
        ) == [1, 3, 67, 67]
        assert len(spec.description.input[0].type.multiArrayType.enumeratedShapes.shapes) == 3
        _assert_torch_coreml_output_shapes(model, spec, traced_model, example_input)

    @pytest.mark.skipif(
        ct.utils._macos_version() < (12, 0),
        reason="Image input with RangeDim works correctly on macOS12+",
    )
    @pytest.mark.parametrize(
        "backend, compute_unit",
        itertools.product(
            backends,
            compute_units,
        ),
    )
    def test_image_input_rangedim(self, backend, compute_unit):
        convert_to = backend[0]
        example_input = torch.rand(1, 3, 50, 50) * 255
        traced_model = torch.jit.trace(TestConvModule().eval(), example_input)

        input_shape = ct.Shape(
            shape=(1, 3, ct.RangeDim(25, 100, default=35), ct.RangeDim(25, 100, default=45))
        )
        model = ct.convert(
            traced_model,
            inputs=[ct.ImageType(shape=input_shape)],
            convert_to=convert_to,
            compute_units=compute_unit,
        )

        spec = model.get_spec()
        assert spec.description.input[0].type.imageType.width == 45
        assert spec.description.input[0].type.imageType.height == 35
        assert spec.description.input[0].type.imageType.imageSizeRange.widthRange.lowerBound == 25
        assert spec.description.input[0].type.imageType.imageSizeRange.widthRange.upperBound == 100
        _assert_torch_coreml_output_shapes(
            model, spec, traced_model, example_input, is_image_input=True
        )

    @pytest.mark.skipif(
        ct.utils._macos_version() < (12, 0),
        reason="Image input with RangeDim works correctly on macOS12+",
    )
    @pytest.mark.parametrize(
        "backend, compute_unit, explicitly_set",
        itertools.product(
            backends,
            compute_units,
            [True, False],
        ),
    )
    def test_image_input_rangedim_infinite(self, backend, compute_unit, explicitly_set):
        convert_to = backend[0]
        example_input = torch.rand(1, 3, 50, 50) * 255
        traced_model = torch.jit.trace(TestConvModule().eval(), example_input)

        second_dim = ct.RangeDim(upper_bound=-1) if explicitly_set else ct.RangeDim()
        input_shape = ct.Shape(shape=(1, 3, second_dim, ct.RangeDim(25, 100, default=45)))

        if convert_to == "mlprogram":
            with pytest.raises(
                ValueError,
                match="For mlprogram, inputs with infinite upper_bound is not allowed. Please set "
                'upper_bound to a positive value in "RangeDim\(\)" for the "inputs" param in '
                "ct.convert\(\).",
            ):
                ct.convert(
                    traced_model,
                    inputs=[ct.ImageType(shape=input_shape)],
                    convert_to=convert_to,
                    compute_units=compute_unit,
                )
        else:
            model = ct.convert(
                traced_model,
                inputs=[ct.ImageType(shape=input_shape)],
                convert_to=convert_to,
                compute_units=compute_unit,
            )
            spec = model.get_spec()
            assert spec.description.input[0].type.imageType.width == 45
            assert spec.description.input[0].type.imageType.height == 1
            assert (
                spec.description.input[0].type.imageType.imageSizeRange.heightRange.lowerBound == 1
            )
            assert (
                spec.description.input[0].type.imageType.imageSizeRange.heightRange.upperBound == -1
            )
            _assert_torch_coreml_output_shapes(
                model, spec, traced_model, example_input, is_image_input=True
            )

    @pytest.mark.parametrize(
        "backend, compute_unit",
        itertools.product(
            backends,
            compute_units,
        ),
    )
    def test_image_input_enumerated(self, backend, compute_unit):
        convert_to = backend[0]
        if convert_to == "mlprogram" and ct.utils._macos_version() < (12, 0):
            return

        example_input = torch.rand(1, 3, 50, 50) * 255
        traced_model = torch.jit.trace(TestConvModule().eval(), example_input)

        input_shape = ct.EnumeratedShapes(
            shapes=[[1, 3, 25, 25], [1, 3, 50, 50], [1, 3, 67, 67]], default=[1, 3, 67, 67]
        )
        model = ct.convert(
            traced_model,
            inputs=[ct.ImageType(shape=input_shape)],
            convert_to=convert_to,
            compute_units=compute_unit,
        )

        spec = model.get_spec()
        assert spec.description.input[0].type.imageType.width == 67
        assert spec.description.input[0].type.imageType.height == 67
        assert len(spec.description.input[0].type.imageType.enumeratedSizes.sizes) == 3
        assert spec.description.input[0].type.imageType.enumeratedSizes.sizes[0].width == 25
        assert spec.description.input[0].type.imageType.enumeratedSizes.sizes[0].height == 25
        _assert_torch_coreml_output_shapes(
            model, spec, traced_model, example_input, is_image_input=True
        )


@pytest.mark.skipif(not _HAS_TF_2 or not ct.utils._is_macos(), reason=MSG_TF2_NOT_FOUND)
class TestFlexibleInputShapesTF:
    @classmethod
    def setup_class(cls):
        """Prepares tf model in different formats (keras model, h5 file, saved_model dir)."""
        input_1 = tf.keras.Input(shape=(None, None, 16), name="input_1")
        input_2 = tf.keras.Input(shape=(None, None, 4), name="input_2")
        x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(input_1) + input_2
        outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
        cls.model = tf.keras.Model(inputs=[input_1, input_2], outputs=outputs)

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.h5_model_path = os.path.join(cls.temp_dir.name, "tf_keras_model.h5")
        cls.model.save(cls.h5_model_path)
        cls.saved_model_path = os.path.join(cls.temp_dir.name, "saved_model")
        cls.model.save(cls.saved_model_path, save_format="tf")

    @classmethod
    def teardown_class(cls):
        """CLean up temp dir that stores the TF models."""
        cls.temp_dir.cleanup()

    @staticmethod
    def _find_unknown_dim_warning(raised_warnings: pytest.WarningsRecorder) -> bool:
        """Find if pytest catches any warning message about the unknown dim warning."""
        for raised_warning in raised_warnings:
            if raised_warning.message.args[0].startswith(
                "Some dimensions in the input shape are unknown, hence they are set to flexible ranges"
            ):
                return True
        return False

    @pytest.mark.parametrize(
        "backend, compute_unit, model_format",
        itertools.product(
            backends,
            compute_units,
            ["keras_model", "h5", "saved_model"],
        ),
    )
    def test_dynamic_shape_no_inputs(self, backend, compute_unit, model_format):
        """
        The `inputs` param in `ct.convert` is not provided, so all inputs in the TF model with `None`
        dim will have a range shape where lower-bound/default/upper-bound are sanitized to finite
        numbers and warns users.
        """
        convert_to = backend[0]
        model_param = self.model
        if model_format == "h5":
            model_param = self.h5_model_path
        elif model_format == "saved_model":
            model_param = self.saved_model_path

        if convert_to == "mlprogram":
            with pytest.warns(
                UserWarning,
                match="Some dimensions in the input shape are unknown, hence they are set to "
                "flexible ranges with lower bound and default value = 1, and upper bound = 2. "
                "To set different values for the default shape and upper bound, please use "
                "the ct.RangeDim.*",
            ):
                mlmodel = ct.convert(
                    model_param,
                    source="tensorflow",
                    convert_to=convert_to,
                    compute_units=compute_unit,
                )
        else:
            mlmodel = ct.convert(
                model_param,
                source="tensorflow",
                convert_to=convert_to,
                compute_units=compute_unit,
            )

        spec = mlmodel.get_spec()
        assert list(spec.description.input[0].type.multiArrayType.shape) == [1, 1, 1, 16]
        assert (
            spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[2].lowerBound == 1
        )
        assert (
            spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[2].upperBound == -1
            if convert_to == "neuralnetwork"
            else 2
        )

    @pytest.mark.parametrize(
        "backend, compute_unit, specify_input",
        itertools.product(
            backends,
            compute_units,
            ["input_1", "input_2"],
        ),
    )
    def test_dynamic_shape_partial_inputs(self, backend, compute_unit, specify_input):
        """
        The `inputs` param in `ct.convert` is partially provided, where the TF model has two inputs
        while we only provide one in `inputs` param. So another input in the TF model with `None`
        dim will have a range shape where lower-bound/default/upper-bound are sanitized to finite
        numbers and warns users.
        """
        convert_to = backend[0]
        last_dim = 16 if specify_input == "input_1" else 4
        inputs = [
            ct.TensorType(
                shape=ct.Shape(
                    shape=(
                        1,
                        3,
                        ct.RangeDim(2, 10, default=8),
                        ct.RangeDim(4, 20, default=last_dim),
                    )
                ),
                name=specify_input,
            )
        ]

        if convert_to == "mlprogram":
            with pytest.warns(
                UserWarning,
                match="Some dimensions in the input shape are unknown, hence they are set to "
                "flexible ranges with lower bound and default value = 1, and upper bound = 2. "
                "To set different values for the default shape and upper bound, please use "
                "the ct.RangeDim.*",
            ):
                mlmodel = ct.convert(
                    self.model,
                    source="tensorflow",
                    inputs=inputs,
                    convert_to=convert_to,
                    compute_units=compute_unit,
                )
        else:
            mlmodel = ct.convert(
                self.model,
                source="tensorflow",
                inputs=inputs,
                convert_to=convert_to,
                compute_units=compute_unit,
            )

        spec = mlmodel.get_spec()
        # Notice the input in spec is not ordered, so need to use name to find input_1 and input_2.
        for input_spec in spec.description.input:
            if input_spec.name == "input_1":
                input_1_spec = input_spec
            elif input_spec.name == "input_2":
                input_2_spec = input_spec
        assert (
            list(input_1_spec.type.multiArrayType.shape) == [1, 3, 8, 16]
            if specify_input == "input_1"
            else [1, 1, 1, 16]
        )
        assert (
            list(input_2_spec.type.multiArrayType.shape) == [1, 3, 8, 4]
            if specify_input == "input_2"
            else [1, 1, 1, 4]
        )
        assert (
            input_1_spec.type.multiArrayType.shapeRange.sizeRanges[2].lowerBound == 2
            if specify_input == "input_1"
            else 1
        )
        assert (
            input_2_spec.type.multiArrayType.shapeRange.sizeRanges[2].lowerBound == 2
            if specify_input == "input_2"
            else 1
        )
        default_upper_bound = -1 if convert_to == "neuralnetwork" else 2
        assert (
            input_1_spec.type.multiArrayType.shapeRange.sizeRanges[2].upperBound == 10
            if specify_input == "input_1"
            else default_upper_bound
        )
        assert (
            input_2_spec.type.multiArrayType.shapeRange.sizeRanges[2].upperBound == 10
            if specify_input == "input_2"
            else default_upper_bound
        )

    @pytest.mark.parametrize(
        "backend, compute_unit",
        itertools.product(
            backends,
            compute_units,
        ),
    )
    def test_multiarray_input_rangedim(self, backend, compute_unit):
        input_shape_1 = ct.Shape(
            shape=(1, 3, ct.RangeDim(8, 20, default=8), ct.RangeDim(10, 100, default=16))
        )
        input_shape_2 = ct.Shape(
            shape=(1, 3, ct.RangeDim(4, 16, default=16), ct.RangeDim(1, 10, default=4))
        )

        with pytest.warns() as raised_warnings:
            model = ct.convert(
                self.model,
                source="tensorflow",
                inputs=[
                    ct.TensorType(shape=input_shape_1, name="input_1"),
                    ct.TensorType(shape=input_shape_2, name="input_2"),
                ],
                convert_to=backend[0],
                compute_units=compute_unit,
            )
            assert not self._find_unknown_dim_warning(raised_warnings)

        spec = model.get_spec()
        assert list(spec.description.input[0].type.multiArrayType.shape) == [1, 3, 8, 16]
        assert (
            spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[2].lowerBound == 8
        )
        assert (
            spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[2].upperBound == 20
        )
        assert list(spec.description.input[1].type.multiArrayType.shape) == [1, 3, 16, 4]
        assert (
            spec.description.input[1].type.multiArrayType.shapeRange.sizeRanges[2].lowerBound == 4
        )
        assert (
            spec.description.input[1].type.multiArrayType.shapeRange.sizeRanges[2].upperBound == 16
        )

    @pytest.mark.parametrize(
        "backend, compute_unit, explicitly_set",
        itertools.product(
            backends,
            compute_units,
            [True, False],
        ),
    )
    def test_multiarray_input_rangedim_infinite(self, backend, compute_unit, explicitly_set):
        convert_to = backend[0]
        second_dim = ct.RangeDim(upper_bound=-1) if explicitly_set else ct.RangeDim()
        input_shape = ct.Shape(shape=(1, 3, second_dim, ct.RangeDim(10, 100, default=16)))

        if convert_to == "mlprogram":
            with pytest.raises(
                ValueError,
                match="For mlprogram, inputs with infinite upper_bound is not allowed. Please set "
                'upper_bound to a positive value in "RangeDim\(\)" for the "inputs" param in '
                "ct.convert\(\).",
            ):
                ct.convert(
                    self.model,
                    source="tensorflow",
                    inputs=[ct.TensorType(shape=input_shape, name="input_1")],
                    convert_to=convert_to,
                    compute_units=compute_unit,
                )
        else:
            model = ct.convert(
                self.model,
                source="tensorflow",
                inputs=[ct.TensorType(shape=input_shape, name="input_1")],
                convert_to=convert_to,
                compute_units=compute_unit,
            )
            spec = model.get_spec()
            assert list(spec.description.input[0].type.multiArrayType.shape) == [1, 3, 1, 16]
            assert (
                spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[2].lowerBound
                == 1
            )
            assert (
                spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[2].upperBound
                == -1
            )

    @pytest.mark.parametrize(
        "backend, compute_unit",
        itertools.product(
            backends,
            compute_units,
        ),
    )
    def test_multiarray_single_input_rangedim(self, backend, compute_unit):
        input_1 = tf.keras.Input(shape=(None, None, 16), name="input_1")
        x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(input_1)
        outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
        single_input_model = tf.keras.Model(inputs=input_1, outputs=outputs)

        # The `inputs` will work without specifying the name.
        model = ct.convert(
            single_input_model,
            source="tensorflow",
            inputs=[
                ct.TensorType(
                    shape=(1, 3, ct.RangeDim(8, 20, default=8), ct.RangeDim(10, 100, default=16))
                )
            ],
            convert_to=backend[0],
            compute_units=compute_unit,
        )
        spec = model.get_spec()
        assert list(spec.description.input[0].type.multiArrayType.shape) == [1, 3, 8, 16]
        assert (
            spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[2].lowerBound == 8
        )
        assert (
            spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[2].upperBound == 20
        )

    @pytest.mark.skipif(
        ct.utils._macos_version() < (12, 0),
        reason="Image input with RangeDim works correctly on macOS12+",
    )
    @pytest.mark.parametrize(
        "backend, compute_unit, explicitly_set",
        itertools.product(
            backends,
            compute_units,
            [True, False],
        ),
    )
    def test_image_input_rangedim_infinite(self, backend, compute_unit, explicitly_set):
        convert_to = backend[0]
        second_dim = ct.RangeDim(upper_bound=-1) if explicitly_set else ct.RangeDim()
        input_shape = ct.Shape(shape=(1, 2, second_dim, ct.RangeDim(1, 10, default=3)))

        if convert_to == "mlprogram":
            with pytest.raises(
                ValueError,
                match="For mlprogram, inputs with infinite upper_bound is not allowed. Please set "
                'upper_bound to a positive value in "RangeDim\(\)" for the "inputs" param in '
                "ct.convert\(\).",
            ):
                ct.convert(
                    self.model,
                    source="tensorflow",
                    inputs=[ct.ImageType(shape=input_shape, name="input_1")],
                    convert_to=convert_to,
                    compute_units=compute_unit,
                )
        else:
            model = ct.convert(
                self.model,
                source="tensorflow",
                inputs=[ct.ImageType(shape=input_shape, name="input_1")],
                convert_to=convert_to,
                compute_units=compute_unit,
            )
            spec = model.get_spec()
            assert spec.description.input[0].type.imageType.width == 1
            assert spec.description.input[0].type.imageType.height == 2
            assert (
                spec.description.input[0].type.imageType.imageSizeRange.widthRange.lowerBound == 1
            )
            assert (
                spec.description.input[0].type.imageType.imageSizeRange.widthRange.upperBound == -1
            )
