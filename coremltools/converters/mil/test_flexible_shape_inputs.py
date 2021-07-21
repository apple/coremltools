# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import coremltools as ct
from coremltools._deps import _HAS_TORCH, MSG_TORCH_NOT_FOUND
import numpy as _np
import PIL.Image
import pytest

if _HAS_TORCH:
    import torch
    torch.manual_seed(10)

    class TestConvModule(torch.nn.Module):
        def __init__(self, in_channels=3, out_channels=10, kernel_size=3):
            super(TestConvModule, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels,
                                        kernel_size)
        def forward(self, x):
            return self.conv(x)

def _numpy_array_to_pil_image(x):
    """
    convert x of shape (1, 3, H, W) to PIL image
    """
    assert len(x.shape) == 4
    assert list(x.shape[:2]) == [1, 3]
    x = x[0, :, :, :] # (3, H, W)
    x = _np.transpose(x, [1, 2, 0]) # (H, W, 3)
    x = x.astype(_np.uint8)
    return PIL.Image.fromarray(x)


def _compute_snr(arr1, arr2):
    arr1 = arr1.flatten()
    arr2 = arr2.flatten()
    noise = arr1 - arr2
    noise_var = _np.sum(noise ** 2) / len(noise) + 1e-7
    signal_energy = _np.sum(arr2 ** 2) / len(arr2)
    max_signal_energy = _np.amax(arr2 ** 2)
    snr = 10 * _np.log10(signal_energy / noise_var)
    psnr = 10 * _np.log10(max_signal_energy / noise_var)
    return snr, psnr

def _assert_torch_coreml_output_shapes(coreml_model, spec, torch_model, torch_example_input, is_image_input=False):
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


@pytest.mark.skipif(not _HAS_TORCH or not ct.utils._is_macos(), reason=MSG_TORCH_NOT_FOUND)
class TestFlexibleInputShapes:

    @pytest.mark.parametrize("convert_to", ['neuralnetwork', 'mlprogram'])
    def test_multiarray_input_rangedim(self, convert_to):
        if convert_to == "mlprogram" and ct.utils._macos_version() < (12, 0):
            return

        example_input = torch.rand(1, 3, 50, 50) * 100
        traced_model = torch.jit.trace(TestConvModule().eval(), example_input)

        input_shape = ct.Shape(shape=(1, 3, ct.RangeDim(25, 100, default=45), ct.RangeDim(25, 100, default=45)))
        model = ct.convert(traced_model,
                           inputs=[ct.TensorType(shape=input_shape)],
                           convert_to=convert_to)

        spec = model.get_spec()
        assert list(spec.description.input[0].type.multiArrayType.shape) == [1, 3, 45, 45]
        assert spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[2].lowerBound == 25
        assert spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[2].upperBound == 100
        _assert_torch_coreml_output_shapes(model, spec, traced_model, example_input)

    @pytest.mark.parametrize("convert_to", ['neuralnetwork', 'mlprogram'])
    def test_multiarray_input_enumerated(self, convert_to):
        if convert_to == "mlprogram" and ct.utils._macos_version() < (12, 0):
            return

        example_input = torch.rand(1, 3, 50, 50) * 100
        traced_model = torch.jit.trace(TestConvModule().eval(), example_input)

        input_shape = ct.EnumeratedShapes(shapes=[[1, 3, 25, 25], [1, 3, 50, 50], [1, 3, 67, 67]],
                                          default=[1, 3, 67, 67])
        model = ct.convert(traced_model,
                           inputs=[ct.TensorType(shape=input_shape)],
                           convert_to=convert_to)

        spec = model.get_spec()
        assert list(spec.description.input[0].type.multiArrayType.shape) == [1, 3, 67, 67]
        assert list(spec.description.input[0].type.multiArrayType.enumeratedShapes.shapes[0].shape) == [1, 3, 67, 67]
        assert len(spec.description.input[0].type.multiArrayType.enumeratedShapes.shapes) == 3
        _assert_torch_coreml_output_shapes(model, spec, traced_model, example_input)

    @pytest.mark.skipif(ct.utils._macos_version() < (12, 0), reason="Image input with RangeDim works correctly on macOS12+")
    @pytest.mark.parametrize("convert_to", ['neuralnetwork', 'mlprogram'])
    def test_image_input_rangedim(self, convert_to):
        example_input = torch.rand(1, 3, 50, 50) * 255
        traced_model = torch.jit.trace(TestConvModule().eval(), example_input)

        input_shape = ct.Shape(shape=(1, 3, ct.RangeDim(25, 100, default=45), ct.RangeDim(25, 100, default=45)))
        model = ct.convert(traced_model,
                           inputs=[ct.ImageType(shape=input_shape)],
                           convert_to=convert_to)

        spec = model.get_spec()
        assert spec.description.input[0].type.imageType.width == 45
        assert spec.description.input[0].type.imageType.height == 45
        assert spec.description.input[0].type.imageType.imageSizeRange.widthRange.lowerBound == 25
        assert spec.description.input[0].type.imageType.imageSizeRange.widthRange.upperBound == 100
        _assert_torch_coreml_output_shapes(model, spec, traced_model, example_input, is_image_input=True)

    @pytest.mark.parametrize("convert_to", ['neuralnetwork', 'mlprogram'])
    def test_image_input_enumerated(self, convert_to):
        if convert_to == "mlprogram" and ct.utils._macos_version() < (12, 0):
            return

        example_input = torch.rand(1, 3, 50, 50) * 255
        traced_model = torch.jit.trace(TestConvModule().eval(), example_input)

        input_shape = ct.EnumeratedShapes(shapes=[[1, 3, 25, 25], [1, 3, 50, 50], [1, 3, 67, 67]],
                                          default=[1, 3, 67, 67])
        model = ct.convert(traced_model,
                           inputs=[ct.ImageType(shape=input_shape)],
                           convert_to=convert_to)

        spec = model.get_spec()
        assert spec.description.input[0].type.imageType.width == 67
        assert spec.description.input[0].type.imageType.height == 67
        assert len(spec.description.input[0].type.imageType.enumeratedSizes.sizes) == 3
        assert spec.description.input[0].type.imageType.enumeratedSizes.sizes[0].width == 25
        assert spec.description.input[0].type.imageType.enumeratedSizes.sizes[0].height == 25
        _assert_torch_coreml_output_shapes(model, spec, traced_model, example_input, is_image_input=True)





