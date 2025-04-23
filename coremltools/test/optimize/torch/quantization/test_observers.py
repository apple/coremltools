#  Copyright (c) 2025, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch
from torch.ao.quantization import FakeQuantize

from coremltools.optimize.torch.quantization._utils import is_per_channel_quant, is_symmetric_quant
from coremltools.optimize.torch.quantization.modules.observers import EMAMinMaxObserver


def volume(shape):
    vol = None
    if shape is None or len(shape) == 0:
        vol = 0
    else:
        vol = 1
        for elem in shape:
            assert elem >= 0
            vol *= elem
    return vol


def generate_binary_tensor(shape):
    assert len(shape) == 4
    assert volume(shape[2:]) > 1

    # Generate 0 or 1 values
    input_tensor = torch.randint(0, 2, shape, dtype=torch.float)

    # Need to guarantee at least one value of 0 and one value of 1 in each channel.
    # This is needed to fix the min-max ranges of the floating point input
    # (on both a per-tensor and per-channel basis) to be at certain extrema.
    # If you fix the min-max ranges, then you can test for correctness using the quantization error
    # (should be 0 because you fixed the ranges to allow for "perfect" quantization of the input tensor).
    for n in range(shape[0]):
        for c in range(shape[1]):
            slice = input_tensor[n, c, :, :]
            contains_zero = (slice == 0.0).any().item()
            contains_one = (slice == 1.0).any().item()
            assert contains_zero or contains_one
            # There must be at least two items in the slice
            # If none are 0.0, then just change the first item to 0.0
            # If none are 1.0, then just change the first item to 1.0
            if not contains_zero:
                slice[0][0] = 0.0
            if not contains_one:
                slice[0][0] = 1.0

    return input_tensor


@pytest.mark.parametrize(
    "qscheme",
    [
        torch.per_tensor_symmetric,
        torch.per_tensor_affine,
        torch.per_channel_symmetric,
        torch.per_channel_affine,
    ],
)
@pytest.mark.parametrize("dtype", [torch.quint8, torch.qint8])
@pytest.mark.parametrize(
    "observer_cls,mode",
    [
        (EMAMinMaxObserver, None),
    ],
)
def test_observer_fixed_data(qscheme, dtype, observer_cls, mode):
    print("\nQSCHEME", qscheme)
    print("DTYPE", dtype)
    print("OBSERVER", observer_cls.__name__)
    print("MODE", mode)

    ch_axis = 0 if is_per_channel_quant(qscheme) else -1
    observer = None
    if mode is None:
        observer = observer_cls(dtype=dtype, qscheme=qscheme, ch_axis=ch_axis)
    else:
        observer = observer_cls(dtype=dtype, qscheme=qscheme, ch_axis=ch_axis, mode=mode)

    shape = (3, 4, 6, 6)

    # Generate torch tensor with only hard-coded values
    # Please see https://pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MinMaxObserver.html
    # for information on how the scale and zp are computed by PyTorch
    # 0 or 255 for unsigned and affine
    # -127 or 127 for unsigned and symmetric
    # -128 or 127 for signed

    # Generate 0 or 1 values
    input_tensor = generate_binary_tensor(shape)

    if is_symmetric_quant(qscheme):
        # Transform to -127 or 127 values
        input_tensor = torch.where(input_tensor == 0.0, -127.0, 127.0)
    elif dtype == torch.quint8:
        # Transform to 0 or 255 values
        input_tensor = input_tensor * 255.0
    else:
        assert dtype == torch.qint8
        # Transform to -128 or 127 values
        input_tensor = torch.where(input_tensor == 0.0, -128.0, 127.0)

    observer(input_tensor)
    scale, zp = observer.calculate_qparams()

    # Compute quantization error
    if ch_axis != -1:
        param_shape = [1] * len(shape)
        param_shape[ch_axis] = shape[ch_axis]
        scale = scale.reshape(param_shape)
        zp = zp.reshape(param_shape)

    quant_min = 0 if dtype == torch.quint8 else -128
    quant_max = 255 if dtype == torch.quint8 else 127

    quantized_tensor = torch.clamp(torch.round((input_tensor / scale) + zp), quant_min, quant_max)
    dequantized_tensor = (quantized_tensor - zp) * scale

    if is_symmetric_quant(qscheme):
        # Round up to mitigate errors introduced by rounding in quantization
        dequantized_tensor = torch.ceil(dequantized_tensor)

    quantization_error = torch.abs(dequantized_tensor - input_tensor).sum()

    print("SCALE PARAMS")
    print(scale.squeeze())
    print("ZP PARAMS")
    print(zp.squeeze())
    print("QUANTIZATION ERROR")
    print(quantization_error.item())

    assert quantization_error.item() == 0.0


@pytest.mark.parametrize(
    "qscheme",
    [
        torch.per_tensor_symmetric,
        torch.per_tensor_affine,
        torch.per_channel_symmetric,
        torch.per_channel_affine,
    ],
)
@pytest.mark.parametrize("dtype", [torch.quint8, torch.qint8])
@pytest.mark.parametrize(
    "observer_cls,mode",
    [
        (EMAMinMaxObserver, None),
    ],
)
def test_observer_checkpoint(qscheme, dtype, observer_cls, mode):
    print("\nQSCHEME", qscheme)
    print("DTYPE", dtype)
    print("OBSERVER", observer_cls.__name__)
    print("MODE", mode)

    ch_axis = 0 if is_per_channel_quant(qscheme) else -1
    observer = None
    if mode is None:
        observer = observer_cls(dtype=dtype, qscheme=qscheme, ch_axis=ch_axis)
    else:
        observer = observer_cls(dtype=dtype, qscheme=qscheme, ch_axis=ch_axis, mode=mode)

    shape = (3, 4, 6, 6)

    # Generate 0 or 1 values
    input_tensor = generate_binary_tensor(shape)

    if is_symmetric_quant(qscheme):
        # Transform to -127 or 127 values
        input_tensor = torch.where(input_tensor == 0.0, -127.0, 127.0)
    elif dtype == torch.quint8:
        # Transform to 0 or 255 values
        input_tensor = input_tensor * 255.0
    else:
        assert dtype == torch.qint8
        # Transform to -128 or 127 values
        input_tensor = torch.where(input_tensor == 0.0, -128.0, 127.0)

    # Calibrate
    observer(input_tensor)

    # Save state dict
    saved_state_dict = observer.state_dict()

    # Create new observer
    new_observer = None
    if mode is None:
        new_observer = observer_cls(dtype=dtype, qscheme=qscheme, ch_axis=ch_axis)
    else:
        new_observer = observer_cls(dtype=dtype, qscheme=qscheme, ch_axis=ch_axis, mode=mode)

    # Load state dict into new observer
    new_observer.load_state_dict(saved_state_dict)

    # Check all state_dict fields are equal
    new_state_dict = new_observer.state_dict()

    assert saved_state_dict == new_state_dict


@pytest.mark.parametrize(
    "qscheme",
    [
        torch.per_tensor_symmetric,
        torch.per_tensor_affine,
        torch.per_channel_symmetric,
        torch.per_channel_affine,
    ],
)
@pytest.mark.parametrize("dtype", [torch.quint8, torch.qint8])
@pytest.mark.parametrize(
    "observer_cls,mode",
    [
        (EMAMinMaxObserver, None),
    ],
)
def test_observer_fake_quantize(qscheme, dtype, observer_cls, mode):
    ch_axis = 0 if is_per_channel_quant(qscheme) else -1

    fq = None

    if ch_axis != -1:
        if mode is None:
            fq = FakeQuantize(observer=observer_cls, dtype=dtype, qscheme=qscheme, ch_axis=ch_axis)
        else:
            fq = FakeQuantize(
                observer=observer_cls,
                dtype=dtype,
                qscheme=qscheme,
                ch_axis=ch_axis,
                mode=mode,
            )
    else:
        if mode is None:
            fq = FakeQuantize(observer=observer_cls, dtype=dtype, qscheme=qscheme)
        else:
            fq = FakeQuantize(observer=observer_cls, dtype=dtype, qscheme=qscheme, mode=mode)

    assert fq is not None
