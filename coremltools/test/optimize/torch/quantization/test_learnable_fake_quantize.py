#  Copyright (c) 2025, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch
import torch.ao.quantization as aoquant

from coremltools.optimize.torch.quantization._utils import is_per_channel_quant, is_symmetric_quant
from coremltools.optimize.torch.quantization.modules.learnable_fake_quantize import (
    LearnableFakeQuantize,
)
from coremltools.optimize.torch.quantization.modules.observers import EMAMinMaxObserver


def create_learnable_fake_quantize(qscheme, dtype, reduce_range, observer_cls, mode):
    lfq = None

    if mode is None:
        lfq = LearnableFakeQuantize(observer_cls, dtype, qscheme, reduce_range)
    else:
        lfq = LearnableFakeQuantize(observer_cls, dtype, qscheme, reduce_range, mode=mode)

    return lfq


def call_setters(lfq, ch_axis, shape):
    if ch_axis != -1:
        lfq.set_ch_axis(ch_axis)
        lfq.set_param_shape(shape)

    lfq.set_device(torch.device("cpu"))


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
@pytest.mark.parametrize("reduce_range", [True, False])
@pytest.mark.parametrize(
    "observer_cls,mode",
    [
        (EMAMinMaxObserver, None),
    ],
)
def test_learnable_fake_quantize_init(qscheme, dtype, reduce_range, observer_cls, mode):
    print("\nQSCHEME", qscheme)
    print("DTYPE", dtype)
    print("OBSERVER", observer_cls.__name__)
    print("MODE", mode)

    ch_axis = 0 if is_per_channel_quant(qscheme) else -1

    # Think about this as a weights tensor
    shape = (3, 4, 6, 6)

    # Test constructor

    lfq = create_learnable_fake_quantize(qscheme, dtype, reduce_range, observer_cls, mode)

    assert lfq is not None
    assert lfq.dtype == dtype
    assert lfq.qscheme == qscheme
    assert lfq.reduce_range == reduce_range
    assert lfq.activation_post_process is not None
    assert type(lfq.activation_post_process) == observer_cls
    assert mode == None or lfq.activation_post_process.mode == mode

    assert lfq.ch_axis == -1
    assert lfq.channel_count == 1

    assert lfq.device is None
    assert lfq.shape is None
    assert lfq.mask_shape is None


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
@pytest.mark.parametrize("reduce_range", [True, False])
@pytest.mark.parametrize(
    "observer_cls,mode",
    [
        (EMAMinMaxObserver, None),
    ],
)
def test_learnable_fake_quantize_setters(qscheme, dtype, reduce_range, observer_cls, mode):
    print("\nQSCHEME", qscheme)
    print("DTYPE", dtype)
    print("OBSERVER", observer_cls.__name__)
    print("MODE", mode)

    ch_axis = 0 if is_per_channel_quant(qscheme) else -1

    # Think about this as a weights tensor
    shape = (3, 4, 6, 6)

    lfq = create_learnable_fake_quantize(qscheme, dtype, reduce_range, observer_cls, mode)

    # Test setters

    call_setters(lfq, ch_axis, shape)

    assert lfq.ch_axis == ch_axis
    if ch_axis == -1:
        assert lfq.shape == None
    else:
        assert lfq.shape == shape
        mask_shape = [1] * len(shape)
        channel_count = shape[ch_axis]
        mask_shape[ch_axis] = channel_count
        assert lfq.mask_shape == mask_shape
        assert lfq.channel_count == channel_count
    assert lfq.device == torch.device("cpu")
    assert lfq.scale.device == torch.device("cpu")
    assert lfq.zero_point.device == torch.device("cpu")


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
@pytest.mark.parametrize("reduce_range", [True, False])
@pytest.mark.parametrize(
    "observer_cls,mode",
    [
        (EMAMinMaxObserver, None),
    ],
)
def test_learnable_fake_quantize_phases(qscheme, dtype, reduce_range, observer_cls, mode):
    print("\nQSCHEME", qscheme)
    print("DTYPE", dtype)
    print("OBSERVER", observer_cls.__name__)
    print("MODE", mode)

    ch_axis = 0 if is_per_channel_quant(qscheme) else -1

    # Think about this as a weights tensor
    shape = (3, 4, 6, 6)

    lfq = create_learnable_fake_quantize(qscheme, dtype, reduce_range, observer_cls, mode)

    call_setters(lfq, ch_axis, shape)

    # Generate 0 or 1 values
    input_tensor = torch.randint(0, 2, shape, dtype=torch.float)

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

    # Test pre-observation phase

    lfq.apply(aoquant.disable_observer)
    lfq.apply(aoquant.disable_fake_quant)

    assert lfq.observer_enabled[0] == 0
    assert lfq.fake_quant_enabled[0] == 0

    scale, zp = lfq.calculate_qparams()

    lfq(input_tensor)

    new_scale, new_zp = lfq.calculate_qparams()

    # There should not be any changes to the scale and zp

    assert (scale == new_scale).all()
    assert (zp == new_zp).all()

    # Test observation phase

    lfq.apply(aoquant.enable_observer)
    lfq.apply(aoquant.disable_fake_quant)

    assert lfq.observer_enabled[0] == 1
    assert lfq.fake_quant_enabled[0] == 0

    lfq(input_tensor)

    new_scale, new_zp = lfq.calculate_qparams()

    # Test fake quant phase

    lfq.apply(aoquant.disable_observer)
    lfq.apply(aoquant.enable_fake_quant)

    assert lfq.observer_enabled[0] == 0
    assert lfq.fake_quant_enabled[0] == 1

    lfq(input_tensor)

    new_scale, new_zp = lfq.calculate_qparams()

    # Test fake quant and observation phase

    lfq.apply(aoquant.enable_observer)
    lfq.apply(aoquant.enable_fake_quant)

    assert lfq.observer_enabled[0] == 1
    assert lfq.fake_quant_enabled[0] == 1

    lfq(input_tensor)

    new_scale, new_zp = lfq.calculate_qparams()


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
@pytest.mark.parametrize("reduce_range", [True, False])
@pytest.mark.parametrize(
    "observer_cls,mode",
    [
        (EMAMinMaxObserver, None),
    ],
)
def test_learnable_fake_quantize_checkpoint(qscheme, dtype, reduce_range, observer_cls, mode):
    print("\nQSCHEME", qscheme)
    print("DTYPE", dtype)
    print("OBSERVER", observer_cls.__name__)
    print("MODE", mode)

    ch_axis = 0 if is_per_channel_quant(qscheme) else -1

    # Think about this as a weights tensor
    shape = (3, 4, 6, 6)

    lfq = create_learnable_fake_quantize(qscheme, dtype, reduce_range, observer_cls, mode)

    call_setters(lfq, ch_axis, shape)

    # Generate 0 or 1 values
    input_tensor = torch.randint(0, 2, shape, dtype=torch.float)

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

    # Perform observation to get sensible quantization parameters

    lfq.apply(aoquant.enable_observer)
    lfq.apply(aoquant.disable_fake_quant)

    lfq(input_tensor)

    scale, zp = lfq.calculate_qparams()

    # Test checkpointing

    saved_state_dict = lfq.state_dict()

    new_lfq = create_learnable_fake_quantize(qscheme, dtype, reduce_range, observer_cls, mode)

    call_setters(new_lfq, ch_axis, shape)

    new_lfq.load_state_dict(saved_state_dict)

    new_scale, new_zp = new_lfq.calculate_qparams()

    # There should not be any changes to the scale and zp

    assert (scale == new_scale).all()
    assert (zp == new_zp).all()

    updated_state_dict = new_lfq.state_dict()

    assert saved_state_dict.keys() == updated_state_dict.keys()
    for key in saved_state_dict.keys():
        val1 = saved_state_dict[key]
        val2 = updated_state_dict[key]
        assert (val1 == val2).all()
