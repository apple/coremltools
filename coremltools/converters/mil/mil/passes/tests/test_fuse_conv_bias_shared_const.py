# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pytest

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import apply_pass_and_basic_check


@pytest.mark.parametrize("conv_type", ["conv", "conv_transpose"])
@pytest.mark.parametrize("has_bias", [False, True])
def test_sub_preserves_constant_used_by_another_branch(conv_type, has_bias):
    offset = np.arange(1, 4, dtype=np.float32).reshape(1, 3, 1, 1)
    old_bias = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    weight_shape = (3, 2, 1, 1) if conv_type == "conv" else (2, 3, 1, 1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 4, 4)),
                             mb.TensorSpec(shape=(1, 3, 4, 4))])
    def prog(x, z):
        shared = mb.const(val=offset.copy())
        conv = getattr(mb, conv_type)(
            x=x, weight=np.ones(weight_shape, dtype=np.float32),
            bias=old_bias.copy() if has_bias else None,
        )
        return mb.relu(x=mb.sub(x=conv, y=shared)), mb.add(x=z, y=shared)

    apply_pass_and_basic_check(prog, "common::fuse_conv_bias")
    conv = prog.functions["main"].find_ops(op_type=conv_type)[0]
    other_consumer = prog.functions["main"].find_ops(op_type="add")[0]
    expected_bias = (old_bias if has_bias else 0) - offset.reshape(-1)
    np.testing.assert_array_equal(conv.bias.val, expected_bias)
    np.testing.assert_array_equal(other_consumer.y.val, offset)


@pytest.mark.parametrize("deduplicate", [False, True])
def test_two_subtractions_do_not_flip_each_others_bias(deduplicate):
    # The default pipeline deduplicates constants of at least 100 elements.
    channels = 128
    offset = np.arange(1, channels + 1, dtype=np.float32).reshape(1, channels, 1, 1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 2, 2))])
    def prog(x):
        shared = mb.const(val=offset.copy())
        second = mb.const(val=offset.copy()) if deduplicate else shared
        first_conv = mb.conv(x=x, weight=np.ones((channels, 1, 1, 1), dtype=np.float32))
        second_conv = mb.conv(x=x, weight=np.full((channels, 1, 1, 1), 2, dtype=np.float32))
        return mb.relu(x=mb.sub(x=first_conv, y=shared)), mb.relu(x=mb.sub(x=second_conv, y=second))

    if deduplicate:
        apply_pass_and_basic_check(prog, "common::const_deduplication")
    apply_pass_and_basic_check(prog, "common::fuse_conv_bias")
    convs = prog.functions["main"].find_ops(op_type="conv")
    assert len(convs) == 2
    for conv in convs:
        np.testing.assert_array_equal(conv.bias.val, -offset.reshape(-1))
