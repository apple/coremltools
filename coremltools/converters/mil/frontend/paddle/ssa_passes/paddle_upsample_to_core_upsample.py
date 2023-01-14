#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools import _logger as logger
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


target_ops = [
    "paddle_upsample_nearest_neighbor",
    "paddle_upsample_bilinear",
]


@register_pass(namespace="paddle")
class paddle_upsample_to_core_upsample(AbstractGraphPass):
    """
    Try to map Paddle dialect ops
    1. `paddle_upsample_nearest_neighbor`
    2. `paddle_upsample_bilinear`
    to `upsample_nearest_neighbor` or `upsample_bilinear` in the core op set if compatible.

    Inputs:

        prog: Program
    """
    def apply(self, prog):
        for f in prog.functions.values():
            _paddle_upsample_to_core_upsample_block(f)

@block_context_manager
def _paddle_upsample_to_core_upsample_block(block):
    for op in block.operations[:]:
        for b in op.blocks:
            _paddle_upsample_to_core_upsample_block(b)

        if op.op_type in target_ops:
            if _try_replace_with_core_upsample(op):
                logger.info("Successfully map {} to core upsample".format(op.op_type))
            else:
                raise ValueError("Unable to map {} to core upsample".format(op.op_type))


def _try_get_upsample_factor(output_size):
    op = output_size
    # If the output has value, than the upsample op itself is derived from the upsample_1d op,
    # so we can just return scale factor 1 for that case
    if op.outputs[0].val is not None:
        assert op.outputs[0].val == 1.
        return 1.

    # output_size = [
    #       (paddle.floor((input.size(i + 2).float() * paddle.tensor(scale_factors[i], dtype=paddle.float32)).float()))
    #        for i in range(dim)
    #    ]
    # source from : https://pypaddle.org/docs/stable/_modules/paddle/nn/functional.html#interpolate
    # We validation if we can trace all the way back to the original scale_factor
    # The whole sequence is mul(input_size, scale_factor) -> cast(fp32) -> floor() -> cast(int32)

    # 1. check if the output_size is type 'cast' with dtype 'int32'
    if op.op_type != "cast" or op.dtype.val != "int32":
        return None

    # 2. check if the op is type 'floor'
    op = op.x.op
    if op.op_type != "floor":
        return None

    # 3. check if the op is type 'cast' with dtype 'fp32'
    op = op.x.op
    if op.op_type != 'cast' or op.dtype.val != "fp32":
        return None

    # 4. check if the op is type mul
    op = op.x.op
    if op.op_type != 'mul':
        return None

    # we successfully trace back the original scale factor
    return np.float32(op.y.val)


def _try_replace_with_core_upsample(op):
    """
    Inputs:

    op (Operation): op.op_type must be either
    1. `paddle_upsample_nearest_neighbor`
    2. `paddle_upsample_bilinear`

    Returns:

    True if op can be represented by mb.upsample_nearest_neighbor or mb.upsample_bilinear op in SSA.
    False otherwise
    """
    assert op.op_type in target_ops

    # 2d upsampling
    if op.op_type in ["paddle_upsample_nearest_neighbor", "paddle_upsample_bilinear"]:
        scales_h = _try_get_upsample_factor(op.output_height.op)
        scales_w = _try_get_upsample_factor(op.output_width.op)

        if scales_h is None or scales_w is None:
            return False

        old_upsample = op.outputs[0]
        block = op.enclosing_block

        if op.op_type == "paddle_upsample_nearest_neighbor":
            new_upsample = mb.upsample_nearest_neighbor(
                x=op.x,
                scale_factor_height=scales_h,
                scale_factor_width=scales_w,
                name=op.name,
                before_op=op,
            )
        elif op.op_type == "paddle_upsample_bilinear":
            new_upsample = mb.upsample_bilinear(
                x=op.x,
                scale_factor_height=scales_h,
                scale_factor_width=scales_w,
                align_corners=op.align_corners,
                name=op.name,
                before_op=op,
            )
        block.replace_uses_of_var_after_op(anchor_op=op, old_var=old_upsample, new_var=new_upsample)
        block.remove_ops([op])

    return True
