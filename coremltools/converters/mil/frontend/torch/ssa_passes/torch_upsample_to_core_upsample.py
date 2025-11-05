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
    "torch_upsample_nearest_neighbor",
    "torch_upsample_bilinear",
]


@register_pass(namespace="torch")
class torch_upsample_to_core_upsample(AbstractGraphPass):
    """
    Try to map Torch dialect ops
    1. `torch_upsample_nearest_neighbor`
    2. `torch_upsample_bilinear`
    to `upsample_nearest_neighbor` or `upsample_bilinear` in the core op set if compatible.

    Inputs:

        prog: Program
    """
    def apply(self, prog):
        for f in prog.functions.values():
            _torch_upsample_to_core_upsample_block(f)

@block_context_manager
def _torch_upsample_to_core_upsample_block(block):
    for op in list(block.operations):
        for b in op.blocks:
            _torch_upsample_to_core_upsample_block(b)

        if op.op_type in target_ops:
            if _try_replace_with_core_upsample(op):
                msg = f"Successfully map {op.op_type} to core upsample"
                logger.info(msg)
            else:
                raise ValueError("Unable to map {} to core upsample".format(op.op_type))

def _try_get_upsample_factor_pattern_2(output_size, expected_gather_indices, target_op):
    """
    This is the pattern corresponds to the python source code:

    class UpsampleBilinear(nn.Module):
        def forward(self, x):
            b, c, h, w = x.shape
            return F.interpolate(x, size=(h*2, w*2), mode='bilinear', align_corners=False)

    The resulting pymil program is:

    function[CoreML5](%x: (1, 3, is0, is1, fp32)(Tensor)) {
      block0() {
        %3_shape: (4,int32)^(Tensor) = shape(x=%x, name="3_shape")
        %gather_0: (int32)^(Scalar) = gather(x=%3_shape, indices=2, axis=0, name="gather_0")
        %6_shape: (4,int32)^(Tensor) = shape(x=%x, name="6_shape")
        %gather_1: (int32)^(Scalar) = gather(x=%6_shape, indices=3, axis=0, name="gather_1")
        %9: (int32)(Scalar) = mul(x=%gather_0, y=2, name="9")
        %10: (int32)(Scalar) = cast(x=%9, dtype="int32", name="10")
        %12: (int32)(Scalar) = mul(x=%gather_1, y=2, name="12")
        %13: (int32)(Scalar) = cast(x=%12, dtype="int32", name="13")
        %17: (1, 3, is2, is3, fp32)(Tensor) = torch_upsample_bilinear(x=%x, output_height=%10, output_width=%13, align_corners=False, name="17")

    We do a pattern matching to extract the scale value.
    """
    # cast op
    op = output_size
    if op.op_type != "cast" or op.dtype.val != "int32":
        return None

    # mul op
    mul_op = op.x.op
    if mul_op.op_type != "mul":
        return None
    mul_op_y = mul_op.y

    # gather op
    gather_op = mul_op.x.op
    if gather_op.op_type != "gather":
        return None
    if gather_op.indices.val != expected_gather_indices:
        return None
    if gather_op.axis.val != 0:
        return None

    # shape op
    shape_op = gather_op.x.op
    if shape_op.op_type != "shape":
        return None
    if shape_op.x != target_op:
        return None

    return mul_op_y.val


def _try_get_upsample_factor_pattern_1(output_size):
    op = output_size
    # If the output has value, then the upsample op itself is derived from the upsample_1d op,
    # so we can just return scale factor 1 for that case
    if op.outputs[0].val is not None:
        assert op.outputs[0].val == 1.
        return 1.

    # output_size = [
    #       (torch.floor((input.size(i + 2).float() * torch.tensor(scale_factors[i], dtype=torch.float32)).float()))
    #        for i in range(dim)
    #    ]
    # source from : https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#interpolate
    # We validate if we can trace all the way back to the original scale_factor
    if op.op_type == "mul":
        # we successfully trace back the original scale factor
        assert op.y.val is not None, "scale factor should be const"
        return np.float32(op.y.val)
    else:
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
        if op.op_type != "cast" or op.dtype.val != "fp32":
            return None
        # 4. check if the op is type mul
        op = op.x.op
        if op.op_type != "mul":
            return None
        # we successfully trace back the original scale factor
        assert op.y.val is not None, "scale factor should be const"
        return np.float32(op.y.val)


def _try_replace_with_core_upsample(op):
    """
    Inputs:

    op (Operation): op.op_type must be either
    1. `torch_upsample_nearest_neighbor`
    2. `torch_upsample_bilinear`

    Returns:

    True if op can be represented by mb.upsample_nearest_neighbor or mb.upsample_bilinear op in SSA.
    False otherwise
    """
    assert op.op_type in target_ops

    # 2d upsampling
    if op.op_type in target_ops:

        # try to resolve the scaling factor - pattern 1
        scales_h = _try_get_upsample_factor_pattern_1(op.output_height.op)
        scales_w = _try_get_upsample_factor_pattern_1(op.output_width.op)

        if scales_h is None or scales_w is None:
            scales_h = _try_get_upsample_factor_pattern_2(op.output_height.op, 2, op.x)
            scales_w = _try_get_upsample_factor_pattern_2(op.output_width.op, 3, op.x)

        if scales_h is None or scales_w is None:
            return False

        old_upsample = op.outputs[0]
        block = op.enclosing_block

        if op.op_type == "torch_upsample_nearest_neighbor":
            new_upsample = mb.upsample_nearest_neighbor(
                x=op.x,
                scale_factor_height=scales_h,
                scale_factor_width=scales_w,
                name=op.name,
                before_op=op,
            )
        elif op.op_type == "torch_upsample_bilinear":
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
