#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Optional, Tuple
import numpy as np

from coremltools.converters.mil.mil import Block
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, types
from coremltools.converters.mil.mil.passes.graph_pass import (
    AbstractGraphPassWithOptimizationConfig,
    AbstractGraphPassWithSampleData,
)
from coremltools.converters.mil.mil.passes.helper import _check_child_op_type, block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
#from coremltools.optimize.coreml._utils import quantize_weight


def get_quant_range(n_bits: int, signed: bool, mode: str) -> Tuple[int, int]:
    """
    Utility to get the quantization range for a given quantization config
    """
    max_q = 2**n_bits
    if not signed:
        quant_min = 0
        quant_max = max_q - 1
        if mode == "LINEAR_SYMMETRIC":
            quant_max -= 1
    else:
        quant_min = -max_q / 2
        quant_max = max_q / 2 - 1
        if mode == "LINEAR_SYMMETRIC":
            quant_min += 1
    return int(quant_min), int(quant_max)


def quantize_weight(
    weight: np.ndarray,
    axes: Tuple[int, ...],
    nbits: int,
    signed: bool,
    quantization_mode: str,
    dtype: np.dtype,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Get quantized data along with metadata (scale, zero_point)."""
    if not np.issubdtype(weight.dtype, np.floating):
        raise ValueError("Only floating numpy arrays are supported.")

    val_min = np.amin(weight, axis=axes, keepdims=True)
    val_max = np.amax(weight, axis=axes, keepdims=True)

    q_val_min, q_val_max = get_quant_range(nbits, signed, quantization_mode)

    zero_point = None
    if quantization_mode == "LINEAR_SYMMETRIC":
        # For the linear_symmetric quantization_mode, the range is symmetrical to 0
        max_abs = np.maximum(np.abs(val_min), np.abs(val_max))
        val_min = -max_abs
        val_max = max_abs

        if not signed:
            zero_point_shift = q_val_max // 2
            zero_point = zero_point_shift * np.ones(val_min.shape)
    else:
        assert quantization_mode == "LINEAR"
        # For the linear quantization_mode, we need to make sure the data range contains `0`
        val_min = np.minimum(0.0, val_min)
        val_max = np.maximum(0.0, val_max)
        zero_point = (q_val_min * val_max - q_val_max * val_min) / (val_max - val_min)
        zero_point = np.round(zero_point)
        zero_point = np.clip(zero_point, q_val_min, q_val_max)

    scale = (val_max - val_min) / (q_val_max - q_val_min)
    quantized_data = np.round(weight / scale)
    if zero_point is not None:
        quantized_data += zero_point
        zero_point = zero_point.squeeze().astype(dtype)
    quantized_data = np.clip(quantized_data, q_val_min, q_val_max).astype(dtype)
    scale = scale.astype(weight.dtype).squeeze()

    return quantized_data, scale, zero_point


@register_pass(namespace="compression")
class insert_quantize_dequantize(AbstractGraphPassWithOptimizationConfig):
    """
    .. code-block::
    dequantize -> conv
    Pattern 1:
        Given:
            %3 = dequantize(%2)
            %4 = conv(%3)
            [NOT] %5 = quantize(%4, axes=-2)
            ...
        Result:
            %3 = dequantize(%2)
            %4 = conv(%3)
            %5 = quantize(%4, axes=-2)
            %6 = dequantize(%5)
            ...
    """

    def apply(self, prog, config):
        visited_ops = {}
        for f in prog.functions.values():
            self._insert_quantize_dequantize(f, config, visited_ops)

    @block_context_manager
    def _insert_quantize_dequantize(self, block: Block, config, visited_ops: dict):

        def help_insert_quantize_dequantize(block: Block) -> bool:
            fusion_occurred = False

            for op in list(block.operations):
                if op.enclosing_block is None:
                    continue

                if op in visited_ops:
                    continue
                visited_ops[op] = 1

                for b in op.blocks:
                    self._insert_quantize_dequantize(b)

                # must start with "dequantize" op,
                if op.op_type != "dequantize":
                    continue

                # try pattern I
                # `dequant` -> `conv`
                if self._try_match_and_transform_pattern(op, block, config, visited_ops):
                    # has to break as the downstream iterator is affected
                    # visited_ops.add(op)
                    return True

                # try pattern II

            return fusion_occurred

        block_changed = True
        while block_changed:
            block_changed = help_insert_quantize_dequantize(block)

    def _try_match_and_transform_pattern(
        self, dequantize_op: Operation, block: Block, config, visited_ops: dict
    ) -> bool:
        """
        This function performs the validation for target pattern.
        identify the pattern:
        - (`quantize` ->) dequantize` -> `conv` -> [new_op]
        - (`quantize` ->) dequantize` -> `conv` -> `relu`-> [new_op]
        Reject if trailing `quantize` -> dequantize` pairs exist.
        - (`quantize` ->) dequantize` -> `conv` -> `quantize` -> dequantize`
        - (`quantize` ->) dequantize` -> `conv` -> `relu` -> `quantize` -> dequantize`
        """
        _allowed_activations = {
            "leaky_relu",
            "tanh",
            "scaled_tanh",
            "sigmoid",
            "hard_sigmoid",
            "relu",
            "relu6",
        }
        # `dequantize` -> `conv`
        if not _check_child_op_type(dequantize_op, "conv"):
            return False
        conv_op = dequantize_op.outputs[0].child_ops[0]
        last_op = conv_op

        # Checking op-level config. Skip if we disable compression on certain ops.
        op_config = config._get_op_config(conv_op)
        if op_config is None:
            return False

        # `dequantize` ->`conv` -> `quantize`
        if _check_child_op_type(conv_op, "quantize"):
            return False
        _child_op = None
        if len(conv_op.outputs[0].child_ops) > 0:
            _child_op = conv_op.outputs[0].child_ops[0]

        # `dequantize` -> `conv` -> activation (relu, etc.) -> `quantize`
        if _child_op is not None:
            if _child_op.op_type in _allowed_activations:
                if len(_child_op.outputs[0].child_ops) > 0:
                    if _check_child_op_type(_child_op, "quantize"):
                        return False

                    _child_child_op = _child_op.outputs[0].child_ops[0]
                    last_op = _child_op
                    _child_op = _child_child_op

        # everything looks good
        return self._try_apply_transform(dequantize_op, last_op, _child_op, block, visited_ops)

    @staticmethod
    def _try_apply_transform(
        dequantize_op: Operation,
        last_op: Operation,
        _child_op: Operation,
        block: Block,
        visited_ops: dict,
    ) -> bool:
        """
        last_op: last op of a valid pattern.
                 E.g. in `conv` -> `relu`, last_op is `relu`; in `conv`, last_op is `conv`.
        _child_op: the child op of the last_op.
        block: current block.
        """
        if _child_op is None:
            return False

        scale_dtype = np.float16 if last_op.outputs[0].dtype == types.fp16 else np.float32

        # -> quantize -> dequantize
        new_quantize_op = mb.quantize(
            input=last_op.outputs[0],
            scale=np.array(1).astype(scale_dtype),
            zero_point=np.int8(0),
            output_dtype="int8",
            before_op=_child_op,
        )
        new_dequantize_op = mb.dequantize(
            input=new_quantize_op,
            scale=np.array(1).astype(scale_dtype),
            zero_point=np.int8(0),
            before_op=_child_op,
        )

        if _child_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=new_dequantize_op.op,
            end_op=_child_op,  #
            old_var=last_op.outputs[0],  #
            new_var=new_dequantize_op,  #
        ):
            visited_ops[new_dequantize_op.op] = 1
            # the name of new quantize/dequantize op change, need to add the new ones to visted list
            return True

        return False


# This is for updating quantize and dequantize op args.
@register_pass(namespace="compression")
class update_quantize_dequantize(AbstractGraphPassWithSampleData):
    """
    .. code-block::
    dequantize -> conv
    Pattern 1:
        Given:
            %2 = quantize(%1) with random scale and zp
            %3 = dequantize(%2) with random scale and zp
            ...
        Result:
            %2 = quantize(%1) with new scale and zp
            %3 = dequantize(%2) with new scale and zp
            ...
    """

    def apply(self, prog, activation_stats):  # [NEW] activation_stats
        visited_ops = {}
        for f in prog.functions.values():
            self._update_quantize_dequantize(f, activation_stats, visited_ops)

    @block_context_manager
    def _update_quantize_dequantize(self, block: Block, activation_stats: dict, visited_ops: dict):

        def help_update_quantize_dequantize(block: Block, activation_stats: dict) -> bool:
            fusion_occurred = False

            for op in list(block.operations):
                if op.enclosing_block is None:
                    continue

                if op in visited_ops:
                    continue
                visited_ops[op] = 1

                for b in op.blocks:
                    self._update_quantize_dequantize(b, activation_stats)

                # must start with "quantize" op
                if op.op_type != "quantize":
                    continue

                # try pattern I
                # `quantize` -> `dequant`
                if self._try_match_and_transform_pattern(op, block, activation_stats, visited_ops):
                    # has to break as the downstream iterator is affected
                    # visited_ops.add(op)
                    return True

                # try pattern II, if any.

            return fusion_occurred

        block_changed = True
        while block_changed:
            block_changed = help_update_quantize_dequantize(block, activation_stats)

    def _try_match_and_transform_pattern(
        self, quantize_op: Operation, block: Block, activation_stats: dict, visited_ops: dict
    ) -> bool:
        """
        This function performs the validation for target pattern.
        identify the pattern:
        - `quantize` -> dequantize` -> [new_op]
        """

        # `quantize` -> dequantize`
        if not _check_child_op_type(quantize_op, "dequantize"):
            return False
        dequantize_op = quantize_op.outputs[0].child_ops[0]
        last_op = dequantize_op

        _child_op = None
        if len(dequantize_op.outputs[0].child_ops) > 0:
            _child_op = dequantize_op.outputs[0].child_ops[0]

        # everything looks good
        return self._try_apply_transform(
            quantize_op, last_op, _child_op, block, activation_stats, visited_ops
        )

    @staticmethod
    def _try_apply_transform(
        quantize_op: Operation,
        last_op: Operation,
        _child_op: Operation,
        block: Block,
        activation_stats: dict,
        visited_ops: dict,
    ) -> bool:
        """
        last_op: last op of a valid pattern. it's 'dequantize' in this case.
        _child_op: the child op of the last_op.
        block: current block.
        """
        ops_to_remove = [quantize_op, last_op]

        if _child_op is None:
            return False

        # name of input var to `quantize`
        in_var_name = quantize_op.inputs["input"].name
        val = np.array([0, 0], dtype=np.float16)
        val[0], val[1] = (
            activation_stats[in_var_name]["rmin"],
            activation_stats[in_var_name]["rmax"],
        )
        # Numerically the scale and zp won't change if the input array only have two elements:
        # the min and max of input array. Plus we don't care about quantized values.
        # That's the trick to re-use quantize_weight.
        _, _scale, _zero_point = quantize_weight(
            val,  # ndarray[rmin, rmax]
            axes=0,  # axes
            nbits=8,
            signed=True,  # not dtype.is_unsigned(),
            quantization_mode="LINEAR_SYMMETRIC",  # mode,
            dtype=types.int8,  # types.nptype_from_builtin(dtype)
        )

        # New quantize -> dequantize
        new_quantize_op = mb.quantize(
            input=quantize_op.input,
            scale=_scale,
            zero_point=_zero_point,
            output_dtype="int8",
            before_op=_child_op,
        )
        new_dequantize_op = mb.dequantize(
            input=new_quantize_op,
            scale=_scale,
            zero_point=_zero_point,
            before_op=_child_op,
        )

        # Replace old ``quantize -> dequantize`` with new ``quantize -> dequantize`` to update scale/zero_point args
        if last_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=last_op,
            end_op=last_op,
            old_var=last_op.outputs[0],
            new_var=new_dequantize_op,
        ):
            block.remove_ops(ops_to_remove)
            visited_ops[
                new_quantize_op.op
            ] = 1  # the name of new quantize/dequantize op change, need to add the new ones to visted list
            return True
        return False
