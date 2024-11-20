#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import Block
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import _check_child_op_type, block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="compression")
class insert_suffix_quantize_dequantize_pair(AbstractGraphPass):
    """
    Insert trailing quantize and dequantize operation pairs after valid patterns.

    .. code-block::
    Pattern 1:
    dequantize -> conv
        Given:
            %2 = dequantize(%1)
            %3 = conv(%2)
            ...
        Result:
            %2 = dequantize(%1)
            %3 = conv(%2)
            %4 = quantize(%3)
            %5 = dequantize(%4)
            ...

    Pattern 2:
    dequantize ->|
                 |-> add
    dequantize ->|
        Given:
            %2 = dequantize(%1)
            %4 = dequantize(%3)
            %5 = add(%2,%4)
            ...
        Result:
            %2 = dequantize(%1)
            %4 = dequantize(%3)
            %5 = add(%2,%4)
            %6 = quantize(%5)
            %7 = dequantize(%6)
            ...
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

    # Graph pass option for setting activation stats.
    _activation_stats = None

    @property
    def activation_stats(self) -> dict:
        return self._activation_stats

    @activation_stats.setter
    def activation_stats(self, activation_stats: dict):
        if not isinstance(activation_stats, dict):
            raise ValueError(
                f"activation_stats only supports dict, but got {type(activation_stats)}"
            )
        self._activation_stats = activation_stats

    # Graph pass option for setting compression config.
    _config = None

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value
        if value._op_selector is not None:
            self.op_selector = value._op_selector

    def apply(self, prog):
        visited_ops = set()
        for f in prog.functions.values():
            self._insert_quantize_dequantize(f, self._config, visited_ops)

    @block_context_manager
    def _insert_quantize_dequantize(self, block: Block, config, visited_ops: set):
        def help_insert_quantize_dequantize(block: Block) -> bool:
            fusion_occurred = False

            for op in list(block.operations):
                if op.enclosing_block is None:
                    continue

                if op in visited_ops:
                    continue
                visited_ops.add(op)

                for b in op.blocks:
                    self._insert_quantize_dequantize(b)

                # Must start with "dequantize" op.
                if op.op_type != "dequantize":
                    continue

                # Try matching valid patterns.
                if self._try_match_and_transform_pattern(op, block, config, visited_ops):
                    fusion_occurred = True

            return fusion_occurred

        block_changed = True
        while block_changed:
            block_changed = help_insert_quantize_dequantize(block)

    def _try_match_and_transform_pattern(
        self,
        dequantize_op: Operation,
        block: Block,
        config,
        visited_ops: set,
    ) -> bool:
        """
        This function performs the pattern match for all target patterns.
        It priorizes longer patterns to shorter ones for more fusions on hardware.
        Reject if the trailing `quantize` and `dequantize` pair already existed.

        A list of valid patterns.
        - conv
        - conv, activation
        - add
        - add, activation
        - pool (max_pool, avg_pool)

        E.g. Identify valid patterns:
        - (`quantize` ->) dequantize` -> `conv`
        - (`quantize` ->) dequantize` -> `conv` -> `relu`
        - (`quantize` ->) dequantize` -> `avg_pool`
        - (`quantize` ->) dequantize` -> `max_pool`
        E.g. Reject if trailing `quantize` -> `dequantize` exist:
        - (`quantize` ->) dequantize` -> `conv` -> `quantize` -> `dequantize`
        - (`quantize` ->) dequantize` -> `conv` -> `relu` -> `quantize` -> `dequantize`
        """

        # Reject if 1st operation is not `conv`/`add`/`pool`.
        SUPPORTED_OP_TYPES = ["conv", "add", "avg_pool", "max_pool"]
        if any([_check_child_op_type(dequantize_op, val) for val in SUPPORTED_OP_TYPES]):
            pass
        else:
            return False

        core_op = dequantize_op.outputs[0].child_ops[0]
        last_op = core_op

        # For operations with two inputs, both need to be `dequantize`.
        if core_op.op_type == "add":
            # Check both inputs
            in_var_x = core_op.inputs["x"]
            in_var_y = core_op.inputs["y"]
            in_x_prev_op = in_var_x.op
            in_y_prev_op = in_var_y.op
            if not (in_x_prev_op.op_type == "dequantize" and in_y_prev_op.op_type == "dequantize"):
                return False

        # Checking op-level config. Skip if we disable compression on certain operations.
        op_config = config._get_op_config(core_op)
        if op_config is None:
            return False

        # Reject if trailing `quantize` -> `dequantize` pair exist.
        if _check_child_op_type(core_op, "quantize"):
            return False

        _child_op = None
        if len(core_op.outputs[0].child_ops) > 0:
            _child_op = core_op.outputs[0].child_ops[0]

        # Check if 2nd operation is part of a valid pattern.
        # E.g. `dequantize` -> `conv` -> activation -> `quantize`.
        if _child_op is not None:
            if _child_op.op_type in self._allowed_activations:
                if len(_child_op.outputs[0].child_ops) > 0:
                    if _check_child_op_type(_child_op, "quantize"):
                        return False

                    _child_child_op = _child_op.outputs[0].child_ops[0]
                    last_op = _child_op
                    _child_op = _child_child_op

        return self._try_apply_transform(last_op, _child_op, block, visited_ops)

    def _try_apply_transform(
        self,
        last_op: Operation,
        _child_op: Operation,
        block: Block,
        visited_ops: set,
    ) -> bool:
        """
        last_op: last op of a valid pattern.
                 E.g. in `conv` -> `relu`, last_op is `relu`; in `conv`, last_op is `conv`.
        _child_op: the child op of the last_op.
        block: current block.
        visited_ops: a dict

        Pattern:
            Given:
                           |-> child_op_1
                last_op -> |-> child_op_2
                           |-> ...
            Result:
                                                     |-> child_op_1
                last_op -> quantize -> dequantize -> |-> child_op_2
                                                     |-> ...
        """
        if _child_op is None:
            return False

        scale_dtype = np.float16 if last_op.outputs[0].dtype == types.fp16 else np.float32

        new_last_op = getattr(mb, last_op.op_type)
        kargs = {}
        for k, v in last_op.inputs.items():
            kargs[k] = v
        kargs["name"] = last_op.name
        kargs["before_op"] = last_op
        new_last_op = new_last_op(**kargs)

        from coremltools.optimize.coreml._utils import get_min_and_max_values, quantize_weight

        var_name = last_op.outputs[0].name
        val = get_min_and_max_values(self._activation_stats, var_name)

        # Numerically the scale and zero point won't change if the input array only have two elements:
        # the min and max values of the input array. That's the trick to re-use quantize_weight util.
        _, _scale, _zero_point = quantize_weight(
            val,
            axes=0,
            nbits=8,
            signed=True,
            quantization_mode="LINEAR_SYMMETRIC",
            dtype=types.int8,
        )

        new_quantize_op = mb.quantize(
            input=new_last_op,
            scale=_scale,
            zero_point=_zero_point,
            output_dtype="int8",
            before_op=last_op,
        )
        new_dequantize_op = mb.dequantize(
            input=new_quantize_op,
            scale=_scale,
            zero_point=_zero_point,
            before_op=last_op,
        )
        ops_to_remove = [last_op]

        last_op_var_name = last_op.outputs[0].name
        # Replace output var of last_op with output of new_dequantize_op.
        if last_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=last_op,
            end_op=last_op,
            old_var=last_op.outputs[0],
            new_var=new_dequantize_op,
        ):
            block.remove_ops(ops_to_remove)
            # The name of new quantize/dequantize may change.
            # Add the new ones to the visited list to avoid revisiting.
            visited_ops.add(new_dequantize_op.op)
            visited_ops.add(new_quantize_op.op)
            new_dequantize_var_name = new_dequantize_op.name
            new_dequantize_op.set_name(f"{new_dequantize_var_name}__post__dequant")
            new_last_op.set_name(f"{last_op_var_name}")
            return True

        return False
