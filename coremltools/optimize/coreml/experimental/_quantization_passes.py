# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import numpy as np
from tqdm import tqdm

from coremltools import _logger as logger
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, Program, types
from coremltools.converters.mil.mil.block import is_current_opset_version_compatible_with
from coremltools.converters.mil.mil.passes.defs.quantization import AbstractQuantizationPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.optimize.coreml._config import OptimizationConfig
from coremltools.optimize.coreml.experimental._config import OpActivationLinearQuantizerConfig

"""
-----------------------------------
Activation compression graph pass -
-----------------------------------
"""


class AbstractActCompressionPass(AbstractQuantizationPass):
    """
    The abstract class for the activation compression graph passes.
    """

    _MINIMUM_OPSET_VERSION = AvailableTarget.iOS17

    def __init__(self, config: OptimizationConfig = None, fake_compression: bool = False):
        if not isinstance(config, (OptimizationConfig, type(None))):
            raise ValueError(f"config must be of type OptimizationConfig. Got {type(config)}.")

        op_selector = None if config is None else config._op_selector

        super().__init__(op_selector=op_selector)

        self.fake_compression = fake_compression
        self._config = config
        if config is not None:
            self._check_config_type(config)

    def apply(self, prog):
        if not isinstance(prog, Program):
            raise TypeError('Transform "{}" can only be applied on PyMIL programs.'.format(self))

        @block_context_manager
        def apply_block(block):
            if not is_current_opset_version_compatible_with(self._MINIMUM_OPSET_VERSION):
                logger.warning(
                    f"The program's opset is not compatible with {self._MINIMUM_OPSET_VERSION}. "
                    f"Skipped the compression pass {self.__class__}."
                )
                return

            valid_consts = []
            for op in list(block.operations):
                for b in op.blocks:
                    apply_block(b)

                if self.is_valid_op(op):
                    need_transform = True
                    if self.op_selector is not None:
                        need_transform = self.op_selector(op)

                    if need_transform:
                        valid_consts.append(op)

            for op in tqdm(
                valid_consts,
                desc=f"Running activation compression pass {self.__class__.__name__}",
                unit=" ops",
            ):
                with mb.set_before_op(op):
                    self.transform_op(op)

        for f in prog.functions.values():
            apply_block(f)

    @property
    def config(self) -> OptimizationConfig:
        return self._config

    @config.setter
    def config(self, value: OptimizationConfig):
        self._check_config_type(value)
        self._config = value
        if value._op_selector is not None:
            self.op_selector = value._op_selector

    def _check_config_type(self, config: OptimizationConfig):
        """
        The utility function is checking the OptimizationConfig is holding correct type of op config.
        """

        def get_supported_types_as_str(supported_type):
            if not isinstance(supported_type, (tuple, list)):
                supported_type = [supported_type]
            return ", ".join([f"{val.__name__}" for val in supported_type])

        all_configs = []
        if config.global_config is not None:
            all_configs.append(config.global_config)
        all_configs.extend(list(config.op_type_configs.values()))
        all_configs.extend(list(config.op_name_configs.values()))

        for config in all_configs:
            if not isinstance(config, self._SUPPORTED_CONFIG_TYPE) and config is not None:
                supported_type_str = get_supported_types_as_str(self._SUPPORTED_CONFIG_TYPE)
                raise ValueError(
                    f"{self.__class__.__name__} only accept {supported_type_str} type config. Got {config.__class__.__name__}."
                )

    def is_valid_op(self, op: Operation):
        return True


@register_pass(namespace="compression")
class insert_prefix_quantize_dequantize_pair(AbstractActCompressionPass):
    """
    This graph pass applies transform on each valid activation quantization pattern.
    A valid activation quantization pattern should be surrounded by a quantize/dequantize pair before and after this pattern.
    This transform adds a quantize/dequantize pair before valid activation quantization patterns.

    .. code-block::
        Input graph:
            ... -> downstream op
        Output graph:
            quantize -> dequantize -> downstream op
    """

    _SUPPORTED_CONFIG_TYPE = OpActivationLinearQuantizerConfig
    _MODE_DTYPE_TO_RANGE = {
        (types.int8, "LINEAR_SYMMETRIC"): (-127, 127),
    }

    SUPPORTED_UNARY_OP_TYPES = ["conv", "avg_pool", "max_pool"]
    SUPPORTED_BINARY_OP_TYPES = ["add"]
    SUPPORTED_OP_TYPES = SUPPORTED_UNARY_OP_TYPES + SUPPORTED_BINARY_OP_TYPES

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

    def transform_op(self, op: Operation):
        if op.op_type not in self.SUPPORTED_OP_TYPES:
            return False

        # Checking op-level config. Skip if we disable compression on certain ops.
        op_config = self.config._get_op_config(op)
        if op_config is None:
            return

        scale_dtype = None
        if op.inputs["x"].dtype == types.fp16:
            scale_dtype = np.float16
        else:
            scale_dtype = np.float32

        # Copy kargs from ``op`` to ``new_core_op``.
        kargs = {}
        for k, v in op.inputs.items():
            kargs[k] = v

        from coremltools.optimize.coreml._utils import get_min_and_max_values, quantize_weight

        if op.op_type in self.SUPPORTED_UNARY_OP_TYPES:
            var_name = op.inputs["x"].name
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
                input=op.inputs["x"],
                scale=_scale,
                zero_point=_zero_point,
                output_dtype="int8",
            )
            new_dequantize_op = mb.dequantize(
                input=new_quantize_op,
                scale=_scale,
                zero_point=_zero_point,
            )
            # Update kargs (input) of ``new_core_op``.
            kargs["x"] = new_dequantize_op

        elif op.op_type in self.SUPPORTED_BINARY_OP_TYPES:
            """
            For op with two live inputs (e.g. add):
            Input graph:
                ... ->|
                      |-> downstream op
                ... ->|
            Output graph:
                quantize -> dequantize |
                                       |-> downstream op
                quantize -> dequantize |
            """

            # Validation check.
            # Both inputs x and y need to be non-const.
            # Reject when either input is const.
            x_is_const = op.inputs["x"].op is not None and op.inputs["x"].op.op_type == "const"
            y_is_const = op.inputs["y"].op is not None and op.inputs["y"].op.op_type == "const"
            if x_is_const != y_is_const:
                return

            # Input "x"
            var_name = op.inputs["x"].name
            val = get_min_and_max_values(self._activation_stats, var_name)
            _, _scale, _zero_point = quantize_weight(
                val,
                axes=0,
                nbits=8,
                signed=True,
                quantization_mode="LINEAR_SYMMETRIC",
                dtype=types.int8,
            )
            new_quantize_op_x = mb.quantize(
                input=op.inputs["x"],
                scale=_scale,
                zero_point=_zero_point,
                output_dtype="int8",
            )
            new_dequantize_op_x = mb.dequantize(
                input=new_quantize_op_x,
                scale=_scale,
                zero_point=_zero_point,
            )

            # Input "y"
            var_name = op.inputs["y"].name
            val = get_min_and_max_values(self._activation_stats, var_name)
            _, _scale, _zero_point = quantize_weight(
                val,
                axes=0,
                nbits=8,
                signed=True,
                quantization_mode="LINEAR_SYMMETRIC",
                dtype=types.int8,
            )
            new_quantize_op_y = mb.quantize(
                input=op.inputs["y"],
                scale=_scale,
                zero_point=_zero_point,
                output_dtype="int8",
            )
            new_dequantize_op_y = mb.dequantize(
                input=new_quantize_op_y,
                scale=_scale,
                zero_point=_zero_point,
            )

            # Update kargs (inputs) of ``new_core_op``.
            kargs["x"] = new_dequantize_op_x
            kargs["y"] = new_dequantize_op_y

        # Update other kargs of ``new_core_op``.
        # These are the same regardless of whether it's a unary or binary op.
        kargs["name"] = op.name
        new_core_op = getattr(mb, op.op_type)(**kargs)
        new_core_op.name = op.outputs[0].name

        if new_core_op.op.enclosing_block.try_replace_uses_of_var_after_op(
            old_var=op.outputs[0],
            new_var=new_core_op,
            anchor_op=new_core_op.op,
            end_op=new_core_op,
        ):
            new_core_op.op.enclosing_block.remove_ops([op])
