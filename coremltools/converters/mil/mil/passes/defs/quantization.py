#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from enum import Enum as _Enum
from typing import Set, Text

import numpy as np

from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.program import Program


class ComputePrecision(_Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class AbstractQuantizationPass(AbstractGraphPass):
    """
    Base class for Post-Training Quantization transforms.

    Derived class needs to implement following two methods:
        - is_valid_op(op)
        - transform_op(op)
    """

    type_eps = {}
    type_min = {}
    type_negmin = {}

    def __init__(self, op_selector=None):
        super().__init__()
        if op_selector is not None and not callable(op_selector):
            raise TypeError(
                "Argument `op_selector` needs to be a callable function which accepts "
                "a MIL operation object and returns a boolean value."
            )
        self.op_selector = op_selector

    def apply(self, prog):
        """
        Walks over each operation in the graph and performs following two steps,
        1. Checks whether an operation is valid for that quantized transform using `is_valid_op` method.
        2. If yes, calls `transform_op` method of the derived quantized transform class.

        :param prog: MIL program
        :return: Transformed MIL program
        """
        if not isinstance(prog, Program):
            raise TypeError('Transform "{}" can only be applied on PyMIL programs.'.format(self))

        if getattr(self, "skip_ops_by_type", set()) and self.op_selector is not None:
            raise ValueError(
                "The graph pass option `skip_ops_by_type` cannot be set along with "
                "the `op_selector` in FP16ComputePrecision. Please only use one "
                "method to control which ops to operate on."
            )

        @block_context_manager
        def apply_block(block):
            for op in list(block.operations):
                for b in op.blocks:
                    apply_block(b)

                if self.is_valid_op(op):
                    need_transform: bool
                    if self.op_selector is not None:
                        need_transform = self.op_selector(op)
                    else:
                        need_transform = op.op_type not in getattr(self, "skip_ops_by_type", set())
                    if need_transform:
                        self.transform_op(op)

        for f in prog.functions.values():
            apply_block(f)

    def transform_op(self, op):
        """
        Replaces an op with a transformed op.

        :param op: MIL operation
        :return: None
        """
        raise NotImplementedError(
            'Op transformation for quantization mode "{}" not implemented.'.format(self)
        )

    def is_valid_op(self, op):
        """
        Checks whether an operation is valid for given quantized transform.

        :param op: MIL operation
        :return: true | false
        """
        raise NotImplementedError(
            'Operation Preconditions for quantization mode "{}" not implemented.'.format(self)
        )

    @classmethod
    def _close_to_zero(cls, val, np_type):
        if np_type not in cls.type_eps:
            cls.type_eps[np_type] = np.finfo(np_type).eps
            cls.type_min[np_type] = np.nextafter(0.0, 1.0, dtype=np_type)
            cls.type_negmin[np_type] = np.nextafter(0.0, -1.0, dtype=np_type)

        return np.isclose(val, 0, atol=cls.type_min[np_type], rtol=cls.type_eps[np_type])

    def __repr__(self):
        return str(self)

    def __str__(self):
        return type(self).__name__


class FP16ComputePrecision(AbstractQuantizationPass):
    """
    This transform does the following, for each valid op and if the "op_selector" return True:
    - For each input of dtype float32, inject a "cast" op to change it to float16 dtype
    - For each output of dtype float16, inject a "cast" op to change it back to float32
    """

    # Activation related ops with alpha/beta parameters.
    _ACTIVATION_ALPHA_OPS: Set[str] = {"elu", "leaky_relu", "prelu", "thresholded_relu"}
    _ACTIVATION_ALPHA_BETA_OPS: Set[str] = {
        "clamped_relu",
        "linear_activation",
        "scaled_tanh",
        "sigmoid_hard",
        "softplus_parametric",
    }
    _ELEMENTWISE_UNARY_EPSILON_OPS: Set[str] = {"inverse", "log", "rsqrt"}

    def __init__(self, op_selector=None):
        super(FP16ComputePrecision, self).__init__(op_selector=op_selector)
        self.target_dtype = "fp16"

        # Var that feeds into multiple ops will be casted once and cached into this dict
        # For reference: Checkout test_single_input_to_multiple_operations in `TestFP16CastTransform`.
        self.cache_vars = {}

    def fp16_overflow(self, op: Operation) -> bool:
        # This overflow check consists of two parts:
        # 1. For valid fp32 numbers (abs < 1e38), we want their exact values,
        #    so we make sure they are within fp16 range [-65504, 65504]
        # 2. For inifinities (abs >= 1e38), their exact values does not matter,
        #    so we can always downcast them to fp16 inf. For example, in attention mask
        #    we just want -inf to make the masked entries have 0 probability after softmax
        for _, inputs in op.inputs.items():
            is_list_input = isinstance(inputs, (list, tuple))
            if not is_list_input:
                inputs = [inputs]
            for var in inputs:
                if (
                    var.op is not None
                    and var.op.op_type == "const"
                    and var.is_tensor_or_scalar_of(dtype="fp32")
                ):
                    value = np.expand_dims(var.op.val.val, 0)
                    abs_value = np.abs(value)
                    if np.max(abs_value[np.where(abs_value < 1e38)], initial=0.0) > 65504:
                        return True
        return False

    def is_valid_op(self, op: Operation) -> bool:
        """Determines if op is valid for fp16 casting."""
        if op.op_type in ["cast", "while_loop", "cond"]:
            return False

        # TODO: Remove after supporting FP16 dynamic quantize transformation for list ops (rdar://74458192)
        if op.op_type in [
            "make_list",
            "list_gather",
            "list_scatter",
            "list_read",
            "list_write",
            "list_length",
        ]:
            return False

        if self.fp16_overflow(op):
            return False

        return True

    def should_cast_parameter(self, op: Operation, param_name: str) -> bool:
        """Determines if a param of an op should be casted to fp16."""
        # Make sure the param is valid for fp16 when type domain is specified.
        type_domain = getattr(op.input_spec.input_types[param_name], "type_domain", None)
        if type_domain and types.fp16 not in type_domain:
            return False

        if op.opset_version >= AvailableTarget.iOS17:
            # In IOS17+ activation ops with alpha/beta support mixed precision, and we don't want to
            # cast alpha/beta to fp16 for better numerical accuracy.
            if op.op_type in self._ACTIVATION_ALPHA_OPS and param_name == "alpha":
                return False
            if op.op_type in self._ACTIVATION_ALPHA_BETA_OPS and param_name in {"alpha", "beta"}:
                return False

            # Element-wise unary ops with epsilon also support mixed precision.
            if op.op_type in self._ELEMENTWISE_UNARY_EPSILON_OPS and param_name == "epsilon":
                return False

        return True

    def _check_underflow_to_zero(self, new_var, var):
        # We check whether there are casted values that "becomes" 0 which is not ideal for eps purposes.
        # However we skip arrays with more than 400 in case we compare through a large sparse matrix.
        if (
            new_var.val is not None
            and len(var.val.flatten()) < 400
            and self._close_to_zero(new_var.val, np.float16).any()
        ):
            value_modified = False
            original_val = var.val.flatten()
            new_val = new_var.val.flatten()

            for idx in range(len(original_val)):
                if not self._close_to_zero(original_val[idx], np.float32) and self._close_to_zero(
                    new_val[idx], np.float16
                ):
                    new_val[idx] = (
                        self.type_min[np.float16]
                        if np.sign(original_val[idx]) > 0
                        else self.type_negmin[np.float16]
                    )
                    value_modified = True

            if value_modified:
                if np.isscalar(new_var.val):
                    new_var._sym_val.val = new_val[0]
                else:
                    new_var._sym_val.val = new_val.reshape(new_var.val.shape)

    def transform_op(self, op):
        block = op.enclosing_block
        casted_inputs = {}
        inputs_modified = False

        for param, inputs in op.inputs.items():
            # First loop, iterates over all the input parameters of an operation.
            if not self.should_cast_parameter(op, param):
                continue

            is_list_input = isinstance(inputs, (list, tuple))
            if not is_list_input:
                inputs = [inputs]

            casted_inputs[param] = list(inputs[:])
            for i, var in enumerate(inputs):
                # Second loop, iterates over all the vars of a python list corresponding to an input parameter.
                if not var.is_tensor_or_scalar_of(dtype="fp32"):
                    continue

                inputs_modified = True
                casted_var_name = var.name + "_to_fp16"
                if (
                    len(var._child_ops) > 1
                    and casted_var_name in self.cache_vars
                    and (block.is_var_visible_in_block(self.cache_vars[casted_var_name]))
                ):
                    casted_inputs[param][i] = self.cache_vars[casted_var_name]
                else:
                    x = mb.cast(x=var, dtype="fp16", name=casted_var_name, before_op=op)
                    self._check_underflow_to_zero(x, var)

                    casted_inputs[param][i] = x
                    if len(var._child_ops) > 1:
                        self.cache_vars[casted_var_name] = casted_inputs[param][i]

            if not is_list_input:
                casted_inputs[param] = casted_inputs[param][0]

        if inputs_modified:
            casted_inputs.update({k: v for k, v in op.inputs.items() if k not in casted_inputs})
            casted_inputs["name"] = op.name + "_cast"
            casted_inputs["before_op"] = op
            quant_output = getattr(mb, op.op_type)(**casted_inputs)

            if not isinstance(quant_output, (list, tuple)):
                quant_output = [quant_output]

            for old_output_var, new_output_var in zip(op.outputs, quant_output):
                if old_output_var.is_tensor_or_scalar_of(dtype="fp32") and (
                    not new_output_var.is_tensor_or_scalar_of(dtype="fp32")
                ):
                    x = mb.cast(
                        x=new_output_var,
                        dtype="fp32",
                        name=new_output_var.name + "_to_fp32",
                        before_op=op,
                    )
                    op.enclosing_block.replace_uses_of_var_after_op(
                        anchor_op=op,
                        old_var=old_output_var,
                        new_var=x,
                        force_replace=True,
                    )
                else:
                    op.enclosing_block.replace_uses_of_var_after_op(
                        anchor_op=op,
                        old_var=old_output_var,
                        new_var=new_output_var,
                        force_replace=True,
                    )

            block.remove_ops([op])


@register_pass(namespace="common")
class add_fp16_cast(FP16ComputePrecision):
    """
    For each input of dtype float32, inject a ``cast`` op to change it to float16 dtype.

    For each output of dtype float16, inject a ``cast`` op to change it back to float32.

    This pass is the registered interface for FP16ComputePrecision, which makes it consistent with
    other passes' interfaces.

    Support options:

    - ``skip_ops_by_type``: Skip op types specified by comma-separated string; for example, ``"mul,const"``.
    """

    _skip_ops_by_type: Set[Text] = set()

    @property
    def skip_ops_by_type(self):
        return self._skip_ops_by_type

    @skip_ops_by_type.setter
    def skip_ops_by_type(self, criteria: Text):
        self._skip_ops_by_type = set(criteria.split(","))
