#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from abc import abstractmethod
from enum import Enum as _Enum
from typing import Dict, Set, Text, Tuple

import numpy as np

from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.input_types import TensorType
from coremltools.converters.mil.mil import Block
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Operation, Var, types
from coremltools.converters.mil.mil.block import is_current_opset_version_compatible_with
from coremltools.converters.mil.mil.ops.registry import SSAOpRegistry
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.program import Program
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from coremltools.converters.mil.mil.types.type_mapping import string_to_builtin


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

        # Var that feeds into multiple ops will be cast once and cached into this dict
        # For reference: Checkout test_single_input_to_multiple_operations in `TestFP16CastTransform`.
        # Note that, we make it a stack of dict to keep tracking the blocks
        self._cache_vars = []

    def current_cache_vars(self) -> Set[Var]:
        return self._cache_vars[-1]

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
            self._cache_vars.append({})
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
            self._cache_vars.pop()

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


class CastTypeQuantization(AbstractQuantizationPass):
    """
    Base class for all type casting related quantization, such as fp32->fp16, int32->int16, etc.

    For each valid op, if the "op_selector" return True:
    - For each input with dtype `origin_dtype`, inject a "cast" op to change it to `target_dtype`.
    - For each output with dtype `target_dtype`, inject a "cast" op to change it back to `origin_dtype`.
    All child classes need to specify `origin_dtype` and `target_dtype`.
    """

    def __init__(self, op_selector=None):
        super().__init__(op_selector=op_selector)

    @property
    @abstractmethod
    def origin_dtype(self) -> str:
        """Original dtype that need to be cast, such as fp32."""
        raise NotImplementedError("origin_dtype must be specified in subclass.")

    @property
    @abstractmethod
    def target_dtype(self) -> str:
        """Target dtype, such as fp16."""
        raise NotImplementedError("target_dtype must be specified in subclass.")

    # TODO: rdar://122845072 ([Infra] Refactor the transform_function_signatures, adjust_io_to_supported_types and update_output_dtypes using a shared graph pass)
    @block_context_manager
    def transform_function_signatures(self, func: Function) -> None:
        """
        This utility transform a function input / output signatures from the original_dtype to
        the target_dtype.

        For instance, in the add_fp16_cast class, this member function transforms the following
        function:

            function(%input(fp32)) {
              block0() {
                % var_1 = op_1(x=%input)
                ...
                % output(fp32) = ...
              } -> (%output)
            }

        into:

            function(%input(fp16)) {
              block0() {
                # input_cast = cast(x=input, dtype="fp32")
                % var_1 = op_1(x=%input_cast)
                ...
                % output(fp32) = ...
              } -> (%output)
            }

        and function.output_types is set to [TensorType(dtype=types.fp16)],
        in which will be used in common::update_output_dtypes to upgrade the function output dtype accordingly.

        """
        # reset input signatures
        old_func_inputs = func.inputs
        new_func_inputs = {}
        cache_vars = {}

        # cast the new input into the original dtype
        for k, v in old_func_inputs.items():
            if v.is_tensor_or_scalar_of(self.origin_dtype):
                new_input = mb.placeholder(
                    shape=v.shape,
                    dtype=string_to_builtin(self.target_dtype),
                    name=v.name,
                ).outputs[0]

                if v in func.outputs:
                    new_outputs = []
                    for val in func.outputs:
                        new_outputs.append(new_input if val == v else val)
                    func.set_outputs(new_outputs)

                new_func_inputs[k] = new_input
                cast_input = mb.cast(
                    x=new_input,
                    dtype=self.origin_dtype,
                    before_op=func.operations[0] if len(func.operations) > 0 else None,
                )
                cache_vars[k] = cast_input
            else:
                new_func_inputs[k] = v
                cache_vars[k] = v

        # replace the use of the old input vars with the new cast var
        for k, v in old_func_inputs.items():
            func.replace_uses_of_var_after_op(
                anchor_op=None,
                old_var=v,
                new_var=cache_vars[k],
            )
        func._input_dict = new_func_inputs

        # reset output signatures
        if func.output_types is None:
            output_types = [TensorType(dtype=v.dtype) for v in func.outputs]
        else:
            output_types = func.output_types

        for idx, v in enumerate(output_types):
            if v.dtype == string_to_builtin(self.origin_dtype):
                output_types[idx] = TensorType(dtype=string_to_builtin(self.target_dtype))

        func.output_types = output_types

    def should_cast_parameter(self, op: Operation, param_name: str) -> bool:
        """
        Determines if a param of an op should be cast to target_dtype.

        There are two cases that an op shouldn't be cast:
        1. The op's parameter doesn't support target_dtype.
        2. The cast op itself doesn't support target_dtype
        """
        type_domain = getattr(op.input_spec.input_types[param_name], "type_domain", None)
        if type_domain and types.string_to_builtin(self.target_dtype) not in type_domain:
            return False
        if self.target_dtype not in SSAOpRegistry._get_core_op_cls("cast").supported_dtypes():
            return False

        return True

    def _get_casted_outputs(self, op: Operation, casted_inputs: Dict[str, Var]) -> Tuple[Var]:
        """
        Given an op and casted_inputs, this utility returns the new resulting outputs.
        """
        return getattr(mb, op.op_type)(**casted_inputs)


    def transform_op(self, op) -> None:
        """Transform the input(s)/output(s) dtypes of the op."""
        block = op.enclosing_block
        casted_inputs = {}
        inputs_modified = False

        for param, inputs in op.inputs.items():
            if not self.should_cast_parameter(op, param):
                continue

            is_list_input = isinstance(inputs, (list, tuple))
            if not is_list_input:
                inputs = [inputs]

            casted_inputs[param] = list(inputs[:])
            for i, var in enumerate(inputs):
                if not var.is_tensor_or_scalar_of(dtype=self.origin_dtype):
                    continue

                inputs_modified = True
                casted_var_name = f"{var.name}_to_{self.target_dtype}"
                if (
                    len(var._child_ops) > 1
                    and casted_var_name in self.current_cache_vars()
                ):
                    casted_inputs[param][i] = self.current_cache_vars()[casted_var_name]
                else:
                    x = mb.cast(
                        x=var,
                        dtype=self.target_dtype,
                        name=casted_var_name,
                        before_op=op,
                    )
                    if self.target_dtype == "fp16":
                        self._check_underflow_to_zero(x, var)
                    Block._copy_metadata(var, x)

                    casted_inputs[param][i] = x
                    if len(var._child_ops) > 1:
                        self.current_cache_vars()[casted_var_name] = casted_inputs[param][i]

            if not is_list_input:
                casted_inputs[param] = casted_inputs[param][0]

        if inputs_modified:
            casted_inputs.update({k: v for k, v in op.inputs.items() if k not in casted_inputs})
            casted_inputs["name"] = f"{op.name}_cast_{self.target_dtype}"
            casted_inputs["before_op"] = op
            quant_output = self._get_casted_outputs(op, casted_inputs)

            if not isinstance(quant_output, (list, tuple)):
                quant_output = [quant_output]

            for old_output_var, new_output_var in zip(op.outputs, quant_output):
                if old_output_var.is_tensor_or_scalar_of(dtype=self.origin_dtype) and (
                    not new_output_var.is_tensor_or_scalar_of(dtype=self.origin_dtype)
                ):
                    x = mb.cast(
                        x=new_output_var,
                        dtype=self.origin_dtype,
                        name=f"{new_output_var.name}_to_{self.origin_dtype}",
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


class FP16ComputePrecision(CastTypeQuantization):
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

    # Unsupported op for fp16 casting
    _UNSUPPORTED_FP16_OPS: Set[str] = {
        "cast",
        "while_loop",
        "cond",
        # TODO: Remove after supporting FP16 dynamic quantize transformation for list ops (rdar://74458192)
        "make_list",
        "list_gather",
        "list_scatter",
        "list_read",
        "list_write",
        "list_length",
        "read_state",
        "coreml_update_state",
    }

    def __init__(self, op_selector=None):
        super(FP16ComputePrecision, self).__init__(op_selector=op_selector)

    @property
    def origin_dtype(self) -> str:
        return "fp32"

    @property
    def target_dtype(self) -> str:
        return "fp16"

    @staticmethod
    def fp16_overflow(op: Operation) -> bool:
        """
        Determines if any of the op's input will overflow when represented by FP16.

        This overflow check consists of two parts:
        1. For valid fp32 numbers (abs < 1e38), we want their exact values,
           so we make sure they are within fp16 range [-65504, 65504]
        2. For inifinities (abs >= 1e38), their exact values does not matter,
           so we can always downcast them to fp16 inf. For example, in attention mask
           we just want -inf to make the masked entries have 0 probability after softmax
        """
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
        if op.op_type in self._UNSUPPORTED_FP16_OPS:
            return False

        if self.fp16_overflow(op):
            return False

        return True

    def should_cast_parameter(self, op: Operation, param_name: str) -> bool:
        """Determines if a param of an op should be cast to fp16."""
        if not super().should_cast_parameter(op, param_name):
            return False

        if is_current_opset_version_compatible_with(AvailableTarget.iOS17):
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


@register_pass(namespace="common")
class add_int16_cast(CastTypeQuantization):
    """
    This transform does the following, for each op that supports int16/uint16:
    - For each input of dtype int32 which supports int16/uint16, inject a "cast" op to change it
      to int16/uint16 dtype.
    - For each output of dtype int16/uint16, inject a "cast" op to change it back to int32.
    Notice that the cast will not be inserted if the const value is out of int16/uint16 range.
    """
    # Ops that prefer int16 params.
    # If an op supports 16-bit only in later iOS (e.g. gather started to support 16-bit from iOS16)
    # then int16 cast will be inserted only if the iOS version is high enough
    # (e.g. nothing will happen for iOS15 gather)
    # This is achieved by type domain confirmation in `CastTypeQuantization.should_cast_parameter`
    _PREFER_INT16_OPS: Set[str] = {"gather", "gather_along_axis", "gather_nd", "squeeze"}

    def __init__(self, op_selector=None):
        super().__init__(op_selector=op_selector)
        # Use variable instead of hard-coded "int16" because the target dtype could be uint16
        # depending on if the param is non-negative const and within uint16 range.
        self._target_dtype: str = "int16"

    @property
    def origin_dtype(self) -> str:
        return "int32"

    @property
    def target_dtype(self) -> str:
        return self._target_dtype

    @target_dtype.setter
    def target_dtype(self, target_dtype: str):
        if target_dtype not in {"int16", "uint16"}:
            raise ValueError("The target_dtype in add_int16_cast must be int16 or uint16")
        self._target_dtype = target_dtype

    def should_cast_parameter(self, op: Operation, param_name: str) -> bool:
        """
        Determine if a parameter should be cast or not.
        If should be cast, determine whether to use int16 or uint16.
        """
        _INT16_MAX = np.iinfo(np.int16).max
        _INT16_MIN = np.iinfo(np.int16).min
        _UINT16_MAX = np.iinfo(np.uint16).max
        _UINT16_MIN = np.iinfo(np.uint16).min

        input_var = op.inputs[param_name]
        if not input_var.is_tensor_or_scalar_of(dtype="int32"):
            return False

        input_op = input_var.op
        if input_op is not None and input_op.op_type == "const":
            if (
                input_op.outputs[0].val.min() >= _UINT16_MIN
                and input_op.outputs[0].val.max() <= _UINT16_MAX
            ):
                self._target_dtype = "uint16"
            elif (
                input_op.outputs[0].val.min() >= _INT16_MIN
                and input_op.outputs[0].val.max() <= _INT16_MAX
            ):
                self._target_dtype = "int16"
            else:
                return False

        # In `gather` and `gather_along_axis`, if the dim size of x is larger than int16
        # upperbound, the dynamic indices could overflow, so it shouldn't be cast.
        if op.op_type in {"gather", "gather_along_axis"} and param_name == "indices":
            if op.indices.val is None and op.x.shape is not None:
                dim_size = op.x.shape[op.axis.val]
                if not is_symbolic(dim_size) and dim_size > _INT16_MAX:
                    return False

        if not super().should_cast_parameter(op, param_name):
            return False

        return True

    def is_valid_op(self, op: Operation) -> bool:
        """Determines if op is valid for int16/uint16 casting."""
        return op.op_type in self._PREFER_INT16_OPS
