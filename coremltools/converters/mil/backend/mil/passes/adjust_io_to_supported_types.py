#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Set

from coremltools import _logger as logger
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as target
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types as types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="mil_backend")
class adjust_io_to_supported_types(AbstractGraphPass):
    """
    Converts all dtypes to types that are supported by the Core ML runtime.
    The runtime supports fp16, fp32, int16, uint16, int32, str, and bool variables.

    General rules:
        * Integer vars with unsupported types are replaced with int32 types.
        * All other types not in the list of runtime supported types are replaced with the fp32 dtype.
          No casts are inserted; the previous type is replaced. The assumption is that all remaining
          types are numerical and can be reasonably replaced with 32 bit float types.

    The "main" function has additional rules since its I/O is mapped to Core ML model I/O:
        * if function.opset_version <  coremltools.target.iOS16, then:
            * Fp16 I/O is replaced with fp32 I/O.
                Casts (fp32 input -> fp16) are inserted at the beginning of the program to preserve 16 bit inputs.
                Casts (fp16 -> fp32 output) are inserted at the end of the program to preserve 16 bit computations.

        * All non-integer I/O that is not fp32 is replaced with fp32 I/O.
          A cast (prev input type -> fp32) is inserted at the beginning of the program to preserve non-fp32 inputs.
          A cast (prev type -> fp32 out) is inserted at the end of the program to preserve non-fp32 computations.
          The assumption is that all remaining types are numerical and it is valid to cast them to/from fp32.

        * The only exception: Int64 outputs are allowed for the classifier op. This is to keep consistency with
          the Core ML API, which uses 64 bit integers to represent classifier labels.

    ------

    func main(bool x, int32 y, fp32 z) {
        bool  out = logical_not(x)
    } -> (out, y, z)

    becomes

    func main(fp32 x, int32 y, fp32 z) {
       bool  x_casted = cast(x)
       bool  out__pre__output__fp32__cast = logical_not(x_casted)
       fp32  out = cast(out__pre__output__fp32__cast)
    } -> (out, y, z)

    ------

    func not_main(bool x, int32 y, fp32 z) {
        bool  out = logical_not(x)
    } -> (out, y, z)

    is unchanged.
    """

    def apply(self, prog):
        for name, func in prog.functions.items():
            is_main_funtion = name == "main"
            _adjust_io_to_supported_types(func, is_main_funtion)


def _adjust_var_dtype_helper(var, dtype):
    if types.is_scalar(var.sym_type):
        var._sym_type = dtype
    else:
        var._sym_type = types.tensor(dtype, var.sym_type.get_shape())


def _get_io_supported_types(opset_version: target) -> Set[type]:
    """Get Core ML I/O supported data types based on opset version."""
    supported_types = {types.fp32, types.int32}
    if opset_version is not None and opset_version >= target.iOS16:
        supported_types.add(types.fp16)
    return supported_types


def _get_runtime_supported_types(opset_version: target) -> Set[type]:
    """Get Core ML Runtime supported data types based on opset version."""
    supported_types = {types.fp16, types.fp32, types.int32, types.str, types.bool}
    if opset_version >= target.iOS17:
        supported_types.update({types.int8, types.uint8, types.int16, types.uint16})
    return supported_types


@block_context_manager
def _adjust_main_inputs(func):
    """
    Adjust the inputs in main func.

    If the input's dtype is not in Core ML I/O supported types, we do following steps:
        1. Change the input's dtype to int32 or fp32 based on original dtype.
        2. If the original dtype is supported in Core ML Runtime, we insert a cast op to cast the
           input from the changed dtype to the original dtype.
    """
    _IO_SUPPORTED_TYPES = _get_io_supported_types(func.opset_version)
    _RUNTIME_SUPPORTED_TYPES = _get_runtime_supported_types(func.opset_version)

    for input_name, input_var in func.inputs.items():
        if (
            types.is_tensor(input_var.sym_type) or types.is_scalar(input_var.sym_type)
        ) and input_var.dtype not in _IO_SUPPORTED_TYPES:
            input_dtype_str = types.builtin_to_string(input_var.dtype)
            convert_to_dtype = types.int32 if types.is_int(input_var.dtype) else types.fp32
            convert_to_dtype_str = types.builtin_to_string(convert_to_dtype)
            should_insert_cast = input_var.dtype in _RUNTIME_SUPPORTED_TYPES
            _adjust_var_dtype_helper(input_var, convert_to_dtype)
            logger.warning(
                f"\nInput '{input_var.name}' is of dtype {input_dtype_str}. The Core ML I/O does "
                f"not support this dtype (supported dtypes are: {_IO_SUPPORTED_TYPES}). Consider "
                f"setting `minimum_deployment_target` to a higher IOS version for more supported "
                f"dtypes. This input is changed to {convert_to_dtype_str}.\n"
            )

            if not should_insert_cast:
                logger.warning(
                    f"The original input dtype {input_dtype_str} is not supported in "
                    f"Core ML Runtime (supported dtypes are: {_RUNTIME_SUPPORTED_TYPES}). Consider "
                    f"setting `minimum_deployment_target` to a higher IOS version for more "
                    f"supported dtypes. We just changed the dtype and won't insert any cast op."
                )
                continue

            logger.warning(
                f"Trying to insert a cast op at the beginning of the program to convert "
                f"the input to the originally defined dtype ({input_dtype_str}).\n"
            )
            try:
                first_op = func.operations[0] if len(func.operations) > 0 else None
                casted_input_var = mb.cast(x=input_var, dtype=input_dtype_str, before_op=first_op)
                # Use force replace as the `input_var.dtype` could be not subtype of the
                # `convert_to_dtype`. For example, int16 cast to int32. As it's only for input
                # dtype cast, this replace should be safe.
                func.replace_uses_of_var_after_op(
                    anchor_op=casted_input_var.op,
                    old_var=input_var,
                    new_var=casted_input_var,
                    force_replace=True,
                    no_check_var_types=True,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to insert the cast op.\n{e}\nThe dtype of the input "
                    f"'{input_var.name}' is changed to {convert_to_dtype_str} without "
                    f"inserting any cast op."
                )


@block_context_manager
def _adjust_main_outputs(func):
    """Adjust the outputs in the main func to make sure they have Core ML I/O supported types."""
    _IO_SUPPORTED_TYPES = _get_io_supported_types(func.opset_version)

    new_outputs = []
    for output_var in func.outputs:
        output_type = output_var.sym_type
        if (
            types.is_tensor(output_type) or types.is_scalar(output_type)
        ) and output_var.dtype not in _IO_SUPPORTED_TYPES:
            output_dtype_str = types.builtin_to_string(output_var.dtype)
            target_dtype = "int32" if types.is_int(output_var.dtype) else "fp32"
            logger.warning(
                f"\nOutput '{output_var.name}' is of dtype {output_dtype_str}. The "
                f"Core ML runtime does not support outputs with this dtype (supported "
                f"dtypes are: {_IO_SUPPORTED_TYPES}). This output will changed to "
                f"{target_dtype} by adding a cast op at the end of the program.\n"
            )
            if output_var.dtype == types.fp16:
                logger.warning(
                    "fp16 dtype output is supported if function.opset_version is chosen to be at "
                    "least iOS16/macOS13.\n"
                )

            output_var_name = output_var.name
            output_var.set_name(f"{output_var_name}__pre__output__{target_dtype}__cast")
            output_var = mb.cast(x=output_var, dtype=target_dtype)
            output_var.set_name(output_var_name)
        new_outputs.append(output_var)
    func.set_outputs(new_outputs)


def _adjust_func_inputs(func):
    """
    Changes the dtype of the provided variable according
    to the rules outlined in the top level pass comment
    (see adjust_io_to_supported_types).
    """
    _RUNTIME_SUPPORTED_TYPES = _get_runtime_supported_types(func.opset_version)

    for input_name, input_var in func.inputs.items():
        if (
            types.is_tensor(input_var.sym_type) or types.is_scalar(input_var.sym_type)
        ) and input_var.dtype not in _RUNTIME_SUPPORTED_TYPES:
            dtype_str = types.builtin_to_string(input_var.dtype)
            convert_to_dtype = types.int32 if types.is_int(input_var.dtype) else types.fp32
            convert_to_dtype_str = types.builtin_to_string(convert_to_dtype)
            _adjust_var_dtype_helper(input_var, convert_to_dtype)
            logger.warning(
                f"Input '{input_var.name}' is of dtype {dtype_str}, which is not"
                f"supported by the Core ML runtime. This input will be changed to "
                f"{convert_to_dtype_str}. No cast will be inserted."
            )


def _adjust_io_to_supported_types(func, is_main):
    if is_main:
        _adjust_main_inputs(func)
        _adjust_main_outputs(func)
    else:
        _adjust_func_inputs(func)
