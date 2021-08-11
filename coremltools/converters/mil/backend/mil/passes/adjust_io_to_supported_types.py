# -*- coding: utf-8 -*-

#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as _mb
from coremltools.converters.mil.mil import types as _types
from coremltools.converters.mil.mil.ops import defs as _ops
from coremltools.converters.mil.mil.passes.pass_registry import register_pass as _register_pass

import warnings as _warnings

@_register_pass(namespace="mil_backend")
def adjust_io_to_supported_types(prog):
    """
    Converts all dTypes to types that are supported by the CoreML runtime.
    The runtime supports only fp16, fp32, int32, str, and bool variables.

    General rules:
        * Integer vars that are not 32 bit are replaced with int32 types.
        * All other types not in the list of runtime supported types are replaced with the fp32 dtype.
          No casts are inserted; the previous type is replaced. The assumption is that all remaining
          types are numerical and can be reasonably replaced with 32 bit float types.

    The "main" function has additional rules since its I/O is mapped to CoreML model I/O:
        * Fp16 I/O is replaced with fp32 I/O.
          Casts (fp32 input -> fp16) are inserted at the beginning of the program to preserve 16 bit inputs.
          Casts (fp16 -> fp32 output) are inserted at the end of the program to preserve 16 bit computations.

        * All non-integer I/O that is not fp32 is replaced with fp32 I/O.
          A cast (prev input type -> fp32) is inserted at the beginning of the program to preserve non-fp32 inputs.
          A cast (prev type -> fp32 out) is inserted at the end of the program to preserve non-fp32 computations.
          The assumption is that all remaining types are numerical and it is valid to cast them to/from fp32.

        * The only exception: Int64 outputs are allowed for the classifier op. This is to keep consistency with
          the CoreML API, which uses 64 bit integers to represent classifier labels.

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
    for name, func in prog.functions.items():
        _adjust_io_to_supported_types(func, name == "main")


__RUNTIME_SUPPORTED_TYPES = [_types.fp16, _types.fp32, _types.int32, _types.str, _types.bool]

#####
# Main Function
#####
def _adjust_var_dtype_helper(var, dtype):
    if (_types.is_scalar(var.sym_type)):
        var._sym_type = dtype
    else:
        var._sym_type = _types.tensor(dtype, var.sym_type.get_shape())

def _adjust_main_inputs(func):
    first_op = func.operations[0] if len(func.operations) > 0 else None
    for input_name, input_var in func.inputs.items():
       if (_types.is_tensor(input_var.sym_type) or _types.is_scalar(input_var.sym_type)) \
            and input_var.dtype != _types.fp32 \
            and input_var.dtype != _types.int32:
            input_dtype_str = _types.builtin_to_string(input_var.dtype)
            if _types.is_int(input_var.dtype):
                # Replace non-int32 input type with int32.
                _warnings.warn("Input" + input_var.name + " is of dType " + input_dtype_str +\
                               ". Only integer variables of bit width 32 are supported by the CoreML runtime. " +\
                               "This input will be assigned a dType of int32. " +\
                               "No cast will be inserted; the previous dtype will be replaced.")
                _adjust_var_dtype_helper(input_var, _types.int32)
            elif input_var.dtype == _types.fp64:
                # Replace float64 input type with fp32.
                _warnings.warn("Input" + input_var.name + " is of dtype fp64. 64 bit float inputs are " +\
                               "not supported by ML program models. This input will be assigned a dType " +\
                               "of fp32. No cast will be inserted; the previous dtype will be replaced.")
                _adjust_var_dtype_helper(input_var, _types.fp32)
            else:
                # This is some other dType. Change the type to fp32 and add a cast.
                # This is only a limitation of main--other functions do not represent CoreML model inputs
                # and do not have the same limitation on input types.
                _warnings.warn("Input" + input_var.name + " is of dType " + input_dtype_str + ". The " +\
                               "CoreML runtime does not support inputs with this dType (only fp32 and " +\
                               "int32 inputs are supported). This input will be assigned a dType of " +\
                               "fp32. A cast will be inserted at the beginning of the program to " +\
                               "convert the input to the originally defined dType.")
                with func:
                    casted_input_var = _mb.cast(x=input_var, dtype=input_dtype_str, before_op=first_op)
                    func.replace_uses_of_var_after_op(anchor_op=casted_input_var.op, old_var=input_var, new_var=casted_input_var)
                    _adjust_var_dtype_helper(input_var, _types.fp32)

def _adjust_main_outputs(func):
    new_outputs = []
    for output_var in func.outputs:
        output_type = output_var.sym_type
        if (_types.is_tensor(output_type) or _types.is_scalar(output_type)) \
            and output_var.dtype != _types.fp32 \
            and output_var.dtype != _types.int32:
            output_dtype_str = _types.builtin_to_string(output_var.dtype)
            _warnings.warn("Output" + output_var.name + " is of dType " + output_dtype_str + ". The " +\
                           "CoreML runtime does not support outputs with this dType (only int32 and " +\
                           "fp32 are supported for outputs). This output will be assigned a dType " +\
                           "of fp32. A cast will be inserted at the end of the program to convert" +\
                           "the original output dType to the dType supported by the CoreML runtime.")

            output_var_name = output_var.name
            output_var.set_name(output_var_name + "__pre__output__fp32__cast")
            # Convert the output to fp32, and add a cast.
            with func:
                output_var = _mb.cast(x=output_var, dtype="fp32")
                output_var.set_name(output_var_name)
        new_outputs.append(output_var)
    func.set_outputs(new_outputs)


#####
# General Functions and Blocks
#####
def _adjust_var(var):
    """
    Changes the dtype of the provided variable according
    to the rules outlined in the top level pass comment
    (see adjust_io_to_supported_types).
    """
    if (_types.is_tensor(var.sym_type) or _types.is_scalar(var.sym_type)) \
        and var.dtype not in __RUNTIME_SUPPORTED_TYPES:
        dtype_str = _types.builtin_to_string(var.dtype)
        if _types.is_int(var.dtype):
            # Replace non-int32 input type with int32.
            _warnings.warn("Input" + var.name + " is of dType " + dtype_str +\
                           ". Only integer variables of bit width 32 are supported by the CoreML runtime. " +\
                           "This input will be assigned a dType of int32. " +\
                           "No cast will be inserted; the previous dtype will be replaced.")
            _adjust_var_dtype_helper(var, _types.int32)
        else:
            # This is some other unsupported dType. Change the input type to fp32.
            _warnings.warn("Var " + var.name + " is of dType " + dtype_str + ". The CoreML runtime " +\
                           "does not support this dType (only fp16, fp32, bool, and int32 are supported). " +\
                           "This input will be assigned a dType of fp32. No cast will be inserted; " +\
                           "the previous dtype will be replaced.")
            _adjust_var_dtype_helper(var, _types.fp32)


def _adjust_func_inputs(func):
    for input_name, input_var in func.inputs.items():
       _adjust_var(input_var)


def _adjust_block_inputs(block):
    for input_var in block.inputs:
       _adjust_var(input_var)


def _adjust_ops(block):
    len_block = len(block.operations)
    i = 0
    while i < len_block:
        op = block.operations[i]

        # Classifier is a special exception to this rule. It can output 64 bit integer labels.
        # Classifier should be inserted after running this pass.
        if op.op_type == "classify":
            raise ValueError("ML Program backend pass adjust_to_supported_types does not support programs" +\
                             " that have already added a classify op.")

        for subblock in op.blocks:
            _adjust_block_inputs(subblock)
            _adjust_ops(subblock)

        for var in op.outputs:
            _adjust_var(var)

        # Cast ops have a param (dtype) that should match the output dtype.
        # If the output dtype or input dtype was previously adjusted,
        # the cast op must change or be removed in kind.
        if op.op_type == "cast":
            output_type_str = _types.builtin_to_string(op.outputs[0].dtype)
            if op.outputs[0].dtype == op.x.dtype:
                # The type of the input or output of this cast op was changed per the rules
                # defined in the top level comment for adjust_io_to_supported_types.
                #
                # That changed output type is the same type as the input to the cast
                # op. Therefore, regardless of whether the user created this cast or
                # not, it is now redundant (noop), and should be removed.
                #
                # The removal isn't covered by the main cast
                # optimization pass since that pass runs before this pass.
                block.replace_uses_of_var_after_op(
                    anchor_op=op, old_var=op.outputs[0], new_var=op.x
                )
                block.remove_ops([op])
                len_block = len(block.operations)
                i -= 1
            elif output_type_str != op.dtype.val:
                # The type of the output of this cast op was changed per the rules
                # defined in the top level comment for adjust_io_to_supported_types.
                #
                # This cast is meaningful, and the "dtype" param now differs from the output
                # type. Replace the dtype cast with a new cast op with a matching dtype param.
                with block:
                    new_cast_out = _mb.cast(x=op.x, dtype=output_type_str, before_op=op)
                    block.replace_uses_of_var_after_op(
                        anchor_op=op, old_var=op.outputs[0], new_var=new_cast_out
                    )
                block.remove_ops([op])
                len_block = len(block.operations)
        i = i + 1
    return block

#####
# The Pass
#####
def _adjust_io_to_supported_types(func, is_main):
    if is_main:
        _adjust_main_inputs(func)
        _adjust_ops(func)
        _adjust_main_outputs(func)
    else:
        _adjust_func_inputs(func)
        _adjust_ops(func)
