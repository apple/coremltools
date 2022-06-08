#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging

from coremltools.converters.mil.mil import Builder as mb, types as types
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as target


@register_pass(namespace="mil_backend")
class adjust_io_to_supported_types(AbstractGraphPass):
    """
    Converts all dtypes to types that are supported by the CoreML runtime.
    The runtime supports only fp16, fp32, int32, str, and bool variables.

    General rules:
        * Integer vars that are not 32 bit are replaced with int32 types.
        * All other types not in the list of runtime supported types are replaced with the fp32 dtype.
          No casts are inserted; the previous type is replaced. The assumption is that all remaining
          types are numerical and can be reasonably replaced with 32 bit float types.

    The "main" function has additional rules since its I/O is mapped to CoreML model I/O:
        * if minimum_deployment_target <  coremltools.target.iOS16, then:
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

    def apply(self, prog):
        for name, func in prog.functions.items():
            is_main_funtion = name == "main"
            _adjust_io_to_supported_types(func, is_main_funtion, self.minimun_deployment_target)

__RUNTIME_SUPPORTED_TYPES = [types.fp16, types.fp32, types.int32, types.str, types.bool]

#####
# Main Function
#####
def _adjust_var_dtype_helper(var, dtype):
    if (types.is_scalar(var.sym_type)):
        var._sym_type = dtype
    else:
        var._sym_type = types.tensor(dtype, var.sym_type.get_shape())

def _adjust_main_inputs(func, min_deployment_target):
    first_op = func.operations[0] if len(func.operations) > 0 else None
    for input_name, input_var in func.inputs.items():
       if (types.is_tensor(input_var.sym_type) or types.is_scalar(input_var.sym_type)) \
            and input_var.dtype != types.fp32 \
            and input_var.dtype != types.int32:
            input_dtype_str = types.builtin_to_string(input_var.dtype)
            if types.is_int(input_var.dtype):
                # Replace non-int32 input type with int32.
                logging.warning("Input" + input_var.name + " is of dtype " + input_dtype_str +\
                               ". Only integer variables of bit width 32 are supported by the CoreML runtime. " +\
                               "This input will be assigned a dtype of int32. " +\
                               "No cast will be inserted; the previous dtype will be replaced.")
                _adjust_var_dtype_helper(input_var, types.int32)
            elif input_var.dtype == types.fp64:
                # Replace float64 input type with fp32.
                logging.warning("Input '" + input_var.name + "' is of dtype fp64. 64 bit float inputs are " +\
                               "not supported by ML program models. This input will be assigned a dtype " +\
                               "of fp32. No cast will be inserted; the previous dtype will be replaced.")
                _adjust_var_dtype_helper(input_var, types.fp32)
            elif input_var.dtype == types.fp16 \
                 and min_deployment_target.value>= target.iOS16.value:
                pass # do nothing, since fp16 is a valid input type for CoreML
            else:
                # This is some other dtype. Change the type to fp32 and add a cast.
                # This is only a limitation of main--other functions do not represent CoreML model inputs
                # and do not have the same limitation on input types.
                supported_dtypes = "{int32, fp32, fp64}" if min_deployment_target < target.iOS16 else \
                                    "{int32, fp16, fp32, fp64}"
                msg = "\nInput '{}' is of dtype {}. The " +\
                               "CoreML runtime does not support inputs with this dtype " +\
                               "(supported dtypes are: {}). This input will be assigned a dtype of " +\
                               "fp32. A cast will be inserted at the beginning of the program to " +\
                               "convert the input to the originally defined dtype.\n"
                if input_var.dtype == types.fp16:
                    msg += "fp16 dtype input is supported if the minimum_deployment_target is chosen to be at least " \
                           "iOS16/macOS13.\n"
                logging.warning(msg.format(
                    input_var.name,
                    input_dtype_str,
                    supported_dtypes))

                with func:
                    casted_input_var = mb.cast(x=input_var, dtype=input_dtype_str, before_op=first_op)
                    func.replace_uses_of_var_after_op(anchor_op=casted_input_var.op, old_var=input_var, new_var=casted_input_var)
                    _adjust_var_dtype_helper(input_var, types.fp32)

def _adjust_main_outputs(func, min_deployment_target):
    new_outputs = []
    for output_var in func.outputs:
        output_type = output_var.sym_type
        if (types.is_tensor(output_type) or types.is_scalar(output_type)) \
            and output_var.dtype != types.fp32 \
            and output_var.dtype != types.int32 \
            and (min_deployment_target < target.iOS16 or output_var.dtype != types.fp16):
            # since fp16 is a valid output type for coreml from ios16 spec onwards, no need to cast
            output_dtype_str = types.builtin_to_string(output_var.dtype)
            supported_dtypes = "{int32, fp32, fp64}" if min_deployment_target < target.iOS16 else \
                                "{int32, fp16, fp32, fp64}"
            msg = "\nOutput '{}' is of dtype {}. The " +\
                           "CoreML runtime does not support outputs with this dtype " +\
                           "(supported dtypes are: {}). This output will be assigned a dtype " +\
                           "of fp32. A cast will be inserted at the end of the program to convert" +\
                           "the original output dtype to the dtype supported by the CoreML runtime.\n"
            if output_var.dtype == types.fp16:
                msg += "fp16 dtype output is supported if the minimum_deployment_target is chosen to be at least " \
                       "iOS16/macOS13.\n"
            logging.warning(msg.format(
                               output_var.name,
                               output_dtype_str,
                               supported_dtypes,
                           ))

            output_var_name = output_var.name
            output_var.set_name(output_var_name + "__pre__output__fp32__cast")
            # Convert the output to fp32, and add a cast.
            with func:
                output_var = mb.cast(x=output_var, dtype="fp32")
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
    if (types.is_tensor(var.sym_type) or types.is_scalar(var.sym_type)) \
        and var.dtype not in __RUNTIME_SUPPORTED_TYPES:
        dtype_str = types.builtin_to_string(var.dtype)
        if types.is_int(var.dtype):
            # Replace non-int32 input type with int32.
            logging.warning("Input '" + var.name + "' is of dtype " + dtype_str +\
                           ". Only integer variables of bit width 32 are supported by the CoreML runtime. " +\
                           "This input will be assigned a dtype of int32. " +\
                           "No cast will be inserted; the previous dtype will be replaced.")
            _adjust_var_dtype_helper(var, types.int32)
        else:
            # This is some other unsupported dtype. Change the input type to fp32.
            logging.warning("Var " + var.name + " is of dtype " + dtype_str + ". The CoreML runtime " +\
                           "does not support this dtype (only fp16, fp32, bool, and int32 are supported). " +\
                           "This input will be assigned a dtype of fp32. No cast will be inserted; " +\
                           "the previous dtype will be replaced.")
            _adjust_var_dtype_helper(var, types.fp32)


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
            output_type_str = types.builtin_to_string(op.outputs[0].dtype)
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
                    new_cast_out = mb.cast(x=op.x, dtype=output_type_str, before_op=op)
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
def _adjust_io_to_supported_types(func, is_main, min_deployment_target):
    if is_main:
        _adjust_main_inputs(func, min_deployment_target)
        _adjust_ops(func)
        _adjust_main_outputs(func, min_deployment_target)
    else:
        _adjust_func_inputs(func)
        _adjust_ops(func)
