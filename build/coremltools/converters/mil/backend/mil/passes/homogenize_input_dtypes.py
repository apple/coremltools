# -*- coding: utf-8 -*-

#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.ops.defs import elementwise_binary, matmul
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types import promote_dtypes, builtin_to_string
from coremltools.converters.mil.mil import Builder as mb

_SUPPORTED_OPS = {
    # Mapping from op_class --> list of those params which needs to be of the same dtype
    elementwise_binary: ["x", "y"],
    matmul: ["x", "y"],
}


def _get_input_params(op):
    for op_class, params in _SUPPORTED_OPS.items():
        if isinstance(op, op_class):
            return params
    return None


def _of_same_dtype(dtype1, dtype2):
    return (dtype1 is dtype2) or (builtin_to_string(dtype1) == builtin_to_string(dtype2))


def _promoted_var(op, var, promoted_dtype):
    x = mb.cast(
        x=var, dtype=builtin_to_string(promoted_dtype), name=var.name + "_promoted", before_op=op
    )
    return x


def _homogenize_input_dtypes_block(block):
    for op in list(block.operations):
        for b in op.blocks:
            _homogenize_input_dtypes_block(b)

        params = _get_input_params(op)
        if params:
            has_mixed_dtypes = False
            input_vars = [getattr(op, v) for v in params]
            promoted_dtype = promote_dtypes([var.dtype for var in input_vars])

            for i,var in enumerate(input_vars):
                if not _of_same_dtype(var.dtype, promoted_dtype):
                    has_mixed_dtypes = True
                    with block:
                        input_vars[i] = _promoted_var(op, var, promoted_dtype)

            if has_mixed_dtypes:
                new_inputs = dict(zip(params, input_vars))
                new_inputs.update({"name": op.name, "before_op": op})
                new_inputs.update(
                    {k: v for k, v in op.inputs.items() if k not in new_inputs}
                )
                with block:
                    new_output = getattr(mb, op.op_type)(**new_inputs)
                    op.enclosing_block.replace_uses_of_var_after_op(
                        anchor_op=op, old_var=op.outputs[0], new_var=new_output, no_check_var_types=True,
                        # Has to set no_check_var_types=True because Matmul PyMIL type inference doesn't enforce same dtypes for x & y
                        # but for output dtype assumes them to be same and chooses one of the two.
                    )
                    block.remove_ops([op])


@register_pass(namespace="mil_backend")
def homogenize_input_dtypes(prog):
    """
    If inputs to an op, doesn't have same dtypes for some parameters, explicit cast operations are injected
    to ensure inputs to that op have same promoted dtype.

    - Only ops specified in dict _SUPPORTED_OPS as its keys, can be affected by this pass
    - Only the named inputs of ops specified in dict _SUPPORTED_OPS as values, are promoted to match dtypes
    """
    for f_name, f in prog.functions.items():
        _homogenize_input_dtypes_block(f)

        for op in f.operations:
            op.type_value_inference(overwrite_output=True)
