# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from ..helper import NameSanitizer
import warnings
from collections import OrderedDict
from coremltools.converters.mil.mil import Function


@register_pass(namespace="mil_backend")
def sanitize_name_strings(prog):
    """
    Sanitize the names of vars and ops to make sure
    that they are of the format as described in the NameSanitizer class, i.e.
    of the format [a-zA-Z_][a-zA-Z0-9_]*
    """

    sanitizer_vars = NameSanitizer(prefix="var_")
    sanitizer_ops = NameSanitizer(prefix="op_")

    for _, f in prog.functions.items():
        _sanitize_block(f, sanitizer_vars, sanitizer_ops, prog.main_input_types)


def _sanitize_block(block, sanitizer_vars, sanitizer_ops, main_input_types=None):

    # iterate over all the block input vars and sanitize the names
    if isinstance(block, Function):
        # this is the "main" block
        # block.inputs is a dict from input names to input vars
        # iterate over the input vars of the main program and sanitize their names
        new_input_dict = OrderedDict()
        input_name_updated = False
        for input_name, var in block.inputs.items():
            msg = "Main block's input name, '{}', is different from its corresponding var's name, '{}'."
            assert input_name == var.name, msg.format(input_name, var.name)
            new_name = sanitizer_vars.sanitize_name(var.name)
            new_input_dict[new_name] = var
            if new_name != var.name:
                msg = "Input, '{}', of the source model, has been renamed to '{}' in the Core ML model."
                warnings.warn(msg.format(var.name, new_name))
                var.set_name(new_name)
                input_name_updated = True
                if main_input_types is not None:
                    # update prog's main_input_types, since we are updating the name of a model input here
                    for i in range(len(main_input_types)):
                        if main_input_types[i].name == input_name:
                            main_input_types[i].name = new_name
                            break
        if input_name_updated:
            block._input_dict = new_input_dict
    else:
        # in this case block is not the "main" function
        # in this case block.inputs is a list of input vars of the block
        for var in block.inputs:
            new_name = sanitizer_vars.sanitize_name(var.name)
            if new_name != var.name:
                var.set_name(new_name)

    # iterate over all the ops and sanitize the names of the output variables
    for op in list(block.operations):
        for b in op.blocks:
            _sanitize_block(b, sanitizer_vars, sanitizer_ops)

        for var in op.outputs:
            new_name = sanitizer_vars.sanitize_name(var.name)
            if new_name != var.name:
                if isinstance(block, Function) and var in block.outputs:
                    msg = "Output, '{}', of the source model, has been renamed to '{}' in the Core ML model."
                    warnings.warn(msg.format(var.name, new_name))
                var.set_name(new_name)

    # iterate over all the ops and sanitize the op names
    for op in list(block.operations):
        if op.name is not None:
            op.name = sanitizer_ops.sanitize_name(op.name)

