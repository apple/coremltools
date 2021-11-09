#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict
import re
import warnings

from coremltools.converters.mil.mil import Function


class NameSanitizer(object):

    def __init__(self, prefix=None):
        # to hold all names encountered,
        # to make sure that all new names are unique
        self.all_names = set()
        self.prefix = "_" if prefix is None else prefix

    def sanitize_name(self, name):
        """
        Sanitize the input string and return it back.
        Input string should be of the format: [a-zA-Z_][a-zA-Z0-9_]*

        If it is not, then it is sanitized in the following manner:
        - first, any character that is not [a-zA-Z0-9_] is replaced with "_"
        - if the starting character is not [a-zA-Z_], it is prefixed with self.prefix
        - the resulting string must be unique. If it has been encountered before,
          it is appended by "_0" or "_1" and so on, until it becomes unique.

        :name: str
            current name

        :return: str
            updated name. Returns the same string, if sanitization not required.
        """

        # replace any character that is not [a-zA-Z0-9_] with an underscore
        new_name = re.sub("[^a-zA-Z0-9_]", "_", name)

        # now check if the name starts with anything but [A-Za-z_]
        # if so, then add the prefix
        if re.match("[^a-zA-Z_]", new_name):
            new_name = self.prefix + new_name

        if new_name == name:
            # return if nothing has changed
            self.all_names.add(name)
            return name
        else:
            # name has changed
            # make sure it is unique, then return
            if new_name in self.all_names:
                idx = 0
                new_name += "_" + str(idx)
                while new_name in self.all_names:
                    idx += 1
                    new_name += "_" + str(idx)
            # now we have a unique name
            self.all_names.add(new_name)
            return new_name



def sanitize_block(block,
                   sanitizer_vars,
                   sanitizer_ops,
                   main_input_types=None,
                   sanitize_model_inputs_outputs_only=False):
    '''
    Sanitize the vars and op names inside the block to adhere to the format [a-zA-Z_][a-zA-Z0-9_]*
    '''

    if sanitize_model_inputs_outputs_only:
        _sanitize_block_input_vars(block, sanitizer_vars, main_input_types, sanitize_main_input_only=True)
        _sanitize_main_outputs_only(block, sanitizer_vars, sanitizer_ops)
    else:
        _sanitize_block_input_vars(block, sanitizer_vars, main_input_types)
        _sanitize_output_vars_and_nested_blocks(block, sanitizer_vars, sanitizer_ops)
        _sanitize_op_names(block, sanitizer_ops)


def _sanitize_block_input_vars(block, sanitizer_vars, main_input_types,
                               sanitize_main_input_only=False):

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
                if var.name in block.placeholder_inputs:
                    block.placeholder_inputs[new_name] = block.placeholder_inputs.pop(var.name)
                    block.placeholder_inputs[new_name].set_name(new_name)
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
    elif not sanitize_main_input_only:
        # in this case block is not the "main" function
        # in this case block.inputs is a list of input vars of the block
        for var in block.inputs:
            new_name = sanitizer_vars.sanitize_name(var.name)
            if new_name != var.name:
                var.set_name(new_name)


def _sanitize_var_names(var, sanitizer_vars, emit_warning=False):
    new_name = sanitizer_vars.sanitize_name(var.name)
    if new_name != var.name:
        if emit_warning:
            msg = "Output, '{}', of the source model, has been renamed to '{}' in the Core ML model."
            warnings.warn(msg.format(var.name, new_name))
        var.set_name(new_name)

def _sanitize_output_vars_and_nested_blocks(block, sanitizer_vars, sanitizer_ops):
    for op in list(block.operations):
        for b in op.blocks:
            sanitize_block(b, sanitizer_vars, sanitizer_ops)

        for var in op.outputs:
            if isinstance(block, Function) and var in block.outputs:
                _sanitize_var_names(var, sanitizer_vars, emit_warning=True)
            else:
                _sanitize_var_names(var, sanitizer_vars)

def _sanitize_main_outputs_only(block, sanitizer_vars, sanitizer_ops):
    for op in list(block.operations):
        for var in op.outputs:
            if isinstance(block, Function) and var in block.outputs:
                _sanitize_var_names(var, sanitizer_vars, emit_warning=True)

def _sanitize_op_names(block, sanitizer_ops):
    # iterate over all the ops and sanitize the op names
    for op in list(block.operations):
        if op.name is not None:
            op.name = sanitizer_ops.sanitize_name(op.name)
