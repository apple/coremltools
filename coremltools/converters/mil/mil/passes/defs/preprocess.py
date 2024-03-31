#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clausefrom

import re
import warnings
from collections import OrderedDict

from coremltools import _logger as logger
from coremltools.converters.mil.input_types import EnumeratedShapes, ImageType, Shape
from coremltools.converters.mil.mil import Block
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="common")
class image_input_preprocess(AbstractGraphPass):
    """
    Plug in to ``transpose`` image input in NHWC format to NCHW format.

    Follow these steps:

    1. Check whether there are any inputs that the users specify as ImageType.
    2. Check the channel's dimension for all inputs that are ImageType.

        a) ``channel_first == True``
           We do not modify this input, since ``channel_first`` is the intended
           behaviour for feeding images for optimal performance.
        b) ``channel_first == False``
           We convert the input into a "channel_first" input, and plug in a
           ``transpose`` for the input to maintain the remaining graph's dimensionality.
    """

    def apply(self, prog):
        for f_name, f in prog.functions.items():
            if f_name == "main":
                # We need to make sure main exist and start here.
                self._image_input_preprocess(prog)

    @staticmethod
    def _image_input_preprocess(prog):
        def _transform_to_channel_first(shape):
            if isinstance(shape, tuple):
                shape = list(shape)
                return tuple(shape[:-3] + [shape[-1]] + shape[-3:-1])
            else:
                return shape[:-3] + [shape[-1]] + shape[-3:-1]

        main_input_types = list(prog.functions["main"].input_types)
        for idx, input_type in enumerate(main_input_types):
            if isinstance(input_type, ImageType) and not input_type.channel_first:
                name = input_type.name
                # Build new ImageType to change data layout
                if isinstance(input_type.shape, Shape):
                    new_shape = _transform_to_channel_first(input_type.shape.shape)
                    new_default = _transform_to_channel_first(input_type.shape.default)
                    shape_type = Shape(shape=new_shape, default=new_default)
                elif isinstance(input_type.shape, EnumeratedShapes):
                    shape_list = []
                    for shape in input_type.shape.shapes:
                        if isinstance(shape, Shape):
                            shape_list.append(_transform_to_channel_first(shape.shape))
                        else:
                            shape_list.append(_transform_to_channel_first(shape))
                    shape_type = EnumeratedShapes(
                        shapes=shape_list,
                        default=_transform_to_channel_first(input_type.shape.default),
                    )
                new_image_type = ImageType(
                    name=name,
                    shape=shape_type,
                    bias=input_type.bias,
                    scale=input_type.scale,
                    color_layout=input_type.color_layout,
                    channel_first=True,
                )
                main_input_types[idx] = new_image_type

                # Reconstruct Placeholder of Function inputs.
                placeholder_op = prog.functions["main"].placeholder_inputs[name]
                old_var = placeholder_op.outputs[0]
                nchw_shape = _transform_to_channel_first(placeholder_op.sym_shape)
                placeholder_op.__init__(
                    nchw_shape, dtype=placeholder_op.dtype, name=placeholder_op.name
                )

                # Update Function input var
                prog.functions["main"]._input_dict[name] = placeholder_op.outputs[0]

                # Add transpose into graph (Transpose from NCHW back to NHWC)
                curr_block = prog.functions["main"]
                curr_var = prog.functions["main"].inputs[name]

                perm = list(range(curr_var.rank))
                perm = perm[:-3] + [perm[-2], perm[-1], perm[-3]]
                with curr_block:
                    new_input = mb.transpose(
                        x=curr_var,
                        perm=perm,
                        before_op=prog.functions["main"].operations[0],
                        name=curr_var.name + "__transpose_from_nchw__",
                    )
                curr_block.replace_uses_of_var_after_op(
                    anchor_op=None, old_var=old_var, new_var=new_input
                )
        prog.functions["main"].input_types = tuple(main_input_types)


class NameSanitizer:
    def __init__(self, prefix=None):
        # to hold all names encountered,
        # to make sure that all new names are unique
        self.all_names = set()
        self.prefix = "_" if prefix is None else prefix

    @staticmethod
    def _replace_invalid_char_with_underscore(name):
        return re.sub("[^a-zA-Z0-9_]", "_", name)

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
        new_name = self._replace_invalid_char_with_underscore(name)

        # now check if the name starts with anything but [A-Za-z_]
        # if so, then add the prefix
        if re.match("[^a-zA-Z_]", new_name):
            new_name = self.prefix + new_name

        reserved_names = [
            "any",
            "bool",
            "program",
            "func",
            "tensor",
            "list",
            "dict",
            "tuple",
            "true",
            "false",
            "string",
            "bf16",
            "fp16",
            "fp32",
            "fp64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        ]
        if new_name in reserved_names:
            new_name += "_workaround"

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

    @staticmethod
    def sanitize_block(
        block,
        sanitizer_vars,
        sanitizer_ops,
        main_input_types=None,
        sanitize_model_inputs_outputs_only=False,
    ):
        """
        Sanitize the vars and op names inside the block to adhere to the format [a-zA-Z_][a-zA-Z0-9_]*
        """

        if sanitize_model_inputs_outputs_only:
            NameSanitizer._sanitize_block_input_vars(
                block, sanitizer_vars, main_input_types, sanitize_main_input_only=True
            )
            NameSanitizer._sanitize_main_outputs_only(block, sanitizer_vars)
        else:
            NameSanitizer._sanitize_block_input_vars(block, sanitizer_vars, main_input_types)
            NameSanitizer._sanitize_output_vars_and_nested_blocks(
                block, sanitizer_vars, sanitizer_ops
            )
            NameSanitizer._sanitize_op_names(block, sanitizer_ops)

    @staticmethod
    def _sanitize_block_input_vars(
        block, sanitizer_vars, main_input_types, sanitize_main_input_only=False
    ):

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

    @staticmethod
    def _sanitize_var_names(var, sanitizer_vars, emit_warning=False):
        new_name = sanitizer_vars.sanitize_name(var.name)
        if new_name != var.name:
            if emit_warning:
                msg = "Output, '{}', of the source model, has been renamed to '{}' in the Core ML model."
                warnings.warn(msg.format(var.name, new_name))
            var.set_name(new_name)

    @staticmethod
    def _sanitize_op_names(block, sanitizer_ops):
        # iterate over all the ops and sanitize the op names
        for op in list(block.operations):
            if op.name is not None:
                op.name = sanitizer_ops.sanitize_name(op.name)

    @staticmethod
    def _sanitize_output_vars_and_nested_blocks(block, sanitizer_vars, sanitizer_ops):
        for op in list(block.operations):
            for b in op.blocks:
                NameSanitizer.sanitize_block(b, sanitizer_vars, sanitizer_ops)

            for var in op.outputs:
                if isinstance(block, Function) and var in block.outputs:
                    NameSanitizer._sanitize_var_names(var, sanitizer_vars, emit_warning=True)
                else:
                    NameSanitizer._sanitize_var_names(var, sanitizer_vars)

    @staticmethod
    def _sanitize_main_outputs_only(block, sanitizer_vars):
        for op in list(block.operations):
            for var in op.outputs:
                if isinstance(block, Function) and var in block.outputs:
                    NameSanitizer._sanitize_var_names(var, sanitizer_vars, emit_warning=True)


@register_pass(namespace="common")
class sanitize_input_output_names(AbstractGraphPass):
    """
    Sanitize the names of model input and output vars to make sure
    that they are of the format as described in the NameSanitizer class; that is,
    of the format ``[a-zA-Z_][a-zA-Z0-9_]*``.
    """

    def apply(self, prog):
        sanitizer_vars = NameSanitizer(prefix="var_")
        sanitizer_ops = NameSanitizer(prefix="op_")

        # sanitize the input/output of the main block
        NameSanitizer.sanitize_block(
            prog.functions["main"],
            sanitizer_vars,
            sanitizer_ops,
            prog.functions["main"].input_types,
            sanitize_model_inputs_outputs_only=True,
        )


# TODO: rdar://122845072 ([Infra] Refactor the transform_function_signatures, adjust_io_to_supported_types and update_output_dtypes using a shared graph pass)
@register_pass(namespace="common")
class update_output_dtypes(AbstractGraphPass):
    """
    Update the dtypes of output vars of each function block to match the dtypes
    provided in ``function.output_types``. The output types for the main function
    is populated by the ``outputs`` argument provided by the user in the ``coremltools.convert()`` API.
    This graph pass assumes that the list of outputs in ``function.output_types`` (if not ``None``),
    are in the same order as the output vars.
    """

    @block_context_manager
    def adjust_function_output_types(self, func: Function) -> None:
        """
        Adjust output dtypes for a pymil function.
        """
        user_provided_output_types = func.output_types
        output_vars = func.outputs
        input_vars = list(func.inputs.values())
        if user_provided_output_types is None or len(user_provided_output_types) == 0:
            return
        if len(output_vars) != len(user_provided_output_types):
            msg = (
                "Number of outputs provided by the user, which is {}, "
                "does not match the number of outputs generated by the model, which is {}"
            )
            raise ValueError(msg.format(len(user_provided_output_types), len(output_vars)))

        new_outputs = []
        for i, output_type in enumerate(user_provided_output_types):
            required_output_dtype = output_type.dtype
            output_var = output_vars[i]
            if (
                required_output_dtype is None
                or not (
                    types.is_tensor(output_var.sym_type) or types.is_scalar(output_var.sym_type)
                )
                or required_output_dtype == output_var.dtype
            ):
                # no need to update the output var's dtype in this case
                new_outputs.append(output_var)
            elif output_var in input_vars:
                # Here is this rare special case, that the program input is also an output
                # For this case, we don't do anything, and throw a warning message
                new_outputs.append(output_var)
                logger.warning(
                    f"Output var '{output_var.name}' is also an input var, hence the "
                    f"dtype cannot be changed: output var '{output_var.name}' remains "
                    f"dtype {types.builtin_to_string(output_var.dtype)}"
                )
            else:
                output_var_name = output_var.name
                output_var.set_name(
                    output_var_name + "_type_" + types.builtin_to_string(output_var.dtype)
                )
                new_output_var = mb.cast(
                    x=output_var, dtype=types.builtin_to_string(required_output_dtype)
                )
                new_output_var.set_name(output_var_name)
                Block._copy_scope_info(output_var, new_output_var)
                new_outputs.append(new_output_var)

        func.set_outputs(new_outputs)

    def apply(self, prog: Program):
        for func in prog.functions.values():
            self.adjust_function_output_types(func)
