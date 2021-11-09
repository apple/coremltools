# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.input_types import ImageType
# import mil internal ops to add it to the builder
from coremltools.converters.mil.mil.ops import defs as _ops
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.types import nptype_from_builtin

import numpy as np

@register_pass(namespace="mil_backend")
class insert_image_preprocessing_ops(AbstractGraphPass):
    """
    Insert preprocessing ops, right after the input if its of type Image
    """
    def apply(self, prog):
        for f_name, f in prog.functions.items():
            if f_name == 'main':
                _insert_image_preprocessing_ops(f, prog)


def _insert_image_preprocessing_ops(block, prog):
    input_types = list(prog.main_input_types)

    for input_type in input_types:
        if isinstance(input_type, ImageType):
            if input_type.name not in block.inputs:
                continue

            input_var = block.inputs[input_type.name]
            placeholder_op = block.placeholder_inputs[input_type.name]
            first_op = block.operations[0]
            old_var = placeholder_op.outputs[0]
            has_bias = np.any(np.array(input_type.bias) != 0)
            with block:
                last_output = input_var
                input_nptype = nptype_from_builtin(type(last_output.dtype()))
                if input_type.scale != 1:
                    last_output = mb.mul(x=last_output,
                                         y=np.array(input_type.scale, dtype=input_nptype),
                                         before_op=first_op, name=input_var.name + "__scaled__")
                if has_bias:
                    if input_type.color_layout == "G":
                        last_output = mb.add(x=last_output,
                                             y=np.array(input_type.bias, dtype=input_nptype),
                                             before_op=first_op, name=input_var.name + "__biased__")
                    else:
                        if len(last_output.shape) == 3:
                            last_output = mb.add(x=last_output,
                                                 y=np.array(input_type.bias, dtype=input_nptype).reshape([3, 1, 1]),
                                                 before_op=first_op, name=input_var.name + "__biased__")
                        elif len(last_output.shape) == 4:
                            last_output = mb.add(x=last_output,
                                                 y=np.array(input_type.bias, dtype=input_nptype).reshape([1, 3, 1, 1]),
                                                 before_op=first_op, name=input_var.name + "__biased__")
                        else:
                            raise TypeError("Unsupported rank for image input type.")

            if last_output != input_var:
                block.replace_uses_of_var_after_op(anchor_op=last_output.op,
                                                   old_var=old_var,
                                                   new_var=last_output)
