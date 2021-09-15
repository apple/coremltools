# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from .name_sanitization_utils import NameSanitizer, sanitize_block

@register_pass(namespace="common")
def sanitize_input_output_names(prog):
    """
    Sanitize the names of model input and output vars to make sure
    that they are of the format as described in the NameSanitizer class, i.e.
    of the format [a-zA-Z_][a-zA-Z0-9_]*
    """

    sanitizer_vars = NameSanitizer(prefix="var_")
    sanitizer_ops = NameSanitizer(prefix="op_")

    # sanitize the input/output of the main block
    sanitize_block(prog.functions["main"],
                   sanitizer_vars,
                   sanitizer_ops,
                   prog.main_input_types,
                   sanitize_model_inputs_outputs_only=True)