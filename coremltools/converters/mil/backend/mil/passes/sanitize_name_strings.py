#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.passes.defs.preprocess import NameSanitizer
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="mil_backend")
class sanitize_name_strings(AbstractGraphPass):
    """
    Sanitize the names of vars and ops to make sure
    that they are of the format as described in the NameSanitizer class, i.e.
    of the format [a-zA-Z_][a-zA-Z0-9_]*
    """
    def apply(self, prog):
        for f in prog.functions.values():
            sanitizer_vars = NameSanitizer(prefix="var_")
            sanitizer_ops = NameSanitizer(prefix="op_")
            NameSanitizer.sanitize_block(
                f, sanitizer_vars, sanitizer_ops, prog.functions["main"].input_types
            )
