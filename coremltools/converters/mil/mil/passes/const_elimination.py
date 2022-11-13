#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@block_context_manager
def _const_elimination_block(block):
    # shallow copy hides changes on f.operations during the loop
    for op in list(block.operations):

        if op.op_type == "const":
            continue

        for b in op.blocks:
            _const_elimination_block(b)

        all_outputs_are_replaced = True
        for i, o in enumerate(op.outputs):
            if o.val is not None:
                res = mb.const(
                    val=o.val,
                    before_op=op,
                    # same var name, but different python
                    # instance does not violate SSA property.
                    name=o.name,
                )
                
                if op.enclosing_block.try_replace_uses_of_var_after_op(
                    anchor_op=op,
                    old_var=o,
                    new_var=res,
                ):
                    # rename the const output
                    o.set_name(o.name + '_ignored')
                else:
                    all_outputs_are_replaced = False
            else:
                all_outputs_are_replaced = False

        if all_outputs_are_replaced:
            op.remove_from_block()

@register_pass(namespace="common")
class const_elimination(AbstractGraphPass):
    """
    prog: Program

    # Replace non-const ops that have const Var
    # outputs replaced with const op. Example:
    #
    # Given:
    #   %2, %3 = non_const_op(...)  # %2 is const, %3 isn't const
    #   %4 = other_op(%2, %3)
    #
    # Result:
    #   _, %3 = non_const_op(...)  # _ is the ignored output
    #   %2_const = const()         # %2_const name is for illustration only
    #   %4 = other_op(%2_const, %3)
    #
    """
    def apply(self, prog):
        for f in prog.functions.values():
            _const_elimination_block(f)
