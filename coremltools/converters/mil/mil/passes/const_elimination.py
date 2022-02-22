#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass

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
    def __init__(self):
        self.ops_to_skip = set()

    def set_ops_to_skip(self, prog):
        pass

    def _const_elimination_block(self, block):
        # shallow copy hides changes on f.operations during the loop
        for op in list(block.operations):

            if op.op_type == "const" or op in self.ops_to_skip:
                continue

            for b in op.blocks:
                self._const_elimination_block(b)

            all_outputs_are_const = True
            for i, o in enumerate(op.outputs):
                if o.val is not None:
                    with block:
                        res = mb.const(
                            val=o.val,
                            before_op=op,
                            # same var name, but different python
                            # instance does not violate SSA property.
                            name=o.name,
                        )
                    op.enclosing_block.replace_uses_of_var_after_op(
                        anchor_op=op, old_var=o, new_var=res
                    )
                    # rename the const output
                    o.set_name(o.name+'_ignored')
                else:
                    all_outputs_are_const = False

            if all_outputs_are_const:
                op.remove_from_block()


    def apply(self, prog):
        self.set_ops_to_skip(prog)
        for f in prog.functions.values():
            self._const_elimination_block(f)
