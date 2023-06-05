#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools import _logger as logger
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Program
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="common")
class const_elimination(AbstractGraphPass):
    """
    Replace non-``const`` ops that have ``const`` Var. Outputs are replaced with the ``const`` op. Example:

    .. code-block::

        Given:
            %2, %3 = non_const_op(...)  # %2 is const, %3 isn't const
            %4 = other_op(%2, %3)

        Result:
            _, %3 = non_const_op(...)  # _ is the ignored output
            %2_const = const()         # %2_const name is for illustration only
            %4 = other_op(%2_const, %3)

    Support options:

    - ``skip_const_by_size``: Skip folding ``const`` ops that have larger number of elements than a threshold.
    """

    _skip_const_by_size = None

    @property
    def skip_const_by_size(self):
        return self._skip_const_by_size

    @skip_const_by_size.setter
    def skip_const_by_size(self, threshold: str):
        try:
            # Convert to float instead of int to support more flexible input such as `1e6`.
            threshold = float(threshold)
        except Exception as e:
            raise ValueError(
                f"Expected to get float threshold, but got `{threshold}` which cannot "
                f"be converted to float. {e}"
            )
        self._skip_const_by_size = float(threshold)

    def apply(self, prog: Program):
        for f in prog.functions.values():
            self._const_elimination_block(f)

    @block_context_manager
    def _const_elimination_block(self, block):
        # shallow copy hides changes on f.operations during the loop
        for op in list(block.operations):
            if op.op_type == "const":
                continue

            for b in op.blocks:
                self._const_elimination_block(b)

            all_outputs_are_replaced = True
            for output in op.outputs:
                if output.can_be_folded_to_const():
                    if (
                        self._skip_const_by_size is not None
                        and len(output.shape) > 0
                        and output.val.size > self._skip_const_by_size
                    ):
                        logger.warning(
                            f"The output ({output}) of op {op} is skipped in const elimination pass "
                            f"because its val size ({output.val.size}) is larger than threshold "
                            f"({self._skip_const_by_size})."
                        )
                        all_outputs_are_replaced = False
                        break

                    res = mb.const(
                        val=output.val,
                        before_op=op,
                        # same var name, but different python
                        # instance does not violate SSA property.
                        name=output.name,
                    )

                    if op.enclosing_block.try_replace_uses_of_var_after_op(
                        anchor_op=op,
                        old_var=output,
                        new_var=res,
                    ):
                        # rename the const output
                        output.set_name(output.name + "_ignored")
                    else:
                        all_outputs_are_replaced = False
                # force const folding of the shape op
                elif output.val is not None and op.op_type == "shape":
                    res = mb.const(
                        val=output.val,
                        before_op=op,
                        # same var name, but different python
                        # instance does not violate SSA property.
                        name=output.name,
                    )
                    op.enclosing_block.replace_uses_of_var_after_op(
                        anchor_op=op,
                        old_var=output,
                        new_var=res,
                        force_replace=True,
                    )
                    # rename the const output
                    output.set_name(output.name + "_ignored")
                else:
                    all_outputs_are_replaced = False

            if all_outputs_are_replaced:
                op.remove_from_block()
