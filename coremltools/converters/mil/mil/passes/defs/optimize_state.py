#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from typing import List

from coremltools.converters.mil.mil import Block
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, Program
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.scope import ScopeInfo


@register_pass(namespace="common")
class canonicalize_inplace_pattern(AbstractGraphPass):
    """
    As a functional-graph framework, Core ML represents in-place operation as

    .. code-block::

        read_state -> functional operation -> write_state

    Due to the non-uniqueness of topological order, in the list representation of ops,
    ``write_state`` can be anywhere after the functional op. We prefer the canonical order,
    i.e. have ``write_state`` immediately follow the functional op

    In practice

    1. In PyMIL, we do not use ``write_state`` op. Instead, we use ``coreml_update_state``,
    which is the composition of ``write_state -> read_state``

    2. The ``read_state`` op does not matter in the pattern match and transform

    So we will match

    .. code-block::

        functional operation -> coreml_update_state

    then reorder the ``coreml_update_state``. For example

    .. code-block::

        Given:

            mul = mul(state, x)
            add = add(mul, y)
            update = coreml_update_state(state, mul)

        Return:

            mul = mul(state, x)
            update = coreml_update_state(state, mul)
            add = add(mul, y)
    """

    def apply(self, prog: Program) -> None:
        for f in prog.functions.values():
            self._apply_block(f)

    @block_context_manager
    def _apply_block(self, block: Block) -> None:
        block_operation_list = list(block.operations)

        for op in block_operation_list:
            # general boilterplate: special case when op manipulates block
            if op.enclosing_block is None:
                continue
            for b in op.blocks:
                self._apply_block(b)

            # Although downstream iterator (op list) gets changed, the change is only in
            # ``coreml_udpate_state`` op, which cannot be the pattern start and will quick return,
            # so no need to break and iterate
            self._try_match_and_transform_pattern(op, block, block_operation_list)

    def _try_match_and_transform_pattern(
        self, op: Operation, block: Block, block_operation_list: List[Operation]
    ) -> None:
        # state op itself is irrelevant
        if op.op_type in ("read_state", "coreml_update_state"):
            return

        coreml_update_state_ops = self._try_find_child_coreml_update_state_ops(op)
        for coreml_update_state_op in coreml_update_state_ops:
            before_op = block_operation_list[block_operation_list.index(op) + 1]
            scopes = self._construct_scope_info_list_from_op_scopes(op)
            with mb.scope(*scopes):
                immediate_coreml_update_state = mb.coreml_update_state(
                    state=coreml_update_state_op.state,
                    value=coreml_update_state_op.value,
                    before_op=before_op,
                )
            # We need to eliminate dead code here,
            # because our dead code elimination graph pass does not work for coreml_update_state
            if block.try_replace_uses_of_var_after_op(
                anchor_op=coreml_update_state_op,
                old_var=coreml_update_state_op.outputs[0],
                new_var=immediate_coreml_update_state,
            ):
                block.remove_ops([coreml_update_state_op])

    @staticmethod
    def _try_find_child_coreml_update_state_ops(op: Operation) -> List[Operation]:
        coreml_update_state_ops = []
        for output in op.outputs:
            for child_op in output.child_ops:
                if child_op.op_type == "coreml_update_state":
                    coreml_update_state_ops.append(child_op)
        return coreml_update_state_ops

    @staticmethod
    def _construct_scope_info_list_from_op_scopes(op: Operation) -> List[ScopeInfo]:
        scope_info_list = []
        for source, data in op.scopes.items():
            scope_info_list.append(ScopeInfo(source=source, data=data))
        return scope_info_list


@register_pass(namespace="common")
class prefer_state_in_downstream(AbstractGraphPass):
    """
    As a functional-graph framework, Core ML represents in-place operation as

    .. code-block::

        read_state -> functional operation -> write_state

    When the output of the in-place operation is used downstream,
    there are 2 possible patterns, one reuses state memory

    .. code-block::

        read_state -> functional operation -> write_state -> read_state -> ...

    the other wastes memory for keeping functional output

    .. code-block::

                                            |-> write_state
        read_state -> functional operation -|
                                            |-> ...

    We prefer the reuse-state one

    In practice

    1. In PyMIL, we do not use ``write_state`` op. Instead, we use ``coreml_update_state``,
    which is the composition of ``write_state -> read_state``

    2. With canonical inplace pattern (guaranteed by graph pass ``canonicalize_inplace_pattern``),
    simply replace the usage of functional output with ``coreml_update_state`` output is enough

    For example

    .. code-block::

        Given:

            mul = mul(state, x)
            update = coreml_update_state(state, mul)
            add = add(mul, y)

        Return:

            mul = mul(state, x)
            update = coreml_update_state(state, mul)
            add = add(update, y)
    """

    def apply(self, prog: Program) -> None:
        for f in prog.functions.values():
            self._apply_block(f)

    @block_context_manager
    def _apply_block(self, block: Block) -> None:
        for op in list(block.operations):
            # general boilterplate: special case when op manipulates block
            if op.enclosing_block is None:
                continue
            for b in op.blocks:
                self._apply_block(b)

            self._try_match_and_transform_pattern(op, block)

    def _try_match_and_transform_pattern(self, op: Operation, block: Block) -> None:
        if op.op_type == "coreml_update_state":
            # if the var is both blck input and output, we should not replace it
            if op.value in block.outputs and op.value in block.inputs.values():
                return
            other_child_ops = [val for val in op.value.child_ops if val != op]

            # if the var doesn't feed into any other op, this pass should do nothing
            if len(other_child_ops) == 0:
                return

            # if the var only feeds into coreml_update_state ops, this pass should do nothing
            if all([val.op_type == "coreml_update_state" for val in other_child_ops]):
                return

            block.try_replace_uses_of_var_after_op(
                anchor_op=op,
                old_var=op.value,
                new_var=op.outputs[0],
            )
