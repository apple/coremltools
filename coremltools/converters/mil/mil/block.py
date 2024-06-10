#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
from collections import Counter, OrderedDict
from typing import List, Optional, Set, Tuple, Union

from coremltools import _OPSET
from coremltools import _logger as logger
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as _target
from coremltools.converters.mil.input_types import InputType

from . import SPACES, types
from .operation import Operation
from .scope import SCOPE_STACK, VALID_OPS_TO_COPY_SCOPE_INFO, ScopeSource, add_graph_pass_scope
from .types.symbolic import is_symbolic, k_used_symbols
from .utils import CacheDoublyLinkedList
from .var import ComplexVar, InternalVar, Var
from .visitors.dot_visitor import DotVisitor

# BLOCK_STACK[-1] is the current block
BLOCK_STACK = []
DEBUG = False


def curr_block():
    if len(BLOCK_STACK) == 0:
        raise ValueError("Must call Builder inside an Function" + " or Block")
    return BLOCK_STACK[-1]

def curr_opset_version():
    block = curr_block()
    while not isinstance(block, Function):
        block = block.outer_op.enclosing_block
    return block.opset_version

def is_current_opset_version_compatible_with(opset_version):
    if curr_opset_version() is None:
        return opset_version <= _target.iOS13
    return curr_opset_version() >= opset_version


class InvalidBlockStateError(Exception):
    pass


class Block:
    __slots__ = [
        "name",
        "_block_inputs",
        "_outputs",
        "operations",
        "_internal_vars",
        "outer_op",
        "cache_operations",
        "_essential_scope_sources",
    ]

    counter = 0

    @classmethod
    def _get_new_name(cls):
        curr_val = cls.counter
        cls.counter += 1
        return "block" + str(curr_val)

    def __init__(self, block_inputs=None, outer_op=None, name=None):
        """
        Inputs:

        block_inputs: python tuple[Var].

            block_inputs is None except when the block represents loop. By
            convention block_inputs should have name ending in '.x', and the
            Variable are not produced by any op (block_inputs[i]._op is None).

            Ex:

            #    main(%a: (1, 2, fp32),
            #         %b: (1, 2, fp32),
            #         %c: (1, 2, fp32)) {
            #      block0() {
            #        %const1: (1, fp32) = const(...)
            #        %loop:0: (1, 2, fp32), %loop:1: (1, 2, fp32) = \
            #        while_loop(loop_vars=(%a, %b))
            #          loop_cond(%a.x, %b.x) {
            #            %blah: (bool) = some_op(x=%a.x, y=%b.x)
            #            %cond_var: (bool) = some_op2(x=%a.x, y=%blah)
            #          } -> (%cond_var)
            #          loop_body(%a.x, %b.x) {
            #            %add_0: (1, 2, fp32) = add(x=%a.x, y=%b.x)
            #          } -> (%add_0, %b.x)
            #        %linear: (1, fp32) = linear(...)
            #      } -> (%loop:0, %loop:1)
            #    }

            %a.x, %b.x are block_inputs.

            `some_op` in `loop_cond` block can access %a, %b, %a.x, %b.x.
            `some_op`, however, cannot take %linear as input.

        outer_op: Operation
            The enclosing op. None iff this Block is an Function.

        function_inputs: tuple[Var]
            function_inputs are always visible for this block and all blocks
            nested within. If function_inputs is None, get it from
            `outer_op.block`
        """
        self.name = name
        if self.name is None:
            self.name = Block._get_new_name()

        # list[Operation]. Topologically sorted.
        self.operations = CacheDoublyLinkedList()

        # Must be set before self.validate()
        self.outer_op = outer_op

        self._block_inputs = block_inputs
        if self._block_inputs is None:
            self._block_inputs = tuple()

        # list[Var]. This is converted to str when generating MIL proto.
        self._outputs = []

        # If we create const, whose inputs (mode, val) cannot be const
        # (infinite recursion). They must be considered as always visible.
        self._internal_vars = set()

        # List[ScopeSource]. During graph pass, those scope source cannot be missed
        self._essential_scope_sources = []

        if self.outer_op is None and not isinstance(self, Function):
            msg = "Block {} is not Function and thus outer_op cannot be None"
            raise ValueError(msg.format(self.name))

        self.validate()

    def _add_essential_scope_source(
        self, scope_source: Union[ScopeSource, List[ScopeSource]]
    ) -> None:
        """
        Add essential scope sources to self._essential_scope_sources.
        When self.validate() is called, we make sure that all source info are not missing.
        """
        if not isinstance(scope_source, list):
            scope_source = [scope_source]

        for source in scope_source:
            if source in self._essential_scope_sources:
                raise ValueError(f"{source} already exist in _essential_scope_sources.")
            self._essential_scope_sources.append(source)

    def _check_has_scope_info(self) -> None:
        """
        Check no ops in the function are missing scope information.
        """

        def _check_has_scope_info_block(block: Block):
            for op in block.operations:
                for b in op.blocks:
                    _check_has_scope_info_block(b)
                for scope in self._essential_scope_sources:
                    if scope not in op.scopes or len(op.scopes[scope]) == 0:
                        raise ValueError(
                            f"op {op.name} with scopes {op.scopes} is missing essential scopes {scope}."
                        )

        _check_has_scope_info_block(self)

    def _check_vars_visibility_in_block(
        self, visible_vars_from_outer_block: Optional[Set[Var]] = None
    ):
        """
        This utils does a one pass program-wise checking of vars visibility.
        That is, each input of an op, should appear before the op in the sequantial order.

        For the debug purpose, if you want to pinpoint the operation which caused the
        invalid program state, please set DEBUG=True, and it will be captured by the ``is_var_visible_in_block`` utils.
        """
        if visible_vars_from_outer_block is None:
            visible_vars_from_outer_block = set()
        block_inputs = list(self.inputs.values()) if isinstance(self, Function) else self.inputs
        visible_vars_in_block = set(block_inputs)

        for op in self.operations:
            for b in op.blocks:
                b._check_vars_visibility_in_block(
                    visible_vars_from_outer_block=visible_vars_from_outer_block.union(
                        visible_vars_in_block
                    )
                )
            for val in op.get_flattened_inputs():
                if (
                    val not in self._internal_vars
                    and val not in visible_vars_in_block
                    and val not in visible_vars_from_outer_block
                ):
                    raise ValueError(f"Var {val} not visible in the block {self.name}.")
            for out_var in op.outputs:
                visible_vars_in_block.add(out_var)

    def validate(
        self,
        force_validate: Optional[bool] = False,
        check_essential_scope: Optional[bool] = False,
    ) -> None:
        """
        Basic validation to protect against some invalid state.
        If force_validate is False, the validation is done only if the global variable DEBUG=True.
        """
        if not DEBUG and not force_validate:
            return

        # Check vars visibility
        if isinstance(self, Function):
            self._check_vars_visibility_in_block()

        # Other validations
        for op in self.operations:
            for b in op.blocks:
                b.validate(force_validate=force_validate)
            if op.outputs is None:
                raise InvalidBlockStateError()

            # Check the input output relationships
            # from outputs -> inputs
            for ov in op.outputs:
                child_op_count = Counter(ov.child_ops)
                for next_op, c in child_op_count.items():
                    c_actual = next_op.get_flattened_inputs().count(ov)
                    if c_actual != c:
                        msg = (
                            "Var {} should be consumed by op {} {}"
                            + " times, but op {} uses it {} times.\n{}"
                        )
                        raise InvalidBlockStateError(
                            msg.format(
                                ov.name,
                                next_op.name,
                                c,
                                next_op.name,
                                c_actual,
                                next_op,
                            )
                        )

            # from inputs -> outputs
            input_var_count = Counter(op.get_flattened_inputs())
            for iv, c in input_var_count.items():
                c_actual = iv.child_ops.count(op)
                if c_actual != c:
                    msg = (
                        "Var {} should be consumed by op {} {}"
                        + " times, but op {} uses it {} times.\n{}"
                    )
                    raise InvalidBlockStateError(
                        msg.format(iv.name, op.name, c_actual, op.name, c, op)
                    )

        # 1 to 1 mapping between Block outputs and Var.consuming_blocks
        for op in self.operations:
            for ov in op.outputs:
                for b in ov.consuming_blocks:
                    if ov not in b.outputs:
                        msg = "Var {} should be output of block {}: {}"
                        raise ValueError(msg.format(ov.name, b.name, b))

        for v in self.outputs:
            if self not in v.consuming_blocks:
                msg = "Var {} should be output of block {}: {}"
                raise ValueError(msg.format(ov.name, b.name, b))

        # checking internal vars are consistent with self._internal_vars
        internal_var_in_block = set()
        for op in self.operations:
            for v in op.internal_inputs.values():
                internal_var_in_block.add(v)
        if not internal_var_in_block == self._internal_vars:
            raise ValueError(
                "internal vars in the block are not consistent with self._internal_vars."
            )

        # check essential scope info are not missing
        if check_essential_scope:
            self._check_has_scope_info()

    def remove_inputs(self, curr_input_vars):
        """
        curr_input_vars: list[Var], whose elements must be in
        self._block_inputs.
        """
        self.validate()
        remove_idx = [self._block_inputs.index(v) for v in curr_input_vars]
        self._block_inputs = [
            v for i, v in enumerate(self._block_inputs) if i not in remove_idx
        ]

    def find_ops(self, prefix=None, op_type=None):
        """
        Return list of ops with name matching `prefix` if specified and
        op_type, if specified. At least one of {prefix, op_type} must be specified.

        prefix: str

        Return list[Operation]. Empty list if no op satisfies.
        """
        if prefix is None and op_type is None:
            raise ValueError("Must specify one of {prefix, op_type}")
        found_ops = []
        for op in self.operations:
            prefix_match = prefix is None or op.name[: len(prefix)] == prefix
            op_type_match = op_type is None or op.op_type == op_type
            if prefix_match and op_type_match:
                found_ops.append(op)
            for b in op.blocks:
                found_ops.extend(b.find_ops(prefix=prefix, op_type=op_type))
        return found_ops

    def add_internal_var(self, internal_var):
        if not isinstance(internal_var, InternalVar):
            raise ValueError("Only InternalVar can be manually added to Block.")
        self._internal_vars.add(internal_var)

    @property
    def inputs(self):
        return self._block_inputs

    @property
    def outputs(self):
        return self._outputs

    def is_var_visible_in_block(self, var: Var, upto_op: Optional[Operation] = None):
        """
        Checks if a var is visible to ops starting from id=`upto_op_with_id` inside the block.

        Var is visible if
        - It is the output of a const op, or
        - It is the output of "preceding" operations in that block, or
        - It is visible in the enclosing block, or
        - It is either a block or a function input

        If upto_op_with_id is None, outputs of all operations inside the block are visible to
        that block.

        For debugging:
        - By default (DEBUG=False), this utils is guarded by the flag in calling code and not running.
        - By setting DEBUG=True, this utils is triggered in multiple places in the code base,
          so the users can pinpoint the exact place where an invalid operation is made by the converter.
          Beware that, the converter could be slow in the debug mode, since the overal conversion
          time will explode to O(N^2) in the average cases by this util.
        """
        if not DEBUG:
            # Only in debug mode, there is a chance that self.operations is type of list when executing this function.
            assert isinstance(
                self.operations, CacheDoublyLinkedList
            ), "operations must be type of CacheDoublyLinkedList."

        if var in self._internal_vars:
            return True

        inputs = list(self.inputs.values()) if isinstance(self, Function) else self.inputs
        if var in inputs:
            return True

        if upto_op is None:
            if var.op in self.operations:
                return True
        else:
            if isinstance(self.operations, list):
                # This could only happen in debug mode
                assert DEBUG is True, "block.operations can only be type of list in debug mode."
                idx = self.find_op_id_in_block(upto_op)
                for i in range(idx - 1, -1, -1):
                    if var.op is self.operations[i]:
                        return True
            else:
                cursor = self.operations._get_node_from_op(upto_op).prev
                while cursor is not None:
                    if cursor.op is var.op:
                        return True
                    cursor = cursor.prev

        if self.outer_op is not None:
            enclosing_block = self.outer_op.enclosing_block
            if enclosing_block.is_var_visible_in_block(var, upto_op=self.outer_op):
                return True

        return False

    def find_op_id_in_block(self, target_op: Operation) -> int:
        if len(self.operations) > 0 and target_op == self.operations[-1]:
            return len(self.operations) - 1

        op_list = self.operations if isinstance(self.operations, list) else list(self.operations)

        try:
            idx = op_list.index(target_op)
        except ValueError:
            raise ValueError("Op {} not found in {}: {}".format(target_op.name, self.name, self))
        return idx

    def set_outputs(self, outputs):
        """
        outputs: list[Var]
        """
        if not isinstance(outputs, list):
            raise ValueError("Outputs must be list of Vars")

        self.validate()

        # check var visibility in debug mode
        if DEBUG:
            for ov in outputs:
                if not self.is_var_visible_in_block(ov):
                    msg = (
                        "Var {} is not visible in block {} and thus cannot "
                        + "be a block output.\n{}"
                    )
                    raise ValueError(msg.format(ov.name, self.name, self))

        # For duplicate vars in self._outputs, only remove block once.
        for ov in set(self._outputs):
            ov.consuming_blocks.remove(self)

        # Need to copy, or block's output would be completely tied to a var's
        # output and we cannot replace a block output with another var's
        # output.
        self._outputs = copy.copy(outputs)
        # For duplicate vars in outputs, only add consuming_blocks once.
        for ov in set(outputs):
            ov.consuming_blocks.append(self)

    def __enter__(self):
        global BLOCK_STACK
        BLOCK_STACK.append(self)
        return self

    def __exit__(self, type, value, traceback):
        self._propagate_nonreplaceable_vars()
        global BLOCK_STACK
        BLOCK_STACK = BLOCK_STACK[:-1]

    def _insert_op_before(self, new_op: Operation, before_op: Optional[Operation] = None):
        """
        A private API used by builder. Please use `builder.YOUR_OP(...,before_op)`.

        new_op's outputs are not used (not input to any other op) after
        this call. All inputs to new_op must be visible at or before
        the before_op (i.e., new_op must be added in topologically sorted
        order). Note that this is more restrictive than MIL, whose Block
        supports lexical scoping and thus an op can reference Var in enclosing
        scopes. new_op.name must be unique in the block.

        before_op=None to append new_op at the end of self.operations.

        Given:   %2 = op0(%1, %1)
                 %4 = op2(%1)
                 %6 = op3(%4, %4)

        Execute: insert_op_before(op1, before_op=op2),
                 where %3 = op1(%1, %2)

        Result:  %2 = op0(%1, %1)
                 %3 = op1(%1, %2)
                 %4 = op2(%1)
                 %6 = op3(%4, %4)

        Comment: We assume op1 has been constructed outside the block with
        %1, %2 as inputs. Typically it's builder's job to create an op and
        insert into the current block.

        Comment: insert_op_before(op1, before_op=op0) would error as %2 (an input to op1)
        is not visible before op0.
        """
        self.validate()

        if isinstance(self.operations, CacheDoublyLinkedList):
            self.operations.insert_op_before(new_op, before_op)
            return

        if before_op is None:
            self.operations.append(new_op)
            return

        # check inputs visibility in debug mode
        if DEBUG:
            for k, v in new_op.inputs.items():
                if not isinstance(v, (Var, tuple)):
                    continue
                vs = [v] if isinstance(v, Var) else v
                for v in vs:
                    if not self.is_var_visible_in_block(v, upto_op=before_op):
                        before_op_name = before_op.name if before_op is not None else "None"
                        msg = "Op '{}' input {}={} is not in scope of {} before {}"
                        raise ValueError(
                            msg.format(new_op.name, k, v.name, self.name, before_op_name)
                        )

        idx = self.find_op_id_in_block(before_op)
        self.operations.insert(idx, new_op)

    def _replace_var(
        self,
        old_var: Var,
        new_var: Var,
        anchor_op: Optional[Operation] = None,
        end_op: Optional[Operation] = None,
        no_check_var_types: Optional[bool] = False,
    ):
        """
        Helper function for replace_uses_of_var_after_op
        """
        self._copy_metadata(old_var, new_var)
        self._copy_scope_info(old_var, new_var)

        num_ops_affected = 0

        # If we start checking right after the old_var, we can reduce the time
        # complexity hugely, by only checking the child_ops, without iterating
        # through whole program.
        # This fix reduce the overall time from O(N) -> O(1).
        replace_vars_right_after_old_var = (
            end_op is None
            and len(self.operations) > 0
            and anchor_op is not None
            and anchor_op is old_var.op
        )

        # We should only compute start_idx and end_idx once if needed.
        start_idx = end_idx = None

        if replace_vars_right_after_old_var:
            op_list = list(old_var.child_ops)
        else:
            if isinstance(self.operations, list):
                start_idx = self.find_op_id_in_block(anchor_op) + 1 if anchor_op is not None else 0
                end_idx = (
                    self.find_op_id_in_block(end_op)
                    if end_op is not None
                    else len(self.operations) - 1
                )
                op_list = self.operations[start_idx : end_idx + 1]
            else:
                assert isinstance(
                    self.operations, CacheDoublyLinkedList
                ), f"Expect operations be type of CacheDoublyLinkedList. Got {type(self.operations)}."
                if len(self.operations) == 0 and anchor_op is not None:
                    raise ValueError(f"anchor op {anchor_op} not in the block.")

                start_node = (
                    self.operations.start
                    if anchor_op is None
                    else self.operations._get_node_from_op(anchor_op).next
                )
                cursor = start_node
                op_list = []
                while cursor is not None:
                    op_list.append(cursor.op)
                    if cursor.op is end_op:
                        break
                    cursor = cursor.next

        for op in op_list:
            new_inputs = {}
            affected = False
            for k, v in op.inputs.items():
                if isinstance(v, (list, tuple)) and old_var in v:
                    new_inputs[k] = tuple(new_var if vv == old_var else vv for vv in v)
                    affected = True
                elif v == old_var:
                    new_inputs[k] = new_var
                    affected = True
                else:
                    new_inputs[k] = v
            if affected:
                num_ops_affected += 1
                op.set_inputs(no_check_var_types=no_check_var_types,
                    **new_inputs)

            # Replace recursively.
            for b in op.blocks:
                num_ops_affected += b._replace_var(old_var, new_var)

        # Replace consuming_blocks's outputs.
        # It is important to use list copy here,
        # since replace_block_output_var is going to change the consuming_blocks
        # Note that, there are some expensive index query in the following implementation,
        # but overally it won't affect the time complexity too much,
        # since we can assume the number of the block outputs in a program as a constant.
        # As the result, the amortized time complexity will not blow up.
        for b in list(old_var.consuming_blocks):
            outer_op = b.outer_op

            if outer_op is not None:
                # Query the start and end index if needed
                if start_idx is None:
                    start_idx = (
                        self.find_op_id_in_block(anchor_op) + 1 if anchor_op is not None else 0
                    )
                if end_idx is None:
                    end_idx = (
                        self.find_op_id_in_block(end_op)
                        if end_op is not None
                        else len(self.operations) - 1
                    )

            op_to_idx = {}
            while outer_op is not None:
                block = outer_op.enclosing_block
                if block is self:
                    if len(op_to_idx) == 0:
                        for idx, op in enumerate(self.operations):
                            op_to_idx[op] = idx
                    op_idx = op_to_idx[outer_op]
                    if op_idx >= start_idx and op_idx <= end_idx:
                        b.replace_block_output_var(old_var, new_var)
                    break
                outer_op = block.outer_op

        if end_op is not None and old_var.op not in op_list:
            return num_ops_affected

        if old_var in self._block_inputs:
            idx = self._block_inputs.index(old_var)
            self._block_inputs = list(self._block_inputs)
            self._block_inputs[idx] = new_var
            self._block_inputs = tuple(self._block_inputs)

        # If old_var is block's output, replace as well.
        self.replace_block_output_var(old_var, new_var)
        return num_ops_affected

    def replace_block_output_var(
            self,
            old_var,
            new_var,
    ):
        """
        If old_var is in the list of block's outputs,
        replace old_var with the new_var.
        """
        found_old_var_in_output = False
        # There could be multiple matched `old_var` in output when the program has duplicate vars
        # in the output.
        for idx, output_var in enumerate(self._outputs):
            if old_var == output_var:
                found_old_var_in_output = True
                self._outputs[idx] = new_var
        if found_old_var_in_output:
            new_var.consuming_blocks.append(self)
            # This block no longer uses `old_var` as its outputs
            old_var.consuming_blocks.remove(self)
            # Ensure output name is consistent
            if isinstance(self, Function):
                if new_var in self.inputs.values() and new_var.name != old_var.name:
                    raise ValueError("It is not allowed to modify function inputs name.")
                new_var.name = old_var.name

    def try_replace_uses_of_var_after_op(
        self,
        anchor_op: Operation,
        old_var: Var,
        new_var: Var,
        end_op: Optional[Operation] = None,
        no_check_var_types: Optional[bool] = False,
    ):
        """
        :param anchor_op: Operation
        :param old_var: Var
        :param new_var: Var
        :param end_op: Operation
        :param no_check_var_types: bool
        :return: True if the old_var can be replaced by new_var. False otherwsie.

        This helper function guards the replace_uses_of_var_after_op function,
        by first checking if the old_var could be replaced by the new_var.

        1. If old_var can be replaced by new_var, the replace_uses_of_var_after_op is called,
        and returns True. 2. Return False if the replacement is not allow.
        """
        if not old_var.can_be_replaced_by_var(new_var):
            return False

        self.replace_uses_of_var_after_op(
            anchor_op=anchor_op,
            end_op=end_op,
            old_var=old_var,
            new_var=new_var,
            no_check_var_types=no_check_var_types,
        )
        return True

    @staticmethod
    def _copy_scope_info(src: Var, dst: Var) -> None:
        """
        Populate meta data from old var (src) to new var (dst)
        """
        curr_scopes = SCOPE_STACK.get_curr_scopes()

        if ScopeSource.COREMLTOOLS_GRAPH_PASS in curr_scopes:

            if src.op in VALID_OPS_TO_COPY_SCOPE_INFO[-1]:
                return

            elif dst.op in VALID_OPS_TO_COPY_SCOPE_INFO[-1]:
                op = dst.op
                assert op is not None, "new_var cannot be a placeholder output"
                VALID_OPS_TO_COPY_SCOPE_INFO[-1].remove(op)

                # If old_var is a placeholder output, we assign defaults values to essential scope source
                old_scopes = src.scopes
                if len(old_scopes) == 0:
                    essential_scope_sources = op.enclosing_block._essential_scope_sources
                    for val in essential_scope_sources:
                        res = None
                        if val == ScopeSource.TORCHSCRIPT_MODULE_TYPE:
                            res = ["__COREML__::TORCHSCRIPT_PLACEHOLDER"]
                        elif val == ScopeSource.TORCHSCRIPT_MODULE_NAME:
                            res = [f"__COREML__::TORCHSCRIPT_PLACEHOLDER_{src.name}"]
                        elif val == ScopeSource.EXIR_STACK_TRACE:
                            res = [None]
                        elif val == ScopeSource.EXIR_DEBUG_HANDLE:
                            res = [None]
                        else:
                            raise ValueError(f"No default placeholder info for {val}.")
                        old_scopes[val] = res

                dst.scopes = add_graph_pass_scope(old_scopes, dst.scopes)

                for input in op.inputs.values():
                    if not isinstance(input, (list, tuple)):
                        input = [input]
                    for i in input:
                        Block._copy_scope_info(src, i)

    @staticmethod
    def _copy_metadata(old_var: Var, new_var: Var) -> None:
        """
        Populate meta data from old var to new var
        """
        return

    def replace_uses_of_var_after_op(
        self,
        anchor_op: Operation,
        old_var: Var,
        new_var: Var,
        end_op: Optional[Operation] = None,
        no_check_var_types: Optional[bool] = False,
        force_replace: Optional[bool] = False,
    ):
        """
        Replace all uses of `old_var` with `new_var` after `anchor_op`,
        and before `end_op` (inclusive).

        That is all the ops that use `old_var` will now use `new_var`.
        The op that produces the `old_var` will continue to produce it, its output
        won't be replaced by `new_var`.

        If `anchor_op` is None, replace all input occurrences of `old_var` in the block. If
        `end_op` is None, all occurrences of `old_var` are replaced in the block starting from
        the op just after `anchor_op`

        no_check_var_types: An error will be raised if the type of new_var is not same as the
        old_var, unless `no_check_var_types` is set to True. Normally type inference is
        re-invoked for all the child ops of `old_var` after updating it to `new_var`. However,
        this is skipped if `no_check_var_types` is set to True.

        old_var, new_var must meet the following conditions:

        - old_var, new_var both existing within the block. This implies that
          the op generating new_var must be inserted prior to this
          replacement.

        - Affected ops (i.e., Operation after anchor_op that take old_var as
          input) must generate the same type inference results as before.

        - new_var must be visible at or before anchor_op in the order of
          self.operations.

        Given:   %2 = op0(%1, %1)
                 %3 = op1(%1, %2)
                 %4 = op2(%1)
                 %6 = op3(%4, %4)

        Execute: replace_uses_of_var_after_op(op2, %4, %3)

        Result:  %2 = op0(%1, %1)
                 %3 = op1(%1, %2)
                 %4 = op2(%1)
                 %6 = op3(%3, %3)     # type inference check against %6


        Comment: Execute: replace_uses_of_var_after_op(op1, %4, %3) would lead to
        identical results, as op2 does not take %4 as input.

        Comment: replace_uses_of_var_after_op(op0, %4, %3) would cause error as %3 is
        after op0

        Comment: To avoid clutter, we drop the names of arguments and return
        Var in the illustration above.


        Another example, usage of "end_op":

        Given:   %2 = op0(%1, %1)
                 %3 = op1()
                 %4 = op2(%1, %2)
                 %5 = op3(%2)

        if execute replace_uses_of_var_after_op(anchor_op=op0, old_var=%2, new_var=%3)

        Result:  %2 = op0(%1, %1)
                 %3 = op1()
                 %4 = op2(%1, %3)
                 %5 = op3(%3)

        if execute replace_uses_of_var_after_op(anchor_op=op0, old_var=%2, new_var=%3, end_op=op2)

        Result:  %2 = op0(%1, %1)
                 %3 = op1()
                 %4 = op2(%1, %3)           # %2 is replaced with %3 till here
                 %5 = op3(%2)               # will continue using %2

        """
        if not force_replace and old_var.op is not None and new_var.op is not None:
            if not old_var.can_be_replaced_by_var(new_var):
                old_nonreplaceable_vars = old_var.nonreplaceable_vars_upstream
                new_nonreplaceable_vars = new_var.nonreplaceable_vars_upstream
                err_var = None
                for _var in old_nonreplaceable_vars:
                    if _var not in new_nonreplaceable_vars:
                        err_var = _var
                        break
                msg = (
                    "var {} cannot be replaced by {}. Since the nonreplaceable var {} might "
                    "potentially "
                    "be removed during the replacement of those vars."
                ).format(old_var, new_var, err_var)
                raise ValueError(msg)

        # It is expensive to check the var visibility, and it should only be done while debugging.
        if DEBUG:
            self.validate()

            visibility_error_msg = (
                    "new_var '{}' is not visible in block '{}' at or before "
                    + "anchor_op '{}'"
            )
            anchor_op_name = "None" if anchor_op is None else anchor_op.name

            if isinstance(new_var, ComplexVar):
                # For ComplexVar, as it's just a temp wrapper to transit the real and imag data, we
                # check the visibility of its real and imaginary Var instead.
                if not self.is_var_visible_in_block(new_var.real, upto_op=anchor_op):
                    raise ValueError(
                        visibility_error_msg.format(new_var.real.name, self.name, anchor_op_name)
                    )
                if not self.is_var_visible_in_block(new_var.imag, upto_op=anchor_op):
                    raise ValueError(
                        visibility_error_msg.format(new_var.imag.name, self.name, anchor_op_name)
                    )
            else:
                if not self.is_var_visible_in_block(new_var, upto_op=anchor_op):
                    raise ValueError(
                        visibility_error_msg.format(new_var.name, self.name, anchor_op_name)
                    )
            start = self.find_op_id_in_block(anchor_op) + 1 if anchor_op is not None else 0
            end_id = self.find_op_id_in_block(end_op) if end_op is not None else -1

            if end_id != -1 and end_id < start:
                msg = "end_op '{}' comes before the anchor_op '{}'"
                raise ValueError(msg.format(end_op.name, anchor_op.name))

        num_ops_affected = self._replace_var(
            old_var,
            new_var,
            anchor_op=anchor_op,
            end_op=end_op,
            no_check_var_types=no_check_var_types,
        )

        logger.debug("Num ops affected in replacing var: {}".format(num_ops_affected))

    def remove_ops(self, ops_to_remove: List[Operation]):
        """
        Remove ops in `ops_to_remove`.

        Args: ops_to_remove: List[Operation]. All ops in this list must be pre-existing in the
        block. It allows duplicated ops, but duplicated ops will only be removed once.

        Raises:
            ValueError if any `op` in `ops_to_remove` meets any of following conditions:
              - `op` is not found in the block
              - any other op in the block uses output Vars of `op`
              - the output var is block's output
        """
        self.validate()

        # Dedup ops because each op can only be deleted once.
        ops_to_remove_set = set(ops_to_remove)
        ops_to_remove = list(ops_to_remove_set)

        for op in ops_to_remove:
            for i, v in enumerate(op.outputs):
                # Check that the output Var isn't block's output
                if v in self._outputs:
                    raise ValueError(
                        f"cannot delete op {op.name} with output {i}: {v.name} that's block {self.name}'s output."
                    )

            for b in op.blocks:
                b.set_outputs([])
                b.remove_ops(b.operations)

            self.operations.remove(op)

            op.enclosing_block = None

            for v in op.get_flattened_inputs():
                v.remove_child_op(op)

            # Remove InternalVar from self._internal_vars
            for v in op.internal_inputs.values():
                self._internal_vars.remove(v)

        # In the end, we check no ops depend on removed op's outputs
        for op in ops_to_remove:
            for i, v in enumerate(op.outputs):
                if len(v.child_ops) > 0:
                    child_op_names = [s.name for s in v.child_ops]
                    raise ValueError(
                        f"Cannot delete op '{op.name}' with active output at id {i}: '{v.name}' used by ops {child_op_names}."
                    )

    def _propagate_nonreplaceable_vars(self):
        def propagate_nonreplaceable_vars_block(block):
            for op in block.operations:
                for b in op.blocks:
                    propagate_nonreplaceable_vars_block(b)
                if op.outputs is None:
                    continue
                for o in op.outputs:
                    o._reset_nonreplaceable_vars_upstream()
                    o._set_nonreplaceable_vars_upstream()
        propagate_nonreplaceable_vars_block(self)

    def indented_str(self, indent: Optional[str] = None, print_attr: Optional[bool] = False) -> str:
        if indent is None:
            indent = ""
        s = (
            indent
            + self.name
            + "("
            + ", ".join([str(var) for var in self._block_inputs])
        )
        s += ") {\n"
        for op in self.operations:
            s += op.indented_str(indent + SPACES * 1, print_attr=print_attr)
        s += indent + "} -> ("
        if self._outputs is not None:
            s += ", ".join(["%" + v.name for v in self._outputs])
        s += ")\n"
        return s

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.indented_str()

    def get_dot_string(
        self,
        function_name="main",
        prefix_id=0,
        highlight_debug_op_types=None,
        highlight_debug_op_names=None,
    ):
        """
        Return the dot string that can be used to show the block
        with dot. Const ops are not added to the dot string.

        * Input vars : yellow
        * output vars : goldenrod2
        * op names that user wants to highlight, provided in "highlight_debug_op_names": cyan
        * op types that user wants to highlight, provided in "highlight_debug_op_types": green

        Examples
        --------
        >>> import graphviz
        >>> graphviz.Source(block.get_dot_string()).view()
        >>> # OR
        >>> graphviz.Source(block.get_dot_string()).view(filename='graph.pdf')
        """
        if highlight_debug_op_types is None:
            highlight_debug_op_types = []
        if highlight_debug_op_names is None:
            highlight_debug_op_names = []

        dotstring = "digraph g {\n" + "\tcompound=true;\n"

        input_var_names = list(self.inputs.keys())
        output_var_names = [v.name for v in self.outputs]

        debug_op_types = []
        if len(highlight_debug_op_types) > 0:
            for op in self.operations:
                if op.op_type in highlight_debug_op_types:
                    debug_op_types.append(op.name)

        vis = DotVisitor()
        vis.highlight_nodes(input_var_names, "yellow").highlight_nodes(
            output_var_names, "goldenrod2"
        ).highlight_nodes(highlight_debug_op_names, "cyan").highlight_nodes(
            debug_op_types, "green"
        )

        vis.visit_all(self, nodename_prefix=str(prefix_id))
        res = vis.get_result("subgraph", "cluster_" + function_name.replace("/", "_"))
        dotstring += "\n".join("\t" + r for r in res.split("\n")) + "\n"
        dotstring += "}"
        return dotstring


class Function(Block):
    def __init__(self, inputs, opset_version=None):
        """
        inputs: str -> placeholder
        opset_version: AvailableTarget enum. Describes the opset version of the function
        """
        self.placeholder_inputs = inputs
        self.opset_version = opset_version
        self.output_types = None
        self.input_types = []

        # str -> Var
        self._input_dict = OrderedDict()
        for k, v in self.placeholder_inputs.items():
            v.set_name(k)  # set to user input name
            self._input_dict[k] = v.outputs[0]

        global k_used_symbols
        global k_num_internal_syms
        for inp in self._input_dict.values():
            if types.is_tensor(inp.dtype):
                shapes = inp.dtype.get_shape()
                for s in shapes:
                    if is_symbolic(s):
                        k_used_symbols.add(s)
        super().__init__()

    # Override Block's input
    @property
    def inputs(self):
        return self._input_dict

    @property
    def opset_version(self):
        return self._opset_version

    @opset_version.setter
    def opset_version(self, version):
        if not (
            isinstance(version, _target) or
            version is None
        ):
            raise ValueError("opset_version must be type of coremltools.AvailableTarget")
        self._opset_version = version

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.to_str("function")

    def to_str(
        self, func_name: Optional[str] = "function", print_attr: Optional[bool] = False
    ) -> str:
        func_name = func_name + "[{}]".format(_OPSET[self.opset_version])
        if len(self._input_dict) == 0:
            s = func_name + "()"
        else:
            inputs = [(in_name, ph) for in_name, ph in self._input_dict.items()]
            s = func_name + "(" + str(inputs[0][1])
            for in_name, ph in inputs[1:]:
                s += ",\n" + " " * (len(func_name) + 1) + str(ph)
            s += ")"
        s += " {\n"
        s += self.indented_str(SPACES, print_attr=print_attr)
        s += "}\n"
        return s

    def get_max_opset_version_and_op(self) -> Tuple[_target, Operation]:
        """
        Find the max opset version among all operations in the function.
        Returns the opset version Enum and the corresponding op.
        """
        max_opset_version = _target.iOS13
        op_with_max_opset_version = None

        def update_max_opset_version_block(block):
            nonlocal max_opset_version
            nonlocal op_with_max_opset_version
            for op in block.operations:
                for b in op.blocks:
                    update_max_opset_version_block(b)
                if not hasattr(op, "_op_variants") or not isinstance(op._op_variants, dict):
                    continue
                if op.opset_version > max_opset_version:
                    max_opset_version = op.opset_version
                    op_with_max_opset_version = op

        update_max_opset_version_block(self)
        return max_opset_version, op_with_max_opset_version

    def set_output_types(self, outputs: Optional[List[InputType]] = None) -> None:
        """
        Set the user defined output type for a function.
        Note: the common::update_output_dtypes graph pass takes this information,
        and changes the function output signature accordingly.
        """
        if outputs is not None:
            if not (
                isinstance(outputs, list) and all([isinstance(out, InputType) for out in outputs])
            ):
                raise TypeError(
                    "main outputs should be a list of type ct.TensorType or ct.ImageType"
                )
        self.output_types = outputs

    def set_input_types(self, input_types: List[InputType]):
        if not isinstance(input_types, tuple):
            raise ValueError("main inputs should be tuple of TensorType or ImageType")
        elif not all([isinstance(inp, InputType) for inp in input_types]):
            raise ValueError("main inputs should be tuple of InputSpec")
        self.input_types = input_types
