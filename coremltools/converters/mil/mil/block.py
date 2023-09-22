#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
from collections import Counter, OrderedDict

from coremltools import _OPSET, _logger as logger
from coremltools.converters.mil._deployment_compatibility import \
    AvailableTarget as _target

from . import SPACES, types
from .types.symbolic import is_symbolic, k_used_symbols
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
        self.operations = []

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

        if self.outer_op is None and not isinstance(self, Function):
            msg = "Block {} is not Function and thus outer_op cannot be None"
            raise ValueError(msg.format(self.name))

        self.validate()

    def validate(self):
        """
        Basic validation to protect against some invalid state.
        """
        if not DEBUG:
            return

        for op in self.operations:
            for b in op.blocks:
                b.validate()
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

    def is_var_visible_in_block(self, var, upto_op_with_id=None):
        """
        Checks if a var is visible to ops starting from id=`upto_op_with_id` inside the block.

        Var is visible if
        - It is the output of a const op, or
        - It is the output of "preceding" operations in that block, or
        - It is visible in the enclosing block, or
        - It is either a block or a function input

        If upto_op_with_id is None, outputs of all operations inside the block are visible to
        that block.
        """

        if var in self._internal_vars:
            return True

        inputs = self.function_inputs if isinstance(self, Function) else self.inputs
        if var in inputs:
            return True

        idx = len(self.operations) if upto_op_with_id is None else upto_op_with_id

        for i in range(idx-1, -1, -1):
            op_outputs = self.operations[i].outputs
            if op_outputs is not None and var in op_outputs:
                return True

        if self.outer_op is not None:
            enclosing_block = self.outer_op.enclosing_block
            outer_op_id = enclosing_block.find_op_id_in_block(self.outer_op)
            if enclosing_block.is_var_visible_in_block(var, upto_op_with_id=outer_op_id):
                return True

        return False

    def find_op_id_in_block(self, target_op):
        try:
            idx = self.operations.index(target_op)
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

    def _insert_op_before(self, new_op, before_op=None):
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

        idx = len(self.operations) if before_op is None else self.find_op_id_in_block(before_op)

        # check inputs are visible
        for k, v in new_op.inputs.items():
            if not isinstance(v, (Var, tuple)):
                continue
            vs = [v] if isinstance(v, Var) else v
            for v in vs:
                if not self.is_var_visible_in_block(v, upto_op_with_id=idx):
                    before_op_name = before_op.name if before_op is not None else "None"
                    msg = "Op '{}' input {}={} is not in scope of {} before {}"
                    raise ValueError(msg.format(new_op.name, k, v.name, self.name, before_op_name))

        # add new_op
        if before_op is None:
            self.operations.append(new_op)
        else:
            self.operations.insert(idx, new_op)

    def _replace_var(
        self,
        old_var,
        new_var,
        start=0,
        end_id=-1,
        no_check_var_types=False,
    ):
        """
        Helper function for replace_uses_of_var_after_op
        """
        num_ops_affected = 0

        if end_id == -1:
            op_list = self.operations[start:]
        else:
            op_list = self.operations[start : end_id + 1]

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

        if end_id != -1 and old_var.op not in op_list:
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
        anchor_op,
        old_var,
        new_var,
        end_op=None,
        no_check_var_types=False,
        no_check_var_visibility=False,
    ):
        """
        :param anchor_op: Operation
        :param old_var: Var
        :param new_var: Var
        :param end_op: Operation
        :param no_check_var_types: bool
        :param no_check_var_visibility: bool
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
            no_check_var_visibility=no_check_var_visibility,
        )
        return True

    def replace_uses_of_var_after_op(
        self,
        anchor_op,
        old_var,
        new_var,
        no_check_var_visibility=False,
        end_op=None,
        no_check_var_types=False,
        force_replace=False,
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

        no_check_var_visibility: True to disable the check ensuring new_var is visible
        (visibility requirement depends on anchor_op).

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

        start = self.find_op_id_in_block(anchor_op) + 1 if anchor_op is not None else 0
        end_id = self.find_op_id_in_block(end_op) if end_op is not None else -1

        if not no_check_var_visibility:
            self.validate()

            idx = start if anchor_op is not None else len(self.operations)
            visibility_error_msg = (
                    "new_var '{}' is not visible in block '{}' at or before "
                    + "anchor_op '{}'"
            )
            anchor_op_name = "None" if anchor_op is None else anchor_op.name

            if isinstance(new_var, ComplexVar):
                # For CompleVar, as it's just a temp wrapper to transit the real and imag data, we
                # check the visibility of its real and imaginary Var instead.
                if not self.is_var_visible_in_block(new_var.real, upto_op_with_id=idx):
                    raise ValueError(
                        visibility_error_msg.format(
                            new_var.real.name, self.name, anchor_op_name
                        )
                    )
                if not self.is_var_visible_in_block(new_var.imag, upto_op_with_id=idx):
                    raise ValueError(
                        visibility_error_msg.format(
                            new_var.imag.name, self.name, anchor_op_name
                        )
                    )
            else:
                if not self.is_var_visible_in_block(new_var, upto_op_with_id=idx):
                    raise ValueError(
                        visibility_error_msg.format(
                            new_var.name, self.name, anchor_op_name
                        )
                    )

        if end_id != -1 and end_id < start:
            msg = "end_op '{}' comes before the anchor_op '{}'"
            raise ValueError(msg.format(end_op.name, anchor_op.name))

        num_ops_affected = self._replace_var(
            old_var,
            new_var,
            start=start,
            end_id=end_id,
            no_check_var_types=no_check_var_types,
        )

        logger.debug("Num ops affected in replacing var: {}".format(num_ops_affected))

    def remove_ops(self, existing_ops):
        """
        Remove ops in `existing_ops`.

        Args: existing_ops: List[Operation]. All ops in this list must be pre-existing in the
        block. It allows duplicated ops, but duplicated ops will only be removed once.

        Raises:
            ValueError if any `op` in `existing_ops` meets any of following conditions:
              - `op` is not found in the block
              - any other op in the block uses output Vars of `op`
              - the output var is block's output
        """
        self.validate()

        # Dedup ops because each op can only be deleted once.
        existing_ops_set = set(existing_ops)
        existing_ops = list(existing_ops_set)
        # Find the idx of each to-be-removed op, and raise errors if any op couldn't be found.
        idxs = [-1] * len(existing_ops)
        for i, op in enumerate(self.operations):
            if op in existing_ops_set:
                idxs[existing_ops.index(op)] = i
        if -1 in idxs:
            not_found = []
            for i, op in zip(idxs, existing_ops):
                if i == -1:
                    not_found.append(op.name)
            raise ValueError(
                "Ops {} not found in block {}".format(not_found, self.name)
            )

        # Remove ops in reverse topological order
        pairs = list(zip(idxs, existing_ops))
        pairs.sort(key=lambda x: x[0], reverse=True)

        for idx, op in pairs:
            for i, v in enumerate(op.outputs):
                # Check that no ops depend on op's outputs
                if len(v.child_ops) > 0:
                    child_op_names = [s.name for s in v.child_ops]
                    msg = (
                        "Cannot delete op '{}' with active output at id {}: '{}' "
                        + "used by ops {}"
                    )
                    raise ValueError(msg.format(op.name, i, v.name, child_op_names))
                # Check that the output Var isn't block's output
                if v in self._outputs:
                    msg = (
                        "cannot delete op {} with output {}: {} "
                        + "that's block {}'s output"
                    )
                    raise ValueError(msg.format(op.name, i, v.name, self.name))

            for b in op.blocks:
                b.set_outputs([])
                b.remove_ops(b.operations)

            # Remove the op (in reverse topological order)
            self.operations.pop(idx)
            op.enclosing_block = None

            for v in op.inputs.values():
                if isinstance(v, (tuple, list)):
                    for vv in v:
                        vv.remove_child_op(op)
                else:
                    v.remove_child_op(op)

    def operations_for_vars(self, end_vs):
        """
        Inputs:

        end_vs: list[Operation].

        Return:

        list[Operation] which are subset of self.operations that are ancestors
        of `end_vs`. Also do recursion into nested blocks.
        """
        used_vars = set(end_vs)
        used_ops = []
        for op in reversed(self.operations):
            # if none of op's output is used, delete op
            if not set(op.outputs).intersection(used_vars):
                continue

            used_ops.append(op)  # append in reverse topological order

            # recursively search for nested blocks
            ops_to_check = []
            for b in op.blocks:
                ops_to_check += b.operations_for_vars(b.outputs)
            ops_to_check.append(op)

            # mark used vars
            for op_to_check in ops_to_check:
                # mark all op's inputs to used
                for _, input_var in op_to_check.inputs.items():
                    if isinstance(input_var, (tuple, list)):
                        used_vars.update(list(input_var))
                    else:
                        used_vars.add(input_var)

        return used_ops[::-1]

    def _propagate_nonreplaceable_vars(self):
        def propagate_nonreplaceable_vars_block(block):
            for op in list(block.operations):
                for b in op.blocks:
                    propagate_nonreplaceable_vars_block(b)
                if op.outputs is None:
                    continue
                for o in op.outputs:
                    o._reset_nonreplaceable_vars_upstream()
                    o._set_nonreplaceable_vars_upstream()
        propagate_nonreplaceable_vars_block(self)

    def indented_str(self, indent=None):
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
            s += op.indented_str(indent + SPACES * 1)
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

        # str -> Var
        self._input_dict = OrderedDict()
        for k, v in self.placeholder_inputs.items():
            v.set_name(k)  # set to user input name
            self._input_dict[k] = v.outputs[0]
        self.function_inputs = tuple(self._input_dict.values())

        global k_used_symbols
        global k_num_internal_syms
        for inp in self.function_inputs:
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

    def to_str(self, func_name="function"):
        func_name = func_name + "[{}]".format(_OPSET[self.opset_version])
        if len(self._input_dict) == 0:
            s = func_name + "()"
        else:
            inputs = [(in_name, ph) for in_name, ph in self._input_dict.items()]
            s = func_name + "(" + str(inputs[0][1])
            for in_name, ph in inputs[1:]:
                s += ",\n" + " " * (len(func_name) + 1) + str(ph)
            s += ") {\n"
            s += self.indented_str(SPACES)
            s += "}\n"
        return s
