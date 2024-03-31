#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import collections
import itertools

from coremltools.converters.mil.mil import Function
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="common")
class dedup_op_and_var_names(AbstractGraphPass):
    """
    For each function, this pass renames ops and variables with the same name
    as any preceding ops/variables across all scopes in the given function,
    where the precedence is implementation-specific. Note that an op name and
    variable names are tracked separately, so an op may have the same name as
    a variable.

    The pass preserves input and output name. Raises ValueError if we cannot
    dedup without changing the input/output var names.

    .. code-block::

        def prog(x):
            x = mb.cast(x=x, dtype="fp16", name="castop")
            x = mb.cast(x=x, dtype="fp32", name="castop")
            x = mb.square(x=x, name="square_last")
            return x

        # Before dedup pass, the op names are ["castop", "castop", "square_last"].
        # After dedup pass, the op names are ["castop", "castop_1", "square_last"].
    """

    def apply(self, prog):
        for func in prog.functions.values():
            # Handle function input/outputs as they cannot be changed (to maintain user interface)
            inputs = list(func.inputs.values())
            io_vars = set(inputs + func.outputs)
            self._ensure_unique_var_names(io_vars)
            seen_var_names = set([v.name for v in io_vars])
            seen_op_names = set()
            self._deduplicate_block(func, set(func.outputs), seen_var_names, seen_op_names)

    @staticmethod
    def _gen_new_name(seen_names, curr_name):
        if curr_name not in seen_names:
            return curr_name
        # make sure the name is unique
        for i in itertools.count(start=1):  # loop from 1 to infinity
            # rename duplicated name start from 1: 'xxx_1'
            new_name = curr_name + "_" + str(i)
            if new_name not in seen_names:
                return new_name

    def _deduplicate_block(self, block, func_outputs, seen_var_names, seen_op_names):
        """
        seen_var_names: set[str]
        seen_op_names: set[str]
        """
        # Add block input (function input is handled separately)
        if not isinstance(block, Function):
            for v in block.inputs:
                v.name = self._gen_new_name(seen_var_names, v.name)
                seen_var_names.add(v.name)

        for op in list(block.operations):
            for b in op.blocks:
                self._deduplicate_block(b, func_outputs, seen_var_names, seen_op_names)
            if op.name is not None:
                op.name = self._gen_new_name(seen_op_names, op.name)
                seen_op_names.add(op.name)
            for v in op.outputs:
                if v in func_outputs:
                    # func output is never renamed
                    continue
                v.name = self._gen_new_name(seen_var_names, v.name)
                seen_var_names.add(v.name)

    @staticmethod
    def _ensure_unique_var_names(v_set):
        """
        v_set: set[Variable]

        All variables in v_set should have different names. Raise ValueError
        otherwise
        """
        names = [v.name for v in v_set]
        dup_names = [name for name, count in collections.Counter(names).items() if count > 1]
        if len(dup_names) > 0:
            raise ValueError(f"Var names {dup_names} is used both as function's input and output")
