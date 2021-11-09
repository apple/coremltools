#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass


def _move_operations_to_the_end_block(block, op_type_to_move):
    # Moves ops with `op_type_to_move` in `block.operations` (list) to the end of the program.
    # Note: ops with `op_type_to_move` and is dead code are moved toward end, which can be eliminated
    # later with dead-code-elimination pass.
    #
    # Inputs:
    #  - block (mil.Block): block to be modified in-place
    #  - op_type_to_move (List[str])
    # Returns:
    #  - set[Var]: Set of vars consumed in block (or returned as block output)

    # first_use maps var to (index, op) representing the first op in block.operation that consumes this var.
    first_use = {}  # var -> op
    ops_to_remove = []  # list of ops to be deleted at the end of pass
    for index, op in enumerate(reversed(block.operations[:])):
        current_op = op

        if op.op_type in op_type_to_move:
            # Mark op for deletion
            ops_to_remove.append(op)

            # Create list of operations consuming each output of current operation
            first_consumers = [first_use[v] for v in op.outputs if v in first_use]

            before_op = None  # None means adding at the end of block
            if len(first_consumers) > 0:
                # Current op should be moved right before this first consumer of one of it's output.
                #  1. Find indices for all the consumer ops of outputs
                #  2. Move current op right before first consumer i.e. smallest index in block.operations
                first_use_indices = [block.operations.index(first_use_op) for first_use_op in first_consumers]
                before_op = block.operations[min(first_use_indices)]

            with block:
                # Create new copy of current operation
                new_var = getattr(mb, op.op_type)(**op.inputs, before_op=before_op)

                if not isinstance(new_var, (list, tuple)):
                    new_var = [new_var]

                # Override current_op to be newly created op to ensure `first_use`
                # points to newly created op instead of old one.
                current_op = new_var[0].op

                for old_output_var, new_output_var in zip(op.outputs, new_var):
                    block.replace_uses_of_var_after_op(
                        anchor_op=None, old_var=old_output_var, new_var=new_output_var)

        # Collect input vars from sub-block if present
        relevant_inputs = set()
        for b in current_op.blocks:
            relevant_inputs |= _move_operations_to_the_end_block(b, op_type_to_move)  # |= is set union

        # Collect vars from operation input
        for v in current_op.inputs.values():
            if isinstance(v, (tuple, list)):
                relevant_inputs |= set(v)
                continue
            relevant_inputs.add(v)

        # Mark current op as first use for all the input vars
        #  a) of it's sub-block
        #  b) of current op
        for v in relevant_inputs:
            # input is seen for the first time or
            # current_op is first_use i.e. appears before earlier recorded first_use.
            # Note: since ops are moved to the end, it's possible that an op is moved right after
            # earlier recorded first_use and in such cases, first_use should not be modified.
            #
            # == Example ==
            #     main( %x: (10, 20, fp32)(Tensor)) {
            #      block0() {
            #         %cast_0: (10, 20, fp16)(Tensor) = cast(x= %x, dtype = "fp16", name = "cast_0")
            #         %cast_1: (10, 20, fp32)(Tensor) = cast(x= %cast_0, dtype = "fp32", name = "cast_1")
            #         %transpose_0: (20, 10, fp16)(Tensor) = transpose(x= %cast_0, perm = [1, 0], name = "transpose_0")
            #         %transpose_1: (10, 20, fp16)(Tensor) = transpose(x= %transpose_0, perm = [1, 0], name = "transpose_1")
            #       } -> (% cast_1, % transpose_1)
            #     }
            # In above example, `%cast_1` will be moved to the end of the block and first_use info for `%cast_0`
            # should point to `%transpose_0` and not to `%cast_1`
            if v not in first_use or block.operations.index(first_use[v]) > block.operations.index(current_op):
                first_use[v] = current_op

    # Remove ops that are reordered
    block.remove_ops(ops_to_remove)

    # Returns set of vars consumed in current block
    vars_consumed_in_block = set([v for v in first_use])
    vars_consumed_in_block.update(block.outputs)
    return vars_consumed_in_block


@register_pass(namespace="common")
class topological_reorder(AbstractGraphPass):
    """
    Topologically reorders the list of operations in a program by re-ordering operations closers to their
    first use or at the end if it's not being consumed by any other operation.

    Currently, This pass re-orders only Transpose and Cast operations.

    Example: input program
        main(x: (2, 4, fp32)) {
            x = mb.cast(x=x, dtype="fp16")
            x1 = mb.square(x=x)
            x1_t = mb.transpose(x=x1, perm=[1, 0])
            x2 = mb.cast(x=x1_t, dtype="fp32")
            x3 = mb.log(x=x)
            x3_t = mb.transpose(x=x3, perm=[1, 0])
            x4 = mb.cast(x=x3_t, dtype="fp32")
            x5 = mb.relu(x=x)
            x6 = mb.cast(x=x5, dtype="fp32")
            x7 = mb.relu(x=x6)
            x8 = mb.relu(x=x)
        } -> x2, x4, x7, x8

    After moving `cast` ops becomes
        main(x: (2, 4, fp32)) {
            x = mb.cast(x=x, dtype="fp16")
            x1 = mb.square(x=x)
            x1_t = mb.transpose(x=x1, perm=[1, 0])
            x3 = mb.log(x=x)
            x3_t = mb.transpose(x=x3, perm=[1, 0])
            x5 = mb.relu(x=x)
            x6 = mb.cast(x=x5, dtype="fp32")
            x7 = mb.relu(x=x6)
            x8 = mb.relu(x=x)
            x4 = mb.cast(x=x3_t, dtype="fp32")
            x2 = mb.cast(x=x1_t, dtype="fp32")
        } -> x2, x4, x7, x8

    After moving `transpose` ops becomes
        main(x: (2, 4, fp32)) {
            x = mb.cast(x=x, dtype="fp16")
            x1 = mb.square(x=x)
            x3 = mb.log(x=x)
            x5 = mb.relu(x=x)
            x6 = mb.cast(x=x5, dtype="fp32")
            x7 = mb.relu(x=x6)
            x8 = mb.relu(x=x)
            x3_t = mb.transpose(x=x3, perm=[1, 0])
            x4 = mb.cast(x=x3_t, dtype="fp32")
            x1_t = mb.transpose(x=x1, perm=[1, 0])
            x2 = mb.cast(x=x1_t, dtype="fp32")
        } -> x2, x4, x7, x8
    """
    def apply(self, prog):
        for f_name, f in prog.functions.items():
            _move_operations_to_the_end_block(f, ['cast', 'transpose'])
