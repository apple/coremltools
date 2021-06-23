# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


def detect_loop_invariants(while_op):
    block = while_op.blocks[1]  # body block
    loop_invariant_ids = []  # list of index in op.loop_vars, block.inputs
    for i, vx_in in enumerate(block.inputs):
        vx_out = block.outputs[i]  # first output is cond var.
        return_input_as_output = vx_in == vx_out
        # this block output is a var from outside of the block
        output_from_outside_of_block = (
            vx_out in block._visible_vars_from_enclosing_block()
        )
        if return_input_as_output or output_from_outside_of_block:
            loop_invariant_ids.append(i)

    # TODO: All outputs that depend on only invariants are invariant. We
    # need to move computation out of while loop.
    return loop_invariant_ids


def loop_invariant_elimination_block(block):
    # Phase 1: Find vars needed to be renamed.
    #
    # while_loop outputs need to be renamed if the output will be eliminated
    # (due to loop invariant) and is returned as block output (which would
    # change the return var name and the program interface).
    #
    # list[(v_src, v_tgt, before_op)]: will rename v_src to v_tgt before
    # before_op (a while_loop)
    output_rename = []
    for op in list(block.operations):
        for b in op.blocks:
            loop_invariant_elimination_block(b)

        if op.op_type != "while_loop":
            continue

        loop_invariant_ids = detect_loop_invariants(op)
        for i in loop_invariant_ids:
            output_rename.append((op.loop_vars[i], op.outputs[i], op))
        if len(loop_invariant_ids) > 0:
            # Avoid the following case:
            # %a, %b = while_loop(..., name="b")
            # becomes
            # %b = identity(..., name="b")
            # %a = while_loop(..., name="b")
            # (two ops with the same name -> name collision)
            op.name = op.name + "_renamed"

    # Phase 2: insert rename ops. This changes block.operations
    for v_src, v_tgt, op in output_rename:
        if v_tgt in block.outputs:
            # rename the loop output to existing block output names
            with block:
                res = mb.identity(x=v_src, before_op=op, name=v_tgt.name)
                op.enclosing_block.replace_uses_of_var_after_op(
                    anchor_op=op, old_var=v_tgt, new_var=res
                )

    # Phase 3: Perform loop invariant elimination without fear!
    for op in list(block.operations):
        if op.op_type != "while_loop":
            continue
        loop_invariant_ids = detect_loop_invariants(op)

        loop_variant_vars = []

        # replace uses of loop_invariants with its source from outside of the
        # while_loop op.
        for i in loop_invariant_ids:
            for block in op.blocks:
                block.replace_uses_of_var_after_op(
                    anchor_op=None, old_var=block.inputs[i],
                    new_var=op.loop_vars[i]
                )

        # replace block inputs
        for block in op.blocks:
            block.remove_inputs([block.inputs[i] for i in loop_invariant_ids])

        # remove invariants from while_loop loop_vars
        for i in loop_invariant_ids:
            # replace usage of while_loop outputs that we'll eliminate.
            op.enclosing_block.replace_uses_of_var_after_op(
                anchor_op=op, old_var=op.outputs[i], new_var=op.loop_vars[i]
            )

        # Remove after replacing to ensure program is valid
        for i in loop_invariant_ids:
            op.loop_vars[i].remove_child_op(op)

        op.loop_vars = tuple(
            v for i, v in enumerate(op.loop_vars) if i not in loop_invariant_ids
        )
        op._input_vars["loop_vars"] = op.loop_vars

        # remove invariants from while_loop body_block outputs
        body_block = op.blocks[1]
        body_block.set_outputs(
            [
                v
                for i, v in enumerate(body_block.outputs)
                if i not in loop_invariant_ids
            ]
        )

        # op._output_vars doesn't include cond var
        op._output_vars = [
            v for i, v in enumerate(op._output_vars) if i not in loop_invariant_ids
        ]

        # check healthy state
        op.enclosing_block.validate()


@register_pass(namespace="common")
def loop_invariant_elimination(prog):
    """
    prog: Program

    # When a block does not modify a block input var, eliminate that block
    # input var and use the corresponding var in the outer scope. Example:
    #
    # Given:
    #    main(%a: (1, 2, fp32),
    #         %b: (1, 2, fp32)) {
    #      block0() {
    #        %loop:0: (1, 2, fp32), %loop:1: (1, 2, fp32) = \
    #        while_loop(loop_vars=(%a, %b))
    #          loop_cond(%a.x, %b.x) {
    #            %cond_var: (bool) = some_op(x=%a.x, y=%b.x)
    #          } -> (%cond_var)
    #          loop_body(%a.x, %b.x) {
    #            %add_0: (1, 2, fp32) = add(x=%a.x, y=%b.x)
    #          } -> (%add_0, %b.x)
    #      } -> (%loop:0, %loop:1)
    #    }
    #
    # (Notice that %b.x is constant through while loop iterates)
    #
    # Result:
    #    main(%a: (1, 2, fp32),
    #         %b: (1, 2, fp32)) {
    #      block0() {
    #        %loop:1: (1, 2, fp32) = identity(x=%b)
    #        %loop:0: (1, 2, fp32) = \
    #        while_loop(loop_vars=(%a))
    #          loop_cond(%a.x) {
    #            %cond_var: (bool) = some_op(x=%a.x, y=%b)
    #          } -> (%cond_var)
    #          loop_body(%a.x) {
    #            %add_0: (1, 2, fp32) = add(x=%a.x, y=%b)
    #          } -> (%add_0)
    #      } -> (%loop:0, %loop:1)
    #    }
    #
    # where we eliminate loop invariant %b.x from while_loop, which returns 1
    # instead of 2 outputs. We also preserve the return var names with
    # identity.
    """
    for f in prog.functions.values():
        loop_invariant_elimination_block(f)
