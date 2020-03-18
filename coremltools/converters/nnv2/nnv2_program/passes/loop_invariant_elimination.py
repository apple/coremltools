# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import numpy as np
import six

from coremltools.converters.nnv2 import CoremlBuilder as cb
from coremltools.converters.nnv2.nnv2_program.passes.pass_registry import register_pass


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

        if op.op_type != 'while_loop':
            continue

        body_block = op.blocks[1]
        for v_in, body_vx_in, body_vx_out, op_out in zip(op.loop_vars,
                body_block.inputs, body_block.outputs, op.outputs):
            if body_vx_in == body_vx_out:
                output_rename.append((v_in, op_out, op))

    # Phase 2: insert rename
    for v_src, v_tgt, op in output_rename:
        if v_tgt in block.outputs:
            # rename the loop output to existing block output names
            with block:
                res = cb.identity(x=v_src, before_op=op, name=v_tgt.name)
                op.enclosing_block.replace_var_after_op(anchor_op=op,
                        old_var=v_tgt, new_var=res)

    # Phase 3: Perform loop invariant elimination without fear!
    for op in list(block.operations):
        for b in op.blocks:
            loop_invariant_elimination_block(b)

        if op.op_type != 'while_loop':
            continue
        cond_block = op.blocks[0]
        body_block = op.blocks[1]
        loop_invariants = set()
        body_block_invariants = []  # only contains loop variant inputs
        cond_block_invariants = []  # only contains loop variant inputs
        loop_variant_vars = []
        for i, (v_in, body_vx_in, cond_vx_in, body_vx_out) in enumerate(\
                zip(op.loop_vars, body_block.inputs, cond_block.inputs,
                    body_block.outputs)):
            if body_vx_in == body_vx_out:
                # body_vx_in != cond_vx_in (SSA property), though
                # body_vx_in.name == cond_vx_in.name.
                loop_invariants.add((i, v_in, body_vx_in, cond_vx_in))
                body_block_invariants.append(body_vx_in)
                cond_block_invariants.append(cond_vx_in)
            else:
                loop_variant_vars.append(v_in)

        # replace block inputs
        body_block.remove_inputs(body_block_invariants)
        cond_block.remove_inputs(cond_block_invariants)

        # replace occurences of loop_variants within the body and cond blocks.
        for i, v_in, body_vx_in, cond_vx_in in loop_invariants:
            body_block.replace_var_after_op(anchor_op=None,
                    old_var=body_vx_in, new_var=v_in)
            cond_block.replace_var_after_op(anchor_op=None,
                    old_var=cond_vx_in, new_var=v_in)

        # remove invariants from while_loop loop_vars
        original_loop_vars = op.loop_vars
        op.loop_vars = tuple(loop_variant_vars)
        op._input_vars['loop_vars'] = op.loop_vars

        # replace usage of while_loop outputs that we'll eliminate.
        for i, v_in, _, _ in loop_invariants:
            op.enclosing_block.replace_var_after_op(anchor_op=op,
                    old_var=op.outputs[i], new_var=v_in)

        # remove invariants from while_loop outputs
        invariant_ids = set([i for i, _, _, _ in loop_invariants])
        body_block._outputs = [v for i, v in enumerate(body_block.outputs) \
                if i not in invariant_ids]
        op._output_vars = [v for i, v in enumerate(op._output_vars) \
                if i not in invariant_ids]

@register_pass(namespace='common')
def loop_invariant_elimination(prog):
    """
    prog: SsaProgram

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
    for f_name, f in prog.functions.items():
        loop_invariant_elimination_block(f)
