# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import numpy as np
import six

from coremltools.converters.nnv2 import CoremlBuilder as cb
from coremltools.converters.nnv2.nnv2_program.passes.pass_registry import register_pass

def commingle_loop_vars_block(block):
    for op in list(block.operations):
        for b in op.blocks:
            commingle_loop_vars_block(b)

        if op.op_type != 'while_loop':
            continue

        cond_block = op.blocks[0]
        body_block = op.blocks[1]

        for v_out, body_vx_in in zip(op.outputs, body_block.inputs):
            # Disable check as v_out is not visible in body_block.
            body_block.replace_uses_of_var_after_op(anchor_op=None,
                    old_var=body_vx_in, new_var=v_out, no_check_var_visibility=True)
        for v_out, cond_vx_in in zip(op.outputs, cond_block.inputs):
            cond_block.replace_uses_of_var_after_op(anchor_op=None,
                    old_var=cond_vx_in, new_var=v_out, no_check_var_visibility=True)

        # replace block inputs
        body_block._block_inputs = op.outputs
        cond_block._block_inputs = op.outputs

@register_pass(namespace='nnv1_backend')
def commingle_loop_vars(prog):
    """
    prog: SsaProgram

    # NNv1 backend expects output vars as loop vars. Example:
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
    # Result:
    #    main(%a: (1, 2, fp32),
    #         %b: (1, 2, fp32)) {
    #      block0() {
    #        %loop:0: (1, 2, fp32), %loop:1: (1, 2, fp32) = \
    #        while_loop(loop_vars=(%a, %b))
    #          loop_cond(%loop:0, %loop:1) {
    #            %cond_var: (bool) = some_op(x=%loop:0, y=%loop:1)
    #          } -> (%cond_var)
    #          loop_body(%loop:0, %loop:1) {
    #            %add_0: (1, 2, fp32) = add(x=%loop:0, y=%loop:1)
    #          } -> (%add_0, %loop:1)
    #      } -> (%loop:0, %loop:1)
    #    }
    #
    # Comment: The resulting program is no longer SSA (multiple assignments on
    # %loop:0).
    """
    for f_name, f in prog.functions.items():
        commingle_loop_vars_block(f)
