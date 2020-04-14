# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

from coremltools.converters.nnv2 import CoremlBuilder as cb
from coremltools.converters.nnv2.nnv2_program.passes.pass_registry import register_pass

def handle_return_unused_inputs_func(f):
    returned_unused_inputs = []
    for v_name, v in f.inputs.items():
        if v not in f.outputs:
            continue
        returned_unused_inputs.append(v)

    with f:
        for v in returned_unused_inputs:
            # copy twice since NNv1 layer cannot have input name == output name
            v_tmp = cb.identity(x=v, name=v.name+'_tmp')
            res = cb.identity(x=v_tmp, name=v.name)
            res.op.enclosing_block.replace_uses_of_var_after_op(anchor_op=res.op,
                    old_var=v, new_var=res)

@register_pass(namespace='nnv1_backend')
def handle_return_unused_inputs(prog):
    """
    prog: SsaProgram

    # NNv1 cannot handle returning input as output. Insert an identity op for
    # those cases. Example:
    #
    # Given:
    #    main(%a: (1, 2, fp32),
    #         %b: (1, 2, fp32)) {
    #      block0() {
    #        %mul_0_y_0: (i32)* = const(val=2)
    #        %mul_0: (1, 2, fp64) = mul(x=%a, y=%mul_0_y_0)
    #      } -> (%mul_0, %b)
    #    }
    #
    # (Notice that %b is returned from input. This causes error in NNv1)
    #
    # Result:
    #    main(%a: (1, 2, fp32),
    #         %b: (1, 2, fp32)) {
    #      block0() {
    #        %mul_0_y_0: (i32)* = const(val=2)
    #        %mul_0: (1, 2, fp64) = mul(x=%a, y=%mul_0_y_0)
    #        %b_tmp: (1, 2, fp32) = identity(x=%b)
    #        %b: (1, 2, fp32) = identity(x=%b_tmp)
    #      } -> (%mul_0, %b)
    #    }
    #
    # where identity is applied twice since NNv1 layer cannot have
    # input name == output name
    """
    for f_name, f in prog.functions.items():
        handle_return_unused_inputs_func(f)
