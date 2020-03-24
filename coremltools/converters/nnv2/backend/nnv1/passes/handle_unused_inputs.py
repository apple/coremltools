# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

from coremltools.converters.nnv2 import CoremlBuilder as cb
from coremltools.converters.nnv2.nnv2_program.passes.pass_registry import register_pass

def handle_unused_inputs_func(f):
    unused_inputs = [v for v_name, v in f.inputs.items() \
            if len(v.child_ops) == 0]

    with f:
        for v in unused_inputs:
            # copy twice since NNv1 layer cannot have input name == output name
            v_tmp = cb.identity(x=v, name=v.name+'_tmp')

@register_pass(namespace='nnv1_backend')
def handle_unused_inputs(prog):
    """
    prog: SsaProgram

    # NNv1 doesn't allow unused inputs. Insert an identity op to consume
    # inputs (though its outputs are not used.). This pass must come after
    # dead code elimination as all inserted code are "dead code". Example:
    #
    # Given:
    #
    #    main(%x: (2, 3, fp32)) {
    #      block0() {
    #        %shape_0_const: (2,i32)* = const(val=[4, 7])
    #      } -> (%shape_0_const)
    #    }
    #
    # (Notice that input %x is not consumed. This causes error in NNv1.)
    #
    # Result:
    #
    #    main(%x: (2, 3, fp32)) {
    #      block0() {
    #        %unused_var: (2, 3, fp32) = identity(x=%x)
    #        %shape_0_const: (2,i32)* = const(val=[4, 7])
    #      } -> (%shape_0_const)
    #    }
    """
    for f_name, f in prog.functions.items():
        handle_unused_inputs_func(f)
