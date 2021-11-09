# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass

def _handle_return_unused_inputs_func(f):

    returned_unused_inputs = filter(lambda x: x in f.outputs, list(f.inputs.values()))

    with f:
        for v in returned_unused_inputs:
            # copy twice since NN layer cannot have input name == output name
            v_tmp = mb.identity(x=v, name=v.name + "_tmp")
            res = mb.identity(x=v_tmp, name=v.name)
            res.op.enclosing_block.replace_uses_of_var_after_op(
                anchor_op=res.op, old_var=v, new_var=res
            )

@register_pass(namespace="nn_backend")
class handle_return_unused_inputs(AbstractGraphPass):
    """
    prog: Program

    # NN cannot handle returning input as output. Insert an identity op for
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
    # (Notice that %b is returned from input. This causes error in NN)
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
    # where identity is applied twice since NN layer cannot have
    # input name == output name
    """
    def apply(self, prog):
        for f in prog.functions.values():
            _handle_return_unused_inputs_func(f)
