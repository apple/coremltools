# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass

def _handle_unused_inputs_func(f):
    unused_inputs = [v for v_name, v in f.inputs.items() if len(v.child_ops) == 0]

    with f:
        for v in unused_inputs:
            # copy the input
            v_tmp = mb.identity(x=v, name=v.name + "_tmp")

@register_pass(namespace="nn_backend")
class handle_unused_inputs(AbstractGraphPass):
    """
    prog: Program

    # NN doesn't allow unused inputs. Insert an identity op to consume
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
    # (Notice that input %x is not consumed. This causes error in NN.)
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
    def apply(self, prog):
        for f in prog.functions.values():
            _handle_unused_inputs_func(f)
