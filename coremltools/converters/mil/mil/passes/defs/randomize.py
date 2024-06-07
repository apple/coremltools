#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="common")
class WeightRandomizer(AbstractGraphPass):
    """
    This graph pass randomizes the weights of each ``const`` op

    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._randomize_weights_block(f)

    @block_context_manager
    def _randomize_weights_block(self, block):
        for op in list(block.operations):
            for b in op.blocks:
                self._randomize_weights_block(b)

            if self.is_valid_op(op):
                self.transform_op(op)

    def is_valid_op(self, op: Operation):
        # lazy import to prevent circular import
        from coremltools.converters.mil.backend.mil.load import should_use_weight_file

        if op.op_type == "const" and should_use_weight_file(op.outputs[0].val):
            return True
        return False

    def transform_op(self, op):
        weight = op.outputs[0].val
        random_weight = np.random.rand(*weight.shape).astype(weight.dtype)
        new_var = mb.const(
            val=random_weight,
            before_op=op,
            name=op.name,
        )

        op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=new_var,
            no_check_var_types=True,
        )

        op.enclosing_block.remove_ops([op])
