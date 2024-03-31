#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="torch")
class torch_tensor_assign_to_core(AbstractGraphPass):
    """
    Map Torch dialect ops `torch_tensor_assign` into core opset.

    Currently, we transform the torch_tensor_assign op using mb.scatter.
    """
    def apply(self, prog):
        for f in prog.functions.values():
            _torch_tensor_assign_to_core_block(f)

@block_context_manager
def _torch_tensor_assign_to_core_block(block):
    for op in list(block.operations):
        for b in op.blocks:
            _torch_tensor_assign_to_core_block(b)

        if op.op_type in ["torch_tensor_assign"]:
            _transform_tensor_assign(op, block)


def _transform_tensor_assign(op, block):
    shape = mb.shape(x=op.x, before_op=op)
    dim_prod = mb.reduce_prod(x=shape, before_op=op)
    ref_indices = mb.range_1d(end=dim_prod, start=0, step=1, before_op=op)
    ref_indices = mb.reshape(x=ref_indices, shape=shape, before_op=op)
    ref_sliced_indices = mb.slice_by_index(
                            x=ref_indices,
                            begin=op.begin,
                            end=op.end,
                            stride=op.stride,
                            begin_mask=op.begin_mask,
                            end_mask=op.end_mask,
                            squeeze_mask=op.squeeze_mask,
                            before_op=op,
                        )
    flatten_indices = mb.reshape(x=ref_sliced_indices, shape=[-1], before_op=op)
    flatten_updates = mb.reshape(x=op.updates, shape=[-1], before_op=op)
    flatten_data = mb.reshape(x=op.x, shape=[-1], before_op=op)
    new_data = mb.scatter(
        data=flatten_data,
        indices=flatten_indices,
        updates=flatten_updates,
        mode="update",
        before_op=op,
    )
    new_data = mb.reshape(x=new_data, shape=shape, before_op=op)

    op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=op, old_var=op.outputs[0], new_var=new_data
    )
    # Remove all the ops at once
    block.remove_ops([op])
