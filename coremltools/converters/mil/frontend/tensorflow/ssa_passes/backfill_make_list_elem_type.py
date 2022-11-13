#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from coremltools.converters.mil.mil.var import ListVar


@register_pass(namespace="tensorflow")
class backfill_make_list_elem_type(AbstractGraphPass):
    """
    TF's TensorArrayV3 (represented as make_list in mil) doesn't necessarily
    contain elem shape/type, which is known when write is performed. We
    backfill elem type info to make_list

    Inputs:

        prog: Program
    """
    def apply(self, prog):
        for f in prog.functions.values():
            _backfill_make_list_elem_type_block(f)

@block_context_manager
def _backfill_make_list_elem_type_block(block):
    # shallow copy hides changes on f.operations during the loop
    for op in block.operations:
        for b in op.blocks:
            _backfill_make_list_elem_type_block(b)

        if op.op_type != "tf_make_list":
            continue

        if op.outputs[0].elem_type != types.unknown:
            # elem_type of the list is known
            continue

        list_var = op.outputs[0]
        elem_type = _infer_elem_type(list_var)  # types.tensor
        if elem_type is None:
            msg = (
                "No list_write or list_scatter op to infer make_list "
                + "'{}' element type. Block:\n{}"
            )
            raise ValueError(msg.format(op.name, op.enclosing_block))

        # elem_shape can be runtime-detemrined, which cannot be inferred here at this point,
        # so we add an internal _const_symbolic node to cover both static and dynamic cases.
        elem_shape = [dim.name if is_symbolic(dim) else dim for dim in elem_type.get_shape()]
        new_list = mb.make_list(
            init_length=op.init_length,
            dynamic_length=op.dynamic_length,
            elem_shape=tuple(elem_shape),
            dtype=op.inputs["dtype"],
            before_op=op,
            name=op.name,
        )

        block.replace_uses_of_var_after_op(
            anchor_op=op, old_var=op.outputs[0], new_var=new_list
        )
        block.remove_ops([op])


def _infer_elem_type(list_var):
    """
    Returns types.tensor. None if failed to infer element type.
    Example:

    Given:

    main(%update: (2,fp32)) {
      block0() {
        %list: List[unknown] = tf_make_list(...) # unknown elem type
        %while_loop_0:0: (i32), %while_loop_0:1: List[(2,fp32)] = while_loop(loop_vars=(...))
          while_loop_0_body(...) {
            %list_write_0: List[(2,fp32)] = list_write(index=..., ls=%list, value=%update)
          } -> (%add_0, %list_write_0)

        Result:

        main(%update: (2,fp32)) {
          block0() {
        %list: List[(2,fp32)] = tf_make_list(...) # Get the elem type from list_write
        %while_loop_0:0: (i32), %while_loop_0:1: List[(2,fp32)] = while_loop(loop_vars=(...))
          while_loop_0_body(...) {
            %list_write_0: List[(2,fp32)] = list_write(index=..., ls=%list, value=%update)
          } -> (%add_0, %list_write_0)
    """
    # Search for child op that have informative element types
    for o in list_var.child_ops:
        if o.op_type in ["list_write", "list_scatter"]:
            return o.outputs[0].elem_type
        if o.op_type == "while_loop":
            idx = list(o.loop_vars).index(list_var)
            block = o.blocks[0]
            # the corresponding Var in body block
            block_var = block.inputs[idx]
            elem_type = _infer_elem_type(block_var)
            if elem_type is not None:

                def _set_types_for_block_inputs(block):
                    block_var = block.inputs[idx]
                    new_block_var = ListVar(name=block_var.name, elem_type=elem_type,
                                            init_length=block_var.sym_type.T[1],
                                            dynamic_length=block_var.sym_type.T[2])
                    block._replace_var(block_var, new_block_var)

                _set_types_for_block_inputs(o.blocks[0])  # condition block
                _set_types_for_block_inputs(o.blocks[1])  # body block

                return elem_type
            # otherwise continue to other block_var (a list_var can be
            # passed into while_loop twice).
    return None
