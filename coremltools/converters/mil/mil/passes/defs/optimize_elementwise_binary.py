#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types as _types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="common")
class divide_to_multiply(AbstractGraphPass):
    """
    Convert divide into multiply if the divisor is ``const``.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._divide_to_multiply_block(f)

    @block_context_manager
    def _divide_to_multiply_block(self, block):
        for op in list(block.operations):
            for b in op.blocks:
                self._divide_to_multiply_block(b)
            if len(op.blocks) > 0:
                # This op can't be divided.
                continue

            # If real_div has integer input, the result is an integer (following TensorFlow spec).
            # Hence, this pass needs disabled if the input is not float, since it translates y
            # to a floating point number. If x or y was originally an integer, and y becomes
            # a floating point number, then the original type
            # signature (with integer output) would not be preserved.
            if op.op_type == "real_div" and op.y.val is not None and _types.is_float(op.x.dtype):
                new_y_val = np.array(1.0, dtype=op.y.val.dtype) / op.y.val
                if not np.isfinite(new_y_val).all():
                    continue

                x = mb.mul(x=op.x, y=new_y_val, name="_inversed_" + op.name, before_op=op)
                op.enclosing_block.replace_uses_of_var_after_op(
                    anchor_op=op, old_var=op.outputs[0], new_var=x
                )
                block.remove_ops([op])


@register_pass(namespace="common")
class fuse_elementwise_to_batchnorm(AbstractGraphPass):
    """
    Fold ``mul`` + ``add`` into a ``batchnorm`` 
    if the ``const`` feeding into the ``mul``/``add`` is of shape ``(1,C,1,1)`` or ``(C,1,1)``
    and input to ``mul`` is of rank 4.

    .. code-block::

        Given:
                 [Const]   [Const]
                    |         |
                    V         V
        [...] --> [Mul] --> [Add] --> [...]

        That is,

            %2 = op1(%1)
            %3 = mul(%2, constant)
            %4 = add(%3, constant)
            %5 = op2(%4)
            ...

        Result:

        [...] --> [BatchNorm] --> [...]

        That is,
            %2 = op1(%1)
            %4 = batchnorm(%2)
            %5 = op2(%4)
            ...
    """

    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self._fuse_elementwise_to_batchnorm_block(f)

    @staticmethod
    def _match_pattern(op):
        if op.outputs[0] in op.enclosing_block.outputs:
            return None

        if op.op_type == "mul":
            # find add
            child_ops = op.outputs[0].child_ops
            if len(child_ops) == 1:
                add_op_candidate = list(child_ops)[0]
                if add_op_candidate.op_type == "add":
                    return add_op_candidate
        return None

    @staticmethod
    def _try_to_transform(mul_op, add_op, block):
        def _find_const_input_val(op):
            if op.x.val is not None:
                return op.x.val
            if op.y.val is not None:
                return op.y.val
            return None

        def _check_shape(arr):
            """
            return True if shape is of form
            (1,C,1,1) or (C,1,1)
            """
            rank = len(arr.shape)
            if not (rank == 3 or rank == 4):
                return False
            C = arr.shape[-3]
            if not (arr.shape == (1, C, 1, 1) or arr.shape == (C, 1, 1)):
                return False
            return True

        non_const_input_mul = mul_op.x if mul_op.x.val is None else mul_op.y
        if non_const_input_mul.rank != 4:
            return False

        gamma = _find_const_input_val(mul_op)
        beta = _find_const_input_val(add_op)
        if gamma is None or beta is None:
            return False

        if not (isinstance(gamma, np.ndarray) and isinstance(beta, np.ndarray)):
            return False

        # check that gamma and beta have shape (1,C,1,1) or (C,1,1)
        # that is they are doing vector addition on the axis=-3, which is what the
        # batchnorm layer does (batchnorm layer only works on rank 4 input tensors)
        if not (_check_shape(gamma) and _check_shape(beta)):
            return False

        C = gamma.shape[-3]
        if C == 1:
            return False

        out_name = add_op.outputs[0].name
        x = mb.batch_norm(
            x=non_const_input_mul,
            mean=np.zeros((C,), np.float32),
            variance=np.ones((C,), np.float32),
            gamma=np.squeeze(gamma),
            beta=np.squeeze(beta),
            name=out_name,
            before_op=mul_op,
        )

        add_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=add_op, old_var=add_op.outputs[0], new_var=x
        )
        # Remove all the ops at once
        block.remove_ops([mul_op, add_op])
        return True

    @block_context_manager
    def _fuse_elementwise_to_batchnorm_block(self, block):
        fusion_status = False
        for op in list(block.operations):
            for b in op.blocks:
                block_changed = True
                while block_changed:
                    block_changed = self._fuse_elementwise_to_batchnorm_block(b)
            if len(op.blocks) > 0:
                # This op can't be mul
                continue

            add_op = self._match_pattern(op)
            if add_op is not None:
                fusion_status = self._try_to_transform(op, add_op, block)
                # has to break as the downstream iterator is affected.
                if fusion_status:
                    return fusion_status
        return fusion_status


@register_pass(namespace="common")
class rank0_expand_dims_swap(AbstractGraphPass):
    """
    Identify the pattern of a ``rank-0`` binary elementwise operation followed by an ``expand_dims`` op.
    In the MIL backend, the output of the ``elementwise`` op becomes rank 1. Hence, an ``expand_dims`` op
    should be added after both of the ``rank-0`` tensors, and the final ``expand_dims`` should be removed.
    If the output var of the binary elementwise op is consumed by more than one op, a ``squeeze`` op
    is inserted.
    
    `Input`

    .. code-block::

            [...](rank-0) --> sub --> expand_dims (axes=[0]) --> [...]
                               ^   |
                               |   |--> op2
                               |   |
                               |   |--> op3
                               |
                         [scalar const]

    `Output`

    .. code-block::

            [...](rank-0) --> expand_dims (axes=[0]) --> sub --> [...]
                                                          ^   |
                                                          |   |--> squeeze ---> op2
                                                          |                |
                                                          |                |--> op3
                                                          |
                                                    expand_dims (axes=[0])
                                                          ^
                                                          |
                                                          |
                                                    [scalar const]

    """

    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self._rank0_expand_dims_swap(f)

    @staticmethod
    def _try_to_transform(op, block):
        op_type = op.op_type
        ops_to_remove = []
        if op.x.rank != 0 or op.y.rank != 0:
            return False

        # One and only one input is a scalar const
        if (op.x.val is None) == (op.y.val is None):
            return False

        var_1, var_2 = op.x, op.y
        ops_to_remove.append(op)

        # check if the output is consumed by exact one expand_dims op and other ops
        expand_dims_ops = []
        other_ops = []
        child_ops = list(op.outputs[0].child_ops)
        for child_op in child_ops:
            if child_op.op_type == "expand_dims":
                expand_dims_ops.append(child_op)
            else:
                other_ops.append(child_op)
        if len(expand_dims_ops) != 1:
            return False

        # check the expand_dim op has axes = [0]
        expand_dims_op = expand_dims_ops[0]
        if expand_dims_op.axes.val != [0]:
            return False
        ops_to_remove.append(expand_dims_op)
        ops_to_remove += other_ops

        for out in op.outputs:
            if out in block.outputs:
                return False

        # add a expand_dims op after each rank-0 tensor
        var_1_expand = mb.expand_dims(x=var_1, axes=[0], before_op=op)
        var_2_expand = mb.expand_dims(x=var_2, axes=[0], before_op=op)

        # add a new elementwise binary op
        elem_op = getattr(mb, op_type)

        # replace var for the expand_dims op
        x = elem_op(
            x=var_1_expand, y=var_2_expand, name=expand_dims_op.outputs[0].name, before_op=op
        )
        expand_dims_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=expand_dims_op, old_var=expand_dims_op.outputs[0], new_var=x
        )

        # replace var for other ops
        if len(other_ops) >= 1:
            elem_op_output = op.outputs[0]
            squeeze = mb.squeeze(x=x, before_op=op)
            for other_op in other_ops:
                new_op = getattr(mb, other_op.op_type)
                kargs = {}
                for k, v in other_op.inputs.items():
                    if v == elem_op_output:
                        kargs[k] = squeeze
                    else:
                        kargs[k] = v
                kargs["name"] = other_op.name
                kargs["before_op"] = other_op
                new_var = new_op(**kargs)
                other_op.enclosing_block.replace_uses_of_var_after_op(
                    anchor_op=other_op, old_var=other_op.outputs[0], new_var=new_var
                )

        # Remove all the ops at once
        block.remove_ops(ops_to_remove)
        return True

    @block_context_manager
    def _rank0_expand_dims_swap(self, block):
        fusion_occurred = False
        for op in list(block.operations):
            for b in op.blocks:
                block_changed = True
                while block_changed:
                    block_changed = self._rank0_expand_dims_swap(b)
            if len(op.blocks) > 0:
                # This op can't be elementwise binary ops
                continue

            if op.op_type in ["add", "sub", "mul", "real_div", "floor_div"]:
                fusion_occurred = self._try_to_transform(op, block)
                # has to break as the downstream iterator is affected.
                if fusion_occurred:
                    return fusion_occurred
        return fusion_occurred
