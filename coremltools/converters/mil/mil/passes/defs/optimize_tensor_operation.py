#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.frontend._utils import value_at
from coremltools.converters.mil.mil import Block
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, Program
from coremltools.converters.mil.mil.block import is_current_opset_version_compatible_with
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import (
    _check_child_op_type,
    _check_var_scalar_value,
    block_context_manager,
)
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.symbolic import any_symbolic


@register_pass(namespace="common")
class fuse_squeeze_expand_dims(AbstractGraphPass):
    """
    Detect the pattern ``input-->squeeze-->expand_dims``, and fuse
    them into an ``identity`` op if ``squeeze`` and ``expand_dims`` cancel out each other.
    Note that, the ``identity`` can be further removed by ``noop_elimination``.

    .. code-block::

        Given:
            %x[3, 1, 4, 1]
            %1[3, 4] = squeeze(%x, axes=[1, 3])
            %2[3, 1, 4, 1] = expand_dims(%1, axes=[1, 3])
            %3 = op(%2)

        Result:
            %x[3, 1, 4, 1]
            %2[3, 1, 4, 1] = identity(%x)
            %3 = op(%2)
    """

    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self.fuse_squeeze_expand_dims_block(f)

    @block_context_manager
    def fuse_squeeze_expand_dims_block(self, block):
        fusion_occurred = False
        for op in list(block.operations):
            if op.enclosing_block is None:
                continue

            for b in op.blocks:
                block_changed = True
                while block_changed:
                    block_changed = self.fuse_squeeze_expand_dims_block(b)

            if len(op.blocks) > 0:
                continue

            squeeze_op = self._match_pattern(op)
            if squeeze_op is not None:
                if self._try_to_transform(squeeze_op, block):
                    fusion_occurred = True
        return fusion_occurred

    @staticmethod
    def _match_pattern(op):
        if op.op_type != "squeeze":
            return None
        if not _check_child_op_type(op, "expand_dims"):
            return None
        return op

    @staticmethod
    def _try_to_transform(op, block):
        expand_dims_op = op.outputs[0].child_ops[0]
        x = op.x
        out_var = expand_dims_op.outputs[0]
        if x.shape != out_var.shape:
            return False
        if op.outputs[0] in block.outputs:
            return False

        new_var = mb.identity(x=x, before_op=op)
        if op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=expand_dims_op,
            old_var=out_var,
            new_var=new_var,
        ):
            # Remove all the ops at once
            block.remove_ops([op, expand_dims_op])
            return True
        return False


@register_pass(namespace="common")
class expand_high_rank_reshape_and_transpose(AbstractGraphPass):
    """
    Detect the pattern ``reshape_1-->transpose-->reshape_2``, where ``reshape_1`` has
    an output tensor with ``rank >= 6``, and ``reshape_2`` produces a tensor with ``rank <= 5``.

    In general, we can expand this pattern into a sequence of rank 4 ``reshape`` and ``transpose`` ops,
    which is supported by the Core ML runtime.

    .. code-block::

        Given:
            %1 = reshape(%x, shape=(d1, d2, d3, d4, ..., dn))
            %2 = transpose(%1, perm=(p1, p2, ..., pn))
            %3 = reshape(%2, shape=(o1, o2, o3, o4, o5))

        Result:
            %t1 = reshape(%x, shape=(y11, y12, y13, y14))
            %h1 = transpose(%t1, perm=[0, 2, 1, 3])
            %t2 = reshape(%h1, shape=(y21, y22, y23, 214))
            %h2 = transpose(%t2, perm=[0, 2, 1, 3])
            ....
            %hn = transpose(%tn, perm=[0, 2, 1, 3])
            %3 = reshape(%hn, shape=(o1, o2, o3, o4, o5))
    """
    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self.expand_high_rank_reshape_and_transpose_block(f)

    @staticmethod
    def _match_pattern(op):
        # We are detecting the
        # reshape(>= rank 6) -> transpose -> reshape(<= rank 5) pattern
        ops = [op]
        if op.op_type != "reshape":
            return None
        if op.outputs[0].rank <= 5:
            return None
        if any_symbolic(op.outputs[0].shape):
            return None

        if not _check_child_op_type(op, "transpose"):
            return None
        transpose_op = op.outputs[0].child_ops[0]
        ops.append(transpose_op)

        if not _check_child_op_type(transpose_op, "reshape"):
            return None
        reshape_op = transpose_op.outputs[0].child_ops[0]
        ops.append(reshape_op)
        if reshape_op.outputs[0].rank >= 6:
            return None

        for candidate_op in ops[:-1]:
            if candidate_op.outputs[0] in op.enclosing_block.outputs:
                return None
        return ops

    @staticmethod
    def _try_to_transform(ops, block):
        def _get_prod(start, end, arr, skip_indices):
            res = 1
            for i in range(start, end):
                if i in skip_indices:
                    continue
                res *= arr[i]
            return res

        reshape_op, transpose_op, last_reshape_op = ops[0], ops[1], ops[2]
        original_shape = reshape_op.outputs[0].shape
        original_perm = transpose_op.perm.val.tolist()

        # Group the consecutive axes in the perm, sometimes this could directly lower the
        # rank under 6.
        #
        # For instance:
        #
        # reshape = mb.reshape(x=x, shape=[1, 2, 3, 4, 5, 6])
        # transpose = mb.transpose(x=reshape, perm=[4, 5, 3, 2, 0, 1])
        # output = mb.reshape(x=transpose, shape=[6, 20, 6])
        #
        # Have 4 groups of axes: [4, 5], [3], [2], [0, 1]
        # We can transform the ops to
        #
        # new_reshape = mb.reshape(x=x, shape=[1*2, 3, 4, 5*6])
        # new_transpose = mb.transpose(x=reshape, perm=[3, 2, 1, 0])
        # output = mb.reshape(x=new_transpose, shape=[6, 20, 6])
        #
        # Note that, the output of new_transpose have different rank than transpose,
        # however, they have the same data layout, so the final output is still unchanged.
        group_axes = []
        i = 0
        res = []
        for i in range(len(original_perm)):
            if i > 0 and original_perm[i] == original_perm[i-1] + 1:
                res.append(original_perm[i])
            else:
                if len(res) > 0:
                    group_axes.append(res)
                res = [original_perm[i]]
            if i == len(original_perm) - 1:
                group_axes.append(res)

        group_shape = []
        for axes in group_axes:
            start, end = axes[0], axes[-1] + 1
            group_shape.append(_get_prod(start, end, original_shape, set()))

        start_group_axis = [axes[0] for axes in group_axes]
        group_axis_order = np.argsort(start_group_axis)
        shape = np.array(group_shape)[group_axis_order].tolist()

        sorted_start_group_axis = np.sort(start_group_axis).tolist()
        perm = [sorted_start_group_axis.index(i) for i in start_group_axis]

        rank = len(perm)
        x = reshape_op.x

        if rank < 6:
            # If the intermediate tensors have rank < 6,
            # we can directly use them to replace the original pattern
            x = mb.reshape(x=x, shape=shape, before_op=reshape_op)
            x = mb.transpose(x=x, perm=perm, before_op=reshape_op)

        else:
            # Otherwise, we need to expand the rank-N tensor into N reshape, and N transpose ops.
            # Note that all intrermediate tensors have rank 4.
            #
            # The algorithm is as followed:
            #
            # reshape shape: [d_1, d_2, ..., d_n]
            # transpose perm: [p_1, p_2, ..., p_n]
            #
            # reshape to [1, d_1*d_2*...*d_(p_1-1), d_(p_1), d_(p_1+1)*...*d_n]
            # transpose to [1, d_(p_1), d_1*d_2*...*d_(p_1-1), d_(p_1+1)*...*d_n]
            #
            # reshape to [d_(p_1), d_1*d_2*...*d_(p_2-1), d_(p_2), d_(p_2+1)*...*d_n]
            # transpose to [d_(p_1), d_(p_2), d_1*d_2*...*d_(p_2-1), d_(p_2+1)*...*d_n]
            #
            # reshape to [d_(p_1)*d_(p_2), d_1*d_2*...*d_(p_3-1), d_(p_3), d_(p_3+1)*...*d_n]
            # ....
            # so on and so forth
            leading_dim = 1
            memo = set()
            for i in range(rank):
                axis = perm[i]
                dim = shape[axis]
                memo.add(axis)
                reshape_shape = [
                    leading_dim,
                    _get_prod(0, axis, shape, memo),
                    dim,
                    _get_prod(axis + 1, rank, shape, memo)
                ]
                x = mb.reshape(x=x, shape=reshape_shape, before_op=reshape_op)
                x = mb.transpose(x=x, perm=[0, 2, 1, 3], before_op=reshape_op)
                leading_dim *= dim


        x = mb.reshape(x=x, shape=last_reshape_op.shape.val, before_op=reshape_op)
        if reshape_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=reshape_op, old_var=last_reshape_op.outputs[0], new_var=x,
        ):
            # Remove all the ops at once
            block.remove_ops(ops)
            return True
        return False

    @block_context_manager
    def expand_high_rank_reshape_and_transpose_block(self, block):
        fusion_occurred = False
        for op in list(block.operations):
            if op.enclosing_block is None:
                continue

            for b in op.blocks:
                block_changed = True
                while block_changed:
                    block_changed = self.expand_high_rank_reshape_and_transpose_block(b)
            if len(op.blocks) > 0:
                continue

            ops = self._match_pattern(op)
            if ops is not None:
                if self._try_to_transform(ops, block):
                    fusion_occurred = True
        return fusion_occurred

@register_pass(namespace="common")
class concat_to_pixel_shuffle(AbstractGraphPass):
    """
    Identify nested, interleaved ``concat`` ops which can be replaced by a single ``concat`` and a `pixel shuffle` layer.

    This pattern occurs with the faster up-convolution from the FCRN model (Laina et al., 2016).

    .. code-block::

        # Before the concat_to_pixel_shuffle pass.
        input(N, C, H, W) -------------------
                                            |
                                            v
        input(N, C, H, W) -----> concat(axis=2, interleave=True) -----> concat(axis=3, interleave=True) ----> output
                                                                                    ^
                                                                                    |
        input(N, C, H, W) -----> concat(axis=2, interleave=True) --------------------
                    |                       ^
                    |                       |
        input(N, C, H, W) -------------------


        # After the concat_to_pixel_shuffle pass.
        input(N, C, H, W) ---------------
                                        |
                                        v
        input(N, C, H, W) -----> concat(axis=1, interleave=True) -----> pixel_shuffle(upscale_factor=2) ----> output
                                        ^
                                        |
        input(N, C, H, W) --------------|
                                        |
                                        |
        input(N, C, H, W) ---------------
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._concat_to_pixel_shuffle_block(f)

    @staticmethod
    def _match_pattern(op):

        # Identify if this is an op we can transform
        if op.op_type != "concat":
            return None

        w_concat = op
        if w_concat.inputs["values"][0].rank != 4:
            return None

        if w_concat.inputs["axis"].val != 3:
            return None
        if not w_concat.inputs["interleave"].val:
            return None

        inputs = list(w_concat.inputs["values"])
        if len(inputs) != 2:
            return None

        if not inputs[0].op or not inputs[1].op:
            return None

        if inputs[0].op.op_type != "concat" or inputs[1].op.op_type != "concat":
            return None

        h_concat_0 = inputs[0].op
        if not h_concat_0.inputs["interleave"].val:
            return None

        h_concat_0_inputs = list(h_concat_0.inputs["values"])
        if len(h_concat_0_inputs) != 2:
            return None

        h_concat_1 = inputs[1].op
        if not h_concat_1.inputs["interleave"].val:
            return None

        h_concat_1_inputs = list(h_concat_1.inputs["values"])
        if len(h_concat_1_inputs) != 2:
            return None

        if h_concat_0.inputs["axis"].val != 2 or h_concat_1.inputs["axis"].val != 2:
            return None

        return w_concat, h_concat_0, h_concat_1

    @staticmethod
    def _replace_ops(block, w_concat, h_concat_0, h_concat_1):

        h_concat_0_inputs = list(h_concat_0.inputs["values"])
        h_concat_1_inputs = list(h_concat_1.inputs["values"])

        all_inputs = [
            h_concat_0_inputs[0],
            h_concat_1_inputs[0],
            h_concat_0_inputs[1],
            h_concat_1_inputs[1],
        ]

        # Concatenate all 4 inputs on the channel axis
        x = mb.concat(values=all_inputs, axis=1, before_op=h_concat_0, interleave=True)
        # Shuffle into place
        x = mb.pixel_shuffle(x=x, upscale_factor=2, before_op=h_concat_0)

        w_concat.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=h_concat_0, old_var=w_concat.outputs[0], new_var=x
        )

        block.remove_ops([w_concat, h_concat_0, h_concat_1])

    @block_context_manager
    def _concat_to_pixel_shuffle_block(self, block):
        for op in list(block.operations):
            layers = self._match_pattern(op)
            if layers:
                self._replace_ops(block, layers[0], layers[1], layers[2])


@register_pass(namespace="common")
class detect_concat_interleave(AbstractGraphPass):
    """
    Detect the pattern ``concat-->reshape--->transpose--->reshape``, where ``concat`` is
    along the channel axis ``(axis=-3)``, and map this pattern to the ``concat`` with ``interleave`` op.

    This pattern occurs, for example, in the ``shufflenet`` model in ``torchvision``.

    .. code-block::

        Given:
            %3 = concat(%1.a, %1.b, ..., axis=-3, interleave=False) #shape = (B, n*C, H, W)
            %4 = reshape(%3) #shape = (B, n, C, H, W)
            %5 = transpose(%4, perm=[0, 2, 1, 3, 4]) # shape = (B, C, n, H, W)
            %6 = reshape(%5) # shape = (B, C*n, H, W)

        Result:
            %6 = concat(%1.a, %1.b, ..., axis=-3, interleave=True)
    """

    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self._fuse_concat_interleave(f)

    @staticmethod
    def _match_pattern(op):
        if op.outputs[0] in op.enclosing_block.outputs:
            return None

        if op.op_type == "concat":
            if op.interleave.val:
                return None

            # check that axis is -3 and rank is 4
            rank = op.values[0].rank
            if rank != 4:
                return None
            axis = op.axis.val
            if axis > 0:
                axis = axis - rank
            if axis != -3:
                return None

            # check that all inputs to concat have fully defined shapes
            for in_ in op.values:
                if any_symbolic(in_.shape):
                    return None

            # check that all inputs to concat have the same shape
            inshape = list(op.values[0].shape)
            for v in op.values[1:]:
                for i in range(rank):
                    if inshape[i] != v.shape[i]:
                        return None

            # check that this concat is connected to exactly 1 reshape op
            child_ops = list(op.outputs[0].child_ops)
            if len(child_ops) == 1:
                if list(child_ops)[0].op_type == "reshape":
                    return op
        return None

    @staticmethod
    def _try_to_transform(concat_op, add_op, block):
        all_ops = [concat_op]
        B, C, H, W = list(concat_op.values[0].shape)
        n = len(concat_op.values)

        # check that reshape shapes the input to (B, n, C, H, W)
        reshape_op1 = concat_op.outputs[0].child_ops[0]
        reshape_shape1 = reshape_op1.shape.val
        if reshape_shape1 is None:
            return False
        if not isinstance(reshape_shape1, np.ndarray):
            return False
        reshape_shape1 = list(reshape_shape1)
        if reshape_shape1 != [B, n, C, H, W]:
            return False
        all_ops.append(reshape_op1)

        # check that after reshape is a transpose op with perm=[0, 2, 1, 3, 4]
        if len(list(reshape_op1.outputs[0].child_ops)) != 1:
            return False
        transpose_op = list(reshape_op1.outputs[0].child_ops)[0]
        if transpose_op.op_type != "transpose":
            return False
        perm = transpose_op.perm.val
        if perm is None:
            return
        if list(perm) != [0, 2, 1, 3, 4]:
            return False
        all_ops.append(transpose_op)

        # check that after transpose is another reshape with [B, . , H, W]
        if len(list(transpose_op.outputs[0].child_ops)) != 1:
            return False
        reshape_op2 = list(transpose_op.outputs[0].child_ops)[0]
        if reshape_op2.op_type != "reshape":
            return False
        reshape_shape2 = reshape_op2.shape.val
        if reshape_shape2 is None:
            return False
        if not isinstance(reshape_shape2, np.ndarray):
            return False
        reshape_shape2 = list(reshape_shape2)
        if len(reshape_shape2) != 4:
            return False
        if [reshape_shape2[0], reshape_shape2[-2], reshape_shape2[-1]] != [B, H, W]:
            return False
        all_ops.append(reshape_op2)

        # check that none of the op in this pattern is connected to the output
        # (except the last mul op)
        for i, op in enumerate(all_ops):
            if i == len(all_ops) - 1:
                continue
            for out in op.outputs:
                if out in block.outputs:
                    return False

        # add a new concat op
        out_name = reshape_op2.outputs[0].name
        x = mb.concat(
            values=concat_op.values,
            axis=concat_op.axis.val,
            interleave=True,
            name=out_name,
            before_op=concat_op,
        )

        reshape_op2.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=reshape_op2, old_var=reshape_op2.outputs[0], new_var=x
        )

        # Remove all the ops at once
        block.remove_ops(all_ops)
        return True

    @block_context_manager
    def _fuse_concat_interleave(self, block):
        fusion_occurred = False
        for op in list(block.operations):
            if op.enclosing_block is None:
                continue

            for b in op.blocks:
                block_changed = True
                while block_changed:
                    block_changed = self._fuse_concat_interleave(b)
            if len(op.blocks) > 0:
                continue

            concat_op = self._match_pattern(op)
            if concat_op is not None:
                if self._try_to_transform(op, concat_op, block):
                    fusion_occurred = True
        return fusion_occurred


@register_pass(namespace="common")
class fuse_onehot_matmul_to_gather(AbstractGraphPass):
    """
    Detect if ``onehot (axis=-1, on_value=1, off_value=0)`` is followed by a ``matmul`` op (no bias).
    If so, they can be replaced by a ``gather`` op.

    .. code-block::

        Input:
            %2 = one_hot(%1, on_value=1, off_value=0, axis=-1)
            %3 = const() # rank 2
            %4  = matmul(%2, %3)

        Output:
            %4 = gather(%3, %2, axis=0)
    """

    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self._fuse_onehot_matmul_to_gather_block(f)

    @staticmethod
    def _try_to_transform(onehot_op, block):
        root_var = onehot_op.indices

        # check that the output of the onehot op is not a block output
        if onehot_op.outputs[0] in block.outputs:
            return False

        # check that onehot op has axis=-1, on_value=1 and off_value=0
        # and constant one_hot_vector_size
        axis = onehot_op.axis.val
        if axis is None:
            return False
        if onehot_op.indices.shape is None:
            return False
        rank = len(onehot_op.indices.shape)
        if axis >= 0:
            axis -= rank
        if axis != -1:
            return False
        if not _check_var_scalar_value(onehot_op.on_value, 1):
            return False
        if not _check_var_scalar_value(onehot_op.off_value, 0):
            return False
        if onehot_op.one_hot_vector_size.val is None:
            return False

        # checks for the following matmul op
        if not _check_child_op_type(onehot_op, "matmul"):
            return False
        matmul_op = list(onehot_op.outputs[0].child_ops)[0]
        if matmul_op.x != onehot_op.outputs[0]:
            return False
        if matmul_op.transpose_x.val or matmul_op.transpose_y.val:
            return False
        W_var = matmul_op.y
        if W_var.val is None:
            return False
        if len(W_var.val.shape) != 2:
            return False

        # remove onehot and matmul and replace with gather op
        if is_current_opset_version_compatible_with(AvailableTarget.iOS17):
            # IOS17 `gather` requires non-negative indices.
            root_var = mb.select(
                cond=mb.greater_equal(x=root_var, y=0, before_op=matmul_op),
                a=root_var,
                b=mb.add(
                    x=root_var,
                    y=value_at(mb.shape(x=W_var, before_op=matmul_op), 0, before_op=matmul_op),
                    before_op=matmul_op,
                ),
                before_op=matmul_op,
            )
        x = mb.gather(
            x=W_var, indices=root_var, axis=0, name=matmul_op.outputs[0].name, before_op=matmul_op
        )

        matmul_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=matmul_op, old_var=matmul_op.outputs[0], new_var=x
        )
        # Remove all the ops at once
        block.remove_ops([onehot_op, matmul_op])
        return True

    @block_context_manager
    def _fuse_onehot_matmul_to_gather_block(self, block):
        fusion_occurred = False
        for op in list(block.operations):
            if op.enclosing_block is None:
                continue

            for b in op.blocks:
                block_changed = True
                while block_changed:
                    block_changed = self._fuse_onehot_matmul_to_gather_block(b)
            if len(op.blocks) > 0:
                # This op can't be pow
                continue

            # start pattern match if one_hot op is encountered
            if op.op_type == "one_hot":
                if self._try_to_transform(op, block):
                    fusion_occurred = True
        return fusion_occurred


@register_pass(namespace="common")
class replace_stack_reshape(AbstractGraphPass):
    """
    A stack followed by a reshape layer can be replaced by a ``concat`` if the reshape
    simply removes the new axis and doubles the size of one of the axes next to it.

    If the new axis is reshaped to the "right" (that is, the axis just after it is
    doubled), then we can use a ``concat``. If it is reshaped to the "left" (the axis
    just before it is doubled), then the ``concat`` needs to set the ``interleaved`` flag.

    Examples:

    .. code-block::

        Given:
            %1 = tensor(1, 5, 3, 4)
            %2 = tensor(1, 5, 3, 4)
            %3 = stack((%1,%2), axis=2) # shape = (1, 5, 2, 3, 4)
            %4 = reshape(%3, shape=[1, 10, 3, 4])

        Result:
            %1 = tensor(1, 5, 3, 4)
            %2 = tensor(1, 5, 3, 4)
            %4 = concat((%1,%2), axis=1, interleave=True) # shape = (1, 10, 3, 4)

        Given:
            %1 = tensor(1, 5, 3, 4)
            %2 = tensor(1, 5, 3, 4)
            %3 = stack((%1, %2), axis=1) # shape = (1, 2, 5, 3, 4)
            %4 = reshape(%3, shape=[1, 10, 3, 4])

        Result:
            %1 = tensor(1, 5, 3, 4)
            %2 = tensor(1, 5, 3, 4)
            %4 = concat((%1, %2), axis = 1) # shape = (1, 10, 3, 4)
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._replace_stack_reshape_block(f)

    @staticmethod
    def _match_operation(stack_op):

        # Identify if this is an op we can transform
        if stack_op.op_type != "stack":
            return None, None

        child_ops = stack_op.outputs[0].child_ops
        if len(child_ops) != 1:
            return None, None

        if child_ops[0].op_type != "reshape":
            return None, None

        stack_axis = stack_op.inputs["axis"]
        if not stack_axis:
            return None, None
        stack_axis_val = stack_axis.val

        reshape_op = child_ops[0]

        # Now, op is a stack op followed by a reshape op
        # So we need to check that the stack really gets eliminated
        stack_output_rank = len(stack_op.outputs[0].shape)
        reshape_output_rank = len(reshape_op.outputs[0].shape)

        if stack_output_rank != (reshape_output_rank + 1):
            return None, None

        # Compare the input to stack to the output from reshape
        # These shapes should differ in either the stack_axis_val place (by a factor of 2),
        # or in the stack_axis_val-1 place by the same factor
        input_shape = list(stack_op.inputs["values"][0].shape)
        concat_axis = [
            idx
            for idx, (x, y) in enumerate(zip(input_shape, reshape_op.outputs[0].shape))
            if x != y
        ]
        if len(concat_axis) != 1:
            return None, None

        concat_axis = concat_axis[0]

        if input_shape[concat_axis] * 2 != reshape_op.outputs[0].shape[concat_axis]:
            return None, None

        if concat_axis != stack_axis_val and concat_axis != stack_axis_val - 1:
            return None, None

        return stack_op, reshape_op

    @staticmethod
    def _replace_stack_reshape_ops(block, stack_op, reshape_op):

        stack_axis = stack_op.inputs["axis"]
        if not stack_axis:
            return None, None
        stack_axis_val = stack_axis.val

        input_shape = list(stack_op.outputs[0].shape)
        input_shape.pop(stack_axis_val)

        concat_axis = [
            idx
            for idx, (x, y) in enumerate(zip(input_shape, reshape_op.outputs[0].shape))
            if x != y
        ]
        if len(concat_axis) != 1:
            return
        concat_axis = concat_axis[0]

        interleave = concat_axis == stack_axis_val - 1

        x = mb.concat(
            values=stack_op.values, axis=concat_axis, before_op=stack_op, interleave=interleave
        )

        reshape_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=stack_op, old_var=reshape_op.outputs[0], new_var=x
        )
        block.remove_ops([stack_op, reshape_op])

    @block_context_manager
    def _replace_stack_reshape_block(self, block):
        for op in list(block.operations):

            stack_op, reshape_op = self._match_operation(op)

            if stack_op:
                self._replace_stack_reshape_ops(block, stack_op, reshape_op)


@register_pass(namespace="common")
class use_reflection_padding(AbstractGraphPass):
    """
    Identify a reflection padding layer composed out of `slices` and `concats`.

    .. code-block::

        Input graph:

                ------------------------------------------------------------------------------------- |
                |                                                                                     v
        input(1, 2, 6, 8) ------> slice_by_index(begin=[0, 0, 0, 1], end=[0, 0, 0, 2]) -----> concat(axis=3) ---> out(1, 2, 6, 10)
                |                                                                                     ^
                ----------------> slice_by_index(begin=[0, 0, 0, -2], end=[0, 0, 0, -1]) -------------|


        Output graph:

        input(1, 2, 6, 8) -----0> pad(mode=reflect, size=[0, 0, 1, 1]) -----> out(1, 2, 6, 10)

    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._reflection_padding_block(f)

    @staticmethod
    def _match_pattern(concat_op, block):
        if concat_op.op_type != "concat":
            return False

        concat_inputs = list(concat_op.inputs["values"])
        # There need to be an odd number of inputs, and at least one model has a concat input of
        # length 1
        if len(concat_inputs) % 2 != 1 or len(concat_inputs) == 1:
            return False

        # The original input will need to be in the middle of the concatenated inputs
        original_input = concat_inputs[len(concat_inputs) // 2]

        axis = None
        slice_ops_out = []
        end_mask = None
        begin_index = len(concat_inputs) // 2

        for slice_op in concat_inputs:

            # one of the concat inputs is the original input (to the slices)
            if slice_op == original_input:
                # We'll now start checking indices from the end
                begin_index = begin_index - 2
                continue

            slice_op = slice_op.op
            if not slice_op:
                return False

            if slice_op.op_type != "slice_by_index":
                return False

            # check that the input to slice op is the original input
            if slice_op.inputs["x"] != original_input:
                return False

            # If the slice is an output
            if slice_op.outputs[0] in block.outputs:
                return False

            if end_mask is None:
                end_mask = slice_op.inputs["end_mask"].val
                axis = list(end_mask).index(False, 0, len(end_mask))

            if end_mask is None:
                return False

            if axis != list(end_mask).index(False, 0, len(end_mask)):
                return False

            # Check that we're only taking a slice of size 1
            end = slice_op.inputs["end"].val
            begin = slice_op.inputs["begin"].val
            if end[axis] - begin[axis] != 1:
                return False

            input_shape = original_input.shape
            # Check that the slices are in order
            if begin[axis] != begin_index and begin[axis] != begin_index + input_shape[axis]:
                return False
            begin_index = begin_index - 1

            slice_ops_out.append(slice_op)

        if axis is None:
            return False

        return use_reflection_padding._replace_ops(
            block, concat_op, slice_ops_out, axis - len(end_mask)
        )

    @staticmethod
    def _replace_ops(block, concat_op, slice_ops, axis):

        pad_size = len(slice_ops) // 2
        if axis == -1:
            pad = [pad_size, pad_size]
        elif axis == -2:
            pad = [pad_size, pad_size, 0, 0]
        else:
            return False

        x = mb.pad(x=slice_ops[0].inputs["x"], pad=pad, mode="reflect", before_op=concat_op)
        concat_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=concat_op, old_var=concat_op.outputs[0], new_var=x
        )

        block.remove_ops([concat_op] + slice_ops)
        return True

    @block_context_manager
    def _reflection_padding_block(self, block):
        for op in list(block.operations):
            self._match_pattern(op, block)


@register_pass(namespace="common")
class fuse_stack_split(AbstractGraphPass):
    """
    Detect the pattern ``inputs -> stack -> split -> squeeze`` and fuse them into an ``identity`` if the pattern
    cancel out each out.
    Note that, the ``identity`` can be further removed by ``noop_elimination``.

    .. code-block::

        Input:
            %4 = stack([%1, %2, %3], axis=0)
            %5, %6, %7 = split(%4, axis=0)
            %8 = squeeze(%5, axes=[0])
            %9 = squeeze(%6, axes=[0])
            %10 = squeeze(%7, axes=[0])

        Output:
            %8 = identity(%1)
            %9 = identity(%2)
            %10 = identity(%3)
    """

    def apply(self, prog: Program) -> None:
        for f in prog.functions.values():
            self.fuse_stack_split_block(f)

    @staticmethod
    def _try_to_transform(block: Block, stack_op: Operation) -> None:
        def _convert_axis_to_positive(axis, rank):
            if axis < 0:
                return axis + rank + 1
            return axis

        def _try_fuse_a_branch(values, rank, axis, split_op):
            ops_to_remove = [split_op]

            # check if the split op have the correct config
            if _convert_axis_to_positive(split_op.axis.val, rank) != axis:
                return

            split_sizes = split_op.split_sizes
            if split_sizes is not None:
                if split_sizes.val.tolist() != [1] * len(values):
                    return

            num_splits = split_op.num_splits
            if num_splits is not None:
                if num_splits.val != len(values):
                    return

            ops_to_remove.append(split_op)

            # check if any of the output var of the stack / split op is the block output
            for val in ops_to_remove:
                for v in val.outputs:
                    if v in block.outputs:
                        return

            # check if the outputs of the split op feed only into squeeze
            split_out_vars = split_op.outputs
            vars_to_replace = []

            for val in split_out_vars:
                if len(val.child_ops) != 1 or val.child_ops[0].op_type != "squeeze":
                    should_fuse = False

                squeeze_op = val.child_ops[0]
                if [
                    _convert_axis_to_positive(val, rank) for val in squeeze_op.axes.val.tolist()
                ] != [axis]:
                    return

                vars_to_replace.append(squeeze_op.outputs[0])
                ops_to_remove.append(squeeze_op)

            for _input, _var in zip(values, vars_to_replace):
                new_var = mb.identity(x=_input, before_op=squeeze_op)
                block.replace_uses_of_var_after_op(
                    anchor_op=squeeze_op,
                    old_var=_var,
                    new_var=new_var,
                )
            block.remove_ops(ops_to_remove)

        if stack_op.outputs[0] in block.outputs:
            return

        # get the params of the stack op
        values = stack_op.values
        rank = values[0].rank
        axis = _convert_axis_to_positive(stack_op.axis.val, rank)

        # go through the split child ops
        for val in list(stack_op.outputs[0].child_ops):
            if val.op_type == "split":
                _try_fuse_a_branch(values, rank, axis, val)

        # remove the stack op if its output no longer consumed by any ops
        if len(stack_op.outputs[0].child_ops) == 0:
            block.remove_ops([stack_op])


    @block_context_manager
    def fuse_stack_split_block(self, block: Block) -> None:
        for op in list(block.operations):
            for b in op.blocks:
                self.fuse_stack_split_block(b)

            if op.op_type != "stack":
                continue

            self._try_to_transform(block, op)
