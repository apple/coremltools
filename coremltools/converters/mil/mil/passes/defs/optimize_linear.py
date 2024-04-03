#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import Block
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, Program, Var
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="common")
class fuse_linear_bias(AbstractGraphPass):
    """
    Convert ``linear + add/sub`` to a single ``linear`` by updating the weight and bias of the ``linear`` layer.

    .. code-block::

        Example 1:
            Original:
                %4 = linear(x=%1, weight=%2, bias=%3) # %2 is a rank-2 const tensor (weight)
                                                      # %3 is a rank-1 const tensor (bias)
                ...
                %6 = add(x=%4, y=%5) # %5 is a const tensor with same shape as %3

            Result:
                %8 = linear(x=%1, weight=%2, bias=%7) # where %7 is a new const tensor with value
                                                      # %7 = %3 + %6

        Example 2:
            Original:
                %4 = linear(x=%1, weight=%2, bias=%3) # %2 is a rank-2 const tensor (weight)
                                                      # %3 is a rank-1 const tensor (bias)
                ...
                %6 = sub(x=%5, y=%4) # %5 is a const tensor with a broacasable shape with %3.
                                       i.e. if %3 has shape (Dout), %5 could be (1, Dout).

            Result:
                %9 = linear(x=%1, weight=%7, bias=%8) # where %7 is a new const tensor with value %7 = -%2
                                                      # %8 = %5 - %3
    """

    def apply(self, prog: Program):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self._fuse_linear_bias_block(f)

    @staticmethod
    def _try_to_transform(linear_op, add_or_sub_op, block):

        if add_or_sub_op.x.val is None and add_or_sub_op.y.val is None:
            return False

        is_sub = add_or_sub_op.op_type == "sub"
        is_first_input = add_or_sub_op.x == linear_op.outputs[0]

        # Return if weight or bias are missing values
        if linear_op.weight.val is None or linear_op.bias.val is None:
            return False

        # compute the new bias
        linear_bias = linear_op.bias.val
        bias = add_or_sub_op.y.val if is_first_input else add_or_sub_op.x.val

        # check if the shape is broadcasable
        if np.prod(linear_bias.shape) != np.prod(bias.shape):
            return False
        Dout = linear_bias.shape[0]
        if bias.shape[-1] != Dout:
            return False
        bias = np.reshape(bias, (Dout,))

        if is_sub:
            if is_first_input:
                bias = -bias
            else:
                linear_bias = -linear_bias

        new_bias = linear_bias + bias

        # compute the new weight
        if is_sub and not is_first_input:
            new_weight = -linear_op.weight.val
        else:
            new_weight = linear_op.weight.val

        # create a new linear op with the new weight, bias value, copying rest of the attributes
        out_name = add_or_sub_op.outputs[0].name
        linear_kargs = {
            "weight": new_weight,
            "bias": new_bias,
            "name": out_name,
            "before_op": linear_op,
        }

        for k, v in linear_op.inputs.items():
            if k in ["weight", "bias"]:
                continue
            linear_kargs[k] = v

        x = mb.linear(**linear_kargs)

        if add_or_sub_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=add_or_sub_op,
            old_var=add_or_sub_op.outputs[0],
            new_var=x,
        ):
            add_or_sub_op.enclosing_block.remove_ops([linear_op, add_or_sub_op])
            return True
        return False

    @block_context_manager
    def _fuse_linear_bias_block(self, block):
        def _find_candicate_op(op):
            if op.op_type != "linear":
                return None
            # abort fusion if op output is also a block output
            if op.outputs[0] in op.enclosing_block.outputs:
                return None
            # find add/sub op
            child_ops = op.outputs[0].child_ops
            if len(child_ops) == 1:
                op_candidate = list(child_ops)[0]
                if op_candidate.op_type in ["add", "sub"]:
                    return op_candidate

        fusion_occurred = False
        for op in list(block.operations):
            if op.enclosing_block is None:
                continue

            for b in op.blocks:
                block_changed = True
                while block_changed:
                    block_changed = self._fuse_linear_bias_block(b)
            if len(op.blocks) > 0:
                # This op can't be conv or conv_transpose
                continue

            add_or_sub_op = _find_candicate_op(op)
            if add_or_sub_op is not None:
                if self._try_to_transform(op, add_or_sub_op, block):
                    fusion_occurred = True
        return fusion_occurred


@register_pass(namespace="common")
class fuse_matmul_weight_bias(AbstractGraphPass):
    """
    Convert ``matmul + add/sub`` to ``linear`` whenever possible.

    .. code-block::

        Given:
            %3 = matmul(x=%1, y=%2)  # %1 or %2 is const and rank 2 (weight)
            ...
            %5 = add(x=%3, y=%4) # %4 is const. add(x=%4, y=%3) is equivalent
                                 # sub is similar.

        Result:
            # assuming %2 above is const and rank 2
            %5 = linear(x=%1, weight=%2, bias=%4)
    """

    def apply(self, prog: Program):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self._fuse_matmul_weight_bias_block(f)

    @staticmethod
    def _find_candidate_op(op):
        _CHILD_OP_TYPES = ["add", "sub"]

        if op.op_type != "matmul":
            return None
        # find add
        child_ops = op.outputs[0].child_ops
        if len(child_ops) == 1:
            add_op_candidate = list(child_ops)[0]
            if add_op_candidate.op_type in _CHILD_OP_TYPES:
                return add_op_candidate

    @staticmethod
    def _transpose(v, before_op, name=None):
        """
        Transpose the last 2 dims.

        - ``v``: (Var, must be a tensor).
        - ``before_op``: (Operation) The op right before the newly added ``transpose`` op.
        - ``name``: Name for the ``transpose`` op if provided.
        """
        perm = list(range(v.rank))
        perm[-2], perm[-1] = perm[-1], perm[-2]

        if name is None:
            return mb.transpose(x=v, perm=perm, before_op=before_op)
        else:
            return mb.transpose(x=v, perm=perm, before_op=before_op, name=name)

    def _try_to_transform(self, matmul_op, add_op, block):
        if matmul_op.x.val is None and matmul_op.y.val is None:
            # This is a dynamic matmul.
            return False
        if add_op.x.val is None and add_op.y.val is None:
            # This is a dynamic add.
            return False

        x_is_weight = matmul_op.x.val is not None
        if x_is_weight:
            weight, linear_x = matmul_op.x, matmul_op.y
            transpose_weight = matmul_op.transpose_x.val
            transpose_x = matmul_op.transpose_y.val
        else:
            weight, linear_x = matmul_op.y, matmul_op.x
            transpose_weight = matmul_op.transpose_y.val
            transpose_x = matmul_op.transpose_x.val

        # We potentially are going to transpose the weight, so if the weight itself is not removable, we skip this path
        if len(weight.nonreplaceable_vars_upstream) > 0:
            return False

        if linear_x.rank < 2 or weight.rank != 2:
            # We don't support these cases yet.
            return False

        # For those weights which are the input for more than one op,
        # we don't do the fusion.
        # The reason is that it might cause memory explosion by adding
        # those weight as a numpy array in the inner product or
        # the batch_mat_mul kernel.
        if len(weight.child_ops) > 1:
            return False

        d_out = weight.shape[1] if not transpose_weight else weight.shape[0]
        bias = add_op.x.val if add_op.x.val is not None else add_op.y.val
        if len(bias.shape) > 1:
            if any([d != 1 for d in bias.shape[:-1]]):
                return  # cannot transform

            # squeeze leading dims of size 1
            bias = np.squeeze(bias)

        if len(bias.shape) != 1 or bias.shape[0] != d_out:
            return  # cannot transform

        if add_op.op_type == "sub":
            bias = -bias
        out_name = add_op.outputs[0].name

        if x_is_weight:
            # If transpose_x == transpose_weight == False:
            # w*x = (x^T w^T)^T = linear(x^T, w)^T
            x_transposed = (
                self._transpose(linear_x, before_op=matmul_op) if not transpose_x else linear_x
            )
            w_no_transpose = (
                weight if not transpose_weight else self._transpose(weight, before_op=matmul_op)
            )
            x = mb.linear(x=x_transposed, weight=w_no_transpose, bias=bias, before_op=matmul_op)
            x = self._transpose(x, before_op=matmul_op, name=out_name)
        else:
            # If transpose_x == transpose_weight == False
            # x*w = x*(w^T)^T = linear(x, w^T)
            x_no_transpose = (
                self._transpose(linear_x, before_op=matmul_op) if transpose_x else linear_x
            )
            w_transposed = (
                weight if transpose_weight else self._transpose(weight, before_op=matmul_op)
            )
            x = mb.linear(
                x=x_no_transpose,
                weight=w_transposed,
                bias=bias,
                before_op=matmul_op,
                name=out_name,
            )

        if add_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=add_op,
            old_var=add_op.outputs[0],
            new_var=x,
        ):
            add_op.enclosing_block.remove_ops([matmul_op, add_op])
            return True
        return False

    @block_context_manager
    def _fuse_matmul_weight_bias_block(self, block):
        fusion_occurred = False
        for op in list(block.operations):
            if op.enclosing_block is None:
                continue

            for b in op.blocks:
                block_changed = True
                while block_changed:
                    block_changed = self._fuse_matmul_weight_bias_block(b)
            if len(op.blocks) > 0:
                # This op can't be matmul
                continue

            add_op = self._find_candidate_op(op)

            if add_op is not None:
                if self._try_to_transform(op, add_op, block):
                    fusion_occurred = True
        return fusion_occurred


@register_pass(namespace="common")
class fuse_transpose_matmul(AbstractGraphPass):
    """
    Fuse ``transpose + matmul`` to ``matmul`` if possible,
    since ``matmul`` has args ``transpose_x`` and ``transpose_y`` to transpose last 2 dims

    .. code-block::

        Positive example:
            Input graph:
                transpose(x=x, perm=(1, 0)) -|
                                             |-> matmul(x=transposed_x, y=transposed_y)
                transpose(x=y, perm=(1, 0)) -|

            Output graph:
                matmul(x=x, y=y, transpose_x=True, transpose_y=True)

        Negative example:
            Input graph:
                transpose(x=x, perm=(1, 0, 2)) -|
                                                |-> matmul(x=transposed_x, y=transposed_y)
                transpose(x=y, perm=(1, 0, 2)) -|

            Output graph:
                Same to input graph, nothing changes
    """

    def apply(self, prog: Program) -> None:
        for f in prog.functions.values():
            self._fuse_transpose_matmul_block(f)

    @block_context_manager
    def _fuse_transpose_matmul_block(self, block: Block) -> None:
        # use shallow copy to hide changes on block.operations during the loop,
        # since we try fusion when loop to matmul, which will not affect downstream
        for op in list(block.operations):
            for b in op.blocks:
                self._fuse_transpose_matmul_block(b)

            if op.op_type == "matmul":
                self._try_fuse_transpose_matmul(op, block)

    @staticmethod
    def is_transposed_and_fusable_to_matmul(x: Var) -> bool:
        """
        1. check if x is transposed
        2. check if x is transposed in the last 2 dimensions,
           since the transpose arg in matmul only transposes the last 2 dimensions
        """

        # x is not transposed, False
        if x.op is None or x.op.op_type != "transpose":
            return False

        rank = x.rank
        # if transposing a rank < 2 tensor, it is a noop and will be elimianted by noop_elimination
        if rank < 2:
            return False

        # canonicalize the input permutation to compare with last-2-dim permutation below
        perm = x.op.perm.val
        perm[np.where(perm < 0)] += rank
        perm[-2:] -= rank

        # permuting only last 2 dims should look like (0, 1, ..., -1, -2)
        perm_only_last_2_dims = np.arange(rank)
        perm_only_last_2_dims[-2] = -1
        perm_only_last_2_dims[-1] = -2

        return np.all(perm == perm_only_last_2_dims)

    def _try_fuse_transpose_matmul(self, op: Operation, block: Block) -> None:
        assert op.op_type == "matmul"

        x = op.x
        y = op.y
        transpose_x = False if op.transpose_x is None else op.transpose_x.val
        transpose_y = False if op.transpose_y is None else op.transpose_y.val

        is_x_transposed_and_fusable_to_matmul = self.is_transposed_and_fusable_to_matmul(x)
        is_y_transposed_and_fusable_to_matmul = self.is_transposed_and_fusable_to_matmul(y)
        # if neither x nor y is transposed and fuseable with matmul, nothing we need to do
        if not is_x_transposed_and_fusable_to_matmul and not is_y_transposed_and_fusable_to_matmul:
            return

        if is_x_transposed_and_fusable_to_matmul:
            x = x.op.x
            transpose_x = not transpose_x
        if is_y_transposed_and_fusable_to_matmul:
            y = y.op.x
            transpose_y = not transpose_y

        fused_transpose_matmul = mb.matmul(
            x=x,
            y=y,
            transpose_x=transpose_x,
            transpose_y=transpose_y,
            before_op=op,
            name=op.name,
        )
        block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=fused_transpose_matmul,
        )
        op.remove_from_block()
