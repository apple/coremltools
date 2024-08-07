#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
from collections import defaultdict
from typing import List, Text, Tuple

import numpy as np

from coremltools import _logger as logger
from coremltools.converters.mil.mil import Block
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Operation
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import _check_child_op_type, block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.symbolic import any_symbolic
from coremltools.converters.mil.mil.types.type_mapping import (
    RangeTuple,
    builtin_to_range,
    builtin_to_resolution,
    string_to_builtin,
)
from coremltools.converters.mil.mil.var import Var


@register_pass(namespace="common")
class merge_consecutive_paddings(AbstractGraphPass):
    """
    Identify two consecutive ``pad`` layers which could be merged into a single ``pad`` layer.

    This is possible only if one of the following conditions is satisfied:

    - The paddings are "constant" and have the same ``constant_val``.
    - The paddings act along different axes.

    .. code-block::

        Input graph:
        input(1, 2, 6, 8) ------> pad([1, 1], mode='reflect) -----> pad([1, 1, 0, 0], mode='reflect') ---> out(1, 2, 8, 10)

        Output graph:
        input(1, 2, 6, 8) ------> pad([1, 1, 1, 1], mode='reflect) ---> out(1, 2, 8, 10)
    """

    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self._merge_padding_block(f)

    def _match_pattern(self, block, padding_op):

        if padding_op.op_type != "pad":
            return False

        if not _check_child_op_type(padding_op, "pad"):
            return False

        child_padding_op = list(padding_op.outputs[0].child_ops)[0]

        if padding_op.inputs["mode"].val != child_padding_op.inputs["mode"].val:
            return False

        # Ensure the paddings have the same length by prepending zeros to the shorter one
        first_pad = padding_op.inputs["pad"].val
        child_pad = child_padding_op.inputs["pad"].val
        if len(first_pad) > len(child_pad):
            child_pad = np.insert(child_pad, 0, [0] * (len(first_pad) - len(child_pad)))
        elif len(child_pad) > len(first_pad):
            first_pad = np.insert(first_pad, 0, [0] * (len(child_pad) - len(first_pad)))
        final_pad = child_pad + first_pad

        if padding_op.inputs["mode"].val == "constant":
            # if the padding is constant, then the values need to be equal
            if padding_op.inputs["constant_val"].val != child_padding_op.inputs["constant_val"].val:
                return False
        else:
            # if the padding is not constant, then we can't merge if both pads affected the same
            # side of the image
            if any(i != 0 and j != 0 for (i, j) in zip(first_pad, child_pad)):
                return False

        return self._replace_ops(block, padding_op, child_padding_op, final_pad)

    @staticmethod
    def _replace_ops(block, padding_op, child_padding_op, final_pad):
        mode = padding_op.inputs["mode"].val
        x = mb.pad(
            x=padding_op.inputs["x"],
            pad=final_pad,
            mode=mode,
            constant_val=padding_op.inputs["constant_val"].val,
            before_op=padding_op,
        )
        padding_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=padding_op, old_var=child_padding_op.outputs[0], new_var=x
        )
        block.remove_ops([padding_op, child_padding_op])
        return True

    @block_context_manager
    def _merge_padding_block(self, block):
        fusion_happens = False
        for op in list(block.operations):
            if op.enclosing_block is None:
                continue

            if self._match_pattern(block, op):
                fusion_happens = True
        return fusion_happens

@register_pass(namespace="common")
class merge_consecutive_transposes(AbstractGraphPass):
    """
    Identify consecutive 'transpose' layers which could be merged into a single 'transpose' layer.

    .. code-block::

        Input graph:
        input ------> transpose -----> 1 or more transpose layers ---> out

        Output graph:
        input ------> transpose ---> out
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._merge_transposes_in_block(f)

    def _match_and_replace_pattern(self, block, transpose_op):
        if not (transpose_op.op_type == "transpose" and _check_child_op_type(transpose_op, "transpose")):
            return False
        if transpose_op.outputs[0] in block.outputs:
            return False

        child_transpose_op = list(transpose_op.outputs[0].child_ops)[0]
        return self._replace_ops(block, transpose_op, child_transpose_op)

    @staticmethod
    def _replace_ops(block, transpose_op, child_transpose_op):
        perm = transpose_op.perm.val
        new_perm = [perm[i] for i in child_transpose_op.perm.val]
        x = mb.transpose(x=transpose_op.x, perm=new_perm, before_op=transpose_op)
        if transpose_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=child_transpose_op,
            old_var=child_transpose_op.outputs[0],
            new_var=x,
        ):
            block.remove_ops([transpose_op, child_transpose_op])
            return True
        return False

    @block_context_manager
    def _merge_transposes_in_block(self, block):
        def help_merge_transpose_ops(block):
            fusion_happens = False
            for op in list(block.operations):
                if op.enclosing_block is None:
                    continue
                if self._match_and_replace_pattern(block, op):
                    fusion_happens = True
            return fusion_happens

        block_changed = True
        while block_changed:
            block_changed = help_merge_transpose_ops(block)


@register_pass(namespace="common")
class merge_consecutive_relus(AbstractGraphPass):
    """
    Identify consecutive ``relu`` layers which could be merged into a single ``relu`` layer.

    .. code-block::

        Input graph:
        input ------> relu -----> 1 or more relu layers ---> out

        Output graph:
        input ------> relu ---> out
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._merge_relus_in_block(f)

    def _match_and_replace_pattern(self, block, relu_op):
        if not (relu_op.op_type == "relu" and _check_child_op_type(relu_op, "relu")):
            return False

        child_relu_op = list(relu_op.outputs[0].child_ops)[0]
        return self._replace_ops(block, relu_op, child_relu_op)

    @staticmethod
    def _replace_ops(block, relu_op, child_relu_op):
        if relu_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=child_relu_op, old_var=child_relu_op.outputs[0], new_var=relu_op.outputs[0]
        ):
            block.remove_ops([child_relu_op])
            return True
        return False

    @block_context_manager
    def _merge_relus_in_block(self, block):
        def help_merge_relu_ops(block):
            fusion_happens = False
            for op in list(block.operations):
                if op.enclosing_block is None:
                    continue
                if self._match_and_replace_pattern(block, op):
                    fusion_happens = True
            return fusion_happens

        block_changed = True
        while block_changed:
            block_changed = help_merge_relu_ops(block)


@register_pass(namespace="common")
class merge_consecutive_reshapes(AbstractGraphPass):
    """
    Identify consecutive ``reshape`` ops which could be merged into a single ``reshape``.

    .. code-block::

        Input graph:
        input -> reshape -> 1 or more reshapes -> output

        Output graph:
        input -> reshape -> output
    """

    # TODO (rdar://105227587): merge a tree of consecutive reshapes

    def apply(self, prog):
        for f in prog.functions.values():
            self._merge_consecutive_reshapes_block(f)

    @staticmethod
    def _match_pattern(reshape_op):
        """
        Given a ``reshape`` op,
        consider it as the head of a sequence of ``reshape`` ops, and
        then end the sequence at a non-removable ``reshape`` op.
        Return this sequence as a list.
        """
        res = []
        op = reshape_op

        while op.op_type == "reshape":
            res.append(op)

            # current reshape has 0 or 2+ child ops:
            # * no child: this is the end of graph
            # * 2+ children: only pattern of sequential reshape ops (1 child)
            #   is supported for now. For more general cases, please see TODO below
            if len(op.outputs[0].child_ops) != 1:
                break
            # current reshape output is a block output, so it is non-removable
            if op.outputs[0] in op.enclosing_block.outputs:
                break

            op = op.outputs[0].child_ops[0]

        return res

    @block_context_manager
    def _merge_consecutive_reshapes_block(self, block):
        @block_context_manager
        def help_merge_consecutive_reshapes_block(block):
            fusion_happens = False
            for op in list(block.operations):
                if op.enclosing_block is None:
                    continue

                for b in op.blocks:
                    block_changed = True
                    while block_changed:
                        block_changed = help_merge_consecutive_reshapes_block(b)
                # move on to the next op if this op is not reshape
                if op.op_type != "reshape":
                    continue

                reshape_ops = self._match_pattern(op)
                # merge the list of consecutive reshape ops
                if len(reshape_ops) > 1:
                    # create a new reshape op
                    reshape_out = mb.reshape(
                        x=reshape_ops[0].x,
                        shape=reshape_ops[-1].shape,
                        name=reshape_ops[-1].outputs[0].name,
                        before_op=reshape_ops[-1],
                    )
                    # replace the consecutive reshape ops with the new reshape op
                    reshape_ops[-1].enclosing_block.replace_uses_of_var_after_op(
                        anchor_op=reshape_ops[-1],
                        old_var=reshape_ops[-1].outputs[0],
                        new_var=reshape_out,
                    )
                    reshape_ops[-1].enclosing_block.remove_ops(reshape_ops)
                    fusion_happens = True

            return fusion_happens

        block_changed = True
        while block_changed:
            block_changed = help_merge_consecutive_reshapes_block(block)

class CastOptimizationNode:
    def __init__(self, op_type, match_criterion=None):
        """
        Parameters
        ----------

        param op_type : Type of an operation.
        param match_criterion : A callable function that matches a MIL op and returns a boolean.

        Examples
        --------

        .. sourcecode:: python

            CastOptimizationNode("mul"),
            CastOptimizationNode("round"),
            CastOptimizationNode("add", lambda op: op.y.val == 0),
            CastOptimizationNode("clip", lambda op: op.alpha.val == -128 and op.beta.val == 127),
            CastOptimizationNode("cast", lambda op: op.dtype.val == "int8"),
            CastOptimizationNode("cast", lambda op: op.dtype.val == "fp32"),

        """

        self.op_type = op_type
        if not match_criterion:
            match_criterion = lambda op: True

        self.match_criterion = match_criterion

@register_pass(namespace="common")
class cast_optimization(AbstractGraphPass):
    """
    This optimization pass performs the following:

    - Removes redundant ``cast`` op; that is, ``cast`` where source and destination tensors have same dtypes.
    - Fuses two consecutive `cast` ops if applicable, repeatedly.

    This is a non-algebraic translation which assumes that the upcasting doesn't change the user's intent.

    (1) Example for redundant ``cast`` op removal:
        .. code-block::

            Input graph:
            input(fp16) -> cast(dtype="fp16") -> relu -> out

            Output graph:
            input -> relu -> out

         The input and output tensors for the ``cast`` op are both with type of ``fp16``.
         Hence, it can be removed.

    (2) Example for two ``cast`` ops fusion:
        .. code-block::

            Input graph:
            input(int8) -> cast(dtype="fp16") -> cast(dtype="fp32") -> out

            Output graph:
            input(int8) -> cast(dtype="fp32") -> out

         The data range and resolution of the above graph are limited by the int8 input,
         so the fusion is allowed.

    (3) Negative example for two ``cast`` ops fusion:
        .. code-block::

            Input graph:
            input(fp32) -> cast(dtype="bool") -> cast(dtype="fp16") -> out

            Output graph:
            Same as input graph.

         The above two ``cast`` ops cannot be merged, since after the first cast,
         the resolution of the numerical output is downcasted to binary (``0, 1``).
         If we fuse them, the output would be in the range and resolution of ``fp16`` instead.

    (4) Another Negative example for two ``cast`` ops fusion:
        .. code-block::

            Input graph:
            input(int32) -> cast(dtype="int8") -> cast(dtype="uint8") -> out

            Output graph:
            Same as input graph.

         The above two ``cast`` ops cannot be merged, since in the original graph,
         by going through two casts, the output numerical range is capped to ``[0, 127]``.
         However, if two ``cast`` ops are reduced to 1 ``cast(dtype="uint8")``, the output
         numerical would in the range of ``[0, 255]``. The fusion would cause numerical
         issue for the numbers between ``[128, 255]``, which is prohibited.

    In general, two ``cast`` ops can be merged if the output data range and resolution is not affected.

    For more examples, please see the unittests that start with prefix ``TestCastOptimization`` in ``test_passes.py``.
    """

    _num_of_visited_ops = 0  # Testing purpose, making sure the algorithm performs in O(N)

    def apply(self, prog):
        self._num_of_visited_ops = 0
        for f in prog.functions.values():
            self._fuse_or_cancel_consecutive_casts_block_wrapper(f)

    def _propagate_range_resolution(self, in_dtype: type, dtype_chain: Tuple[type]):
        """
        Given an input type ``in_dtype``, and a chain of casting, return the resulting output data range and resolution.

        For example, ``in_dtype = fp32`` and ``dtype_chain = [int8, int32]``. This means an input data with type ``fp32``,
        is propagated through ``cast(dtype="int8")`` and ``cast(dtype="int32")`` in order.

        1. The input fp32 data range is ``[-3.4e+38, 3.4e+38]`` with resolution ``1e-06``.
        2. After the first ``cast(dtype="int8")`` downcast, the range becomes ``[-128, 127]`` with resolution ``1``.
        3. Even the ``int32`` has a larger range, the resulting range is still capped to ``[-128, 127]``.

        For the above example, this function returns range of ``[-128, 127]`` and resolution ``1``.
        """
        assert isinstance(dtype_chain, tuple)
        cur_range, cur_resolution = builtin_to_range(in_dtype), builtin_to_resolution(in_dtype)
        for v in dtype_chain:
            tmp_range, tmp_resolution = builtin_to_range(v), builtin_to_resolution(v)
            cur_range = RangeTuple(
                max(cur_range.low, tmp_range.low), min(cur_range.high, tmp_range.high)
            )
            cur_resolution = max(cur_resolution, tmp_resolution)
        return cur_range, cur_resolution

    def _is_cast_ops_fusable(self, cast_1: Operation, cast_2: Operation):
        """
        Check if two cast ops can be fused by verifying the consistency between the range and resolution before and after fusion.

        Take the same example shown in ``_propagate_range_resolution``:

            input(fp32) -> cast(dtype="int8") -> cast(dtype="int32")

        The original pattern has output range and resolution ``[-128, 127]``, ``1``.

        However, if the two ``cast`` ops are fused:

            input(fp32) -> cast(dtype="int32")

        The output range becomes the range of int32, which is not ``[-128, 127]``.
        As the result, the fusion is prohibited.
        """
        x_dtype, cast_1_dtype, cast_2_dtype = (
            cast_1.x.dtype,
            string_to_builtin(cast_1.dtype.val),
            string_to_builtin(cast_2.dtype.val),
        )

        ref_range, ref_resolution = self._propagate_range_resolution(
            x_dtype, (cast_1_dtype, cast_2_dtype)
        )
        out_range, out_resolution = self._propagate_range_resolution(x_dtype, (cast_2_dtype,))

        return out_range == ref_range and out_resolution == ref_resolution

    def _dup_if_affect_io(self, new_var: Var, old_var: Var, before_op: Operation):
        """
        We cannot replace old_var with new_var, if:
        1. old_var is a function output
        2. new_var is a function input
        Since the name of the function is going to be changed and become invalid.

        For this special corner case, we use an identity op to duplicate the new_var.
        """
        block_1 = before_op.enclosing_block
        is_new_var_function_input = (
            isinstance(block_1, Function) and new_var in block_1.inputs.values()
        )
        block_2 = old_var.op.enclosing_block
        is_old_var_function_output = isinstance(block_2, Function) and old_var in block_2.outputs

        if is_new_var_function_input and is_old_var_function_output:
            return mb.identity(x=new_var, before_op=before_op)
        return new_var

    def _fuse_cast_ops(self, cast_ops: List[Operation], reuse_input_var: bool = False):
        """
        Fuse the pattern of:
            input -> cast_1(dtype=dtype_1) -> cast_2(dtype=dtype_2) -> out

        If ``reuse_input_var = True``, the pattern is reduced to:
            input -> out

        otherwise, a new ``cast`` op with the same ``dtype`` as ``cast_2`` is created:
            input -> cast_3(dtype=dtype_2) -> out
        """
        if not isinstance(cast_ops[0], tuple):
            cast_ops = tuple((cast_ops,))

        ops_to_remove = []

        for cast_1, cast_2 in cast_ops:
            if reuse_input_var:
                new_output_var = self._dup_if_affect_io(cast_1.x, cast_2.outputs[0], cast_1)
            else:
                fused_output_var_name = cast_1.x.name + "_to_{}".format(cast_2.dtype.val)
                new_output_var = mb.cast(
                    x=cast_1.x,
                    dtype=cast_2.dtype,
                    name=fused_output_var_name,
                    before_op=cast_2,
                )
            # It's important to use `cast_2.enclosing_block` since `cast_2` might be present in a block nested under `cast_1.enclosing_block`
            cast_2.enclosing_block.replace_uses_of_var_after_op(
                anchor_op=cast_2,
                old_var=cast_2.outputs[0],
                new_var=new_output_var,
            )
            # Remove just the last cast op and let dce eliminate the rest of the ops if needed,
            # The reason is that first cast op could be feeding into other non-cast ops.
            ops_to_remove.append(cast_2)

        ops_to_remove[0].enclosing_block.remove_ops(ops_to_remove)

    def _try_to_transform(self, root_op, cast_ops_across_blocks):
        block = root_op.enclosing_block
        if block is None:
            return False

        # Scenario: Redundant cast when source and destination dtype are same.
        if root_op.op_type == "cast" and root_op.x.is_tensor_or_scalar_of(dtype=root_op.dtype.val):
            new_var = root_op.x
            old_var = root_op.outputs[0]
            new_var = self._dup_if_affect_io(root_op.x, old_var, root_op)
            block.replace_uses_of_var_after_op(
                anchor_op=root_op,
                old_var=old_var,
                new_var=new_var,
            )
            block.remove_ops([root_op])
            return True

        # Scenario: Consecutive casts
        candidate_child_ops = []
        for op in root_op.outputs[0].child_ops:
            if op.op_type == "cast":
                candidate_child_ops.append(op)

        fusion_happens = False
        for child_op in candidate_child_ops:
            if not self._is_cast_ops_fusable(root_op, child_op):
                continue

            if root_op.x.is_tensor_or_scalar_of(dtype=child_op.dtype.val):
                # when consecutive casts cancel each other
                # Please check out: test_linear_consecutive_cast_ops_cancellation in TestCastOptimization
                self._fuse_cast_ops((root_op, child_op), reuse_input_var=True)
                fusion_happens = True
            else:
                if child_op.enclosing_block != block:
                    # If cast_2 is in an inner block, we handle it at once in a separated function `_fuse_casts_ops_across_blocks`
                    cast_ops_across_blocks[child_op.enclosing_block].add((root_op, child_op))
                    continue
                self._fuse_cast_ops((root_op, child_op))
                fusion_happens = True
        return fusion_happens

    @block_context_manager
    def _fuse_casts_ops_across_blocks(self, block: Block, ops_to_fused: Tuple[Operation]):
        self._fuse_cast_ops(ops_to_fused)

    @block_context_manager
    def _fuse_or_cancel_consecutive_casts_block_wrapper(self, block):
        def _fuse_or_cancel_consecutive_casts_block(block, cast_ops_across_blocks):
            # We first make sure all the inner blocks are optimized
            # It is important to do it seperately in the very beginning, to ensure the last step of optimization cast ops across the block boundary is correct.
            for op in block.operations:
                for b in op.blocks:
                    self._fuse_or_cancel_consecutive_casts_block_wrapper(b)

            fusion_happens = False
            for op in list(block.operations):
                self._num_of_visited_ops += 1
                # start pattern match if cast op is encountered
                if op.op_type == "cast":
                    if self._try_to_transform(op, cast_ops_across_blocks):
                        # It is important not to exist the loop right away when a fusion happens,
                        # in order to make the time conplexity low.
                        # For instance, given a program of the pattern:
                        # relu -> relu -> cast -> cast -> cast,
                        # the three cast ops can be fused into a single cast op in one shot.
                        # On the other hand, if we break the loop right away, the
                        # two relu ops will be visited 3 times, and makes the overal
                        # time complexity O(N^2).
                        fusion_happens = True
            return fusion_happens

        block_changed = True
        cast_ops_across_blocks = defaultdict(set)
        while block_changed:
            block_changed = _fuse_or_cancel_consecutive_casts_block(block, cast_ops_across_blocks)

        # fuse the cast ops across the inner / outer block boundary
        for k, v in cast_ops_across_blocks.items():
            self._fuse_casts_ops_across_blocks(k, tuple(v))

class TransformAxisUpdateOps:
    """
    Parent class for every axis update op's class

    An axis update op is an op that can be updated, such that it can allow a transpose layer to "pass" through it.
    That is,

    op(transpose(x)) == transpose(op_updated(x))

    where
    "op" : original op,
    "op_updated": op after being updated.

    Example:

    if x is a tensor of rank 2, and transpose has perm=[1,0],
    then

    reduce_mean[axis=1](transpose(x)) == transpose(reduce_mean[axis=0](x))

    here reduce_mean op with axis=1 can be updated to a reduce_mean op with axis=0,
    to allow the transpose to "pass" through it, i.e. get applied after it.

    """

    def __init__(self, op, transpose_axes, var_to_hypothetical_value_dict=None):
        self.op = op
        self.transpose_axes = transpose_axes
        self.var_to_hypothetical_value_dict = var_to_hypothetical_value_dict

    def can_transpose_pass(self):
        """
        Each "axis" op must determine whether it can act like a unary op
        and allow the transpose to pass through.
        Return True if it can allow the transpose to pass through, otherwise return False.

        :return: bool
        """
        raise NotImplementedError("This function must be implemented by each op")

    def update(self):
        """
        A method that updates some attribute of the axis op,
        based on the transpose axes value.
        This method only gets called if "can_transpose_pass" returns True.

        Update the op such that the output %i2 should be equal to %o2

        Before:
        %i_1 = transpose_op(%i_0, perm=transpose_axes)
        %i2 = op(%i1)

        After:
        %o1 = op_updated(%i0)
        %o2 = transpose_op(%o1, perm=transpose_axes)

        :return: None
        """
        raise NotImplementedError("This function must be implemented by each op")

    @staticmethod
    def _find_transpose_compliment(perm):
        """
        return the permutation value that when applied will reverse the
        effect of the given permutation.

        e.g.: if perm == (1, 2, 3, 0), then return (3, 0, 1, 2), which will undo the
        first permutation's effect
        """
        rank = len(perm)
        all_positive_perm = [p + rank if p < 0 else p for p in perm]
        perm_inverse = [0] * rank
        for i in range(rank):
            perm_inverse[i] = all_positive_perm.index(i)
        return perm_inverse


class _HypotheticalValue:
    """
    A hypothetical value that simply wraps a Var. Actual Var it wraps doesn't really matter, as
    its mainly for debugging.
    This class really exists to differentiate a "_LazyTransposeHypotheticalValue" type with a
    non-"_LazyTransposeHypotheticalValue" type.
    """

    def __init__(self, var=None):
        self.value = var  # type : Var


class _LazyTransposeHypotheticalValue:
    """
    A hypothetical value that represents a transpose op on top of a hypothetical value, or a
    collection of transpose_ops, which have the same "perm" parameter.
    """

    def __init__(self, hypothetical_value, transpose_ops, perm):
        # Input hypothetical value to the transpose op.
        # When there are multiple transpose ops, this is the incoming hypothetical value to any one of those
        self.wrapped_hypothetical_value = hypothetical_value  # type : _HypotheticalValue

        if not isinstance(hypothetical_value, _HypotheticalValue):
            raise ValueError(
                "transpose optimization pass: incorrect type passed for hypothetical_value"
            )

        for op in transpose_ops:
            if op.op_type != "transpose":
                raise ValueError(
                    "transpose optimization pass: _LazyTransposeHypotheticalValue can only be made with transpose ops"
                )
            perm_op = list(op.inputs["perm"].val)
            if perm_op != perm:
                raise ValueError(
                    "transpose optimization pass: _LazyTransposeHypotheticalValue can only be made with transpose ops with the same 'perm' values"
                )

        self.perm = perm  # type : list[int], perm parameter of all the transpose ops
        self.transpose_ops = transpose_ops  # type : Set(op)


class _TransposeOptimization:
    _DEBUG = False  # Set to true to plot the block before and after the transformation.

    # Dictionary from axis update op to its class
    # This is filled in by child classes of the class "TransformAxisUpdateOps".
    _AXIS_UPDATE_OPS = dict()

    # TODO: instead of a hard-coded set, use op-traits
    # These are the ops that satisfy the following property:
    # - single non constant input
    # - single output
    # - non rank changing
    # - doesn't need to be updated of a transpose passes through it. i.e.
    #  Transpose(op(x)) == op(Transpose(x))
    _UNARY_LIKE_OP_TYPES = {
        "relu",
        "log",
        "relu6",
        "abs",
        "acos",
        "asin",
        "atan",
        "atanh",
        "ceil",
        "clip",
        "cos",
        "cosh",
        "erf",
        "exp",
        "exp2",
        "floor",
        "identity",
        "logical_not",
        "round",
        "rsqrt",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "square",
        "pow",
        "tan",
        "tanh",
        "threshold",
        "clamped_relu",
        "elu",
        "gelu",
        "leaky_relu",
        "linear_activation",
        "scaled_tanh",
        "sigmoid",
        "sigmoid_hard",
        "softplus",
        "softplus_parametric",
        "softsign",
        "thresholded_relu",
    }

    def __init__(self, block):
        self.block = block

        # for each var in the block, this dictionary stores the hypothetical value that is assigned to it during
        # graph traversal
        self.var_to_hypothetical_value = (
            {}
        )  # type : var : _HypotheticalValue or _LazyTransposeHypotheticalValue
        # start out by filling this dictionary with all the inputs of the block
        for _, input_var in block.inputs.items():
            self.var_to_hypothetical_value[input_var] = _HypotheticalValue(input_var)

        # Dictionaries below are used to store transpose cancellation/fusion information.
        # These are filled during the traversal of the graph,
        # after which they are used by the `_apply_transform` method

        # transpose op to the list of transpose ops that are its compliments and can be cancelled away with it
        self.transpose_op_to_cancel_ops = defaultdict(lambda: [])  # type : op : List[op]

        # transpose op to the list of ops before which it has to materialize, i.e. the root transpose op
        #  can be moved downstream in the graph, as far as these materialize ops
        self.transpose_op_to_materialize_ops = defaultdict(
            lambda: []
        )  # type : op : List[Tuple(op, Var)]

        # list of the ops that need to be updated (either their axis parameter or one of their constant inputs)
        # if the transpose op is fused away or moved downstream in the graph
        self.transpose_op_to_axis_update_ops = defaultdict(lambda: [])  # type : op : List[op]

        # for book keeping
        self.ops_updated = set()
        self.materialized_ops_handled = set()
        self.transpose_ops_removed = set()

        # save the output sinks' information
        self.old_output_vars = []
        self.output_sink_ops = []

        # We modify the graph temporarily for outputs
        self._add_output_sinks()

    def _add_output_sinks(self):
        # We add an identity sink for all outputs.
        self.old_output_vars = {var: var.name for var in self.block.outputs}
        new_outputs = []
        output_sinks_var = {}
        for out_var in self.block.outputs:
            if out_var not in output_sinks_var:
                out_sink = mb.identity(x=out_var)
                output_sinks_var[out_var] = out_sink
            else:
                out_sink = output_sinks_var[out_var]
            new_outputs.append(out_sink)
            self.output_sink_ops.append(out_sink.op)
        self.block.set_outputs(new_outputs)

    def _visit_unary_like_op(self, op, input_var=None):
        # pass the input var's hypothetical_value to the output var's, since shape invariant ops do
        # not modify the incoming hypothetical_value

        if input_var is None:
            input_var = op.inputs["x"]

        if len(op.outputs) > 1:
            msg = (
                "transpose optimization pass: op '{}', of type = '{}', has multiple outputs, hence it"
                "cannot be handled like a unary op"
            )
            raise ValueError(msg.format(op.name, op.op_type))
        self.var_to_hypothetical_value[op.outputs[0]] = self.var_to_hypothetical_value[input_var]

    def _visit_materialize_op(self, op):
        # this is the catch all category of ops
        # these are the "not-lazy-transpose-pass-through" kind of ops
        # output hypothetical_value is same as the vars
        for out_var in op.outputs:
            self.var_to_hypothetical_value[out_var] = _HypotheticalValue(out_var)

        # check for the inputs
        # if there is a lazy transpose hypothetical value as an input,
        # all the transpose ops it hold,
        # need to be materialized here now, i.e., we should update "transpose_op_to_materialize_ops"
        for input_var in self._get_input_vars(op):
            input_hypothetical_value = self.var_to_hypothetical_value[input_var]
            if isinstance(input_hypothetical_value, _LazyTransposeHypotheticalValue):
                all_lazy_transpose_ops = input_hypothetical_value.transpose_ops
                for transpose_op in all_lazy_transpose_ops:
                    self.transpose_op_to_materialize_ops[transpose_op].append((op, input_var))

    def _visit_axis_update_op(self, op):
        """
        Check:
        - at least one of the non-constant inputs to this op is of type _LazyTransposeHypotheticalValue
        - for all non-constant inputs, that are of type _LazyTransposeHypotheticalValue, they
        have the same perm value.
        These checks are common for all "axis update" ops.
        """
        input_vars = self._get_input_vars(op, only_nonconst_vars=True)
        perm = None
        num_lazy_input_vars = 0
        for var in input_vars:
            hypothetical_value = self.var_to_hypothetical_value[var]
            if isinstance(hypothetical_value, _LazyTransposeHypotheticalValue):
                num_lazy_input_vars += 1
                if perm is None:
                    perm = hypothetical_value.perm
                elif perm != hypothetical_value.perm:
                    self._visit_materialize_op(op)
                    return

        if num_lazy_input_vars == 0:
            self._visit_materialize_op(op)
            return

        # checks specific to the op type
        op_cls = self._AXIS_UPDATE_OPS.get(op.op_type, None)
        if op_cls is None:
            raise ValueError("Transform class for op of type '{}' not found".format(op.op_type))

        if not op_cls(
            **{
                "op": op,
                "transpose_axes": perm,
                "var_to_hypothetical_value_dict": self.var_to_hypothetical_value,
            }
        ).can_transpose_pass():
            self._visit_materialize_op(op)
            return

        # add this op to the dictionary "transpose_op_to_axis_update_ops"
        # and update self.var_to_hypothetical_value[op.outputs[0]]
        all_lazy_transpose_ops = set()
        wrapped_hypothetical_value = None
        for var in input_vars:
            input_hypothetical_value = self.var_to_hypothetical_value[var]
            if isinstance(input_hypothetical_value, _LazyTransposeHypotheticalValue):
                all_lazy_transpose_ops.update(input_hypothetical_value.transpose_ops)
                wrapped_hypothetical_value = input_hypothetical_value.wrapped_hypothetical_value

        for transpose_op in all_lazy_transpose_ops:
            self.transpose_op_to_axis_update_ops[transpose_op].append(op)

        for output in op.outputs:
            self.var_to_hypothetical_value[output] = _LazyTransposeHypotheticalValue(
                wrapped_hypothetical_value,
                all_lazy_transpose_ops,
                perm,
            )

    @staticmethod
    def _do_transposes_cancel(perm1, perm2):
        if len(perm1) != len(perm2):
            return False
        x = list(range(len(perm1)))
        x1 = [x[i] for i in perm1]
        x2 = [x1[i] for i in perm2]
        if x == x2:
            return True
        return False

    def _visit_transpose_op(self, op):
        input_var = op.inputs["x"]
        if op.inputs["perm"].val is None:
            self._visit_materialize_op(op)
            return
        perm = list(op.inputs["perm"].val)
        input_hypothetical_value = self.var_to_hypothetical_value[input_var]

        """
        There are 3 cases to handle:

        1. input type == _HypotheticalValue
        2. input type == _LazyTransposeHypotheticalValue and this op is the transpose compliment of it
        3. input type == _LazyTransposeHypotheticalValue and this op is NOT the transpose compliment of it
        """

        if isinstance(input_hypothetical_value, _HypotheticalValue):
            # case 1
            # the input is not a lazy transpose.
            # Since the current node is a transpose, there are two sub-cases.
            #  a) It's a output node. We materialize it directly.
            #  b) It might get cancelled downstream, so make the output var's
            #     hypothetical_value a lazy transpose
            if op.outputs[0] in self.old_output_vars:
                self._visit_materialize_op(op)
            else:
                self.var_to_hypothetical_value[op.outputs[0]] = _LazyTransposeHypotheticalValue(
                    input_hypothetical_value, set([op]), perm
                )
            return

        # input is a Lazy transpose hypothetical value. Lets first check whether the current
        # transpose cancels it or not
        do_cancel = self._do_transposes_cancel(input_hypothetical_value.perm, perm)
        if do_cancel:
            # case 2
            # transposes cancel, so now the hypothetical_value of the output will
            # be same as the hypothetical value wrapped inside the upstream lazy transpose
            self.var_to_hypothetical_value[
                op.outputs[0]
            ] = input_hypothetical_value.wrapped_hypothetical_value
            # also update the dictionary "transpose_op_to_cancel_ops"
            all_lazy_transpose_ops = input_hypothetical_value.transpose_ops
            for transpose_op in all_lazy_transpose_ops:
                self.transpose_op_to_cancel_ops[transpose_op].append(op)
        else:
            # case 3
            # transposes don't cancel
            # this is same as a materialize op then
            self._visit_materialize_op(op)

    def _visit_op(self, op):

        input_vars = self._get_input_vars(op)
        for var in input_vars:
            assert (
                var in self.var_to_hypothetical_value
            ), "transpose optimization pass: hypothetical value for var '{}', not found".format(
                var.name
            )

        if op in self.output_sink_ops:
            self._visit_materialize_op(op)
        elif op.op_type in self._UNARY_LIKE_OP_TYPES:
            self._visit_unary_like_op(op)
        elif op.op_type in self._AXIS_UPDATE_OPS:
            self._visit_axis_update_op(op)
        elif op.op_type == "transpose":
            self._visit_transpose_op(op)
        elif op.op_type == "const":
            self.var_to_hypothetical_value[op.outputs[0]] = _HypotheticalValue(op.outputs[0])
        else:
            self._visit_materialize_op(op)

    def block_traversal(self):

        # Since the ops are already organized in a topological manner,
        # simply iterate through all the ops

        for op in self.block.operations:
            self._visit_op(op)

    def _verify_cancellable_transposes(self):

        # invert "transpose_op_to_cancel_ops"
        transpose_cancel_ops_to_starting_transpose_set = defaultdict(lambda: set())
        for op, cancel_ops_list in self.transpose_op_to_cancel_ops.items():
            for cancel_op in cancel_ops_list:
                transpose_cancel_ops_to_starting_transpose_set[cancel_op].update(set([op]))

        for op in transpose_cancel_ops_to_starting_transpose_set:
            assert (
                op not in self.transpose_op_to_cancel_ops
            ), "transpose reduction optimization: transpose op '{}' cannot be both a starting and cancel op".format(
                op.name
            )

        # invert "transpose_op_to_materialize_ops"
        materizalize_ops_to_starting_transpose_set = defaultdict(lambda: set())
        for op, materialize_ops in self.transpose_op_to_materialize_ops.items():
            for materialize_op, edge in materialize_ops:
                materizalize_ops_to_starting_transpose_set[materialize_op].update(set([op]))

                # the starting transpose op may not be in "transpose_op_to_cancel_ops"
                # but it needs to be removed if it materializes later, hence we need to add it
                # to the "transpose_op_to_cancel_ops", with an empty value, i.e. no other ops to cancel because of it
                if op not in self.transpose_op_to_cancel_ops:
                    self.transpose_op_to_cancel_ops[op] = []

        # (starting transpose ops) and (transpose cancel ops + materialize ops) form a bipartite graph.
        # Find the connected components of this graph, by doing a BFS traversal
        connected_components = []  # List[(Set(op), Set(op)), Set(op)]
        visited = {}
        for op in list(self.transpose_op_to_cancel_ops.keys()):
            if op in visited:
                continue
            visited[op] = 1
            set_a = set([op])  # set of starting transpose ops
            set_b1 = set()  # set of transpose cancel ops connected to set_a
            set_b2 = set()  # set of materialize ops connected to set_a
            queue = []
            queue.extend(self.transpose_op_to_cancel_ops[op])
            if op in self.transpose_op_to_materialize_ops:
                materialize_ops_list = list(list(zip(*self.transpose_op_to_materialize_ops[op]))[0])
                queue.extend(materialize_ops_list)
            while len(queue) > 0:
                o = queue.pop(0)
                visited[o] = 1
                # enqueue nodes connected to o
                if o in self.transpose_op_to_cancel_ops:
                    set_a.update(set([o]))
                    for neighbor_op in self.transpose_op_to_cancel_ops[o]:
                        if neighbor_op not in visited:
                            queue.append(neighbor_op)
                    if o in self.transpose_op_to_materialize_ops:
                        materialize_ops_list = list(
                            list(zip(*self.transpose_op_to_materialize_ops[o]))[0]
                        )
                        for neighbor_op in materialize_ops_list:
                            if neighbor_op not in visited:
                                queue.append(neighbor_op)
                elif o in transpose_cancel_ops_to_starting_transpose_set:
                    set_b1.update(set([o]))
                    for neighbor_op in transpose_cancel_ops_to_starting_transpose_set[o]:
                        if neighbor_op not in visited:
                            queue.append(neighbor_op)
                else:
                    set_b2.update(set([o]))
                    for neighbor_op in materizalize_ops_to_starting_transpose_set[o]:
                        if neighbor_op not in visited:
                            queue.append(neighbor_op)
            connected_components.append((set_a, set_b1, set_b2))

        starting_ops_to_remove = set()  # starting ops to remove from the optimization list

        # now for each connected component, make a decision whether to cancel it or not
        # (either all transpose ops in a set get cancelled or they don't)
        for op_set, op_cancel_set, materialize_op_set in connected_components:

            block_output = False
            # check that output is not directly connected to a starting transpose op
            for op in op_set:
                if op.outputs[0] in self.block.outputs:
                    starting_ops_to_remove.update(op_set)
                    block_output = True
                    break
            if block_output:
                continue

            materizalize_set = set(list(materialize_op_set))
            if len(materizalize_set) >= len(op_set) + len(op_cancel_set):
                starting_ops_to_remove.update(op_set)

        # remove ops
        for op in starting_ops_to_remove:
            self.transpose_op_to_cancel_ops.pop(op, None)

    def _remove_transpose_ops(self, starting_transpose_op):

        perm = list(starting_transpose_op.inputs["perm"].val)
        starting_transpose_op_out_var = starting_transpose_op.outputs[0]
        starting_transpose_op_input_var = starting_transpose_op.inputs["x"]

        # update all the "axis_update" ops
        for op in self.transpose_op_to_axis_update_ops.get(starting_transpose_op, []):
            if op not in self.ops_updated:
                op_cls = self._AXIS_UPDATE_OPS.get(op.op_type, None)
                op_cls(
                    **{
                        "op": op,
                        "transpose_axes": perm,
                        "var_to_hypothetical_value_dict": self.var_to_hypothetical_value,
                    }
                ).update()
                self.ops_updated.add(op)

        # short circuit starting_transpose_op and its cancel ops

        to_be_removed_ops = []
        name_changed_vars = set()

        for op in [starting_transpose_op] + self.transpose_op_to_cancel_ops[starting_transpose_op]:
            if op in self.transpose_ops_removed:
                continue

            to_be_removed_ops.append(op)
            self.transpose_ops_removed.add(op)

            input_var = op.inputs["x"]  # input to the transpose op
            output_var = op.outputs[0]  # output of the transpose op
            parent_op = input_var.op  # parent op of the transpose op

            if output_var in self.old_output_vars:
                # output is a block output, so this must be one of the "edge" transpose compliment ops
                # We need to set `input_var` as the block output var
                # Change the name of the input_var to match the block output if input_var is not changed.
                # If the same input_var is in output twice, we can't rename it twice, therefore we initiate an
                # Identity op to match the name
                if input_var in self.block.inputs.values():
                    input_var = mb.identity(x=input_var, before_op=op, name=output_var.name)
                    parent_op = None  # set anchor op as None.
                elif input_var not in name_changed_vars:
                    input_var.name = output_var.name
                    input_var.op.name = output_var.op.name
                    name_changed_vars.update([input_var])
                else:
                    input_var = mb.identity(x=input_var, before_op=op, name=output_var.name)
                    parent_op = input_var.op

            # connect all the child ops of the output_var to the parent of the transpose op.
            self.block.replace_uses_of_var_after_op(
                anchor_op=parent_op,
                old_var=output_var,
                new_var=input_var,
                no_check_var_types=True,
            )

        """
        Insert a transpose op JUST before each one of the materialize ops
        i.e.
        Given:  %i1 = op(...)
                ...
                ... = materialize_op(..., %i1 ,...)
                ...

        Result: %i1 = op(...)
                ...
                %i2 = transpose_op(%i1, %perm)
                ... = materialize_op(..., %i2 ,...)
                ...
        """
        for op, input_var in self.transpose_op_to_materialize_ops.get(starting_transpose_op, []):
            if (op, input_var) in self.materialized_ops_handled:
                continue

            self.materialized_ops_handled.add((op, input_var))
            if input_var == starting_transpose_op_out_var:
                # materialize op is connected to the starting transpose op
                # in this case, connect to its parent
                if op in self.output_sink_ops:
                    continue
                i1 = starting_transpose_op_input_var
            else:
                i1 = input_var

            if op in self.output_sink_ops:
                # The input_var of output sink is itself a output. We can safely
                # modify the name of the input_var since it should only be consumed
                # by block output here.
                if i1 not in name_changed_vars:
                    x = mb.transpose(x=i1, perm=perm, before_op=op, name=i1.name)
                    i1.name = "_before_transpose_op_" + x.op.name
                    i1.op.name = "_before_transpose_op_" + x.op.name
                else:
                    x = mb.transpose(x=i1, perm=perm, before_op=op, name=self.old_output_vars[i1])
            else:
                x = mb.transpose(x=i1, perm=perm, before_op=op)

            self.block.replace_uses_of_var_after_op(
                anchor_op=x.op,
                end_op=op,
                old_var=i1,
                new_var=x,
                no_check_var_types=True,
            )

        self.block.remove_ops(to_be_removed_ops)

    def apply_transform(self):
        """
        Take in the data collected during graph traversal
        and transform the graph by cancelling out transpose ops that can be removed.
        """

        logger.debug("Block before optimize transpose transform:\n{}".format(self.block))
        if self._DEBUG:
            import graphviz

            graphviz.Source(
                self.block.get_dot_string(
                    highlight_debug_op_names=[], highlight_debug_op_types=["transpose"]
                )
            ).view(filename="/tmp/block_before_reduce_transpose")

        """
        First check which transposes can be cancelled.
        After this function call we get an updated dictionary "transpose_op_to_cancel_ops"
        with only the transpose ops that can really be cancelled in the graph
        Reasons to not cancel:
        - materialize_ops are greater than cancel_ops, so removing transpose will instead end up increasing the count of transposes
        - removing a transpose op can only be successful, if all of its cancel ops are removed, removing all the cancel ops
          is only successful if all of their starting transpose ops are removed and so on. This check is also done in
           "_verify_cancellable_transposes()"
        """
        self._verify_cancellable_transposes()

        # apply transform
        for transpose_op in self.transpose_op_to_cancel_ops:
            self._remove_transpose_ops(transpose_op)
        self.block.set_outputs([sink_op.x for sink_op in self.output_sink_ops])
        self.block.remove_ops(list(self.output_sink_ops))

        if self._DEBUG:
            graphviz.Source(
                self.block.get_dot_string(
                    highlight_debug_op_names=[], highlight_debug_op_types=["transpose"]
                )
            ).view(filename="/tmp/block_after_reduce_transpose")

        logger.debug("Block after optimize transpose transform:\n{}".format(self.block))

        for op in self.block.operations:
            op.type_value_inference(overwrite_output=True)

    @staticmethod
    def register_axis_update_op(ops: List[Text]):
        """
        :param ops: Ops that will be registered. For example: the class "_TransformReduceMean" can
        be used to register ops including "reduce_prod", "reduce_sum" etc.
        """

        def class_wrapper(op_update_cls):
            for op_type in ops:
                if op_type in _TransposeOptimization._AXIS_UPDATE_OPS:
                    raise ValueError(
                        "Update class for op of type '{}' already defined".format(op_type)
                    )
                _TransposeOptimization._AXIS_UPDATE_OPS[op_type] = op_update_cls
            return op_update_cls

        return class_wrapper

    @staticmethod
    def _get_input_vars(op, only_nonconst_vars=False) -> List[Var]:
        input_vars = []
        for name, val in op.inputs.items():
            if isinstance(val, Var):
                if only_nonconst_vars:
                    if val.op and val.op.op_type == "const":
                        continue
                input_vars.append(val)
            elif isinstance(val, (list, tuple)):
                for var in val:
                    if not isinstance(var, Var):
                        raise ValueError(
                            f"transpose optimization pass: unrecognized input type of "
                            f"op='{op.name}', input='{name}'"
                        )
                    if only_nonconst_vars:
                        if var.op and var.op.op_type == "const":
                            continue
                    input_vars.append(var)
            else:
                raise ValueError(
                    f"transpose optimization pass: unrecognized input type of "
                    f"op='{op.name}', input='{name}'"
                )
        return input_vars


@_TransposeOptimization.register_axis_update_op(ops=["concat"])
class _TransformConcat(TransformAxisUpdateOps):
    def __init__(self, **kwargs):
        super(_TransformConcat, self).__init__(**kwargs)
        self.axis_var = self.op.inputs["axis"]

    def can_transpose_pass(self):
        # Check that all non const inputs are of type _LazyTransposeHypotheticalValue.
        # That they have the same perm value has already been checked before.
        input_vars = _TransposeOptimization._get_input_vars(self.op, only_nonconst_vars=True)
        for var in input_vars:
            hypothetical_value = self.var_to_hypothetical_value_dict[var]
            if not isinstance(hypothetical_value, _LazyTransposeHypotheticalValue):
                return False
        if self.axis_var.val is not None:
            return True
        return False

    def update(self):
        new_axis_val = self.transpose_axes[self.axis_var.val]

        # to be used, if there is a constant inputs to the concat op
        self._update_const_inputs()

        # insert a new constant for the new axis, JUST before the op
        with self.op.enclosing_block:
            new_axis_var = mb.const(val=new_axis_val, before_op=self.op)

        self.op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=new_axis_var.op,
            end_op=self.op,
            old_var=self.axis_var,
            new_var=new_axis_var,
            no_check_var_types=True,
        )

    def _update_const_inputs(self):
        transpose_perm_for_const = [0] * len(self.transpose_axes)
        for i, axis in enumerate(self.transpose_axes):
            transpose_perm_for_const[axis] = i

        # if there is a constant input, transpose it
        inputs = list(self.op.inputs["values"])
        for input_var in inputs:
            if input_var.op.op_type == "const":
                const_val = input_var.val
                new_const_val = np.transpose(const_val, transpose_perm_for_const)
                # insert a new constant JUST before the op
                with self.op.enclosing_block:
                    new_const_input_var = mb.const(val=new_const_val, before_op=self.op)
                self.op.enclosing_block.replace_uses_of_var_after_op(
                    anchor_op=new_const_input_var.op,
                    end_op=self.op,
                    old_var=input_var,
                    new_var=new_const_input_var,
                    no_check_var_types=True,
                )


@_TransposeOptimization.register_axis_update_op(ops=["split"])
class _TransformSplit(_TransformConcat):
    def __init__(self, **kwargs):
        super(_TransformSplit, self).__init__(**kwargs)

    # The split op is handled the same as the concat op, except it does not need
    # to transform const inputs
    def _update_const_inputs(self):
        pass


@_TransposeOptimization.register_axis_update_op(ops=["pad"])
class _TransformPad(TransformAxisUpdateOps):
    def __init__(self, **kwargs):
        super(_TransformPad, self).__init__(**kwargs)
        self.pad_var = self.op.inputs["pad"]
        self.pad_op = self.pad_var.op
        self.mode = self.op.mode.val
        self.pad_amounts_new = None

    def _compute_new_pad_values(self):
        pad_amounts = np.reshape(self.pad_var.val, [-1, 2])
        rank_diff = len(self.transpose_axes) - pad_amounts.shape[0]
        self.pad_amounts_new = copy.deepcopy(pad_amounts)
        # append "rank_diff" rows of zeros to the top
        self.pad_amounts_new = np.concatenate(
            (np.zeros((2 * rank_diff)).reshape(-1, 2), self.pad_amounts_new)
        )
        self.pad_amounts_new = self.pad_amounts_new.astype(pad_amounts.dtype)
        pad_amounts = np.concatenate((np.zeros((2 * rank_diff)).reshape(-1, 2), pad_amounts))
        for i, axis in enumerate(self.transpose_axes):
            self.pad_amounts_new[axis][0] = pad_amounts[i][0]
            self.pad_amounts_new[axis][1] = pad_amounts[i][1]
        # get the top "rank_diff" rows
        top_rows = self.pad_amounts_new[:rank_diff, :]
        if not np.all(top_rows == 0):
            return False
        # cut "rank_diff" from the top
        self.pad_amounts_new = self.pad_amounts_new[rank_diff:, :]
        self.pad_amounts_new = self.pad_amounts_new.flatten()
        return True

    def can_transpose_pass(self):
        if (
            len(_TransposeOptimization._get_input_vars(self.op, only_nonconst_vars=True)) != 1
            or self.pad_op.op_type != "const"
        ):
            return False
        if len(self.transpose_axes) < 2:
            return False
        if not self._compute_new_pad_values():
            return False
        # check that if mode is not constant, the updated padding
        # would stay limited to last 2 axes
        if self.mode != "constant" and not np.all(self.pad_amounts_new[:-4] == 0):
            return False
        return True

    def update(self):
        self._compute_new_pad_values()
        # insert a new constant for pad val, JUST before the op
        with self.op.enclosing_block:
            new_pad_var = mb.const(val=self.pad_amounts_new, before_op=self.op)
        self.op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=new_pad_var.op,
            end_op=self.op,
            old_var=self.pad_var,
            new_var=new_pad_var,
            no_check_var_types=True,
        )


@_TransposeOptimization.register_axis_update_op(
    ops=[
        "reduce_l1_norm",
        "reduce_l2_norm",
        "reduce_max",
        "reduce_log_sum",
        "reduce_log_sum_exp",
        "reduce_mean",
        "reduce_min",
        "reduce_prod",
        "reduce_sum",
        "reduce_sum_square",
    ]
)
class _TransformReduceMean(TransformAxisUpdateOps):
    def __init__(self, **kwargs):
        super(_TransformReduceMean, self).__init__(**kwargs)
        self.axes_var = self.op.inputs["axes"]
        self.axes_op = self.axes_var.op

    def can_transpose_pass(self):
        # allow transpose to push through it only if keep_dims are True since that doesn't change the rank
        if self.op.inputs["keep_dims"].val:
            if self.axes_op.op_type == "const":
                return True
        return False

    def update(self):
        # update axis of the op
        old_axes_val = self.axes_var.val
        new_axes_val = [0] * len(old_axes_val)
        for i, axis in enumerate(old_axes_val):
            new_axes_val[i] = self.transpose_axes[axis]

        # insert a new constant for the axis, JUST before the op
        with self.op.enclosing_block:
            new_axis_var = mb.const(val=new_axes_val, before_op=self.op)

        self.op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=new_axis_var.op,
            end_op=self.op,
            old_var=self.axes_var,
            new_var=new_axis_var,
            no_check_var_types=True,
        )


@_TransposeOptimization.register_axis_update_op(
    ops=["add", "mul", "sub", "real_div", "maximum", "minimum"]
)
class _TransformAdd(TransformAxisUpdateOps):
    def __init__(self, **kwargs):
        super(_TransformAdd, self).__init__(**kwargs)
        # self.tranpose_input: this is the input coming from an upstream transpose op. If both inputs are
        #                      connected to an upstream transpose, this will be set to one of those
        # self.other_input: the other input, that is not coming from a transpose
        is_x_input_lazy_transpose = isinstance(
            self.var_to_hypothetical_value_dict[self.op.x], _LazyTransposeHypotheticalValue
        )
        is_y_input_lazy_transpose = isinstance(
            self.var_to_hypothetical_value_dict[self.op.y], _LazyTransposeHypotheticalValue
        )
        if is_x_input_lazy_transpose and is_y_input_lazy_transpose:
            self.other_input = None
            self.tranpose_input = self.op.x
        elif is_y_input_lazy_transpose and not is_x_input_lazy_transpose:
            self.other_input = self.op.x
            self.tranpose_input = self.op.y
        elif is_x_input_lazy_transpose and not is_y_input_lazy_transpose:
            self.other_input = self.op.y
            self.tranpose_input = self.op.x
        else:
            # we should not be here since this class is only invoked,
            # when there is at least one input var of type _LazyTransposeHypotheticalValue
            self.tranpose_input = None
            self.other_input = None

    def can_transpose_pass(self):
        """
         Return True if the one of the following is true:
         - (scenario 1) both inputs are of type _LazyTransposeHypotheticalValue, with the same perm value
         - one input is of type _LazyTransposeHypotheticalValue and the other satisfies one of the following:
             - (scenario 2) it is constant. In this case, the constant can be updated accordingly to allow the transpose to pass through
             - (scenario 3) if its non constant, then all of the following must be true
                 - its shape is fully defined
                 - the transpose compliment operation on the other input can be expressed via a reshape. This can
                   be done if there is only 1 non unit dimension in its shape, or if there are more than 1 non unit dims,
                   the transpose compliment operation only permutes the unit dimensions.

        In scenario 3, the transpose will be removed, by adding an extra static reshape.
        This is based on the assumption that a static reshape op will be less expensive than transpose.
        An example of scenario 3 is displayed below:

         Input pattern:

         (shape=(10, 20, 30))
              |
              |
              V
          Transpose op
          (shape = (20, 30, 10))
              |
              |
              V
          this op  <--------- (shape = (10,)) (other non const input)
              |
              V


         After transpose passes through:

         (shape=(10, 20, 30))
              |
              |
              V
          this op  <--------- (shape = (10, 1, 1)) Reshape op <---------- (shape = (10,)) (other non const input)
              |
              V
          Transpose op
          (shape = (20, 30, 10))
              |
              V

        """

        # ---------------------
        # check for scenario 1
        # --------------------
        # are both inputs _LazyTransposeHypotheticalValue?
        if self.other_input is None:
            return True

        # ---------------------
        # check for scenario 2
        # --------------------
        # is the second input a constant?
        rank = len(self.tranpose_input.shape)
        if len(self.transpose_axes) != rank:
            return False
        other_input_shape = self.other_input.shape
        if any_symbolic(other_input_shape):
            return False
        if len(other_input_shape) > rank:
            return False
        if isinstance(self.other_input.val, (np.ndarray, np.generic)):
            return True

        # ---------------------
        # check for scenario 3
        # --------------------
        # can other input be "reshaped" to allow the transpose to pass through?
        if any_symbolic(self.other_input.shape):
            return False
        transpose_compliment_perm = self._find_transpose_compliment(self.transpose_axes)
        # make the rank of the other input, same as that of the transpose input,
        # by broadcasting
        if len(other_input_shape) < rank:
            other_input_shape = [1] * (rank - len(other_input_shape)) + list(other_input_shape)

        # how many non unit dimensions in the other input's shape?
        if other_input_shape.count(1) in [rank, rank - 1]:
            # 0 or 1 non unit dimension
            return True
        else:
            # more than 1 non unit dimensions in other input
            # check if transpose is moving only dimensions that have values 1
            # if true, then the transpose compliment can be expressed via a reshape
            for i, axis in enumerate(transpose_compliment_perm):
                if i != axis and other_input_shape[axis] != 1:
                    return False
            return True

    def update(self):
        # ----------------------
        # update for scenario 1
        # ----------------------
        if self.other_input is None:
            # nothing to update
            return

        # --------------------------
        # update for scenario 2 & 3
        # --------------------------
        if len(self.other_input.shape) == 0:
            # other input is a scalar, no need to modify it
            return

        # broadcast the shape of other input to match the rank
        rank = len(self.tranpose_input.shape)
        other_input_shape = self.other_input.shape
        if len(other_input_shape) < rank:
            other_input_shape = [1] * (rank - len(other_input_shape)) + list(other_input_shape)

        # find new shape after transpose compliment
        transpose_compliment_perm = self._find_transpose_compliment(self.transpose_axes)
        new_shape = [0] * rank
        for i, axis in enumerate(transpose_compliment_perm):
            new_shape[i] = other_input_shape[axis]

        if self.other_input.val is not None:
            # update the const (scenario 2)
            const_value = self.other_input.val
            new_const_val = np.transpose(
                const_value.reshape(other_input_shape), transpose_compliment_perm
            )
            # insert a new constant JUST before the op
            with self.op.enclosing_block:
                new_const_var = mb.const(val=new_const_val, before_op=self.op)

            self.op.enclosing_block.replace_uses_of_var_after_op(
                anchor_op=new_const_var.op,
                end_op=self.op,
                old_var=self.other_input,
                new_var=new_const_var,
                no_check_var_types=True,
            )
        else:
            # insert a reshape (scenario 3)
            with self.op.enclosing_block:
                new_other_var = mb.reshape(x=self.other_input, shape=new_shape, before_op=self.op)
            self.op.enclosing_block.replace_uses_of_var_after_op(
                anchor_op=new_other_var.op,
                end_op=self.op,
                old_var=self.other_input,
                new_var=new_other_var,
                no_check_var_types=True,
            )


@register_pass(namespace="common")
class reduce_transposes(AbstractGraphPass):
    """
    Reduce transposes when it is applicable. For example:

    .. code-block::

        # Example 1
            Input graph:
            input -----> transpose(axis=[1,0]) -----> transpose(axis=[1,0]) ---> out

            Output graph:
            input -----> identity -----> out

        # Example 2
            Input graph:
            input---->transpose(axis=[0,3,1,2])---->relu---->transpose(axis=[0,2,3,1])--->out

            Output graph:
            input----->relu----->out

        # Example 3
            Input graph:
            input(shape=10,2,3,5)--->transpose(axis=[0,2,3,1])----->relu---->pool----->out1
                                                               |
                                                               |
                                                               --->relu----->log---->transpose(axis=[0,3,1,2])---->out2

            Output graph:
            input(shape=10,2,3,5)----->relu---->transpose(axis=[0,2,3,1])---->pool----->out1
                                   |
                                   |
                                   --->relu----->log---->out2

    Please see ``TransposeOptimizationPass`` for more details.

    Notes
    -----

    This pass is divided into 3 phases:

    `1st phase:` Information gathering.

    - Plug in Identity ops for all output nodes. This allows us to treat all ops uniformly during traversal.
    - Block is traversed in the topological order, starting from the ops connected to the inputs.
    - During the traversal, a value is associated with every var in the block.
      This value can be either of type ``_HypotheticalValue`` or ``_LazyTransposeHypotheticalValue``.
      The main purpose of type ``_HypotheticalValue`` is to indicate that it is `not` of type ``_LazyTransposeHypotheticalValue``.
    - ``_LazyTransposeHypotheticalValue`` represents either one or multiple transpose ops with the same perm value. This information
      is stored in this class. It also wraps a ``_HypotheticalValue`` that was the last hypothetical value which was generated
      prior to the origin of ``_LazyTransposeHypotheticalValue``.
    - Each op decides which type of hypothetical value to associate with its output vars, based on its op type,
      attributes, and the types of the hypothetical values of its input vars.
    - Ops are classified into 4 categories: `unary like`, `axis update`, `transpose`, and `materialize` (for all the rest).
    - Transpose ops are the ops from which a ``_LazyTransposeHypotheticalValue`` originate.
        - If the input to it is a ``_HypotheticalValue``, its output will be a ``_LazyTransposeHypotheticalValue``,
          indicating that this ``transpose`` op is available to get cancelled downstream.
        - If the input to it is a ``_LazyTransposeHypotheticalValue``, then it is checked whether this op cancels it or not.
            - If the op cancels it, a ``_HypotheticalValue`` value is generated at the output and the information about this ``transpose`` cancellation
              is recorded in the dictionary ``transpose_op_to_cancel_ops``.
            - If the op does not cancel, the current ``transpose`` op is categrorized as a `materialize` op. Therefore, the information in
              dictionary ``transpose_op_to_materialize_ops`` is updated accordingly. The output of the op is now mapped to a
              ``_HypotheticalValue``.
    - Unary like ops: These simply transfer their input hypothetical value type to the output.
    - Axis update ops: If a ``transpose`` can pass through them, they are treated like a unary op and the dictionary
      ``transpose_op_to_axis_update_ops`` is updated. If the op cannot be updated in any manner to
      allow a ``transpose`` to pass through, this op is then categorized as a `materialize` op and handled accordingly.
    - Materialize ops: All ``_LazyTransposeHypotheticalValue`` input vars, if present, materialize here. Output of this op
      is always of type ``_HypotheticalValue``. If the input is a ``_LazyTransposeHypotheticalValue``, update the dictionary
      ``transpose_op_to_materialize_ops``.
    - To treat an op like a unary op, add its type to ``_UNARY_LIKE_OP_TYPES``. In future changes we want to make this process
      automatic by detecting an op as a `unary like` by its "traits".
    - To treat an op like `axis update` op, add a class specific to the op implementing the class ``TransformAxisUpdateOps``.
      For examples, see classes ``_TransformConcat``, ``_TransformPad``, and so on. The dictionary ``AXIS_UPDATE_OPS`` is automatically filled
      in by the decorator ``_TransposeOptimization.register_axis_update_op``.

    `2nd phase:` Determining which ``transpose`` ops to remove from the graph.

    All ``transpose`` ops that have a corresponding compliment op in dict ``transpose_op_to_cancel_ops`` is a candidate.
    However, you need to ensure the following:

    - If a ``transpose`` op is removed, then all of its ``cancel`` ops in ``transpose_op_to_cancel_ops`` must also be removed,
      to ensure correctness of the graph. The same is true in the reverse direction as well;
      that is, for every ``cancel`` op that is removed, all its parent ``transpose`` ops upstream must also be removed.
    - ``transpose`` ops should be removed only if the number of ``cancel`` ops is greater than the number of ``transpose`` ops
      that would get freshly introduced to the block as a result of materialization ops. Currently in the algorithm,
      each materialization op/output var (dicts ``transpose_op_to_materialize_ops``/``old_output_vars``)
      results in one more ``transpose`` op, although this can be further optimized in the future.

    To resolve this, we recognize that nodes consisting of sets ``(a)`` and ``(b)`` form a bipartitle graph, where,
    ``(a) ==`` starting ``transpose`` ops (originators of ``_LazyTransposeHypotheticalValue``)
    and ``(b) ==`` set of ``transpose`` ``cancel`` ops and ``materialize`` ops.

    - In this bipartite graph, we find all the connected components for each connected component.
      Either the entire set of ``transpose`` ops in it are removed/materialized, or none
      of them are touched.
    - Thus for each set, a determination is made based on counting the number of ``cancel`` ops and ``materialize`` ops.
    - Based on this determination, the final set of ``transpose`` ops to be removed is updated.

    `3rd phase:` Transforming the graph.

    - ``transpose`` starting ops and the ``cancel`` ops are removed.
    - Axis update ops, affected by these ``transpose`` ops, are updated.
    - Transposes are materialized; that is, added just before the ``materialize`` ops, which are linked to the starting ``transpose`` ops.
      The starting ``transpose`` op can be materialized (inserted) multiple times, before each of the ``materialize`` ops downstream.
    - Block outputs are handled in a similar fashion as the `materialize` ops.
    - Type inference on all ops is invoked after all the transformations.
    - All Identity ops that are plugged into the graph to treat outputs as materialized are removed.

    `Debugging`

    If the ``debug`` flag is set to ``True``, the block before and after the transformation is plotted,
    with transpose nodes highlighted.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._reduce_transposes_block(f)

    @staticmethod
    def _reduce_transposes_block(block):
        """
        Only apply the optimization if the block is flat,
        i.e, it does not contain any op which contains a sub-block.
        TODO:
        Removing transposes and transpose compliments requires re-running
        type inference for the set of ops in between the fused transpose ops,
        which is simpler to do when all the ops in the block are free of sub blocks.
        The case of transpose fusion with sub-block containing ops needs to be handled with more care and test cases.
        """
        for op in block.operations:
            if len(op.blocks) > 0:
                return

        with block:
            opt_transposes = _TransposeOptimization(block)
            opt_transposes.block_traversal()
            opt_transposes.apply_transform()
