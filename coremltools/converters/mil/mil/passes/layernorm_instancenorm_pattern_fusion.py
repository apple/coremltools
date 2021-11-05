# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging
from typing import List, Optional

import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, Block, Var, Program
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass

DEBUG = False  # set to true to plot the block before and after the transformation

def _check_no_output_connection(block: Block, ops: List[Operation]) -> bool:
    """
    Check that none of the op in this pattern is connected to the output
    (except the last add op)

    :param block: Block
    :param ops: List of operations to check on.
    """
    for op in ops[:-1]:
        for out in op.outputs:
            if out in block.outputs:
                return False
    return True


def _check_reduce_op(reduce_op: Operation, mode: str = "reduce_mean") -> bool:
    """
    Check whether or not the reduction op satisfy following conditions:
    - Mode is expected.
    - Does not change rank (keep_dims is True).
    - Axes is known at compile time.

    :param reduce_op: reduce op to check on
    :param mode: reduce mode
    """
    if reduce_op is None:
        return False
    if reduce_op.op_type != mode:
        return False
    if reduce_op.keep_dims is None or reduce_op.keep_dims.val is None:
        return False
    if reduce_op.keep_dims.val is False:
        return False
    if reduce_op.axes is None or reduce_op.axes.val is None:
        return False
    return True


def _check_child_op_types(
    op: Operation, child_op_types: List[str], check_order: bool = True
) -> bool:
    """
    Returns True child op types match child_op_types, otherwise returns False.

    :param op: Current op
    :param child_op_type: Expected child op types
    :param check_order: Ensure child in given order, defaults to True.
    """
    if op is None or len(op.outputs) != 1:
        return False
    child_ops = list(op.outputs[0].child_ops)
    if len(child_ops) != len(child_op_types):
        return False
    ops_types = [c.op_type for c in child_ops]
    if check_order is False:
        ops_types = sorted(ops_types)
        child_op_types = sorted(child_op_types)
    return ops_types == child_op_types


def _try_get_child_op_type(
    op: Operation, child_op_type: str, index: int = 0
) -> Optional[Operation]:
    """
    Returns child op if type matches, otherwise returns None.

    :param op: Current op
    :param child_op_type: Expected child op type
    :param index: Child op index
    """
    if op is None:
        return None
    if len(op.outputs) != 1:
        return None
    child_ops = list(op.outputs[0].child_ops)
    if index >= len(child_ops):
        return None
    if child_ops[index].op_type != child_op_type:
        return None
    return child_ops[index]


def _try_apply_transform(
    reduce_op: Operation,
    block: Block,
    gamma_var: Var,
    beta_var: Var,
    epsilon_var: Var,
    end_op: Operation,
    ops_to_remove: List[Operation],
) -> bool:
    """
    Insert instance_norm / layer_norm and delete all ops.

    :param reduce_op: Start operation of the pattern.
    :param block: Block
    :param gamma_var: Gamma variable.
    :param beta_var: Beta variable.
    :param epsilon_var: Epsilon variable.
    :param end_op: End operation of the pattern.
    :param ops_to_remove: Operations to remove.
    """
    if not _check_no_output_connection(block, ops_to_remove):
        return False

    axes = reduce_op.axes.val
    rank = len(reduce_op.x.shape)

    # check whether the pattern is instance_norm or layer_norm
    is_layernorm = False
    is_instancenorm = False
    is_require_rank4_transpose = False

    negative_axes = [a - rank if a >= 0 else a for a in axes]
    negative_axes.sort()

    if len(gamma_var.val.shape) == len(axes) and len(beta_var.val.shape) == len(axes):
        # axes for layer_norm must be [-1] or [-1, -2] or [-1, -2, -3] and so on
        if negative_axes == list(range(-len(negative_axes), 0)):
            is_layernorm = True

    if rank == 4 and (negative_axes == [-2, -1] or negative_axes == [-3, -2]):
        if (
            len(np.squeeze(gamma_var.val).shape) == 1
            and len(np.squeeze(beta_var.val).shape) == 1
        ):
            is_instancenorm = True
        if negative_axes == [-3, -2]:
            is_require_rank4_transpose = True

    if not (is_instancenorm or is_layernorm):
        return False

    # remove all the ops, and replace with a layer_norm or instance_norm op
    out_name = end_op.outputs[0].name

    if is_require_rank4_transpose:
        x = mb.transpose(
            x=reduce_op.x,
            perm=[0, 3, 1, 2],
            name=out_name + "_transpose_nhwc_nchw",
            before_op=end_op,
        )
    if is_instancenorm:
        x = mb.instance_norm(
            x=x if is_require_rank4_transpose else reduce_op.x,
            gamma=np.squeeze(gamma_var.val),
            beta=np.squeeze(beta_var.val),
            epsilon=epsilon_var,
            name=out_name + "_instancenorm" if is_require_rank4_transpose else out_name,
            before_op=end_op,
        )
    else:  # is_layernorm
        x = mb.layer_norm(
            x=x if is_require_rank4_transpose else reduce_op.x,
            axes=axes,
            gamma=gamma_var,
            beta=beta_var,
            epsilon=epsilon_var,
            name=out_name + "_layernorm" if is_require_rank4_transpose else out_name,
            before_op=end_op,
        )
    if is_require_rank4_transpose:
        x = mb.transpose(
            x=x,
            perm=[0, 2, 3, 1],
            name=out_name + "_transpose_nchw_nhwc",
            before_op=end_op,
        )

    end_op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=end_op, old_var=end_op.outputs[0], new_var=x
    )
    # Remove all the ops at once
    block.remove_ops(ops_to_remove)
    return True


def _try_match_and_transform_pattern_1(reduce_op, block) -> bool:
    """
    Identify the pattern:

    y = gamma * (x - mean) / sqrt(variance + epsilon) + beta

    y = x * [gamma * rsqrt(variance + eps)] + (beta - mean * [gamma * rsqrt(variance + eps)])

    x --> reduce_mean --> sub --> square --> reduce_mean --> add(epsilon) --> rsqrt
    |             |        ^                                                    |
    |             |        |                                                    V
    |-----------------------                                              mul (gamma)
    |             |                                                           |
    |             |                                                   --------|---------
    |             |                                                   |                |
    |             |                                                   |                V
    |             |---------------------------------------------------------------->  mul
    |                                                                 |                |
    |                                                                 V                |
    |--------------------------------------------------------------> mul               |
                                                                      |                V
                                                                      |              sub (beta) --> add --> [...]
                                                                      |                              ^
                                                                      |-------------------------------

    This pattern corresponds to either layer_norm or instance_norm.

    It is instance_norm if all of the following are true:
        - input is rank 4
        - axes of reduce_mean is [-2, -1] or [-3, -2]
          (when [-3, -2], a channel first to channel last transpose would be inserted)
        - gamma and beta are rank 1, after squeeze

    It is layer_norm if all of the following are true:
        - axes is either [-1] or [-1, -2] or [-1, -2, -3] and so on
        - rank of gamma and beta is equal to the length of the axes
    """
    ops_to_remove = []
    root_var = reduce_op.x

    if root_var.shape is None:
        return False

    rank = len(root_var.shape)

    # check that root_var feeds into exactly 3 ops
    if len(list(root_var.child_ops)) != 3:
        return False
    if root_var.op is not None and not _check_child_op_types(
        root_var.op, child_op_types=["reduce_mean", "sub", "mul"]
    ):
        return False

    # check 1st reduce_mean op
    if not _check_reduce_op(reduce_op):
        return False
    ops_to_remove.append(reduce_op)

    # check 1st sub op
    if not _check_child_op_types(reduce_op, ["sub", "mul"], check_order=False):
        return False
    child_ops_reduce_mean = list(reduce_op.outputs[0].child_ops)
    op_a = child_ops_reduce_mean[0]
    op_b = child_ops_reduce_mean[1]
    sub_op1 = op_a if op_a.op_type == "sub" else op_b
    if not (sub_op1.x == root_var and sub_op1.y == reduce_op.outputs[0]):
        return False
    ops_to_remove.append(sub_op1)

    # check square op
    square_op = _try_get_child_op_type(sub_op1, "square")
    if square_op is None:
        return False
    ops_to_remove.append(square_op)

    # check second reduce mean
    reduce_op2 = _try_get_child_op_type(square_op, "reduce_mean")
    if not _check_reduce_op(reduce_op2):
        return False
    ops_to_remove.append(reduce_op2)

    # check add op (with epsilon)
    add_op1 = _try_get_child_op_type(reduce_op2, "add")
    if add_op1 is None:
        return False
    epsilon_var = add_op1.y if add_op1.x == reduce_op2.outputs[0] else add_op1.x
    if epsilon_var.val is None or len(epsilon_var.val.shape) != 0:
        return False  # must be scalar
    ops_to_remove.append(add_op1)

    # check rsqrt
    rsqrt_op = _try_get_child_op_type(add_op1, "rsqrt")
    if rsqrt_op is None:
        return False
    ops_to_remove.append(rsqrt_op)

    # check mul (gamma)
    mul_op1 = _try_get_child_op_type(rsqrt_op, "mul")
    if mul_op1 is None:
        return False
    gamma_var = mul_op1.y if mul_op1.x == rsqrt_op.outputs[0] else mul_op1.x
    if gamma_var.val is None:
        return False
    ops_to_remove.append(mul_op1)

    # check 2 muls after the gamma mul
    if not _check_child_op_types(mul_op1, ["mul", "mul"]):
        return False
    child_ops = list(mul_op1.outputs[0].child_ops)
    mul_op2 = child_ops[0]
    mul_op3 = child_ops[1]
    mul_op2_other_var = mul_op2.x if mul_op2.y == mul_op1.outputs[0] else mul_op2.y
    mul_op3_other_var = mul_op3.x if mul_op3.y == mul_op1.outputs[0] else mul_op3.y
    if not (
        (mul_op2_other_var == root_var and mul_op3_other_var == reduce_op.outputs[0])
        or (mul_op2_other_var == reduce_op.outputs[0] and mul_op3_other_var == root_var)
    ):
        return False
    if mul_op2_other_var == root_var:
        mul_root_op = mul_op2
        mul_mean_op = mul_op3
    else:
        mul_root_op = mul_op3
        mul_mean_op = mul_op2
    ops_to_remove.append(mul_mean_op)
    ops_to_remove.append(mul_root_op)

    # check sub with beta
    sub_op2 = _try_get_child_op_type(mul_mean_op, "sub")
    if sub_op2 is None:
        return False
    if sub_op2.y != mul_mean_op.outputs[0]:
        return False
    beta_var = sub_op2.x
    if beta_var.val is None:
        return False
    ops_to_remove.append(sub_op2)

    # check last add op
    add_op2 = _try_get_child_op_type(sub_op2, "add")
    if add_op2 is None:
        return False
    if not (add_op2.x == mul_root_op.outputs[0] or add_op2.y == mul_root_op.outputs[0]):
        return False
    ops_to_remove.append(add_op2)

    return _try_apply_transform(
        reduce_op, block, gamma_var, beta_var, epsilon_var, add_op2, ops_to_remove
    )


def _try_match_and_transform_pattern_2(reduce_op, block) -> bool:
    """
    Identify the pattern:
    y = (x - mean) / pow(variance + epsilon) * gamma + beta

    This pattern corresponds to, should be fused as instance_norm.
    All of the following must be satisty:
    1) Input is rank 4 tensor
    2) Reduce operates on spatial dimensions axes=[-2, -1], or axes=[-3, -2] (a
       channel first to channel last transpose would be inserted in such case)
    3) Gamma and beta are both shape (C,) after squeeze, where C is number of channels

    |----> sub -----|                            const (0.5)
    |       ^       |                                |
    |       |       V                                V
    x ---> mean  square --> mean1 --> add_eps ---> pow       const_gamma   const_beta
    |       |                                        |             |            |
    |       V                                        V             V            V
    |----> sub1 --------------------------------> real_div --> mul_gamma --> add_beta --> ...
    """
    ops_to_remove = []
    root_var = reduce_op.x

    if root_var.shape is None:
        return False

    rank = len(root_var.shape)

    # check that root_var feeds into exactly 3 ops
    if len(root_var.child_ops) != 3:
        return False
    if root_var.op is not None and not _check_child_op_types(
        root_var.op, child_op_types=["reduce_mean", "sub", "sub"]
    ):
        return False

    # check 1st reduce_mean op
    if not _check_reduce_op(reduce_op):
        return False
    ops_to_remove.append(reduce_op)

    # check 1st sub op
    if not _check_child_op_types(reduce_op, ["sub", "sub"]):
        return False
    child_ops_reduce_mean = list(reduce_op.outputs[0].child_ops)
    reduce_mean_child_op_a = child_ops_reduce_mean[0]
    reduce_mean_child_op_b = child_ops_reduce_mean[1]
    # One of sub op directly goes square, the other one goes real_div
    if list(reduce_mean_child_op_a.outputs[0].child_ops)[0].op_type == "square":
        sub_op0 = reduce_mean_child_op_a
        sub_op1 = reduce_mean_child_op_b
    else:
        sub_op0 = reduce_mean_child_op_b
        sub_op1 = reduce_mean_child_op_a
    if not (sub_op0.x == root_var and sub_op0.y == reduce_op.outputs[0]):
        return False
    if not (sub_op1.x == root_var and sub_op1.y == reduce_op.outputs[0]):
        return False
    ops_to_remove.append(sub_op0)
    ops_to_remove.append(sub_op1)

    # check square op
    square_op = _try_get_child_op_type(sub_op0, "square")
    if square_op is None:
        return False
    ops_to_remove.append(square_op)

    # check second reduce mean
    reduce_op2 = _try_get_child_op_type(square_op, "reduce_mean")
    if not _check_reduce_op(reduce_op2):
        return False
    ops_to_remove.append(reduce_op2)

    # check add op (with epsilon)
    add_eps_op = _try_get_child_op_type(reduce_op2, "add")
    if add_eps_op is None:
        return False
    epsilon_var = (
        add_eps_op.y if add_eps_op.x == reduce_op2.outputs[0] else add_eps_op.x
    )
    if epsilon_var.val is None or len(epsilon_var.val.shape) != 0:
        return False  # must be scalar
    ops_to_remove.append(add_eps_op)

    # check pow
    pow_op = _try_get_child_op_type(add_eps_op, "pow")
    if pow_op is None:
        return False
    if pow_op.y.val is None or not np.isclose(pow_op.y.val, 0.5):
        return False
    ops_to_remove.append(pow_op)

    # check real_div
    real_div_op = _try_get_child_op_type(pow_op, "real_div")
    if real_div_op is None:
        return False
    if not (real_div_op.x == sub_op1.outputs[0] and real_div_op.y == pow_op.outputs[0]):
        return False
    ops_to_remove.append(real_div_op)

    # check mul with gamma
    mul_gamma_op = _try_get_child_op_type(real_div_op, "mul")
    if mul_gamma_op is None:
        return False
    gamma_var = (
        mul_gamma_op.y if mul_gamma_op.x == real_div_op.outputs[0] else mul_gamma_op.x
    )
    if gamma_var.val is None:
        return False
    ops_to_remove.append(mul_gamma_op)

    # check add with beta
    add_beta_op = _try_get_child_op_type(mul_gamma_op, "add")
    if add_beta_op is None:
        return False
    beta_var = (
        add_beta_op.y if add_beta_op.x == mul_gamma_op.outputs[0] else add_beta_op.x
    )
    if beta_var.val is None:
        return False
    ops_to_remove.append(add_beta_op)

    return _try_apply_transform(
        reduce_op, block, gamma_var, beta_var, epsilon_var, add_beta_op, ops_to_remove
    )


def _try_match_and_transform_pattern_3(reduce_op, block) -> bool:
    """
    Detect InstanceNorm pattern in TensorFlow-Addons.

    This pattern corresponds to, should be fused as instance_norm.
    All of the following must be satisty:
    1) Input is rank 4 tensor
    2) Reduce operates on spatial dimensions axes=[-2, -1], or axes=[-3, -2] (a
       channel first to channel last transpose would be inserted in such case)
    3) Gamma and beta are absent. Default values for gamma and beta would be used.

           |-------------------------------------------------|
           |                                                 |
           |                                                 V
    x --> mean   square --> mean1 --> add_eps --> rsqrt --> mul2 --> mul_sub
    |      |       ^                                |                   |
    |      V       |                                |                   |
    | --> sub -----|                                |                   |
    |                                               V                   V
    |--------------------------------------------> mul1 -------------> add --> ...
    """
    ops_to_remove = []
    root_var = reduce_op.x

    if root_var.shape is None:
        return False

    rank = len(root_var.shape)

    # check that root_var feeds into exactly 3 ops
    if len(root_var.child_ops) != 3:
        return False
    if root_var.op is not None and not _check_child_op_types(
        root_var.op, ["sub", "mul", "reduce_mean"]
    ):
        return False

    # check 1st reduce_mean op
    if not _check_reduce_op(reduce_op):
        return False
    ops_to_remove.append(reduce_op)

    # check 1st sub op
    if not _check_child_op_types(reduce_op, ["sub", "mul"], check_order=False):
        return False
    child_ops_reduce_mean = list(reduce_op.outputs[0].child_ops)
    reduce_mean_child_op_a = child_ops_reduce_mean[0]
    reduce_mean_child_op_b = child_ops_reduce_mean[1]
    sub_op1 = (
        reduce_mean_child_op_a
        if reduce_mean_child_op_a.op_type == "sub"
        else reduce_mean_child_op_b
    )
    if not (sub_op1.x == root_var and sub_op1.y == reduce_op.outputs[0]):
        return False
    ops_to_remove.append(sub_op1)

    # check square op
    square_op = _try_get_child_op_type(sub_op1, "square")
    if square_op is None:
        return False
    ops_to_remove.append(square_op)

    # check second reduce mean
    reduce_op2 = _try_get_child_op_type(square_op, "reduce_mean")
    if reduce_op2 is None or not _check_reduce_op(reduce_op2):
        return False
    ops_to_remove.append(reduce_op2)

    # check add op (with epsilon)
    add_eps_op = _try_get_child_op_type(reduce_op2, "add")
    if add_eps_op is None:
        return False
    epsilon_var = (
        add_eps_op.y if add_eps_op.x == reduce_op2.outputs[0] else add_eps_op.x
    )
    if epsilon_var.val is None or len(epsilon_var.val.shape) != 0:
        return False  # must be scalar
    ops_to_remove.append(add_eps_op)

    # check rsqrt
    rsqrt_op = _try_get_child_op_type(add_eps_op, "rsqrt")
    if rsqrt_op is None:
        return False
    ops_to_remove.append(rsqrt_op)

    # check mul 1
    mul_op1 = _try_get_child_op_type(rsqrt_op, "mul")
    if mul_op1 is None:
        return False
    if not (
        (mul_op1.x == root_var and mul_op1.y == rsqrt_op.outputs[0])
        or (mul_op1.x == rsqrt_op.outputs[0] and mul_op1.y == root_var)
    ):
        return False
    ops_to_remove.append(mul_op1)

    # check mul 2
    mul_op2 = _try_get_child_op_type(rsqrt_op, "mul", index=1)
    if mul_op2 is None:
        return False
    if not (
        (mul_op2.x == reduce_op.outputs[0] and mul_op2.y == rsqrt_op.outputs[0])
        or (mul_op2.x == rsqrt_op.outputs[0] and mul_op2.y == reduce_op.outputs[0])
    ):
        return False
    ops_to_remove.append(mul_op2)

    # check mul (sub)
    mul_sub_op = _try_get_child_op_type(mul_op2, "mul")
    if mul_sub_op is None:
        return False
    if mul_sub_op.y.val is None or mul_sub_op.y.val != -1:
        return False
    ops_to_remove.append(mul_sub_op)

    # check last add op
    add_op = _try_get_child_op_type(mul_sub_op, "add")
    if add_op is None:
        return False
    if not (
        (add_op.x == mul_op1.outputs[0] and add_op.y == mul_sub_op.outputs[0])
        or (add_op.x == mul_sub_op.outputs[0] and add_op.y == mul_op1.outputs[0])
    ):
        return False
    ops_to_remove.append(add_op)

    gamma_var = mb.const(
        val=np.ones(shape=(1, root_var.shape[1], 1, 1)),
        name="_fuse_layernorm_or_instancenorm_gamma",
    )
    beta_var = mb.const(
        val=np.zeros(shape=(1, root_var.shape[1], 1, 1)),
        name="_fuse_layernorm_or_instancenorm_beta",
    )

    return _try_apply_transform(
        reduce_op, block, gamma_var, beta_var, epsilon_var, add_op, ops_to_remove
    )


def _try_match_and_transform_pattern_4(reduce_op: Operation, block: Block) -> bool:
    """
    Identify the pattern:
    y = x * [gamma * rsqrt(variance + eps)] + (beta - mean * [gamma * rsqrt(variance + eps)])

    This pattern corresponds to, should be fused as instance_norm.
    All of the following must be satisty:
    1) Input is rank 4 tensor
    2) Reduce operates on spatial dimensions axes=[-2, -1], or axes=[-3, -2] (a
       channel first to channel last transpose would be inserted in such case)
    3) Gamma and beta are both shape (C,) after squeeze, where C is number of channels

    |-----------|
    |           V
    |------> mul_square1 -----> sum1 -----> mul_mean1
    |                                           |
    |                                           V
    x --> sum --> mul_mean ==> mul_square --> sub_variance --> add_eps --> rsqrt
    |                |                                                      |
    |                |                                                      V
    |                |                                                  mul_gamma
    |                |                                                      |
    |                |                                            |----------------|
    |                |                                            |                V
    |                |--------------------------------------------+-------------> mul2
    |                                                             V                |
    |----------------------------------------------------------> mul1              |
                                                                  |                V
                                                                  |             sub_beta --> add --> [...]
                                                                  |                           ^
                                                                  |---------------------------|
    """
    ops_to_remove = []
    root_var = reduce_op.x

    if root_var.shape is None:
        return False

    rank = len(root_var.shape)

    # check that root_var feeds into exactly 4 ops
    if len(root_var.child_ops) != 4:
        return False
    if root_var.op is not None and not _check_child_op_types(
        root_var.op, child_op_types=["mul", "mul", "reduce_sum", "mul"]
    ):
        return False

    # check 1st reduce_sum op
    if not _check_reduce_op(reduce_op, mode="reduce_sum"):
        return False
    ops_to_remove.append(reduce_op)

    # check mul (mean) op
    mul_mean_op = _try_get_child_op_type(reduce_op, "mul")
    if mul_mean_op is None:
        return False
    if mul_mean_op.y.shape != ():
        return False
    ops_to_remove.append(mul_mean_op)

    # check 1st mul (square) op
    if not _check_child_op_types(mul_mean_op, child_op_types=["mul", "mul", "mul"]):
        return False
    # both 0 and 1 should be mul square op
    mul_square_op = _try_get_child_op_type(mul_mean_op, "mul")
    if mul_square_op is None:
        return False
    if _try_get_child_op_type(mul_mean_op, "mul", index=1) is None:
        return False
    ops_to_remove.append(mul_square_op)

    # Check another branch

    # check 2nd mul (square) op
    # both 0 and 1 should be mul square op 1
    mul_square_op2 = list(root_var.child_ops)[0]
    ops_to_remove.append(mul_square_op2)

    # check 2nd reduce sum
    reduce_op2 = _try_get_child_op_type(mul_square_op2, child_op_type="reduce_sum")
    if not _check_reduce_op(reduce_op2, "reduce_sum"):
        return False
    ops_to_remove.append(reduce_op2)

    # check mul after 2nd reduce op
    mul_mean_op2 = _try_get_child_op_type(reduce_op2, "mul")
    if mul_mean_op2 is None:
        return False
    if mul_mean_op2.y.shape != ():
        return False
    ops_to_remove.append(mul_mean_op2)

    # check sub (variance)
    sub_variance_op = _try_get_child_op_type(mul_mean_op2, "sub")
    if sub_variance_op is None:
        return False
    if sub_variance_op.y != mul_square_op.outputs[0]:
        return False
    ops_to_remove.append(sub_variance_op)

    # check add op (epsilon)
    add_eps_op = _try_get_child_op_type(sub_variance_op, "add")
    if add_eps_op is None:
        return False
    epsilon_var = (
        add_eps_op.y if add_eps_op.x == sub_variance_op.outputs[0] else add_eps_op.x
    )
    if epsilon_var.val is None or len(epsilon_var.val.shape) != 0:
        return False  # must be scalar
    ops_to_remove.append(add_eps_op)

    # check rsqrt
    rsqrt_op = _try_get_child_op_type(add_eps_op, "rsqrt")
    if rsqrt_op is None:
        return False
    ops_to_remove.append(rsqrt_op)

    # check mul (gamma)
    mul_gamma_op = _try_get_child_op_type(rsqrt_op, "mul")
    if mul_gamma_op is None:
        return False
    gamma_var = (
        mul_gamma_op.y if mul_gamma_op.x == rsqrt_op.outputs[0] else mul_gamma_op.x
    )
    if gamma_var.val is None:
        return False
    ops_to_remove.append(mul_gamma_op)

    # check 2 muls after the gamma mul
    if not _check_child_op_types(mul_gamma_op, ["mul", "mul"]):
        return False
    mul_gamma_child_ops = list(mul_gamma_op.outputs[0].child_ops)
    mul_op1 = mul_gamma_child_ops[0]
    mul_op2 = mul_gamma_child_ops[1]
    mul_op1_other_var = mul_op1.x if mul_op1.y == mul_gamma_op.outputs[0] else mul_op1.y
    mul_op2_other_var = mul_op2.x if mul_op2.y == mul_gamma_op.outputs[0] else mul_op2.y
    if not (
        (mul_op1_other_var == root_var and mul_op2_other_var == mul_square_op.x)
        or (mul_op1_other_var == mul_square_op.x and mul_op2_other_var == root_var)
    ):
        return False
    if mul_op1_other_var == root_var:
        mul_op1, mul_op2 = mul_op1, mul_op2
    else:
        mul_op2, mul_op1 = mul_op1, mul_op2
    ops_to_remove.append(mul_op1)
    ops_to_remove.append(mul_op2)

    # check sub with beta
    sub_beta_op = _try_get_child_op_type(mul_op2, "sub")
    if sub_beta_op is None:
        return False
    if sub_beta_op.y != mul_op2.outputs[0]:
        return False
    beta_var = sub_beta_op.x
    if beta_var.val is None:
        return False
    ops_to_remove.append(sub_beta_op)

    # check last add op
    add_op = _try_get_child_op_type(sub_beta_op, "add")
    if add_op is None:
        return False
    if not (
        (add_op.x == mul_op1.outputs[0] and add_op.y == sub_beta_op.outputs[0])
        or (add_op.y == mul_op1.outputs[0] and add_op.x == sub_beta_op.outputs[0])
    ):
        return False
    ops_to_remove.append(add_op)

    return _try_apply_transform(
        reduce_op, block, gamma_var, beta_var, epsilon_var, add_op, ops_to_remove
    )


def _fuse_layernorm_or_instancenorm_block(block: Block):
    fusion_status = False
    for i, op in enumerate(list(block.operations)):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = _fuse_layernorm_or_instancenorm_block(b)
        if len(op.blocks) > 0:
            continue

        # start pattern match if reduce_mean op is encountered
        if op.op_type == "reduce_mean":
            with block:
                if fusion_status is False:
                    fusion_status = _try_match_and_transform_pattern_1(op, block)
                if fusion_status is False:
                    fusion_status = _try_match_and_transform_pattern_2(op, block)
                if fusion_status is False:
                    fusion_status = _try_match_and_transform_pattern_3(op, block)
            # has to break as the downstream iterator is affected.
            if fusion_status:
                return fusion_status
        elif op.op_type == "reduce_sum":
            with block:
                if fusion_status is False:
                    fusion_status = _try_match_and_transform_pattern_4(op, block)
            # has to break as the downstream iterator is affected.
            if fusion_status:
                return fusion_status
    return fusion_status


@register_pass(namespace="common")
class fuse_layernorm_or_instancenorm(AbstractGraphPass):
    """
    A graph optimization pass on PyMIL to detect and fuse several different
    vairants of layer_norm or instance_norm. Pattern 1 is corresponding to
    either layer_norm or instance_norm. Pattern 2-4 are instance_norm.

    :param prog: PyMIL Program to work on.
    """
    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                if DEBUG:
                    import graphviz

                    graphviz.Source(
                        f.get_dot_string(highlight_debug_op_types=["instance_norm"],)
                    ).view(filename="/tmp/block_before_fuse_layernorm_or_instancenorm")
                logging.debug(
                    "Block before fuse_layernorm_or_instancenorm transform:\n{}".format(f)
                )

                block_changed = _fuse_layernorm_or_instancenorm_block(f)

                if DEBUG:
                    graphviz.Source(
                        f.get_dot_string(highlight_debug_op_types=["instance_norm"],)
                    ).view(filename="/tmp/block_after_fuse_layernorm_or_instancenorm")

                logging.debug(
                    "Block after fuse_layernorm_or_instancenorm transform:\n{}".format(f)
                )
