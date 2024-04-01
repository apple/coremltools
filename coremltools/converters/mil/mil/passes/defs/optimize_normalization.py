#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import List, Optional

import numpy as np

from coremltools import _logger as logger
from coremltools.converters.mil.mil import Block
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, Program, Var
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import (
    _check_no_output_connection,
    block_context_manager,
)
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="common")
class fuse_layernorm_or_instancenorm(AbstractGraphPass):
    """
    A graph optimization pass on PyMIL to detect and fuse several variants of ``layer_norm`` or
    ``instance_norm``. Pattern 1 corresponds to either ``layer_norm`` or ``instance_norm``. Patterns 2-4
    are ``instance_norm``. Pattern 5 is ``layer_norm``. You can find these patterns in the methods for
    this class in the source code. To quickly view the source code, click the **[source]** button at
    the end of the class definition.

    """

    _DEBUG = False  # set to true to plot the block before and after the transformation

    def apply(self, prog: Program):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                if self._DEBUG:
                    import graphviz

                    graphviz.Source(
                        f.get_dot_string(
                            highlight_debug_op_types=["instance_norm"],
                        )
                    ).view(filename="/tmp/block_before_fuse_layernorm_or_instancenorm")
                logger.debug("Block before fuse_layernorm_or_instancenorm transform:\n{}".format(f))

                block_changed = self._fuse_layernorm_or_instancenorm_block(f)

                if self._DEBUG:
                    graphviz.Source(
                        f.get_dot_string(
                            highlight_debug_op_types=["instance_norm"],
                        )
                    ).view(filename="/tmp/block_after_fuse_layernorm_or_instancenorm")

                logger.debug("Block after fuse_layernorm_or_instancenorm transform:\n{}".format(f))

    @staticmethod
    def _check_reduce_op(reduce_op: Operation, mode: str = "reduce_mean") -> bool:
        """
		Check whether or not the ``reduction`` op satisfies following conditions:

			- Mode is expected.
			- Does not change rank (``keep_dims`` is ``True``).
			- The ``axes`` is known at compile time.

		Parameters
		----------

		param reduce_op : ``reduce_op`` to check on.

		param mode : ``reduce`` mode

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

    @staticmethod
    def _check_child_op_types(
        op: Operation, child_op_types: List[str], check_order: bool = True
    ) -> bool:
        """
        Returns ``True`` for child op types matching ``child_op_types``, otherwise returns ``False``.

                Parameters
                ----------

        param op : Current op.

        param child_op_type : Expected child op type.

        param check_order : Ensure child in given order, defaults to ``True``.
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

    @staticmethod
    def _try_get_child_op_type(
        op: Operation, child_op_type: str, index: int = 0
    ) -> Optional[Operation]:
        """
        Returns child op if type matches, otherwise returns ``None``.

                Parameters
                ----------

        param op : Current op.

        param child_op_type : Expected child op type.

        param index : Child op index.
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

    @staticmethod
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

        gamma_rank = gamma_var.rank if gamma_var is not None else -1
        beta_rank = beta_var.rank if beta_var is not None else -1

        if gamma_rank == len(axes) and beta_rank == len(axes):
            # axes for layer_norm must be [-1] or [-1, -2] or [-1, -2, -3] and so on
            if negative_axes == list(range(-len(negative_axes), 0)):
                is_layernorm = True

        if rank == 4 and negative_axes == [-3]:
            is_layernorm = (gamma_var is None and beta_var is None) or (gamma_rank == 1 and beta_rank == 1)

            if gamma_var:
                ops_to_remove.append(gamma_var.op)
                gamma_var = gamma_var.val
            else:
                gamma_var = None

            if beta_var:
                ops_to_remove.append(beta_var.op)
                beta_var = beta_var.val
            else:
                beta_var = None

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
            ops_to_remove.extend([gamma_var.op, beta_var.op])
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

    def _try_match_and_transform_pattern_1(self, reduce_op, block) -> bool:
        """
        Identify the pattern:

        ``y = gamma * (x - mean) / sqrt(variance + epsilon) + beta``

        ``y = x * [gamma * rsqrt(variance + eps)] + (beta - mean * [gamma * rsqrt(variance + eps)])``

        .. code-block::

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

        This pattern corresponds to either ``layer_norm`` or ``instance_norm``.

        It is ``instance_norm`` if all of the following are true:
            - ``input`` is rank 4.
            - ``axes`` of ``reduce_mean`` is ``[-2, -1]`` or ``[-3, -2]``
              (when ``[-3, -2]``, a channel first to channel last transpose would be inserted).
            - ``gamma`` and ``beta`` are rank 1, after ``squeeze``.

        It is ``layer_norm`` if all of the following are true:
            - ``axes`` is either ``[-1]``, ``[-1, -2]``, or ``[-1, -2, -3]``, and so on.
            - ``rank`` of ``gamma`` and ``beta`` is equal to the length of the ``axes``.

        """
        ops_to_remove = []
        root_var = reduce_op.x

        if root_var.shape is None:
            return False

        # check that root_var feeds into exactly 3 ops
        if len(list(root_var.child_ops)) != 3:
            return False
        if root_var.op is not None and not self._check_child_op_types(
            root_var.op, child_op_types=["reduce_mean", "sub", "mul"]
        ):
            return False

        # check 1st reduce_mean op
        if not self._check_reduce_op(reduce_op):
            return False
        ops_to_remove.append(reduce_op)

        # check 1st sub op
        if not self._check_child_op_types(reduce_op, ["sub", "mul"], check_order=False):
            return False
        child_ops_reduce_mean = list(reduce_op.outputs[0].child_ops)
        op_a = child_ops_reduce_mean[0]
        op_b = child_ops_reduce_mean[1]
        sub_op1 = op_a if op_a.op_type == "sub" else op_b
        if not (sub_op1.x == root_var and sub_op1.y == reduce_op.outputs[0]):
            return False
        ops_to_remove.append(sub_op1)

        # check square op
        square_op = self._try_get_child_op_type(sub_op1, "square")
        if square_op is None:
            return False
        ops_to_remove.append(square_op)

        # check second reduce mean
        reduce_op2 = self._try_get_child_op_type(square_op, "reduce_mean")
        if not self._check_reduce_op(reduce_op2):
            return False
        ops_to_remove.append(reduce_op2)

        # check add op (with epsilon)
        add_op1 = self._try_get_child_op_type(reduce_op2, "add")
        if add_op1 is None:
            return False
        epsilon_var = add_op1.y if add_op1.x == reduce_op2.outputs[0] else add_op1.x
        if epsilon_var.val is None or len(epsilon_var.val.shape) != 0:
            return False  # must be scalar
        ops_to_remove.append(add_op1)

        # check rsqrt
        rsqrt_op = self._try_get_child_op_type(add_op1, "rsqrt")
        if rsqrt_op is None:
            return False
        ops_to_remove.append(rsqrt_op)

        # check mul (gamma)
        mul_op1 = self._try_get_child_op_type(rsqrt_op, "mul")
        if mul_op1 is None:
            return False
        gamma_var = mul_op1.y if mul_op1.x == rsqrt_op.outputs[0] else mul_op1.x
        if gamma_var.val is None:
            return False
        ops_to_remove.append(mul_op1)

        # check 2 muls after the gamma mul
        if not self._check_child_op_types(mul_op1, ["mul", "mul"]):
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
        sub_op2 = self._try_get_child_op_type(mul_mean_op, "sub")
        if sub_op2 is None:
            return False
        if sub_op2.y != mul_mean_op.outputs[0]:
            return False
        beta_var = sub_op2.x
        if beta_var.val is None:
            return False
        ops_to_remove.append(sub_op2)

        # check last add op
        add_op2 = self._try_get_child_op_type(sub_op2, "add")
        if add_op2 is None:
            return False
        if not (add_op2.x == mul_root_op.outputs[0] or add_op2.y == mul_root_op.outputs[0]):
            return False
        ops_to_remove.append(add_op2)

        return self._try_apply_transform(
            reduce_op, block, gamma_var, beta_var, epsilon_var, add_op2, ops_to_remove
        )

    def _try_match_and_transform_pattern_2(self, reduce_op, block) -> bool:
        """
        Identify the pattern:

        ``y = (x - mean) / pow(variance + epsilon) * gamma + beta``

        This pattern corresponds to, and should be fused as, ``instance_norm``.

        All of the following conditions must be satisfied:

        1. ``input`` is rank 4 tensor.
        2. ``reduce`` operates on spatial dimensions ``axes=[-2, -1]``, or ``axes=[-3, -2]`` (a
           channel first to channel last transpose would be inserted in such cases).
        3. ``gamma`` and ``beta`` are both shape ``(C,)`` after ``squeeze``, where ``C`` is number of channels.

        .. code-block::

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

        # check that root_var feeds into exactly 3 ops
        if len(root_var.child_ops) != 3:
            return False
        if root_var.op is not None and not self._check_child_op_types(
            root_var.op, child_op_types=["reduce_mean", "sub", "sub"]
        ):
            return False

        # check 1st reduce_mean op
        if not self._check_reduce_op(reduce_op):
            return False
        ops_to_remove.append(reduce_op)

        # check 1st sub op
        if not self._check_child_op_types(reduce_op, ["sub", "sub"]):
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
        square_op = self._try_get_child_op_type(sub_op0, "square")
        if square_op is None:
            return False
        ops_to_remove.append(square_op)

        # check second reduce mean
        reduce_op2 = self._try_get_child_op_type(square_op, "reduce_mean")
        if not self._check_reduce_op(reduce_op2):
            return False
        ops_to_remove.append(reduce_op2)

        # check add op (with epsilon)
        add_eps_op = self._try_get_child_op_type(reduce_op2, "add")
        if add_eps_op is None:
            return False
        epsilon_var = add_eps_op.y if add_eps_op.x == reduce_op2.outputs[0] else add_eps_op.x
        if epsilon_var.val is None or len(epsilon_var.val.shape) != 0:
            return False  # must be scalar
        ops_to_remove.append(add_eps_op)

        # check pow
        pow_op = self._try_get_child_op_type(add_eps_op, "pow")
        if pow_op is None:
            return False
        if pow_op.y.val is None or not np.isclose(pow_op.y.val, 0.5):
            return False
        ops_to_remove.append(pow_op)

        # check real_div
        real_div_op = self._try_get_child_op_type(pow_op, "real_div")
        if real_div_op is None:
            return False
        if not (real_div_op.x == sub_op1.outputs[0] and real_div_op.y == pow_op.outputs[0]):
            return False
        ops_to_remove.append(real_div_op)

        # check mul with gamma
        mul_gamma_op = self._try_get_child_op_type(real_div_op, "mul")
        if mul_gamma_op is None:
            return False
        gamma_var = mul_gamma_op.y if mul_gamma_op.x == real_div_op.outputs[0] else mul_gamma_op.x
        if gamma_var.val is None:
            return False
        ops_to_remove.append(mul_gamma_op)

        # check add with beta
        add_beta_op = self._try_get_child_op_type(mul_gamma_op, "add")
        if add_beta_op is None:
            return False
        beta_var = add_beta_op.y if add_beta_op.x == mul_gamma_op.outputs[0] else add_beta_op.x
        if beta_var.val is None:
            return False
        ops_to_remove.append(add_beta_op)

        return self._try_apply_transform(
            reduce_op, block, gamma_var, beta_var, epsilon_var, add_beta_op, ops_to_remove
        )

    def _try_match_and_transform_pattern_3(self, reduce_op, block) -> bool:
        """
        Detect ``InstanceNorm`` pattern in TensorFlow-Addons.

        This pattern corresponds to, and should be fused as, ``instance_norm``.

        All of the following conditions must be satisfied:

        1. ``input`` is rank 4 tensor.
        2. ``reduce`` operates on spatial dimensions ``axes=[-2, -1]``, or ``axes=[-3, -2]`` (a
           channel first to channel last transpose would be inserted in such cases).
        3. ``gamma`` and ``beta`` are absent. Default values for ``gamma`` and ``beta`` would be used.

        .. code-block::

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

        # check that root_var feeds into exactly 3 ops
        if len(root_var.child_ops) != 3:
            return False
        if root_var.op is not None and not self._check_child_op_types(
            root_var.op, ["sub", "mul", "reduce_mean"]
        ):
            return False

        # check 1st reduce_mean op
        if not self._check_reduce_op(reduce_op):
            return False
        ops_to_remove.append(reduce_op)

        # check 1st sub op
        if not self._check_child_op_types(reduce_op, ["sub", "mul"], check_order=False):
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
        square_op = self._try_get_child_op_type(sub_op1, "square")
        if square_op is None:
            return False
        ops_to_remove.append(square_op)

        # check second reduce mean
        reduce_op2 = self._try_get_child_op_type(square_op, "reduce_mean")
        if reduce_op2 is None or not self._check_reduce_op(reduce_op2):
            return False
        ops_to_remove.append(reduce_op2)

        # check add op (with epsilon)
        add_eps_op = self._try_get_child_op_type(reduce_op2, "add")
        if add_eps_op is None:
            return False
        epsilon_var = add_eps_op.y if add_eps_op.x == reduce_op2.outputs[0] else add_eps_op.x
        if epsilon_var.val is None or len(epsilon_var.val.shape) != 0:
            return False  # must be scalar
        ops_to_remove.append(add_eps_op)

        # check rsqrt
        rsqrt_op = self._try_get_child_op_type(add_eps_op, "rsqrt")
        if rsqrt_op is None:
            return False
        ops_to_remove.append(rsqrt_op)

        # check mul 1
        mul_op1 = self._try_get_child_op_type(rsqrt_op, "mul")
        if mul_op1 is None:
            return False
        if not (
            (mul_op1.x == root_var and mul_op1.y == rsqrt_op.outputs[0])
            or (mul_op1.x == rsqrt_op.outputs[0] and mul_op1.y == root_var)
        ):
            return False
        ops_to_remove.append(mul_op1)

        # check mul 2
        mul_op2 = self._try_get_child_op_type(rsqrt_op, "mul", index=1)
        if mul_op2 is None:
            return False
        if not (
            (mul_op2.x == reduce_op.outputs[0] and mul_op2.y == rsqrt_op.outputs[0])
            or (mul_op2.x == rsqrt_op.outputs[0] and mul_op2.y == reduce_op.outputs[0])
        ):
            return False
        ops_to_remove.append(mul_op2)

        # check mul (sub)
        mul_sub_op = self._try_get_child_op_type(mul_op2, "mul")
        if mul_sub_op is None:
            return False
        if mul_sub_op.y.val is None or mul_sub_op.y.val != -1:
            return False
        ops_to_remove.append(mul_sub_op)

        # check last add op
        add_op = self._try_get_child_op_type(mul_sub_op, "add")
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

        return self._try_apply_transform(
            reduce_op, block, gamma_var, beta_var, epsilon_var, add_op, ops_to_remove
        )

    def _try_match_and_transform_pattern_4(self, reduce_op: Operation, block: Block) -> bool:
        """
        Identify the pattern:

        ``y = x * [gamma * rsqrt(variance + eps)] + (beta - mean * [gamma * rsqrt(variance + eps)])``

        This pattern corresponds to, and should be fused as, ``instance_norm``.

        All of the following conditions must be satisfied:

        1. ``input`` is rank 4 tensor.
        2. ``reduce`` operates on spatial dimensions ``axes=[-2, -1]`` or ``axes=[-3, -2]`` (a
           channel first to channel last transpose would be inserted in such cases).
        3. ``gamma`` and ``beta`` are both shape ``(C,)`` after ``squeeze``, where ``C`` is number of channels.

        .. code-block::

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

        # check that root_var feeds into exactly 4 ops
        if len(root_var.child_ops) != 4:
            return False

        if (
            root_var.op is not None
            and not self._check_child_op_types(
                root_var.op, child_op_types=["mul", "mul", "reduce_sum", "mul"]
            )
            and not self._check_child_op_types(
                # The _check_child_op_types checks for the exact order of the child_ops.
                root_var.op,
                child_op_types=["mul", "mul", "mul", "reduce_sum"],
            )
        ):
            return False

        # check 1st reduce_sum op
        if not self._check_reduce_op(reduce_op, mode="reduce_sum"):
            return False
        ops_to_remove.append(reduce_op)

        # check mul (mean) op
        mul_mean_op = self._try_get_child_op_type(reduce_op, "mul")
        if mul_mean_op is None:
            return False
        if mul_mean_op.y.shape != ():
            return False
        ops_to_remove.append(mul_mean_op)

        # check 1st mul (square) op
        if not self._check_child_op_types(mul_mean_op, child_op_types=["mul", "mul", "mul"]):
            return False
        # both 0 and 1 should be mul square op
        mul_square_op = self._try_get_child_op_type(mul_mean_op, "mul")
        if mul_square_op is None:
            return False
        if self._try_get_child_op_type(mul_mean_op, "mul", index=1) is None:
            return False
        ops_to_remove.append(mul_square_op)

        # Check another branch

        # check 2nd mul (square) op
        # both 0 and 1 should be mul square op 1
        mul_square_op2 = list(root_var.child_ops)[0]
        ops_to_remove.append(mul_square_op2)

        # check 2nd reduce sum
        reduce_op2 = self._try_get_child_op_type(mul_square_op2, child_op_type="reduce_sum")
        if not self._check_reduce_op(reduce_op2, "reduce_sum"):
            return False
        ops_to_remove.append(reduce_op2)

        # check mul after 2nd reduce op
        mul_mean_op2 = self._try_get_child_op_type(reduce_op2, "mul")
        if mul_mean_op2 is None:
            return False
        if mul_mean_op2.y.shape != ():
            return False
        ops_to_remove.append(mul_mean_op2)

        # check sub (variance)
        sub_variance_op = self._try_get_child_op_type(mul_mean_op2, "sub")
        if sub_variance_op is None:
            return False
        if sub_variance_op.y != mul_square_op.outputs[0]:
            return False
        ops_to_remove.append(sub_variance_op)

        # check add op (epsilon)
        add_eps_op = self._try_get_child_op_type(sub_variance_op, "add")
        if add_eps_op is None:
            return False
        epsilon_var = add_eps_op.y if add_eps_op.x == sub_variance_op.outputs[0] else add_eps_op.x
        if epsilon_var.val is None or len(epsilon_var.val.shape) != 0:
            return False  # must be scalar
        ops_to_remove.append(add_eps_op)

        # check rsqrt
        rsqrt_op = self._try_get_child_op_type(add_eps_op, "rsqrt")
        if rsqrt_op is None:
            return False
        ops_to_remove.append(rsqrt_op)

        # check mul (gamma)
        mul_gamma_op = self._try_get_child_op_type(rsqrt_op, "mul")
        if mul_gamma_op is None:
            return False
        gamma_var = mul_gamma_op.y if mul_gamma_op.x == rsqrt_op.outputs[0] else mul_gamma_op.x
        if gamma_var.val is None:
            return False
        ops_to_remove.append(mul_gamma_op)

        # check 2 muls after the gamma mul
        if not self._check_child_op_types(mul_gamma_op, ["mul", "mul"]):
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
        sub_beta_op = self._try_get_child_op_type(mul_op2, "sub")
        if sub_beta_op is None:
            return False
        if sub_beta_op.y != mul_op2.outputs[0]:
            return False
        beta_var = sub_beta_op.x
        if beta_var.val is None:
            return False
        ops_to_remove.append(sub_beta_op)

        # check last add op
        add_op = self._try_get_child_op_type(sub_beta_op, "add")
        if add_op is None:
            return False
        if not (
            (add_op.x == mul_op1.outputs[0] and add_op.y == sub_beta_op.outputs[0])
            or (add_op.y == mul_op1.outputs[0] and add_op.x == sub_beta_op.outputs[0])
        ):
            return False
        ops_to_remove.append(add_op)

        return self._try_apply_transform(
            reduce_op, block, gamma_var, beta_var, epsilon_var, add_op, ops_to_remove
        )

    def _try_match_and_transform_pattern_5(self, reduce_op, block) -> bool:
        """
        Detect BC1S ``LayerNorm`` pattern as in ml-ane-transformers.

        Identify two patterns, the first:

        ``y = (x - mean(x)) * rsqrt(variance(X) + eps)``

        ``y = (x - mean(x)) * rsqrt(mean((x - mean(x))^2) + eps)``

        .. code-block::

            x --> reduce_mean --|
            |                   |  |---|
            |                   V  |   V
            |----------------> sub --> mul --> reduce_mean --> add(epsilon) --> rsqrt
                                |                                                   |
                                |                                                   V
                                |-------------------------------------------------> mul --> [...]

        If the optional elementwise weight and bias are set, the second pattern is:

        ``y = [(x - mean(x)) * rsqrt(mean((x - mean(x))^2) + eps) + beta] * gamma``

        Note that this is different from the torch and MIL definitions of beta and gamma
        so beta is be scaled by gamma and applied before it.

        .. code-block::

            x --> reduce_mean --|
            |                   |  |---|
            |                   V  |   V
            |----------------> sub --> mul --> reduce_mean --> add(epsilon) --> rsqrt
                                |                                                   |
                                |                                                   V
                                |-------------------------------------------------> mul
                                                                                    |
                                                                                    V
                                                                                add(beta)
                                                                                    |
                                                                                    V
                                                                                mul(gamma) --> [...]

        These pattern corresponds to a specific ``layer_norm``:
            - ``rank`` is 4.
            - ``axes`` is ``[1]``
            - ``gamma`` and ``beta`` are applied as in ml-ane-transformers, in the opposite order of torch.

         """
        ops_to_remove = []
        root_var = reduce_op.x

        # check that root_var feeds into at least 2 ops
        if len(list(root_var.child_ops)) < 2:
            return False

        # Do not enforce that the only child ops are reduce_mean and sub as in other
        # patterns. There are models where the root op is used after the layer norm.

        # check 1st reduce_mean op
        if not self._check_reduce_op(reduce_op):
            return False
        if len(reduce_op.axes.val) != 1 or reduce_op.axes.val != [1] or not reduce_op.keep_dims.val:
            return False
        ops_to_remove.append(reduce_op)

        # check 1st sub op
        if not self._check_child_op_types(reduce_op, ["sub"], check_order=False):
            return False
        child_ops_reduce_mean = list(reduce_op.outputs[0].child_ops)
        sub_op1 = child_ops_reduce_mean[0]
        if sub_op1 is None or not self._check_child_op_types(
            sub_op1, child_op_types=["mul", "mul", "mul"]
        ):
            return False
        if not (sub_op1.x == root_var and sub_op1.y == reduce_op.outputs[0]):
            return False
        ops_to_remove.append(sub_op1)

        # check mul op (equivalent to a square op)
        square_op = self._try_get_child_op_type(sub_op1, "mul")
        if square_op is None or not self._check_child_op_types(
            square_op, child_op_types=["reduce_mean"]
        ):
            return False
        if square_op.x != square_op.y:
            return False
        ops_to_remove.append(square_op)

        # check second reduce mean
        reduce_op2 = self._try_get_child_op_type(square_op, "reduce_mean")
        if not self._check_reduce_op(reduce_op2) or not self._check_child_op_types(
            reduce_op2, child_op_types=["add"]
        ):
            return False
        if len(reduce_op2.axes.val) != 1 or reduce_op2.axes.val != [1] or not reduce_op2.keep_dims.val:
            return False
        ops_to_remove.append(reduce_op2)

        # check add op (with epsilon)
        add_op1 = self._try_get_child_op_type(reduce_op2, "add")
        if add_op1 is None or not self._check_child_op_types(
            add_op1, child_op_types=["rsqrt"]
        ):
            return False
        epsilon_var = add_op1.y if add_op1.x == reduce_op2.outputs[0] else add_op1.x
        if epsilon_var.val is None or len(epsilon_var.val.shape) != 0:
            return False  # must be scalar
        ops_to_remove.append(add_op1)

        # check rsqrt
        rsqrt_op = self._try_get_child_op_type(add_op1, "rsqrt")
        if rsqrt_op is None or not self._check_child_op_types(
            rsqrt_op, child_op_types=["mul"]
        ):
            return False
        ops_to_remove.append(rsqrt_op)

        # Last op in pattern if there is no elementwise affine.
        mul_op = self._try_get_child_op_type(rsqrt_op, "mul")
        if mul_op is None:
            return False
        if mul_op.y != sub_op1.outputs[0] and mul_op.x != sub_op1.outputs[0]:
            return False
        ops_to_remove.append(mul_op)

        # Default values if no gamma or beta ops.
        end_op = mul_op
        gamma_var = None
        beta_var = None

        add_beta_op = self._try_get_child_op_type(mul_op, "add")
        mul_gamma_op = self._try_get_child_op_type(add_beta_op, "mul")

        has_beta_and_gamma = add_beta_op is not None and mul_gamma_op is not None

        # mul_op cannot be used except as an input to add_beta_op.
        if has_beta_and_gamma and not self._check_child_op_types(
            mul_op, child_op_types=["add"]
        ):
            # It would be possible to fuse this pattern as:
            # layer_norm(x, gamma=None, beta=None) -> add(beta) -> mul(gamma) -> ...
            #                                      |-> other mul_op child ops
            # For simplicity don't handle this edge case.
            return False

        # add_beta_op cannot be used except as an input to mul_gamma_op.
        if has_beta_and_gamma and not self._check_child_op_types(
            add_beta_op, child_op_types=["mul"]
        ):
            # It would be possible to fuse this pattern as:
            # layer_norm(x, gamma=None, beta=None) -> add(beta) -> mul(gamma) -> ...
            #                                                   |-> other add_beta_op child ops
            # For simplicity don't handle this edge case.
            return False

        if add_beta_op is None and mul_gamma_op is None:
            # Gamma and beta are optional in layer_norm.
            pass
        elif add_beta_op is None or mul_gamma_op is None:
            # If only one of gamma or beta is present, they could
            # be folded into the layer_norm op. For simplicity
            # don't handle this edge case.
            return False

        if has_beta_and_gamma:
            beta_var = add_beta_op.y if add_beta_op.x == mul_op.outputs[0] else add_beta_op.x

            gamma_var = mul_gamma_op.y if mul_gamma_op.x == add_beta_op.outputs[0] else mul_gamma_op.x
            gamma_var = mb.const(
                val=np.squeeze(gamma_var.val),
                name="_fuse_layernorm_gamma",
            )

            # Scale beta by gamma. Note: this un-scaling introduces a small amount
            # of precision loss.
            # https://github.com/apple/ml-ane-transformers/blob/da64000fa56cc85b0859bc17cb16a3d753b8304a/ane_transformers/huggingface/distilbert.py#L31
            beta_var = mb.const(
                val=np.squeeze(beta_var.val) * gamma_var.val,
                name="_fuse_layernorm_beta"
            )

            ops_to_remove.extend([add_beta_op, mul_gamma_op])
            end_op = mul_gamma_op

        return self._try_apply_transform(
            reduce_op, block, gamma_var, beta_var, epsilon_var, end_op, ops_to_remove
        )

    @block_context_manager
    def _fuse_layernorm_or_instancenorm_block(self, block: Block):
        fusion_occurred = False
        for op in list(block.operations):
            if op.enclosing_block is None:
                continue

            for b in op.blocks:
                block_changed = True
                while block_changed:
                    block_changed = self._fuse_layernorm_or_instancenorm_block(b)
            if len(op.blocks) > 0:
                continue

            # start pattern match if reduce_mean op is encountered
            if op.op_type == "reduce_mean":
                if self._try_match_and_transform_pattern_1(op, block):
                    fusion_occurred = True
                elif self._try_match_and_transform_pattern_2(op, block):
                    fusion_occurred = True
                elif self._try_match_and_transform_pattern_3(op, block):
                    fusion_occurred = True
                elif self._try_match_and_transform_pattern_5(op, block):
                    fusion_occurred = True
            elif op.op_type == "reduce_sum":
                if self._try_match_and_transform_pattern_4(op, block):
                    fusion_occurred = True
        return fusion_occurred
