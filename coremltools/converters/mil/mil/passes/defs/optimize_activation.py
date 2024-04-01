#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.experimental.passes.generic_pass_infrastructure import (
    fuse_all_blocks,
)
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import (
    _check_child_op_type,
    _check_var_scalar_value,
    _check_var_scalar_value_in_interval,
    block_context_manager,
)
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="common")
class fuse_gelu_exact(AbstractGraphPass):
    """
    Identify the pattern that corresponds to the exact version of ``gelu``, and replace it with a single
    ``gelu`` layer with ``mode=EXACT``. The pattern is ``y = 0.5 * x * (1 + erf (x / srqt (2))``, which
    can be represented by one of the following:

    .. code-block::

        (1)
            [...] ----> div (1.414) ---> erf ---> add (1) -----> mul (0.5) ---> mul ---> [...]
              |                                                                  ^
              |                                                                  |
              |-------------------------------------------------------------------

        (2)
            [...] ----> div (1.414) ---> erf ---> add (1) -----> mul ---> mul (0.5) ---> [...]
              |                                                   ^
              |                                                   |
              |----------------------------------------------------

        (3)
            [...] ----> div (1.414) ---> erf ---> add (1) -----> mul ------> [...]
              |                                                   ^
              |                                                   |
              |---------------> mul(0.5) --------------------------

        All of them are converted to:
            [...] ----> gelu (mode=EXACT) ---> [...]
    """

    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self._fuse_gelu_exact_block(f)

    @staticmethod
    def _try_to_transform(op, block):
        ops_to_remove = []
        if op.x.val is None and op.y.val is None:
            return False

        # check either the op is mul(1/sqrt(2)) or real_div(sqrt(2))
        root_var = op.x if op.y.val is not None else op.y
        if op.op_type == "real_div":
            if not _check_var_scalar_value(op.y, 2**0.5):
                return False
        elif op.op_type == "mul":
            if not (
                _check_var_scalar_value(op.x, 2**-0.5) or _check_var_scalar_value(op.y, 2**-0.5)
            ):
                return False
        ops_to_remove.append(op)

        # check if the child op is erf
        if not _check_child_op_type(op, "erf"):
            return False
        erf_op = list(op.outputs[0].child_ops)[0]
        ops_to_remove.append(erf_op)

        # check if the child op is add
        if not _check_child_op_type(erf_op, "add"):
            return False
        add_op = list(erf_op.outputs[0].child_ops)[0]
        if not (_check_var_scalar_value(add_op.x, 1) or _check_var_scalar_value(add_op.y, 1)):
            return False
        ops_to_remove.append(add_op)

        # check if the child op is mul
        if not _check_child_op_type(add_op, "mul"):
            return False
        mul_op = list(add_op.outputs[0].child_ops)[0]

        # now we have two case:
        # (1) first mul by 0.5 and by the root var
        if _check_var_scalar_value(mul_op.x, 0.5) or _check_var_scalar_value(mul_op.y, 0.5):
            ops_to_remove.append(mul_op)
            if not _check_child_op_type(mul_op, "mul"):
                return False
            mul_op_2 = list(mul_op.outputs[0].child_ops)[0]
            if not (mul_op_2.x == root_var or mul_op_2.y == root_var):
                return False
            ops_to_remove.append(mul_op_2)

        # (2) first mul by the root var and then mul by 0.5
        elif mul_op.x == root_var or mul_op.y == root_var:
            ops_to_remove.append(mul_op)
            if not _check_child_op_type(mul_op, "mul"):
                return False
            mul_op_2 = list(mul_op.outputs[0].child_ops)[0]
            if not (
                _check_var_scalar_value(mul_op_2.x, 0.5) or _check_var_scalar_value(mul_op_2.y, 0.5)
            ):
                return False
            ops_to_remove.append(mul_op_2)

        else:
            other_parent_op = mul_op.x.op if mul_op.y == add_op.outputs[0] else mul_op.y.op
            if other_parent_op.op_type != "mul":
                return False
            if not (
                _check_var_scalar_value(other_parent_op.x, 0.5)
                or _check_var_scalar_value(other_parent_op.y, 0.5)
            ):
                return False
            if not (other_parent_op.x == root_var or other_parent_op.y == root_var):
                return False
            ops_to_remove.append(other_parent_op)
            ops_to_remove.append(mul_op)
            mul_op_2 = mul_op

        # check that none of the op in this pattern is connected to the output
        # (except the last mul op)
        for op in ops_to_remove[:-1]:
            for out in op.outputs:
                if out in block.outputs:
                    return False

        # remove all the ops, and replace with a gelu op
        out_name = mul_op_2.outputs[0].name
        x = mb.gelu(x=root_var, mode="EXACT", name=out_name, before_op=op)

        mul_op_2.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=mul_op_2, old_var=mul_op_2.outputs[0], new_var=x
        )
        # Remove all the ops at once
        block.remove_ops(ops_to_remove)
        return True

    @block_context_manager
    def _fuse_gelu_exact_block(self, block):
        fusion_occurred = False
        for op in list(block.operations):
            if op.enclosing_block is None:
                continue

            for b in op.blocks:
                block_changed = True
                while block_changed:
                    block_changed = self._fuse_gelu_exact_block(b)

            if len(op.blocks) > 0:
                # This op can't be real_div or mul
                continue

            if op.op_type in ["mul", "real_div"]:
                if self._try_to_transform(op, block):
                    fusion_occurred = True
        return fusion_occurred


@register_pass(namespace="common")
class fuse_gelu_tanh_approximation(AbstractGraphPass):
    """
    Identify the pattern that corresponds to the ``tanh`` approximate version of ``gelu``, and replace it
    with a single ``gelu`` layer with ``mode=TANH_APPROXIMATION``.

    The implementation of this pass uses the generic graph pattern matching and transform algorithm
    implemented in ``coremltools.converters.mil.experimental.passes.generic_pass_infrastructure`` and
    documented in ``coremltools/converters/mil/experimental/passes/readme.md``.

    `Graph for` ``get_gelu_pattern1()``

    ``y = x * (0.5 * (tanh(((.0447)x^3 + x ) * sqrt(2/pi)) + 1))``

    .. code-block::

            [...] -----> pow (3) ----> mul (.044715) ---> add -----> mul (sqrt(2/pi)) ---> tanh ----> add (1) ----> mul (0.5) -----> mul ---> [...]
              |                                            ^                                                                          ^
              |                                            |                                                                          |
              |------------------------------------------------------------------------------------------------------------------------


    `Graph for` ``get_gelu_pattern2()``

    ``y = (0.5 * x) * (tanh(((.0447)x^3 + x ) * sqrt(2/pi)) + 1)``

    .. code-block::

                       --------------------------------------------------------------------------------------------------------
                       ^                                                                                                      |
                       |                                                                                                      V
        [...] -----> mul(0.5)    pow (3) ----> mul (.044715) ---> add -----> mul (sqrt(2/pi)) ---> tanh ----> add (1) -----> mul ---> [...]
          |                        ^                               ^
          |                        |                               |
          |---------------------------------------------------------

    """

    def apply(self, prog):
        fuse_all_blocks(
            ops_arrangement=self.get_gelu_pattern1(),
            var_constraints=self.is_var_constraint_satisifed,
            transform_pattern=self.transform_pattern,
            prog=prog,
        )

        fuse_all_blocks(
            ops_arrangement=self.get_gelu_pattern2(),
            var_constraints=self.is_var_constraint_satisifed,
            transform_pattern=self.transform_pattern,
            prog=prog,
        )

    @staticmethod
    def is_var_constraint_satisifed(pattern):

        passed = _check_var_scalar_value(pattern.mul.y, 0.5) or _check_var_scalar_value(
            pattern.mul.x, 0.5
        )
        passed = passed and _check_var_scalar_value(pattern.pow.y, 3.0)

        passed = passed and (
            _check_var_scalar_value(pattern.mul_1.y, 0.044715)
            or _check_var_scalar_value(pattern.mul_1.x, 0.044715)
        )

        passed = passed and (
            _check_var_scalar_value(pattern.mul_2.y, 0.79788)
            or _check_var_scalar_value(pattern.mul_2.x, 0.79788)
        )

        passed = passed and (
            _check_var_scalar_value(pattern.add_1.y, 1)
            or _check_var_scalar_value(pattern.add_1.x, 1)
        )

        return passed

    @staticmethod
    def transform_pattern(pattern):
        # remove all the ops, and replace with a gelu op
        out_name = pattern.mul_3.outputs[0].name
        x = mb.gelu(
            x=pattern.root_var, mode="TANH_APPROXIMATION", name=out_name, before_op=pattern.mul
        )

        pattern.mul_3.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=pattern.mul_3, old_var=pattern.mul_3.outputs[0], new_var=x
        )

        # Remove all the ops at once
        pattern.block.remove_ops(pattern.op_list())

    @staticmethod
    def get_gelu_pattern1():
        """
        ``y = x * (0.5 * (tanh(((.0447)x^3 + x ) * sqrt(2/pi)) + 1))``

        .. code-block::

			[...] -----> pow (3) ----> mul (.044715) ---> add -----> mul (sqrt(2/pi)) ---> tanh ----> add (1) ----> mul (0.5) -----> mul ---> [...]
			  |                                            ^                                                                          ^
			  |                                            |                                                                          |
			  |------------------------------------------------------------------------------------------------------------------------

        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=([get_new_symbol(), get_new_symbol(), get_new_symbol()])),
            ]
        )
        def gelu_to_detect_1(x):
            # MIL operation takes named inputs (instead of positional inputs).
            # Here `name` argument is MANDATORY.
            pow = mb.pow(x=x, y=3.0, name="pow")
            mul_1 = mb.mul(x=0.044714998453855515, y=pow, name="mul_1")
            add = mb.add(x=x, y=mul_1, name="add")
            mul_2 = mb.mul(x=0.7978845834732056, y=add, name="mul_2")
            tanh = mb.tanh(x=mul_2, name="tanh")
            add_1 = mb.add(x=1.0, y=tanh, name="add_1")
            mul = mb.mul(x=0.5, y=add_1, name="mul")
            mul_3 = mb.mul(x=mul, y=x, name="mul_3")
            return mul_3

        return gelu_to_detect_1

    @staticmethod
    def get_gelu_pattern2():
        """
        ``y = (0.5 * x) * (tanh(((.0447)x^3 + x ) * sqrt(2/pi)) + 1)``

        .. code-block::

                           --------------------------------------------------------------------------------------------------------
                           ^                                                                                                      |
                           |                                                                                                      V
            [...] -----> mul(0.5)    pow (3) ----> mul (.044715) ---> add -----> mul (sqrt(2/pi)) ---> tanh ----> add (1) -----> mul ---> [...]
              |                        ^                               ^
              |                        |                               |
              |---------------------------------------------------------

        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=([get_new_symbol(), get_new_symbol(), get_new_symbol()])),
            ]
        )
        def gelu_to_detect_2(x):
            pow = mb.pow(x=x, y=3.0, name="pow")
            mul_1 = mb.mul(x=0.044714998453855515, y=pow, name="mul_1")
            add = mb.add(x=x, y=mul_1, name="add")
            mul_2 = mb.mul(x=0.7978845834732056, y=add, name="mul_2")
            tanh = mb.tanh(x=mul_2, name="tanh")
            add_1 = mb.add(x=1.0, y=tanh, name="add_1")
            mul = mb.mul(x=0.5, y=x, name="mul")
            mul_3 = mb.mul(x=mul, y=add_1, name="mul_3")
            return mul_3

        return gelu_to_detect_2


@register_pass(namespace="common")
class fuse_leaky_relu(AbstractGraphPass):
    """
    Detect the ``mul`` ---> ``max`` pattern than can be mapped to ``leaky_relu``.

    `In code form - Input`

    .. code-block::

       %2 = const(value = alpha) # where 0 <= alpha <= 1
       %3 = mul(%1, %2) # alpha * x
       %4 = max(%3, %1) # max(alpha * x, x)


    `In code form - Output`

    .. code-block::

       %4 = leaky_relu(x=%1, alpha=%2)


    `In graphical form - Input graph`

    .. code-block::

                 const (val = alpha)
                     |
        input ----> mul ---------------> maximum -----------> output
          |                                 |
          |----------------------------------


    `In graphical form - Output graph`

    .. code-block::

        input --------> leaky_relu ---------> output

    """

    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self._fuse_leaky_relu_block(f)

    @staticmethod
    def _try_to_transform(mul_op, block):

        ops_to_remove = []

        # check that one of the inputs of the mul op is a constant that is between 0 and 1
        if _check_var_scalar_value_in_interval(mul_op.x, 0, 1):
            alpha_input_var = mul_op.x
            parent_var = mul_op.y
        elif _check_var_scalar_value_in_interval(mul_op.y, 0, 1):
            alpha_input_var = mul_op.y
            parent_var = mul_op.x
        else:
            return False

        # check that output of mul is not a block output
        if mul_op.outputs[0] in block.outputs:
            return False
        ops_to_remove.append(mul_op)

        # check if the child op of the mul op is maximum
        if not _check_child_op_type(mul_op, "maximum"):
            return False

        # check that the other input of the max op is same as the parent of the mul op
        max_op = list(mul_op.outputs[0].child_ops)[0]
        if not (
            (max_op.x == mul_op.outputs[0] and max_op.y == parent_var)
            or (max_op.y == mul_op.outputs[0] and max_op.x == parent_var)
        ):
            return False
        ops_to_remove.append(max_op)

        # remove all the ops, and replace with a leaky relu op
        out_name = max_op.outputs[0].name
        x = mb.leaky_relu(x=parent_var, alpha=alpha_input_var.val, name=out_name, before_op=max_op)
        max_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=max_op, old_var=max_op.outputs[0], new_var=x
        )
        block.remove_ops(ops_to_remove)
        return True

    @block_context_manager
    def _fuse_leaky_relu_block(self, block):
        fusion_occurred = False
        for op in list(block.operations):
            if op.enclosing_block is None:
                continue

            for b in op.blocks:
                block_changed = True
                while block_changed:
                    block_changed = self._fuse_leaky_relu_block(b)

            if len(op.blocks) > 0:
                continue

            # start pattern match if mul op is encountered
            if op.op_type == "mul":
                if self._try_to_transform(op, block):
                    fusion_occurred = True
        return fusion_occurred


class FusePreluPattern1:
    @staticmethod
    def is_var_constraint_satisifed(pattern):
        # input must be rank 4
        if pattern.root_var.rank != 4:
            return False
        # output must be rank 4
        if pattern.out_op.outputs[0].rank != 4:
            return False
        if not (
            _check_var_scalar_value(pattern.neg.y, -1) or _check_var_scalar_value(pattern.neg.x, -1)
        ):
            return False
        if pattern.alpha_mul.x.val is not None:
            alpha = pattern.alpha_mul.x.val
        elif pattern.alpha_mul.y.val is not None:
            alpha = pattern.alpha_mul.y.val
        else:
            return False
        # alpha must be of shape (1, C, 1, 1) or (C, 1, 1)
        if len(alpha.shape) not in (3, 4):
            return False
        if alpha.size != alpha.shape[-3]:
            return False

        return True

    @staticmethod
    def transform_pattern(pattern):
        # remove all the ops, and replace with a prelu op
        out_var = pattern.out_op.outputs[0]
        if pattern.alpha_mul.x.val is not None:
            alpha = pattern.alpha_mul.x.val
        else:
            alpha = pattern.alpha_mul.y.val

        alpha_vector = -1 * alpha.flatten()
        x = mb.prelu(
            x=pattern.root_var, alpha=alpha_vector, name=out_var.name, before_op=pattern.out_op
        )
        pattern.out_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=pattern.out_op, old_var=out_var, new_var=x
        )
        # Remove all the ops at once
        pattern.block.remove_ops(pattern.op_list())

    @staticmethod
    def get_prelu_pattern():
        """
        ``y = a * relu(-1 * x) + relu(x)``

        When ``x`` is rank 4, and ``a`` is of shape ``(1, C, 1, 1)`` or ``(C, 1, 1)``,
        this is equivalent to ``prelu`` with ``alpha = -a.flatten()``.
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(
                    shape=([get_new_symbol(), get_new_symbol(), get_new_symbol(), get_new_symbol()])
                ),
            ]
        )
        def prelu_pattern(x):
            return fuse_prelu._prelu_pattern(x)

        return prelu_pattern


class FusePreluPattern2:
    @staticmethod
    def is_var_constraint_satisifed(pattern):
        perm = pattern.transpose.perm.val
        if not np.array_equal(perm, np.array([0, 2, 3, 1])):
            return False
        # output must be rank 4
        if pattern.out_op.outputs[0].rank != 4:
            return False
        if not (
            _check_var_scalar_value(pattern.neg.y, -1) or _check_var_scalar_value(pattern.neg.x, -1)
        ):
            return False
        if pattern.alpha_mul.x.val is not None:
            alpha = pattern.alpha_mul.x.val
        elif pattern.alpha_mul.y.val is not None:
            alpha = pattern.alpha_mul.y.val
        else:
            return False
        # alpha must be of shape (C,) or (1,C) or (1,1,C) or (1,1,1,C)
        if alpha.size != alpha.shape[-1]:
            return False

        return True

    @staticmethod
    def transform_pattern(pattern):
        # remove all the ops, and replace with a prelu op + transpose op
        perm = pattern.transpose.perm.val
        out_var = pattern.out_op.outputs[0]
        if pattern.alpha_mul.x.val is not None:
            alpha = pattern.alpha_mul.x.val
        else:
            alpha = pattern.alpha_mul.y.val

        alpha_vector = -1 * alpha.flatten()
        x = mb.prelu(x=pattern.root_var, alpha=alpha_vector, before_op=pattern.out_op)
        x = mb.transpose(x=x, perm=perm, name=out_var.name, before_op=pattern.out_op)
        pattern.out_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=pattern.out_op, old_var=out_var, new_var=x
        )
        # Remove all the ops at once
        pattern.block.remove_ops(pattern.op_list())

    @staticmethod
    def get_prelu_pattern():
        """
        ``x1 = transpose(perm=(0,2,3,1))(x)``

        ``y = a * relu(-1 * x1) + relu(x1)``

        When ``x`` is rank 4, and ``a`` is of shape (``C,)``, ``(1, C)``, ``(1,1,C)``, or ``(1,1,1,C)``,
        this is equivalent to ``prelu`` with ``alpha = -a.flatten()``, followed by a ``transpose``
        with ``perm (0,2,3,1)``.
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(
                    shape=([get_new_symbol(), get_new_symbol(), get_new_symbol(), get_new_symbol()])
                ),
            ]
        )
        def prelu_pattern(x):
            # perm value can be anything, it will be checked in "is_var_constraint_satisifed" method
            x = mb.transpose(x=x, perm=[0, 1, 2, 3], name="transpose")
            return fuse_prelu._prelu_pattern(x)

        return prelu_pattern


@register_pass(namespace="common")
class fuse_prelu(AbstractGraphPass):
    """
    Detect the following patterns that can be mapped to a ``prelu`` op.
    Essentially, the ``prelu`` op can be broken down into the following ops:

    ``y = a * relu(-1 * x) + relu(x)``

    `Pattern 1`

    .. code-block::

                          | ------------> relu --------------------|
                          |                                        V
           x (BCHW) ------|                                       add -----> y (BCHW)
                          |                                        ^
                          --------> mul -------> relu -----> mul---|
                                     ^                        ^
                                     |                        |
                                Const(val=-1)            Const(name=a, shape=(C,1,1) or (1,C,1,1))


    This will be mapped to:

    .. code-block::

            x (BCHW) ------> prelu(alpha=a, shape=(C,)) ---------> y (BCHW)


    `Pattern 2`

    .. code-block::

                                          | ------------> relu --------------------|
                                          |                                        V
          x (BCHW) -->transpose(BHWC)---->|                                       add -----> y (BHWC)
                                          |                                        ^
                                          --------> mul -------> relu -----> mul---|
                                                     ^                        ^
                                                     |                        |
                                            Const(val=-1)    Const(shape=(C,) or (1,C) or (1,1,C) or (1,1,1,C))


    This will be mapped to:

    .. code-block::

            x (BCHW) ------> prelu ---------> transpose ------> y (BHWC)
    """

    def apply(self, prog):
        for pattern in (FusePreluPattern1, FusePreluPattern2):
            fuse_all_blocks(
                ops_arrangement=pattern.get_prelu_pattern(),
                var_constraints=pattern.is_var_constraint_satisifed,
                transform_pattern=pattern.transform_pattern,
                prog=prog,
            )

    @staticmethod
    def _prelu_pattern(x):
        # MIL operation takes named inputs (instead of positional inputs).
        # Here `name` argument is MANDATORY.
        neg = mb.mul(x=x, y=-1.0, name="neg")
        relu1 = mb.relu(x=neg, name="relu1")
        # Use any constant here to match, rank and shape will be verified in
        # `is_var_constraint_satisifed`.
        mul = mb.mul(x=relu1, y=np.random.rand(2, 2, 2, 2), name="alpha_mul")
        relu2 = mb.relu(x=x, name="relu2")
        out = mb.add(x=relu2, y=mul, name="out_op")
        return out


@register_pass(namespace="common")
class prelu_to_lrelu(AbstractGraphPass):
    """
    If ``prelu`` has the same leakage factor across all channels, it will be converted to ``leaky_relu``.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._prelu_to_lrelu_block(f)

    @block_context_manager
    def _prelu_to_lrelu_block(self, block):
        for op in list(block.operations):
            for b in op.blocks:
                self._prelu_to_lrelu_block(b)
            if len(op.blocks) > 0:
                # This op can't be prelu.
                continue

            if op.op_type == "prelu":
                alpha_val = op.alpha.val
                common_leakage_factor = True
                for c in range(1, op.alpha.val.shape[0]):
                    if alpha_val[c] != alpha_val[0]:
                        common_leakage_factor = False
                        break
                if common_leakage_factor:
                    lrelu_out = mb.leaky_relu(
                        x=op.x, alpha=alpha_val[0], name=op.outputs[0].name, before_op=op
                    )
                    op.enclosing_block.replace_uses_of_var_after_op(
                        anchor_op=op, old_var=op.outputs[0], new_var=lrelu_out
                    )
                    block.remove_ops([op])
