#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.experimental.passes.generic_pass_infrastructure import \
    fuse_all_blocks
from coremltools.converters.mil.mil import get_new_symbol
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import \
    _check_var_scalar_value
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


def _prelu_pattern(x):
    # MIL operation takes named inputs (instead of positional inputs).
    # Here `name` argument is MANDATORY.
    neg = mb.mul(x=x, y=-1., name="neg")
    relu1 = mb.relu(x=neg, name="relu1")
    # use any constant here to match, rank and shape will be verified in "is_var_constraint_satisifed" method
    mul = mb.mul(x=relu1, y=np.random.rand(2, 2, 2, 2), name="alpha_mul")
    relu2 = mb.relu(x=x, name="relu2")
    out = mb.add(x=relu2, y=mul, name="out_op")
    return out



class Pattern1:
    @staticmethod
    def is_var_constraint_satisifed(pattern):
        # input must be rank 4
        if pattern.root_var.rank != 4:
            return False
        # output must be rank 4
        if pattern.out_op.outputs[0].rank != 4:
            return False
        if not (_check_var_scalar_value(pattern.neg.y, -1) or _check_var_scalar_value(pattern.neg.x, -1)):
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
        x = mb.prelu(x=pattern.root_var, alpha=alpha_vector, name=out_var.name, before_op=pattern.out_op)
        pattern.out_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=pattern.out_op, old_var=out_var, new_var=x
        )
        # Remove all the ops at once
        pattern.block.remove_ops(pattern.op_list())

    @staticmethod
    def get_prelu_pattern():
        """
        y = a * relu(-1 * x) + relu(x)

        when x is rank 4, and "a" is of shape (1, C, 1, 1) or (C, 1, 1),
        this is equivalent to prelu with alpha = -a.flatten(),
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=([get_new_symbol(), get_new_symbol(),
                                                       get_new_symbol(), get_new_symbol()])), ])
        def prelu_pattern(x):
            return _prelu_pattern(x)

        return prelu_pattern



class Pattern2:
    @staticmethod
    def is_var_constraint_satisifed(pattern):
        perm = pattern.transpose.perm.val
        if not np.array_equal(perm, np.array([0,2,3,1])):
            return False
        # output must be rank 4
        if pattern.out_op.outputs[0].rank != 4:
            return False
        if not (_check_var_scalar_value(pattern.neg.y, -1) or _check_var_scalar_value(pattern.neg.x, -1)):
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
        x1 = transpose(perm=(0,2,3,1))(x)
        y = a * relu(-1 * x1) + relu(x1)

        when x is rank 4, and "a" is of shape (C,) or (1, C) or (1,1,C) or (1,1,1,C),
        this is equivalent to prelu with alpha = -a.flatten(), followed by a transpose
        with perm (0,2,3,1)
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=([get_new_symbol(), get_new_symbol(),
                                                       get_new_symbol(), get_new_symbol()])), ])
        def prelu_pattern(x):
            # perm value can be anything, it will be checked in "is_var_constraint_satisifed" method
            x = mb.transpose(x=x, perm=[0,1,2,3], name="transpose")
            return _prelu_pattern(x)

        return prelu_pattern


@register_pass(namespace="common")
class fuse_prelu(AbstractGraphPass):
    """
    Detect the following patterns that can be mapped to a prelu op.
    Essentially prelu op can be broken down into the following ops: y = a * relu(-1 * x) + relu(x)

    Pattern 1:


                      | ------------> relu --------------------|
                      |                                        V
       x (BCHW) ------|                                       add -----> y (BCHW)
                      |                                        ^
                      --------> mul -------> relu -----> mul---|
                                ^                         ^
                                |                         |
                            Const(val=-1)               Const(name=a, shape=(C,1,1) or (1,C,1,1))

    This will be mapped to:
        x (BCHW) ------> prelu(alpha=a, shape=(C,)) ---------> y (BCHW)


    Pattern 2:

                                      | ------------> relu --------------------|
                                      |                                        V
      x (BCHW) -->transpose(BHWC)---->|                                       add -----> y (BHWC)
                                      |                                        ^
                                      --------> mul -------> relu -----> mul---|
                                                 ^                        ^
                                                 |                        |
                                        Const(val=-1)    Const(shape=(C,) or (1,C) or (1,1,C) or (1,1,1,C))

    This will be mapped to:
        x (BCHW) ------> prelu ---------> transpose ------> y (BHWC)
    """


    def apply(self, prog):
        for pattern in (Pattern1, Pattern2):
            fuse_all_blocks(ops_arrangement=pattern.get_prelu_pattern(),
                            var_constraints=pattern.is_var_constraint_satisifed,
                            transform_pattern=pattern.transform_pattern,
                            prog=prog)





