#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Callable, List, Optional

import numpy as np

from coremltools.converters.mil.mil import Block, Operation
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass


class classproperty(property):
    """
    A decorator class that allow us to have a class-level property
    """
    def __get__(self, owner, cls):
        return self.fget(cls)


def block_context_manager(_func: Optional[Callable] = None):
    """
    This decorator executes a function under the context manager `with block`.
    For instance, given a function `func` with an input block and other arguments:

    def func(block, *args):
        ...
        with block:
            op_1 = mb.add(...)
        ...
        with block:
            op_2 = mb.relu...()

    It can be be streamlined as:

    @block_context_manager
    def func(block, *args):
        ...
        op_1 = mb.add(...)
        ...
        op_2 = mb.relu...()

    Note that, the first argument of the function must have type Block.
    It is highly recommended to decorate a function with block_context_manager if it is calling `with block` multiple times,
    since when the code exit `block`, an expensive _propagate_nonreplaceable_vars() is invoked.
    The decorator reduces the amount of calling `with block` overally.
    """

    def wrapper(*args):
        # Make it compatible with class method.
        if isinstance(args[0], AbstractGraphPass):
            block = args[1]
        else:
            block = args[0]

        if not isinstance(block, Block):
            raise ValueError(
                "The function decorated with block_context_manager must have a Block "
                "type argument as the first input."
            )

        with block:
            return _func(*args)

    return wrapper


def _check_child_op_type(op, child_op_type):
    """
    :param op: operation
    :param child_op_type: str
    :return: Return True if op has 1 child and type of that child matches child_op_type
    """
    if len(op.outputs) != 1:
        return False
    child_ops = list(op.outputs[0].child_ops)
    if len(child_ops) != 1:
        return False
    if child_ops[0].op_type == child_op_type:
        return True
    return False


def _check_no_output_connection(block: Block, ops: List[Operation]) -> bool:
    """
    Check that none of the op in this pattern is connected to the output
    (except the last op)

    :param block: Block
    :param ops: List of operations to check on.
    """
    for op in ops[:-1]:
        for out in op.outputs:
            if out in block.outputs:
                return False
    return True


def _check_var_scalar_value_in_interval(x, lower_bound, upper_bound):
    """
    :param x: var
    :param lower_bound: a scalar value
    :param upper_bound: a scalar value
    :return: True if the value of var is in the interval [lower_bound, upper_bound]
    """
    if x.val is None:
        return False
    if not isinstance(x.val, (np.ndarray, np.generic)):
        return False

    if isinstance(x.val, np.ndarray):
        if x.val.size != 1:
            return False
        x_val = x.val[:][0] if len(x.val.shape) > 0 else x.val[()]
    else:
        x_val = x.val

    if x_val >= lower_bound and x_val <= upper_bound:
        return True
    return False


def _check_var_scalar_value(x, val, tol=1e-3):
    """
    :param x: var
    :param val: a scalar value
    :return: True if x.val is equal to val otherwise return False
    """
    if x.val is None:
        return False
    if not isinstance(x.val, np.ndarray) and not np.isscalar(x.val):
        return False

    if isinstance(x.val, np.ndarray):
        if x.val.size != 1:
            return False
        if len(x.val.shape) == 0:
            x_val = x.val
        else:
            x_val = x.val[:][0] if len(x.val.shape) > 0 else x.val[()]
    else:
        x_val = x.val

    if abs(x_val - val) < tol:
        return True
    return False
