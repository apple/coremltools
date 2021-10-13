# -*- coding: utf-8 -*-

#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

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