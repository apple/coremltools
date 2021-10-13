# -*- coding: utf-8 -*-

#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os
import numpy as np

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.experimental.passes.generic_pass_infrastructure import register_generic_pass
from coremltools.converters.mil.mil import get_new_symbol

arbitrary_shape = (get_new_symbol(), get_new_symbol())
np.random.seed()
arbitrary_weight = np.random.rand(4,3)
arbitrary_bias =  np.random.rand(4)

@mb.program(input_specs=[mb.TensorSpec(shape=arbitrary_shape)])
def pattern_add(x):
    """
    Original:
        % 4 = linear(x= % 1, weight = % 2, bias = % 3)  # %2 is a rank-2 const tensor (weight)
        # %3 is a rank-1 const tensor (bias)
        ...
        % 6 = add(x= % 4, y = % 5)  # %5 is a const tensor with same shape as %3

    Result:
        % 8 = linear(x= % 1, weight = % 2, bias = % 7)  # where %7 is a new const tensor with value
        # %7 = %3 + %6
    """
    linear = mb.linear(x=x, weight=arbitrary_weight, bias=arbitrary_bias, name="linear")
    add_or_sub = mb.add(x=linear, y=arbitrary_bias, name="add_or_sub")
    return add_or_sub

@mb.program(input_specs=[mb.TensorSpec(shape=arbitrary_shape)])
def pattern_sub(x):
    """
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
    linear = mb.linear(x=x, weight=arbitrary_weight, bias=arbitrary_bias, name="linear")
    add_or_sub = mb.sub(x=linear, y=arbitrary_bias, name="add_or_sub")
    return add_or_sub


def var_constraints(pattern):
    passed = True
    passed = passed and pattern.add_or_sub.x.val is not None or pattern.add_or_sub.y.val is not None

    is_sub, is_first_input = _get_is_sub_and_is_first_input(pattern)
    linear_bias, bias, Dout = _get_linear_bias_bias_Dout(pattern, is_first_input)

    # check if the shape is broadcasable
    passed = passed and np.prod(linear_bias.shape) == np.prod(bias.shape)
    passed = passed and bias.shape[-1] == Dout
    return passed


def _get_is_sub_and_is_first_input(pattern):
    is_sub = pattern.add_or_sub.op_type == "sub"
    is_first_input = pattern.add_or_sub.x == pattern.linear.outputs[0]
    return is_sub, is_first_input


def _get_linear_bias_bias_Dout(pattern, is_first_input):
    linear_bias = pattern.linear.bias.val
    bias = pattern.add_or_sub.y.val if is_first_input else pattern.add_or_sub.x.val
    Dout = linear_bias.shape[0]
    return linear_bias, bias, Dout


def transform_pattern(pattern):
    is_sub, is_first_input = _get_is_sub_and_is_first_input(pattern)
    linear_bias, bias, Dout = _get_linear_bias_bias_Dout(pattern, is_first_input)
    bias = np.reshape(bias, (Dout,))

    if is_sub and is_first_input: bias = -bias
    if is_sub and not is_first_input: linear_bias = -linear_bias

    new_bias = linear_bias + bias

    # compute the new weight
    if is_sub and not is_first_input:
        new_weight = -pattern.linear.weight.val
    else:
        new_weight = pattern.linear.weight.val

    # create a new linear op with the new weight, bias value, copying rest of the attributes
    out_name = pattern.add_or_sub.outputs[0].name
    linear_kargs = {"weight": new_weight, "bias": new_bias, "name": out_name, "before_op": pattern.linear}

    linear_kargs.update({k: v for k, v in pattern.linear.inputs.items() if k not in ["weight", "bias"]})

    x = mb.linear(**linear_kargs)

    pattern.add_or_sub.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=pattern.add_or_sub, old_var=pattern.add_or_sub.outputs[0], new_var=x
    )
    # Remove all the ops at once
    pattern.block.remove_ops(pattern.op_list())


if os.getenv('ENABLE_EXPERIMENTAL_PASSES') == '1':
    register_generic_pass(
        ops_arrangement=pattern_add,
        var_constraints=var_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_linear_bias",
        namespace="common",
    )

    register_generic_pass(
        ops_arrangement=pattern_sub,
        var_constraints=var_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_linear_bias",
        namespace="common",
    )