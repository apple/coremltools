#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.experimental.passes.generic_pass_infrastructure import \
    fuse_all_blocks
from coremltools.converters.mil.mil import get_new_symbol
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import \
    _check_var_scalar_value
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


def is_var_constraint_satisifed(pattern):

    passed = _check_var_scalar_value(pattern.mul.y, 0.5) or _check_var_scalar_value(pattern.mul.x, 0.5)
    passed = passed and _check_var_scalar_value(pattern.pow.y, 3.0)

    passed = passed and (
                        _check_var_scalar_value(pattern.mul_1.y, 0.044715) or
                        _check_var_scalar_value(pattern.mul_1.x, 0.044715)
                        )

    passed = passed and (
                        _check_var_scalar_value(pattern.mul_2.y, 0.79788) or
                        _check_var_scalar_value(pattern.mul_2.x, 0.79788)
                        )

    passed = passed and (
                        _check_var_scalar_value(pattern.add_1.y, 1) or
                        _check_var_scalar_value(pattern.add_1.x, 1)
                        )

    return passed


def transform_pattern(pattern):
    # remove all the ops, and replace with a gelu op
    out_name = pattern.mul_3.outputs[0].name
    x = mb.gelu(x=pattern.root_var, mode="TANH_APPROXIMATION", name=out_name, before_op=pattern.mul)

    pattern.mul_3.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=pattern.mul_3, old_var=pattern.mul_3.outputs[0], new_var=x
    )

    # Remove all the ops at once
    pattern.block.remove_ops(pattern.op_list())

def get_gelu_pattern1():
    """
    y = x * (0.5 * (tanh(((.0447)x^3 + x ) * sqrt(2/pi)) + 1))


    [...] -----> pow (3) ----> mul (.044715) ---> add -----> mul (sqrt(2/pi)) ---> tanh ----> add (1) ----> mul (0.5) -----> mul ---> [...]
      |                                            ^                                                                          ^
      |                                            |                                                                          |
      |------------------------------------------------------------------------------------------------------------------------

    """
    @mb.program(input_specs=[mb.TensorSpec(shape=([get_new_symbol(), get_new_symbol(), get_new_symbol()])), ])
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


def get_gelu_pattern2():
    """
    y = (0.5 * x) * (tanh(((.0447)x^3 + x ) * sqrt(2/pi)) + 1)

                    ---------------------------------------------------------------------------------------------------------
                    ^                                                                                                       |
                    |                                                                                                       V
     [...] -----> mul(0.5)    pow (3) ----> mul (.044715) ---> add -----> mul (sqrt(2/pi)) ---> tanh ----> add (1) -----> mul ---> [...]
      |                         ^                               ^
      |                         |                               |
      |------------------------------------------------------------
    """
    @mb.program(input_specs=[mb.TensorSpec(shape=([get_new_symbol(), get_new_symbol(), get_new_symbol()])), ])
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
class fuse_gelu_tanh_approximation(AbstractGraphPass):
    """
    Identify the pattern that corresponds to the tanh approximate version of gelu, and replace it with a single
    gelu layer with mode=TANH_APPROXIMATION

    The implementation of this pass uses the generic graph pattern matching and transform algorithm implemented in
    coremltools.converters.mil.experimental.passes.generic_pass_infrastructure and documented in
    coremltools/converters/mil/experimental/passes/readme.md
    """

    def apply(self, prog):
        fuse_all_blocks(ops_arrangement=get_gelu_pattern1(),
                        var_constraints=is_var_constraint_satisifed,
                        transform_pattern=transform_pattern,
                        prog=prog)

        fuse_all_blocks(ops_arrangement=get_gelu_pattern2(),
                        var_constraints=is_var_constraint_satisifed,
                        transform_pattern=transform_pattern,
                        prog=prog)
