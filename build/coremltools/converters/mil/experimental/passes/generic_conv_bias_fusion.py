# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import os
import numpy as np
import logging

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.experimental.passes.generic_pass_infrastructure import register_generic_pass
from coremltools.converters.mil.mil import types

"""
Fold add/sub into bias of conv and conv_transpose
That is, convert conv + add/sub to conv, when add/sub is adding a constant

There are two main patterns supported now. The first one is:

Pattern 1:
Given:
   %2 = conv(%1)
   ...
   %3 = add(%2, constant) # where constant has shape (1,C,1)/(C,1) for 1d conv, (1,C,1,1)/(C,1,1) for 2d conv etc
   ...

Result:
   %3 = conv(%1)
   ...
   
The second one is:

Pattern 2:
   Given:
       %2 = conv(%1)
       %3 = transpose(%2)
       ...
       %4 = add(%3, constant) # where constant has a broacasable shape
       ...

   Result:
       %2 = conv(%1)
       %4 = transpose(%2)
       ...
       
When taking all of the conv/conv_tranpose, transpose/no transpose, and add/sub into account,
We end up with a total of 8 patterns (2^3). These patterns are paramaterized by the pattern_to_detect
function below.
"""

arbitrary_cin = 5
arbitrary_cout = 8
arbitrary_scalar = 5
np.random.seed()
arbitrary_perm = [0,1,2,3]
arbitrary_input = (3, arbitrary_cin, 224, 224)
arbitrary_weight = np.random.rand(arbitrary_cout, arbitrary_cin, 10, 10)


def pattern_to_detect(conv_transpose, transpose, sub):
    """
    Wrapper to create 8 patterns to detect for conciseness.
    """

    @mb.program(input_specs=[mb.TensorSpec(shape=arbitrary_input)])
    def conv_bias_pattern(x):
        if not conv_transpose:
            conv = mb.conv(x=x, weight=arbitrary_weight, pad_type="valid", name="conv")
        else:
            conv = mb.conv_transpose(x=x, weight=arbitrary_weight, pad_type="valid", name="conv")

        if transpose:
            transpose_layer = mb.transpose(x=conv, perm=arbitrary_perm, name="transpose")

        if sub:
            add_or_sub = mb.sub(x=transpose_layer if transpose else conv, y=arbitrary_scalar, name="add_or_sub")
        else:
            add_or_sub = mb.add(x=transpose_layer if transpose else conv, y=arbitrary_scalar, name="add_or_sub")
        return add_or_sub

    return conv_bias_pattern


def var_constraints(pattern):
    bias_value = _get_bias_var(pattern).val
    rank = pattern.conv.x.rank
    is_bias_scalar = True if not isinstance(bias_value, np.ndarray) else False
    old_bias = pattern.conv.inputs.get("bias", None)
    old_bias_value = old_bias.val if old_bias is not None and old_bias.val is not None else None

    passed = True
    passed = passed and isinstance(bias_value, (np.ndarray, np.generic))
    passed = passed and rank is not None
    passed = passed and (rank == 3 or rank == 4 or rank == 5)

    # check compatibility of bias value with the rank of the conv op
    # either bias value should be a scalar or:
    # rank=3 ==> (B,C,D), which means bias must be (1,C,1) or (C,1)
    # rank=4 ==> (B,C,D1,D2), which means bias must be (1,C,1,1) or (C,1,1)
    # rank=5 ==> (B,C,D1,D2,D3), which means bias must be (1,C,1,1,1) or (C,1,1,1)
    if not is_bias_scalar:
        # check that there is at most one dimension in the shape that is not 1
        passed = passed and len(np.squeeze(bias_value).shape) <= 1
        # check that addition is not happening on the batch dimension
        passed = passed and (len(bias_value) != rank or bias_value.shape[0] == 1)
        # check that last rank-2 entries in the shape vector are all 1s
        passed = passed and np.prod(bias_value.shape[-(rank - 2):]) == 1

    bias_value = np.array([bias_value]) if is_bias_scalar else np.squeeze(bias_value)

    passed = passed and (
            old_bias is not None
            or np.prod(bias_value.shape) != 1
            or pattern.conv.weight.val is not None
    )

    if old_bias is not None:
        try:
            new_bias_value = old_bias_value + bias_value
        except:
            return False

    return passed


def var_constraints_tranpose(pattern):
    bias = pattern.add_or_sub.x.val if pattern.add_or_sub.x.val is not None else pattern.add_or_sub.y.val
    Cout = pattern.conv.outputs[0].shape[1]

    passed = True
    passed = passed and pattern.add_or_sub.x.val is not None or pattern.add_or_sub.y.val is not None
    passed = passed and _bias_mod_and_validity(bias, Cout, pattern) is not None
    return passed

def transform_pattern(pattern):
    bias_value = _get_bias_var(pattern).val

    is_conv_op = (pattern.conv.op_type == "conv")

    is_bias_scalar = False
    if not isinstance(bias_value, np.ndarray):
        is_bias_scalar = True

    # find rank of the conv input
    rank = pattern.conv.x.rank

    bias_value = np.array([bias_value]) if is_bias_scalar else np.squeeze(bias_value)

    if pattern.add_or_sub.op_type == "sub":
        bias_value *= -1

    # everything looks good, now find the new updated bias
    old_bias = pattern.conv.inputs.get("bias", None)
    old_bias_value = None
    if old_bias is not None and old_bias.val is not None:
        old_bias_value = old_bias.val
    if old_bias is None:
        # need to create a fresh numpy array for bias
        if np.prod(bias_value.shape) == 1:
            # its a scalar bias
            # need to find the value of Cout to form a new bias
            # conv_transpose has weight format [K, C_out, spatial dims]
            # conv has weight format [C_out, K, spatial dims]
            Cout = pattern.conv.weight.val.shape[0 if is_conv_op else 1]
            new_bias_value = np.broadcast_to(bias_value, (Cout,))
        else:
            new_bias_value = bias_value
    else:
        # just need to update the existing bias array
        new_bias_value = old_bias_value + bias_value

    # create a new conv op with the new bias value, copying rest of the attributes
    out_name = pattern.add_or_sub.outputs[0].name
    if new_bias_value.dtype != np.float32 and new_bias_value.dtype != np.float16:
        # cast the bias to match the weight type
        weight_np_type = types.nptype_from_builtin(pattern.conv.inputs["weight"].sym_type.get_primitive())
        logging.warning("conv_bias_fusion pass: casting bias "
                        "from {} to {} to match the dtype of the weight of the conv layer".format(
                        new_bias_value.dtype, weight_np_type
                        )
        )
        new_bias_value = new_bias_value.astype(weight_np_type)
    new_bias_var = mb.const(val=new_bias_value, before_op=pattern.conv)

    conv_kargs = {"bias": new_bias_var, "name": out_name, "before_op": pattern.conv}

    for k, v in pattern.conv.inputs.items():
        if k == "bias":
            continue
        conv_kargs[k] = v

    if is_conv_op:
        x = mb.conv(**conv_kargs)
    else:
        x = mb.conv_transpose(**conv_kargs)

    pattern.add_or_sub.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=pattern.add_or_sub, old_var=pattern.add_or_sub.outputs[0], new_var=x
    )
    # Remove all the ops at once
    pattern.block.remove_ops(pattern.op_list())


def transform_transpose_pattern(pattern):
    is_deconv = pattern.conv.op_type == "conv_transpose"

    # get the bias
    bias = pattern.add_or_sub.x.val if pattern.add_or_sub.x.val is not None else pattern.add_or_sub.y.val
    is_first_input = pattern.add_or_sub.y.val is not None
    is_sub = pattern.add_or_sub.op_type == "sub"

    # get the conv bias/weight
    conv_shape = pattern.conv.outputs[0].shape
    Cout = conv_shape[1]
    conv_weight = pattern.conv.weight.val
    conv_weight_type = conv_weight.dtype
    conv_bias = np.zeros(Cout).astype(conv_weight_type) if pattern.conv.bias is None else pattern.conv.bias.val

    bias = _bias_mod_and_validity(bias, Cout, pattern)

    # compute the new bias
    if is_sub:
        if is_first_input:
            bias = -bias
        else:
            conv_bias = -conv_bias

    new_bias = conv_bias + bias

    # compute the new weight
    if is_sub and not is_first_input:
        new_weight = -conv_weight
    else:
        new_weight = conv_weight

    # create a new conv op with the new weight, bias value, copying rest of the attributes
    conv_kargs = {"weight": new_weight, "bias": new_bias, "before_op": pattern.conv}

    for k, v in pattern.conv.inputs.items():
        if k in ["weight", "bias"]:
            continue
        conv_kargs[k] = v

    if is_deconv:
        x = mb.conv_transpose(**conv_kargs)
    else:
        x = mb.conv(**conv_kargs)

    # create a new transpose op
    out_name = pattern.add_or_sub.outputs[0].name
    tranpose_kargs = {"x": x, "name": out_name, "before_op": pattern.transpose}
    for k, v in pattern.transpose.inputs.items():
        if k == "x":
            continue
        tranpose_kargs[k] = v
    x = mb.transpose(**tranpose_kargs)

    pattern.add_or_sub.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=pattern.add_or_sub, old_var=pattern.add_or_sub.outputs[0], new_var=x
    )

    # Remove all the ops at once
    pattern.block.remove_ops(pattern.op_list())

def _bias_mod_and_validity(bias, Cout, pattern):
 # check if the bias is compatible for fusion
    is_bias_scalar = True
    if isinstance(bias, np.ndarray):
        if bias.shape == ():
            bias = bias.tolist()
        elif np.prod(bias.shape) == 1:
            bias = np.squeeze(bias).tolist()
        else:
            is_bias_scalar = False

    if not is_bias_scalar:
        if np.prod(bias.shape) != Cout:
            return None
        rank = pattern.transpose.outputs[0].rank
        cout_dim = pattern.transpose.perm.val.tolist().index(1) - rank
        if bias.shape[cout_dim] != Cout:
            return None
        bias = np.reshape(bias, (Cout))

    return bias

def _get_bias_var(pattern):
    if pattern.add_or_sub.op_type == "sub":
        bias_var = pattern.add_or_sub.y
    else:
        bias_var = pattern.add_or_sub.x if pattern.add_or_sub.x.val is not None else pattern.add_or_sub.y

    return bias_var


if os.getenv("ENABLE_EXPERIMENTAL_PASSES") == "1":

    # conv -> add
    register_generic_pass(
        ops_arrangement=pattern_to_detect(False, False, False),
        var_constraints=var_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_conv_bias",
        namespace="common",
    )

    # conv -> sub
    register_generic_pass(
        ops_arrangement=pattern_to_detect(False, False, True),
        var_constraints=var_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_conv_bias",
        namespace="common",
    )

    # conv_transpose -> add
    register_generic_pass(
        ops_arrangement=pattern_to_detect(True, False, False),
        var_constraints=var_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_conv_bias",
        namespace="common",
    )

    # conv_transpose -> sub
    register_generic_pass(
        ops_arrangement=pattern_to_detect(True, False, True),
        var_constraints=var_constraints,
        transform_pattern=transform_pattern,
        pass_name="fuse_conv_bias",
        namespace="common",
    )

    # conv -> transpose -> add
    register_generic_pass(
        ops_arrangement=pattern_to_detect(False, True, False),
        var_constraints=var_constraints_tranpose,
        transform_pattern=transform_transpose_pattern,
        pass_name="fuse_conv_bias",
        namespace="common",
    )

    # conv -> transpse -> sub
    register_generic_pass(
        ops_arrangement=pattern_to_detect(False, True, True),
        var_constraints=var_constraints_tranpose,
        transform_pattern=transform_transpose_pattern,
        pass_name="fuse_conv_bias",
        namespace="common",
    )

    # conv_transpose -> transpose -> add
    register_generic_pass(
        ops_arrangement=pattern_to_detect(True, True, False),
        var_constraints=var_constraints_tranpose,
        transform_pattern=transform_transpose_pattern,
        pass_name="fuse_conv_bias",
        namespace="common",
    )

    # conv_transpose -> transpose -> sub
    register_generic_pass(
        ops_arrangement=pattern_to_detect(True, True, True),
        var_constraints=var_constraints_tranpose,
        transform_pattern=transform_transpose_pattern,
        pass_name="fuse_conv_bias",
        namespace="common",
    )
