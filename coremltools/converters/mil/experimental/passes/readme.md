# Generic Pattern Matching Infrastructure Documentation

## _**Introduction**_

This document contains the **motivation**, **user flow**, and **documentation**, and **instructions** for adding/running a pass for Arjun Singla’s Generic Pattern Matching Infrastructure.

## _**What We Know**_

* Existing TensorFlow and Pytorch models are converted to intermediate representations, GraphDef and TorchScript respectively, by the frameworks themselves when they are compiled. These intermediate representations are “verbose” - each operation is expanded into the combination of its most basic operations.
* Then, our Apple infrastructure performs a one to one mapping, taking these intermediate representations and converting them into a MIL (Model Intermediate Language) representation. As this mapping is one to one, the MIL representation is “verbose” as well.
*  Now, the goal becomes to take these “verbose” MIL representations, and make them compact again - taking sets of simple operations and consolidating them into their more complicated cousins - the same ones that the user defined in the original TensorFlow and Pytorch models. These are executed when we convert the MIL representation into the final CoreML one.
* The project
    * My project is working on a very specific subproblem to this larger issue. The goal is to take these “verbose” MIL representations, detect **any** sequence of operations, and replace it with **any** other sequence of operations. 

## _**The User Flow: Documentation**_

* We are assuming that the user has a very high understanding of PyMil. So, we will have the user define a PyMil program, which will be the pattern to detect in the larger machine learning model. Attached is a code snippet, taken from the PyMil docs, on how to define a program:

```
#import builder
from coremltools.converters.mil import Builder as mb

# Input to MIL program is a list of tensors. Here we have one input with
# shape (1, 100, 100, 3) and implicit dtype == fp32

@mb.program(input_specs=[mb.TensorSpec(shape=(1, 100, 100, 3)),])
def prog(x): 
    
    # MIL operation takes named inputs (instead of positional inputs). 
    # Here name argument is optional.
    
    x = mb.relu(x=x, name='relu') 
    x = mb.transpose(x=x, perm=[0, 3, 1, 2], name='transpose') 
    x = mb.reduce_mean(x=x, axes=[2, 3], keep_dims=False, name='reduce') 
    x = mb.log(x=x, name='log') 
    return x 
```

* It is important that the user follows these constraints when writing their MIL program:
  * **This program must only have one root variable**
  * **This program has exactly one proper last operation topologically.**
  * **Each operation in the program must have a UNIQUE NAME!!!**
    ```
    # Read from left to right, this pattern has two "last" operations,
    # and is not permitted
            
     --> op1 --- op2 --- op3 -->
                  |
                  | ---- op4 -->
                     
    # Read from left to right, this pattern has one "last" operation,
    # and is permitted. The only thing that must be
    # singular here is the last operation (and, of course, the root var)
    
    --> op1 --- op2 --- op3 --- op5 -->
                 |               |
                 | ---- op4 -----|    



* The second function the user needs to define is the following:
`def var_constraints(pattern):`
    * Parameters
        * a `Pattern` object
            * What is a pattern object, you may ask? Excellent question!
            * A `Pattern` object stores the captured operations in the larger machine learning model. So, let’s say that the user defined a pattern ` return mb.relu(x=x, name='mycoolrelu') ` . Then, `pattern.mycoolrelu` would return the **captured** relu operation in the larger machine learning model!
            * The pattern also has the following additional attributes: 
                * `pattern.root_var`, which is the root variable of the first operation of the captured pattern (and corresponds to the user defined pattern’s root variable)
                * `pattern.final_op`, the operation in the larger machine learning model that corresponds to the last operation in the user defined pattern.
                * `pattern.block`, the block in the larger machine learning model where the pattern was found
                * `pattern.op_set`, a set of all the operations captured from the larger machine learning model. The user should call `pattern.op_list() `to return a list verision of the set (without duplicates)
            * Note: The user can add additional attributes to the pattern object using this method if they choose: 
          `pattern.add_attribute("attr_name", attribute)`
        * Returns `True` if the pattern satisfies certain constraints (ie constant input values, rank, etc). Basically, anything beyond its topological order with respect to operation types, which is already identical to that of the user defined pattern. Returns `False` otherwise.



* The third function the user needs to define is the following:
`def transform_pattern(pattern):`

    * Parameters
        * a `Pattern` object
    * This function needs to replace the captured operations (stored in the pattern object) with whatever you want! Feel free to define another MIL program and replace the pattern with that second program.



* The last thing the user needs to do is **call** the following function
`register_generic_pass(ops_arrangement, var_constraints, 
                            transform_pattern, pass_name, namespace)`

    * Parameters
        * `ops_arrangement`: the user defined pattern
        * `var_constraints`: the user defined function (see above)
        * `transform_pattern`: the user defined function (see above)
        * `pass_name`: a string representing the name of the pass.
        * `namespace`: a string representing the namespace of the pass (ie `"common"`)
    * Calling this function will register the pass will the given `passname` and `namespace`, so that it will be called when the passes are run.
    * If you have multiple patterns to detect for a single pass, just call this function multiple times with the respective `ops_arrangement`, `var_constraints`, and `transform_pattern`, but have the `pass_name` and `namespace` be the same. That way, all of these “mini passes” will be registered under the same pass!



## Gelu Example - Everything the User Does

```
# Full source @ coreml/coremltools/coremltools/converters/mil/experimental/passes/generic_gelu_tanh_approximation_fusion.py 
# This is a simple function defined by the user
# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.passes.helper import _check_var_scalar_value
from coremltools.converters.mil.experimental.passes.generic_pass_infrastructure import register_generic_pass

# This is the user defined pattern to detect
@mb.program(input_specs=[mb.TensorSpec(shape=([1, 1024, 4096])), ])
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
"""
y = x * (0.5 * (tanh(((.0447)x^3 + x ) * sqrt(2/pi)) + 1))


[...] -----> pow (3) ----> mul (.044715) ---> add -----> mul (sqrt(2/pi)) ---> tanh ----> add (1) ----> mul (0.5) -----> mul ---> [...]
  |                                            ^                                                                          ^
  |                                            |                                                                          |
  |------------------------------------------------------------------------------------------------------------------------

"""

# This is another user defined pattern to detect
# In this pattern, 0.5 is first multiplied with the input which is then multiplied with the tanh term.
# In pattern1, 0.5 is first multiplied with the tanh term, and then multiplied with input
@mb.program(input_specs=[mb.TensorSpec(shape=([1, 1024, 4096])), ])
def gelu_to_detect_2(x):
    pow = mb.pow(x=x, y=3.0, name ="pow")
    mul_1 = mb.mul(x=0.044714998453855515, y=pow, name="mul_1")
    add = mb.add(x=x, y=mul_1, name="add")
    mul_2 = mb.mul(x=0.7978845834732056, y=add, name="mul_2")
    tanh = mb.tanh(x=mul_2, name="tanh")
    add_1 = mb.add(x=1.0, y=tanh, name="add_1")
    mul = mb.mul(x = 0.5, y=x, name="mul")
    mul_3 = mb.mul(x=mul, y=add_1, name="mul_3")
    return mul_3

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

# Constraint enforcement
def var_constraints(pattern):
    passed = True

    passed = passed and (_check_var_scalar_value(pattern.mul.y, 0.5) or _check_var_scalar_value(pattern.mul.x, 0.5))
    passed = passed and _check_var_scalar_value(pattern.pow.y, 3.0)

    passed = passed and (
                        _check_var_scalar_value(pattern.mul_1.y, 0.044715) or
                        _check_var_scalar_value(pattern.mul_1.x,  0.044715)
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

# Transformation Logic
def transform_pattern(pattern):
    # remove all the ops, and replace with a gelu op
    out_name = pattern.mul_3.outputs[0].name
    x = mb.gelu(x=pattern.root_var, mode="TANH_APPROXIMATION", name=out_name, before_op=pattern.mul)

    pattern.mul_3.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=pattern.mul_3, old_var=pattern.mul_3.outputs[0], new_var=x
    )

    # Remove all the ops at once
    pattern.block.remove_ops(pattern.op_list())

# Registering the Pass
register_generic_pass(ops_arrangement=gelu_to_detect_1, var_constraints=var_constraints,
                        transform_pattern = transform_pattern, pass_name="fuse_gelu_tanh_approximation", namespace="common")

register_generic_pass(ops_arrangement=gelu_to_detect_2, var_constraints = var_constraints,
                        transform_pattern = transform_pattern, pass_name="fuse_gelu_tanh_approximation", namespace="common")


```



## Linear Bias Example - Everything the User Does

```
# Full source @ coreml/coremltools/coremltools/converters/mil/mil/passes/linear_bias_fusion.py arbitrary_shape = (get_new_symbol(), get_new_symbol())
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
```



## Layernorm/Instancenorm Fusion - Everything the User Does (for one of the patterns)

```
# Full source @coreml/coremltools/coremltools/converters/mil/experimental/passes/generic_layernorm_instancenorm_pattern_fusion.py 
@mb.program(input_specs=[mb.TensorSpec(shape=(1, 100, 100, 3)),])
def layernorm(x): 
    
    # MIL operation takes named inputs (instead of positional inputs). 
    
    y = mb.reduce_mean(x = x, keep_dims = False, name = "reduce_mean")
    x = sub(x = x, y =y, name = "add")
    ...
    x = add(x = x, y = sub, name = "last_add")
    return x 

# User defined helper function
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

# User defined helper function
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


def var_constraints(pattern) -> bool:
   
    root_var = pattern.reduce_op.x
    epsilon_var = pattern.add_op1.y if add_op1.x == pattern.reduce_op2.outputs[0] else pattern.add_op1.x
    gamma_var = pattern.mul_op1.y if pattern.mul_op1.x == pattern.rsqrt_op.outputs[0] else pattern.mul_op1.x
    beta_var = pattern.sub_op2.x
    rank = len(root_var.shape)

    passed = True

    passed = passed and _check_no_output_connection(pattern.block, pattern.to_list)

    passed = passed and root_var.shape is not None
    passed = passed and rank == 4
    passed = passed and _check_reduce_op(pattern.reduce_op)
    passed = passed and not(epsilon_var.val is None or len(epsilon_var.val.shape) != 0)
    passed = passed and gamma_var.val is not None
    passed = passed and beta_var.val is not None

    pattern.add_attribute('epsilon_var', epsilon_var)
    pattern.add_attribute('gamma_var', gamma_var)
    pattern.add_attribute('beta_var', beta_var)

    return constraints_passed


def transform_pattern(pattern):
    
    # Insert instance_norm / layer_norm and delete all ops.

    axes = pattern.reduce_op.axes.val
    rank = len(pattern.reduce_op.x.shape)

    # check whether the pattern is instance_norm or layer_norm
    is_layernorm = False
    is_instancenorm = False
    is_require_rank4_transpose = False

    negative_axes = [a - rank if a >= 0 else a for a in axes]
    negative_axes.sort()

    if len(pattern.gamma_var.val.shape) == len(axes) and len(pattern.beta_var.val.shape) == len(axes):
        # axes for layer_norm must be [-1] or [-1, -2] or [-1, -2, -3] and so on
        if negative_axes == list(range(-len(negative_axes), 0)):
            is_layernorm = True

    if rank == 4 and (negative_axes == [-2, -1] or negative_axes == [-3, -2]):
        if (
            len(np.squeeze(pattern.gamma_var.val).shape) == 1
            and len(np.squeeze(pattern.beta_var.val).shape) == 1
        ):
            is_instancenorm = True
        if negative_axes == [-3, -2]:
            is_require_rank4_transpose = True

    if not (is_instancenorm or is_layernorm):
        return False

    # remove all the ops, and replace with a layer_norm or instance_norm op
    out_name = pattern.end_op.outputs[0].name

    if is_require_rank4_transpose:
        x = mb.transpose(
            x=pattern.reduce_op.x,
            perm=[0, 3, 1, 2],
            name=out_name + "_transpose_nhwc_nchw",
            before_op=pattern.end_op,
        )
    if is_instancenorm:
        x = mb.instance_norm(
            x=x if is_require_rank4_transpose else pattern.reduce_op.x,
            gamma=np.squeeze(pattern.gamma_var.val),
            beta=np.squeeze(pattern.beta_var.val),
            epsilon=pattern.epsilon_var,
            name=out_name + "_instancenorm" if is_require_rank4_transpose else out_name,
            before_op=pattern.end_op,
        )
    else:  # is_layernorm
        x = mb.layer_norm(
            x=x if is_require_rank4_transpose else pattern.reduce_op.x,
            axes=axes,
            gamma=pattern.gamma_var,
            beta=pattern.beta_var,
            epsilon=pattern.epsilon_var,
            name=out_name + "_layernorm" if is_require_rank4_transpose else out_name,
            before_op=pattern.end_op,
        )
    if is_require_rank4_transpose:
        x = mb.transpose(
            x=x,
            perm=[0, 2, 3, 1],
            name=out_name + "_transpose_nchw_nhwc",
            before_op=pattern.end_op,
        )

    pattern.end_op.enclosing_block.replace_uses_of_var_after_op(
        anchor_op=pattern.end_op, old_var=pattern.end_op.outputs[0], new_var=x
    )
    # Remove all the ops at once
    block.remove_ops(pattern.to_list)
    return True


`register_generic_pass`(ops_arrangement=layernorm, var_constraints = var_constraints,
                transform_pattern = transform_pattern, 
                pass_name="layernorm_pass", namespace="common")
```



## _**Understanding the Infrastructure: Implementation Details**_

* This is a list of all the internal functions in my infrastructure, and what they each do. Remember, the goal is to detect a small user-defined MIL program inside a larger machine learning model (also a MIL program). Most of these functions are in the `coreml/coremltools/coremltools/converters/mil/experimental/passes/generic_pass_infrastructure.py` file
* The first (highest level) function:
`register_generic_pass(ops_arrangement, var_constraints, transform_pattern, pass_name, namespace)`
    * Parameters
        * `ops_arragement` : The user defined MIL program we are trying to detect
        * `var_constraints` : The user defined function that takes in a `Pattern` object (which stores captured operations in the larger machine learning model) as a parameter, and returns whether the captured operations in that object satisfy certain constraints
        * `transform_pattern` : The user defined function that takes in a `Pattern` object (which stores captured operations in the larger machine learning model) as a parameter, and replaces those captured operations with the desired operations in the larger machine learning model
        * `pass_name` : A string that is the name of the pass
        * `namespace`: A string that is the namespace where the pass is registered
    * Results
        * This function registers a pass with the given parameters
* The second function, called by the one above:
`fuse_all_blocks(ops_arrangement, var_constraints, transform_pattern, prog)`
    * Parameters
        * `ops_arragement` : The user defined MIL program we are trying to detect
        * `var_constraints` : The user defined function that takes in a `Pattern` object (which stores captured operations in the larger machine learning model) as a parameter, and returns whether the captured operations in that object satisfy certain constraints
        * `transform_pattern` : The user defined function that takes in a `Pattern` object (which stores captured operations in the larger machine learning model) as a parameter, and replaces those captured operations with the desired operations in the larger machine learning model
        * `prog` : The large machine learning model (represented in MIL) in which we are tying to detect `ops_arragement`
    * Results
        * This function replaces all instances of `ops_arragement` in `prog` with the desired replacement code in `transform_pattern`
* The third function, called by the one above:
`fuse_one_block(block, ops_arrangement, var_constraints, transform_pattern)`
    * Parameters
        * `block`: The block in the main machine learning model that we are looking into right now
        * `ops_arragement` : The user defined MIL program we are trying to detect
        * `var_constraints` : The user defined function that takes in a `Pattern` object (which stores captured operations in the larger machine learning model) as a parameter, and returns whether the captured operations in that object satisfy certain constraints
        * `transform_pattern` : The user defined function that takes in a `Pattern` object (which stores captured operations in the larger machine learning model) as a parameter, and replaces those captured operations with the desired operations in the larger machine learning model
    * Results
        * This function replaces the first instance of `ops_arragement` in `block` with the desired replacement code in `transform_pattern`
* The fourth function, called by the one above:
`detect_pattern(program_op, ops_arrangement_root_var, block)`

    * Parameters
        * `program_op`: A single operation in the main machine learning model
        * `ops_arrangement_root_var` : The root variable for the user defined MIL program we are trying to detect. **Assumption: this program has only one root variable**
        * `block`: The block in the main machine learning model that we are looking into right now
    * Results
        * This function does the following:
            * Creates a `Pattern` object to capture operations and other relevant details from the main machine learning model
            * Sets the `Pattern` object’s `block` and `root_var` attributes. `Root_var` is to the **single** variable input of the `program_op` that corresponds to the `ops_arrangement_root_var`. In other words, if you remember your SAT prep from high school, `ops_arrangement_root_var` is to `ops_arragement` as `pattern.root_var` is to the main machine learning model. **Since we are assuming that the user defined pattern has only 1 root variable, if `program_op` has more than 1 input variable, we loop through these inputs to find the one that corresponds to the one in the user defined pattern. Here, “corresponds” is defined as recursively having the same number and type of child operations, in the same topological order.**
            * Sets the `Pattern` object operation attributes. Each of these attribute's names correspond to the names given to the operations in the user defined pattern.
            * Sets the Pattern object `final_op` attribute. This is the operation in the main machine learning model that corresponds to the last operation in the user defined pattern. For this last operation, we always only verify that the operation types are the same (we don’t care about child operations). 
                * We also check that this is the only operation in the captured pattern that has an output that is also in the block’s output. If it is not, we return `False, None`
                * **Assumption: Here, we are assuming that the user defined pattern has exactly one proper last operation. If the user defined pattern has multiple “last operations” (ie, operations with 0 child operations) then the `final_op` will be set to only one of these last operations, and the check mentioned above will fail - therefore, not capturing the pattern.**
            * Returns `True, pattern` if the user defined pattern is found in the main machine learning model starting at `program_op` ‘s root variable, and `False, None` otherwise
* The fifth function, called by the one above:
`pattern_detected(pattern, program_op, pattern_op, program_root_var, pattern_root_var, block)`
    * Parameters
        * `pattern`: A `Pattern` object
        * `program_op` : The current operation in the main machine learning model that we are looking at
        * `pattern_op` : The current operation in the user defined pattern that we are looking at
        * `program_root_var` : The variable in the main machine learning model that is analogous to the root variable in the user defined pattern. 
        * `pattern_root_var` : The root variable in the user defined pattern
        * `block`: The block in the main machine learning model that we are looking into right now
    * Results
        * This recursively looks at operations and their children in the main machine learning model, and returns true if the following conditions are met, and false otherwise
            * Every operation in the user defined pattern has the same operation type and number of outputs as its counterpart in the main machine learning model
            * Every operation in the user defined pattern has the same number and type of child operations as its counterpart in the main machine learning model (recursive call). This constraint is not enforced if the operation in the user defined pattern has 0 children.
            * **Assumption: If an `program_op` and the `pattern_op` have the same number of outputs, we are assuming that, if there is a match, those outputs are stored in the same order. Child operations do not have to be ordered.**
* The sixth function, called by the one above:
`lists_op_equality(oplist1, oplist2)`
    * Parameters
        * `oplist1`: A list of operations
        * `oplist2` : A list of operations
    * Results
        * Returns True if the operations in `oplist1` are in the same order and have the same operation type as the operations in `oplist2` and False otherwise.
* The `Pattern` class
    * Stores a bunch of stuff, including operations, and in addition has `root_var`, `bock`, `op_set` and `final_op` attributes. The user can, of course, add more attributes to the pattern in their functions if they wish, using `pattern.add_attribute(attribute_name, attribute`)
    * `pattern.op_list()` Returns a list of all unique operations stored in the pattern
* The `PassContainer` class
    * In the new infrastructure, each new pattern that the user wants to detect needs to be defined and registered separately. If the user wants to group each of these “subpasses” together, they can register them with the same name and namespace, and all the “subpasses” will be stored in a `PassContainer` instance, where they will eventually all be executed. 
    * `PassContainer(pass_name)`: makes a new `PassContainer` object with a single pass name (String)
    *  `passContainer.add(pass_func)` adds a pass function to the `PassContainer’s` list of pass functions. A pass function is a function that takes in a machine learning model as a parameter and transforms it into the compressed, transformed machine learning model. This is a partial function of `fuse_all_blocks` defined above.
    * `PassContainer.__call__(prog)` : Executes all `pass_functions` stored in this `PassContainer` object with respect to the given machine learning model

## _**How to Add/Run a Pass**_

* Write the pass, and save it in a file in the `coreml/coremltools/coremltools/converters/mil/experimental/passes` folder
* Add an import line to the `coreml/coremltools/coremltools/converters/mil/mil/passes/init.py` file
* Run the experimental (generic) passes by setting the `ENABLE_EXPERIMENTAL_PASSES` environment variable to 1, which will override the regular passes with the same name
