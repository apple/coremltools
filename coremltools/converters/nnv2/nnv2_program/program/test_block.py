# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb

"""
Test manipulating variable and operations in the SsaBlock.

In the test, we are actually testing SsaFunction, which is a child class of
SsaBlock. Technically SsaFunction should not inherit from SsaBlock, which is a
debt to be resolved in the future.

SsaFunction has some different behaviors from SsaBlock that are irrelevant to
the core API being tested here.
"""


def test_empty_block():
    """Test an empty program
    """

    @cb.program(input_specs=[cb.TensorSpec(shape=(2, 4))])
    def prog(x0):
        return x0

    block = prog.functions["main"]
    assert len(block.operations) == 0
    assert len(block.inputs) == 1
    assert len(block.outputs) == 1
    assert block.inputs["x0"] == block.outputs[0]
    print(prog)


def test_add_op():
    """Test add statement to an empty program, also change the output
    """

    @cb.program(input_specs=[cb.TensorSpec(shape=(2, 4))])
    def prog(x0):
        return x0

    print("before:\n{}".format(prog))
    block = prog.functions["main"]
    x0 = block.inputs["x0"]
    with block:
        x1 = cb.log(x=x0)
    block.set_outputs([x1])
    print("after:\n{}".format(prog))
    assert block.inputs["x0"] == block.find_ops(op_type="log")[0].inputs["x"]
    assert len(block.operations) == 1
    assert block.operations[0].op_type == "log"
    assert block.outputs[0] == x1


def test_remove_op():
    """Test remove all ops and return empty program
    """

    @cb.program(input_specs=[cb.TensorSpec(shape=(2, 4))])
    def prog(x0):
        x1 = cb.log(x=x0)
        return x1

    print("before:\n{}".format(prog))
    block = prog.functions["main"]
    x0 = block.inputs["x0"]
    ops = block.find_ops(op_type="log")
    block.set_outputs([x0])
    block.remove_ops(ops)
    print("after:\n{}".format(prog))
    assert len(block.operations) == 0
    assert len(block.inputs) == 1
    assert len(block.outputs) == 1
    assert block.inputs["x0"] == block.outputs[0]


def test_remove_op2():
    """Test remove ops with multiple identical inputs
    """

    @cb.program(input_specs=[cb.TensorSpec(shape=(2, 4))])
    def prog(x0):
        x1 = cb.add(x=x0, y=x0)
        return x1

    print("before:\n{}".format(prog))
    block = prog.functions["main"]
    x0 = block.inputs["x0"]
    ops = block.find_ops(op_type="add")
    block.set_outputs([x0])
    block.remove_ops(ops)
    print("after:\n{}".format(prog))
    assert len(block.operations) == 0
    assert len(block.inputs) == 1
    assert len(block.outputs) == 1
    assert block.inputs["x0"] == block.outputs[0]


def test_simple_substituion():
    """Replace log(x+y) with log(x*y)
    """

    @cb.program(input_specs=[cb.TensorSpec(shape=(2, 4)), cb.TensorSpec(shape=(2, 4))])
    def prog(x0, y0):
        x1 = cb.add(x=x0, y=y0)
        z = cb.log(x=x1)
        return z

    print("before:\n{}".format(prog))
    block = prog.functions["main"]
    assert len(block.find_ops(op_type="log")) == 1
    assert len(block.find_ops(op_type="add")) == 1
    assert len(block.find_ops(op_type="mul")) == 0

    add = block.find_ops(op_type="add")[0]

    x0 = add.inputs["x"]
    y0 = add.inputs["y"]
    x1 = add.outputs[0]

    with block:
        # It's important add 'mul' before 'add' (its even better to do it
        # immeidately after 'add' but we don't have the API)
        # because we need to replace any op affected by add with 'mul'
        x2 = cb.mul(x=x0, y=y0, before_op=add)

    assert len(block.find_ops(op_type="mul")) == 1
    assert len(block.find_ops(op_type="add")) == 1
    assert len(block.find_ops(op_type="log")) == 1

    # It's important to set anchor_op = 'mul' becuase new_var is only visible
    # after 'mul'.
    block.replace_var_after_op(anchor_op=x2.op, old_var=x1, new_var=x2)
    block.remove_ops([add])

    print("after:\n{}".format(prog))
    assert len(block.find_ops(op_type="add")) == 0
    assert len(block.find_ops(op_type="mul")) == 1
    assert len(block.find_ops(op_type="log")) == 1


def test_substitute_nested_op():
    """"Replace an conditional op with nested block"""

    @cb.program(input_specs=[cb.TensorSpec(shape=(2, 4)), cb.TensorSpec(shape=(2, 4))])
    def prog(x0, y0):
        pred = cb.less(x=x0, y=y0)
        z = cb.cond(
            pred=pred, _true_fn=lambda: cb.abs(x=x0), _false_fn=lambda: cb.abs(x=y0)
        )
        z1 = cb.log(x=z)
        return z1

    print("before:\n{}".format(prog))
    block = prog.functions["main"]
    assert len(block.find_ops(op_type="less")) == 1
    assert len(block.find_ops(op_type="abs")) == 2
    assert len(block.find_ops(op_type="cond")) == 1
    assert len(block.find_ops(op_type="log")) == 1

    cond = block.find_ops(op_type="cond")[0]
    x0 = block.inputs["x0"]
    z = cond.outputs[0]
    block.replace_var_after_op(anchor_op=None, old_var=z, new_var=x0)

    # removing cond will also remove the abs ops within its block
    block.remove_ops([cond])

    print("after:\n{}".format(prog))
    assert len(block.find_ops(op_type="less")) == 1
    assert len(block.find_ops(op_type="log")) == 1
    assert len(block.find_ops(op_type="cond")) == 0
    assert len(block.find_ops(op_type="abs")) == 0


def test_simple_transpose_squash():
    """Test eliminiate consecutive transpose can be canceled
    """

    @cb.program(input_specs=[cb.TensorSpec(shape=(2, 4))])
    def prog(x0):
        x1 = cb.transpose(x=x0, perm=[1, 0])
        x2 = cb.transpose(x=x1, perm=[1, 0])
        x3 = cb.log(x=x2)
        x4 = cb.transpose(x=x3, perm=[1, 0])
        x5 = cb.transpose(x=x4, perm=[1, 0])
        x6 = cb.transpose(x=x5, perm=[1, 0])
        x7 = cb.transpose(x=x6, perm=[1, 0])
        return x7

    print("before:\n{}".format(prog))
    block = prog.functions["main"]
    assert len(block.find_ops(op_type="transpose")) == 6

    def can_squash(trans1, trans2):
        return (
            len(trans1.outputs) == 1
            and len(trans2.outputs) == 1
            and all(trans1.perm.val == trans2.perm.val)
        )

    # Find all candidate pairs tranposes
    # we ignore all const (transpose_perm_%x), and add pairs of transpose op as
    # candidate. This won't generalize to more complicated program with other
    # shape invariant ops in between.
    candidates = []
    non_const_ops = [op for op in block.operations if op.op_type != "const"]
    for i in range(len(non_const_ops) - 1):
        op = non_const_ops[i]
        if len(candidates) and op == candidates[-1][1]:
            # op is already a squash candidate
            continue
        next_op = non_const_ops[i + 1]
        if (
            op.op_type == "transpose"
            and next_op.op_type == "transpose"
            and can_squash(op, next_op)
        ):
            candidates.append((op, next_op))

    # Remove each candidate pairs
    for (trans1, trans2) in candidates:
        before = trans1.inputs["x"]
        after = trans2.outputs[0]
        block.replace_var_after_op(anchor_op=trans2, old_var=after, new_var=before)
        block.remove_ops([trans1, trans2])

    print("after:\n{}".format(prog))
    assert len(block.find_ops(op_type="transpose")) == 0
