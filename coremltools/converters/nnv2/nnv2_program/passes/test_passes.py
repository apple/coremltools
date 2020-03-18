from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from coremltools.converters.nnv2.testing_utils import (
        assert_op_count_match, assert_model_is_valid,
        assert_same_output_names)
from coremltools.converters.nnv2.nnv2_program.program import Symbol
from coremltools.converters.nnv2.nnv2_program.passes.pass_registry import PASS_REGISTRY
import copy

import numpy as np

np.random.seed(1984)
validate_model = True

# TODO: rdar://58993652 (Add recursive block test cases for graph pass tests)


def test_const_elimination():
    @cb.program(input_specs=[cb.TensorSpec(shape=(2, 4))])
    def prog(x):
        a = np.random.rand(2, 4).astype(np.float32)
        double_a = cb.add(x=a, y=a)
        return cb.add(x=x, y=double_a)

    assert_op_count_match(prog, expect=2, op='const')
    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY['common::const_elimination'](prog)
    assert_same_output_names(prev_prog, prog)
    assert_op_count_match(prog, expect=3, op='const')

    if validate_model:
        assert_model_is_valid(prog, {'x': (2, 4)})


def test_matmul_to_linear():
    @cb.program(input_specs=[cb.TensorSpec(shape=(2, 4))])
    def prog(x):
        weights_val = np.random.rand(2, 4).T.astype(np.float32)
        weights = cb.const(val=weights_val, mode='immediate_value')
        bias_val = np.random.rand(2).astype(np.float32)
        bias = cb.const(val=bias_val, mode='immediate_value')

        matmul = cb.matmul(x=x, y=weights)
        return cb.add(x=matmul, y=bias)

    assert_op_count_match(prog, expect=1, op='matmul')
    assert_op_count_match(prog, expect=0, op='linear')
    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY['common::matmul_to_linear'](prog)
    assert_same_output_names(prev_prog, prog)
    assert_op_count_match(prog, expect=0, op='matmul')
    assert_op_count_match(prog, expect=1, op='linear')

    if validate_model:
        assert_model_is_valid(prog, {'x': (2, 4)})


def test_dead_code_elimination():
    @cb.program(
        input_specs=[
            cb.TensorSpec(shape=(2, 4)),
            cb.TensorSpec(shape=(2, 4)),
            ])
    def program0(x, y):
        # following three unused op should be eliminated
        a = cb.const(val=np.zeros(shape=(1, )), mode='immediate_value')
        b = cb.const(val=np.zeros(shape=(1, )), mode='immediate_value')
        _ = cb.add(x=a, y=b)
        return cb.add(x=x, y=y)

    assert_op_count_match(program0, expect=4)
    prev_prog = copy.deepcopy(program0)
    PASS_REGISTRY['common::dead_code_elimination'](program0)
    assert_same_output_names(prev_prog, program0)
    assert_op_count_match(program0, expect=1)

    if validate_model:
        assert_model_is_valid(program0, {'x': (2, 4), 'y': (2, 4)})

    @cb.program(input_specs=[cb.TensorSpec(shape=(2, 4))])
    def program1(x):
        weights_val = np.random.rand(2, 4).T.astype(np.float32)
        weights = cb.const(val=weights_val, mode='immediate_value')
        bias_val = np.random.rand(4).astype(np.float32)
        bias = cb.const(val=bias_val, mode='immediate_value')

        # unused op and its inputs should be eliminated
        cb.matmul(x=x, y=weights)

        return cb.linear(x=x, weight=weights, bias=bias)

    assert_op_count_match(program1, expect=6)
    prev_prog = copy.deepcopy(program1)
    PASS_REGISTRY['common::dead_code_elimination'](program1)
    assert_same_output_names(prev_prog, program1)
    assert_op_count_match(program1, expect=3)

    if validate_model:
        assert_model_is_valid(program1, {'x': (2, 4)})

def test_remove_symbolic_reshape():
    sym_b = Symbol('s0')
    original_shape = (sym_b, Symbol('s1'), 2)
    reshape_name = 'reshape'
    @cb.program(input_specs=[cb.TensorSpec(shape=(sym_b, 4))])
    def prog(x):
        # const cannot represent symbolic values. Use _const_symbolic
        shape = cb._const_symbolic(val=original_shape)
        return cb.reshape(x=x, shape=shape, name=reshape_name)

    reshape_op = prog.find_ops(prefix=reshape_name, op_type='reshape',
            exactly_one=True)[0]
    shape_var = reshape_op.shape
    reshaped_var = reshape_op.outputs[0]
    assert np.all(shape_var.sym_val == original_shape)
    assert np.all(reshaped_var.shape == (sym_b, 2, 2))

    # Note: we cannot deepcopy prog with symbol.
    prev_outputs = [o.name for o in prog['main'].outputs]
    PASS_REGISTRY['common::remove_symbolic_reshape'](prog)
    curr_outputs = [o.name for o in prog['main'].outputs]
    assert curr_outputs == prev_outputs

    reshape_op = prog.find_ops(prefix=reshape_name, op_type='reshape',
            exactly_one=True)[0]
    shape_var = reshape_op.shape
    reshaped_var = reshape_op.outputs[0]
    # shape param cannot be symbolic after the pass
    assert np.all(shape_var.sym_val == (-1, 2, 2))
    # output shape is still symbolic
    assert np.all(reshaped_var.shape == (sym_b, 2, 2))

    if validate_model:
        assert_model_is_valid(prog, {'x': (3, 4)})


def test_loop_invariant_elimination():
    def body(a, b):
        return cb.add(x=a, y=b), b

    def cond(a, b):
        a_mean = cb.reduce_mean(x=a, axes=[0, 1])
        b_mean = cb.reduce_mean(x=b, axes=[0, 1])
        return cb.less(x=a_mean, y=b_mean)

    @cb.program(input_specs=[
        cb.TensorSpec(shape=(1,2)),
        cb.TensorSpec(shape=(1,2)),
        ])
    def prog(a, b):
        return cb.while_loop(_cond=cond, _body=body, loop_vars=(a, b))

    while_op = prog.find_ops(op_type='while_loop', exactly_one=True)[0]
    assert len(while_op.blocks[0].inputs) == 2
    assert len(while_op.blocks[1].inputs) == 2
    assert len(while_op.outputs) == 2
    assert len(while_op.loop_vars) == 2
    assert while_op.blocks[0].inputs[0].name == 'a.x'
    assert while_op.blocks[0].inputs[1].name == 'b.x'
    assert while_op.blocks[1].inputs[0].name == 'a.x'
    assert while_op.blocks[1].inputs[1].name == 'b.x'

    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY['common::loop_invariant_elimination'](prog)
    assert_same_output_names(prev_prog, prog)

    while_op = prog.find_ops(op_type='while_loop', exactly_one=True)[0]
    assert len(while_op.blocks[0].inputs) == 1
    assert len(while_op.blocks[1].inputs) == 1
    assert len(while_op.outputs) == 1
    assert len(while_op.loop_vars) == 1
    assert while_op.blocks[0].inputs[0].name == 'a.x'
    assert while_op.blocks[1].inputs[0].name == 'a.x'

    if validate_model:
        assert_model_is_valid(prog, {'a': (1, 2), 'b': (1, 2)})
