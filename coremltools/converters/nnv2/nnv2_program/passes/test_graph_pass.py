from coremltools.converters.nnv2.nnv2_program import passes
from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as builder
from coremltools.converters.nnv2.builtin_types.symbolic import any_symbolic
from coremltools.converters.nnv2.testing_utils import build_main_program, assert_op_count_match, assert_model_is_valid
from coremltools.converters.nnv2.nnv2_program.program import Symbol
from coremltools.converters.nnv2.nnv2_program.passes.pass_registry import PASS_REGISTRY

import numpy as np

np.random.seed(1984)
validate_model = True

# TODO: rdar://58993652 (Add recursive block test cases for graph pass tests)


def test_const_elimination():
    @build_main_program(inputs={'x': builder.placeholder(shape=(2, 4))})
    def program(x):
        a = np.random.rand(2, 4).astype(np.float32)
        double_a = builder.add(x=a, y=a)
        return builder.add(x=x, y=double_a)

    assert_op_count_match(program, expect=2, op='const')
    PASS_REGISTRY['const_elimination'](program)
    assert_op_count_match(program, expect=3, op='const')

    if validate_model:
        assert_model_is_valid(program, {'x': (2, 4)})


def test_matmul_to_linear():
    @build_main_program(inputs={'x': builder.placeholder(shape=(2, 4))})
    def program(x):
        weights_val = np.random.rand(2, 4).T.astype(np.float32)
        weights = builder.const(val=weights_val, mode='immediate_value')
        bias_val = np.random.rand(2).astype(np.float32)
        bias = builder.const(val=bias_val, mode='immediate_value')

        matmul = builder.matmul(x=x, y=weights)
        return builder.add(x=matmul, y=bias)

    assert_op_count_match(program, expect=1, op='matmul')
    assert_op_count_match(program, expect=0, op='linear')
    PASS_REGISTRY['matmul_to_linear'](program)
    assert_op_count_match(program, expect=0, op='matmul')
    assert_op_count_match(program, expect=1, op='linear')

    if validate_model:
        assert_model_is_valid(program, {'x': (2, 4)})


def test_dead_code_elimination():
    @build_main_program(
        inputs={
            'x': builder.placeholder(shape=(2, 4)),
            'y': builder.placeholder(shape=(2, 4))
        })
    def program0(x, y):
        # following three unused op should be eliminated
        a = builder.const(val=np.zeros(shape=(1, )), mode='immediate_value')
        b = builder.const(val=np.zeros(shape=(1, )), mode='immediate_value')
        _ = builder.add(x=a, y=b)
        return builder.add(x=x, y=y)

    assert_op_count_match(program0, expect=4)
    PASS_REGISTRY['dead_code_elimination'](program0)
    assert_op_count_match(program0, expect=1)

    if validate_model:
        assert_model_is_valid(program0, {'x': (2, 4), 'y': (2, 4)})

    @build_main_program(inputs={'x': builder.placeholder(shape=(2, 4))})
    def program1(x):
        weights_val = np.random.rand(2, 4).T.astype(np.float32)
        weights = builder.const(val=weights_val, mode='immediate_value')
        bias_val = np.random.rand(4).astype(np.float32)
        bias = builder.const(val=bias_val, mode='immediate_value')

        # unused op and its inputs should be eliminated
        builder.matmul(x=x, y=weights)

        return builder.linear(x=x, weight=weights, bias=bias)

    assert_op_count_match(program1, expect=6)
    PASS_REGISTRY['dead_code_elimination'](program1)
    assert_op_count_match(program1, expect=3)

    if validate_model:
        assert_model_is_valid(program1, {'x': (2, 4)})

def test_remove_symbolic_reshape():
    sym_b = Symbol('s0')
    original_shape = (sym_b, Symbol('s1'), 2)
    func_inputs = {"x": builder.placeholder(shape=(sym_b, 4))}
    reshape_name = 'reshape'
    @build_main_program(inputs=func_inputs)
    def program(x):
        # const cannot represent symbolic values. Use _const_symbolic
        shape = builder._const_symbolic(val=original_shape)
        return builder.reshape(x=x, shape=shape, name=reshape_name)

    reshape_op = program.find_ops(prefix=reshape_name, op_type='reshape',
            exactly_one=True)[0]
    shape_var = reshape_op.shape
    reshaped_var = reshape_op.outputs[0]
    assert np.all(shape_var.sym_val == original_shape)
    assert np.all(reshaped_var.shape == (sym_b, 2, 2))
    PASS_REGISTRY['remove_symbolic_reshape'](program)
    reshape_op = program.find_ops(prefix=reshape_name, op_type='reshape',
            exactly_one=True)[0]
    shape_var = reshape_op.shape
    reshaped_var = reshape_op.outputs[0]
    # shape param cannot be symbolic after the pass
    assert np.all(shape_var.sym_val == (-1, 2, 2))
    # output shape is still symbolic
    assert np.all(reshaped_var.shape == (sym_b, 2, 2))

    if validate_model:
        assert_model_is_valid(program, {'x': (3, 4)})
