import copy
from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from coremltools.converters.nnv2.nnv2_program.passes.pass_registry import PASS_REGISTRY
from coremltools.converters.nnv2.testing_utils import assert_model_is_valid
from coremltools.converters.nnv2.testing_utils import (
        assert_same_output_names)

def test_commingle_loop_vars():
    def body(a, b):
        # b is a loop invariant
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
    assert while_op.blocks[0].inputs[0].name == 'a.x'
    assert while_op.blocks[0].inputs[1].name == 'b.x'

    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY['nnv1_backend::commingle_loop_vars'](prog)
    assert_same_output_names(prev_prog, prog)

    while_op = prog.find_ops(op_type='while_loop', exactly_one=True)[0]
    assert while_op.blocks[0].inputs[0].name == while_op.outputs[0].name
    assert while_op.blocks[0].inputs[1].name == while_op.outputs[1].name

    prog.validate()

    # The program is not ssa and thus cannot be converted


def test_handle_return_return_inputs_as_outputs():
    @cb.program(input_specs=[
        cb.TensorSpec(shape=(1,2)),
        cb.TensorSpec(shape=(1,2)),
        ])
    def prog(a, b):
        return cb.mul(x=a, y=2), b

    prev_main_output_names = [o.name for o in prog['main'].outputs]
    assert prog['main'].outputs[1].op is None  # output comes from input

    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY['nnv1_backend::handle_return_inputs_as_outputs'](prog)
    assert_same_output_names(prev_prog, prog)

    assert prog['main'].outputs[1].op is not None  # output comes from an op
    assert prog['main'].outputs[1].op.op_type == 'identity'

    assert_model_is_valid(prog, {'a': (1, 2), 'b': (1, 2)})

def test_handle_unused_inputs():
    @cb.program(input_specs=[
        cb.TensorSpec(shape=(1,2)),
        ])
    def prog(unused_input):
        return cb.const(val=[3, 2])

    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY['nnv1_backend::handle_unused_inputs'](prog)
    assert_same_output_names(prev_prog, prog)

    id_op = prog.find_ops(op_type='identity', exactly_one=True)[0]
    # Assert that input var is consumed by an identity op.
    assert id_op in prog['main'].inputs['unused_input'].child_ops

    assert_model_is_valid(prog, {'unused_input': (1, 2)})
