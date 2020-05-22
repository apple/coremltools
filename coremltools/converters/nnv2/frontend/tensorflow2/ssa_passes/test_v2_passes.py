from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from coremltools.converters.nnv2.testing_utils import (
    assert_op_count_match,
    assert_model_is_valid,
    assert_same_output_names)
from coremltools.converters.nnv2.nnv2_program.passes.pass_registry import PASS_REGISTRY
from coremltools.converters.nnv2.builtin_types import builtins
import copy

import numpy as np

np.random.seed(1984)
validate_model = True


def test_remove_vacuous_cond():
    @cb.program(input_specs=[
        cb.TensorSpec(shape=(1,), dtype=builtins.bool),
        cb.TensorSpec(shape=(2, 3))])
    def prog(a, b):
        def then_branch():
            return cb.identity(x=b)

        def else_branch():
            return cb.identity(x=b)

        pred = cb.squeeze(x=a)
        return cb.cond(pred=pred, _true_fn=then_branch, _false_fn=else_branch)

    cond_op = prog.find_ops(op_type='cond', exactly_one=True)[0]
    original_cond_op_name = cond_op.name
    assert len(cond_op.blocks[0].operations) == 1
    assert len(cond_op.blocks[1].operations) == 1
    assert cond_op.blocks[0].operations[0].op_type == 'identity'
    assert cond_op.blocks[1].operations[0].op_type == 'identity'

    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY['tensorflow2::remove_vacuous_cond'](prog)
    assert_same_output_names(prev_prog, prog)

    cond_op = prog.find_ops(op_type='cond')
    assert len(cond_op) == 0
    identity_op = prog.find_ops(prefix=original_cond_op_name, exactly_one=True)[0]
    assert identity_op.op_type == 'identity'

    if validate_model:
        assert_model_is_valid(prog, {'a': (1,), 'b': (2, 3)})
