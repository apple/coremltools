from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from coremltools.converters.nnv2.nnv2_program.passes.pass_registry import PASS_REGISTRY 
from coremltools.converters.nnv2.testing_utils import (
        assert_model_is_valid, assert_same_output_names)
import copy
import pytest
pytest.importorskip('tensorflow', minversion='1.14.0')

def test_backfill_make_list_elem_type():

    # The while_loop appends [1, 2]*i to `ls` for each iteration
    # i = 0, ... num_iters-1.

    elem_shape = (2,)
    @cb.program(input_specs=[
        cb.TensorSpec(shape=elem_shape),
        ])
    def prog(update):
        def body(i, ls):
            return cb.add(x=i, y=1), \
                    cb.list_write(ls=ls, index=i, value=update)

        def cond(i, ls):
            return cb.less(x=i, y=num_iters)

        i = 0
        ls = cb.tf_make_list(init_length=1)
        num_iters = 3
        _, final_tensor_list = cb.while_loop(_cond=cond, _body=body,
                loop_vars=(i, ls))
        list_len = cb.list_length(ls=final_tensor_list)
        indices = cb.range_1d(start=0, end=list_len, step=1)
        return cb.list_gather(ls=final_tensor_list,
                indices=indices)

    # tf_make_list has no elem_type info
    make_list_op = prog.find_ops(op_type='tf_make_list', exactly_one=True)[0]
    assert make_list_op.outputs[0].elem_type == builtins.unknown

    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY['tensorflow::backfill_make_list_elem_type'](prog)
    assert_same_output_names(prev_prog, prog)
    prog.validate()

    # tf_make_list is replaced with make_list and should have elem_type now
    make_list_op = prog.find_ops(op_type='make_list', exactly_one=True)[0]
    assert make_list_op.outputs[0].elem_type.get_shape() == elem_shape

    assert_model_is_valid(prog, {'update': elem_shape})
