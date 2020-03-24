import six
from coremltools.converters.nnv2.builtin_types.builtins.type_mapping import numpy_val_to_builtin_val
from coremltools.converters.nnv2.nnv2_program.program.program import (
        SsaBlock, SYMBOL)
from coremltools.converters.nnv2.nnv2_program.program.var import Var
from ._op_reqs import *

# rdar://58622145
@register_op(doc_str='TODO')
class cond(Operation):
    input_spec = InputSpec(
            pred = BoolInputType(),
            _true_fn = PyFunctionInputType(),
            _false_fn = PyFunctionInputType(),
            )

    def __init__(self, **kwargs):
        super(cond, self).__init__(**kwargs)

    def build_nested_blocks(self):
        # Cond block
        true_block_name = self.name + '_true'
        with SsaBlock(name=true_block_name, outer_op=self) as true_block:
            true_func = self._true_fn.val
            true_ret_vars = true_func()
            if isinstance(true_ret_vars, tuple):
                true_ret_vars = list(true_ret_vars)
            if not isinstance(true_ret_vars, list):
                true_ret_vars = [true_ret_vars]
            true_block.set_outputs(true_ret_vars)
            self.blocks.append(true_block)

        false_block_name = self.name + '_false'
        with SsaBlock(name=false_block_name, outer_op=self) as false_block:
            false_func = self._false_fn.val
            false_ret_vars = false_func()
            if isinstance(false_ret_vars, tuple):
                false_ret_vars = list(false_ret_vars)
            if not isinstance(false_ret_vars, list):
                false_ret_vars = [false_ret_vars]
            false_block.set_outputs(false_ret_vars)
            self.blocks.append(false_block)

    def type_inference(self):
        true_ret_vars = self.blocks[0].outputs
        false_ret_vars = self.blocks[1].outputs
        # Verify true_ret_vars has the same types as false_ret_vars
        for i, (vt, vf) in enumerate(zip(true_ret_vars, false_ret_vars)):
            if vt.sym_type != vf.sym_type:
                msg = "true branch output {} type {} mismatch false branch" +\
                    " output type {}"
                raise ValueError(msg.format(vt.name, vt.sym_type, vf.sym_type))

        return tuple(v.sym_type for v in true_ret_vars)


# rdar://58622145
@register_op(doc_str='TODO')
class const(Operation):
    input_spec = InputSpec(
            mode = InternalStringInputType(const=True,
                default="immediate_value"),
            val = InternalScalarOrTensorInputType(const=True),
            )

    def __init__(self, **kwargs):
        super(const, self).__init__(**kwargs)

    def type_inference(self):
        builtin_type, _ = self._get_type_val(self.val.val)
        return builtin_type

    def value_inference(self):
        _, val = self._get_type_val(self.val.val)
        return val

    def _get_type_val(self, value):

        if isinstance(value, float):
            value = np.float32(value)
        elif isinstance(value, bool):
            value = np.bool(value)
        elif isinstance(value, six.integer_types):
            value = np.int32(value)
        elif isinstance(value, (tuple, list, np.ndarray)):
            value = np.array(value)
            if value.dtype == np.int64:
                # We use int32 by default.
                value = value.astype(np.int32)

        if not isinstance(value, (np.generic, np.ndarray,
            six.string_types, bool)):
            raise ValueError("Unknown value for constant: {}".format(value))

        _, builtin_type = numpy_val_to_builtin_val(value)
        return builtin_type, value


# Internal const can have symbolic value (for testing purpose)
@register_op(doc_str='TODO')
class _const_symbolic(const):
    def __init__(self, **kwargs):
        super(_const_symbolic, self).__init__(**kwargs)

    def type_inference(self):
        builtin_type, _ = self._get_type_val(self.val.sym_val)
        return builtin_type

    def value_inference(self):
        # We allow symbolic values in _const_symbolic
        _, val = self._get_type_val(self.val.sym_val)
        return val


@register_op(doc_str="""
Returns the elements selected from either a or b, depending on the cond.
Shape of cond, a, b must be broadcastable.

Inputs

* cond <*, T>
    * Tensor, when True (non-zero), select element from x, otherwise, y
* a <*, T> Optional
    * Tensor, values selected at indices where condition is True
    * Defaults to None.
* b <*, T> Optional
    * Tensor, values selected at indices where condition is False
    * Defaults to None.

Outputs

* <*, T>
    *  A tensor of shape equal to the broadcasted shape.

Type Domains

* T: f32
""")
class select(Operation):
    input_spec = InputSpec(
        cond=TensorInputType(),
        a=TensorInputType(),
        b=TensorInputType()
    )

    def __init__(self, **kwargs):
        super(select, self).__init__(**kwargs)

    def type_inference(self):
        a_type = self.a.sym_type
        b_type = self.b.sym_type
        if all([a_type, b_type]):
            compatible, ret_type = builtins.is_tensor_and_is_compatible_general_shape(
                a_type, b_type
            )
            if compatible:
                return ret_type
            elif a_type == b_type:
                return a_type
            else:
                raise ValueError('Type mismatch {} vs. {}'.format(a_type, b_type))
        return a_type if a_type is not None else b_type

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.where(self.cond.val, self.a.val, self.b.val)

# rdar://58622145
@register_op(doc_str='TODO')
class while_loop(Operation):
    input_spec = InputSpec(
            # arg name with underscore prefix won't be printed.
            _cond = PyFunctionInputType(),
            _body = PyFunctionInputType(),
            loop_vars = TupleInputType(),
            )

    def __init__(self, **kwargs):
        super(while_loop, self).__init__(**kwargs)

    def build_nested_blocks(self):
        # self.loop_vars is python tuple of Vars
        # Cond block
        cond_block_name = self.name + '_cond'
        # SsaBlock takes a python tuple[Var]
        with SsaBlock(block_inputs=self.loop_vars, outer_op=self,
                name=cond_block_name) as cond_block:
            cond_func = self._cond.val
            cond_var = cond_func(*cond_block.inputs)
            cond_block.set_outputs([cond_var])
            self.blocks.append(cond_block)
        if not isinstance(cond_var, Var) or cond_var.dtype != builtins.bool:
            msg = "Cond in while_loop {} should return bool, but got {}"
            raise ValueError(msg.format(self.name, cond_var.sym_type))

        # Body block
        body_block_name = self.name + '_body'
        with SsaBlock(block_inputs=self.loop_vars, outer_op=self,
                name=body_block_name) as body_block:
            body_func = self._body.val
            exit_vars = body_func(*body_block.inputs)
            body_block.set_outputs(list(exit_vars))
            self.blocks.append(body_block)

        # Verify exit_vars has the same types as loop_vars
        for v_in, v_out in zip(self.loop_vars, exit_vars):
            if v_in.sym_type != v_out.sym_type:
                msg = "loop_vars {} changes in the body while_loop {}"
                raise ValueError(msg.format(v_in.name, self.name))

    def type_inference(self):
        return tuple(v.sym_type for v in self.blocks[1].outputs)




# identity is used for renaming and is rarely necessary. See
# `loop_invariant_elimination` pass for a rare use case.
@register_op(doc_str='TODO')
class identity(Operation):
    input_spec = InputSpec(
            x = ScalarOrTensorInputType(),
            )

    def __init__(self, **kwargs):
        super(identity, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE|SYMBOL)
    def value_inference(self):
        return self.x.sym_val
