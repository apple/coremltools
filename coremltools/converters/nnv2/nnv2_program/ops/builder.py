from collections import defaultdict
import logging
import copy

from ..program.program import (curr_block,
        SsaProgram, SsaFunction, Placeholder, is_internal_input)
from ..program.input_type import (_InputType, InternalStringInputType,
                                 InternalScalarOrTensorInputType,
                                 ScalarOrTensorInputType, TupleInputType,
                                 InputSpec, InternalInputType,
                                 PyFunctionInputType)
from ..program.var import InternalVar, Var


class CoremlBuilder:
    """
    Singleton builder.

    Example:

    from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
    from coremltools.converters.nnv2.nnv2_program.program import SsaProgram, SsaFunction

    prog = SsaProgram()
    func_inputs = {"x": cb.placeholder(_shape=[2,3]),
                   "y": cb.placeholder(_shape=[2,3])}
    with SsaFunction(func_inputs) as ssa_fun:
      x, y = ssa_fun.inputs['x'], ssa_fun.inputs['x']
      res_var = cb.add(x=x, y=y) # created within ssa_fun block
      ssa_fun.set_outputs([res_var])
    prog.add_function("main", ssa_fun)
    """
    name_count = defaultdict(int)

    @classmethod
    def _get_free_name(cls, name):
        new_name = name + "_" + str(cls.name_count[name])
        cls.name_count[name] += 1
        return new_name

    @classmethod
    def _maybe_set_name(cls, kwargs, op_type):
        if 'name' not in kwargs:
            kwargs['name'] = cls._get_free_name(op_type)
        return kwargs

    @classmethod
    def _add_const_immediate_value(cls, val, name, before_op):
        const_name = cls._get_free_name(name)
        logging.debug("Adding immediate value op {}".format(const_name))
        output_var = cls.const(mode="immediate_value",
                               val=val,
                               name=const_name,
                               before_op=before_op)
        return output_var

    @classmethod
    def _create_input_vars(cls, input_spec, op_name, op_cls, before_op,
                           kwargs):
        """
        1. Create Var for optional input types with default values that's not
        specified.

        2. Convert python primitive types to Var.

        Inputs:

        input_spec (InputSpec)
        op_name (str): op name.
        before_op: created all vars / const op will come right before
                   `before_op` in the block's order. None to append at the end.
        """
        update_dict = {}
        for in_name, in_type in input_spec.input_types.items():
            new_var_name = op_name + "_" + in_name
            if not in_type.optional and in_name not in kwargs:
                raise ValueError("Input {} is required for op {}.".format(\
                        in_name, op_cls.__name__))

            if in_name in kwargs and isinstance(kwargs[in_name], Var):
                # check const
                if in_type.const and kwargs[in_name].val is None:
                    msg = "Input {} of op {} ({}) must be const at compile time."
                    raise ValueError(
                        msg.format(in_name, op_name, op_cls.__name__))

            elif in_name in kwargs:
                # Provided value is not Var. Create a Var from kwargs[in_name]
                val = kwargs[in_name]
                # create Var for numpy / python primitive
                if isinstance(in_type, InternalInputType):
                    # Shove all internal inputs to InternalVar (unknown type).
                    var = InternalVar(val, name=new_var_name)
                    curr_block().add_internal_var(var)
                else:
                    if isinstance(in_type, TupleInputType):
                        var = []
                        for i, v in enumerate(val):
                            if isinstance(v, Var):
                                var.append(v)
                                continue
                            var.append(cls._add_const_immediate_value(
                                v, new_var_name+str(i), before_op))
                    elif isinstance(in_type, ScalarOrTensorInputType):
                        var = cls._add_const_immediate_value(
                            val, new_var_name, before_op)
                    else:
                        msg = "Cannot convert input {} of type {} to Var (op: {})"
                        raise ValueError(
                            msg.format(in_name,
                                       type(in_type).__name__, op_name))
                update_dict[in_name] = var

            elif in_name not in kwargs and in_type.default is not None:
                if isinstance(in_type, PyFunctionInputType):
                    msg = "Default value is not allowed for PyFunctionInputType"
                    raise ValueError(msg)
                # Create a Var from the default value.
                if is_internal_input(in_name):
                    var = InternalVar(in_type.default, name=new_var_name)
                    curr_block().add_internal_var(var)
                elif isinstance(in_type, TupleInputType):
                    var = tuple(cls._add_const_immediate_value(
                        v, new_var_name+str(i), before_op) \
                                for i, v in enumerate(in_type.default))
                else:
                    var = cls._add_const_immediate_value(
                        in_type.default, new_var_name, before_op)
                update_dict[in_name] = var

        kwargs.update(update_dict)

        return kwargs

    @classmethod
    def _add_op(cls, op_cls, **kwargs):
        """
        Add an op of type `op_cls` (e.g., convolution) to current block.
        """
        kwargs = cls._maybe_set_name(kwargs, op_cls.__name__)
        logging.info("Adding op {} of type {}".format(kwargs["name"],
                                                      op_cls.__name__))
        before_op = kwargs.get('before_op', None)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs = cls._create_input_vars(op_cls.input_spec, kwargs['name'], op_cls,
                                        before_op, kwargs)
        new_op = op_cls(**kwargs)
        curr_block().insert_op_before(new_op, before_op=before_op)
        new_op.build_nested_blocks()
        new_op.type_value_inference()
        if len(new_op.outputs) == 1:
            return new_op.outputs[0]
        return new_op.outputs

    @staticmethod
    def placeholder(shape, dtype=None):
        return Placeholder(shape, dtype)

    @staticmethod
    def TensorSpec(shape, dtype=None):
        return Placeholder(shape, dtype)

    @staticmethod
    def program(input_specs=None):
        """
        Usage:

        @cb.program(input_specs=[cb.TensorSpec(shape=(1,2))])
        def prog(a):
            return cb.add(x=a, y=2)
        """
        if input_specs is None:
            input_specs = []
        def wrapper(main_block):
            program = SsaProgram()
            num_args = main_block.__code__.co_argcount
            arg_names = list(main_block.__code__.co_varnames)[:num_args]
            if len(input_specs) != num_args:
                msg = '{} expects {} inputs: {}. Got {} input_specs.'
                raise ValueError(msg.format(main_block.__name__,
                    num_args, arg_names, len(input_specs)))
            input_spec_dict = {k: v for k, v in \
                    zip(arg_names, input_specs)}
            with SsaFunction(input_spec_dict) as func:
                input_vars = [func.inputs[a] for a in arg_names]
                outputs = main_block(*input_vars)
                if isinstance(outputs, tuple):
                    outputs = list(outputs)
                elif not isinstance(outputs, list):
                    outputs = [outputs]
                func.set_outputs(outputs)
                program.add_function('main', func)
            return program
        return wrapper
