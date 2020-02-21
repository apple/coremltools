from collections import defaultdict
import logging
import copy

from ..program.program import (curr_block, Placeholder, is_internal_input)
from ..program.input_type import (_InputType, InternalStringInputType,
                                 InternalScalarOrTensorInputType,
                                 ScalarOrTensorInputType, TupleInputType,
                                 InputSpec, InternalInputType,
                                 PyFunctionInputType,
                                 PyTupleInputType)
from ..program.var import InternalVar, Var


class CoremlBuilder:
    """
    Singleton builder.

    Example:

    from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
    from coremltools.converters.nnv2.nnv2_program.program import SsaProgram, SsaFunction, SsaValue

    prog = SsaProgram()
    prog.add_parameters({"w1": SsaValue(...)})
    func_inputs = {"x": cm.placeholder(_shape=[2,3]),
                   "y": cm.placeholder(_shape=[2,3])}
    with SsaFunction(func_inputs) as ssa_fun:
      a, b = ssa_func.inputs['x'], ssa_func.inputs['x']
      res_var = cb.add(a, b) # created within ssa_fun block
      ssa_func.set_outputs([res_var])
      prog.add_function("main", ssa_func)
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
        const_input_types = InputSpec(
            mode=InternalStringInputType(const=True,
                                         default="immediate_value"),
            val=InternalScalarOrTensorInputType(const=True),
        )
        logging.debug("Adding immediate value op {}".format(const_name))
        output_var = cls.const(mode="immediate_value",
                               val=val,
                               name=const_name,
                               input_types=const_input_types,
                               before_op=before_op)
        return output_var

    @staticmethod
    def _collect_input_types(op_cls):
        """
        Return input_name (str) --> input type (_InputType) mapping.

        If Op1 subclasses Op2, which subclass Operation,
        CoremlBuilder._collect_input_types(Op1) return input types from both
        Op1 and Op2
        """
        input_types = InputSpec()
        for x in op_cls.mro()[:-2]:  # excluding Operation and object
            op_input_types = getattr(x, "input_types", None)
            if op_input_types is not None:
                input_types.update(copy.deepcopy(op_input_types))
        input_types = input_types.input_types

        for k, v in input_types.items():
            if not isinstance(v, _InputType):
                raise RuntimeError(
                    "Input {} should be an _InputType".format(k))
        return input_types

    @classmethod
    def _create_input_vars(cls, input_types, op_name, op_cls, before_op,
                           kwargs):
        """
        1. Create Var for optional input types with default values that's not
        specified.

        2. Convert python primitive types to Var.

        Inputs:

        input_types (dict of str --> _InputType)
        op_name (str): op name.
        before_op: created all vars / const op will come right before
                   `before_op` in the block's order. None to append at the end.
        """
        update_dict = {}
        for in_name, in_type in input_types.items():
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
                        var = cls._make_tuple(elems=val)
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
                if isinstance(in_type, (PyFunctionInputType, TupleInputType)):
                    msg = "Default value is not allowed for PyFunctionInputType " +\
                        "and TupleInputType"
                    raise ValueError(msg)
                # Create a Var from the default value.
                if is_internal_input(in_name):
                    var = InternalVar(in_type.default, name=new_var_name)
                    curr_block().add_internal_var(var)
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
        input_types = CoremlBuilder._collect_input_types(op_cls)
        before_op = kwargs.get('before_op', None)
        kwargs = cls._create_input_vars(input_types, kwargs['name'], op_cls,
                                        before_op, kwargs)
        kwargs['input_types'] = input_types
        new_op = op_cls(**kwargs)
        curr_block().insert_op_before(new_op, before_op=before_op)
        if len(new_op.outputs) == 1:
            return new_op.outputs[0]
        return new_op.outputs

    @staticmethod
    def placeholder(shape, dtype=None):
        return Placeholder(shape, dtype)
