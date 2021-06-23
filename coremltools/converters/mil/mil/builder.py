#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import defaultdict
import copy
import logging
import numbers
import numpy as np

from coremltools.converters.mil.mil.types.symbolic import any_symbolic
from .program import Program, Placeholder
from .block import curr_block, Function
from .operation import is_internal_input
from .input_type import (
    _InputType,
    InternalStringInputType,
    InternalScalarOrTensorInputType,
    ScalarOrTensorInputType,
    ListOrScalarOrTensorInputType,
    TupleInputType,
    InputSpec,
    InternalInputType,
    PyFunctionInputType,
)
from .var import InternalVar, Var

def is_python_value(val):
    return (
        isinstance(val, (np.generic, np.ndarray))
        or isinstance(val, numbers.Number)
        or isinstance(val, str)
        or isinstance(val, bool)
        or (isinstance(val, (tuple, list)) and all(is_python_value(v) for v in val))
    )


class Builder:
    """
    Singleton builder.

    Example:

    from coremltools.converters.mil.mil import Builder as mb
    from coremltools.converters.mil.mil import Program, Function

    prog = Program()
    func_inputs = {"x": mb.placeholder(_shape=[2,3]),
                   "y": mb.placeholder(_shape=[2,3])}
    with Function(func_inputs) as ssa_fun:
      x, y = ssa_fun.inputs['x'], ssa_fun.inputs['x']
      res_var = mb.add(x=x, y=y) # created within ssa_fun block
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
        if "name" not in kwargs:
            kwargs["name"] = cls._get_free_name(op_type)
        return kwargs

    @classmethod
    def _add_const(cls, val, name, before_op):
        if not is_python_value(val):
            raise ValueError("Cannot add const {}".format(val))
        if any_symbolic(val):
            msg = (
                "Python native vals (list, tuple), np.array that are"
                + "operation inputs cannot have symbolic values. Consider feeding"
                + "symbolic shape in through placeholder and use mb.shape() "
                + "operator. Input {}: {}"
            )
            raise ValueError(msg.format(name, val))
        const_name = cls._get_free_name(name)
        logging.debug("Adding const op '{}'".format(const_name))
        output_var = cls.const(val=val, name=const_name,
            before_op=before_op)
        return output_var


    @classmethod
    def _create_vars(cls, input_spec, op_name, before_op,
        candidate_kv):
        """
        For each key K in `candidate_kv`, create a Var if the
        followings are satisfied:

        - K exists in input_spec and is not an InternalInputType
        - candidate_kv[K] is not already a Var

        Inputs
        ------
        - candidate_kv: Dict[str, Any]
          Key-values may be inputs to an op (whose inputs is defined by
          input_spec)

        Returns
        -------
        - var_kv: Dict[str, Var]
          For the K satisfying the above, var_kv[K] is the newly
          created Var
        """
        update_dict = {}
        for k, val in candidate_kv.items():
            if isinstance(val, Var):
                continue # already a Var

            if k not in input_spec.input_types:
                continue # k is not an op input

            in_type = input_spec.input_types[k]
            if isinstance(in_type, InternalInputType):
                new_var_name = op_name + "_" + k
                var = InternalVar(val, name=new_var_name)
                curr_block().add_internal_var(var)
                update_dict[k] = var
                continue # Not a regular Var

            new_var_name = op_name + "_" + k
            if isinstance(in_type, TupleInputType):
                var = []
                for i, v in enumerate(val):
                    if isinstance(v, Var):
                        var.append(v)
                        continue
                    var.append(
                        cls._add_const(v, new_var_name + str(i),
                          before_op)
                    )
                update_dict[k] = var
                continue

            if isinstance(in_type, (ScalarOrTensorInputType,
              ListOrScalarOrTensorInputType)):
                var = cls._add_const(val, new_var_name, before_op)
                update_dict[k] = var

        return update_dict

    @classmethod
    def _add_op(cls, op_cls, **kwargs):
        """
        Add an op of type `op_cls` (e.g., convolution) to current block.
        """
        kwargs = cls._maybe_set_name(kwargs, op_cls.__name__)
        logging.info(
            "Adding op '{}' of type {}".format(kwargs["name"], op_cls.__name__)
        )
        before_op = kwargs.get("before_op", None)
        # Shallow copy list inputs to ensure op inputs are immutable
        kwargs = {k: v if not isinstance(v, (list, tuple)) else v[:] for k, v in kwargs.items() if v is not None}
        kwargs.update(cls._create_vars(
            input_spec=op_cls.input_spec,
            op_name=kwargs["name"], before_op=before_op,
            candidate_kv=kwargs))
        new_op = op_cls(**kwargs)

        # Initialize optional input Vars if it wasn't in kwargs
        default_inputs = new_op.default_inputs()
        # Shallow copy list inputs to ensure op inputs are immutable
        missing_optional_vals = {k: v if not isinstance(v, (list, tuple)) else v[:] for k, v in default_inputs.items()
            if k not in kwargs and v is not None}
        missing_optional_vars = cls._create_vars(
            input_spec=op_cls.input_spec,
            op_name=kwargs["name"], before_op=before_op,
            candidate_kv=missing_optional_vals)
        new_op.set_inputs(type_inference=False,
            **missing_optional_vars)

        curr_block()._insert_op_before(new_op, before_op=before_op)
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

        @mb.program(input_specs=[mb.TensorSpec(shape=(1,2))])
        def prog(a):
            return mb.add(x=a, y=2)
        """
        if input_specs is None:
            input_specs = []

        def wrapper(main_block):
            program = Program()
            num_args = main_block.__code__.co_argcount
            arg_names = list(main_block.__code__.co_varnames)[:num_args]
            if len(input_specs) != num_args:
                msg = "{} expects {} inputs: {}. Got {} input_specs."
                raise ValueError(
                    msg.format(
                        main_block.__name__, num_args, arg_names, len(input_specs)
                    )
                )
            input_spec_dict = {k: v for k, v in zip(arg_names, input_specs)}
            with Function(input_spec_dict) as func:
                input_vars = [func.inputs[a] for a in arg_names]
                outputs = main_block(*input_vars)
                if isinstance(outputs, tuple):
                    outputs = list(outputs)
                elif not isinstance(outputs, list):
                    outputs = [outputs]
                func.set_outputs(outputs)
                program.add_function("main", func)
            return program

        return wrapper


"""importing ops triggers installation of all ops into Builder"""
from .ops import defs as _ops
