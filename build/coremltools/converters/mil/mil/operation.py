#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.types import is_compatible_type
from coremltools.converters.mil.mil.types.symbolic import is_symbolic, any_symbolic
from . import SPACES
from .block import curr_block
from .input_type import TupleInputType, DefaultInputs
from .var import Var, InternalVar, ListVar

VALUE = 1
SYMBOL = 2
NONE = 4
ALL = 7


def _is_compatible_symbolic_array(a, b):
    """
    A helper function which check if two numpy array with symbolic value.
    For instance, a = np.array([is0, is2])
                  b = np.array([is1, 1])
    are considered compatible.
                  a = np.array([is0, 1])
                  b = np.array([is1, -1])
    are not.
    """
    if not a.shape == b.shape:
        return False
    a = a.flatten()
    b = b.flatten()
    for t, v in zip(a, b):
        if not is_symbolic(t) and not is_symbolic(v):
            if t != v:
                return False
    return True


def precondition(allow=ALL):
    """
    A helper decorator for value_inference method.
    Decorate value_inference with parameter VALUE/SYMBOL/NONE or ALL.
    For VALUE/SYMBOL/NONE use logical or ( | ) for multiple allowance.
    Note that:
        1. ALL == VALUE | SYMBOL | NONE
        2. Chosen flag (some or all VALUE/SYMBOL/NONE) must be satisfied
           by EVERY INPUTS for the precondition to be satisfied.

    The meaning for each flag is:
    VALUE: value that can be materialized during compile time
    SYMBOL: value that cannot be materialized by exist as a symbol value
    NONE: a None value

    Usage:
    @precondition(allow=VALUE|SYMBOL)
    def value_inference(self):
        '''some value_inference implementation'''
    """
    ALLOW_VALUE = allow & VALUE
    ALLOW_SYMBOL = allow & SYMBOL
    ALLOW_NONE = allow & NONE

    def process(v, has_value, has_symbol, has_none):
        """
        v: Var

        Return updated has_value, has_symbol, has_none
        """
        if any_symbolic(v.sym_val):
            return has_value, True, has_none
        elif v.val is None:
            return has_value, has_symbol, True
        return True, has_symbol, has_none

    def decorator(func):
        def wrapper(self):
            HAS_VALUE = False
            HAS_SYMBOL = False
            HAS_NONE = False
            for in_name, in_type in self._input_types.items():
                if in_type.optional:
                    # Optional inputs are not required to invoke value_inference()
                    continue

                if isinstance(in_type, TupleInputType):
                    for v in self._input_vars[in_name]:
                        HAS_VALUE, HAS_SYMBOL, HAS_NONE = process(
                            v, HAS_VALUE, HAS_SYMBOL, HAS_NONE
                        )
                else:
                    HAS_VALUE, HAS_SYMBOL, HAS_NONE = process(
                        self._input_vars[in_name], HAS_VALUE, HAS_SYMBOL, HAS_NONE
                    )

            if HAS_VALUE and not ALLOW_VALUE:
                msg = "Implementation of value_inference() for op {} doesn't support input with VALUE"
                raise NotImplementedError(msg.format(self.op_type))
            elif HAS_SYMBOL and not ALLOW_SYMBOL:
                msg = "Implementation of value_inference() for op {} doesn't support input with SYMBOL"
                raise NotImplementedError(msg.format(self.op_type))
            elif HAS_NONE and not ALLOW_NONE:
                msg = "Implementation of value_inference() for op {} doesn't support input with NONE"
                raise NotImplementedError(msg.format(self.op_type))
            else:
                return func(self)

        return wrapper

    return decorator


def is_internal_input(arg_name):
    return arg_name[0] == "_"


class mil_list(object):
    '''
    A wrapper around python list
    '''

    def __init__(self, ls=None):
        self.ls = ls if ls is not None else []
        if not isinstance(self.ls, list):
            raise TypeError("Type of 'ls' must be list in the 'mil_list' class")


class Operation(object):
    """
    Represents Operation in MIL.

    # Properties
    name (str):
        The name of the operation

    input_types (InputSpec, class attr):
        Read-only named input types from all subclasses. Input types are used
        to validate `inputs`.

    inputs [_input_vars] (dict of str --> Var):
        An Operation (subclass of Operation) only has access to input Var,
        which is already validated against `input_spec`.

    outputs [_output_vars] (list of Var):
        List of output var based on type inference. Read-only
    """

    def __init__(self, **kwargs):
        self._input_types = self.input_spec.input_types
        self.name = kwargs.get("name", None)

        self._output_vars = None
        self._input_vars = {}
        self.blocks = []
        self.enclosing_block = curr_block()

        # Initialize inputs as object attributes (all None)
        for k in self._input_types.keys():
            setattr(self, k, None)
            self._input_vars[k] = None

        self._check_expected_inputs(kwargs)

        # Set inputs from kwargs
        input_kv = {k: v for k, v in kwargs.items() \
            if k in self._input_types and v is not None}
        self._validate_and_set_inputs(input_kv)
        self._ensure_required_inputs()

    def _check_expected_inputs(self, kwargs):
        """
        Check that all kwargs inputs are one of the followings:

        - system inputs (non-attributes)
        - op inputs (self._input_types.keys())
        """
        non_attributes = [
            "name",
            "symbolic_datatype",
            "datatype",
            "symbolic_value",
            "value",
            "version",
            "before_op",
            "no_check_var_visibility",  # no_check_var_visibility==True to deviate from SSA
            "no_check_var_types",  # no_check_var_types==True to force set inputs, even if type does not match with earlier ones
        ]
        for k in kwargs.keys():
            if k not in non_attributes and k not in self._input_types:
                raise ValueError(
                    "Unknown input '{}' for op '{}'".format(
                      k, self.op_type)
                    )

    def set_inputs(self,
        no_check_var_types=False,
        type_inference=False,
        **input_kvs):
        """
        Parameters
        ----------
        - input_kvs: Dict[str, Var]
          Value cannot be None

        - type_inference: bool
          True to perform type inference and recreate output Var.
        """
        self._validate_and_set_inputs(input_kvs,
            no_check_var_types=no_check_var_types)
        if type_inference and not no_check_var_types:
            self.type_value_inference()
        self._ensure_required_inputs()

    def get_flattened_inputs(self):
        """
        Returns:
        list[Var]. Flatten all tuple inputs
        """
        flat_inputs = []
        for v in self.inputs.values():
            if isinstance(v, (list, tuple)):
                flat_inputs.extend(v)
            else:
                flat_inputs.append(v)
        return flat_inputs

    def type_value_inference(self, overwrite_output=False):
        """
        Perform type inference and auto_val computation based on new input Vars
        in kwargs. If self._output_vars is None then we generate _output_vars;
        otherwise no new Var is created, but type inference result is verified
        against existing _output_vars, if overwrite_output is False.

        If overwrite_output is True, then the type inference result overwrites the
        existing _output_vars
        """
        output_types = self.type_inference()
        if not isinstance(output_types, tuple):
            output_types = (output_types,)
        output_vals = self._auto_val(output_types)
        try:
            output_names = self.output_names()
            if not isinstance(output_names, tuple):
                output_names = (output_names,)
        except NotImplementedError as e:
            if len(output_types) > 1:
                output_names = tuple(str(i) for i, _ in enumerate(output_types))
            else:
                output_names = ("",)  # output name same as op name.

        # Combine (output_names, output_types, output_vals) to create output
        # Vars.
        if self._output_vars is None:
            self._output_vars = []
            for i, (n, sym_type, sym_val) in enumerate(
                zip(output_names, output_types, output_vals)
            ):
                name = self.name + "_" + n if n != "" else self.name
                if types.is_list(sym_type):
                    new_var = ListVar(
                        name,
                        elem_type=sym_type.T[0],
                        init_length=sym_type.T[1],
                        dynamic_length=sym_type.T[2],
                        sym_val=sym_val if (sym_val is not None and isinstance(sym_val.val, list)) else None,
                        op=self,
                        op_output_idx=i,
                    )
                else:
                    new_var = Var(name, sym_type, sym_val, op=self, op_output_idx=i)
                self._output_vars.append(new_var)
        else:
            # Check new inference result against existing self._output_vars.
            for i, (n, sym_type, sym_val) in enumerate(
                zip(output_names, output_types, output_vals)
            ):
                out_var = self._output_vars[i]
                # Check type inference
                if overwrite_output:
                    out_var._sym_type = sym_type
                elif not is_compatible_type(sym_type, out_var.sym_type):
                    msg = "Output Var {} in op {} type changes with new input Vars"
                    raise ValueError(msg.format(out_var.name, self.name))

                # Check value inference
                if overwrite_output:
                    out_var._sym_val = sym_val

                if sym_val is not None and out_var.sym_val is not None:
                    if np.any(sym_val.val != out_var.sym_val):
                        if overwrite_output:
                            out_var._sym_val = sym_val
                        else:
                            msg = 'value_inference differs for var {} in op {}'
                            if not _is_compatible_symbolic_array(sym_val.val, out_var.sym_val):
                                raise ValueError(msg.format(out_var.name, self.name))

    def _auto_val(self, output_types):
        """
        # Evaluation is two stage:
        #
        # Stage 1: Check whether the method value_inference() is implemented
        #
        # Stage 2: Check if there's an value_inference() implementation
        #          for given input types.
        #
        # Suppose input are all SYMBOL:
        # Case 1: No value_inference() implemented => fail at stage 1
        # Case 2: If value_inference() implemented, but requires all VALUE not
        #         SYMBOL => fail at stage 2
        # Case 3: If value_inference() implemented, and has no restriction on
        #         input types => Success
        #
        # If either stage fails, outputs[i].val is None.
        # Otherwise, output[i].sym_val is not None.

        output_types: tuple of builtin types

        Returns:
            output_vals: tuple of builtin type with value, or tuple of None
        """
        do_auto_val = True

        if do_auto_val:
            # Is self.value_inference implemented for corresponding input?
            try:
                vals = self.value_inference()
            except NotImplementedError as e:
                do_auto_val = False

        if not do_auto_val:
            # No auto_val possible.
            return tuple(None for _ in output_types)

        if not isinstance(vals, (tuple, list)):
            vals = (vals,)
        for val in vals:
            if val is None:
                do_auto_val = False
        if not do_auto_val:
            # No auto_val possible.
            return tuple(None for _ in output_types)

        auto_val = []
        for t, v in zip(output_types, vals):
            builtin_val = t()
            if isinstance(v, mil_list):
                builtin_val.val = v.ls
            else:
                builtin_val.val = v
            auto_val.append(builtin_val)
        return auto_val

    def value_inference(self):
        """
        Optional Python implementation of the op based on (materialized) values
        in `self.input_var`. Return a builtin value (single output) or a tuple of
        builtin values (multi-outputs) of the same length as returned by `
        type_inference`
        """
        msg = "value_inference() is not implemented by op {}"
        raise NotImplementedError(msg.format(self.op_type))

    def default_inputs(self):
        """
        Optional. Returns default values for optional inputs. The
        function is guaranteed to have access to all required inputs and
        possibly some optional inputs should the user supply them.
        They may be used to construct default values, such as
        `strides=[1]*num_spatial_dims` in conv, where
        `num_spatial_dims` may be inferred from the rank of
        required inputs
        """
        return DefaultInputs()

    def output_names(self):
        """
        Optional. If implemented, we set the output var i name as
        self.name + "/" + output_names[i]

        Returns a string (single output) or tuple of strings
        """
        msg = "output_names() is not implemented by op {}"
        raise NotImplementedError(msg.format(self.op_type))

    def type_inference(self):
        """
        Return (builtin_type, builtin_val) pair from type inference.
        builtin_val may be None if symbolic_value is not attainable at compile
        time.
        """
        raise NotImplementedError("This function must be implemented by each op")

    def build_nested_blocks(self):
        """
        Build nested blocks (for cond and while_loop and other composite
        blocks)
        """
        pass

    def _ensure_required_inputs(self):
        """
        Raise value error if required inputs aren't present
        """
        for name, input_type in self._input_types.items():
            if not input_type.optional and \
                self._input_vars[name] is None:
                msg_prefix = 'Op \"{}\" (op_type: {}) '.format(self.name,
                self.op_type)
                raise ValueError(msg_prefix + \
                    "Required input {} is missing".format(name))

    def _validate_and_set_inputs(self, input_kvs,
        no_check_var_types=False):
        """
        For each k, v in `input_kvs`, perform the followings:

        - Check k exists in `self.input_specs`
        - Check that v satisfies the correspodning `InputType`
        - Set input, possibly replacing existing input.

        Note that it does not ensure all required inputs are satisfied.
        Use _ensure_required_inputs() for that.

        Parameters
        ----------
        - input_kvs: Dict[str, Var]
          Each key in input_kvs must exist in `self.input_specs`. Its values
          must be a Var.

        - no_check_var_types: bool
          True to check var types against input_specs only, but not
          enforcing new input vars to be a subtype of existing input vars
        """
        for key in input_kvs.keys():
            if key not in self._input_types:
                raise RuntimeError(
                    "Unknown input '{}' for op '{}'".format(key, self.op_type)
                )

        def check_and_detach(v_new, v_old, op, no_check_var_types):
            # Check new var's sym_type is compatible with the
            # existing's sym_type.
            if (
                not is_compatible_type(v_new.sym_type, v_old.sym_type)
                and not no_check_var_types
            ):
                msg = "New var type {} not a subtype of " + "existing var type {}"
                raise ValueError(msg.format(v_new.sym_type, v_old.sym_type))
            v_old.remove_child_op(op, no_check_var_types)

        self.input_spec.validate_inputs(self.name, self.op_type, input_kvs)

        for name, var in input_kvs.items():
            # TODO: remove InternalVar check
            #if not isinstance(var, InternalVar):

            # Remove this operation itself from existing input
            # Var's child_ops
            existing_input_var = self._input_vars[name]
            if existing_input_var is not None:
                if isinstance(existing_input_var, (list, tuple)):
                    for v_old, v_new in zip(existing_input_var, var):
                        check_and_detach(v_new, v_old, self, no_check_var_types)
                else:
                    check_and_detach(
                        var, existing_input_var, self, no_check_var_types
                    )

            # Set var as input_var
            if isinstance(var, Var):
                var.add_child_op(self)
            elif isinstance(var, (tuple, list)):
                for v in var:
                    v.add_child_op(self)
            # ignore function inputs
            self._input_vars[name] = var
            setattr(self, name, var)


    @property
    def inputs(self):
        """
        Returns
        -------
        - inputs: Dict[str, Union[Var, Tuple[Var]]]
        """
        # Filter out InternalVar
        return {k: v for k, v in self._input_vars.items() if not
            isinstance(v, InternalVar) and v is not None}

    @property
    def outputs(self):
        return self._output_vars

    @property
    def op_type(self):
        return type(self).__name__

    def remove_from_block(self):
        """
        Remove / detach itself from the enclosing block. See Block.remove_ops
        for details.
        """
        self.enclosing_block.remove_ops([self])

    @staticmethod
    def var_to_str(v):
        if isinstance(v, (tuple, list)):
            return "(" + ", ".join(["%" + s.name for s in v]) + ")"
        elif v.op and v.op.op_type == "const":
            val = v.op.val.sym_val
            if isinstance(val, (np.generic, np.ndarray)):
                # for small tensors, serialize as string; skip large tensors.
                if val.size <= 10:
                    return str(val.tolist())
            else:
                # other types are small enough they can be serialized
                return (
                    '"' + val + '"'
                    if isinstance(val, str)
                    else str(val)
                )

        return "%" + v.name

    def indented_str(self, indent=""):
        if self.op_type == "const":
            return ""
        s = indent
        if self.outputs is not None:
            s += ", ".join([str(o) for o in self.outputs])
        s += " = " + self.op_type + "("
        s += ", ".join(
            [
                k + "=" + Operation.var_to_str(self.inputs[k])
                for k in self._input_types.keys()
                if k in self.inputs and not is_internal_input(k)
            ]
        )
        s += ', name="{}")\n'.format(self.name)
        for b in self.blocks:
            s += b.indented_str(indent=indent + SPACES)
        return s

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.indented_str(SPACES)
