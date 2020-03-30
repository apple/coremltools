# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import copy
import logging
import numpy as np
import sympy as sm
import six

from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2.builtin_types.symbolic import any_symbolic
from .var import Var, InternalVar
from .input_type import TupleInputType

# BLOCK_STACK[-1] is the current block
BLOCK_STACK = []

SPACES = "  "

def curr_block():
    if len(BLOCK_STACK) == 0:
        raise ValueError("Must call CoremlBuilder inside an SsaFunction" +
                         " or SsaBlock")
    return BLOCK_STACK[-1]


def is_internal_input(arg_name):
    return arg_name[0] == "_"

VALUE = 1
SYMBOL = 2
NONE = 4
ALL = 7
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
        if v.val is None:
            return has_value, has_symbol, True
        elif any_symbolic(v.val):
            return has_value, True, has_none
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
                        HAS_VALUE, HAS_SYMBOL, HAS_NONE = process(v,
                                HAS_VALUE, HAS_SYMBOL, HAS_NONE)
                else:
                    HAS_VALUE, HAS_SYMBOL, HAS_NONE = process(
                            self._input_vars[in_name],
                            HAS_VALUE, HAS_SYMBOL, HAS_NONE)

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

class Operation(object):
    """
    Represents Operation in NNv2.Program.

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
        self.name = kwargs.get('name', None)

        self._output_vars = None
        self._input_vars = {}
        self.blocks = []
        self.enclosing_block = curr_block()
        self._validate_and_set_inputs(**kwargs)

    def set_inputs(self, **kwargs):
        self._validate_and_set_inputs(**kwargs)
        self.type_value_inference()

    def type_value_inference(self):
        """
        Perform type inference and auto_val computation based on new input Vars
        in kwargs. If self._output_vars is None then we generate _output_vars;
        otherwise no new Var is created, but type inference result is verified
        against existing _output_vars.
        """
        output_types = self.type_inference()
        if not isinstance(output_types, tuple):
            output_types = (output_types, )
        output_vals = self._auto_val(output_types)
        try:
            output_names = self.output_names()
            if not isinstance(output_names, tuple):
                output_names = (output_names, )
        except NotImplementedError as e:
            if len(output_types) > 1:
                output_names = tuple(
                    str(i) for i, _ in enumerate(output_types))
            else:
                output_names = ("", )  # output name same as op name.

        # Combine (output_names, output_types, output_vals) to create output
        # Vars.
        if self._output_vars is None:
            self._output_vars = []
            for i, (n, sym_type, sym_val) in enumerate(
                    zip(output_names, output_types, output_vals)):
                name = self.name + ":" + n if n != '' else self.name
                new_var = Var(name,
                              sym_type,
                              sym_val,
                              op=self,
                              op_output_idx=i)
                self._output_vars.append(new_var)
        else:
            # Check new inference result against existing self._output_vars.
            for i, (n, sym_type, sym_val) in enumerate(
                    zip(output_names, output_types, output_vals)):
                out_var = self._output_vars[i]
                # Check type inference
                if sym_type != out_var.sym_type:
                    msg = "Output Var {} in op {} type changes with new input Vars"
                    raise ValueError(msg.format(out_var.name, self.name))

                # Check value inference
                if sym_val is not None and out_var.sym_val is None:
                    msg = 'value_inference failed with new inputs for op {}'
                    raise ValueError(msg.format(self.name))
                if sym_val is not None and out_var.sym_val is not None:
                    if np.any(sym_val.val != out_var.sym_val):
                        msg = 'value_inference differs for var {} in op {}'
                        raise ValueError(msg.format(output_var.name, self.name))

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
            vals = (vals, )
        for val in vals:
            if val is None:
                do_auto_val = False
        if not do_auto_val:
            # No auto_val possible.
            return tuple(None for _ in output_types)

        auto_val = []
        for t, v in zip(output_types, vals):
            builtin_val = t()
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
        raise NotImplementedError(
            "This function must be implemented by each op")

    def build_nested_blocks(self):
        """
        Build nested blocks (for cond and while_loop and other composite
        blocks)
        """
        pass

    def _validate_and_set_inputs(self, **kwargs):
        non_attributes = [
            "name",
            "symbolic_datatype",
            "datatype",
            "symbolic_value",
            "value",
            "version",
            "before_op",
            "no_check",  # no_check==True to deviate from SSA
        ]
        op_inputs = list(self._input_types.keys())
        legal_args = op_inputs + non_attributes
        no_check = kwargs.get('no_check', False)

        for key in kwargs.keys():
            if key not in legal_args:
                raise RuntimeError("Unknown input {} for op {}".format(
                    key, self.op_type))

        parsed_inputs = self.input_spec.parse_inputs(kwargs)
        for (name, var) in parsed_inputs:
            setattr(self, name, var)
            if var is not None and not isinstance(var, InternalVar):
                # Remove this operation itself from existing input Var's child_ops
                existing_input_var = self._input_vars.get(name, None)
                if existing_input_var is not None:
                    if isinstance(existing_input_var, (list, tuple)):
                        for v_old, v_new in zip(existing_input_var, var):
                            if v_old.sym_type != v_new.sym_type:
                                msg = 'New var type {} != existing var type {}'
                                raise ValueError(msg.format(v_new.sym_type,
                                    v_old.sym_type))
                            v_old.remove_child_op(self, no_check)
                    else:
                        # Check new var's sym_type is compatible with the
                        # existing's sym_type.
                        if existing_input_var.sym_type != var.sym_type:
                            msg = 'New var type {} != existing var type {}'
                            raise ValueError(msg.format(var.sym_type,
                                existing_input_var.sym_type))
                        existing_input_var.remove_child_op(self, no_check)

                # Set var as input_var
                if isinstance(var, Var):
                    var.add_child_op(self)
                elif isinstance(var, (tuple,list)):
                    for v in var:
                        v.add_child_op(self)
                # ignore function inputs
                self._input_vars[name] = var

    @property
    def inputs(self):
        return self._input_vars

    @property
    def outputs(self):
        return self._output_vars

    @property
    def op_type(self):
        return type(self).__name__

    def remove_from_block(self):
        """

        Remove / detach itself from the enclosing block. See SsaBlock.remove_ops
        for details.
        """
        self.enclosing_block.remove_ops([self])

    @staticmethod
    def var_to_str(v):
        if isinstance(v, (tuple, list)):
            return "(" + ", ".join(["%" + s.name for s in v]) + ")"
        else:
            return "%" + v.name

    def indented_str(self, indent=""):
        s = indent + ", ".join([str(o) for o in self.outputs])
        s += " = " + self.op_type + "("
        if self.op_type == 'const':
            if self.mode.val == 'immediate_value':
                if isinstance(self.val.sym_val, (np.generic, np.ndarray)):
                    val_str = str(self.val.sym_val.tolist())
                else:
                    val_str = "\"" + self.val.sym_val + "\"" if \
                        isinstance(self.val.sym_val, six.string_types) else \
                                str(self.val.sym_val)
                s += "val=" + val_str
            else:
                s += "val=(file_value)"
        else:
            s += ", ".join([k + "=" + Operation.var_to_str(self.inputs[k]) \
                    for k in self._input_types.keys() if \
                    k in self.inputs and not is_internal_input(k)])
        s += ", name=\"{}\")\n".format(self.name)
        for b in self.blocks:
            s += b.indented_str(indent=indent + SPACES)
        return s

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.indented_str(SPACES)


class InvalidBlockStateError(Exception):
    pass

class SsaBlock(object):
    __slots__ = [
        "name", "_block_inputs", "_outputs", "operations", "_internal_vars",
        "outer_op", "_shadowed_vars"
    ]

    counter = 0

    @classmethod
    def _get_new_name(cls):
        curr_val = cls.counter
        cls.counter += 1
        return "block" + str(curr_val)

    def __init__(self, block_inputs=None, outer_op=None, name=None):
        """
        Inputs:

        block_inputs: python tuple[Var].
            block_inputs is None except when the block represents loop.

        outer_op: Operation
            The enclosing op. None iff this SsaBlock is an SsaFunction.

        function_inputs: tuple[Var]
            function_inputs are always visible for this block and all blocks
            nested within. If function_inputs is None, get it from
            `outer_op.block`
        """
        self.name = name
        if self.name is None:
            self.name = SsaBlock._get_new_name()

        # list[Operation]. Topologically sorted.
        self.operations = []

        self.set_inputs(block_inputs)

        # list[Var]. This is converted to str when generating NNv2 proto.
        self._outputs = None

        # If we create const, whose inputs (mode, val) cannot be const
        # (infinite recursion). They must be considered as always visible.
        self._internal_vars = set()

        self.outer_op = outer_op
        if self.outer_op is None and not isinstance(self, SsaFunction):
            msg = "SsaBlock {} is not SsaFunction and thus outer_op cannot be None"
            raise ValueError(msg.format(self.name))

    def validate(self):
        """
        Basic validation to protect against some invalid state.
        """
        for op in self.operations:
            for b in op.blocks:
                b.validate()
            if op.outputs is None:
                raise InvalidBlockStateError()

    def set_inputs(self, block_inputs):
        """
        block_inputs must be a var in enclosing block, which will be shadowed
        within the block. Example:

        #    main(%a: (1, 2, fp32),
        #         %b: (1, 2, fp32),
        #         %c: (1, 2, fp32)) {
        #      block0() {
        #        %const1: (1, fp32) = const(...)
        #        %loop:0: (1, 2, fp32), %loop:1: (1, 2, fp32) = \
        #        while_loop(loop_vars=(%a, %b))
        #          loop_cond(%a.x, %b.x) {
        #            %blah: (bool) = some_op(x=%a.x, y=%b.x)
        #            %cond_var: (bool) = some_op2(x=%a.x, y=%blah)
        #          } -> (%cond_var)
        #          loop_body(%a.x, %b.x) {
        #            %add_0: (1, 2, fp32) = add(x=%a.x, y=%b.x)
        #          } -> (%add_0, %b.x)
        #        %linear: (1, fp32) = linear(...)
        #      } -> (%loop:0, %loop:1)
        #    }

        loop_cond block can (%a, %b) which is visible to the block. %a, %b are
        shadowed and becomes %a.x, %b.x. loop_cond block, however, cannot take
        %linear as input.
        """
        self.validate()
        # block_inputs: list[Var]
        if block_inputs is not None:
            self._block_inputs = tuple(copy.deepcopy(v) for v in block_inputs)
            # Keep track the vars we shadow
            self._shadowed_vars = block_inputs
            for v in self._block_inputs:
                v._op = None
                v.op_output_idx = None
                v._child_ops = set()
                v.name = v.name + ".x"
                v._sym_val = None
        else:
            self._block_inputs = tuple()
            self._shadowed_vars = tuple()

    def remove_inputs(self, curr_input_vars):
        """
        curr_input_vars: list[Var], whose elements must be in
        self._block_inputs.
        """
        self.validate()
        remove_idx = [self._block_inputs.index(v) for v in curr_input_vars]
        self._block_inputs = [v for i, v in enumerate(self._block_inputs) \
                if i not in remove_idx]
        self._shadowed_vars = [v for i, v in enumerate(self._shadowed_vars) \
                if i not in remove_idx]

    def find_ops(self, prefix=None, op_type=None):
        """
        Return list of ops with name matching `prefix` if specified and
        op_type, if specified. At least one of {prefix, op_type} must be specified.

        prefix: str

        Return list[Operation]. Empty list if no op satisfies.
        """
        if prefix is None and op_type is None:
            raise ValueError('Must specify one of {prefix, op_type}')
        found_ops = []
        for op in self.operations:
            prefix_match = prefix is None or \
                    op.name[:len(prefix)] == prefix
            op_type_match = op_type is None or op.op_type == op_type
            if prefix_match and op_type_match:
                found_ops.append(op)
            for b in op.blocks:
                found_ops.extend(b.find_ops(prefix=prefix, op_type=op_type))
        return found_ops

    def add_internal_var(self, internal_var):
        if not isinstance(internal_var, InternalVar):
            raise ValueError(
                "Only InternalVar can be manually added to SsaBlock.")
        self._internal_vars.add(internal_var)

    @property
    def inputs(self):
        return self._block_inputs

    @property
    def outputs(self):
        return self._outputs

    def set_outputs(self, outputs):
        """
        outputs: list[Var]
        """
        if not isinstance(outputs, list):
            raise ValueError("Outputs must be list of Vars")

        # Need to copy, or block's output would be completely tied to a var's
        # output and we cannot replace a block output with another var's
        # output.
        self._outputs = copy.copy(outputs)

    def __enter__(self):
        global BLOCK_STACK
        BLOCK_STACK.append(self)
        return self

    def __exit__(self, type, value, traceback):
        global BLOCK_STACK
        BLOCK_STACK = BLOCK_STACK[:-1]

    def _visible_vars_in_block(self, target_op=None, inclusive=True):
        """
        Returns:

        index (int) of target_op in self.operations if target_op not None,
        undefined otherwise.

        Raises:

        ValueError if target_op not None and not found in self.operations.

        visible_vars: set[Var]
            Vars returned by ops in the block (self) visible (and equal to
            if inclusive==True) target_op.  If target_op is not found or None,
            include all vars output by self.operations. Examples:

        #    main(%a: (1, 2, fp32),
        #         %b: (1, 2, fp32),
        #         %c: (1, 2, fp32)) {
        #      block0() {
        #        %const1: (1, fp32) = const(...)
        #        %loop:0: (1, 2, fp32), %loop:1: (1, 2, fp32) = \
        #        while_loop(loop_vars=(%a, %b))
        #          loop_cond(%a.x, %b.x) {
        #            %blah: (bool) = some_op(x=%a.x, y=%b.x)
        #            %cond_var: (bool) = some_op2(x=%a.x, y=%blah)
        #          } -> (%cond_var)
        #          loop_body(%a.x, %b.x) {
        #            %add_0: (1, 2, fp32) = add(x=%a.x, y=%b.x)
        #          } -> (%add_0, %b.x)
        #        %linear: (1, fp32) = linear(...)
        #      } -> (%loop:0, %loop:1)
        #    }
        #

        Let V0 and V1 be the set of internal_vars of block0 and loop_cond
        block that supplies const vals (for const).

        Ex1: self = block0, target_op = linear.
        idx = 2
        visible_vars = {%const1, %loop:0, %loop:1, %linear, V0}

        Ex2: self = loop_cond, target_op = None.
        idx = undefined
        visible_vars = {%a.x, %b.x, %blah, %cond_var, V1}

        Ex3: self = loop_cond, target_op = some_op.
        idx = 0
        visible_vars = {%a.x, %b.x, %blah, V1}

        Ex4: self = loop_cond, target_op = linear.
        raises ValueError (linear not found in loop_cond block)
        """
        visible_vars = set(self._internal_vars)
        visible_vars.update(self.inputs)
        idx = -1
        # find the location of target_op
        for i, op in enumerate(self.operations):
            if op == target_op:
                if inclusive:
                    visible_vars.update(op.outputs)
                return i, visible_vars
            visible_vars.update(op.outputs)
        if target_op is not None:
            msg = 'Op {} not found in {}: {}'
            raise ValueError(msg.format(target_op.name, self.name, self))
        return idx, visible_vars

    def _visible_vars_from_enclosing_block(self):
        """
        Returns:

        visible_vars: Vars from lexical scopes visible at the beginning of the
        block, up to but not including outputs from before_op. Given program:

        #    main(%a: (1, 2, fp32),
        #         %b: (1, 2, fp32),
        #         %c: (1, 2, fp32)) {
        #      block0() {
        #        %const1: (1, fp32) = const(...)
        #        %loop:0: (1, 2, fp32), %loop:1: (1, 2, fp32) = \
        #        while_loop(loop_vars=(%a, %b))
        #          loop_cond(%a.x, %b.x) {
        #            %blah: (bool) = some_op(x=%a.x, y=%b.x)
        #            %cond_var: (bool) = some_op2(x=%a.x, y=%blah)
        #          } -> (%cond_var)
        #          loop_body(%a.x, %b.x) {
        #            %add_0: (1, 2, fp32) = add(x=%a.x, y=%b.x)
        #          } -> (%add_0, %b.x)
        #        %const2: (1, fp32) = const(...)
        #      } -> (%loop:0, %loop:1)
        #    }

        Let V0 be the set of internal_vars of block0 block that supplies const
        vals (for const).

        Ex1: self = block0
             visible_vars = {%a, %b, %c} (function input)

        Ex2: self = loop_cond.
            visible_vars = {%c} (%a, %b are shadowed by %a.x, %b.x)
                           + {%const1, V0}
                           (%const2 is not part of the set)
        """
        visible_vars = set()

        # function inputs are considered external to the block.
        if isinstance(self, SsaFunction):
            # block in function only has function_inputs as from enclosing
            # block (Ex1 above).
            visible_vars.update(self.function_inputs)
            return visible_vars

        if self.outer_op is not None:
            enclosing_block = self.outer_op.enclosing_block
            vars_at_start = enclosing_block._visible_vars_from_enclosing_block()
            visible_vars.update(vars_at_start)
            _, visible_vars_in_block = enclosing_block._visible_vars_in_block(
                    self.outer_op, inclusive=False)
            visible_vars.update(visible_vars_in_block)

        if len(self._shadowed_vars) > 0:
            # Shadow vars.
            for v in self._shadowed_vars:
                if v not in visible_vars:
                    msg = 'Failed to shadow var {} (not in enclosing ' + \
                            'scope of block {})'
                    raise ValueError(msg.format(v, self))
                visible_vars.remove(v)

        return visible_vars

    def _insert_op_before(self, new_op, before_op=None):
        """
        A private API used by builder. Please use `builder.YOUR_OP(...,before_op)`.

        new_op's outputs are not used (not input to any other op) after
        this call. All inputs to new_op must visible at or before
        before_op (i.e., new_op must be added in topologically sorted
        order). Note that this is more restrictive than NNv2, whose Block
        supports lexical scoping and can thus op can reference Var in enclosing
        scopes. new_op.name must be unique in the block.

        before_op=None to append new_op at the end of self.operations.

        Given:   %2 = op0(%1, %1)
                 %4 = op2(%1)
                 %6 = op3(%4, %4)
        Execute: insert_op_before(op2, op1)
        Result:  %2 = op0(%1, %1)
                 %3 = op1(%1, %2)
                 %4 = op2(%1)
                 %6 = op3(%4, %4)

        Comment: We assume op1 has been constructed outside the block with
        %1, %2 as inputs. Typically it's builder's job to create an op and
        insert into the current block.

        Comment: insert_op_before(op0, op1) would error as %2 (an input to op1)
        is not visible before op0.
        """
        self.validate()
        visible_vars = self._visible_vars_from_enclosing_block()
        if before_op is not None:
            idx, visible_vars_in_block = self._visible_vars_in_block(before_op,
                    inclusive=True)
            visible_vars.update(visible_vars_in_block)
        else:
            _, visible_vars_in_block = self._visible_vars_in_block()
            visible_vars.update(visible_vars_in_block)

        # check inputs are visible
        for k, v in new_op.inputs.items():
            if not isinstance(v, (Var, tuple)):
                continue
            if isinstance(v, Var):
                vs = [v]
            else:
                vs = v
            for s in vs:
                if s not in visible_vars:
                    before_op_name = before_op.name \
                            if before_op is not None else "None"
                    msg = "Op {} input {}={} is not in scope of {} before {}"
                    raise ValueError(
                        msg.format(new_op.name, k, s.name, self.name,
                                   before_op_name))

        # add new_op
        if before_op is None:
            self.operations.append(new_op)
        else:
            self.operations.insert(idx, new_op)

    def _replace_var(self, old_var, new_var, start=0, no_check=False):
        """Helper function for replace_var_after_op"""
        num_ops_affected = 0
        for op in self.operations[start:]:
            new_inputs = {'no_check': no_check}
            affected = False
            for k, v in op.inputs.items():
                if isinstance(v, (list, tuple)) and old_var in v:
                    new_inputs[k] = tuple(new_var \
                            if vv == old_var else vv for vv in v)
                    affected = True
                elif v == old_var:
                    new_inputs[k] = new_var
                    affected = True
                else:
                    new_inputs[k] = v
            if affected:
                num_ops_affected += 1
                op.set_inputs(**new_inputs)

            # Replace recursively.
            for b in op.blocks:
                num_ops_affected += b._replace_var(old_var, new_var)

        # If old_var is block's output, replace as well.
        if old_var in self._outputs:
            idx = self._outputs.index(old_var)
            self._outputs[idx] = new_var
        return num_ops_affected

    def replace_var_after_op(self, anchor_op, old_var, new_var,
            no_check=False):
        """
        Replace all uses of `old_var` with `new_var` after `anchor_op`. If
        `anchor_op` is None, replace all occurences of `old_var` in the block.

        no_check: True to disable the check ensuring new_var is visible
        (visibility requirement depends on anchor_op).

        old_var, new_var must meet the following conditions:

        - old_var, new_var both existing within the block. This implies that
          the op generating new_var must be inserted prior to this
          replacement.

        - Affected ops (i.e., Operation after anchor_op that take old_var as
          input) must generate the same type inference results as before.

        - new_var must be visible at or before anchor_op in the order of
          self.operations.

        Given:   %2 = op0(%1, %1)
                 %3 = op1(%1, %2)
                 %4 = op2(%1)
                 %6 = op3(%4, %4)
        Execute: replace_var_after_op(op2, %4, %3)
        Result:  %2 = op0(%1, %1)
                 %3 = op1(%1, %2)
                 %4 = op2(%1)
                 %6 = op3(%3, %3)     # type inference check against %6


        Comment: Execute: replace_var_after_op(op1, %4, %3) would lead to
        identical results, as op2 does not take %4 as input.

        Comment: replace_var_after_op(op0, %4, %3) cause error as %3 is
        after op0

        Comment: To avoid clutter, we drop the names of arguments and return
        Var in the illustration above.
        """
        self.validate()
        # Get visible vars from enclosing block
        visible_vars = self._visible_vars_from_enclosing_block()
        if anchor_op is not None:
            # Get visible vars from the current block
            idx, block_vars = self._visible_vars_in_block(anchor_op,
                    inclusive=True)
            visible_vars.update(block_vars)

            # start from the next op, excluding `anchor_op`
            start = idx + 1
        else:
            visible_vars.update(self._block_inputs)
            visible_vars.update(self._internal_vars)
            # Perform replacement from beginning
            start = 0

        if not no_check and new_var not in visible_vars:
            msg = "new_var {} is not visible in block {} at or before "\
                    + "anchor_op {}"
            anchor_op_name = "None" if anchor_op is None else anchor_op.name
            raise ValueError(
                msg.format(new_var.name, self.name, anchor_op_name))

        num_ops_affected = self._replace_var(old_var, new_var,
                start=start, no_check=no_check)

        logging.debug(
            "Num ops affected in replacing var: {}".format(num_ops_affected))

    def remove_ops(self, existing_ops):
        """
        Remove `existing_ops` (list[Operation]) that must be pre-existing in
        the block. Error if any other op in the block uses output Vars of
        `existing_ops`
        """
        self.validate()
        idxs = [-1] * len(existing_ops)
        existing_ops_set = set(existing_ops)
        for i, op in enumerate(self.operations):
            if op in existing_ops_set:
                idxs[existing_ops.index(op)] = i
        if -1 in idxs:
            not_found = []
            for i, op in zip(idxs, existing_ops):
                if i == -1:
                    not_found.append(op.name)
            raise ValueError("Ops {} not found in block {}".format(
                not_found, self.name))

        # Remove ops in reverse topological order
        pairs = list(zip(idxs, existing_ops))
        pairs.sort(key=lambda x: x[0], reverse=True)

        for idx, op in pairs:
            for i, v in enumerate(op.outputs):
                # Check that no ops depend on op's outputs
                if len(v.child_ops) > 0:
                    child_op_names = [s.name for s in v.child_ops]
                    msg = "Cannot delete op {} with active output {}: {} " +\
                            "used by ops {}"
                    raise ValueError(
                        msg.format(op.name, i, v.name, child_op_names))
                # Check that the output Var isn't block's output
                if v in self._outputs:
                    msg = "cannot delete op {} with output {}: {} " +\
                            "that's block {}'s output"
                    raise ValueError(msg.format(op.name, i, v.name, self.name))

            # remove the op (in reverse topological order)
            self.operations.pop(idx)
            op.enclosing_block = None

            # an binary op may take the same input, e.g. add(x0, x0)
            # we should only call remove add from x0 once.
            unique_inputs = set()
            for v in op.inputs.values():
                if isinstance(v, (tuple,list)):
                    for vv in v:
                        unique_inputs.add(vv)
                else:
                    unique_inputs.add(v)
            for v in unique_inputs:
                v.remove_child_op(op)

    def indented_str(self, indent=None):
        if indent is None:
            indent = ""
        s = indent + self.name + "(" + ", ".join(
            ["%" + var.name for var in self._block_inputs])
        s += ") {\n"
        for op in self.operations:
            s += op.indented_str(indent + SPACES * 1)
        s += indent + "} -> ("
        s += ", ".join(["%" + v.name for v in self._outputs])
        s += ")\n"
        return s

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.indented_str()


class SsaFunction(SsaBlock):
    """
    """
    def __init__(self, inputs):
        """
        inputs: str -> placeholder
        """
        self.placeholder_inputs = inputs
        # str -> Var
        self._input_dict = {}
        for k, v in self.placeholder_inputs.items():
            v.set_name(k)  # set to user input name
            self._input_dict[k] = v.outputs[0]
        self.function_inputs = tuple(self._input_dict.values())
        super(SsaFunction, self).__init__()

    # Override SsaBlock's input
    @property
    def inputs(self):
        return self._input_dict

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.to_str("function")

    def to_str(self, func_name='function'):
        if len(self._input_dict) == 0:
            s = func_name + "() {"
        else:
            inputs = [(in_name, ph)
                      for in_name, ph in self._input_dict.items()]
            s = func_name + "(" + str(inputs[0][1])
            for in_name, ph in inputs[1:]:
                s += ",\n" + " " * (len(func_name) + 1) + str(ph)
            s += ") {\n"
            s += self.indented_str(SPACES)
            s += "}\n"
        return s


class SsaProgram(object):
    def __init__(self):
        self.functions = {}
        self.parameters = {}

    def add_function(self, name, ssa_func):
        if not isinstance(ssa_func, SsaFunction):
            raise ValueError("Only SsaFunction can be added to SsaProgram.")
        self.functions[name] = ssa_func

    def add_parameters(self, name, ssa_val):
        raise NotImplementedError()

    def find_ops(self, prefix=None, op_type=None, exactly_one=False):
        """
        Return list of ops with name matching `prefix` if specified, and
        op_type, if specified. At least one of {prefix, op_type} must be
        specified.

        If `exactly_one` == True, raise ValueError if we find <1 or >1 ops satisfying
        the criteria.

        prefix: str

        Return list[Operation]. Empty list if no op satisfies.
        """
        found_ops = []
        for f_name, f in self.functions.items():
            found_ops.extend(f.find_ops(prefix=prefix, op_type=op_type))
        if exactly_one and len(found_ops) != 1:
            msg = 'Found matching ops not exactly one. Found ops: {}'
            raise ValueError(msg.format(found_ops))
        return found_ops

    def __getitem__(self, func_name):
        if func_name not in self.functions:
            msg = 'Function {} not found in among functions {}.'
            raise KeyError(msg.format(func_name, self.functions.keys()))
        return self.functions[func_name]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = ""
        for f_name, f in self.functions.items():
            s += f.to_str(f_name)
        return s


class Placeholder(object):
    counter = 0

    def __init__(self, sym_shape, dtype=None, name=None):
        """
        sym_shape: () or [] for scalar. list, tuple, np.ndarray for tensor. May
        contain Symbol as symbolic shape (but not string).

        dtype: builtins.float or other scalar builtin types.
        """
        if not isinstance(sym_shape, (list, tuple, np.generic, np.ndarray)):
            raise ValueError(
                "Illegal shape for Placeholder: {}".format(sym_shape))
        self.sym_shape = sym_shape
        self.dtype = dtype
        if self.dtype is None:
            self.dtype = builtins.float
        sym_type = self.type_inference()

        # Globally unique var name for placeholders
        name = 'placeholder_' + str(self.__class__.counter)
        self.__class__.counter += 1

        # List of output vars (consistent w/ other ops)
        self.outputs = [Var(name, sym_type)]

    def set_name(self, name):
        self.name = name
        self.outputs[0].name = name

    def type_inference(self):
        if len(self.sym_shape) == 0:
            return self.dtype
        return builtins.tensor(self.dtype, self.sym_shape)

    def __str__(self):
        return str(self.outputs[0])


k_used_symbols = set()
k_num_internal_syms = 0

def get_new_variadic_symbol():
    global k_num_internal_syms
    s = Symbol('*is' + str(k_num_internal_syms))
    k_num_internal_syms += 1
    return s

def get_new_symbol():
    global k_num_internal_syms
    s = Symbol('is' + str(k_num_internal_syms))
    k_num_internal_syms += 1
    return s

class Symbol(sm.Symbol):
    def __init__(self, sym_name):
        """
        Essentially sympy.Symbol representing an i32 value in shape.

        sym_name: str. If first character is *, then this symbol represents
        variadic rank. Otherwise the symbol name should start with a alpha
        character. `sym_name` must be unique if specified, or it'd be auto
        generated (to a non-variadic symbol). Furthermore, sym_name may not
        start with 'is' (internal symbol)
        """
        if not (sym_name[0].isalpha() or sym_name[0] == '*'):
            msg = 'Symbol name must start with a letter or *. Got {}'
            raise ValueError(msg.format(sym_name))
        if sym_name in k_used_symbols:
            msg = 'Symbol `{}` is used already.'
            raise ValueError(msg.format(sym_name))
        k_used_symbols.add(sym_name)
        self.name = sym_name
