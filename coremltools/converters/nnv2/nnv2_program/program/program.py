# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import copy
import logging
import numpy as np
import sympy as sm

from coremltools.converters.nnv2.builtin_types import builtins
from .var import Var, TupleVar, InternalVar

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


class Operation(object):
    """
    Represents Operation in NNv2.Program.

    # Properties
    name (str):
        The name of the operation

    input_types (InputTypes):
        Read-only named input types from all subclasses. Input types are used
        to validate `inputs`.

    inputs [_input_vars] (dict of str --> Var):
        An Operation (subclass of Operation) only has access to input Var,
        which is already validated against `input_types`.

    outputs [_output_vars] (list of Var):
        List of output var based on type inference. Read-only
    """
    def __init__(self, **kwargs):
        """
        kwargs:

        input_types (str -> _InputType): required
        kwargs[] = Var for each i in required input_types
        """
        # TODO: input_types might just need to be local var, not class
        # instance member.
        self._input_types = kwargs['input_types']
        self.name = kwargs.get('name', None)

        self._output_vars = None
        self._input_vars = {}
        self.blocks = []
        self.enclosing_block = curr_block()
        self.set_inputs(**kwargs)

    def set_inputs(self, **kwargs):
        """
        Perform type inference and auto_val computation based on new input Vars
        in kwargs. If self._output_vars is None then we generate _output_vars;
        otherwise no new Var is created, but type inference result is verified
        against existing _output_vars.
        """
        self._validate_and_set_inputs(**kwargs)
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
                if builtins.is_tuple(sym_type):
                    # ignore sym_type, as TupleVar.__init__ creates that from
                    # sym_val
                    new_var = TupleVar(name,
                                       elems=sym_val,
                                       op=self,
                                       op_output_idx=i)
                else:
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
                if sym_type != out_var.sym_type or sym_val != out_var.sym_val:
                    msg = "Output Var {} type changes with new input Vars"
                    raise ValueError(msg.format(out_var.name))

    def _auto_val(self, output_types):
        """
        # Evaluation is two stage:
        #
        # Stage 1: Attempt to evaluate non-symbolic value. This succeeds if
        #
        #   - all required inputs can be materialized (in_var.val is not None
        #     for all required in_var).
        #   - eval() is implemented.
        #
        # Stage 2: If Stage 1 fails, attempt to evaluate symbolic_value. This
        #          succeeds if:
        #
        #   - all required inputs have symbolic value (in_var.sym_val is not
        #     None for all required in_var).
        #   - sym_eval() is implemented.
        #
        # If stage 1 succeeds, outputs[i].val is not None. If only stage 2
        # succeeds, outputs[i].sym_val is not none but outputs[i].val is None.
        # If neither stage succeeds, outputs[i].val and outputs[i].sym_val is
        # None

        output_types: tuple of builtin types

        Returns:
            output_vals: tuple of builtin type with value, or tuple of None
        """
        do_auto_val = True
        do_auto_sym_val = True
        # Determine if AUTO_VAL is possible
        for in_name, in_type in self._input_types.items():
            if in_type.optional:
                # Optional inputs are not required to invoke eval() or
                # sym_eval()
                continue

            if self._input_vars[in_name].val is None:
                do_auto_val = False
            if self._input_vars[in_name].sym_val is None:
                do_auto_sym_val = False

        if do_auto_val:
            # Is eval implemented?
            try:
                self.eval()
            except NotImplementedError as e:
                do_auto_val = False
        if not do_auto_val and do_auto_sym_val:
            # Is ref_impl_sym implemented?
            try:
                self.sym_eval()
            except NotImplementedError as e:
                do_auto_sym_val = False

        if not do_auto_val and not do_auto_sym_val:
            # No sym_val or val possible.
            return tuple(None for _ in output_types)

        # perform auto val
        if do_auto_val:
            vals = self.eval()
        else:
            vals = self.sym_eval()
        if not isinstance(vals, (tuple, list)):
            vals = (vals, )
        auto_val = []
        for t, v in zip(output_types, vals):
            builtin_val = t()
            builtin_val.val = v
            auto_val.append(builtin_val)
        return auto_val

    def eval(self):
        """
        Optional Python implementation of the op based on (materialized) values
        in `self.input_var`. Return a builtin value (single output) or a tuple of
        builtin values (multi-outputs) of the same length as returned by `
        type_inference`
        """
        msg = "eval() is not implemented by op {}"
        raise NotImplementedError(msg.format(self.op_type))

    def sym_eval(self):
        """
        Optional Python implementation of the op based on (symbolic) values
        in `self.input_var`. Return a builtin value (single output) or a tuple of
        builtin values (multi-outputs) of the same length as returned by `
        type_inference`
        """
        msg = "sym_eval() is not implemented by op {}"
        raise NotImplementedError(msg.format(self.op_type))

    def output_names(self):
        """
        Optional. If implemented, we set the output var i name as
        self.name + "/" + output_names[i]

        Returns a string (single output) or tuple of strings
        """
        msg = "output_names() is not implemented by op {}"
        raise NotImplementedError(msg.format(self.op_type))

    def get_input_types(self):
        raise NotImplementedError(
            "This function must be implemented by each op")

    def type_inference(self):
        """
        Return (builtin_type, builtin_val) pair from type inference.
        builtin_val may be None if symbolic_value is not attainable at compile
        time.
        """
        raise NotImplementedError(
            "This function must be implemented by each op")

    def _validate_and_set_inputs(self, **kwargs):
        non_attributes = [
            "name",
            "symbolic_datatype",
            "datatype",
            "symbolic_value",
            "value",
            "version",
            "input_types",
            "before_op",
        ]
        op_inputs = list(self._input_types.keys())
        legal_args = op_inputs + non_attributes

        for key in kwargs.keys():
            if key not in legal_args:
                raise RuntimeError("Unknown input {} for op {}".format(
                    key, self.op_type))

        # kwargs MUST contain a value (usually Var) compatible with
        # corresponding _InputType for each
        # - required _InputType
        # - optional _InputType with default value.
        #
        # kwargs MAY additionally contain a compatible value for optional
        # _InputType without default value (when user specifies it)
        for in_name, in_type in self._input_types.items():
            if in_name not in kwargs and in_type.default is None:
                setattr(self, in_name, None)
                continue
            if in_name not in kwargs and not in_type.optional:
                raise RuntimeError("Input {} is required".format(in_name))
            # in_name must be in kwargs here
            var = kwargs[in_name]
            if not in_type.is_compatible(var):
                msg = "Op {} of type {} and input {}: {} not " + \
                    "compatible with input type {}"
                raise ValueError(
                    msg.format(self.name, self.op_type, in_name, var.sym_type,
                               in_type.input_type))

            # Remove this operation itself from existing input Var's child_ops
            existing_input_var = self._input_vars.get(in_name, None)
            if existing_input_var is not None:
                existing_input_var.remove_child_op(self)

            # Set var as input_var
            if isinstance(var, Var):
                var.add_child_op(self)
            elif isinstance(var, tuple):
                for v in var:
                    v.add_child_op(self)
            # ignore function inputs
            self._input_vars[in_name] = var
            setattr(self, in_name, var)

    @property
    def inputs(self):
        return self._input_vars

    @property
    def outputs(self):
        return self._output_vars

    @property
    def op_type(self):
        return type(self).__name__

    def replace_var_after_op(self, old_var, new_var):
        """
        See SsaBlock.replace_var_after_op
        """
        self.enclosing_block.replace_var_after_op(self, old_var, new_var)

    def remove_from_block(self):
        """

        Remove / detach itself from the enclosing block. See SsaBlock.remove_ops
        for details.
        """
        self.enclosing_block.remove_ops([self])

    @staticmethod
    def var_to_str(v):
        if isinstance(v, tuple):
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
                    val_str = str(self.val.sym_val)
                s += "val=" + val_str
            else:
                s += "val=(file_value)"
        else:
            s += ", ".join([k + "=" + Operation.var_to_str(self.inputs[k]) \
                    for k in self._input_types.keys() if \
                    k in self.inputs and not is_internal_input(k)])
        s += ")\n"
        for b in self.blocks:
            s += b.indented_str(indent=indent + SPACES)
        return s

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.indented_str(SPACES)


class SsaBlock(object):
    __slots__ = [
        "name", "_block_inputs", "_outputs", "operations", "_internal_vars",
        "outer_op"
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
            function_inputs are always available for this block and all blocks
            nested within. If function_inputs is None, get it from
            `outer_op.block`
        """
        self.name = name
        if self.name is None:
            self.name = SsaBlock._get_new_name()

        # list[Var] map. This is converted to str:var_name when generating
        # NNv2 proto.
        if block_inputs is not None:
            self._block_inputs = tuple(copy.deepcopy(v) for v in block_inputs)
            for v in self._block_inputs:
                v._op = None
                v.op_output_idx
                v._child_ops = set()
                v.name = v.name + ".x"
        else:
            self._block_inputs = tuple()

        # list[Operation]. Topologically sorted.
        self.operations = []

        # list[Var]. This is converted to str when generating NNv2 proto.
        self._outputs = None

        # If we create const, whose inputs (mode, val) cannot be const
        # (infinite recursion). They must be considered as always available.
        self._internal_vars = set()

        self.outer_op = outer_op
        if self.outer_op is None and not isinstance(self, SsaFunction):
            msg = "SsaBlock {} is not SsaFunction and thus outer_op cannot be None"
            raise ValueError(msg.format(self.name))

    def find_ops(self, prefix=None, op_type=None):
        """
        Return list of ops with name matching `prefix` if specified and
        op_type, if specified. At least one of {prefix, op_type} must be specified.

        prefix: str

        Return list[Operation]. Empty list if no op satisfies.
        """
        found_ops = []
        for op in self.operations:
            prefix_match = prefix is not None and op.name[:len(prefix)] == prefix
            op_type_match = op_type is not None and op.op_type == op_type
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
            raise ValueError("outputs must be list of Vars")
        self._outputs = outputs

    def __enter__(self):
        global BLOCK_STACK
        BLOCK_STACK.append(self)
        return self

    def __exit__(self, type, value, traceback):
        global BLOCK_STACK
        BLOCK_STACK = BLOCK_STACK[:-1]

    def _op_idx_preceeding_vars(self, target_op):
        """
        Returns:

        index (int) of target_op in self.operations if found, -1 otherwise.

        preceeding_vars: set[Var]
            outputs of (set) up to, including target_op. If target_op is not
            found, include all vars output by self.operations.
        """
        idx = -1
        preceeding_vars = set()
        # find the location of target_op
        for i, op in enumerate(self.operations):
            [preceeding_vars.add(o) for o in op.outputs]
            if op == target_op:
                idx = i
                break
        return idx, preceeding_vars

    def _op_idx_available_vars(self, target_op):
        """
        Find target_op index in the current block, and the available vars from
        lexical scopes. If target_op is None, return -1, all Vars available at
        the end of current block.

        Available vars consists of:
        - InternalVars in current and enclosing scopes
        - Vars in current and enclosing scopes "before" target_op in the
          ordering of block.operations
        """
        # InternalVars
        avail_vars = set(self._internal_vars)
        avail_vars.update(self._block_inputs)
        if self.outer_op is not None:
            _, outer_vars = \
                    self.outer_op.enclosing_block._op_idx_available_vars(
                            self.outer_op)
            avail_vars.update(outer_vars)
        idx, preceeding_vars = self._op_idx_preceeding_vars(target_op)
        avail_vars.update(preceeding_vars)
        if isinstance(self, SsaFunction):
            avail_vars.update(self.function_inputs)
        return idx, avail_vars

    def insert_op_before(self, new_op, before_op=None):
        """
        new_op's outputs are not used (not input to any other op) after
        this call. All inputs to new_op must available at or before
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

        Comment: We assume op1 has been constructed outside of the block using
        available_vars in the block.

        Comment: insert_op_before(op0, op1) would error as %2 (an input to op1)
        is not available before op0.
        """
        # Find available vars in all lexical scoping. May be inefficient (N^2
        # cost).
        idx, available_vars = self._op_idx_available_vars(before_op)
        if before_op is not None and idx == -1:
            raise ValueError("before_op {} is not in the block".format(
                before_op.name))

        # check inputs are available
        for k, v in new_op.inputs.items():
            if not isinstance(v, (Var, tuple)):
                continue
            if isinstance(v, Var):
                vs = [v]
            else:
                vs = v
            for s in vs:
                if s not in available_vars:
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

    def replace_var_after_op(self, anchor_op, old_var, new_var):
        """
        Replace all uses of `old_var` with `new_var` after `anchor_op`
        old_var, new_var must meet the following conditions:

        - old_var, new_var both existing within the block. This implies that
          the op generating new_var must be inserted prior to this
          replacement.

        - Affected ops (i.e., Operation after anchor_op that take old_var as
          input) must generate the same type inference result as before.

        - new_var must be available at or before anchor_op in the order of
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
        idx, available_vars = self._op_idx_available_vars(anchor_op)
        if idx == -1:
            raise ValueError("anchor_op {} not found in block {}".format(
                anchor_op.name, self.name))
        if new_var not in available_vars:
            msg = "new_var {} is not available in block {} at or before " + \
                "anchor_op {}"
            raise ValueError(
                msg.format(new_var.name, self.name, anchor_op.name))

        num_ops_affected = 0
        for op in self.operations[idx:]:
            new_inputs = {}
            affected = False
            for k, v in op.inputs.items():
                if v == old_var:
                    new_inputs[k] = new_var
                    affected = True
                else:
                    new_inputs[k] = v
            if affected:
                num_ops_affected += 1
                op.set_inputs(**new_inputs)
        logging.debug(
            "Num ops affected in replacing var: {}".format(num_ops_affected))

        # If old_var is block's output, replace as well.
        if old_var in self._outputs:
            idx = self._outputs.index(old_var)
            self._outputs[idx] = new_var

    def remove_ops(self, existing_ops):
        """
        Remove `existing_ops` (list[Operation]) that must be pre-existing in
        the block. Error if any other op in the block uses output Vars of
        `existing_ops`
        """
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
            for v in op.inputs.values():
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
        raise self.indented_str()


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
        Return list of ops with name matching `prefix` if specified and
        op_type, if specified. At least one of {prefix, op_type} must be specified.

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
