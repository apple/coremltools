#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as _np
import sympy as _sm

from coremltools import _logger as logger
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as _target
from coremltools.converters.mil.mil.input_type import InternalInputType
from coremltools.converters.mil.mil.ops.helper import _get_version_of_op
from coremltools.converters.mil.mil.var import ListVar

from . import types
from .block import Function
from .operation import Operation
from .scope import ScopeSource
from .types.symbolic import k_num_internal_syms, k_used_symbols
from .var import Var


class Program:
    @staticmethod
    def _get_opset_str_value(op):
        return f"coremltools.target.{op.name}"

    def __init__(self):
        self.functions = {}
        self.skip_all_passes = False

    def _add_essential_scope_source(
        self, scope_source: Union[ScopeSource, List[ScopeSource]]
    ) -> None:
        """
        Add essential scope sources to functions.
        """
        for func in self.functions.values():
            func._add_essential_scope_source(scope_source)

    def _get_dialect_namespaces(self) -> Dict[str, List[Operation]]:
        """
        Return a dict which maps the dialect namespace into a list of corresponding operations.
        """
        res = defaultdict(list)

        def get_dialect_namespaces_block(block):
            for op in block.operations:
                for b in op.blocks:
                    get_dialect_namespaces_block(b)
                if hasattr(op, "_dialect_namespace"):
                    dialect_namespace = op._dialect_namespace
                    res[dialect_namespace].append(op)

        for func in self.functions.values():
            get_dialect_namespaces_block(func)
        return res

    def _get_max_opset_version_and_op(self):
        max_opset_version = _target.iOS13
        op_with_max_opset_version = None
        for func in self.functions.values():
            cur_max_opset, cur_op = func.get_max_opset_version_and_op()
            if cur_max_opset > max_opset_version:
                max_opset_version = cur_max_opset
                op_with_max_opset_version = cur_op
        return max_opset_version, op_with_max_opset_version

    def _check_ops_version_compatibility(self, max_opset_version):
        def check_version_compatibility_block(block):
            for op in block.operations:
                for b in op.blocks:
                    check_version_compatibility_block(b)
                if not hasattr(op, "_op_variants") or not isinstance(op._op_variants, dict):
                    continue
                expected_op_cls = _get_version_of_op(op._op_variants, max_opset_version)
                if type(op) is not expected_op_cls:
                    msg = (
                        "Op {} with an out of date version {} is detected. Please use @mb.program(input_specs=..., "
                        "opset_version={})"
                    ).format(
                        op.op_type,
                        self._get_opset_str_value(op.opset_version),
                        self._get_opset_str_value(max_opset_version),
                    )
                    raise ValueError(msg)
        for func in self.functions.values():
            check_version_compatibility_block(func)

    def _check_or_set_functions_opset_version(self, max_opset_version):
        funcs = list(self.functions.values())
        for func in funcs:
            if func.opset_version is None:
                func.opset_version = max_opset_version
            else:
                if func.opset_version < max_opset_version:
                    msg = "function should have at least opset_version {}. Got {}".format(
                        self._get_opset_str_value(max_opset_version),
                        self._get_opset_str_value(func.opset_version),
                    )
                    raise ValueError(msg)
        for func in funcs:
            if func.opset_version != funcs[0].opset_version:
                msg = "all functions must have the same opset_version. Got {} and {}.".format(
                    self._get_opset_str_value(func.opset_version),
                    self._get_opset_str_value(funcs[0].opset_version),
                )
                raise ValueError(msg)

    def _check_program_opset_version(self):
        max_opset_version, _ = self._get_max_opset_version_and_op()
        self._check_ops_version_compatibility(max_opset_version)
        self._check_or_set_functions_opset_version(max_opset_version)

    @staticmethod
    def _get_runtime_supported_dialect_opset() -> List[str]:
        """
        Return a list of supported dialect opsets at runtime.
        """
        return []

    def _check_invalid_opset(self):
        """
        Check if the program consists of opsets not supported by runtime.
        """
        dialect_namespaces = self._get_dialect_namespaces()
        if len(dialect_namespaces) != 0:
            for dialect_key in list(dialect_namespaces.keys()):
                if dialect_key not in self._get_runtime_supported_dialect_opset():
                    invalid_op = dialect_namespaces[dialect_key][0]
                    raise ValueError(
                        f'Core ML only support core opset. Got unsupported op "{invalid_op.name}" with type "{invalid_op.op_type}" of dialect namespace "{invalid_op._dialect_namespace}".'
                    )

    def _check_invalid_tensor_rank(self):
        """
        Check if the program consists of tensors with rank >= 6.
        """
        def _check_invalid_tensor_rank_block(block):
            for op in block.operations:
                for b in op.blocks:
                    _check_invalid_tensor_rank_block(b)
                for o in op.outputs:
                    if not isinstance(o, ListVar) and (o.rank < 0 or o.rank >= 6):
                        if op.op_type == "const" or op.op_type.startswith("constexpr_"):
                            if all(
                                child_op.op_type.startswith("constexpr_")
                                for child_op in o.child_ops
                            ):
                                # For const/constexpr op's constexpr output, tensor with rank > 5 is ok.
                                continue
                        raise ValueError(
                            f'Core ML only supports tensors with rank <= 5. Layer "{op.name}", '
                            f'with type "{op.op_type}", outputs a rank {o.rank} tensor. '
                        )

        for f in self.functions.values():
            _check_invalid_tensor_rank_block(f)

    def _check_invalid_const_tensor_input(self):
        """
        Check if non const tensor feed into const input.
        This might happen in the early stage of conversion, for instance:
            constexpr_ -> reshape -> transpose -> linear

        However, the pattern is optimized into the following in a graph pass.
            constexpr_ -> linear
        """
        def _check_invalid_const_tensor_input_block(block):
            for op in block.operations:
                for b in op.blocks:
                    _check_invalid_const_tensor_input_block(b)

                for k, v in op.inputs.items():
                    input_type = op.input_spec.input_types[k]

                    if (
                        input_type.const
                        and not isinstance(input_type, InternalInputType)
                        and not (v.op.op_type.startswith("constexpr_") or v.val is not None)
                    ):
                        raise ValueError(
                            f"In op {op.name}. Input {k} ({v.name}) must be const or constexpr ops."
                        )

        for f in self.functions.values():
            _check_invalid_const_tensor_input_block(f)

    def _check_early_error_out_for_invalid_program(self):
        """
        Early error out for
        1. tensor with rank >= 6
        2. non const tensor feed into const input
        3. program consist of non mil core ops
        """
        self._check_invalid_tensor_rank()
        self._check_invalid_const_tensor_input()
        self._check_invalid_opset()

    def add_function(self, name, ssa_func):
        if not isinstance(ssa_func, Function):
            raise ValueError("Only Function can be added to Program.")
        self.functions[name] = ssa_func
        self._check_program_opset_version()

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
            msg = "Found matching ops not exactly one. Found ops: {}"
            raise ValueError(msg.format(found_ops))
        return found_ops

    def validate(self, check_essential_scope: Optional[bool] = False) -> None:
        for f in self.functions.values():
            f.validate(force_validate=True, check_essential_scope=check_essential_scope)

    def construct_debug_handle_to_ops_mapping(self) -> Dict:
        """
        For PyMIL program translated from ExecuTorch only: Based on scope info inherited from EXIR,
        construct a debug handle to ops mapping. The mapping format is something like
        {
          1: [
            {"Type": "Program"},
            {"Type": "Function", "Name": "main"},
            {"Type": "Block"},
            {"Type": "Operation", "Operator": "add", "Output": "z"}
          ]
        }
        where `1`, `"main"`, `"add"`, and `"z"` are example values of
        the debug handle, function name, operation type,
        and output var name (or the name of the first output var, if multiple outputs)
        """
        debug_handle_to_ops_mapping = {}
        for function_name, function in self.functions.items():
            for operation in function.operations:
                # TODO (rdar://115846569): Handle multi-block case from EXIR
                if len(operation.blocks) > 0:
                    raise NotImplementedError("Multi-block case has not been supported yet")
                debug_handle = operation.scopes.get(ScopeSource.EXIR_DEBUG_HANDLE)
                if debug_handle is None:
                    continue
                debug_handle = debug_handle[0]
                if debug_handle not in debug_handle_to_ops_mapping:
                    debug_handle_to_ops_mapping[debug_handle] = []
                debug_handle_to_ops_mapping[debug_handle].append(
                    [
                        {"Type": "Program"},
                        {"Type": "Function", "Name": function_name},
                        {"Type": "Block"},
                        {
                            "Type": "Operation",
                            "Operator": operation.op_type,
                            "Output": operation.outputs[0].name,
                        },
                    ]
                )
        return debug_handle_to_ops_mapping

    def __getitem__(self, func_name):
        if func_name not in self.functions:
            msg = "Function {} not found in among functions {}."
            raise KeyError(msg.format(func_name, self.functions.keys()))
        return self.functions[func_name]

    def __repr__(self):
        return self.__str__()

    def __str__(self, print_attr: Optional[bool] = False) -> str:
        s = ""
        for f_name, f in self.functions.items():
            s += "\n"
            s += f.to_str(f_name, print_attr=print_attr)
        return s


class Placeholder:
    counter = 0

    def __init__(self, sym_shape, dtype=None, name=None, allow_rank0_input=False):
        """
        sym_shape: () or [] for scalar. list, tuple, np.ndarray for tensor. May
        contain Symbol as symbolic shape (but not string).

        dtype: types.float or other scalar builtin types.
        allow_rank0_input: A flag that allows the rank 0 placeholder.
        """
        if not isinstance(sym_shape, (list, tuple, _np.ndarray)):
            raise ValueError("Illegal shape for Placeholder: {}".format(sym_shape))

        if len(sym_shape) == 0:
            if not allow_rank0_input:
                raise ValueError('Rank-0 (input {}) is unsupported'.format(name))
            else:
                logger.warning('Rank-0 (input {}) is unsupported in coreml. You might run into error while\
                running this model'.format(name))

        for i, d in enumerate(sym_shape):
            if not isinstance(d, (_np.generic, int, Symbol)):
                msg = 'Placeholder dim {} in {} is not integer or symbol'
                raise ValueError(msg.format(i, sym_shape))
        self.sym_shape = sym_shape
        self.dtype = dtype
        if self.dtype is None:
            self.dtype = types.float
        self.name = name
        self._infer_output_var()

    def set_name(self, name):
        self.name = name
        self.outputs[0].name = name

    def type_inference(self):
        if len(self.sym_shape) == 0:
            return self.dtype
        return types.tensor(self.dtype, self.sym_shape)

    def __str__(self):
        return str(self.outputs[0])

    def _infer_output_var(self):
        sym_type = self.type_inference()

        # Globally unique var name for placeholders
        if self.name is None:
            self.name = f"{self.__class__.__name__}_{self.__class__.counter}"
            self.__class__.counter += 1

        # List of output vars (consistent w/ other ops)
        self.outputs = [Var(self.name, sym_type)]

def get_new_variadic_symbol():
    global k_num_internal_syms
    s = Symbol("*is" + str(k_num_internal_syms))
    k_num_internal_syms += 1
    return s


def get_new_symbol(name=None):
    """
    Returns a new symbol, optionally named.

    name: str (optional)
        Optional name that provides more readability. If the name specified is
        not available, an extra integer will be appended.
    """
    global k_used_symbols
    global k_num_internal_syms

    if name is not None:
        s = Symbol(name)
        if s in k_used_symbols:
            new_name = name + k_num_internal_syms
            msg = 'Symbol name "{}" already occupied. Renaming to {}'
            logger.warning(msg.format(name, new_name))
            s = Symbol(new_name)
    else:
        s = Symbol("is" + str(k_num_internal_syms))
    k_num_internal_syms += 1
    return s

def get_existing_symbol(name):
    global k_used_symbols
    if name not in k_used_symbols:
        msg = 'Symbol name {} does not exist'
        raise ValueError(msg.format(name))
    return k_used_symbols[name]


class Symbol(_sm.Symbol):
    def __init__(self, sym_name):
        """
        Essentially sympy.Symbol representing an i32 value in shape.

        sym_name: str. If first character is *, then this symbol represents
        variadic rank. Otherwise the symbol name should start with a alpha
        character. `sym_name` must be unique if specified, or it'd be auto
        generated (to a non-variadic symbol). Furthermore, sym_name may not
        start with 'is' (internal symbol)
        """
        if not (sym_name[0].isalpha() or sym_name[0] == "*"):
            msg = "Symbol name must start with a letter or *. Got {}"
            raise ValueError(msg.format(sym_name))
        global k_used_symbols
        if sym_name in k_used_symbols:
            msg = "Symbol `{}` is used already."
            raise ValueError(msg.format(sym_name))
        k_used_symbols[sym_name] = self
        self.name = sym_name
