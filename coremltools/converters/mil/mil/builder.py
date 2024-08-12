#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numbers
from collections import defaultdict
from typing import Any, Callable, List, Optional, Tuple, Type

import numpy as np

from coremltools import _logger as logger
from coremltools.converters.mil import mil
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.mil.types.symbolic import any_symbolic

from .block import Function, curr_block
from .input_type import (
    InternalInputType,
    ListOrTensorOrDictInputType,
    TensorInputType,
    TupleInputType,
)
from .program import Placeholder, StateTensorPlaceholder
from .scope import (
    SCOPE_STACK,
    VALID_OPS_TO_COPY_SCOPE_INFO,
    ScopeContextManger,
    ScopeInfo,
    ScopeSource,
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
    This class is a singleton builder to construct a MIL program. For more
    information, see `Create a MIL program <https://coremltools.readme.io/docs/model-intermediate-language#create-a-mil-program>`_.

    Importing ``.ops`` triggers the installation of all MIL ops into the Builder.
    For details on each op, see `MIL ops <https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html>`_.

    Examples
    --------

    >>> from coremltools.converters.mil.mil import Builder as mb
    >>> from coremltools.converters.mil.mil import Program, Function

    >>> prog = Program()
    >>> func_inputs = {"x": mb.placeholder(shape=[2,3]),
    >>>                "y": mb.placeholder(shape=[2,3])}
    >>> with Function(func_inputs) as ssa_fun:
    >>>   x, y = ssa_fun.inputs['x'], ssa_fun.inputs['y']
    >>>   res_var = mb.add(x=x, y=y) # created within ssa_fun block
    >>>   ssa_fun.set_outputs([res_var])
    >>> prog.add_function("main", ssa_fun)

    >>> # Importing ops triggers installation of all ops into Builder.
    >>> from .ops import defs as _ops

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
            err_msg = f"Cannot add const {val}"
            if any_symbolic(val):
                err_msg += (
                    "\nPython native vals (list, tuple), np.array that are"
                    + "operation inputs cannot have symbolic values. Consider feeding"
                    + "symbolic shape in through placeholder and use mb.shape() "
                    + f"operator. Input {name}: {val}"
                )
            raise ValueError(err_msg)
        const_name = cls._get_free_name(name)
        logger.debug("Adding const op '{}'".format(const_name))
        output_var = cls.const(val=val, name=const_name,
            before_op=before_op)
        return output_var


    @classmethod
    def _create_vars(cls, input_spec, op_name, before_op,
        candidate_kv):
        """
        For each key K in `candidate_kv`, create a Var if the
        following are satisfied:

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
                if not isinstance(val, (list, tuple)):
                    raise ValueError(f"Invalid type {type(val)} for TupleInputType param.")
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

            if isinstance(in_type, (TensorInputType, ListOrTensorOrDictInputType)):
                var = cls._add_const(val, new_var_name, before_op)
                update_dict[k] = var

        return update_dict

    @classmethod
    def _add_op(cls, op_cls, **kwargs):
        """
        Add an op of type `op_cls` (e.g., convolution) to current block.
        """
        kwargs = cls._maybe_set_name(kwargs, op_cls.__name__)
        logger.debug(
            "Adding op '{}' of type {}".format(kwargs["name"], op_cls.__name__)
        )
        before_op = kwargs.get("before_op", None)
        # Shallow copy list inputs to ensure op inputs are immutable
        kwargs = {k: v if not isinstance(v, (list, tuple)) else v[:] for k, v in kwargs.items() if v is not None}
        kwargs.update(cls._create_vars(
            input_spec=op_cls.input_spec,
            op_name=kwargs["name"], before_op=before_op,
            candidate_kv=kwargs))
        kwargs["enclosing_block"] = curr_block()

        # Add scope information
        current_scopes = SCOPE_STACK.get_curr_scopes()
        kwargs["scopes"] = current_scopes
        new_op = op_cls(**kwargs)

        # We record if the op is created under graph pass
        if len(current_scopes) == 1 and ScopeSource.COREMLTOOLS_GRAPH_PASS in current_scopes:
            VALID_OPS_TO_COPY_SCOPE_INFO[-1].add(new_op)

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
    def placeholder(
        shape: Tuple[Any],
        dtype: Optional[Type] = None,
        allow_rank0_input: Optional[bool] = False,
        name: Optional[str] = None,
    ) -> Placeholder:
        return Placeholder(shape, dtype, allow_rank0_input=allow_rank0_input, name=name)

    @staticmethod
    def TensorSpec(shape, dtype=None):
        return Placeholder(shape, dtype)

    @staticmethod
    def StateTensorSpec(shape, dtype=None):
        return StateTensorPlaceholder(shape, dtype)

    @staticmethod
    def state_tensor_placeholder(shape, dtype=None):
        return StateTensorPlaceholder(shape, dtype)

    @staticmethod
    def _create_function(
        main_block: Callable,
        input_specs: Optional[List[Placeholder]] = None,
        opset_version: Optional[AvailableTarget] = None,
    ):
        """
        Utility to construct a pymil function.
        """
        if input_specs is None:
            input_specs = []

        # validate number of function inputs
        num_args = main_block.__code__.co_argcount
        arg_names = list(main_block.__code__.co_varnames)[:num_args]
        if len(input_specs) != num_args:
            raise ValueError(
                f"{main_block.__name__} expects {num_args} inputs: {arg_names}. Got {len(input_specs)} input_specs."
            )

        # create the function
        input_spec_dict = {k: v for k, v in zip(arg_names, input_specs)}
        with Function(input_spec_dict, opset_version) as func:
            input_vars = [func.inputs[a] for a in arg_names]
            outputs = main_block(*input_vars)
            if isinstance(outputs, tuple):
                outputs = list(outputs)
            elif not isinstance(outputs, list):
                outputs = [outputs]
            func.set_outputs(outputs)

        # infer the opset version if not provided
        max_opset_version, _ = func.get_max_opset_version_and_op()
        if opset_version is None:
            func.opset_version = max_opset_version

        return func

    @staticmethod
    def function(
        input_specs: Optional[List[Placeholder]] = None,
        opset_version: Optional[AvailableTarget] = None,
    ):
        """
        The ``mb.function`` decorator creates a MIL function.

        Parameters
        ----------
        input_specs: List[TensorSpec]
            Describes the function inputs

        opset_version: AvailableTarget enum
            Describes the opset version of the function

        Examples
        --------
        >>> import coremltools as ct
        >>> @mb.function(input_specs=[mb.TensorSpec(shape=(1,2))], opset_version=ct.target.iOS16)
        >>> def func(a):
        >>>     return mb.add(x=a, y=2)

        """
        def wrapper(main_block):
            return Builder._create_function(main_block, input_specs, opset_version)

        return wrapper

    @staticmethod
    def program(
        input_specs: Optional[List[Placeholder]] = None,
        opset_version: Optional[AvailableTarget] = None,
        function_name: Optional[str] = "main",
    ):
        """
        The ``mb.program`` decorator creates a MIL program with a single
        function with name ``function_name``.

        Parameters
        ----------
        input_specs: List[TensorSpec]
            Describes the function inputs

        opset_version: AvailableTarget enum
            Describes the opset version of the program

        function_name: str
            Name of the function

        Examples
        --------
        >>> import coremltools as ct
        >>> from coremltools.converters.mil.mil import Builder as mb
        >>>
        >>> @mb.program(input_specs=[mb.TensorSpec(shape=(1,2))], opset_version=ct.target.iOS16)
        >>> def prog(a):
        >>>     return mb.add(x=a, y=2)

        """
        def wrapper(main_block):
            function = Builder._create_function(main_block, input_specs, opset_version)
            program = mil.Program()
            program.add_function(function_name, function)
            return program
        return wrapper

    @staticmethod
    def scope(
        *scopes: List[ScopeInfo],
    ) -> ScopeContextManger:
        """
        The ``mb.scope`` creates a context manager, which makes the operations created within it have the corresponding scope information.

        Parameters
        ----------
        scopes: Optional[List[ScopeInfo]] (Optional)
            * A list of ScopeInfo under the context manager.
            * The source in each ScopeInfo cannot be duplicated.
            * If not provided, this context manager does no affects.

        Examples
        --------
        The following is an example of creating a scope for torchscript module heirarchy with type and name information.

        .. sourcecode:: python

            @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
            def prog(x):
                with mb.scope(
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["Module1"]),
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=["module_1"]),
                ):
                    return mb.add(x=x, y=4.3, name="add_1")


        In the previous example, the "add_1" op will have two scope attributes, for torchscipt module type and name:
            * TORCHSCRIPT_MODULE_TYPE: ["Module1"]
            * TORCHSCRIPT_MODULE_NAME: ["module_1"]

        The following is an example of creating nested scopes:

        .. sourcecode:: python

            @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
            def prog(x):
                with mb.scope(
                    ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["Module1"]),
                ):
                    x = mb.add(x=x, y=4.3, name="add_1")
                    with mb.scope(
                        ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["Module2"]),
                        ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=["module_2"]),
                    ):
                        return mb.add(x=x, y=3.2, name="add_2")

        In the previous example, the "add_1" op would have a scope attribute:
            * TORCHSCRIPT_MODULE_TYPE: ["Module1"]

        while the "add_2" op would have scope attributes:
            * TORCHSCRIPT_MODULE_TYPE: ["Module1", "Module2"]
            * TORCHSCRIPT_MODULE_NAME: ["module_2"]
        """
        return ScopeContextManger(*scopes)
