#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from typing import Dict, Set, Tuple

from coremltools import _logger as logger
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Placeholder, Program, Var, types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.symbolic import any_symbolic, is_symbolic


@register_pass(namespace="common")
class materialize_symbolic_shape_program(AbstractGraphPass):
    """
    If we realize that only a few fixed shapes are used in a symbolic-shape model,
    we may prefer materialization into a fixed-shape (multifunction) model,
    which has the potential to be further optimized

    Supported options:

    - ``function_name_to_materialization_map``: Dict[str, Dict[str, Tuple[int]]]
        A dictionary specifying the name of new functions to be created,
        and for each new function what is the new fixed shapes for inputs.
        If a new function has the same name as an old function,
        then the old function will be overridden

    - ``source_function_name``: str
        The name of the source symbolic-shape function to be materialized, default = main

    Example:

    Suppose we have a symbolic shape model with 2 symbols ``is0`` and ``is1``

    .. code-block::

        symbolic_shape_mlmodel: ct.models.MLModel
        symbolic_shape_prog = symbolic_shape_mlmodel._mil_program

    We may invoke this graph pass to materialize some fixed shapes (e.g. ``is0 = 2, is1 = 5``
    and ``is0 = 4, is1 = 7``), then run every other optimization passes

    .. code-block::

        pass_pipeline: PassPipeline = ct.PassPipeline.DEFAULT
        pass_pipeline.insert_pass(0, "common::materialize_symbolic_shape_program")
        pass_pipeline.set_options(
            "common::materialize_symbolic_shape_program",
            {
                "function_name_to_materialization_map": {
                    # As an example, let us assume the input is x (is0, is1, 1024)
                    "materialization_2_5": {"x": (2, 5, 1024)},
                    "materialization_4_7": {"x": (4, 7, 1024)},
                }
            },
        )
        PassPipelineManager.apply_pipeline(symbolic_shape_prog, pass_pipeline)

    We will arrive at

    .. code-block::

        main[CoreML8](%x: (is0, is1, 1024, fp16)(Tensor)) {
          block0() {
            ...
          } -> (%y)
        }

        materialization_2_5[CoreML8](%x: (2, 5, 1024, fp16)(Tensor)) {
          block5() {
            ...
          } -> (%y)
        }

        materialization_4_7[CoreML8](%x: (4, 7, 1024, fp16)(Tensor)) {
          block6() {
            ...
          } -> (%y)
        }
    """

    def __init__(self) -> None:
        self._function_name_to_materialization_map: Dict[str, Dict[str, Tuple[int]]] = None
        self._source_function_name: str = "main"

    @property
    def function_name_to_materialization_map(self) -> Dict[str, Dict[str, Tuple[int]]]:
        return self._function_name_to_materialization_map

    @function_name_to_materialization_map.setter
    def function_name_to_materialization_map(
        self, function_name_to_materialization_map_: Dict[str, Dict[str, Tuple[int]]]
    ) -> None:
        if not isinstance(function_name_to_materialization_map_, dict):
            raise ValueError(
                "function_name_to_materialization_map must be type of dict, "
                f"but got {type(function_name_to_materialization_map_)}"
            )
        for function_name, materialization_map in function_name_to_materialization_map_.items():
            if not isinstance(function_name, str):
                raise ValueError(
                    f"Materialized new function name must be type of str, but got {type(function_name)}"
                )
            if not isinstance(materialization_map, dict):
                raise ValueError(
                    f"Materialization map must be type of dict, but got {type(materialization_map)}"
                )
            for input_name, shape in materialization_map.items():
                if not isinstance(input_name, str):
                    raise ValueError(
                        f"Materialization map key (input name) must be type of str, but got {type(input_name)}"
                    )
                if not isinstance(shape, tuple):
                    raise ValueError(
                        f"Materialization map value (shape) must be type of tuple, but got {type(shape)}"
                    )
                for size in shape:
                    if not isinstance(size, int):
                        raise ValueError(f"Shape element must be type of int, but got {type(size)}")
        self._function_name_to_materialization_map = function_name_to_materialization_map_

    @property
    def source_function_name(self) -> str:
        return self._source_function_name

    @source_function_name.setter
    def source_function_name(self, source_function_name_: str) -> None:
        if not isinstance(source_function_name_, str):
            raise ValueError(
                f"Source function name must be type of str, but got {type(source_function_name_)}"
            )
        self._source_function_name = source_function_name_

    @staticmethod
    def _canonicalize_materialization_map(
        source_function: Function,
        function_name_to_materialization_map: Dict[str, Dict[str, Tuple[int]]],
    ) -> Dict[str, Dict[str, int]]:
        """
        User input ``materialization_map`` maps input names to fixed shapes,
        but what a rigorous graph pass really need is a map from symbols to integers,
        so here we construct the canonical materialization map from user input
        """
        function_name_to_canonical_materialization_map: Dict[str, Dict[str, int]] = {}
        for function_name, materialization_map in function_name_to_materialization_map.items():
            canonical_materialization_map: Dict[str, int] = {}
            for source_input_var in source_function.inputs.values():
                input_name: str = source_input_var.name
                if input_name in materialization_map:
                    fixed_shape = materialization_map[input_name]
                    for size, integer in zip(source_input_var.shape, fixed_shape):
                        if is_symbolic(size):
                            if size.name not in canonical_materialization_map:
                                canonical_materialization_map[size.name] = integer
                            else:
                                existing_integer = canonical_materialization_map[size.name]
                                if existing_integer != integer:
                                    raise ValueError(
                                        f"Inconsistent symbol materialization in new function {function_name}: "
                                        f"symbol {size.name} is to be materialized into {existing_integer} and {integer}. "
                                        f"Please make sure input {input_name} has compatible shape with others"
                                    )
                        else:
                            if size != integer:
                                raise ValueError(
                                    f"Already fixed size cannot be altered: new function {function_name}, "
                                    f"input {input_name}, original size is {size}, but user specified new size {integer}"
                                )
                else:
                    logger.warning(
                        f"In new function {function_name}, "
                        f"although materialization for input {input_name} is not specified, "
                        f"it may still be materialized if it shares symbol with other inputs"
                    )
            function_name_to_canonical_materialization_map[
                function_name
            ] = canonical_materialization_map
        return function_name_to_canonical_materialization_map

    @staticmethod
    def _validate_inputs(
        source_function: Function,
        function_name_to_canonical_materialization_map: Dict[str, Dict[str, int]],
    ) -> None:
        # Get existing symbols in program
        symbols: Set[str] = set()
        for source_input_var in source_function.inputs.values():
            for size in source_input_var.shape:
                if is_symbolic(size):
                    symbols.add(size.name)
        # Compare existing symbols vs user specified materialization map
        for (
            function_name,
            canonical_materialization_map,
        ) in function_name_to_canonical_materialization_map.items():
            symbols_to_be_materialized = set(canonical_materialization_map.keys())
            # Q: Why we only check symbols is subset of symbols_to_be_materialized,
            #    but not symbols_to_be_materialized is subset of symbols?
            # A: Since our API has user specify {input name: fixed shape tuple},
            #    we will not receive any redundant symbol,
            #    i.e. symbols_to_be_materialized will always be a subset of symbols
            if not symbols.issubset(symbols_to_be_materialized):
                logger.warning(
                    f"In new function {function_name}, these symbols will not be materialized: "
                    f"{symbols - symbols_to_be_materialized}"
                )

    @staticmethod
    def _maybe_materialize_symbolic_shape(
        shape: Tuple, canonical_materialization_map: Dict[str, int]
    ) -> Tuple:
        if any_symbolic(shape):
            materialized_shape = []
            for size in shape:
                if is_symbolic(size) and size.name in canonical_materialization_map:
                    materialized_shape.append(canonical_materialization_map[size.name])
                else:
                    materialized_shape.append(size)
            return tuple(materialized_shape)
        else:
            return shape

    @staticmethod
    def _create_placeholders(
        source_function: Function, canonical_materialization_map: Dict[str, int]
    ) -> Dict[str, Placeholder]:
        placeholders: Dict[str, Placeholder] = {}
        for source_input_name, source_input_var in source_function.inputs.items():
            target_input_shape = (
                materialize_symbolic_shape_program._maybe_materialize_symbolic_shape(
                    source_input_var.shape, canonical_materialization_map
                )
            )
            if types.is_state(source_input_var.sym_type):
                placeholders[source_input_name] = mb.state_tensor_placeholder(
                    target_input_shape, source_input_var.dtype
                )
            else:
                placeholders[source_input_name] = mb.placeholder(
                    target_input_shape, source_input_var.dtype
                )
        return placeholders

    @staticmethod
    def _copy_construct_const_var(source_const_var: Var) -> Var:
        target_const_var = mb.const(val=source_const_var.val, name=source_const_var.name)
        if (
            hasattr(source_const_var.op, "weight_key")
            and source_const_var.op.weight_key is not None
        ):
            target_const_var.op.weight_key = source_const_var.op.weight_key
        return target_const_var

    def apply(self, prog: Program) -> None:
        if self.source_function_name not in prog.functions:
            raise ValueError(
                f"Source function {self.source_function_name} not found, "
                f"available functions are {list(prog.functions.keys())}"
            )
        source_function = prog.functions[self.source_function_name]

        function_name_to_canonical_materialization_map = self._canonicalize_materialization_map(
            source_function, self.function_name_to_materialization_map
        )
        self._validate_inputs(source_function, function_name_to_canonical_materialization_map)

        for (
            target_function_name,
            canonical_materialization_map,
        ) in function_name_to_canonical_materialization_map.items():
            context: Dict[str, Var] = {}
            with Function(
                inputs=self._create_placeholders(source_function, canonical_materialization_map),
                opset_version=source_function.opset_version,
            ) as target_function:
                # Extract function input variables
                for target_input_name, target_input_var in target_function.inputs.items():
                    context[target_input_name] = target_input_var

                # Rebuild all operations with new variables
                for source_operation in source_function.operations:
                    # Instead of building constants as we encounter,
                    # we will build them when we find them in operation input,
                    # otherwise we will mess up with block internal variable
                    if source_operation.op_type == "const":
                        continue
                    else:
                        # prepare operation inputs
                        target_name_to_input: Dict[str, Var] = {}
                        for source_input_name, source_input_vars in source_operation.inputs.items():
                            # operation input may either be Var or Tuple[Var]
                            is_source_single_input = isinstance(source_input_vars, Var)
                            if is_source_single_input:
                                source_input_vars = [source_input_vars]
                            target_input_vars = []
                            for source_input_var in source_input_vars:
                                # build const input that is currently missing from context
                                if source_input_var.name not in context:
                                    assert (
                                        source_input_var.op.op_type == "const"
                                    ), "Only const may be absent from context"
                                    context[source_input_var.name] = self._copy_construct_const_var(
                                        source_input_var
                                    )
                                target_input_vars.append(context[source_input_var.name])
                            if is_source_single_input:
                                target_name_to_input[source_input_name] = target_input_vars[0]
                            else:
                                target_name_to_input[source_input_name] = tuple(target_input_vars)
                        # build operation
                        outputs = getattr(mb, source_operation.op_type)(
                            **target_name_to_input, name=source_operation.name
                        )
                        # operation output may either be Var or Tuple[Var]
                        if isinstance(outputs, Var):
                            outputs = [outputs]
                        for output, source_output in zip(outputs, source_operation.outputs):
                            output.set_name(source_output.name)
                            context[output.name] = output

                # Set function output variables
                target_function.set_outputs(
                    [
                        context[source_output_var.name]
                        for source_output_var in source_function.outputs
                    ]
                )

                prog.add_function(target_function_name, target_function)

        # For some reason, if we run const_deduplication._deduplicate_const_across_functions here,
        # the populated `const.weight_id` will get lost if we run pass pipeline afterwards,
        # so we have no choice but to let user manually deduplicate after all passes are done
        # TODO (rdar://131680531): Investigate why it happens & whether we can change this behavior
        logger.warning(
            "(If you are using ct.utils.materialize_dynamic_shape_mlmodel, "
            "you are safe to ignore this warning message) "
            "Weights are duplicated in each materialized new function, "
            "so you may want to run const_deduplication._deduplicate_const_across_functions "
            "on your pymil program before serialization to milproto"
        )
