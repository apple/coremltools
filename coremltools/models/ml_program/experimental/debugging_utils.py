# Copyright (c) 2025, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import gc
import random
from collections import OrderedDict, deque
from collections.abc import AsyncIterator
from enum import Enum
from logging import getLogger as _getLogger
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

import numpy as np
import tqdm

from coremltools import _SPECIFICATION_VERSION_IOS_16, ComputeUnit, proto

from ...model import MLModel
from .async_wrapper import MLModelAsyncWrapper
from .remote_device import Device, _DeviceCtlError

_logger = _getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")

class _Trie(Generic[K, V]):
    class Node:
        def __init__(
            self,
            key: Optional[K],
            values: Optional[Iterable[V]],
        ):
            self.key = key
            self.values = values
            self.children = {}

    def __init__(self):
        self.root = _Trie.Node(key=None, values=None)

    def insert(
        self,
        keys: List[K],
        value: V,
    ):
        if len(keys) == 0:
            raise ValueError("Parameter 'keys' must not be empty.")

        curr_node = self.root
        if len(keys) > 1:
            for key in keys[0 : len(keys) - 1]:
                node = curr_node.children.get(key, None)
                if node is None:
                    node = _Trie.Node(key=key, values=None)
                    curr_node.children[key] = node
                curr_node = node

        node = curr_node.children.get(keys[-1], None)
        if node is not None:
            node.values = [] if node.values is None else node.values
            node.values.append(value)
        else:
            curr_node.children[keys[-1]] = _Trie.Node(key=keys[-1], values=[value])

    def find(
        self,
        keys: List[K],
    ) -> Optional[Iterable[V]]:
        if len(keys) == 0:
            raise ValueError("keys must not be empty.")

        curr_node = self.root
        for key in keys[0 : len(keys) - 1]:
            node = curr_node.children.get(key, None)
            if node is None:
                return None

            curr_node = node

        node = curr_node.children.get(keys[-1], None)
        return node.values if node is not None else None

    def remove(
        self,
        keys: List[K],
    ) -> Optional[Iterable[V]]:
        if len(keys) == 0:
            raise ValueError("keys must not be empty.")

        curr_node = self.root
        for key in keys[0 : len(keys) - 1]:
            node = curr_node.children.get(key, None)
            if node is None:
                return None

            curr_node = node

        node = curr_node.children.pop(keys[-1], None)
        removed_values = node.values if node is not None else None
        return removed_values

    def starts_with(
        self,
        keys: List[K],
    ) -> Optional[Iterable[V]]:
        curr_node = self.root
        for key in (key for key in keys if len(key) > 0):
            node = curr_node.children.get(key, None)
            if node is None:
                return None
            curr_node = node

        values = []
        queue = deque()
        queue.append(curr_node)
        while len(queue) > 0:
            node = queue.popleft()
            if node.values is not None:
                values.extend(node.values)

            for child in node.children.values():
                queue.append(child)

        return values


class MLModelInspector:
    """
    A class for inspecting an ML model.

    This class provides functionality to retrieve intermediate outputs of an ML model.

    Examples
    --------
    .. sourcecode:: python
        inspector = coremltools_internal.models.debugging_utils.MLModelInspector(model)
        input_data = {"input_1": np.random.rand(1, 3, 224, 224).astype(np.float32)}
        # The intermediate outputs we want to inspect
        output_names = ["conv1_output", "relu1_output", "pool1_output"]
        async for output_name, output_value in inspector.inspect(
            inputs=input_data, output_names=output_names
        ):
            print(f"Name: {output_name}")
            print(f"Output: {output_value}")

    """

    @staticmethod
    def _clone_spec(
        spec: "proto.Model_pb2.Model",
    ) -> "proto.Model_pb2.Model":
        spec_class = spec.__class__
        new_spec = spec_class()
        new_spec.CopyFrom(spec)
        return new_spec

    @staticmethod
    def _create_output_name_to_op_map(
        ops: Iterable[proto.MIL_pb2.Operation],
    ) -> "OrderedDict[str, proto.MIL_pb2.Operation]":
        result = OrderedDict()
        for op in ops:
            for output in op.outputs:
                result[output.name] = op
        return result

    @staticmethod
    def _init_check(
        model: MLModel,
        compute_units: ComputeUnit,
        function_name: Optional[str],
        optimization_hints: Optional[Dict[str, Any]],
        device: Optional[Device],
    ):
        if not isinstance(model, MLModel):
            raise TypeError('The "model" parameter must be of type "MLModel"')

        if not isinstance(compute_units, ComputeUnit):
            raise TypeError('The "compute_units" parameter must be of type "ComputeUnit"')

        if function_name is not None and not isinstance(function_name, str):
            raise TypeError('The "function_name" parameter must be of type "str"')

        if optimization_hints is not None and not isinstance(optimization_hints, Mapping):
            raise TypeError(f"The 'optimization_hints' must be of mapping type (e.g., dict)")

        if device is not None and not isinstance(device, Device):
            raise TypeError('The "device" parameter must be of type "Device"')

        if device is not None and not device.session.is_alive:
            raise ValueError(
                f"The device '{device.name}' is not ready for debugging. "
                "Please ensure the following steps have been completed:\n"
                "1. The device is properly connected.\n"
                "2. You have called 'prepare_for_model_debugging' on the device.\n"
                "3. The debugging session has been successfully initiated.\n"
            )

    def __init__(
        self,
        model: MLModel,
        compute_units: ComputeUnit = None,
        function_name: Optional[str] = None,
        optimization_hints: Optional[Dict[str, Any]] = None,
        device: Optional[Device] = None,
    ):
        """
        Initializes the MLModelInspector.

        Parameters
        ----------
        model : MLModel
            The MLModel to inspect.

        compute_units : coremltools.ComputeUnit
            The compute units to use. Defaults to the model's compute unit.

        function_name : Optional[str]
            The function name. Defaults to the model's function name.

        optimization_hints : Optional[Dict[str, Any]]
            Keys are the names of the optimization hint, either 'reshapeFrequency' or 'specializationStrategy'.
            Values are enumeration values of type ``coremltools.ReshapeFrequency`` or ``coremltools.SpecializationStrategy``.

        device: Device
           The device on which the model will execute.
        """
        compute_units = compute_units if compute_units is not None else model.compute_unit
        MLModelInspector._init_check(
            model=model,
            compute_units=compute_units,
            function_name=function_name,
            optimization_hints=optimization_hints,
            device=device,
        )

        spec = model.get_spec()
        if spec.WhichOneof("Type") != "mlProgram":
            raise ValueError("MLModelInspector only supports ML program.")

        self.model = model
        self._cached_models = {}
        self._data_type_to_feature_type = {
            proto.MIL_pb2.DataType.FLOAT16: proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16,
            proto.MIL_pb2.DataType.FLOAT64: proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE,
            proto.MIL_pb2.DataType.FLOAT32: proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32,
            proto.MIL_pb2.DataType.INT32: proto.FeatureTypes_pb2.ArrayFeatureType.INT32,
        }
        self._model_spec = spec
        function_name = function_name if function_name is not None else model.function_name
        func_spec = spec.mlProgram.functions.get(
            function_name if function_name is not None else "main", None
        )
        if func_spec is None:
            raise ValueError(f"Missing function for name : {function_name}")

        block_spec = func_spec.block_specializations.get(func_spec.opset, None)
        if block_spec is None:
            raise ValueError(f"Missing block specialization for opset : {func_spec.opset}")

        self._function_name = function_name
        self._optimization_hints = (
            optimization_hints if optimization_hints is not None else model.optimization_hints
        )
        self._output_name_to_op_map = MLModelInspector._create_output_name_to_op_map(
            block_spec.operations
        )
        self.compute_units = compute_units
        self.device = device

    @property
    def output_names(self) -> List[str]:
        """
        Returns a list of all output names in the model.

        Returns
        List[str]
            A list of output names.
        """
        return list(reversed(self._output_name_to_op_map.keys()))

    @property
    def output_name_to_op_map(self) -> "OrderedDict[str, proto.MIL_pb2.Operation]":
        """
        Returns a dictionary mapping output names to their corresponding operations.

        Returns
        -------
        Dict[str, proto.MIL_pb2.Operation]
            A dictionary of output names to operations.
        """
        return self._output_name_to_op_map.copy()

    async def _create_model_with_outputs(
        self,
        output_names: List[str],
        ignore_const_ops: bool,
    ) -> MLModelAsyncWrapper:
        model_key = frozenset(output_names)
        model = None
        for cached_model_key in self._cached_models:
            if model_key in cached_model_key:
                model = self._cached_models.get(model_key, None)
                break

        if model is not None:
            return model

        cloned_spec = MLModelInspector._clone_spec(self._model_spec)
        cloned_spec.specificationVersion = max(
            _SPECIFICATION_VERSION_IOS_16, cloned_spec.specificationVersion
        )
        func_spec = cloned_spec.mlProgram.functions.get(
            self._function_name if self._function_name is not None else "main", None
        )
        block_spec = func_spec.block_specializations.get(func_spec.opset, None)
        const_ops = {
            "const",
            "compression.constexpr_blockwise_shift_scale",
            "constexpr_lut_to_dense",
            "constexpr_sparse_to_dense",
            "constexpr_lut_to_sparse",
            "constexpr_affine_dequantize",
            "constexpr_cast",
        }
        for output_name in output_names:
            if output_name in block_spec.outputs:
                continue

            op = self._output_name_to_op_map.get(output_name, None)
            if op is None or (op.type in const_ops and ignore_const_ops):
                continue

            output_type = next(
                iter([output.type for output in op.outputs if output.name == output_name]), None
            )
            if output_type is None or output_type.WhichOneof("type") != "tensorType":
                raise ValueError(
                    "Only tensor type is supported as an intermediate output {}".format(output_type)
                )

            data_type = output_type.tensorType.dataType
            feature_type = self._data_type_to_feature_type.get(data_type, None)
            if feature_type is None:
                _logger.warning(
                    f"Skipping ({output_name}), ({data_type}) is not supported as model output."
                )
                continue
            block_spec.outputs.append(output_name)
            output_desc = proto.Model_pb2.FeatureDescription()
            output_desc.name = output_name
            output_desc.type.multiArrayType.dataType = feature_type
            cloned_spec.description.output.append(output_desc)

        model = MLModelAsyncWrapper.from_spec_or_path(
            spec_or_path=cloned_spec,
            weights_dir=self.model.weights_dir,
            compute_units=self.compute_units,
            function_name=self._function_name,
            optimization_hints=self._optimization_hints,
            device=self.device,
        )

        await model.load()
        self._cached_models[model_key] = model
        return model

    async def _generate_models_with_outputs(
        self,
        output_names: List[str],
        ignore_const_ops: bool,
    ) -> AsyncIterator[Tuple[MLModelAsyncWrapper, List[str]]]:
        if len(output_names) == 0:
            return
        all_output_names = [output_names]
        while len(all_output_names) > 0:
            curr_output_names = all_output_names[0]
            del all_output_names[0]
            model = None
            error_message = None
            try:
                model = await self._create_model_with_outputs(
                    curr_output_names,
                    ignore_const_ops=ignore_const_ops,
                )
            except _DeviceCtlError as e:
                # Re-raise devicectl errors
                raise e
            except Exception as e:
                error_message = str(e)

            if model is None:
                _logger.warning(
                    f"Failed to create model with outputs names ({curr_output_names}), "
                    f"error ({error_message if error_message is not None else 'unknown'})"
                )

                if len(curr_output_names) > 1:
                    """
                    Failed to retrieve intermediate outputs. As a recovery strategy,
                    we're splitting the op list and retrying. This divide-and-conquer
                    approach helps isolate potentially problematic ops. Each part will
                    be processed separately in subsequent attempts, allowing us to bypass ops
                    that may be causing issues while continuing with the retrieval process for the
                    remaining ops.
                    """
                    pivot_index = random.randint(1, len(curr_output_names) - 1)
                    _logger.warning(
                        f"Retrying with ({curr_output_names[:pivot_index]}) and ({curr_output_names[pivot_index:]})"
                    )
                    all_output_names.insert(0, curr_output_names[pivot_index:])
                    all_output_names.insert(0, curr_output_names[:pivot_index])
            else:
                yield (model, curr_output_names)

    def clear_cached_models(self):
        """
        Clears the cache of generated models.
        """
        for model in self._cached_models.values():
            model.cleanup()

        self._cached_models.clear()
        gc.collect()

    async def inspect(
        self,
        inputs: Dict[str, np.array],
        output_names: List[str] = None,
        num_predict_intermediate_outputs: Optional[int] = None,
        ignore_const_ops: bool = True,
    ) -> AsyncIterator[Tuple[str, Optional[np.array]]]:
        """
        Retrieves intermediate outputs from the model for given inputs.
        Parameters
        ----------
        inputs : Dict[str, np.array]
            Input data for the model.

        output_names : List[str]
            Names of outputs to retrieve. Defaults to all outputs.

        num_predict_intermediate_outputs : Optional[int]
            The number of intermediate outputs to retrieve in each ``MLModel`` prediction call. Defaults to None.

        ignore_const_ops : bool
            Whether to ignore constant operations. Defaults to True.

        Returns
        -------
        AsyncIterator[Tuple[str, Optional[np.array]]]
            An iterator of tuples containing output names and their values.

        Yields
        ------
        Tuple[str, Optional[np.array]]
            A tuple of (output_name, output_value) for each requested output.

        Examples
        --------
        .. sourcecode:: python
            inspector = coremltools.models.ml_program.experimental.debugging_utils.MLModelInspector(
                model
            )
            input_data = {"input_1": np.random.rand(1, 3, 224, 224).astype(np.float32)}
            # The intermediate outputs we want to inspect
            output_names = ["conv1_output", "relu1_output", "pool1_output"]
            async for output_name, output_value in inspector.inspect(
                inputs=input_data, output_names=output_names
            ):
                print(f"Name: {output_name}")
                print(f"Output: {output_value}")
        """

        def batch(iterable, size: int):
            l = len(iterable)
            for index in range(0, l, size):
                yield iterable[index : min(index + size, l)]

        def find_index(lst, value, start):
            if start < 0 or start >= len(lst):
                raise ValueError(f"Invalid start index: {start}")
            try:
                return lst.index(value, start)
            except ValueError:
                return None

        if output_names is None:
            output_names = self.output_names
        else:
            invalid_names = [
                name for name in output_names if name not in self._output_name_to_op_map
            ]
            if len(invalid_names) > 0:
                raise ValueError(
                    f"Invalid output names ({invalid_names}). Available output names are: {list(reversed(self._output_name_to_op_map.keys()))}"
                )

        if len(output_names) == 0:
            return

        output_names = list(OrderedDict.fromkeys(output_names))
        num_predict_intermediate_outputs = (
            num_predict_intermediate_outputs
            if num_predict_intermediate_outputs is not None
            else len(output_names)
        )
        for curr_output_names in batch(output_names, num_predict_intermediate_outputs):
            """
            Handle potential missing outputs gracefully:
            1. Iterate through the requested output names in the given order.
            2. For each output name:
                - If the value is available in the outputs, include it.
                - If the value is not found, use None as the value.
            """
            processed_index = -1
            async for model, processed_output_names in self._generate_models_with_outputs(
                output_names=curr_output_names,
                ignore_const_ops=ignore_const_ops,
            ):
                if len(processed_output_names) == 0:
                    continue

                outputs = await model.predict(inputs)
                max_idx = find_index(
                    curr_output_names,
                    processed_output_names[-1],
                    processed_index + 1,
                )
                if max_idx is None:
                    continue

                max_idx = max(max_idx, processed_index + 1)
                for i in range(processed_index + 1, max_idx + 1):
                    output_name = curr_output_names[i]
                    output_value = outputs.get(output_name, None)
                    yield (output_name, output_value)

                processed_index = max_idx

            for i in range(processed_index + 1, len(curr_output_names)):
                output_name = curr_output_names[i]
                yield (output_name, None)

    def __del__(self):
        self.clear_cached_models()

    async def retrieve_outputs(
        self,
        inputs: Dict[str, np.array],
        output_names: List[str],
        num_predict_intermediate_outputs: Optional[int] = None,
        ignore_const_ops: bool = True,
    ) -> Dict[str, Optional[np.array]]:
        """
        Asynchronously retrieve intermediate outputs for the specified output names and inputs.

        This method inspects the model with given inputs and collects the intermediate outputs
        for the specified output names.

        Parameters
        ----------
        output_names : List[str]
            Names of intermediate outputs to retrieve.

        inputs : Dict[str, np.ndarray])
            Input data for the model.

        num_predict_intermediate_outputs : Optional[int]
            The number of intermediate outputs to retrieve in each ``MLModel`` prediction call. Defaults to None.

        ignore_const_ops : bool
            Whether to ignore constant operations. Defaults to True.

        Returns
        -------
        Dict[str, Optional[np.array]]

            A dictionary mapping output names to their corresponding numpy array values.
            If an output is not available, its value will be None.

         Examples
        --------
        .. sourcecode:: python
            inspector = coremltools.models.ml_program.experimental.debugging_utils.MLModelInspector(
                model
            )
            input_data = {"input_1": np.random.rand(1, 3, 224, 224).astype(np.float32)}
            # The intermediate outputs we want to retrieve
            outputs = await inspector.retrieve_outputs(
                inputs=inputs,
                output_names=["conv1_output", "relu1_output", "pool1_output"],
            )

            for output_name, output_value in outputs.items():
                print(f"Name: {output_name}")
                print(f"Output: {output_value}")
        """
        outputs = {}
        async for name, value in self.inspect(
            inputs=inputs,
            output_names=output_names,
            ignore_const_ops=ignore_const_ops,
            num_predict_intermediate_outputs=num_predict_intermediate_outputs,
        ):
            outputs[name] = value

        return outputs


class _Status(Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


def _retrieve_direct_dependency_names(
    op: proto.MIL_pb2.Operation,
) -> List[str]:
    result = []
    for arg in op.inputs.values():
        for binding in arg.arguments:
            if binding.name is not None:
                result.append(binding.name)
    return result


def _retrieve_direct_dependencies(
    op: proto.MIL_pb2.Operation,
    output_name_to_op_map: Dict[str, proto.MIL_pb2.Operation],
) -> List[proto.MIL_pb2.Operation]:
    result = []
    for name in _retrieve_direct_dependency_names(op):
        dep = output_name_to_op_map.get(name, None)
        if dep is not None:
            result.append(dep)

    return result


def _retrieve_dependency_names(
    op: proto.MIL_pb2.Operation,
    output_name_to_op_map: Dict[str, proto.MIL_pb2.Operation],
    processed_output_names: Set[str],
    max_dependencies: int,
    skip_op: Callable[[proto.MIL_pb2.Operation], bool] = lambda op: False,
) -> List[str]:
    result = OrderedDict()
    visited: set[int] = set()
    queue: list[proto.MIL_pb2.Operation] = list()

    def enqueue_deps(curr_op: proto.MIL_pb2.Operation):
        nonlocal visited
        nonlocal queue

        deps = _retrieve_direct_dependencies(
            op=curr_op, output_name_to_op_map=output_name_to_op_map
        )

        for dep in deps:
            if id(dep) in visited or skip_op(dep):
                continue
            visited.add(id(dep))
            queue.append(dep)

    enqueue_deps(op)
    initial_len = len(queue)
    processed = 0

    while (processed < initial_len or len(result) < max_dependencies) and len(queue) > 0:
        current_op = queue.pop(0)
        processed += 1
        for output in current_op.outputs:
            if output.name not in processed_output_names:
                result[output.name] = None
        enqueue_deps(current_op)

    return list(result.keys())


def _is_failure_source(
    op: proto.MIL_pb2.Operation,
    statuses: Dict[int, _Status],
    output_name_to_op_map: Dict[str, proto.MIL_pb2.Operation],
) -> bool:
    if statuses.get(id(op), None) != _Status.FAIL:
        return False

    queue = deque()

    def add_deps(curr_op: proto.MIL_pb2.Operation):
        deps = _retrieve_direct_dependencies(
            op=curr_op,
            output_name_to_op_map=output_name_to_op_map,
        )

        for dep in deps:
            queue.insert(0, dep)

    add_deps(op)
    while len(queue) > 0:
        dep = queue.popleft()
        status = statuses.get(id(dep), None)
        if status is None:
            return False
        elif status == _Status.FAIL:
            return False
        elif status == _Status.UNKNOWN:
            # If the dep status is unknown then we check its direct dependencies.
            add_deps(dep)

    # The operation itself is likely the source of the failure, as its dependencies have passed
    # the validation check.
    return True


def skip_op_by_type(
    op: proto.MIL_pb2.Operation,
    op_types: Iterable[str] = {
        "expand_dims",
        "reshape",
        "reshape_like",
        "squeeze",
        "transpose",
        "slice_by_index",
        "slice_by_size",
        "shape",
        "split",
        "stack",
        "concat",
        "fill",
        "const",
        "compression.constexpr_blockwise_shift_scale",
        "constexpr_lut_to_dense",
        "constexpr_sparse_to_dense",
        "constexpr_lut_to_sparse",
        "constexpr_affine_dequantize",
        "constexpr_cast",
        "cast",
    },
) -> bool:
    """
    Determines if an operation should be skipped based on its type.
    """
    return op.type in op_types


def _skip_op_excluding_output_names(
    skip_op: Callable[[proto.MIL_pb2.Operation], bool],
    output_names: List[str],
) -> Callable[[proto.MIL_pb2.Operation], bool]:
    def wrapper(op: proto.MIL_pb2.Operation):
        if any(output.name in output_names for output in op.outputs):
            return False

        return skip_op(op)

    return wrapper

def _get_unique_dependencies(
    op: proto.MIL_pb2.Operation,
    output_name_to_op_map: Dict[str, proto.MIL_pb2.Operation],
) -> Dict[int, proto.MIL_pb2.Operation]:
    """
    Computes the unique dependencies of an operation.
    """
    queue = deque()
    queue.append(op)
    unique_ops = {}
    while len(queue) > 0:
        curr_op = queue.popleft()
        if id(curr_op) in unique_ops:
            continue

        unique_ops[id(curr_op)] = curr_op
        for dep in _retrieve_direct_dependencies(
            op=curr_op,
            output_name_to_op_map=output_name_to_op_map,
        ):
            queue.append(dep)

    return unique_ops

class MLModelValidator:
    """
    A validator class for diagnosing numerical issues in an ML model.

    This class provides methods to traverse and validate operations within an ML model,
    specifically focusing on detecting issues such as NaN (Not a Number) or infinite outputs.
    It uses a graph traversal approach to identify the source of numerical instabilities.

    Examples
    --------
    .. sourcecode:: python
        validator = coremltools.models.ml_program.experimental.debugging_utils.MLModelValidator(
            model
        )
        failing_ops = validator.find_failing_ops(
            inputs={"input": np.array([1, 2, 3])},
            validate_output=lambda op, out: np.isnan(out).any(),
            output_names=["output1", "output2"],
        )
    """

    def __init__(
        self,
        model: MLModel,
        function_name: Optional[str] = None,
        compute_units: ComputeUnit = None,
        optimization_hints: Optional[Dict[str, Any]] = None,
        num_predict_intermediate_outputs: int = 20,
        device: Optional[Device] = None,
    ):
        """
        Initializes the MLModelValidator.

        Parameters
        ----------
        model : MLModel
            The model to be validated.

        function_name : Optional[str]
            The function name. Defaults to the model's function name.

        compute_units : coremltools.ComputeUnit:
            The compute units to use. Defaults to the model's compute unit.

        optimization_hints : Optional[Dict[str, Any]]
            Keys are the names of the optimization hint, either 'reshapeFrequency' or 'specializationStrategy'.
            Values are enumeration values of type ``coremltools.ReshapeFrequency`` or ``coremltools.SpecializationStrategy``.

        num_predict_intermediate_outputs : int
            The number of intermediate outputs to retrieve in each ``MLModel`` prediction call. Defaults to 20.

        device: Device
           The device on which the model will execute.
        """

        model_inspector = MLModelInspector(
            model,
            function_name=function_name,
            compute_units=compute_units,
            optimization_hints=optimization_hints,
            device=device,
        )

        self.model = model
        self._outputs = {}
        self._visited = set()
        self._num_predict_intermediate_outputs = num_predict_intermediate_outputs
        self._model_inspector = model_inspector
        self._output_name_to_op_map = self._model_inspector.output_name_to_op_map

    async def _process_op(
        self,
        op: proto.MIL_pb2.Operation,
        inputs: Dict[str, np.array],
        queue: List[proto.MIL_pb2.Operation],
        validate_output: Callable[[proto.MIL_pb2.Operation, np.array], bool],
        skip_op: Callable[[proto.MIL_pb2.Operation], bool],
    ) -> _Status:
        def add_deps(curr_op: proto.MIL_pb2.Operation):
            # The operation validation failed; process its dependencies.
            for dep in _retrieve_direct_dependencies(
                op=curr_op,
                output_name_to_op_map=self._output_name_to_op_map,
            ):
                if id(dep) not in self._visited:
                    queue.insert(0, dep)

        self._visited.add(id(op))

        if skip_op(op):
            add_deps(op)
            return _Status.UNKNOWN

        processed_output_names = set(self._outputs.keys())
        dep_names = _retrieve_dependency_names(
            op=op,
            max_dependencies=self._num_predict_intermediate_outputs,
            processed_output_names=processed_output_names,
            output_name_to_op_map=self._output_name_to_op_map,
            skip_op=skip_op,
        )

        op_output_names = [output.name for output in op.outputs]
        output_names = list(op_output_names) + dep_names
        # Retrieve the operation outputs along with selected dependencies.
        # This will reduce the number of prediction calls.
        outputs = await self._model_inspector.retrieve_outputs(
            output_names=output_names,
            inputs=inputs,
            num_predict_intermediate_outputs=self._num_predict_intermediate_outputs,
        )
        self._outputs.update(outputs)
        self._model_inspector.clear_cached_models()

        status = _Status.PASS

        for output_name in op_output_names:
            value = self._outputs.get(output_name, None)
            self._outputs.pop(output_name, None)
            _logger.info(f"Validating ({output_name})")

            if value is None:
                _logger.warning(f"Failed to retrieve value for ({output_name}) from model.)")
                status = _Status.UNKNOWN
                break

            if not validate_output(op, value):
                # The operation validation failed; process its dependencies.
                _logger.info(f"Failed validation for ({output_name})")
                status = _Status.FAIL
                break
            else:
                _logger.info(f"Passed validation for ({output_name})")

        if status != _Status.PASS:
            add_deps(op)

        return status

    async def find_failing_ops(
        self,
        inputs: Dict[str, np.array],
        validate_output: Callable[[proto.MIL_pb2.Operation, np.array], bool],
        output_names: Optional[List[str]] = None,
        skip_op: Callable[[proto.MIL_pb2.Operation], bool] = skip_op_by_type,
    ) -> List[proto.MIL_pb2.Operation]:
        """
        Identify operations in the model that fail the specified validation criteria.

        This method traverses the model's operation graph, starting from the specified
        output operations, and applies the given validation function to each operation's
        output. It returns a list of operations that are likely the source of failures.

        Parameters
        ----------
        inputs : Dict[str, np.array]
            A dictionary of input tensors for the model. Keys are input names, and values are numpy arrays.

        validate_output : Callable[[proto.MIL_pb2.Operation, np.array], bool]
            A function that takes an operation and its output array, and returns False if the output
            fails the validation criteria, True otherwise.

        output_names : Optional[List[str]]
            A list of specific output names to start the traversal from. If None, all model outputs are used.

        skip_op: Callable[[proto.MIL_pb2.Operation], bool]
            A function that determines if an operation should be skipped.

        Returns
        -------
        List[proto.MIL_pb2.Operation]:
            A list of operations that are identified as likely sources of failure based on the validation criteria.

        Notes
        -----
            - The method uses a breadth-first search strategy to traverse the operation graph.
            - An operation is considered a failure source if it fails validation, but its direct inputs pass.

        Examples
        --------
        .. sourcecode:: python
            validator = coremltools.models.ml_program.experimental.debugging_utils.MLModelValidator(
                model
            )
            failing_ops = await validator.find_failing_ops(
                inputs={"input": np.array([1, 2, 3])},
                validate_output=lambda op, out: not np.isnan(out).any(),
                output_names=["output1", "output2"],
            )
        """

        if output_names is None:
            output_names = [
                output_name for output_name in self._model_inspector.model.output_description
            ]

        queue = []
        all_deps = {}
        for output_name in output_names:
            op = self._output_name_to_op_map.get(output_name, None)
            if op is not None:
                queue.append(op)
                all_deps.update(
                    _get_unique_dependencies(
                        op=op,
                        output_name_to_op_map=self._output_name_to_op_map,
                    )
                )

        result = []
        statuses = {}

        with tqdm.tqdm(total=len(all_deps), desc="\033[Analyzing operations...\033[0m") as pbar:
            while len(queue) > 0:
                op = queue[0]
                status = statuses.get(id(op), None)
                if status is not None:
                    if _is_failure_source(
                        op=op,
                        statuses=statuses,
                        output_name_to_op_map=self._output_name_to_op_map,
                    ):
                        """
                        The operation did not pass validation, but its dependencies did.
                        This suggests that the failure likely comes from this operation.
                        """
                        result.append(op)

                    queue.pop(0)
                else:
                    status = await self._process_op(
                        op=op,
                        inputs=inputs,
                        queue=queue,
                        validate_output=validate_output,
                        skip_op=_skip_op_excluding_output_names(
                            skip_op=skip_op, output_names=output_names
                        ),
                    )
                    output_name = next((output.name for output in op.outputs), "")
                    pbar.set_description(
                        desc=f"\033[1mAnalyzed operation:\033[0m {output_name}, \033[1mtype:\033[0m {op.type}"
                    )
                    pbar.update(n=1)
                statuses[id(op)] = status

        self._outputs = {}
        self._visited = set()

        return result

    async def find_failing_ops_with_infinite_output(
        self,
        inputs: Dict[str, np.array],
        output_names: List[str] = None,
        skip_op: Callable[[proto.MIL_pb2.Operation], bool] = skip_op_by_type,
    ) -> List[proto.MIL_pb2.Operation]:
        """
        Identify operations in the model that produce infinite outputs.

        This method traverses the model's operation graph and checks for infinite values
        in the output of each operation. It returns a list of operations that are likely the
        source of failures. It's useful for debugging numerical instability issues in the model.

        Parameters
        ----------
        inputs : dict[str, np.array]
            A dictionary of input tensors for the model. Keys are input names, and values are numpy arrays.

        output_names : Optional[List[str]]
            A list of specific output names to start the traversal from. If None, all model outputs are used.

        skip_op: Callable[[proto.MIL_pb2.Operation], bool]
            A function that determines if an operation should be skipped.

        Returns
        -------
        List[proto.MIL_pb2.Operation]
            A list of operations that are identified as likely sources of failure.

        Notes
        -----
            - The method uses a breadth-first search strategy to traverse the operation graph.
            - An operation is considered a failure source if it outputs infinite values while its direct inputs do not.

        Examples
        --------
        .. sourcecode:: python
            validator = coremltools.models.ml_program.experimental.debugging_utils.MLModelValidator(
                model
            )
            failing_ops = await validator.find_failing_ops_with_infinite_output(
                inputs={"input": np.array([1, 2, 3])}, output_names=["output1", "output2"]
            )
        """

        def validate_output(op: proto.MIL_pb2.Operation, value: np.array):
            return not np.isinf(value).any()

        return await self.find_failing_ops(
            inputs=inputs,
            validate_output=validate_output,
            output_names=output_names,
            skip_op=skip_op,
        )

    async def find_failing_ops_with_nan_output(
        self,
        inputs: Dict[str, np.array],
        output_names: List[str] = None,
        skip_op: Callable[[proto.MIL_pb2.Operation], bool] = skip_op_by_type,
    ) -> List[proto.MIL_pb2.Operation]:
        """
        Identify operations in the model that produce NaN (Not a Number) outputs.

        This method traverses the model's operation graph and checks for NaN values
        in the output of each operation. It returns a list of operations that are likely
        the source of failures. It's useful for debugging numerical instability issues in the model.

        Parameters
        ----------
        inputs : Dict[str, np.array]
            A dictionary of input tensors for the model. Keys are input names, and values are numpy arrays.

        output_names : Optional[List[str]]
            A list of specific output names to start the traversal from. If None, all model outputs are used.

        skip_op: Callable[[proto.MIL_pb2.Operation], bool]
            A function that determines if an operation should be skipped.

        Returns
        -------
        List[proto.MIL_pb2.Operation]
            A list of operations that are identified as likely sources of failure.

        Notes
        -----
            - The method uses a breadth-first search strategy to traverse the operation graph.
            - An operation is considered a failure source if it outputs NaN while its direct inputs do not.

        Examples
        --------
        .. sourcecode:: python
            validator = coremltools.models.ml_program.experimental.debugging_utils.MLModelValidator(
                model
            )
            failing_ops = await validator.find_failing_ops_with_nan_output(
                inputs={"input": np.array([1, 2, 3])}, output_names=["output1", "output2"]
            )
        """

        def validate_output(op: proto.MIL_pb2.Operation, value: np.array):
            return not np.isnan(value).any()

        return await self.find_failing_ops(
            inputs=inputs,
            validate_output=validate_output,
            output_names=output_names,
            skip_op=skip_op,
        )


def compute_snr_and_psnr(
    x: np.array,
    y: np.array,
) -> Tuple[float, float]:
    """
    Compute the Signal-to-Noise Ratio (SNR) and Peak Signal-to-Noise Ratio (PSNR) between two signals.

    This function calculates the SNR and PSNR between two input signals, typically used to compare
    an original signal (y) with a processed or noisy version of that signal (x).

    Parameters
    ----------
    x : np.array
        The processed or noisy signal.
    y : np.array
        The original or reference signal.

    Returns
    -------
    Tuple[float, float]
    A tuple containing two float values:
        - snr (float): The Signal-to-Noise Ratio in decibels (dB).
        - psnr (float): The Peak Signal-to-Noise Ratio in decibels (dB).

    Raises:
        AssertionError: If the lengths of x and y are not equal.


    .. sourcecode:: python
        original = np.array([1, 2, 3, 4, 5])
        noisy = np.array([1.1, 2.1, 2.9, 4.2, 5.1])
        snr, psnr = compute_snr_and_psnr(noisy, original)
        print(f"SNR: {snr:.2f} dB, PSNR: {psnr:.2f} dB")
    """
    assert x.shape == y.shape
    eps = 1e-5
    eps2 = 1e-10
    noise = x - y
    noise_var = np.sum(noise**2) / len(noise)
    signal_energy = np.sum(y**2) / len(y)
    max_signal_energy = np.amax(y**2)
    snr = 10 * np.log10((signal_energy + eps) / (noise_var + eps2))
    psnr = 10 * np.log10((max_signal_energy + eps) / (noise_var + eps2))
    return snr, psnr


def _create_name_index(
    names: Iterable[str],
) -> _Trie[str, str]:
    result = _Trie[str, str]()
    for name in names:
        keys = name.split("_")
        result.insert(
            keys=keys,
            value=name,
        )

    return result


async def _retrieve_target_outputs(
    inspector: MLModelInspector,
    reference_output_names: List[str],
    inputs: Dict[str, np.array],
    target_output_names_index: _Trie[str, str],
    num_predict_intermediate_outputs: Optional[int],
) -> Dict[str, Optional[np.array]]:
    target_output_names = []
    target_to_reference_name_map = {}
    for reference_output_name in reference_output_names:
        key = reference_output_name.split("_")
        target_output_name = target_output_names_index.find(key)
        if target_output_name is None:
            target_output_name = target_output_names_index.starts_with(key + ["cast"])

        target_output_name = (
            next(iter(target_output_name), None) if target_output_name is not None else None
        )
        if target_output_name is not None:
            target_output_names.append(target_output_name)
            target_to_reference_name_map[target_output_name] = reference_output_name

    target_outputs = await inspector.retrieve_outputs(
        output_names=target_output_names,
        inputs=inputs,
        num_predict_intermediate_outputs=num_predict_intermediate_outputs,
    )

    outputs = {}
    for key, value in target_outputs.items():
        outputs[target_to_reference_name_map[key]] = value

    return outputs

class MLModelComparator:
    """
    A class for comparing two MLModel objects and identifying discrepancies in their outputs.

    This class provides functionality to compare the outputs of a reference model and a target model,
    helping to identify operations that produce different results.

    The ModelComparator is designed to compare models derived from the same source.
    Using it with reference and target models originating from different sources may
    lead to unreliable or meaningless results.

    Examples
    --------
    .. sourcecode:: python
        # Load the reference and target models
        reference_model = coremltools.models.MLModel(
            "model.mlpackage", compute_unit=coremltools.ComputeUnit.CPU_ONLY
        )
        target_model = coremltools.models.MLModel(
            "model.mlpackage", compute_unit=coremltools.ComputeUnit.CPU_AND_GPU
        )
        # Create an instance of MLModelComparator
        comparator = (
            coremltools.models.ml_program.experimental.debugging_utils.MLModelComparator(
                reference_model, target_model
            )
        )
        # Prepare input data
        input_data = {"input_1": np.random.rand(1, 3, 224, 224).astype(np.float32)}
        # Find failing operations with a PSNR of less than 40.
        failing_ops = await comparator.find_failing_ops(
            inputs=input_data, compare_output=lambda x, y: compute_snr_and_psnr(x, y)[1] >= 40.0
        )
    """

    def __init__(
        self,
        reference_model: MLModel,
        target_model: MLModel,
        function_name: Optional[str] = None,
        optimization_hints: Optional[Dict[str, Any]] = None,
        num_predict_intermediate_outputs: int = 20,
        reference_device: Optional[Device] = None,
        target_device: Optional[Device] = None,
    ):
        """
        Initializes the MLModelComparator.

        Parameters
        ----------
        reference_model : MLModel
             The reference MLModel.

        target_model : MLModel
            The target MLModel to compare against the reference.

        function_name : Optional[str]
            The function name. Defaults to the model's function name.

        optimization_hints : Optional[Dict[str, Any]]
            Keys are the names of the optimization hint, either 'reshapeFrequency' or 'specializationStrategy'.
            Values are enumeration values of type ``coremltools.ReshapeFrequency`` or ``coremltools.SpecializationStrategy``.

        num_predict_intermediate_outputs : int
            The number of intermediate outputs to retrieve in each ``MLModel`` prediction call. Defaults to 20.

        reference_device: Device
            The device on which the reference model will execute.

        target_device: Device
            The device on which the target model will execute.
        """

        reference_model_inspector = MLModelInspector(
            model=reference_model,
            function_name=function_name,
            optimization_hints=optimization_hints,
            device=reference_device,
        )

        target_model_inspector = MLModelInspector(
            model=target_model,
            function_name=function_name,
            optimization_hints=optimization_hints,
            device=target_device,
        )

        self.reference_model = reference_model
        self.target_model = target_model
        self._reference_model_inspector = reference_model_inspector
        self._target_model_inspector = target_model_inspector
        self._reference_output_name_to_op_map = (
            self._reference_model_inspector.output_name_to_op_map
        )
        self._target_output_names_index = _create_name_index(
            self._target_model_inspector.output_name_to_op_map.keys(),
        )
        self._reference_outputs = {}
        self._target_outputs = {}
        self._visited = set()
        self._num_predict_intermediate_outputs = num_predict_intermediate_outputs

    async def _process_op(
        self,
        op: proto.MIL_pb2.Operation,
        inputs: Dict[str, np.array],
        queue: List[proto.MIL_pb2.Operation],
        compare_outputs: Callable[[proto.MIL_pb2.Operation, np.array, np.array], bool],
        skip_op: Callable[[proto.MIL_pb2.Operation], bool] = skip_op_by_type,
    ) -> _Status:
        def add_deps(curr_op: proto.MIL_pb2.Operation):
            # The operation validation failed; process its dependencies.
            for dep in _retrieve_direct_dependencies(
                op=curr_op,
                output_name_to_op_map=self._reference_output_name_to_op_map,
            ):
                if id(dep) not in self._visited:
                    queue.insert(0, dep)

        self._visited.add(id(op))

        if skip_op(op):
            add_deps(op)
            return _Status.UNKNOWN

        processed_output_names = set(self._reference_outputs.keys())
        dep_names = _retrieve_dependency_names(
            op=op,
            max_dependencies=self._num_predict_intermediate_outputs,
            processed_output_names=processed_output_names,
            output_name_to_op_map=self._reference_output_name_to_op_map,
            skip_op=skip_op,
        )

        op_output_names = [output.name for output in op.outputs]
        output_names = list(op_output_names) + dep_names
        # Retrieve the operation outputs along with selected dependencies.
        # This will reduce the number of prediction calls.
        reference_outputs = await self._reference_model_inspector.retrieve_outputs(
            output_names=output_names,
            inputs=inputs,
            num_predict_intermediate_outputs=self._num_predict_intermediate_outputs,
        )
        self._reference_outputs.update(reference_outputs)

        target_outputs = await _retrieve_target_outputs(
            inspector=self._target_model_inspector,
            reference_output_names=output_names,
            target_output_names_index=self._target_output_names_index,
            inputs=inputs,
            num_predict_intermediate_outputs=self._num_predict_intermediate_outputs,
        )
        self._target_outputs.update(target_outputs)

        self._reference_model_inspector.clear_cached_models()
        self._target_model_inspector.clear_cached_models()

        status = _Status.PASS

        for output_name in op_output_names:
            reference_value = self._reference_outputs.get(output_name, None)
            target_value = self._target_outputs.get(output_name, None)

            self._reference_outputs.pop(output_name, None)
            self._target_outputs.pop(output_name, None)

            _logger.info(f"Comparing values for ({output_name})")

            if reference_value is None:
                _logger.warning(
                    f"Failed to retrieve value for ({output_name}) from reference model.)"
                )
                status = _Status.UNKNOWN
                break

            if target_value is None:
                _logger.warning(f"Failed to retrieve value for ({output_name}) from target model.)")
                status = _Status.UNKNOWN
                break

            if not compare_outputs(op, reference_value, target_value):
                _logger.info(f"Failed comparison for ({output_name})")
                # The operation validation failed; process its dependencies.
                status = _Status.FAIL
                break
            else:
                _logger.info(f"Passed comparison for ({output_name})")

        if status != _Status.PASS:
            add_deps(op)

        return status

    async def find_failing_ops(
        self,
        inputs: Dict[str, np.array],
        compare_outputs: Callable[[proto.MIL_pb2.Operation, np.array, np.array], bool],
        output_names: Optional[List[str]] = None,
        skip_op: Callable[[proto.MIL_pb2.Operation], bool] = skip_op_by_type,
    ) -> List[proto.MIL_pb2.Operation]:
        """
        Identifies operations that produce different outputs in the reference and target models.

        This method compares the outputs of the reference and target models for specified operations,
        identifying those that fail the comparison criteria.

        Parameters
        ----------
        inputs : Dict[str, np.array]
            Input data for the models.

        compare_outputs : Callable[[proto.MIL_pb2.Operation, np.array, np.array], bool])
            A function to compare outputs of an operation between the two models.

        output_names : Optional[List[str]], optional)
            Names of specific outputs to compare. If None, all model outputs are compared. Defaults to None.

        skip_op: Callable[[proto.MIL_pb2.Operation], bool]
            A function that determines if an operation should be skipped.

        Notes
        -----
        - The method uses a breadth-first search strategy to traverse the operation graph.
        - An operation is considered a failure source if it fails comparison while its direct inputs do not.

        Returns:
        List[proto.MIL_pb2.Operation]
            A list of operations that failed the comparison.

        Examples
        --------
        .. sourcecode:: python
            # Load the reference and target models
            reference_model = coremltools.models.MLModel(
                "model.mlpackage", compute_unit=coremltools.ComputeUnit.CPU_ONLY
            )
            target_model = coremltools.models.MLModel(
                "model.mlpackage", compute_unit=coremltools.ComputeUnit.CPU_AND_GPU
            )
            # Create an instance of MLModelComparator
            comparator = (
                coremltools.models.ml_program.experimental.debugging_utils.MLModelComparator(
                    reference_model, target_model
                )
            )
            # Prepare input data
            input_data = {"input_1": np.random.rand(1, 3, 224, 224).astype(np.float32)}

            # Define a custom comparison function
            def compare_outputs(op, reference_output, target_output):
                return np.allclose(reference_output, target_output, rtol=1e-3, atol=1e-3)


            # Find failing operations
            failing_ops = await comparator.find_failing_ops(
                inputs=input_data, compare_outputs=compare_outputs
            )
        """
        if output_names is None:
            output_names = [
                output_name
                for output_name in self._reference_model_inspector.model.output_description
            ]

        queue = []
        all_deps = {}
        for output_name in output_names:
            op = self._reference_output_name_to_op_map.get(output_name, None)
            if op is not None:
                queue.append(op)
                all_deps.update(
                    _get_unique_dependencies(
                        op=op,
                        output_name_to_op_map=self._reference_output_name_to_op_map,
                    )
                )


        result = []
        statuses = {}

        with tqdm.tqdm(total=len(all_deps), desc="\033[1mAnalyzing operations...\033[0m") as pbar:
            while len(queue) > 0:
                op = queue[0]
                status = statuses.get(id(op), None)
                if status is not None:
                    if _is_failure_source(
                        op=op,
                        statuses=statuses,
                        output_name_to_op_map=self._reference_output_name_to_op_map,
                    ):
                        """
                        The operation did not pass comparison check, but its dependencies did.
                        This suggests that the failure likely comes from this operation.
                        """
                        result.append(op)

                    queue.pop(0)
                else:
                    status = await self._process_op(
                        op=op,
                        inputs=inputs,
                        queue=queue,
                        compare_outputs=compare_outputs,
                        skip_op=_skip_op_excluding_output_names(
                            skip_op=skip_op, output_names=output_names
                        ),
                    )
                    output_name = next((output.name for output in op.outputs), "")
                    pbar.set_description(
                        desc=f"\033[1mAnalyzed operation:\033[0m {output_name}, \033[1mtype:\033[0m {op.type}"
                    )
                    pbar.update(n=1)
                statuses[id(op)] = status

        self._visited = set()
        self._reference_outputs = {}
        self._target_outputs = {}

        return result
