# Copyright (c) 2025, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import atexit
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

from coremltools import ComputeUnit, proto
from coremltools._deps import _HAS_TORCH

from ...compute_plan import MLComputePlan, MLModelStructure
from ...model import MLModel, MLState
from ...utils import compile_model
from .remote_device import Device, _RemoteMLModelService

if _HAS_TORCH:
    from torch import Tensor

class MLModelAsyncWrapper(ABC):
    @staticmethod
    def init_check(
        spec_or_path: Union["proto.Model_pb2.Model", str],
        weights_dir: str,
        compute_units: ComputeUnit,
        function_name: Optional[str] = None,
        optimization_hints: Optional[Dict[str, Any]] = None,
    ):
        if not isinstance(spec_or_path, str) and not isinstance(
            spec_or_path, proto.Model_pb2.Model
        ):
            raise TypeError(
                'The "spec_or_path" parameter must be of type "str" or "proto.Model_pb2.Model"'
            )

        if not isinstance(weights_dir, str):
            raise TypeError('The "weights_dir" parameter must be of type "str"')

        if not isinstance(compute_units, ComputeUnit):
            raise TypeError('The "compute_units" parameter must be of type "ComputeUnit"')

        if function_name is not None and not isinstance(function_name, str):
            raise TypeError('The "function_name" parameter must be of type "str"')

        if optimization_hints is not None and not isinstance(optimization_hints, Mapping):
            raise TypeError("The 'optimization_hints' must be of mapping type (e.g., dict)")

    """
    An abstract base class for asynchronous wrappers of ``MLModel``.
    """

    def __init__(
        self,
        spec_or_path: Union["proto.Model_pb2.Model", str],
        weights_dir: str,
        compute_units: ComputeUnit = ComputeUnit.ALL,
        function_name: Optional[str] = None,
        optimization_hints: Optional[Dict[str, Any]] = None,
    ):
        MLModelAsyncWrapper.init_check(
            spec_or_path=spec_or_path,
            weights_dir=weights_dir,
            compute_units=compute_units,
            function_name=function_name,
            optimization_hints=optimization_hints,
        )
        self.spec_or_path = spec_or_path
        self.weights_dir = weights_dir
        self.compute_units = compute_units
        self.function_name = function_name
        self.optimization_hints = optimization_hints
        self._temp_asset_path = None

    @abstractmethod
    async def load(self):
        """
        Asynchronously loads the ``MLModel``.
        """
        pass

    @abstractmethod
    async def unload(self):
        """
        Asynchronously unloads the ``MLModel``.
        """
        pass

    @abstractmethod
    async def predict(
        self,
        inputs: Dict[str, np.array],
        state: Optional[MLState],
    ) -> Dict[str, np.array]:
        """
        Asynchronously performs predictions using the loaded ``MLModel``.
        """
        pass

    @property
    def temp_asset_path(self) -> Optional[str]:
        return self._temp_asset_path

    @temp_asset_path.setter
    def temp_asset_path(self, value: Optional[str]):
        if self._temp_asset_path != value:
            self._temp_asset_path = value
            atexit.register(MLModelAsyncWrapper._remove_directory, value)

    @abstractmethod
    def make_state_if_needed(self) -> Optional["MLState"]:
        pass

    @abstractmethod
    async def retrieve_compute_plan(self) -> MLComputePlan:
        """
        Asynchronously retrieves the compute plan for the loaded ``MLModel``.
        """
        pass

    @property
    @abstractmethod
    def load_duration_in_nano_seconds(self) -> Optional[int]:
        """
        Retrieves the duration of the model loading process in nanoseconds.
        """
        pass

    @property
    @abstractmethod
    def last_predict_duration_in_nano_seconds(self) -> Optional[int]:
        """
        Retrieves the duration of the last predict operation in nanoseconds.
        This method returns the time taken for the most recent prediction made by
        the model, measured in nanoseconds.
        """
        pass

    @staticmethod
    def _remove_directory(path: Optional[str]):
        if path is not None and os.path.exists(path):
            shutil.rmtree(path)

    def cleanup(self):
        MLModelAsyncWrapper._remove_directory(self.temp_asset_path)

    def __del__(self):
        self.cleanup()

    @staticmethod
    def from_spec_or_path(
        spec_or_path: Union["proto.Model_pb2.Model", str],
        weights_dir: str,
        compute_units: ComputeUnit = ComputeUnit.ALL,
        function_name: Optional[str] = None,
        optimization_hints: Optional[Dict[str, Any]] = None,
        device: Device = None,
    ) -> "MLModelAsyncWrapper":
        """
        Creates an MLModelAsyncWrapper instance from a model specification or model path.

        This static method constructs an ``MLModelAsyncWrapper`` object based on the provided
        model specification and additional parameters.

        If the device parameter is``None``, the model is loaded on the local system. Otherwise, it
        is loaded on the specified device.

        Parameters
        ----------
        spec_or_path : Union["proto.Model_pb2.Model", str]
            Either a protobuf specification of the model (``proto.Model_pb2.Model``) or a string
            representing the file path to the model (``mlpackage``).

        weights_dir: str
            The model weights directory.

        function_name : Optional[str]
            The function name. Defaults to the model's function name.

        compute_units : coremltools.ComputeUnit:
            The compute units to use. Defaults to the model's compute unit.

        optimization_hints : Optional[Dict[str, Any]]
            Keys are the names of the optimization hint, either 'reshapeFrequency' or 'specializationStrategy'.
            Values are enumeration values of type ``coremltools.ReshapeFrequency`` or ``coremltools.SpecializationStrategy``.

        device: Device
            The device on which the model will execute.

        Returns
        -------
        MLModelAsyncWrapper
            An instance of MLModelAsyncWrapper.
        """
        if device is None:
            return LocalMLModelAsyncWrapper(
                spec_or_path=spec_or_path,
                weights_dir=weights_dir,
                compute_units=compute_units,
                function_name=function_name,
                optimization_hints=optimization_hints,
            )
        else:
            return RemoteMLModelAsyncWrapper(
                spec_or_path=spec_or_path,
                weights_dir=weights_dir,
                compute_units=compute_units,
                function_name=function_name,
                optimization_hints=optimization_hints,
                device=device,
            )


class LocalMLModelAsyncWrapper(MLModelAsyncWrapper):
    def __init__(
        self,
        spec_or_path: Union["proto.Model_pb2.Model", str],
        weights_dir: str,
        compute_units: ComputeUnit = ComputeUnit.ALL,
        function_name: Optional[str] = None,
        optimization_hints: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            spec_or_path=spec_or_path,
            weights_dir=weights_dir,
            compute_units=compute_units,
            function_name=function_name,
            optimization_hints=optimization_hints,
        )
        self._model = None

    async def load(self):
        model = MLModel(
            model=self.spec_or_path,
            weights_dir=self.weights_dir,
            compute_units=self.compute_units,
            function_name=self.function_name,
            optimization_hints=self.optimization_hints,
        )

        if model is None or model.__proxy__ is None:
            raise ValueError("Failed to load model")

        self._model = model
        self.temp_asset_path = self._model.package_path

    def make_state_if_needed(self) -> Optional["MLState"]:
        if self._model._is_stateful():
            return self._model.make_state()

        return None

    async def predict(
        self,
        inputs: Dict[str, np.array],
        state: Optional[MLState] = None,
    ) -> Dict[str, np.array]:
        if self._model is None:
            await self.load()

        return self._model.predict(inputs, state=state)

    async def retrieve_compute_plan(self) -> MLComputePlan:
        if self._model is None:
            await self.load()

        compile_model_path = self._model.get_compiled_model_path()
        if compile_model_path is not None:
            return MLComputePlan.load_from_path(
                path=compile_model_path,
                compute_units=self._model.compute_unit,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            model_package_path = self._model.package_path
            if model_package_path is None:
                model_package_path = str((Path(temp_dir) / "model.mlpackage").resolve())
                self._model.save(model_package_path)

            compiled_model_path = str((Path(temp_dir) / "model.mlmodelc").resolve())
            compile_model(
                model=model_package_path,
                destination_path=str(compiled_model_path.resolve()),
            )

            return MLComputePlan.load_from_path(
                path=compile_model_path,
                compute_units=self._model.compute_unit,
            )


    async def unload(self):
        self._model = None

    @property
    def load_duration_in_nano_seconds(self) -> Optional[int]:
        return self._model.load_duration_in_nano_seconds if self._model is not None else None

    @property
    def last_predict_duration_in_nano_seconds(self) -> Optional[int]:
        return (
            self._model.last_predict_duration_in_nano_seconds if self._model is not None else None
        )


class RemoteMLModelAsyncWrapper(MLModelAsyncWrapper):
    """
    A concrete implementation of the ``MLModelAsyncWrapper`` for a remote ``MLModel``.
    """

    def __init__(
        self,
        spec_or_path: Union["proto.Model_pb2.Model", str],
        weights_dir: str,
        device: Device,
        compute_units: ComputeUnit = ComputeUnit.ALL,
        function_name: Optional[str] = None,
        optimization_hints: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(device, Device):
            raise TypeError("The 'device' must be of Device type")

        super().__init__(
            spec_or_path=spec_or_path,
            weights_dir=weights_dir,
            compute_units=compute_units,
            function_name=function_name,
            optimization_hints=optimization_hints,
        )
        self.device = device
        self._remote_service = None
        self._compiled_model_path = None

    async def load(self):
        working_directory = Path(tempfile.mkdtemp())
        self.temp_asset_path = str(working_directory.resolve())

        if isinstance(self.spec_or_path, str):
            model_path = self.spec_or_path
        else:
            model = MLModel(
                model=self.spec_or_path,
                weights_dir=self.weights_dir,
                compute_units=self.compute_units,
                function_name=self.function_name,
                optimization_hints=self.optimization_hints,
                skip_model_load=True,
            )

            model_path = str((working_directory / "model.mlpackage").resolve())
            model.save(model_path)

        compiled_model_path = working_directory / "model.mlmodelc"
        compile_model(
            model=model_path,
            destination_path=str(compiled_model_path.resolve()),
        )

        remote_service = await _RemoteMLModelService.load_on_device(
            device=self.device,
            compiled_model_path=compiled_model_path,
            compute_units=self.compute_units,
            function_name=self.function_name,
            optimization_hints=self.optimization_hints,
        )

        self._compiled_model_path = compiled_model_path
        self._remote_service = remote_service

    def make_state_if_needed(self) -> Optional[MLState]:
        return None

    async def predict(
        self,
        inputs: Dict[str, np.array],
        state: Optional[MLState] = None,
    ) -> Dict[str, np.array]:
        def convert_to_np_array(input: Any):
            if isinstance(input, np.ndarray):
                return input
            elif _HAS_TORCH and isinstance(input, Tensor):
                return input.detach().numpy()
            else:
                return np.array(input)

        if self._remote_service is None:
            await self.load()

        return await self._remote_service.predict(
            inputs={key: convert_to_np_array(value) for key, value in inputs.items()}
        )

    async def retrieve_compute_plan(self) -> MLComputePlan:
        try:
            from .compute_plan_utils import _MLComputePlanRemoteProxy
        except Exception as e:
            error_message = f"Error importing _MLComputePlanRemoteProxy: {str(e)}"
            raise ValueError(
                f"Remote device functionality for retrieving the compute plan is unavailable. {error_message}"
            )

        if self._remote_service is None:
            await self.load()

        compute_plan = await self._remote_service.retrieve_compute_plan()
        model_structure = MLModelStructure.load_from_path(
            compiled_model_path=str(self._compiled_model_path.resolve()),
        )

        remote_proxy = _MLComputePlanRemoteProxy(
            device=self.device,
            compute_plan=compute_plan,
            model_structure=model_structure,
        )

        return MLComputePlan(proxy=remote_proxy)

    async def unload(self):
        await self._remote_service.unload()

    @property
    def load_duration_in_nano_seconds(self) -> Optional[int]:
        return (
            self._remote_service.load_duration_in_nano_seconds
            if self._remote_service is not None
            else None
        )

    @property
    def last_predict_duration_in_nano_seconds(self) -> Optional[int]:
        return (
            self._remote_service.last_predict_duration_in_nano_seconds
            if self._remote_service is not None
            else None
        )
