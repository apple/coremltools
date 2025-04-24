# Copyright (c) 2025, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import gc
import random
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from coremltools import ComputeUnit, proto

from ...compute_plan import MLComputePlanDeviceUsage
from ...model import MLModel
from .async_wrapper import MLModelAsyncWrapper
from .model_structure_path import (
    ModelStructurePath,
    map_model_spec_to_path,
    map_model_structure_to_path,
)
from .remote_device import Device

def _convert_nanoseconds_to_milliseconds(nanoseconds: int):
    return nanoseconds / 1000000


def _gen_random_multiarray_value(type: "proto.FeatureTypes_pb2.ArrayFeatureType") -> np.array:
    data_types = {
        proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.FLOAT32: np.float32,
        proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.FLOAT16: np.float16,
        proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.DOUBLE: np.float64,
        proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.INT32: np.int32,
    }

    np_data_type = data_types.get(type.dataType, None)
    if np_data_type is None:
        supported_types = ", ".join(str(t) for t in data_types.keys())
        raise ValueError(
            f"Unsupported data type: {type.dataType}. Supported types are: {supported_types}"
        )

    shape = tuple(type.shape)
    if np_data_type == np.int32:
        return np.random.randint(0, 10, shape)

    return np.random.random(shape)


def _gen_random_image_value(type: "proto.FeatureTypes_pb2.ImageFeatureType") -> np.array:
    try:
        from PIL import Image
    except:
        raise ValueError(
            "The Python Imaging Library (PIL) is required but not installed. "
            "Please install it by running 'pip install pillow'."
        )

    def gen_random_pixel_value(
        colorSpace: "proto.FeatureTypes_pb2.ImageFeatureType.ColorSpace",
    ) -> Any:
        if (
            colorSpace == proto.FeatureTypes_pb2.ImageFeatureType.RGB
            or colorSpace == proto.FeatureTypes_pb2.ImageFeatureType.BGR
        ):
            return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        elif colorSpace == proto.FeatureTypes_pb2.ImageFeatureType.GRAYSCALE:
            return random.randint(0, 255)
        elif colorSpace == proto.FeatureTypes_pb2.ImageFeatureType.GRAYSCALE_FLOAT16:
            return random.random()
        else:
            raise ValueError(f"Unsupported image color space={colorSpace}")

    def get_image_type(
        colorSpace: "proto.FeatureTypes_pb2.ImageFeatureType.ColorSpace",
    ) -> str:
        if (
            colorSpace == proto.FeatureTypes_pb2.ImageFeatureType.RGB
            or colorSpace == proto.FeatureTypes_pb2.ImageFeatureType.BGR
        ):
            return "RGB"
        elif colorSpace == proto.FeatureTypes_pb2.ImageFeatureType.GRAYSCALE:
            return "L"
        elif colorSpace == proto.FeatureTypes_pb2.ImageFeatureType.GRAYSCALE_FLOAT16:
            return "F"
        else:
            raise ValueError(f"Unsupported image color space: {colorSpace}")

    colorSpace = type.colorSpace
    # Create a new image with the specified size
    image = Image.new(get_image_type(colorSpace=colorSpace), (type.width, type.height))

    # Get the pixel data
    pixels = image.load()

    # Fill the image with random BGR values
    for x in range(type.width):
        for y in range(type.height):
            pixels[x, y] = gen_random_pixel_value(colorSpace)

    return image


def _gen_random_feature_value(
    type: "proto.FeatureTypes_pb2.FeatureType",
) -> Any:
    feature_type = type.WhichOneof("Type")
    if feature_type == "int64Type":
        return random.randint(1, 10)

    elif feature_type == "doubleType":
        return random.random()

    elif feature_type == "stringType":
        return "".join(random.choices(string.ascii_letters + string.digits, k=10))

    elif feature_type == "multiArrayType":
        return _gen_random_multiarray_value(type=type.multiArrayType)

    elif type == "imageType":
        return _gen_random_image_value(type=type.imageType)

    else:
        raise ValueError(f"Unsupported feature type: {type}")


def _gen_random_inputs(
    model_description: "proto.Model_pb2.ModelDescription",
) -> Dict[str, Any]:

    result = {}
    input = model_description.input
    for feature_description in input:
        name = feature_description.name
        type = feature_description.type
        result[name] = _gen_random_feature_value(type=type)

    return result


class MLModelBenchmarker:
    """
    A class for benchmarking an MLModel.

    This class provides functionality to measure and analyze the performance of an MLModel,
    including loading time, prediction time, and individual operation execution times.
    """

    @dataclass(frozen=True)
    class Statistics:
        minimum: float
        maximum: float
        average: float
        std_dev: float
        median: float

        @staticmethod
        def from_values(values: List[float]) -> Optional["MLModelBenchmarker.Statistics"]:
            if len(values) == 0:
                return None

            minimum = float(np.min(values))
            maximum = float(np.max(values))
            average = float(np.mean(values))
            std_dev = float(np.std(values))
            median = float(np.median(values))

            return MLModelBenchmarker.Statistics(
                minimum=minimum,
                maximum=maximum,
                average=average,
                std_dev=std_dev,
                median=median,
            )

    @dataclass(frozen=True)
    class Measurement:
        statistics: Optional["MLModelBenchmarker.Statistics"]
        samples: List[float]

        @staticmethod
        def from_samples(samples: List[float]) -> "MLModelBenchmarker.Measurement":
            return MLModelBenchmarker.Measurement(
                statistics=MLModelBenchmarker.Statistics.from_values(values=samples),
                samples=samples,
            )

        def add_samples(self, samples: List[float]) -> "MLModelBenchmarker.Measurement":
            if len(samples) == 0:
                return self

            if len(self.samples) == 0:
                return MLModelBenchmarker.Measurement.from_samples(samples=samples)

            if len(self.samples) != len(samples):
                raise ValueError(
                    f"Number of samples must match existing samples. "
                    f"Expected {len(self.samples)}, but got {len(samples)}."
                )

            return MLModelBenchmarker.Measurement.from_samples(
                samples=[sample1 + sample2 for sample1, sample2 in zip(self.samples, samples)]
            )

        @property
        def sort_key(self) -> Tuple[bool, Optional[float]]:
            return (
                self.statistics is not None,
                self.statistics.median if self.statistics is not None else None,
            )

    @dataclass
    class OperationExecutionInfo:
        spec: proto.MIL_pb2.Operation
        path: ModelStructurePath
        compute_device_usage: Optional[MLComputePlanDeviceUsage]
        measurement: "MLModelBenchmarker.Measurement"

    def __init__(
        self,
        model: MLModel,
        device: Optional[Device] = None,
    ) -> None:
        self.model = model
        self.model_spec = model.get_spec()
        self.device = device
        self._loaded_model = None

    @property
    def compute_units(self) -> ComputeUnit:
        return self.model.compute_unit

    @compute_units.setter
    def compute_units(self, compute_units: ComputeUnit):
        if self.model.compute_unit != compute_units:
            self.model = MLModel(
                model=self.model_spec,
                compute_units=compute_units,
                weights_dir=self.model.weights_dir,
                function_name=self.model.function_name,
                optimization_hints=self.model.optimization_hints,
            )

            self._loaded_model = None

    @staticmethod
    async def _create_loaded_model(
        model: MLModel,
        device: Optional[Device],
    ) -> MLModelAsyncWrapper:
        model_wrapper = MLModelAsyncWrapper.from_spec_or_path(
            spec_or_path=model.get_spec(),
            weights_dir=model.weights_dir,
            compute_units=model.compute_unit,
            function_name=model.function_name,
            optimization_hints=model.optimization_hints,
            device=device,
        )

        await model_wrapper.load()
        return model_wrapper

    async def benchmark_load(
        self,
        iterations: int = 1,
    ) -> "MLModelBenchmarker.Measurement":
        """
        Measures the loading time of the model.

        This method creates and loads the model multiple times, measuring the duration
        of each load operation. It then unloads the model and performs garbage collection
        after each iteration to ensure consistent measurements.

        Parameters
        ----------
        iterations: int
                The number of times to load the model. Defaults to 1.

        Returns
        -------
        MLModelBenchmarker.Measurement
            A Measurement object containing statistics and samples of the load durations in milliseconds.

        """
        if iterations < 1:
            raise ValueError("The number of iterations must be at least 1. Received: {iterations}")

        load_durations = []
        for _ in range(0, iterations):
            loaded_model = await MLModelBenchmarker._create_loaded_model(
                model=self.model,
                device=self.device,
            )

            load_duration = loaded_model.load_duration_in_nano_seconds
            if load_duration is None:
                raise ValueError(
                    "Failed to retrieve model load duration. The model may not have been loaded properly or the timing information is unavailable."
                )

            load_durations.append(_convert_nanoseconds_to_milliseconds(load_duration))

            if self._loaded_model is None:
                self._loaded_model = loaded_model
            else:
                await loaded_model.unload()
                del loaded_model
                gc.collect()

        return MLModelBenchmarker.Measurement.from_samples(
            samples=load_durations,
        )

    @staticmethod
    async def _benchmark_model_predict(
        model: MLModel,
        iterations: int,
        inputs: Dict[str, Any],
        warmup: bool,
    ) -> "MLModelBenchmarker.Measurement":
        predict_durations = []
        model_state = None
        iterations = iterations + 1 if warmup else iterations
        for iteration in range(0, iterations):
            model_state = model.make_state_if_needed()
            _ = await model.predict(inputs, state=model_state)
            predict_duration = model.last_predict_duration_in_nano_seconds
            if predict_duration is None:
                raise ValueError(
                    "Failed to retrieve prediction duration, timing information is unavailable."
                )

            if iteration > 0 or iteration == 0 and not warmup:
                predict_durations.append(_convert_nanoseconds_to_milliseconds(predict_duration))

        return MLModelBenchmarker.Measurement.from_samples(
            samples=predict_durations,
        )

    async def benchmark_predict(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        iterations: int = 1,
        warmup: bool = False,
    ) -> "MLModelBenchmarker.Measurement":
        """
        Measures the prediction time of the model.

        This method loads the model, then runs predictions multiple times, measuring
        the duration of each prediction. It supports an optional warmup iteration.

        Parameters
        ----------
        inputs : Optional[Dict[str, Any]]
            The input data for the model prediction. If None, random input data will be generated.

        iterations: int
            The number of prediction iterations to run. Defaults to 1.

        warmup: bool
            Whether to perform a warmup iteration. Defaults to False.

        Returns
        -------
        MLModelBenchmarker.Measurement
            A Measurement object containing statistics and
            samples of the prediction durations in milliseconds.

        Raises:
            ValueError: If the number of iterations is less than 1.

        Note:
            This method is asynchronous and should be awaited when called.
            The warmup iteration, if enabled, is not included in the returned measurements.
        """
        if iterations < 1:
            raise ValueError("The number of iterations must be at least 1. Received: {iterations}")

        if self._loaded_model is None:
            loaded_model = await MLModelBenchmarker._create_loaded_model(
                model=self.model,
                device=self.device,
            )

            self._loaded_model = loaded_model

        if inputs is None:
            inputs = _gen_random_inputs(model_description=self.model_spec.description)

        return await MLModelBenchmarker._benchmark_model_predict(
            model=self._loaded_model,
            iterations=iterations,
            inputs=inputs,
            warmup=warmup,
        )

    async def benchmark_operation_execution(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        iterations: int = 1,
        warmup: bool = False,
    ) -> List[OperationExecutionInfo]:
        """
        Measures the execution time of individual operations in the model.

        This method loads the model, runs predictions, and retrieves the execution time
        of each operation within the model.

        Parameters
        ----------
        inputs : Optional[Dict[str, Any]]
            The input data for the model prediction. If None, random input data will be generated.

        iterations: int
            The number of prediction iterations to run. Defaults to 1.

        warmup: bool
            Whether to perform a warmup iteration. Defaults to False.

        Returns
        -------
        List[OperationExecutionInfo]
            A list of OperationExecutionInfo objects, each containing
            details about an operation's execution, sorted by execution time in descending order.

        Notes
        -----
        - The returned list is sorted by execution time, with the most time-consuming operations first.
        - Execution times are estimated based on the overall prediction time and the model's compute plan.
        """

        predict_measurement = await self.benchmark_predict(
            inputs=inputs,
            iterations=iterations,
            warmup=warmup,
        )

        compute_plan = await self._loaded_model.retrieve_compute_plan()

        path_to_op_spec_map = {
            path: spec for spec, path in map_model_spec_to_path(model_spec=self.model_spec)
        }

        path_to_op_structure_map = {
            path: structure
            for structure, path in map_model_structure_to_path(
                model_structure=compute_plan.model_structure
            )
        }

        result = []
        for path, op_structure in path_to_op_structure_map.items():
            op_spec = path_to_op_spec_map.get(path, None)
            if op_spec is None:
                continue

            compute_device_usage = compute_plan.get_compute_device_usage_for_mlprogram_operation(
                op_structure
            )

            estimated_cost = compute_plan.get_estimated_cost_for_mlprogram_operation(op_structure)

            durations = (
                [sample * estimated_cost.weight for sample in predict_measurement.samples]
                if estimated_cost is not None
                else None
            )

            measurement = MLModelBenchmarker.Measurement.from_samples(
                samples=durations if durations is not None else []
            )

            execution_info = MLModelBenchmarker.OperationExecutionInfo(
                spec=op_spec,
                path=path,
                compute_device_usage=compute_device_usage,
                measurement=measurement,
            )

            result.append(execution_info)

        return sorted(result, key=lambda value: value.measurement.sort_key, reverse=True)
