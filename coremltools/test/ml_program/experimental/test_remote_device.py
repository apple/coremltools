# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from threading import Timer
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.models._compiled_model import CompiledMLModel
from coremltools.models.ml_program.experimental.remote_device import (
    Device,
    DeviceState,
    DeviceType,
    _DeviceCtlError,
    _DeviceCtlRemoteService,
    _JSONRPCError,
    _JSONRPCRequest,
    _JSONRPCResponse,
    _JSONRPCSocket,
    _ModelRunnerAppBuilder,
    _RemoteMLModelService,
    _TensorDescriptor,
)


class MockDeviceCtlSocket(_JSONRPCSocket):
    def __init__(
        self,
    ) -> None:
        self.models = {}
        self.results = {}
        self.responses = {}
        self.timers = []
        self.device = Device(
            name="Test",
            type=DeviceType.IPHONE,
            identifier="",
            udid="",
            os_version="",
            os_build_number="",
            developer_mode_state="",
            state=DeviceState.CONNECTED,
            session=None,
        )

    def __del__(self):
        for timer in self.timers:
            timer.cancel()

    def _set_response(
        self,
        response: _JSONRPCResponse,
    ) -> None:
        self.responses[response.id] = response

    def set_response_after_random_delay(
        self,
        response: _JSONRPCResponse,
        max_delay_in_seconds: float = 1.0,
    ) -> Timer:
        # Generate a random delay between 0 and ``max_delay_in_seconds`` seconds
        random_delay = random.uniform(0, max_delay_in_seconds)
        # Create a Timer object
        timer = Timer(interval=random_delay, function=lambda: self._set_response(response=response))
        # Start the timer
        timer.start()
        return timer

    def send(
        self,
        request: _JSONRPCRequest,
        resource: Optional[Path],
    ) -> None:
        error = None
        result = None
        response_resource = None

        model_id = request.params["modelID"]
        compute_units = ct.ComputeUnit[request.params["computeUnits"]]
        model_key = frozenset([model_id, compute_units.value])

        def process_model_load(params: Dict[str, Any]):
            nonlocal result
            model = self.models.get(model_key, None)
            duration = 0
            is_cached = True
            if model is None:
                start = time.time()
                model = CompiledMLModel(
                    path=str(resource.absolute()),
                    compute_units=compute_units,
                )
                end = time.time()
                duration = end - start
                is_cached = False

            self.models[model_key] = model
            result = {
                "modelID": model_id,
                "computeUnits": compute_units.value,
                "duration": duration,
                "isCached": is_cached,
            }

        def process_model_unload(params: Dict[str, Any]):
            nonlocal result
            nonlocal error

            model = self.models.pop(model_key, None)
            if model is None:
                message = f"The model with ID {model_id} and {compute_units.value} is not currently loaded."
                error = _JSONRPCError(code=-32001, message=message)
            else:
                result = {
                    "modelID": model_id,
                    "computeUnits": compute_units.value,
                }

        def process_model_predict(params: Dict[str, Any]):
            nonlocal result
            nonlocal error
            nonlocal response_resource

            inputs = params["inputs"]
            model = self.models.get(model_key, None)
            if model is None:
                message = (
                    f"The model with ID {model_id} and {compute_units} is not currently loaded."
                )
                error = _JSONRPCError(code=-32001, message=message)
            else:
                duration = 0
                data_file = tempfile.NamedTemporaryFile("w+b", suffix=".bin", delete=False)
                data_file.seek(0)
                with open(resource, "rb") as fp:
                    model_inputs = {}
                    for name, value in inputs.items():
                        descriptor = _TensorDescriptor.from_dict(value)
                        array = descriptor.to_array(
                            file=fp,
                        )
                        model_inputs[name] = array

                    start = time.time()
                    model_outputs = model.predict(model_inputs)
                    end = time.time()
                    duration = end - start

                    outputs = {}
                    for (name, value) in model_outputs.items():
                        outputs[name] = _TensorDescriptor.from_array(
                            array=value,
                            file=data_file,
                        ).as_dict()
                    data_file.close()
                    response_resource = Path(data_file.name)

                result = {
                    "modelID": model_id,
                    "computeUnits": compute_units.value,
                    "outputs": outputs,
                    "duration": duration,
                }

        if request.method == "MLModelService.Load":
            process_model_load(request.params)

        elif request.method == "MLModelService.Unload":
            process_model_unload(request.params)

        elif request.method == "MLModelService.Prediction":
            process_model_predict(request.params)

        else:
            message = f"Request type {request.method} is not supported by the ModelService."
            error = _JSONRPCError(code=-32001, message=message)
        # Schedule a response.
        response = _JSONRPCResponse(
            id=request.id,
            result=result,
            resource=response_resource,
            error=error,
        )

        timer = self.set_response_after_random_delay(
            response=response,
        )
        self.timers.append(timer)

    def receive(
        self,
        id: str,
    ) -> Optional[_JSONRPCResponse]:
        return self.responses.get(id, None)

    @property
    def is_alive(self) -> bool:
        return True


class TestRemoteMLModelService:
    @staticmethod
    def _get_test_program1():
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 10)), mb.TensorSpec(shape=(1, 10))])
        def prog(x, y):
            return (mb.add(x=x, y=y), mb.sub(x=x, y=y))

        return prog

    @staticmethod
    def _get_test_program2():
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 10)), mb.TensorSpec(shape=(1, 10))])
        def prog(x, y):
            return (mb.sub(x=x, y=y), mb.add(x=x, y=y))

        return prog

    @staticmethod
    def _get_test_models() -> ct.models.MLModel:
        mlmodel1 = ct.convert(
            TestRemoteMLModelService._get_test_program1(),
            minimum_deployment_target=ct.target.iOS16,
            skip_model_load=False,
        )

        mlmodel2 = ct.convert(
            TestRemoteMLModelService._get_test_program2(),
            minimum_deployment_target=ct.target.iOS16,
            skip_model_load=False,
        )

        return [mlmodel1, mlmodel2]

    @staticmethod
    def _get_test_remote_model_services(
        socket: MockDeviceCtlSocket,
        mlmodels: List[ct.models.MLModel],
    ) -> List[_RemoteMLModelService]:
        services = []
        for mlmodel in mlmodels:
            devicectl_service = _DeviceCtlRemoteService(socket=socket)
            service = _RemoteMLModelService(
                compiled_model_path=mlmodel.get_compiled_model_path(),
                compute_units=ct.ComputeUnit.ALL,
                service=devicectl_service,
            )

            services.append(service)

        return services

    @pytest.mark.asyncio
    async def test_model_load(self):
        socket = MockDeviceCtlSocket()
        mlmodels = TestRemoteMLModelService._get_test_models()
        services = TestRemoteMLModelService._get_test_remote_model_services(
            socket=socket,
            mlmodels=mlmodels,
        )

        for service in services:
            await service.load()

    @pytest.mark.asyncio
    async def test_model_unload(self):
        socket = MockDeviceCtlSocket()
        mlmodels = TestRemoteMLModelService._get_test_models()
        services = TestRemoteMLModelService._get_test_remote_model_services(
            socket=socket,
            mlmodels=mlmodels,
        )

        for service in services:
            await service.load()
            await service.unload()

    @pytest.mark.asyncio
    async def test_model_unload_without_load(self):
        socket = MockDeviceCtlSocket()
        mlmodels = TestRemoteMLModelService._get_test_models()
        services = TestRemoteMLModelService._get_test_remote_model_services(
            socket=socket,
            mlmodels=mlmodels,
        )

        for service in services:
            try:
                await service.unload()
            except _DeviceCtlError as e:
                message = f"The model with ID {service.model_id} and {service.compute_units.value} is not currently loaded."
                assert e.message == message

    @pytest.mark.asyncio
    async def test_model_load_multiple_times(self):
        socket = MockDeviceCtlSocket()
        mlmodels = TestRemoteMLModelService._get_test_models()
        services = TestRemoteMLModelService._get_test_remote_model_services(
            socket=socket,
            mlmodels=mlmodels,
        )

        for service in services:
            for _ in range(5):
                await service.load()

    @pytest.mark.asyncio
    async def test_model_predict_without_load(self):
        socket = MockDeviceCtlSocket()
        mlmodels = TestRemoteMLModelService._get_test_models()
        inputs = {
            "x": np.random.rand(1, 10).astype(np.float32),
            "y": np.random.rand(1, 10).astype(np.float32),
        }
        services = TestRemoteMLModelService._get_test_remote_model_services(
            socket=socket,
            mlmodels=mlmodels,
        )

        for service in services:
            try:
                await service.predict(inputs)
            except _DeviceCtlError as e:
                message = f"The model with ID {service.model_id} and {service.compute_units.value} is not currently loaded."
                assert e.message == message

    @pytest.mark.asyncio
    async def test_model_predict(self):
        socket = MockDeviceCtlSocket()
        mlmodels = TestRemoteMLModelService._get_test_models()
        inputs = {
            "x": np.random.rand(1, 10).astype(np.float32),
            "y": np.random.rand(1, 10).astype(np.float32),
        }
        expected_outputs = [mlmodel.predict(inputs) for mlmodel in mlmodels]
        services = TestRemoteMLModelService._get_test_remote_model_services(
            socket=socket,
            mlmodels=mlmodels,
        )

        for service, expected_output in zip(services, expected_outputs):
            await service.load()
            outputs = await service.predict(inputs)
            for name, value in expected_output.items():
                assert np.allclose(value, outputs[name], atol=1e-2)
            await service.unload()


class TestModelRunnerApp:
    @staticmethod
    @pytest.mark.skipif(
        shutil.which("xcodebuild") is None, reason="xcodebuild doesn't exist, skipping test"
    )
    def test_model_runner():
        def can_sudo_without_password():
            try:
                subprocess.run(["sudo", "-n", "true"], check=True, stderr=subprocess.DEVNULL)
                return True
            except subprocess.CalledProcessError:
                return False

        workspace_path = str(
            (_ModelRunnerAppBuilder._get_project_path() / "ModelRunner.xcworkspace").resolve()
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            command = f'xcodebuild test -workspace {workspace_path} -scheme ModelRunnerTests SYMROOT={temp_dir} CONFIGURATION_BUILD_DIR={temp_dir} CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO'
            if can_sudo_without_password():
                # On CI without login `xcodebuild test` fails.
                command = f"sudo login -f local bash -c '{command}'"

            output = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
            )
            try:
                output.check_returncode()
            except subprocess.CalledProcessError as e:
                # Extract stdout and stderr safely
                stdout = e.stdout if e.stdout else "No output"
                stderr = e.stderr if e.stderr else "No error details"

                # Raise a ValueError with detailed information
                raise ValueError(
                    f"Failed to execute tests.\n"
                    f"Command: {e.cmd}\n"
                    f"Exit Code: {e.returncode}\n"
                    f"STDOUT:\n{stdout}\n"
                    f"STDERR:\n{stderr}"
                )
