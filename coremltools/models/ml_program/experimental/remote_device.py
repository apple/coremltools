# Copyright (c) 2025, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import asyncio as _asyncio
import json as _json
import os as _os
import shutil as _shutil
import subprocess as _subprocess
import tempfile as _tempfile
import uuid as _uuid
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from collections.abc import Mapping as _Mapping
from collections.abc import Sequence as _Sequence
from dataclasses import dataclass as _dataclass
from enum import Enum as _Enum
from pathlib import Path as _Path
from typing import Any as _Any
from typing import BinaryIO as _BinaryIO
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Type as _Type
from typing import Union as _Union

import numpy as _np
import tqdm as _tqdm

from coremltools import ComputeUnit as _ComputeUnit
from coremltools import _logger

from .model_structure_path import ModelStructurePath


@_dataclass(frozen=True)
class AppSigningCredentials:
    """
    Represents the credentials required for signing an iOS application.

    This class encapsulates the essential information needed for code signing an iOS app,
    including the development team identifier and optionally the bundle identifier.

    Attributes
    ----------
    development_team : Optional[str]
        The development team identifier associated with the Apple Developer account.

    provisioning_profile_uuid : Optional[str]
        The UUID of the provisioning profile. This is used to specify which provisioning
        profile should be applied during the code-signing process.

    bundle_identifier : Optional[str]
        The bundle identifier for the application. This is an optional parameter that, if provided,
        should be in the format of a reverse domain name (e.g., "com.example.app"). If None, the
        bundle identifier from the project's settings will be used.

    Notes
    -----
       Either ``provisioning_profile_uuid`` or ``development_team`` must be provided:
        - If ``provisioning_profile_uuid`` is provided, then ``bundle_identifier`` must match
          the one defined in the provisioning profile.

        - If ``development_team`` is provided, Xcode will automatically create and manage
          a provisioning profile during the build process.

    """

    development_team: _Optional[str] = ""
    provisioning_profile_uuid: _Optional[str] = None
    bundle_identifier: _Optional[str] = None

class DeviceType(_Enum):
    """
    Enumeration of device types.

    Attributes
    ----------
    MAC : str
       Represents a Mac device.

    IPHONE : str
       Represents an iPhone device.

    IPAD : str
       Represents an iPad device

    APPLETV : str
       Represents an Apple TV device.

    WATCH : str
       Represents an Apple Watch device.

    HOMEPOD : str
       Represents a HomePod device.

    UNKNOWN : str
        Represents an unknown device type.

    """

    MAC = "mac"
    IPHONE = "iphone"
    IPAD = "ipad"
    APPLETV = "apppletv"
    WATCH = "applewatch"
    HOMEPOD = "homepod"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value):
        return DeviceType.UNKNOWN


class DeviceState(_Enum):
    """
    Enumeration of device states.

    Attributes
    ----------
    CONNECTING : str
        The device is connecting.

    CONNECTED : str
        The device is connected.

    AVAILABLE : str
        The device is available for use.

    UNAVAILABLE : str
        The device is not available for use.

    DISCONNECTED :str
        The device is disconnected.

    UNKNOWN : str
        Represents an unknown device state.
    """

    CONNECTING = "connecting"
    CONNECTED = "connected"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DISCONNECTED = "disconnected"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value):
        return DeviceState.UNKNOWN


@_dataclass(frozen=True)
class Device:
    """
    Represents a device.

    Attributes
    -----------
    name : str
        The name of the device.

    type : DeviceType
        The type of the device.

    identifier : str
        A unique identifier for the device.

    udid : str
        The device identifier.

    os_version : str
        The operating system version of the device.

    os_build_number : str
        The build number of the os.

    developer_mode_state : str
        The state of developer mode on the device.
    """

    name: str
    type: DeviceType
    identifier: str
    udid: str
    os_version: str
    os_build_number: str
    developer_mode_state: str
    state: DeviceState
    session: _Optional[_Type["_AppSession"]]

    @staticmethod
    def get_devices() -> _List[_Type["Device"]]:
        """
        Retrieve a list of available devices.

        This method fetches and returns a list of all accessible devices
        using the DeviceCtl utility.

        Returns
        -------
        List[Device]
            A list of Device objects representing the available devices.
        """
        return _DeviceCtl.get_devices()

    @staticmethod
    def get_connected_devices(device_type: DeviceType) -> _List[_Type["Device"]]:
        """
        Retrieve a list of connected devices of a specified type.

        This function uses the DeviceCtl utility to fetch all accessible devices and filters them
        based on the following criteria:
        1. The device is in a connected state
        2. The device matches the specified device_type

        Parameters
        ----------
        device_type : DeviceType
            The type of device to filter for (e.g., iPhone, iPad, Apple TV).

        Returns
        -------
        List[Device]
        """

        def is_device_connected(device: Device) -> bool:
            return device.state == DeviceState.CONNECTED and device.type == device_type

        return [device for device in Device.get_devices() if is_device_connected(device=device)]

    def _install_and_launch_app(
        self,
        app_path: _Path,
        working_directory: _Path,
    ) -> _Type["_AppSession"]:
        """
         Install and launch an application on the device.

         This method takes the path to an application, installs it on the device
         associated with this instance, and then launches the application.

        Parameters
        ----------
         app_path : Path
             The path to the application file or directory to be installed.
             This should be a valid path object pointing to the app bundle.

        Returns
        -------
        AppSession
             An AppSession object representing the launched application session.
             This object typically contains:
             - process: The subprocess object for the running application.
             - device: The Device object on which the app was installed and launched.
             - bundle_identifier: The bundle ID of the installed and launched application.
             - database_uuid: The database UUID of the installed application.

        """
        return _DeviceCtl.install_and_launch_app(
            device=self,
            app_path=app_path,
            working_directory=working_directory,
        )

    def _launch_app(
        self,
        app_path: _Path,
        working_directory: _Path,
    ) -> _Type["_AppSession"]:
        """
         Install and launch an application on the device.

         This method takes the path to an application, installs it on the device
         associated with this instance, and then launches the application.

         Parameters
         ----------
         app_path : Path
             The path to the application file or directory to be installed.
             This should be a valid path object pointing to the app bundle.

        Returns
        -------
        AppSession
             An AppSession object representing the launched application session.
             This object typically contains:
             - process: The subprocess object for the running application.
             - device: The Device object on which the app was installed and launched.
             - bundle_identifier: The bundle ID of the installed and launched application.
             - database_uuid: The database UUID of the installed application.

        """
        return _DeviceCtl.launch_app(
            device=self,
            app_path=app_path,
            working_directory=working_directory,
        )

    def _build_and_launch_model_runner_app(
        self,
        build_directory: _Path,
        working_directory: _Path,
        credentials: AppSigningCredentials = AppSigningCredentials(),
        clean: bool = False,
    ) -> _Type["_AppSession"]:
        def command_exists(command):
            """Check if a command exists in the system's PATH."""
            return _shutil.which(command) is not None

        if not command_exists("xcodebuild"):
            raise ValueError(
                "The 'xcodebuild' command is required. Please ensure that Xcode Command Line Tools are installed and accessible."
            )

        app_path = _ModelRunnerAppBuilder.build(
            device=self,
            build_directory=build_directory,
            credentials=credentials,
            clean=clean,
        )

        return self._install_and_launch_app(
            app_path=app_path,
            working_directory=working_directory,
        )

    async def prepare_for_model_debugging(
        self,
        credentials: _Optional[AppSigningCredentials] = None,
        working_directory: _Optional[_Path] = None,
        clean: bool = False,
    ) -> _Type["Device"]:
        """
        Prepares the device for model debugging by building and launching the model runner app.

        This method checks if the device is in the correct state for debugging, sets up the working
        directory, builds the ``modelrunner`` application, and launches it on the device.

        Parameters
        ----------
        credentials : Optional[AppSigningCredentials
            The credentials required for signing ``modelrunner`` application.

        working_directory: Optional[Path]
            The directory utilized for storing files required for building and communicating
            with the ``modelrunner`` application. If None, a temporary director will be created.
            Defaults to None.

        clean : bool
            If True, performs a clean build. Defaults to False.

        Returns
        -------
        Device
            A new Device instance with the prepared session.
        """

        async def report_progress(
            future: _asyncio.Future,
            message: str,
            delay: float = 1.0,
        ):
            with _tqdm.tqdm(desc=message, bar_format="\033[1m{desc} {elapsed}s\033[0m") as pbar:
                while not future.done():
                    await _asyncio.sleep(delay)
                    pbar.update(0)

        if not (self.state == DeviceState.CONNECTED):
            raise ValueError(
                "The device cannot be used for model debugging. "
                f"Current state: {self.state}. "
                "Required: State must be CONNECTED."
            )

        if working_directory is None:
            working_directory = _tempfile.mkdtemp()

        if isinstance(working_directory, str):
            working_directory = _Path(working_directory)

        if not working_directory.is_dir():
            raise ValueError(
                f"The specified working directory '{working_directory}' is invalid. Please ensure it exists and is a directory."
            )

        if credentials is None:
            credentials = AppSigningCredentials(
                development_team="",
                bundle_identifier=None,
                provisioning_profile_uuid=None,
            )

        build_directory = working_directory / "build"
        loop = _asyncio.get_event_loop()

        session_future = loop.run_in_executor(
            None,
            self._build_and_launch_model_runner_app,
            working_directory,
            build_directory,
            credentials,
            clean,
        )

        progress_task = _asyncio.create_task(
            report_progress(
                future=session_future,
                message=f"Setting up '{self.name}' for model debugging. This may take a few moments...",
            )
        )

        try:
            session, _ = await _asyncio.gather(session_future, progress_task)
        finally:
            progress_task.cancel()  # Ensure progress task is cancelled

        return Device(
            name=self.name,
            type=self.type,
            identifier=self.identifier,
            udid=self.udid,
            state=self.state,
            os_version=self.os_version,
            os_build_number=self.os_build_number,
            developer_mode_state=self.developer_mode_state,
            session=session,
        )

@_dataclass(frozen=True)
class _DataTransferInfo:
    source: str
    destination: str


@_dataclass(frozen=True)
class _AppInstallationInfo:
    """
    Represents the result of an application installation.
    """

    bundle_identifier: str
    database_uuid: str


class _AppSession:
    """
    Represents a session for an application running on a device.

    This class manages the lifecycle of an application session, including
    file transfers to and from the application.
    """

    def __init__(
        self,
        device: Device,
        process: _subprocess.Popen,
        installation_info: _AppInstallationInfo,
        working_directory: _Path,
    ) -> None:
        """
        Initialize an AppSession.

        Parameters
        ----------
        device : Device
            The device on which the application is running.

        process : subprocess.Popen
            The subprocess representing the launched application.

        installation_info : str
            The application installation info.

        working_directory: Path
            The directory utilized for storing files required for communication.

        Returns
        -------
        AppSession
            An AppSession object representing the running instance
        """
        self.device = device
        self.installation_info = installation_info
        self._process = process
        self.working_directory = working_directory

    def stop(self):
        """
        Stop the application session.

        This method terminates the subprocess associated with the application.
        """
        self._process.terminate()

    def is_alive(self) -> bool:
        """
        Check if the application session is still active.

        Returns
        -------
        bool:
            True if the application is still running, False otherwise.
        """
        poll = self._process.poll()
        return poll is None

    @property
    def bundle_identifier(self):
        """
        Bundle identifier of the application.

        Returns
        -------
        str:
           Returns the bundle identifier of the application.
        """
        return self.installation_info.bundle_identifier

    def send_file_to_app(
        self,
        source_path: _Path,
        destination: str,
    ) -> _DataTransferInfo:
        """
        Send a file to the application on the device.

        Parameters
        ----------
        source_path : Path
            The path to the file to be sent.

        destination : str
            The destination path within the application's sandbox.

        Returns
        -------
        DataTransferInfo
            Information about the file transfer.
        """
        return _DeviceCtl.send_file_to_app(
            bundle_identifier=self.bundle_identifier,
            source_path=source_path,
            destination=destination,
            device=self.device,
        )

    def receive_file_from_app(
        self,
        destination_path: _Path,
        source: str,
    ) -> _DataTransferInfo:
        """
        Receive a file from the application on the device.

        Parameters
        ----------
        destination_path : Path
            The path where the received file will be saved.

        source : str
            The source path of the file within the application's sandbox.

        Returns
        -------
        DataTransferInfo
            Information about the file transfer.
        """
        return _DeviceCtl.receive_file_from_app(
            bundle_identifier=self.bundle_identifier,
            source=source,
            destination_path=destination_path,
            device=self.device,
        )

class _DeviceCtlError(Exception):
    """Custom exception for device ctl errors."""

    def __init__(self, message, error_code=None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

    def __str__(self):
        return f"[Error Code {self.error_code}] {self.message}"


class _DeviceCtl:
    """
    The DeviceCtl class provides a set of methods to perform various operations
    on devices, such as listing connected devices, installing and launching applications,
    transferring files, and more. It serves as a high-level interface to the 'xcrun devicectl'
    and related command-line tools.
    """

    _validation_done = False

    @staticmethod
    def _validate():
        def command_exists(command):
            """Check if a command exists in the system's PATH."""
            return _shutil.which(command) is not None

        """Perform validation checks."""
        if not _DeviceCtl._validation_done:
            if not command_exists(command="devicectl"):
                raise ValueError(
                    "The 'devicectl' command is required. Please ensure that you have the latest version of Xcode Command Line Tools installed."
                )
        _DeviceCtl._validation_done = True

    @staticmethod
    def _run_command(
        command: str,
    ) -> _Dict[str, any]:
        _DeviceCtl._validate()
        _logger.info(f"Executing devicectl command: {command}")
        with _tempfile.NamedTemporaryFile() as fp:
            fp.write(bytes())
            try:
                output = _subprocess.run(
                    f"{command} -j {fp.name}",
                    shell=True,
                    stdout=_subprocess.PIPE,
                    stderr=_subprocess.PIPE,
                    stdin=_subprocess.PIPE,
                )
                output.check_returncode()
            except _subprocess.CalledProcessError as e:
                raise _DeviceCtlError(
                    error_code=e.returncode,
                    message=e.stderr,
                )

            with open(fp.name) as json_data:
                result = _json.load(json_data)
                return result

    @staticmethod
    def _run_command_async(
        command: str,
    ) -> _subprocess.Popen:
        _DeviceCtl._validate()
        _logger.info(f"Executing devicectl command: {command}")
        return _subprocess.Popen(
            command,
            shell=True,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.PIPE,
            stdin=_subprocess.PIPE,
        )

    @staticmethod
    def _check_tools():
        def command_exists(command):
            """Check if a command exists in the system's PATH."""
            return _shutil.which(command) is not None

        if not command_exists("devicectl"):
            raise ValueError(
                "The 'devicectl' command is required. Please ensure that you have the latest version of Xcode Command Line Tools installed."
            )

    @staticmethod
    def install_app(
        device: Device,
        app_path: _Path,
    ) -> _AppInstallationInfo:
        """
        Install an application on the specified device.

        This method attempts to install an application on the given device using the
        'xcrun devicectl' command-line tool. It parses the installation result and
        returns information about the installed application.

        Parameters
        ----------
        device: Device
            The target device on which to install the application.

        app_path: Path
            The path to the application file (.ipa or .app) to be installed. This should
            be a Path object pointing to a valid application file.

        Returns
        -------
        AppInstallationInfo
            If successful, returns an AppInstallationInfo object containing:
            - bundle_identifier: The bundle ID of the installed application.
            - database_uuid: The database UUID assigned to the installed application.

        """
        json_data = _DeviceCtl._run_command(
            f"xcrun devicectl device install app -d {device.identifier} {app_path.resolve()}"
        )
        result = json_data.get("result", None)

        if result is None:
            raise _DeviceCtlError("App installation failed: No result data returned")

        installed_applications = result.get("installedApplications", [])
        if not isinstance(installed_applications, _Sequence):
            raise _DeviceCtlError("App installation failed: Invalid installed applications data")

        installed_application = next(iter(installed_applications), None)
        if not isinstance(installed_application, _Mapping):
            raise _DeviceCtlError("App installation failed: No applications were installed")

        bundle_identifier = installed_application.get("bundleID", "")
        database_uuid = installed_application.get("databaseUUID", "")

        return _AppInstallationInfo(
            bundle_identifier=bundle_identifier,
            database_uuid=database_uuid,
        )

    @staticmethod
    def launch_app(
        device: Device,
        installation_info: _AppInstallationInfo,
        working_directory: _Path,
    ) -> _AppSession:
        """
        Launch an installed application on the specified device and create a DeviceSession.

        This method launches an application that has been previously installed on the device
        using the 'xcrun devicectl' command-line tool. It starts the app in console mode and
        returns a DeviceSession object representing the running application session.

        Parameters
        ----------
        device: Device
            The target device on which to launch the application.

        installation_info : AppInstallationInfo
            The result of a previous app installation, containing the bundle identifier
            and database UUID of the installed application.

        Returns
        -------
        AppSession:
            An AppSession object representing the launched application session. This object contains:
            - process: The subprocess object for the running 'xcrun devicectl' command.
            - device: The Device on which the app was launched.
            - bundle_identifier: The bundle ID of the launched application.
            - database_uuid: The database UUID of the launched application.

        """

        process = _DeviceCtl._run_command_async(
            f"xcrun devicectl device process launch --console -d {device.identifier} {installation_info.bundle_identifier}"
        )
        return _AppSession(
            process=process,
            device=device,
            installation_info=installation_info,
            working_directory=working_directory,
        )

    @staticmethod
    def install_and_launch_app(
        device: Device,
        app_path: _Path,
        working_directory: _Path,
    ) -> _AppSession:
        """
        Install and launch an application on the specified device in a single operation.

        This method combines the process of installing an application and launching it
        on the target device. It first installs the app using the provided app path,
        and then immediately launches the newly installed application.

        Parameters
        ----------
        device : Device
            The target device on which to install and launch the application.

        app_path : Path
            The path to the application file (.ipa or .app) to be installed.

        Returns
        -------
        _AppSession:
            An _AppSession object representing the launched application session.
            This object typically contains:
            - process: The subprocess object for the running application.
            - device: The Device object on which the app was installed and launched.
            - bundle_identifier: The bundle ID of the installed and launched application.
            - database_uuid: The database UUID of the installed application.

        """
        installation_info = _DeviceCtl.install_app(device=device, app_path=app_path)

        return _DeviceCtl.launch_app(
            device=device,
            installation_info=installation_info,
            working_directory=working_directory,
        )

    @staticmethod
    def _parse_data_transfer_result(
        json_data: _Dict[str, any],
    ) -> _DataTransferInfo:
        result = json_data.get("result", None)
        if result is None:
            raise _DeviceCtlError("No 'result' field in the JSON data")

        source = result.get("source", None)
        if not isinstance(source, str):
            raise _DeviceCtlError("No 'source' field in the result data")

        destination = result.get("destination", None)
        if not isinstance(destination, str):
            raise _DeviceCtlError("No 'destination' field in the result data")

        return _DataTransferInfo(
            source=source,
            destination=destination,
        )

    @staticmethod
    def send_file_to_app(
        bundle_identifier: str,
        source_path: _Path,
        destination: str,
        device: Device,
    ) -> _DataTransferInfo:
        """
        Send a file to a specific application's data container on the target device.

        This method uses the 'devicectl' command-line tool to copy a file from the host
        machine to an application's data container on the specified device.

        Parameters
        ----------
        bundle_identifier: str
            The bundle identifier of the target application on the device.

        source_path : Path
            The path to the file on the host machine that should be sent to the device.
            This should be a Path object pointing to an existing file.

        destination: str
            The destination path within the app's data container where the file should
            be copied. This is relative to the app's container root.

        device: Device
            The target device to which the file will be sent.

        Returns
        -------
        _DataTransferInfo
            An object containing information about the completed file transfer, typically
            including:
            - source: The source path of the transferred file.
            - destination: The destination path of the transferred file on the device.
        """
        command = (
            f"devicectl device copy to "
            f"-d {device.identifier} "
            f"--source {source_path.resolve()} "
            f"{'' if device.type == DeviceType.MAC else '--user mobile '}"
            "--domain-type=appDataContainer "
            f"--domain-identifier={bundle_identifier} "
            f"--destination {destination}"
        )
        json_data = _DeviceCtl._run_command(command=command)
        return _DeviceCtl._parse_data_transfer_result(json_data=json_data)

    @staticmethod
    def receive_file_from_app(
        bundle_identifier: str,
        source: str,
        destination_path: _Path,
        device: Device,
    ) -> _DataTransferInfo:
        """
        Retrieve a file from a specific application's data container on the target device.

        This method uses the 'devicectl' command-line tool to copy a file from an application's
        data container on the specified device to the host machine.

        Parameters
        ----------
        bundle_identifier : str
            The bundle identifier of the source application on the device.
            This identifies the app's data container from which the file will be copied.

        source : str
            The source path of the file within the app's data container on the device.
            This path is relative to the app's container root.

        destination_path : Path
            The path on the host machine where the file should be saved.
            This should be a Path object pointing to the desired location for the received file.

        device : Device
            The source device from which the file will be retrieved. This object should have
            an 'identifier' attribute that uniquely identifies the device.

        Returns
        -------
        DataTransferInfo
            An object containing information about the completed file transfer, typically
            including:
            - source: The source path of the transferred file on the device.
            - destination: The destination path of the transferred file on the host machine.
        """
        command = (
            f"devicectl device copy from "
            f"-d {device.identifier} "
            f"--source {source} "
            f"{'' if device.type == DeviceType.MAC else '--user mobile '}"
            "--domain-type=appDataContainer "
            f"--domain-identifier={bundle_identifier} "
            f"--destination {destination_path.resolve()}"
        )
        json_data = _DeviceCtl._run_command(command=command)
        return _DeviceCtl._parse_data_transfer_result(json_data=json_data)

    @staticmethod
    def get_devices() -> _List[Device]:
        """
        Retrieve a list of all available devices connected to the host.

        This method uses the 'xcrun devicectl' command-line tool to fetch information
        about all connected devices and parses the result to create Device objects.

        Returns
        -------
        List[Device]
            A list of Device objects, each representing a connected device.
            Each Device object typically contains:
            - name: The name of the device.
            - identifier: A unique identifier for the device.
            - os_version: The operating system version of the device.
            - os_build_number: The build number of the operating system.
            - developer_mode_state: The state of developer mode on the device.
            - type: The type of the device (e.g., iPhone, iPad, Mac).
            - state: The current state of the device (e.g., connected, available).
            - udid: The Unique Device Identifier.
        """
        json_data = _DeviceCtl._run_command("xcrun devicectl list devices")
        result = json_data.get("result", None)
        if result is None:
            raise _DeviceCtlError("No 'result' field in the JSON data")

        devices = result.get("devices", None)
        if not isinstance(devices, _Sequence):
            raise _DeviceCtlError("No 'devices' field in the JSON data")

        result: _List[Device] = []

        for device in devices:
            device_properties = device.get("deviceProperties", {})
            hardware_properties = device.get("hardwareProperties", {})
            connection_properties = device.get("connectionProperties", {})

            if not isinstance(device_properties, _Mapping):
                raise _DeviceCtlError("No 'deviceProperties' field in the JSON data")

            if not isinstance(hardware_properties, _Mapping):
                raise _DeviceCtlError("No 'hardwareProperties' field in the JSON data")

            if not isinstance(connection_properties, _Mapping):
                raise _DeviceCtlError("No 'connectionProperties' field in the JSON data")

            type = DeviceType.UNKNOWN
            identifier = device.get("identifier", "")
            type = DeviceType(hardware_properties.get("deviceType", "").lower())
            udid = hardware_properties.get("udid", "")

            name = device_properties.get("name", "")
            if (
                len(identifier) == 0
                or len(udid) == 0
                or len(name) == 0
                or type == DeviceType.UNKNOWN
            ):
                _logger.warning(
                    f"Missing device identifier={identifier} or udid={udid} or type={type}, skipping."
                )
                continue

            os_version = device_properties.get("osVersionNumber", "")
            os_build_number = device_properties.get("osBuildUpdate", "")
            developer_mode_state = device_properties.get("developerModeStatus", "")
            state = DeviceState.UNKNOWN
            tunnel_state = connection_properties.get("tunnelState", DeviceState.UNKNOWN.value)
            state = DeviceState(tunnel_state.lower())
            device = Device(
                name=name,
                identifier=identifier,
                os_version=os_version,
                os_build_number=os_build_number,
                developer_mode_state=developer_mode_state,
                type=type,
                state=state,
                udid=udid,
                session=None,
            )
            result.append(device)

        return result


class _AppBuilder:
    """
    A utility class for building Xcode projects for different device types.
    """

    class Error(Exception):
        """Custom exception for build errors."""

        def __init__(self, message, error_code=None):
            self.message = message
            self.error_code = error_code
            super().__init__(self.message)

        def __str__(self):
            return f"[Error Code {self.error_code}] {self.message}"

    @classmethod
    def get_scheme_name(
        cls,
        workspace_name: str,
        device: Device,
    ) -> str:
        """
        Determine the appropriate scheme name based on the workspace name and device type.

        Parameters
        ----------
        workspace_name: str
            The name of the Xcode workspace.

        device: Device
            The target device for which the scheme is being determined.

        Returns
        --------
        str:
            The scheme name to use for the build.
        """
        if device.type == DeviceType.WATCH:
            return f"{workspace_name.lower()}-watchos"
        elif (
            device.type == DeviceType.IPHONE
            or device.type == DeviceType.IPAD
            or device.type == DeviceType.MAC
        ):
            return workspace_name.lower()
        else:
            raise ValueError(f"No scheme for device={device}, type = {device.type}")

    @classmethod
    def get_sdk_name(
        cls,
        device: Device,
        credentials: AppSigningCredentials,
    ) -> str:
        """
        Get the SDK name for the given device type.

        Parameters
        ----------
        device: Device
            The target device for which the SDK name is being determined.

        Returns
        -------
        str
            The SDK name to use for the build.
        """
        if device.type == DeviceType.MAC:
            return "macosx"
        elif device.type == DeviceType.IPHONE:
            return "iphoneos"
        elif device.type == DeviceType.IPAD:
            return "iphoneos"
        elif device.type == DeviceType.WATCH:
            return "watchos"
        else:
            raise ValueError(f"No sdk for device={device}, type = {device.type}")

    @classmethod
    def get_destination(
        cls,
        device: Device,
    ) -> str:
        """
        Get the destination string for ``xcodebuild`` based on the device type.

        Parameters
        ----------
        device : Device
            The target device for which the destination string is being determined.

        Returns
        -------
        str
            The destination string to use for the ``xcodebuild`` command.
        """
        if device.type == DeviceType.MAC:
            return f"platform=macOS,arch=arm64,x86_64,id={device.udid}"
        elif device.type == DeviceType.IPHONE:
            return f"platform=iOS,id={device.udid}"
        elif device.type == DeviceType.IPAD:
            return f"platform=iOS,id={device.udid}"
        elif device.type == DeviceType.WATCH:
            return f"platform=watchOS,arch=arm64_32,id={device.udid}"
        else:
            raise ValueError(f"No destination for device={device}, type = {device.type}")

    @classmethod
    def get_build_path(
        cls,
        device: Device,
        build_directory: _Path,
    ) -> _Path:
        """
        Determine the build path for the given device and build directory.

        Parameters
        ----------
        device : Device
            The target device for which the build path is being determined.

        build_directory : Path
            The base build directory.

        Returns
        -------
        Path
            The full path where the build artifacts will be stored.
        """
        return build_directory / device.udid

    @classmethod
    def check_app_signing_credentials(
        cls,
        credentials: AppSigningCredentials,
    ):
        if (credentials.provisioning_profile_uuid is None) and (
            credentials.development_team is None
        ):
            raise ValueError(
                "Invalid signing credentials: Either 'provisioning_profile_uuid' or 'development_team' must be provided.\n"
                "Note:\n"
                "- When using 'provisioning_profile_uuid', ensure 'bundle_identifier' matches the profile's bundle ID\n"
                "- When using 'development_team', Xcode will automatically manage provisioning profiles"
            )

    @classmethod
    def build(
        cls,
        device: Device,
        build_directory: _Path,
        workspace_path: _Path,
        credentials: AppSigningCredentials,
        clean: bool = False,
    ) -> _Path:
        """
        Build the Xcode project for the specified device.

        This method handles the entire build process, including cleaning if requested,
        setting up the build directory, and running the ``xcodebuild`` command.

        Parameters:
        -----------
        device: Device
            The target device for which to build the app.

        build_directory: Path
            The directory where build artifacts will be stored.

        workspace_path: Path
            The path to the Xcode workspace file.

        credentials: AppSigningCredentials
            The application code signing credentials.

        clean: bool
            Whether to clean the build directory before building (default is False).

        Returns
        -------
        Path
            The path to the built .app file if successful, None otherwise.

        """
        cls.check_app_signing_credentials(credentials=credentials)

        build_path = cls.get_build_path(
            device=device,
            build_directory=build_directory,
        )

        result = None
        if clean and build_path.is_dir():
            _shutil.rmtree(build_path.resolve())
        else:
            result = next(build_path.rglob("*.app"), None)

        if result is not None:
            return result

        build_path.mkdir(parents=True, exist_ok=True)

        if not workspace_path.is_dir():
            raise ValueError(f"Workspace path={workspace_path} is not a directory.")

        workspace_name = _os.path.splitext(_os.path.basename(workspace_path))[0]

        bundle_identifier = credentials.bundle_identifier
        bundle_identifier = (
            bundle_identifier if bundle_identifier is not None else "com.coremltools.modelrunnerd"
        )

        command = (
            f"xcodebuild -workspace {str(workspace_path.resolve())} "
            f"-scheme {cls.get_scheme_name(workspace_name= workspace_name, device=device)} "
            f"-sdk {cls.get_sdk_name(device, credentials)} "
            f"-destination {cls.get_destination(device)} "
            "-configuration Release "
            f"SYMROOT={build_path.resolve()} "
            "CODE_SIGN_STYLE=AUTOMATIC "
        )

        if credentials.provisioning_profile_uuid is not None:
            command += f"PROVISIONING_PROFILE={credentials.provisioning_profile_uuid} "
        else:
            command += "-allowProvisioningUpdates -allowProvisioningDeviceRegistration "

        if credentials.development_team is not None and len(credentials.development_team) > 0:
            command += f"DEVELOPMENT_TEAM={credentials.development_team} "
        else:
            command += 'CODE_SIGN_IDENTITY="-" '

        command += f"PRODUCT_BUNDLE_IDENTIFIER={bundle_identifier} "

        try:
            output = _subprocess.run(
                command,
                shell=True,
                stdout=_subprocess.PIPE,
                stderr=_subprocess.STDOUT,
                stdin=_subprocess.PIPE,
            )
            output.check_returncode()
        except _subprocess.CalledProcessError as e:
            raise _AppBuilder.Error(
                error_code=e.returncode,
                message=e.stderr,
            )

        result = next(build_path.rglob("*.app"), None)
        return result


class _ModelRunnerAppBuilder(_AppBuilder):
    """
    A utility class for building the modelrunner Xcode project.
    """

    @staticmethod
    def _get_project_path():
        return _Path(__file__).parent.parent.parent.parent / "modelrunner"

    @classmethod
    def build(
        cls,
        device: Device,
        build_directory: _Path,
        credentials: AppSigningCredentials,
        clean: bool = False,
    ) -> _Path:
        """
        Build the ``modelrunner`` Xcode project for the specified device.

        This method handles the entire build process, including cleaning if requested,
        setting up the build directory, and running the ``xcodebuild`` command.

        Parameters:
        -----------
        device: Device
            The target device for which to build the app.

        build_directory: Path
            The directory where build artifacts will be stored.

        clean: bool
            Whether to clean the build directory before building (default is False).

        Returns
        -------
        Path
            The path to the built .app file if successful, None otherwise.

        """

        def command_exists(command):
            """Check if a command exists in the system's PATH."""
            return _shutil.which(command) is not None

        project_path = _ModelRunnerAppBuilder._get_project_path()

        if not project_path.is_dir():
            raise ValueError(
                f"Model runner directory not found at expected path: {project_path.resolve()}"
            )

        workspace_path = project_path / "ModelRunner.xcworkspace"

        if not workspace_path.is_dir():
            raise ValueError(
                f"Workspace directory not found at expected path: {workspace_path.resolve()}"
            )

        if not command_exists("xcodebuild"):
            raise ValueError(
                "The 'xcodebuild' command is required. Please ensure that Xcode Command Line Tools are installed and accessible."
            )

        return super().build(
            device=device,
            build_directory=build_directory.resolve(),
            workspace_path=workspace_path.resolve(),
            credentials=credentials,
            clean=clean,
        )


@_dataclass(frozen=True)
class _JSONRPCRequest:
    """
    Represents a JSON-RPC request.
    """

    id: str
    method: str
    params: any


class _JSONRPCError(Exception):
    """
    Represents a JSON-RPC error.
    """

    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"Error {self.code}: {self.message}"


@_dataclass(frozen=True)
class _JSONRPCResponse:
    """
    Represents a JSON-RPC response with an associated resource.
    """

    id: str
    result: _Optional[_Any]
    error: _Optional[_JSONRPCError]
    resource: _Path


class _JSONRPCSocket(_ABC):
    """
    An abstract base class representing a JSON-RPC socket for communication.

    This class defines the interface for sending JSON-RPC requests and receiving
    responses, potentially with associated resources.
    """

    @_abstractmethod
    def send(
        self,
        request: _JSONRPCRequest,
        resource_path: _Optional[_Path] = None,
    ) -> None:
        """
        Send a JSON-RPC request through the socket.

        This method is responsible for transmitting a JSON-RPC request, optionally
        with an associated resource.

        Parameters
        ----------
        request: JSONRPCRequest
            The JSON-RPC request to be sent.

        resource_path: Optional[Path]
            The path to an associated resource file, if any. Defaults to None.

        """
        pass

    @_abstractmethod
    def receive(
        self,
        id: str,
    ) -> _Optional[_JSONRPCResponse]:
        """
        Receive a JSON-RPC response for a given request ID.

        This method is responsible for receiving and returning a JSON-RPC response
        associated with a specific request ID, potentially including resource information.

        Parameters
        ----------
        id: str
            The ID of the JSON-RPC request for which to receive the response.

        Returns
        -------
        Optional[JSONRPCResourcefulResponse]
            The received JSON-RPC response, potentially including resource information.
        """
        pass

    @property
    @_abstractmethod
    def is_alive(self) -> bool:
        """
        Checks if the socket is active.

        Returns
        -------
        bool:
            True if the socket is active, False otherwise.
        """
        pass


class _DeviceCtlSocket(_JSONRPCSocket):
    """
    A class for managing communication with a device application using JSON-RPC over a custom socket-like interface.

    This class handles sending requests to and receiving responses from an application running on a device.
    """

    @staticmethod
    def _parse_error(
        value: _Optional[_Any],
    ) -> _Optional[_JSONRPCError]:
        if value is None:
            return None

        if not isinstance(value, _Mapping):
            raise ValueError(f"Error = {value} is not a dict")

        code = value["code"]
        message = value["message"]
        return _JSONRPCError(
            code=code,
            message=message,
        )

    def __init__(
        self,
        device: Device,
        source: str,
    ) -> None:
        """
        Initialize a new DeviceCtlSocket instance.

        Parameters
        ----------
        device: Device
            The device object.

        working_dir: Path
            The base working directory for storing temporary files and resources.

        source: str
            The source identifier for the communication.
        """
        self.device = device
        self.working_directory = device.session.working_directory / str(_uuid.uuid4())
        self.working_directory.mkdir(parents=True, exist_ok=False)
        self.source = source
        self.requests_directory = self.working_directory / "requests"
        self.responses_directory = self.working_directory / "responses"

    def send(
        self,
        request: _JSONRPCRequest,
        resource_path: _Optional[_Path] = None,
    ) -> None:
        """
        Send a JSON-RPC request to the device application.

        This method prepares and sends a JSON-RPC request, optionally including a resource file.

        Parameters
        ----------
        request: JSONRPCRequest
           The JSON-RPC request to send.

        resource_path: Optional[Path]
           Path to a resource file to be sent with the request.

        Returns
        -------
        DataTransferInfo
           Information about the data transfer.
        """
        message = {}
        message["jsonrpc"] = "2.0"
        message["id"] = request.id
        message["method"] = request.method
        message["params"] = request.params

        requests_directory = self.requests_directory / request.id
        if requests_directory.is_dir():
            _shutil.rmtree(requests_directory.resolve())

        if resource_path is not None:
            resource_name = _os.path.basename(resource_path)
            self.device.session.send_file_to_app(
                resource_path,
                destination=f"{self.source}/requests/resources/{request.id}/{resource_name}",
            )
            message["resource"] = resource_name

        requests_directory.mkdir(parents=True, exist_ok=False)
        message_json_file = requests_directory / "message.json"
        with open(message_json_file, "w") as file:
            json_string = _json.dumps(message)
            file.write(json_string)

        self.device.session.send_file_to_app(
            message_json_file,
            destination=f"{self.source}/requests/messages/{request.id}/message.json",
        )

    def receive(
        self,
        id: str,
    ) -> _Optional[_JSONRPCResponse]:
        """
        Receive a JSON-RPC response from the device application.

        This method retrieves and parses a JSON-RPC response, including any associated resource files.

        Parameters
        ----------
        id : str
            The ID of the request for which to receive the response.

        Returns
        -------
        Optional[JSONRPCResponse]
            The received response, or None if no response is available.
        """
        responses_directory = self.responses_directory / id
        if responses_directory.is_dir():
            _shutil.rmtree(responses_directory.resolve())

        responses_directory.mkdir(parents=True, exist_ok=False)
        payload = self.device.session.receive_file_from_app(
            destination_path=responses_directory / "message.json",
            source=f"{self.source}/responses/{id}/message.json",
        )

        if payload is None:
            return None

        message_json_file = responses_directory / "message.json"
        if not message_json_file.is_file():
            raise ValueError(f"Response for request-id{id} is missing message.json file.")

        result = None
        error = None
        resource = None
        resource_name = None
        with open(message_json_file, "r") as file:
            message = _json.load(file)
            if not isinstance(message, _Mapping):
                raise ValueError(f"Response for request-id{id} is malformed {message}.")

            response_id = message.get("id", "")
            if response_id != id:
                raise ValueError(f"Response for request-id{id} is malformed {message}.")

            result = message.get("result", None)
            error = _DeviceCtlSocket._parse_error(message.get("error", None))
            resource_name = message.get("resource", None)

        if result is None and error is None:
            raise ValueError(
                f"Response for request-id{id} is malformed {message}, missing result={result} or error={error}."
            )

        resource = None
        if resource_name is not None:
            resource_payload = self.device.session.receive_file_from_app(
                destination_path=responses_directory / resource_name,
                source=f"{self.source}/responses/{id}/{resource_name}",
            )

            if resource_payload is None:
                raise ValueError(
                    f"Response for request-id={id} is missing resource file={resource_name}."
                )

            resource = responses_directory / resource_name

        return _JSONRPCResponse(
            id=id,
            result=result,
            error=error,
            resource=resource,
        )

    @property
    def is_alive(self) -> bool:
        """
        Checks if the socket is active.

        Returns
        -------
        bool
            True if the socket is active, False otherwise.
        """
        return self.device.session.is_alive


class _RemoteService(_ABC):
    """
    An abstract base class representing a remote service capable of executing methods.

    This class defines the interface for remote method invocation, allowing
    for asynchronous calls with optional resource handling.
    """

    async def call_method(
        self,
        name: str,
        params: _Any,
        resource: _Optional[_Path] = None,
    ) -> _Tuple[_Any, _Optional[_Path]]:
        """
        Asynchronously call a method on the remote service.

        This method is responsible for invoking a named method on the remote service,
        passing the provided parameters, and optionally handling a resource.

        Parameters
        ----------
        name: str
            The name of the method to be called on the remote service.

        params: Any
            The parameters to be passed to the remote method. This can be of any type, depending on what the remote method expects.


        resource: Optional[Path]
            A path to a resource that may be required or modified by the method call. Defaults to None.

        Returns
        -------
        Tuple[Any, Optional[Path]]
            A tuple containing:
                - The result of the method call (Any type)
                - An optional Path object, which may represent a new or modified resource
        """
        pass

    def fire_and_forget(
        self,
        name: str,
        params: _Any,
        resource: _Optional[_Path] = None,
    ) -> None:
        """
        Executes a method asynchronously without waiting for its completion.

        This method uses the event loop's executor to run the `call_method` function
        in a separate thread, allowing the calling code to continue execution
        immediately without waiting for the method to complete.

        Parameters
        ----------
        name: str
            The name of the method to call on the remote device.

        params: Any
            The parameters to pass to the method. Can be any JSON-serializable object.

        resource: Optional[Path]
            Path to a resource file to be sent with the request. Defaults to None.

        """
        pass

    @property
    def is_alive(self) -> bool:
        """
        Checks if the service is active.

        Returns
        -------
        bool:
            True if the service is active, False otherwise.
        """
        pass


class _DeviceCtlRemoteService(_RemoteService):
    """
    A service class for managing device control operations using a DeviceCtlSocket.

    This class provides an asynchronous interface for calling methods on a remote device
    using JSON-RPC over a custom socket implementation.
    """

    def __init__(self, socket: _DeviceCtlSocket, poll_time: float = 0.5, max_attempts=20) -> None:
        """
        Initialize a DeviceCtlService instance.

        Parameters
        ----------
        socket: DeviceCtlSocket
            The socket used for communication with the device.

        poll_time: float
            The time interval (in seconds) between polling for responses. Defaults to 0.5 seconds.
        """
        self.socket = socket
        self.poll_time = poll_time
        self.max_attempts = max_attempts
        self._event_loop = _asyncio.get_event_loop()

    @property
    def working_dir(self) -> _Path:
        """
        Get the working directory of the associated socket.
        """
        return self.socket.working_dir

    async def call_method(
        self,
        name: str,
        params: _Any,
        resource: _Optional[_Path] = None,
    ) -> _Tuple[_Any, _Optional[_Path]]:
        """
        Asynchronously call a method on the remote device.

        This method sends a JSON-RPC request to the device and waits for the response.
        It handles the creation of the request, sending it through the socket, and polling
        for the response.

        Parameters
        ----------
        name: str
            The name of the method to call on the remote device.

        params: Any
            The parameters to pass to the method. Can be any JSON-serializable object.

        resource: Optional[Path]
            Path to a resource file to be sent with the request. Defaults to None.

        Returns
        -------
        Tuple[Any, Optional[Path]]
            A tuple containing:
            - The result of the method call (Any)
            - The path to any resource returned by the method (Optional[Path])
        """

        def should_throw_exception(attempt: int) -> bool:
            return not self.socket.is_alive or (attempt + 1) == self.max_attempts

        request = _JSONRPCRequest(
            id=str(_uuid.uuid4()),
            method=name,
            params=params,
        )

        for attempt in _tqdm.tqdm(
            range(self.max_attempts),
            desc=f"\033[1mInitiating data transfer to '{self.socket.device.name}' for {name}. Attempts \033[0m",
        ):
            if resource is not None and not resource.exists():
                raise _DeviceCtlError(f"Resource file {resource.absolute()} does not exist.")
            try:
                await self._event_loop.run_in_executor(None, self.socket.send, request, resource)
                break
            except _DeviceCtlError as e:
                _logger.info(
                    f"Failed to send request: {request}, code: {e.error_code}, message: {e.message}"
                )
                if should_throw_exception(attempt=attempt):
                    raise e

            await _asyncio.sleep(self.poll_time)

        response = None
        for attempt in _tqdm.tqdm(
            range(self.max_attempts),
            desc=f"\033[1mInitiating data transfer from {self.socket.device.name} for {name}. Attempts \033[0m",
        ):
            try:
                response = await self._event_loop.run_in_executor(
                    None, self.socket.receive, request.id
                )
                if response is not None:
                    break
            except _DeviceCtlError as e:
                _logger.info(
                    f"Failed to receive response of id: {request.id}, code: {e.error_code}, message: {e.message}"
                )
                if should_throw_exception(attempt=attempt):
                    raise e

            await _asyncio.sleep(self.poll_time)

        if response.error is not None:
            raise _DeviceCtlError(error_code=response.error.code, message=response.error.message)

        return (response.result, response.resource)

    def fire_and_forget(
        self,
        name: str,
        params: _Any,
        resource: _Optional[_Path] = None,
    ) -> None:
        """
        Executes a method asynchronously without waiting for its completion.

        This method uses the event loop's executor to run the `call_method` function
        in a separate thread, allowing the calling code to continue execution
        immediately without waiting for the method to complete.

        Parameters
        ----------
        name: str
            The name of the method to call on the remote device.

        params: Any
            The parameters to pass to the method. Can be any JSON-serializable object.

        resource: Optional[Path]
            Path to a resource file to be sent with the request. Defaults to None.

        """
        self._event_loop.run_in_executor(
            None,
            self.call_method,
            name,
            params,
            resource,
        )

    @property
    def is_alive(self) -> bool:
        """
        Checks if the socket is active.

        Returns
        -------
        bool:
            True if the socket is active, False otherwise.
        """
        return self.socket.is_alive


@_dataclass(frozen=True)
class _TensorStorage:
    """
    Represents a segment of data with an offset and size.
    """

    offset: int
    size: int

    def as_dict(
        self,
    ) -> _Dict[str, _Any]:
        """
        Convert the TensorStorage to a dictionary.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the TensorStorage.
        """
        return {"offset": self.offset, "size": self.size}

    @staticmethod
    def from_dict(
        dict: _Dict[str, _Any],
    ) -> _Type["_TensorStorage"]:
        """
        Create a TensorStorage instance from a dictionary.

        Parameters
        ----------
        dict: Dict[str, Any]
            A dictionary containing 'offset' and 'size' keys.

        Returns
        -------
        TensorStorage
            A new TensorStorage instance.
        """
        return _TensorStorage(
            offset=dict.get("offset", 0),
            size=dict.get("size", 0),
        )


@_dataclass(frozen=True)
class _TensorDescriptor:
    """
    Describes the properties of a tensor.
    """

    class DataType(_Enum):
        Float16 = "Float16"
        Float32 = "Float32"
        Float64 = "Float64"
        Int32 = "Int32"

    shape: _List[int]
    strides: _List[int]
    data_type: DataType
    storage: _TensorStorage

    @staticmethod
    def _to_multi_array_dtype(
        dtype: _np.dtype,
    ) -> str:
        if dtype == _np.float16:
            return _TensorDescriptor.DataType.Float16
        elif dtype == _np.float32:
            return _TensorDescriptor.DataType.Float32
        elif dtype == _np.float64:
            return _TensorDescriptor.DataType.Float64
        elif dtype == _np.int32:
            return _TensorDescriptor.DataType.Int32
        else:
            raise ValueError(f"{dtype} is not supported")

    @staticmethod
    def _to_numpy_dtype(
        dtype: str,
    ) -> str:
        if dtype == _TensorDescriptor.DataType.Float16:
            return _np.float16
        elif dtype == _TensorDescriptor.DataType.Float32:
            return _np.float32
        elif dtype == _TensorDescriptor.DataType.Float64:
            return _np.float64
        elif dtype == _TensorDescriptor.DataType.Int32:
            return _np.int32
        else:
            raise ValueError(f"{dtype} is not supported")

    @staticmethod
    def from_array(
        array: _np.array,
        file: _BinaryIO,
    ) -> _Type["_TensorDescriptor"]:
        """
        Create a TensorDescriptor from a numpy array and write the data to a binary file.

        This static method converts a numpy array into a TensorDescriptor and writes the
        array data to the provided binary file.

        Parameters
        ----------
        array: np.array
            The numpy array to convert.

        file: BinaryIO
            A binary file object to which the array data will be written.

        Returns
        -------
        TensorDescriptor
             A new TensorDescriptor instance representing the input array.
        """
        shape = list(array.shape)
        strides = [x // array.dtype.itemsize for x in list(array.strides)]
        data_type = _TensorDescriptor._to_multi_array_dtype(array.dtype)

        offset = file.tell()
        buffer = array.tobytes()
        size = len(buffer)
        file.write(buffer)

        storage = _TensorStorage(
            offset=offset,
            size=size,
        )

        return _TensorDescriptor(
            shape=shape,
            strides=strides,
            data_type=data_type,
            storage=storage,
        )

    def to_array(
        self,
        file: _BinaryIO,
    ) -> _np.array:
        """
        Convert the TensorDescriptor back to a numpy array, reading data from a binary file.

        This method reconstructs a numpy array from the TensorDescriptor, reading the necessary
        data from the provided binary file.

        Parameters
        ----------
        file: BinaryIO
            A binary file object from which to read the array data.
            This file should be openable in binary read mode ('rb').

        Returns
        -------
        np.array
            A numpy array reconstructed from the TensorDescriptor and file data.
        """

        data_type = _TensorDescriptor._to_numpy_dtype(self.data_type)
        file.seek(self.storage.offset)
        data = file.read(self.storage.size)
        result = _np.frombuffer(data, data_type)
        strides = [x * result.itemsize for x in self.strides]
        strided_view = _np.lib.stride_tricks.as_strided(result, self.shape, strides)
        result = _np.array(strided_view)
        return result

    def as_dict(
        self,
    ) -> _Dict[str, _Any]:
        """
        Convert the TensorDescriptor to a dictionary representation.

        This method creates a dictionary containing all the attributes of the TensorDescriptor,
        with keys matching the attribute names.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the TensorDescriptor with the following keys:
                - 'shape': List[int], the shape of the tensor
                - 'strides': List[int], the strides of the tensor
                - 'dataType': str, the string representation of the data type
                - 'storage': Dict, the dictionary representation of the TensorStorage
        """
        result = {}
        result["shape"] = self.shape
        result["strides"] = self.strides
        result["dataType"] = self.data_type.value
        result["storage"] = self.storage.as_dict()

        return result

    @staticmethod
    def from_dict(
        dict: _Dict[str, _Any],
    ) -> _Type["_TensorDescriptor"]:
        """
        Create a TensorDescriptor instance from a dictionary.

        Parameters
        ----------
        dict: Dict[str, Any]
            A dictionary containing the TensorDescriptor attributes.

        Returns
        -------
        TensorDescriptor
            A new TensorDescriptor instance created from the dictionary data.
        """
        return _TensorDescriptor(
            shape=dict.get("shape", []),
            strides=dict.get("strides", []),
            data_type=_TensorDescriptor.DataType(
                dict.get("dataType", _TensorDescriptor.DataType.Float32.value)
            ),
            storage=_TensorStorage.from_dict(dict.get("storage", {})),
        )


@_dataclass(frozen=True)
class ComputePlan:
    @_dataclass(frozen=True)
    class CPUDevice:
        @staticmethod
        def from_dict(dict: _Dict[str, _Any]) -> _Optional["ComputePlan.CPUDevice"]:
            value = dict.get("cpu", None)
            return ComputePlan.CPUDevice() if value is not None else None

        def to_dict(self) -> _Dict[str, _Any]:
            return {"cpu": {}}

    @_dataclass(frozen=True)
    class GPUDevice:
        @staticmethod
        def from_dict(dict: _Dict[str, _Any]) -> _Optional["ComputePlan.GPUDevice"]:
            value = dict.get("gpu", None)
            return ComputePlan.GPUDevice() if value is not None else None

        def to_dict(self) -> _Dict[str, _Any]:
            return {"gpu": {}}

    @_dataclass(frozen=True)
    class NeuralEngineDevice:
        total_core_count: int

        @staticmethod
        def from_dict(dict: _Dict[str, _Any]) -> _Optional["ComputePlan.NeuralEngineDevice"]:
            value = dict.get("neuralEngine", None)
            if not isinstance(value, _Mapping):
                return None

            total_core_count = value.get("totalCoreCount", 0)
            return ComputePlan.NeuralEngineDevice(total_core_count=total_core_count)

        def to_dict(self) -> _Dict[str, _Any]:
            return {"neuralEngine": {"totalCoreCount": self.total_core_count}}

    Device = _Union[
        "ComputePlan.CPUDevice",
        "ComputePlan.GPUDevice",
        "ComputePlan.NeuralEngineDevice",
    ]

    @_dataclass(frozen=True)
    class DeviceUsage:
        preferred: "ComputePlan.Device"
        supported: _List["ComputePlan.Device"]

        @staticmethod
        def from_dict(
            dict: _Dict[str, _Any],
        ) -> _Optional["ComputePlan.Device"]:
            def device_from_dict(dict: _Dict[str, _Any]) -> _Optional[Device]:
                device_types = [
                    ComputePlan.CPUDevice,
                    ComputePlan.GPUDevice,
                    ComputePlan.NeuralEngineDevice,
                ]

                for device_type in device_types:
                    parsed_device = device_type.from_dict(dict=dict)
                    if parsed_device is not None:
                        return parsed_device

                raise ValueError(
                    f"Failed to parse device: {dict}, it does not match any known device type"
                )

            preferred_device = dict.get("preferred", None)
            if not isinstance(preferred_device, _Mapping):
                raise ValueError(f"Failed to parse preferred device: {preferred_device}")
            preferred = device_from_dict(preferred_device)

            supported_devices = dict.get("supported", None)
            if not isinstance(supported_devices, _Sequence):
                raise ValueError(f"Failed to parse supported devices: {supported_devices}")

            supported = [device_from_dict(device) for device in supported_devices]

            return ComputePlan.DeviceUsage(preferred=preferred, supported=supported)

        def to_dict(self) -> _Dict[str, _Any]:
            return {
                "preferred": self.preferred.to_dict(),
                "supported": [device.to_dict() for device in self.supported],
            }

    @_dataclass(frozen=True)
    class OperationOrLayerInfo:
        device_usage: "ComputePlan.DeviceUsage"
        estimated_cost: _Optional[float]
        path: ModelStructurePath

        @staticmethod
        def from_dict(
            dict: _Dict[str, _Any],
        ) -> _Optional["ComputePlan.OperationOrLayerInfo"]:
            device_usage = ComputePlan.DeviceUsage.from_dict(dict.get("deviceUsage", {}))
            estimated_cost = dict.get("estimatedCost", None)
            path = ModelStructurePath.from_dict(dict.get("path", {}))

            return ComputePlan.OperationOrLayerInfo(
                device_usage=device_usage,
                estimated_cost=estimated_cost,
                path=path,
            )

        def to_dict(self) -> _Dict[str, _Any]:
            return {
                "deviceUsage": self.device_usage.to_dict(),
                "estimatedCost": self.estimated_cost,
                "path": self.path.to_dict(),
            }

    infos: _Dict[ModelStructurePath, "ComputePlan.OperationOrLayerInfo"]

    @staticmethod
    def from_dict(dict: _Dict[str, _Any]) -> _Optional["ComputePlan"]:
        infos = dict.get("infos", None)
        if not isinstance(infos, _Sequence):
            return None

        infos = (ComputePlan.OperationOrLayerInfo.from_dict(info) for info in infos)
        return ComputePlan(infos={info.path: info for info in infos})

    def to_dict(self) -> _Dict[str, _Any]:
        return {
            "infos": {key: value.to_dict() for key, value in self.infos.items()},
        }


class _RemoteMLModelService:
    """
    This class provides a service interface for interacting with an MLModel
    on a remote device using a DeviceCtlService.
    """

    class State(_Enum):
        Loaded = "loaded"
        Unloaded = "unloaded"

    @staticmethod
    def _validate_args(
        service: _RemoteService,
        compiled_model_path: _Union[_Path, str],
        compute_units: _ComputeUnit,
        function_name: _Optional[str],
        optimization_hints: _Optional[_Dict[str, _Any]],
    ):
        if not isinstance(service, _RemoteService):
            raise TypeError(
                f"Parameter 'service' must be a RemoteService instance, not {type(service).__name__}"
            )

        if not isinstance(compiled_model_path, _Path) and not isinstance(compiled_model_path, str):
            raise TypeError(
                f"Parameter 'compiled_model_path' must be a Path or str, not {type(compiled_model_path).__name__}"
            )

        if not isinstance(compute_units, _ComputeUnit):
            raise TypeError(
                f"Parameter 'compute_units' must be a ComputeUnit enum, not {type(compute_units).__name__}"
            )

        if function_name is not None and not isinstance(function_name, str):
            raise TypeError(
                f"Parameter 'function_name' must be a str, not {type(function_name).__name__}"
            )

        if optimization_hints is not None and not isinstance(optimization_hints, _Mapping):
            raise TypeError(
                f"Parameter 'optimization_hints' must be a mapping type (e.g., dict), not {type(optimization_hints).__name__}"
            )

    def __init__(
        self,
        service: _RemoteService,
        compiled_model_path: _Union[_Path, str],
        compute_units: _ComputeUnit,
        function_name: _Optional[str] = None,
        optimization_hints: _Optional[_Dict[str, _Any]] = None,
    ) -> None:
        """
        Initialize a RemoteMLModelService instance.

        Parameter
        ---------
        compiled_model_path: Path
            Path to the compiled model file.

        compute_units: ComputeUnit
            The compute units to be used for model execution.

        service: RemoteService
            The service used for communication with the remote device.
        """

        _RemoteMLModelService._validate_args(
            service=service,
            compiled_model_path=compiled_model_path,
            compute_units=compute_units,
            function_name=function_name,
            optimization_hints=optimization_hints,
        )

        if isinstance(compiled_model_path, str):
            compiled_model_path = _Path(compiled_model_path)

        self._compiled_model_path = compiled_model_path
        self._compute_units = compute_units
        self._model_id = str(_uuid.uuid4())
        self._service = service
        self._model_name = _os.path.basename(compiled_model_path)
        self._function_name = function_name
        self._optimization_hints = optimization_hints
        self._state = _RemoteMLModelService.State.Unloaded
        self._load_duration_in_nano_seconds = None
        self._last_predict_duration_in_nano_seconds = None

    @property
    def load_duration_in_nano_seconds(self) -> _Optional[int]:
        return self._load_duration_in_nano_seconds

    @property
    def last_predict_duration_in_nano_seconds(self) -> _Optional[int]:
        return self._last_predict_duration_in_nano_seconds

    def __del__(self):
        if self._state == _RemoteMLModelService.State.Unloaded:
            return

        params = {
            "modelID": self._model_id,
            "computeUnits": self._compute_units.name,
        }

        self._service.fire_and_forget(
            name="MLModelService.Unload",
            params=params,
        )

    @property
    def model_id(self) -> str:
        """
        Get the unique identifier for this model instance.

        Returns
        -------
        str
            The model ID.
        """
        return self._model_id

    @property
    def compute_units(self) -> _ComputeUnit:
        """
        Get the compute units used for this model.

        Returns
        -------
        ComputeUnit
            The compute units.
        """
        return self._compute_units

    def _validate_result(
        self,
        result: _Dict[str, _Any],
    ) -> None:
        if not isinstance(result, _Mapping):
            raise TypeError(
                f"Expected result to be a mapping type (e.g., dict), but got {type(result).__name__}"
            )

        if result.get("modelID", None) != self.model_id:
            raise ValueError(
                f"Mismatched model ID: Expected '{self.model_id}', but got '{result.get('modelID', 'None')}'"
            )

    @staticmethod
    def _validate_device_state(
        device: Device,
    ):
        if device.session is None:
            raise ValueError(
                "Device is not prepared for debugging. Please call `prepare_for_model_debugging()` on the device before attempting to load the model."
            )

        if not device.session.is_alive:
            raise ValueError(
                "The device session is not active. Please call `prepare_for_model_debugging()` on the device to establish a new session."
            )

    async def load(
        self,
    ) -> None:
        """
        Loads the model on the remote device asynchronously.
        """

        if self._state == _RemoteMLModelService.State.Loaded:
            return

        params = {
            "modelID": self._model_id,
            "computeUnits": self._compute_units.name,
            "name": self._model_name,
            "functionName": self._function_name,
        }

        if self._optimization_hints is not None:
            optimization_hints_str_vals = {k: v.name for k, v in self._optimization_hints.items()}
            params["optimizationHints"] = optimization_hints_str_vals


        (result, _) = await self._service.call_method(
            name="MLModelService.Load",
            params=params,
            resource=self._compiled_model_path,
        )

        self._validate_result(result=result)
        self._state = _RemoteMLModelService.State.Loaded
        self._load_duration_in_nano_seconds = result.get("duration", None)

    @staticmethod
    def _prepare_data_for_transfer(
        values: _Dict[str, _np.array],
    ) -> _Tuple[_Dict[str, _Any], _Path]:
        data_file = _tempfile.NamedTemporaryFile("w+b", suffix=".bin", delete=False)
        data_file.seek(0)

        result = {}
        for name, value in values.items():

            result[name] = _TensorDescriptor.from_array(
                array=value,
                file=data_file,
            ).as_dict()
        data_file.close()
        data_file_path = _Path(data_file.name)

        return (result, data_file_path)

    async def predict(
        self,
        inputs: _Dict[str, _np.array],
    ) -> _Dict[str, _np.array]:
        """
        Perform a prediction using the remote ML model.

        This asynchronous method sends input data to the remote model, executes the prediction,
        and returns the results.

        Parameters
        ----------
        inputs : Dict[str, np.array]
            A dictionary mapping input names to numpy arrays containing the input data for the prediction.

        Returns
        -------
        Dict[str, np.array]
            A dictionary mapping output names to numpy arrays containing the prediction results.
        """
        if self._state == _RemoteMLModelService.State.Unloaded:
            message = f"The model with ID {self.model_id} and {self.compute_units.value} is not currently loaded."
            raise _DeviceCtlError(error_code=-32001, message=message)

        (values, data_file_path) = self._prepare_data_for_transfer(values=inputs)

        params = {
            "modelID": self._model_id,
            "computeUnits": self._compute_units.name,
            "inputs": values,
        }

        (result, resource) = await self._service.call_method(
            name="MLModelService.Prediction",
            params=params,
            resource=data_file_path,
        )

        self._validate_result(result=result)

        outputs = result.get("outputs", None)
        if not isinstance(outputs, _Mapping):
            raise TypeError(
                f"Expected outputs to be a mapping type (e.g., dict), but got {type(outputs).__name__}"
            )

        values: _Dict[str, _np.array] = {}
        if resource is None:
            raise ValueError("Required resource for tensor data storage is missing.")

        with open(resource, "rb") as fp:
            for name, value in outputs.items():
                if not isinstance(value, _Mapping):
                    raise TypeError(
                        f"Expected output to be a mapping type (e.g., dict), but got {type(value).__name__}"
                    )

                descriptor = _TensorDescriptor.from_dict(value)
                array = descriptor.to_array(fp)
                values[name] = array

        self._last_predict_duration_in_nano_seconds = result.get("duration", None)

        return values

    async def retrieve_compute_plan(
        self,
    ) -> ComputePlan:
        """
        Retrieves the compute plan of a model loaded on the remote device asynchronously.
        """
        params = {
            "modelID": self._model_id,
        }

        (result, _) = await self._service.call_method(
            name="MLModelService.ComputePlan",
            params=params,
        )

        self._validate_result(result=result)
        return ComputePlan.from_dict(dict=result)

    async def unload(
        self,
    ) -> None:
        """
        Unloads the model on the remote device asynchronously.
        """

        if self._state == _RemoteMLModelService.State.Unloaded:
            return
        params = {
            "modelID": self._model_id,
            "computeUnits": self._compute_units.name,
        }

        (result, _) = await self._service.call_method(
            name="MLModelService.Unload",
            params=params,
        )

        self._validate_result(result=result)
        self._state = _RemoteMLModelService.State.Unloaded

    @staticmethod
    async def load_on_device(
        device: Device,
        compiled_model_path: _Union[_Path, str],
        compute_units: _ComputeUnit,
        function_name: _Optional[str] = None,
        optimization_hints: _Optional[_Dict[str, _Any]] = None,
    ) -> _Type["_RemoteMLModelService"]:
        _RemoteMLModelService._validate_device_state(device=device)

        socket = _DeviceCtlSocket(device=device, source="Documents/modelrunnerd")
        remote_service = _DeviceCtlRemoteService(socket=socket)
        remote_model_service = _RemoteMLModelService(
            service=remote_service,
            compiled_model_path=compiled_model_path,
            compute_units=compute_units,
            function_name=function_name,
            optimization_hints=optimization_hints,
        )

        await remote_model_service.load()

        return remote_model_service
