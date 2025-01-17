# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from abc import ABC as _ABC
from typing import List as _List

from coremltools import _logger

try:
    from ..libcoremlpython import _MLModelProxy
except Exception as e:
    _logger.warning(f"Failed to load _MLModelProxy: {e}")
    _MLModelProxy = None

try:
    from ..libcoremlpython import _MLCPUComputeDeviceProxy
except Exception as e:
    _logger.warning(f"Failed to load _MLCPUComputeDeviceProxy: {e}")
    _MLCPUComputeDeviceProxy = None

try:
    from ..libcoremlpython import _MLGPUComputeDeviceProxy
except Exception as e:
    _logger.warning(f"Failed to load _MLGPUComputeDeviceProxy: {e}")
    _MLGPUComputeDeviceProxy = None

try:
    from ..libcoremlpython import _MLNeuralEngineComputeDeviceProxy
except Exception as e:
    _logger.warning(f"Failed to load _MLNeuralEngineComputeDeviceProxy: {e}")
    _MLNeuralEngineComputeDeviceProxy = None


class MLComputeDevice(_ABC):
    """
    Represents a compute device.

    The represented device is capable of running machine learning computations and other tasks like
    analysis and processing of images, sound, etc.
    """

    @classmethod
    def get_all_compute_devices(
        cls,
    ) -> _List["MLComputeDevice"]:
        """
        Returns the list of all of the compute devices that are accessible.

        Returns
        -------
        List[MLComputeDevice]
            The accessible compute devices.

        Examples
        --------
        .. sourcecode:: python

            compute_devices = (
                coremltools.models.compute_device.MLComputeDevice.get_all_compute_devices()
            )
        
        """
        return _MLModelProxy.get_all_compute_devices()


class MLCPUComputeDevice(MLComputeDevice):
    """
    Represents a CPU compute device.
    """

    def __init__(self, proxy):
        if _MLCPUComputeDeviceProxy is None or not isinstance(proxy, _MLCPUComputeDeviceProxy):
            raise TypeError("The proxy parameter must be of type _MLCPUComputeDeviceProxy.")
        self.__proxy__ = proxy


class MLGPUComputeDevice(MLComputeDevice):
    """
    Represents a GPU compute device.
    """

    def __init__(self, proxy):
        if _MLGPUComputeDeviceProxy is None or not isinstance(proxy, _MLGPUComputeDeviceProxy):
            raise TypeError("The proxy parameter must be of type _MLGPUComputeDeviceProxy.")
        self.__proxy__ = proxy


class MLNeuralEngineComputeDevice(MLComputeDevice):
    """
    Represents a Neural Engine compute device.
    """

    def __init__(self, proxy):
        if _MLNeuralEngineComputeDeviceProxy is None or not isinstance(
            proxy, _MLNeuralEngineComputeDeviceProxy
        ):
            raise TypeError(
                "The proxy parameter must be of type _MLNeuralEngineComputeDeviceProxy."
            )
        self.__proxy__ = proxy

    @property
    def total_core_count(self) -> int:
        """
        Get the total number of cores in the Neural Engine.

        Returns
        -------
        int
            The total number of cores in the Neural Engine.

        Examples
        --------
        .. sourcecode:: python

            compute_devices = (
                coremltools.models.compute_device.MLComputeDevice.get_all_compute_devices()
            )
            compute_devices = filter(
                lambda compute_device: isinstance(
                    compute_device, coremltools.models.compute_device.MLNeuralEngineComputeDevice
                ),
                compute_devices,
            )
            neural_engine_compute_device = next(compute_devices, None)
            neural_engine_core_count = (
                neural_engine_compute_device.total_core_count
                if neural_engine_compute_device is not None
                else 0
            )

        """
        return self.__proxy__.get_total_core_count()
