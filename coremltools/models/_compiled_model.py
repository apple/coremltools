# Copyright (c) 2023, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from os.path import expanduser as _expanduser
from typing import Optional as _Optional
from typing import Type as _Type

from coremltools import ComputeUnit as _ComputeUnit
from coremltools.models.model import MLState as _MLState

from .model import MLModel as _MLModel
from .model import MLModelAsset as _MLModelAsset
from .model import _verify_optimization_hint_input
from .utils import _macos_version

try:
    from ..libcoremlpython import _MLModelProxy
except:
    _MLModelProxy = None


class CompiledMLModel:

    @staticmethod
    def _init_check(
        path: str,
        compute_units: _ComputeUnit,
        function_name: str,
        optimization_hints: _Optional[dict] = None,
        asset: _Optional[_MLModelAsset] = None,
    ):
        if _macos_version() < (10, 13):
            raise Exception("Loading compiled Core ML models is only support on macOS 10.13 or higher.")
        if _MLModelProxy is None:
            raise Exception("Unable to load any compiled models. This is most likely because"
                            " coremltools was installed from an egg rather than a wheel.")

        if not isinstance(path, str):
            raise TypeError('The "path" parameter must be of type "str".')

        if not asset is None and not isinstance(asset, _MLModelAsset):
            raise TypeError('The "asset" parameter must be of type "MLModelAsset".')

        if not isinstance(compute_units, _ComputeUnit):
            raise TypeError('The "compute_units" parameter must be of type: "coremltools.ComputeUnit".')

        if not isinstance(function_name, str):
            raise TypeError('The "function_name" parameter must be of type "str".')

        _verify_optimization_hint_input(optimization_hints)

    def __init__(
        self,
        path: str,
        compute_units: _ComputeUnit = _ComputeUnit.ALL,
        function_name: _Optional[str] = None,
        optimization_hints: _Optional[dict] = None,
        asset: _Optional[_MLModelAsset] = None,
    ):
        """
        Loads a compiled Core ML model.

        Parameters
        ----------
        path : str
            The path to a compiled model directory, ending in ``.mlmodelc``.

        compute_units : coremltools.ComputeUnit
            An enum with the following possible values:
                - ``coremltools.ComputeUnit.ALL``: Use all compute units available, including the
                  neural engine.
                - ``coremltools.ComputeUnit.CPU_ONLY``: Limit the model to only use the CPU.
                - ``coremltools.ComputeUnit.CPU_AND_GPU``: Use both the CPU and GPU, but not the
                  neural engine.
                - ``coremltools.ComputeUnit.CPU_AND_NE``: Use both the CPU and neural engine, but
                  not the GPU. Available only for macOS >= 13.0.

        optimization_hints : dict or None
            Keys are the names of the optimization hint, either 'reshapeFrequency' or 'specializationStrategy'.
            Values are enumeration values of type ``coremltools.ReshapeFrequency`` or ``coremltools.SpecializationStrategy``.

        asset : MLModelAsset or None
            The model asset.

        Examples
        --------
        .. sourcecode:: python

            my_compiled_model = ct.models.CompiledMLModel("my_model_path.mlmodelc")
            y = my_compiled_model.predict({"x": 3})

        See Also
        --------
        predict
        """
        if function_name is None:
            function_name = ""

        self._init_check(
            path=path,
            asset=asset,
            compute_units=compute_units,
            function_name=function_name,
            optimization_hints=optimization_hints,
        )

        path_or_asset = _expanduser(path)
        if asset is not None:
            path_or_asset = asset

        self.compute_unit = compute_units
        self.function_name = function_name
        self.path_or_asset = path_or_asset
        if optimization_hints is not None:
            self.optimization_hints = optimization_hints.copy()
        else:
            self.optimization_hints = None

        asset_proxy = asset.__proxy__ if asset is not None else None

        if self.optimization_hints is not None:
            optimization_hints_str_vals = {k: v.name for k, v in self.optimization_hints.items()}
        else:
            optimization_hints_str_vals = {}
        self._proxy = _MLModelProxy(
            path, compute_units.name, function_name, optimization_hints_str_vals, asset_proxy
        )

    @classmethod
    def from_asset(
        cls,
        asset: _MLModelAsset,
        compute_units: _ComputeUnit = _ComputeUnit.ALL,
        function_name: _Optional[str] = None,
        optimization_hints: _Optional[dict] = None,
    ) -> _Type["CompiledMLModel"]:
        """
        Creates a CompiledModel instance from an asset.

        Parameters
        ----------
        asset: MLModelAsset
            The model asset to create the compiled model from.

        compute_units : coremltools.ComputeUnit
            An enum with the following possible values:
                - ``coremltools.ComputeUnit.ALL``: Use all compute units available, including the
                  neural engine.
                - ``coremltools.ComputeUnit.CPU_ONLY``: Limit the model to only use the CPU.
                - ``coremltools.ComputeUnit.CPU_AND_GPU``: Use both the CPU and GPU, but not the
                  neural engine.
                - ``coremltools.ComputeUnit.CPU_AND_NE``: Use both the CPU and neural engine, but
                  not the GPU. Available only for macOS >= 13.0.

        optimization_hints : dict or None
            Keys are the names of the optimization hint, either 'reshapeFrequency' or 'specializationStrategy'.
            Values are enumeration values of type ``coremltools.ReshapeFrequency`` or ``coremltools.SpecializationStrategy``.

        Returns
        -------
        CompiledMLModel
            An instance of ``CompiledMLModel`` loaded from the provided asset.

        Examples
        --------
        .. sourcecode:: python

            my_model_asset = MLModelAsset.from_memory(spec_data)
            my_compiled_model = CompiledMLModel.from_asset(my_model_asset)
            y = my_compiled_model.predict({"x": 3})

        """
        return cls(
            asset=asset,
            path="",
            compute_units=compute_units,
            function_name=function_name,
            optimization_hints=optimization_hints,
        )


    def predict(self, data, state: _Optional[_MLState] = None):
        """
        Return predictions for the model.

        Parameters
        ----------
        data: dict[str, value] or list[dict[str, value]]
            Dictionary of data to use for predictions, where the keys are the names of the input features.
            For batch predictons, use a list of such dictionaries.

        state : MLState
            Optional state object as returned by ``make_state()``.

        Returns
        -------
        dict[str, value]
            Predictions as a dictionary where each key is the output feature name.

        list[dict[str, value]]
            For batch prediction, returns a list of the above dictionaries.

        Examples
        --------
        .. sourcecode:: python

            data = {"bedroom": 1.0, "bath": 1.0, "size": 1240}
            predictions = model.predict(data)

            data = [
                {"bedroom": 1.0, "bath": 1.0, "size": 1240},
                {"bedroom": 4.0, "bath": 2.5, "size": 2400},
            ]
            batch_predictions = model.predict(data)

        """
        _MLModel._check_predict_data(data)

        return _MLModel._get_predictions(
            self._proxy, _MLModel._update_float16_multiarray_input_to_float32, data, state
        )


    def make_state(self) -> _MLState:
        """
        Returns a new state object, which can be passed to the ``predict`` method.

        Examples
        --------
        .. sourcecode:: python

            state = model.make_state()
            predictions = model.predict(x, state)

        See Also
        --------
        predict
        """
        return _MLState(self._proxy.newState())
