#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pytest
import torch
import torch.nn as nn

import coremltools as ct
import coremltools.models.utils as coremltoolsutils
from coremltools import RangeDim, TensorType
from coremltools._deps import _IS_MACOS
from coremltools.converters.mil.mil.types.type_mapping import \
    nptype_from_builtin
from coremltools.converters.mil.testing_utils import ct_convert

from ..converter import torch_to_mil_types


class ModuleWrapper(nn.Module):
    """
    Helper class to transform torch function into torch nn module.
    This helps to keep the testing interface same for torch functional api.
    """
    def __init__(self, function, kwargs=None):
        super(ModuleWrapper, self).__init__()
        self.function = function
        self.kwargs = kwargs if kwargs else {}

    def forward(self, *args):
        return self.function(*args, **self.kwargs)


np.random.seed(1984)


def _flatten(objects):
    flattened_list = []
    for item in objects:
        if isinstance(item, (list, tuple)):
            flattened_list.extend(_flatten(item))
        else:
            flattened_list.append(item)
    return flattened_list


def _copy_input_data(input_data):
    if isinstance(input_data, (list, tuple)):
        return [_copy_input_data(x) for x in input_data]
    return input_data.clone().detach()


def contains_op(torch, op_string):
    return hasattr(torch, op_string)


def convert_to_coreml_inputs(input_description, inputs):
    """
    Convenience function to combine a CoreML model's input description and
    set of raw inputs into the format expected by the model's predict function.
    """
    flattened_inputs = _flatten(inputs)
    coreml_inputs = {
        str(x): inp.numpy().astype(np.float32) for x, inp in zip(input_description, flattened_inputs)
    }

    for k, v in coreml_inputs.items():
        if isinstance(v, np.ndarray) and v.ndim == 0:
            coreml_inputs[k] = np.expand_dims(v, axis=-1)

    return coreml_inputs


def convert_to_mlmodel(model_spec, tensor_inputs, backend=("neuralnetwork", "fp32"),
                       converter_input_type=None, compute_unit=ct.ComputeUnit.CPU_ONLY,
                       minimum_deployment_target=None):
    def _convert_to_inputtype(inputs):
        if isinstance(inputs, list):
            return [_convert_to_inputtype(x) for x in inputs]
        elif isinstance(inputs, tuple):
            return tuple([_convert_to_inputtype(x) for x in inputs])
        elif isinstance(inputs, TensorType):
            return inputs
        elif isinstance(inputs, torch.Tensor):
            return TensorType(shape=inputs.shape, dtype=torch_to_mil_types[inputs.dtype])
        else:
            raise ValueError(
                "Unable to parse type {} into InputType.".format(type(inputs))
            )

    if converter_input_type is None:
        inputs = list(_convert_to_inputtype(tensor_inputs))
    else:
        inputs = converter_input_type

    return ct_convert(model_spec, inputs=inputs, convert_to=backend,
                      source="pytorch", compute_units=compute_unit,
                      minimum_deployment_target=minimum_deployment_target)


def generate_input_data(input_size, rand_range=(0, 1)):
    r1, r2 = rand_range

    def random_data(spec):
        if isinstance(spec, TensorType):
            spec_shape = spec.shape.shape
            dtype = nptype_from_builtin(spec.dtype)
        else:
            spec_shape = spec
            dtype = np.float32

        static_shape = tuple([np.random.randint(dim.lower_bound, dim.upper_bound if dim.upper_bound > 0 else 10)
                              if isinstance(dim, RangeDim) else dim for dim in spec_shape])

        data = np.random.rand(*static_shape) if static_shape != () else np.random.rand()
        data = (r1 - r2) * data + r2
        return torch.from_numpy(np.array(data).astype(dtype))

    if isinstance(input_size, list):
        return [random_data(size) for size in input_size]
    else:
        return random_data(input_size)


def trace_model(model, input_data):
    model.eval()
    if isinstance(input_data, list):
        input_data = tuple(input_data)
    torch_model = torch.jit.trace(model, input_data)
    return torch_model


def flatten_and_detach_torch_results(torch_results):
    if isinstance(torch_results, (list, tuple)):
        return [x.detach().numpy() for x in _flatten(torch_results) if x is not None]
    # Do not need to flatten
    return [torch_results.detach().numpy()]


def convert_and_compare(
    input_data,
    model_spec,
    expected_results=None,
    atol=1e-4,
    rtol=1e-05,
    backend=("neuralnetwork", "fp32"),
    converter_input_type=None,
    compute_unit=ct.ComputeUnit.CPU_ONLY,
    minimum_deployment_target=None
):
    """
    If expected results is not set, it will by default
    be set to the flattened output of the torch model.

    Inputs:

    - input_data: torch.tensor or list[torch.tensor]
    """
    if isinstance(model_spec, str):
        torch_model = torch.jit.load(model_spec)
    else:
        torch_model = model_spec

    if not isinstance(input_data, (list, tuple)):
        input_data = [input_data]

    if expected_results is None:
        torch_input = _copy_input_data(input_data)
        expected_results = torch_model(*torch_input)
    expected_results = flatten_and_detach_torch_results(expected_results)
    mlmodel = convert_to_mlmodel(model_spec, input_data, backend=backend,
                                 converter_input_type=converter_input_type,
                                 compute_unit=compute_unit,
                                 minimum_deployment_target=minimum_deployment_target,)

    coreml_inputs = convert_to_coreml_inputs(mlmodel.input_description, input_data)

    if not _IS_MACOS or (mlmodel.is_package and coremltoolsutils._macos_version() < (12, 0)):
        return model_spec, mlmodel, coreml_inputs, None

    _, dtype = backend
    if mlmodel.compute_unit != ct.ComputeUnit.CPU_ONLY or (dtype == "fp16"):
        atol = max(atol * 100.0, 5e-1)
        rtol = max(rtol * 100.0, 5e-2)

    if not coremltoolsutils._has_custom_layer(mlmodel._spec):
        coreml_preds = mlmodel.predict(coreml_inputs)
        coreml_outputs = mlmodel._spec.description.output
        coreml_results = [
            coreml_preds[output.name] for output in coreml_outputs
        ]
        for torch_result, coreml_result in zip(expected_results,
                                               coreml_results):

            if torch_result.shape == ():
                torch_result = np.array([torch_result])
            np.testing.assert_equal(coreml_result.shape, torch_result.shape)
            np.testing.assert_allclose(coreml_result, torch_result, atol=atol, rtol=rtol)
    return model_spec, mlmodel, coreml_inputs, coreml_preds


class TorchBaseTest:
    testclassname = ''
    testmodelname = ''

    @pytest.fixture(autouse=True)
    def store_testname_with_args(self, request):
        TorchBaseTest.testclassname = type(self).__name__
        TorchBaseTest.testmodelname = request.node.name

    @staticmethod
    def run_compare_torch(
            input_data,
            model,
            expected_results=None,
            atol=1e-04,
            rtol=1e-05,
            input_as_shape=True,
            backend=("neuralnetwork", "fp32"),
            rand_range=(-1.0, 1.0),
            use_scripting=False,
            converter_input_type=None,
            compute_unit=ct.ComputeUnit.CPU_ONLY,
            minimum_deployment_target=None,
    ):
        """
        Traces a model and runs a numerical test.
        Args:
            input_as_shape <bool>: If true generates random input data with shape.
            expected_results <iterable, optional>: Expected result from running pytorch model.
            converter_input_type: If not None, then pass it to the "inputs" argument to the
                ct.convert() call.
        """
        model.eval()
        if input_as_shape:
            input_data = generate_input_data(input_data, rand_range)

        if use_scripting:
            model_spec = torch.jit.script(model)
        else:
            model_spec = trace_model(model, _copy_input_data(input_data))

        model_spec, mlmodel, coreml_inputs, coreml_results = \
            convert_and_compare(
                input_data,
                model_spec,
                expected_results=expected_results,
                atol=atol,
                rtol=rtol,
                backend=backend,
                converter_input_type=converter_input_type,
                compute_unit=compute_unit,
                minimum_deployment_target=minimum_deployment_target,
            )

        return model_spec, mlmodel, coreml_inputs, coreml_results, \
            TorchBaseTest.testclassname, TorchBaseTest.testmodelname
