import numpy as np
import torch
from coremltools.converters import convert
from coremltools.models import MLModel


def _flatten(object):
    flattened_list = []
    for item in object:
        if isinstance(item, (list, tuple)):
            flattened_list.extend(_flatten(item))
        else:
            flattened_list.append(item)
    return flattened_list


def convert_to_coreml_inputs(input_description, inputs):
    """Convenience function to combine a CoreML model's input description and
    set of raw inputs into the format expected by the model's predict function.
    """
    flattened_inputs = _flatten(inputs)
    coreml_inputs = {
        str(x): inp.numpy() for x, inp in zip(input_description, flattened_inputs)
    }
    return coreml_inputs


def convert_to_mlmodel(model_spec, inputs):
    if isinstance(inputs, tuple):
        inputs = list(inputs)
        
    mlmodel = convert(
        model_spec, inputs=inputs,
    )
    return mlmodel


def generate_input_data(input_size):
    if isinstance(input_size, list):
        return [torch.rand(_size) for _size in input_size]
    else:
        return torch.rand(input_size)


def trace_model(model, input_data):
    model.eval()
    if isinstance(input_data, list):
        input_data = tuple(input_data)
    torch_model = torch.jit.trace(model, input_data)
    return torch_model


def run_numerical_test(input_shape, model, places=5):
    input_data = generate_input_data(input_shape)
    torch_model = trace_model(model, input_data)
    convert_and_compare(torch_model, input_data, atol=10.0 ** -places)


def flatten_and_detach_torch_results(torch_results):
    if isinstance(torch_results, (list, tuple)):
        return [x.detach().numpy() for x in _flatten(torch_results)]
    # Do not need to flatten
    return [torch_results.detach().numpy()]


def convert_and_compare(model_spec, inputs, expected_results=None, atol=1e-5):
    """
        If expected results is not set, it will by default 
        be set to the flattened output of the torch model.
    """
    if isinstance(model_spec, str):
        torch_model = torch.jit.load(model_spec)
    else:
        torch_model = model_spec

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    if not expected_results:
        expected_results = flatten_and_detach_torch_results(torch_model(*inputs))

    mlmodel = convert_to_mlmodel(model_spec, inputs)
    coreml_inputs = convert_to_coreml_inputs(mlmodel.input_description, inputs)
    coreml_results = mlmodel.predict(coreml_inputs)
    sorted_coreml_results = [
        coreml_results[key] for key in sorted(coreml_results.keys())
    ]

    for torch_result, coreml_result in zip(expected_results, sorted_coreml_results):
        np.testing.assert_equal(coreml_result.shape, torch_result.shape)
        np.testing.assert_allclose(coreml_result, torch_result, atol=atol)
