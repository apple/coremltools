import numpy as np
import torch
import coremltools


def convert_to_coreml_inputs(input_description, inputs):
    """Convenience function to combine a CoreML model's input description and
    set of raw inputs into the format expected by the model's predict function.
    """
    coreml_inputs = {str(x): inp.numpy() for x, inp in zip(input_description, inputs)}
    return coreml_inputs

def convert_to_mlmodel(model_spec, inputs):
    return coremltools.converters.torch.convert(model_spec, inputs)

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

def convert_and_compare(model_spec, inputs, atol=1e-5):
    if isinstance(model_spec, str):
        torch_model = torch.jit.load(model_spec)
    else:
        torch_model = model_spec

    if not isinstance(inputs, list):
        inputs = [inputs]
    mlmodel = convert_to_mlmodel(model_spec, inputs)

    coreml_inputs = convert_to_coreml_inputs(mlmodel.input_description, inputs)
    coreml_outputs = [str(x) for x in mlmodel.output_description]
    coreml_result = mlmodel.predict(coreml_inputs)
    coreml_result = coreml_result[coreml_outputs[0]]

    torch_result = torch_model(*inputs)
    torch_result = torch_result.detach().numpy()

    np.testing.assert_equal(coreml_result.shape, torch_result.shape)
    np.testing.assert_allclose(
        coreml_result, torch_result, atol=atol,
    )

def run_numerical_test(input_shape, model, places=5):
    input_data = generate_input_data(input_shape)
    torch_model = trace_model(model, input_data)
    convert_and_compare(torch_model, input_data, atol=10.0 ** -places)