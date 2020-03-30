import numpy as np
import torch
import coremltools


def convert_to_coreml_inputs(input_description, inputs):
    """Convenience function to combine a CoreML model's input description and
    set of raw inputs into the format expected by the model's predict function.
    """
    coreml_inputs = {str(x): inp.numpy() for x, inp in zip(input_description, inputs)}
    return coreml_inputs


def convert_and_compare(model_spec, inputs, atol=1e-5):
    if isinstance(model_spec, str):
        torch_model = torch.jit.load(model_spec)
    else:
        torch_model = model_spec
    mlmodel = coremltools.converters.torch.convert(model_spec, [inputs])

    coreml_inputs = convert_to_coreml_inputs(mlmodel.input_description, [inputs])
    coreml_outputs = [str(x) for x in mlmodel.output_description]
    coreml_result = mlmodel.predict(coreml_inputs)
    coreml_result = coreml_result[coreml_outputs[0]]

    torch_result = torch_model(inputs)
    torch_result = torch_result.detach().numpy()

    np.testing.assert_equal(coreml_result.shape, torch_result.shape)
    np.testing.assert_allclose(
        coreml_result, torch_result, atol=atol,
    )
