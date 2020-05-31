from __future__ import absolute_import

import gc
import coremltools

from coremltools.converters.mil.converter import _convert
from coremltools.converters.mil.mil import Program
from coremltools._deps import HAS_TORCH, HAS_TF, HAS_TF_2
from coremltools.converters._profile_utils import profile

if HAS_TORCH:
    import torch
    from coremltools.converters.mil.frontend.torch.load import _torchscript_from_model as pytorch_load
if HAS_TF:
    from coremltools.converters.mil.frontend.tensorflow.load import _tf_graph_from_model as tf1_load
if HAS_TF_2:
    from coremltools.converters.mil.frontend.tensorflow2.load import _tf_graph_from_model as tf2_load


@profile
def convert(model,
            source="auto",
            inputs=None,
            outputs=None,
            **kwargs):
    """

    Method to convert neural networks represented in Tensorflow or Pytorch formats
    to the Core ML model format. This method will choose a convert method based on
    the type of model passed in. For kwargs specific to a model type (Tensorflow, Torch etc.), 
    look in the load.py of the converter for that model type.

    Parameters
    ----------
    model:
        an object representing a neural network model defined in one of Tensorflow 1,
        Tensorflow 2 or Pytorch formats

        Depending on the source framework, type of model is one of the following:

        For Tensorflow versions 1.x:
            - frozen tf.Graph object
            - path to a frozen .pb file
        For Tensorflow versions 2.x:
            - tf.Graph object
            - tf.keras model object
            - path to a .h5 saved keras model file
            - path to a saved model directory
            - list of concrete functions
        For Pytorch:
            - a TorchScript object
            - path to a .pt file

    source: str (optional)
        one of "auto" (default), "tensorflow", "pytorch"

    inputs: list (optional)
        For Tensorflow:
            list of tuples or list of strings
            If [tuple] : each tuple contains input tensor name and shape
            If [str]: each string is the name of the Placeholder input op in the TF graph
        For Pytorch
            a list of example inputs, which are any of:
            1. tensor
            2. tuple shape
            3. tuple of (name, (1. or 2.))

    outputs: list[str] (optional)
        For Tensorflow:
            (required)
            list of output op names
        For Pytorch:
            (not required)
            list of output op names

    Returns
    -------
    model: MLModel
    A Core ML MLModel object

    Examples
    --------
    TODO: to format properly

    Tensorflow 1:

        mlmodel = coremltools.converters.convert(model='frozen_model_mobilenet.pb',
                                                 inputs=[('input', (1, 224, 224, 3)),
                                                 outputs=['Softmax']]

        mlmodel.save('model_mobilenet.mlmodel')


    Tensorflow 2:

        mlmodel = coremltools.converters.convert(model='frozen_model_mobilenet.h5',
                                                 outputs=['Softmax']]

        mlmodel.save('model_mobilenet.mlmodel')

    Pytorch :

        model = torchvision.models.mobilenet_v2()
        model.eval()
        example_input = torch.rand(1, 3, 256, 256)
        traced_model = torch.jit.trace(model, example_input)

        mlmodel = coremltools.converters.convert(traced_model,
                                                inputs=[('input_name',example_input)]
                                                outputs=['output_name'])

        mlmodel.save('mobilenetv2.mlmodel')

    """

    source = source.lower()

    if inputs is not None:
        if not isinstance(inputs, list):
            msg = "\"inputs\" must be of type list"
            raise ValueError(msg)

    if source == "auto" and HAS_TF:
        try:
            tf1_load(model, outputs=outputs)
            source = "tensorflow"
        except:
            pass

    if source in {"auto", "tensorflow"} and HAS_TF_2:
        try:
            tf2_load(model)
            source = "tensorflow2"
        except:
            pass

    if source == "auto" and HAS_TORCH:
        try:
            pytorch_load(model)
            source = "pytorch"
        except:
            pass
            
    if source == "auto" and isinstance(model, Program):
        source = "mil"

    convert_to = kwargs.get('convert_to', 'nn_proto')
    kwargs.pop('convert_to', None)

    if source == "auto":
        msg = "Unable to determine the type of the model, i.e. the source framework. " \
              "Please provide the value of argument \"source\", from one of " \
              "\"tensorflow\", \"pytorch\"."
        raise ValueError(msg)

    elif source in {"tensorflow", "tensorflow2"}:

        if not HAS_TF:
            raise ValueError('Converter was called with source=tensorflow, but missing tensorflow package')
        if source == "tensorflow2" and not HAS_TF_2:
            raise ValueError('Converter was called with source=tensorflow2, but missing tensorflow2 package')

        if inputs is not None and len(inputs) > 0 and isinstance(inputs[0], tuple):
            inputs = dict(inputs)

        proto_spec = _convert(
                    model,
                    convert_from=source,
                    convert_to=convert_to,
                    inputs=inputs,
                    outputs=outputs,
                    **kwargs
                    )

    elif source == "pytorch":
        if 'example_inputs' in kwargs:
            msg = "Unexpected argument \"example_inputs\" found"
            raise ValueError(msg)

        if inputs is not None:
            if not isinstance(inputs, list):
                msg = "\"inputs\" must be of type list. Recieved: {}".format(inputs)
                raise ValueError(msg)
            if not all([isinstance(_input, (torch.Tensor, tuple)) for _input in inputs]):
                msg = "\"inputs\" list must contain torch.Tensors or tuples. Recieved: {}".format(inputs)
                raise ValueError(msg)

        if outputs is not None:
            if not isinstance(outputs, list):
                msg = "\"outputs\" must be of type list. Recieved: {}".format(outputs)
                raise ValueError(msg)
            if not all([isinstance(output, str) for output in outputs]):
                msg = "\"inputs\" list must contain strings. Recieved: {}".format(outputs)
                raise ValueError(msg)

        proto_spec = _convert(
                    model,
                    convert_from="torch",
                    convert_to=convert_to,
                    example_inputs=inputs,
                    outputs=outputs,
                    **kwargs
                    )
    
    elif source == "mil":
        if not isinstance(model, Program):
            msg = "Converter was asked to convert MIL input, but input is not a MIL program!"
            raise ValueError(msg)

        proto_spec = _convert(
                    model,
                    convert_from="mil",
                    convert_to=convert_to,
                    example_inputs=inputs,
                    **kwargs
                    )

    else:
        msg = "Unrecognized value of argument \"source\": {}. " \
              "It must be one of \"auto\", \"tensorflow\", \"pytorch\"."
        raise ValueError(msg.format(source))

    mlmodel = coremltools.models.MLModel(proto_spec, useCPUOnly=True)
    del proto_spec
    gc.collect()
    return mlmodel



