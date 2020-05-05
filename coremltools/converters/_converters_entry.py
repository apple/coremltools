import coremltools
from coremltools.converters.nnv2.converter import _convert
from coremltools._deps import HAS_TORCH, HAS_TF_1, HAS_TF_2

if HAS_TORCH:
    from coremltools.converters.nnv2.frontend.torch.load import _torchscript_from_model as pytorch_load
if HAS_TF_1:
    from coremltools.converters.nnv2.frontend.tensorflow.load import _tf_graph_from_model as tf1_load
if HAS_TF_2:
    from coremltools.converters.nnv2.frontend.tensorflow2.load import _tf_graph_from_model as tf2_load



def convert(model,
            source="auto",
            inputs=None,
            outputs=None,
            **kwargs):
    """

    Method to convert neural networks represented in Tensorflow or Pytorch formats
    to the Core ML model format.

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
            a list of example inputs, in torch.tensor format

    outputs: list[str] (optional)
        For Tensorflow:
            (required)
            list of output op names
        For Pytorch:
            (not required)

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
                                                inputs=[example_input])

        mlmodel.save('mobilenetv2.mlmodel')

    """

    source = source.lower()
    if source.startswith('tensorflow'):
        source = 'tensorflow'

    if inputs is not None:
        if not isinstance(inputs, list):
            msg = "\"inputs\" must be of type list"
            raise ValueError(msg)

    if source == "auto" and HAS_TF_1:
        try:
            tf1_load(model)
            source = "tensorflow"
        except:
            pass

    if source == "auto" and HAS_TF_2:
        try:
            tf2_load(model)
            source = "tensorflow"
        except:
            pass

    if source == "auto" and HAS_TORCH:
        try:
            pytorch_load(model)
            source = "pytorch"
        except:
            pass

    convert_to = kwargs.get('convert_to', 'nnv1_proto')
    kwargs.pop('convert_to', None)

    if source == "auto":
        msg = "Unable to determine the type of the model, i.e. the source framework. " \
              "Please provide the value of argument \"source\", from one of " \
              "\"tensorflow\", \"pytorch\"."
        raise ValueError(msg)

    elif source == "tensorflow":

        if inputs is not None and isinstance(inputs[0], tuple):
            inputs = dict(inputs)

        if HAS_TF_1:
            proto_spec = _convert(
                        model,
                        convert_from="tensorflow",
                        convert_to=convert_to,
                        inputs=inputs,
                        outputs=outputs,
                        **kwargs
                        )
        elif HAS_TF_2:
            proto_spec = _convert(
                        model,
                        convert_from='tensorflow2',
                        convert_to=convert_to,
                        inputs=inputs,
                        outputs=outputs,
                        **kwargs
                        )
        else:
            raise ValueError('Converter was called with source=tensorflow, but missing tensorflow package')

    elif source == "pytorch":
        if 'example_inputs' in kwargs:
            msg = "Unexpected argument \"example_inputs\" found"
            raise ValueError(msg)

        proto_spec = _convert(
                    model,
                    convert_from="torch",
                    convert_to=convert_to,
                    example_inputs=inputs,
                    **kwargs
                    )

    else:
        msg = "Unrecognized value of argument \"source\": {}. " \
              "It must be one of \"auto\", \"tensorflow\", \"pytorch\"."
        raise ValueError(msg.format(source))

    mlmodel = coremltools.models.MLModel(proto_spec, useCPUOnly=True)
    return mlmodel



