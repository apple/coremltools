from __future__ import absolute_import

import gc
import coremltools
from six import string_types

from coremltools.converters.mil.input_types import InputType, ClassifierConfig
from coremltools.converters.mil.converter import _convert
from coremltools.converters.mil.mil import Program
from coremltools._deps import HAS_TORCH, HAS_TF_1, HAS_TF_2
from coremltools.converters._profile_utils import profile
from coremltools import __version__ as ct_version
from coremltools.models import _METADATA_VERSION, _METADATA_SOURCE


if HAS_TF_1:
    import tensorflow as tf
    from coremltools.converters.mil.frontend.tensorflow.load import TF1Loader
if HAS_TF_2:
    import tensorflow as tf
    from coremltools.converters.mil.frontend.tensorflow2.load import TF2Loader

if HAS_TORCH:
    import torch
    from coremltools.converters.mil.frontend.torch.load import _torchscript_from_model as pytorch_load


@profile
def convert(model,
            source="auto",
            inputs=None,
            outputs=None,
            classifier_config=None,
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
        If inputs is provided, it should be a list of TensorType or ImageType

    outputs: list[str] (optional)
        For Tensorflow 1:
            (required)
            list of output op names
        For Tensorflow 2:
            (not required)
            list of output op names
        For Pytorch:
            (not required)
            list of output op names

    classifier_config: ClassifierConfig class (optional)
        The configuration if the mlmodel is intended to be a classifier.

    Returns
    -------
    model: MLModel
    A Core ML MLModel object

    Examples
    --------
    TODO: to format properly

    Tensorflow 1:

        input = coremltools.TensorType(name='input', shape=(1, 224, 224, 3))
        mlmodel = coremltools.convert(model='frozen_model_mobilenet.pb',
                                      inputs=[input],
                                      outputs=['Softmax']]

        mlmodel.save('model_mobilenet.mlmodel')


    Tensorflow 2:

        mlmodel = coremltools.convert(model='frozen_model_mobilenet.h5',
                                      outputs=['Softmax']]

        mlmodel.save('model_mobilenet.mlmodel')

    Pytorch :

        model = torchvision.models.mobilenet_v2()
        model.eval()
        example_input = torch.rand(1, 3, 256, 256)
        traced_model = torch.jit.trace(model, example_input)

        input = coremltools.TensorType(name='input_name', shape=(1, 3, 256, 256))
        mlmodel = coremltools.convert(traced_model,
                                                inputs=[input]
                                                outputs=['output_name'])

        mlmodel.save('mobilenetv2.mlmodel')

    Unknown/Flexible Shapes:
        # -1 in shape is meant to be a fully flexible shape, or a shape within range [1, inf)

        from coremltools import TensorType
        flexible_input = TensorType(name='input', shape=(1, -1, -1, 3))
        mlmodel = coremltools.convert(model='frozen_style_transfer.pb',
                                                 inputs=[flexible_input],
                                                 outputs=['Softmax'])

    Ranged Shapes:
        # By using RangeDim, we allow users to provide input that is ONLY within the interval

        from coremltools import TensorType, RangeDim
        range_input = TensorType(name='input', shape=(1, RangeDim(220, 230), -1, 3))
        mlmodel = coremltools.convert(model='frozen_style_transfer.pb',
                                                 inputs=[range_input],
                                                 outputs=['Softmax'])

    Enumerated Shapes:
        # EnumeratedShapes allows the mlmodel to take different kinds of shapes.

        from coremltools import TensorType, EnumeratedShapes
        shape_1 = (1, 224, 224, 3)
        shape_2 = (1, 225, 225, 3)
        shape_3 = (1, 300, 300, 3)
        enumerated_shapes = EnumeratedShapes(shapes=[shape1, shape2, shape3])
        enumerated_inputs = TensorType(name='input', shape=enumerated_shapes)
        mlmodel = coremltools.convert(model='frozen_style_transfer.pb',
                                                 inputs=[enumerated_inputs],
                                                 outputs=['Softmax'])

    Optimized Shaping:
        # If default is provided in Shape class, the default shape of the produced model will be set.
        # By pre-setting the default shape, memory pre-allocation could be done when model is being loaded.

        from coremltools import Shape, TensorType, RangeDim, EnumeratedShapes

        flexible_input = TensorType(name='input', shape=Shape(shape=(1, -1, -1, 3), default=(1, 224, 224, 3)))
        range_input = TensorType(name='input', shape=Shape(shape=(1, RangeDim(220, 230), -1, 3), default=(1, 224, 224, 3)))

        shape_1 = (1, 224, 224, 3)
        shape_2 = (1, 225, 225, 3)
        shape_3 = (1, 300, 300, 3)
        enumerated_shapes = EnumeratedShapes(shapes=[shape1, shape2, shape3], default=shape_1)
        enumerated_inputs = TensorType(name='input', shape=enumerated_shapes)

        mlmodel = coremltools.convert(model='frozen_style_transfer.pb',
                                                 inputs=[flexible_input], # This could be flexible_input/range_input/enumerated_inputs
                                                 outputs=['Softmax'])

    """
    source = source.lower()
    if not source in {'auto', 'tensorflow', 'pytorch'}:
        msg = "Unrecognized value of argument \"source\": {}. " \
              "It must be one of \"auto\", \"tensorflow\", \"pytorch\"."
        raise ValueError(msg.format(source))

    if inputs is not None:
        if not isinstance(inputs, list):
            msg = "\"inputs\" must be of type list"
            raise ValueError(msg)

    if classifier_config is not None:
        if not isinstance(classifier_config, ClassifierConfig):
            msg = "\"classfier_config\" must be of type ClassifierConfig"
            raise ValueError(msg)

    if source == "tensorflow" and HAS_TF_2:
        source = "tensorflow2"

    if source == "auto" and HAS_TF_1:
        try:
            loader = TF1Loader(model, outputs=outputs)
            loader._graph_def_from_model(outputs=outputs)
            source = "tensorflow"
        except:
            pass

    if source == "auto" and HAS_TF_2:
        try:
            loader = TF2Loader(model, outputs=outputs)
            loader._graph_def_from_model(outputs=outputs)
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

        if source == 'tensorflow' and not HAS_TF_1:
            raise ValueError('Converter was called with source=tensorflow, but missing tensorflow package')

        if inputs is not None and not all([isinstance(_input, InputType) for _input in inputs]):
            raise ValueError('Input should be a list of TensorType or ImageType')

        proto_spec = _convert(model,
                              convert_from=source,
                              convert_to=convert_to,
                              inputs=inputs,
                              outputs=outputs,
                              classifier_config=classifier_config,
                              **kwargs
                              )

    elif source == "pytorch":
        if 'example_inputs' in kwargs:
            msg = "Unexpected argument \"example_inputs\" found"
            raise ValueError(msg)

        def _flatten_list(_inputs):
            ret = []
            for _input in _inputs:
                if isinstance(_input, (list, tuple)):
                    ret.extend(_flatten_list(_input))
                elif isinstance(_input, InputType):
                    ret.append(_input)
                else:
                    raise ValueError("Unknown type {} for flattening into InputType.".format(type(_input)))
            return ret

        if inputs is not None and not all([isinstance(_input, InputType) for _input in _flatten_list(inputs)]):
            raise ValueError('Input should be a list/tuple (or nested lists/tuples) of TensorType or ImageType')
        if outputs is not None:
            if not isinstance(outputs, list):
                msg = "\"outputs\" must be of type list. Received: {}".format(outputs)
                raise ValueError(msg)
            if not all([isinstance(output, string_types) for output in outputs]):
                msg = "\"inputs\" list must contain strings. Received: {}".format(outputs)
                raise ValueError(msg)

        proto_spec = _convert(
                    model,
                    convert_from="torch",
                    convert_to=convert_to,
                    inputs=inputs,
                    outputs=outputs,
                    classifier_config=classifier_config,
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
                    classifier_config=classifier_config,
                    **kwargs
                    )


    model = coremltools.models.MLModel(proto_spec, useCPUOnly=True)

    del proto_spec
    gc.collect()

    # recording metadata: coremltools version, source framework and version
    if source in {'tensorflow', 'tensorflow2'} and (HAS_TF_1 or HAS_TF_2):
        src_pkg_version = "tensorflow=={0}".format(tf.__version__)
    elif source == 'pytorch' and HAS_TORCH:
        src_pkg_version = "torch=={0}".format(torch.__version__)
    else:
        src_pkg_version = 'unknown'

    model.user_defined_metadata[_METADATA_VERSION] = ct_version
    model.user_defined_metadata[_METADATA_SOURCE] = src_pkg_version

    return model
