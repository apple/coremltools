#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.models import MLModel
from coremltools.converters._profile_utils import _profile
from . import InputType, ImageType
from .mil.passes.common_pass import common_pass


class ConverterRegistry:
    frontends = {}
    backends = {}

    @staticmethod
    def frontend(converter):
        ConverterRegistry.frontends[converter.name] = converter
        return converter

    @staticmethod
    def backend(converter):
        ConverterRegistry.backends[converter.name] = converter
        return converter


@ConverterRegistry.frontend
class MILFrontend:
    name = "mil"

    def __call__(self, model, *args, **kwargs):
        if "inputs" in kwargs and kwargs["inputs"] is not None:
            inputs = kwargs["inputs"]
            if not isinstance(inputs, (list, tuple)):
                raise ValueError(
                    "Type of inputs should be list or tuple, got {} instead.".format(
                        type(inputs)
                    )
                )
            if not all([isinstance(i, InputType) for i in inputs]):
                raise ValueError(
                    "Type of inputs should be list or tuple of TensorType or ImageType, got {} instead.".format(
                        [type(i) for i in inputs]
                    )
                )

            for idx, inp in enumerate(inputs):
                # We set the default image format in MIL as NCHW, since only NCHW is
                # natively supported by MIL ops (ex. Conv/Pool/etc.)
                if isinstance(inp, ImageType) and inputs[idx].channel_first is None:
                    inputs[idx].channel_first = True
            model.set_main_input_types(tuple(inputs))
        return model


@ConverterRegistry.frontend
class TensorFlowFrontend:
    name = "tensorflow"

    def __call__(self, *args, **kwargs):
        from .frontend.tensorflow.load import TF1Loader

        tf1_loader = TF1Loader(*args, **kwargs)
        return tf1_loader.load()


@ConverterRegistry.frontend
class TensorFlow2Frontend:
    name = "tensorflow2"

    def __call__(self, *args, **kwargs):
        from .frontend.tensorflow2.load import TF2Loader

        tf2_loader = TF2Loader(*args, **kwargs)
        return tf2_loader.load()


@ConverterRegistry.frontend
class TorchFrontend:
    name = "pytorch"

    def __call__(self, *args, **kwargs):
        from .frontend.torch import load

        return load(*args, **kwargs)


@ConverterRegistry.backend
class NNProtoBackend:
    name = "nn_proto"

    def __call__(self, *args, **kwargs):
        from .backend.nn import load

        return load(*args, **kwargs)


@_profile
def mil_convert(
    model,
    convert_from,
    convert_to,
    **kwargs
):
    """
    Convert model from a specified frontend `convert_from` to a specified
    converter backend `convert_to`.

    Parameters
    ----------
    model: TF, PyTorch, or `coremltools.converters.mil.Program`.
        See `coremltools.converters.convert`

    convert_from: str
        The value must be one of ['tensorflow', 'tensorflow2',
        'pytorch', 'mil'] (aka name of a `ConverterRegistry.frontend`).

    convert_to: str
       Value must be one of ['nn_proto', 'mil'] (aka name of
       `ConverterRegistry.backend`). See `coremltools.converters.convert`

    Returns
    -------
    model: `coremltools.models.MLModel` or
    `coremltools.converters.mil.Program`
        See `coremltools.converters.convert`
    """
    proto = mil_convert_to_proto(model, convert_from, convert_to,
        ConverterRegistry, **kwargs)
    if convert_to == 'mil':
        return proto
    useCPUOnly = kwargs.get("useCPUOnly", False)
    return MLModel(proto, useCPUOnly=useCPUOnly)


def mil_convert_to_proto(
    model,
    convert_from,
    convert_to,
    converter_registry,
    **kwargs
):
    """
    Convert model to proto object.

    Parameters
    ----------
    model: See `mil_convert`

    convert_from: See `mil_convert`

    convert_to: See `mil_convert`

    converter_registry: `ConverterRegistry`
      Available frontend and backend converters

    Returns
    -------
    model: `coremltools.models.MLModel` or
    `coremltools.converters.mil.Program`
        See `coremltools.converters.convert`
    """
    frontend_converter_type = converter_registry.frontends.get(convert_from.lower())
    if not frontend_converter_type:
        msg = 'Frontend converter "{}" not implemented, must be one of: {}'
        raise NotImplementedError(
            msg.format(convert_from, list(converter_registry.frontends.keys()))
        )
    frontend_converter = frontend_converter_type()

    prog = frontend_converter(model, **kwargs)
    common_pass(prog)

    if convert_to == 'mil':
        return prog # Returns `coremltools.converters.mil.Program`

    backend_converter_type = converter_registry.backends.get(convert_to.lower())
    if not backend_converter_type:
        msg = 'Backend converter "{}" not implemented, must be one of: {}'
        raise NotImplementedError(
            msg.format(convert_to, list(converter_registry.backends.keys()))
        )
    backend_converter = backend_converter_type()
    out = backend_converter(prog, **kwargs)

    return out
