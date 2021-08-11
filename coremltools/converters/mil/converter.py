#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
from . import InputType, ImageType
from .mil.passes.apply_common_pass_pipeline import apply_common_pass_pipeline
from coremltools.converters._profile_utils import _profile
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.quantization_passes import AbstractQuantizationPass
from coremltools.models import MLModel
from coremltools.models.model import _MODEL_FILE_NAME, _WEIGHTS_DIR_NAME
from coremltools.models.utils import _MLMODEL_EXTENSION, _MLPACKAGE_EXTENSION

try:
    from coremltools.libmodelpackage import ModelPackage
except ModuleNotFoundError:
    pass

import os as _os
import shutil as _shutil
import stat as _stat
import tempfile as _tempfile
import warnings as _warnings


class ConverterRegistry:
    frontends = {}
    backends = {}
    backend_alias_names = {}

    @staticmethod
    def frontend(converter):
        ConverterRegistry.frontends[converter.name] = converter
        return converter

    @staticmethod
    def backend(converter):
        ConverterRegistry.backends[converter.name] = converter
        if 'alias_names' in converter.__dict__:
            for name in converter.alias_names:
                ConverterRegistry.backend_alias_names[name] = converter.name
        return converter


@ConverterRegistry.frontend
class MILFrontend:
    name = "milinternal"

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
    name = "neuralnetwork"
    alias_names = []

    def __call__(self, *args, **kwargs):
        from .backend.nn import load

        return load(*args, **kwargs)

@ConverterRegistry.backend
class MILProtoBackend:
    name = "mlprogram"
    alias_names = []

    def __call__(self, *args, **kwargs):
        from .backend.mil import load as backend_load

        return backend_load(*args, **kwargs)


def _reset_conversion_state():
    '''
    Reset any stateful properties/variables that are populated during conversion.
    '''

    # Clear the "name_count" dict,
    # which is used to generate unique op names in the mil builder class.
    mb.name_count.clear()

@_profile
def mil_convert(
    model,
    convert_from,
    convert_to,
    compute_units,
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
        'pytorch', 'milinternal'] (aka name of a `ConverterRegistry.frontend`).

    compute_units: coremltools.ComputeUnit
        A enum with three possible values:
            - coremltools.ComputeUnit.ALL - use all compute units available, including the
                neural engine.
            - coremltools.ComputeUnit.CPU_ONLY - limit the model to only use the CPU.
            - coremltools.ComputeUnit.CPU_AND_GPU - use both the CPU and GPU, but not the
                neural engine.

    convert_to: str
       Value must be one of ['neuralnetwork', 'mlprogram', 'milinternal']
       See `coremltools.converters.convert`

    Returns
    -------
    model: `coremltools.models.MLModel` or
    `coremltools.converters.mil.Program`
        See `coremltools.converters.convert`
    """
    return _mil_convert(model, convert_from, convert_to, ConverterRegistry, MLModel, compute_units, **kwargs)


def _mil_convert(
    model,
    convert_from,
    convert_to,
    registry,
    modelClass,
    compute_units,
    **kwargs
):

    # Map "convert_to" values that correspond to the alias_names, to the actual supported registries
    if convert_to in registry.backend_alias_names:
        msg = "Please use '{}' instead of '{}' with the 'convert_to' argument. The latter will be removed in the future."
        _warnings.warn(msg.format(registry.backend_alias_names[convert_to], convert_to))
        convert_to = registry.backend_alias_names[convert_to]

    if convert_to == 'mlprogram':
        # mil_convert_to_proto places weight files inside the weights_dir
        weights_dir = _tempfile.mkdtemp()
        kwargs['weights_dir'] = weights_dir

        # To make sure everyone can read and write to this directory (on par with os.mkdir())
        _os.chmod(weights_dir, _stat.S_IRWXU | _stat.S_IRWXG | _stat.S_IRWXO)

    proto, mil_program = mil_convert_to_proto(
                            model,
                            convert_from,
                            convert_to,
                            registry,
                            **kwargs
                         )

    _reset_conversion_state()

    if convert_to == 'milinternal':
        return mil_program # mil program

    elif convert_to == 'mlprogram':
        # Save proto to disk
        proto_spec_str = proto.SerializeToString()
        spec_file = _tempfile.NamedTemporaryFile(suffix=_MLMODEL_EXTENSION)
        spec_file.write(proto_spec_str)
        spec_file.flush()

        # To make sure everyone can read this file
        _os.chmod(spec_file.name, _stat.S_IRUSR | _stat.S_IWUSR | _stat.S_IRGRP | _stat.S_IROTH)

        # If package directory is already provided, use that
        package_path = kwargs.get("package_dir")
        if not package_path:
            package_path = _tempfile.mkdtemp(suffix=_MLPACKAGE_EXTENSION)

        if _os.path.exists(package_path):
            _shutil.rmtree(package_path)

        package = ModelPackage(package_path)

        # Root model file is copied into the model package.
        package.setRootModel(spec_file.name, _MODEL_FILE_NAME, "com.apple.CoreML", "CoreML Model Specification");
        spec_file.close() # clean up spec file now that it is part of the model package

        # Weights bundle is copied into the model package. Changes to in-memory JSON is commited to disk when package goes out of scope.
        package.addItem(weights_dir, _WEIGHTS_DIR_NAME, "com.apple.CoreML", "CoreML Model Weights")
        _shutil.rmtree(weights_dir) # clean up weights now that it is part of the model package

        package = None

        return modelClass(package_path,
                          is_temp_package=not kwargs.get('package_dir'),
                          mil_program=mil_program,
                          skip_model_load=kwargs.get('skip_model_load', False),
                          compute_units=compute_units)

    return modelClass(proto,
                      mil_program=mil_program,
                      skip_model_load=kwargs.get('skip_model_load', False),
                      compute_units=compute_units)

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

    kwargs.setdefault("convert_to", convert_to)
    frontend_converter = frontend_converter_type()

    prog = frontend_converter(model, **kwargs)

    if convert_to.lower() != "neuralnetwork":
        passes = kwargs.get("transforms", list())
    else:
        # Post training Quantization Passes not available on NN proto backend.
        passes = [p for p in kwargs.get("transforms", list()) if not isinstance(p, AbstractQuantizationPass)]

    apply_common_pass_pipeline(prog, passes)

    if convert_to == 'milinternal':
        return None, prog # Returns (None, coremltools.converters.mil.Program)

    backend_converter_type = converter_registry.backends.get(convert_to.lower())
    if not backend_converter_type:
        msg = 'Backend converter "{}" not implemented, must be one of: {}'
        raise NotImplementedError(
            msg.format(convert_to, list(converter_registry.backends.keys()))
        )
    backend_converter = backend_converter_type()
    out = backend_converter(prog, **kwargs)

    return out, prog # Returns (Model_pb2.Model, coremltools.converters.mil.Program)
