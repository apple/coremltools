#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import tempfile as _tempfile
import warnings as _warnings
from typing import Optional, Text, Tuple

from coremltools.converters._profile_utils import _profile
from coremltools.converters.mil import Program
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.types.symbolic import k_num_internal_syms, k_used_symbols
from coremltools.models import MLModel
from coremltools.models.model import _create_mlpackage

from . import ImageType, InputType
from .mil.passes.pass_pipeline import PassPipeline, PassPipelineManager


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
        specification_version = kwargs.get("specification_version", None)
        if specification_version is not None:
            max_opset_version, op = model._get_max_opset_version_and_op()
            if max_opset_version > specification_version:
                msg = (
                    "Please update the minimum_deployment_target to coremltools.target.{},"
                    " since op {} is only available in opset coremltools.target.{} or newer."
                ).format(max_opset_version.name, op.op_type, max_opset_version.name)
                raise ValueError(msg)

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
        from .frontend.torch.load import load

        return load(*args, **kwargs)


@ConverterRegistry.backend
class NNProtoBackend:
    name = "neuralnetwork"
    alias_names = []

    def __call__(self, *args, **kwargs):
        from .backend.nn.load import load

        return load(*args, **kwargs)


@ConverterRegistry.backend
class MILProtoBackend:
    name = "mlprogram"
    alias_names = []

    def __call__(self, *args, **kwargs):
        from .backend.mil.load import load as backend_load

        return backend_load(*args, **kwargs)


def _reset_conversion_state():
    '''
    Reset any stateful properties/variables that are populated during conversion.
    '''

    # Clear the "name_count" dict,
    # which is used to generate unique op names in the mil builder class.
    mb.name_count.clear()

    # Clear "k_used_symbols" dict, and the int counter "k_num_internal_syms" that are used to track symbolic names
    global k_used_symbols
    global k_num_internal_syms
    k_used_symbols.clear()
    k_num_internal_syms = 0


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
        weights_dir = _tempfile.TemporaryDirectory()
        kwargs["weights_dir"] = weights_dir.name

    proto, mil_program = mil_convert_to_proto(
                            model,
                            convert_from,
                            convert_to,
                            registry,
                            **kwargs
                         )

    _reset_conversion_state()

    if convert_to == 'milinternal':
        return mil_program  # mil program
    elif convert_to == 'milpython':
        return proto  # internal mil data structure

    elif convert_to == "mlprogram":
        package_path = _create_mlpackage(
            proto, kwargs.get("weights_dir"), kwargs.get("package_dir")
        )
        return modelClass(
            package_path,
            is_temp_package=not kwargs.get("package_dir"),
            mil_program=mil_program,
            skip_model_load=kwargs.get("skip_model_load", False),
            compute_units=compute_units,
        )

    return modelClass(proto,
                      mil_program=mil_program,
                      skip_model_load=kwargs.get('skip_model_load', False),
                      compute_units=compute_units)


def mil_convert_to_proto(
    model, convert_from, convert_to, converter_registry, main_pipeline=None, **kwargs
) -> Tuple[Optional[MLModel], Program]:
    """
    Convert model to proto object.

    Parameters
    ----------
    model: See `mil_convert`

    convert_from: See `mil_convert`

    convert_to: See `mil_convert`

    converter_registry: `ConverterRegistry`
      Available frontend and backend converters

    main_pipeline: `PassPipeline`
      The main pipeline with options set by users.
    """
    frontend_converter_type = converter_registry.frontends.get(convert_from.lower())
    if not frontend_converter_type:
        raise NotImplementedError(
            f'Frontend converter "{convert_from}" not implemented, must be '
            f"one of: {list(converter_registry.frontends.keys())}"
        )

    kwargs.setdefault("convert_from", convert_from)
    kwargs.setdefault("convert_to", convert_to)

    if main_pipeline is None:
        # If the client calls `mil_convert` directly, the `pass_pipeline` is None. To keep the
        # behaviour same as before, the quantization pass is removed in this situation.
        # TODO: rdar://106111553 ([Infra] Quantization Pass is skipped when `mil_convert` is called directly.)
        main_pipeline = PassPipeline()
        main_pipeline.remove_passes({"common::add_fp16_cast"})
    frontend_pipeline, backend_pipeline = _construct_other_pipelines(
        main_pipeline, convert_from, convert_to
    )

    frontend_converter = frontend_converter_type()
    prog = frontend_converter(model, **kwargs)
    PassPipelineManager.apply_pipeline(prog, frontend_pipeline)

    PassPipelineManager.apply_pipeline(prog, main_pipeline)

    prog._check_invalid_program()

    if convert_to == 'milinternal':
        return None, prog

    PassPipelineManager.apply_pipeline(prog, backend_pipeline)
    backend_converter_type = converter_registry.backends.get(convert_to.lower())
    if not backend_converter_type:
        raise NotImplementedError(
            f'Backend converter "{convert_to}" not implemented, must be '
            f"one of: {list(converter_registry.backends.keys())}"
        )
    backend_converter = backend_converter_type()
    out = backend_converter(prog, **kwargs)

    return out, prog


def _construct_other_pipelines(
    main_pipeline: PassPipeline, convert_from: Text, convert_to: Text
) -> Tuple[PassPipeline, PassPipeline]:
    """
    Construct other pipelines based on the main pipeline. It includes:
    - The frontend pipeline which will run in the frontend converter
    - The backend pipeline which will run in the backend converter
    As the main pipeline could have passes which also exists in the frontend/backend passes, we
    need to make sure the pass options are set properly in all pipelines.
    For example, if users set options to skip some vars in `const_elimination` pass, we want to make
    sure those vars are skipped not only in main_pipeline, but also in other pipelines wherever the
    `const_elimination` pass runs.

    TODO: rdar://106046237 ([Infra] Expose Backend and Frontend Pipeline to External Users)
    Currently users only control the passes in the main pipeline by passing `pass_pipeline` param.
    There are two reasons why we don't expose the frontend/backend pipelines at the current stage:
    - The frontend and backend specific passes need to be well documented.
    - The interface need more carefully design, as we don't want to provide too many params such as
      ct.convert(..., frontend_pipeline=xxx, backend_pipelien=xxx, main_pipeline=xxx) to overwhelm
      users.
    """
    # Set the main pipeline options specified by the user in frontend/backend pipeline.
    frontend_pipeline = PassPipeline.get_pipeline(f"frontend_{convert_from.lower()}")
    frontend_pipeline.set_options_by_another_pipeline(main_pipeline)
    backend_pipeline = PassPipeline.get_pipeline(f"backend_{convert_to.lower()}")
    backend_pipeline.set_options_by_another_pipeline(main_pipeline)

    # If a pass is skipped in the main pipeline, we also skip it in the frontend/backend pipeline.
    default_main_pipeline = PassPipeline.get_pipeline("default")
    passes_skipped_in_main = set(default_main_pipeline.passes) - set(main_pipeline.passes)
    frontend_pipeline.remove_passes(passes_skipped_in_main)
    backend_pipeline.remove_passes(passes_skipped_in_main)

    return frontend_pipeline, backend_pipeline
