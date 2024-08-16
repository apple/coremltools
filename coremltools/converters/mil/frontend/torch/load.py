#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os.path as _os_path
from typing import List, Optional, Union

import torch as _torch
from torch.jit._script import RecursiveScriptModule

from coremltools import _logger as logger
from coremltools._deps import _HAS_TORCH_EXPORT_API
from coremltools.converters.mil.frontend.torch.converter import TorchConverter
from coremltools.converters.mil.input_types import StateType, TensorType
from coremltools.converters.mil.mil.program import Program

from .converter import TorchConverter

if _HAS_TORCH_EXPORT_API:
    from torch.export import ExportedProgram


def load(
    spec: Union[RecursiveScriptModule, "ExportedProgram", str],
    inputs: List[TensorType],
    specification_version: int,
    debug: bool = False,
    outputs: Optional[List[TensorType]] = None,
    cut_at_symbols: Optional[List[str]] = None,
    use_default_fp16_io: bool = False,
    states: Optional[List[StateType]] = None,
    **kwargs
) -> Program:
    """
    Convert PyTorch model to mil CoreML format.

    Parameters
    ----------
    spec: It could be one of the following:
        - String path to .pt file containing serialized torchscript model
        - In memory TorchScript model of type torch.jit.ScriptModule
        - In memory EXIR program of type ExportedProgram
    inputs: Can be a singular element or list of elements of the following form
        1. Any subclass of InputType
        2. torch.Tensor (only shape and dtype will be used)
        3. list of (1. or 2.)
        Inputs are parsed in the flattened order that the model accepts them.
        If names are not specified: input keys for calling predict on the converted model
        will be internal symbols of the input to the graph.
        User can specify a subset of names.
    debug: bool, optional. Defaults to False.
        This flag should generally be False except for debugging purposes
        for diagnosing conversion errors. Setting this flag to True will
        print the list of supported and unsupported ops found in the model
        if conversion fails due to an unsupported op.
    outputs (optional): list[ct.InputType] or None
        list of either ct.TensorTypes or ct.ImageTypes (both of which are child classes of InputType)
        This is the value of the "outputs" argument, passed on by the user in "coremltools.convert" API.
    cut_at_symbols (optional): List of internal symbol name strings. Graph conversion will
        terminate once these symbols have been generated. For debugging use
        only.
    use_default_fp16_io (optional): bool. Defaults to False.
        When minimum_deployment_target set >= ct.target.iOS16 (the same as ct.target.macOS13),
        and the compute precision set to fp16, this flag is True.
        When True, fp32 i/o defaults to fp16.
    """

    if _HAS_TORCH_EXPORT_API and isinstance(spec, ExportedProgram):
        # TODO: rdar://115845792 ([Executorch] Handle user provided inputs/outputs in the convert API)
        if states:
            raise AssertionError("'states' argument should be None for ExportedProgram")
        model = spec
    else:
        model = _torchscript_from_spec(spec)

    converter = TorchConverter(
        model,
        inputs,
        outputs,
        cut_at_symbols,
        specification_version,
        use_default_fp16_io,
        states,
    )

    return _perform_torch_convert(converter, debug)


def is_torch_model(model_spec: Union[str, RecursiveScriptModule]) -> bool:
    if isinstance(model_spec, str) and (model_spec.endswith(".pt") or model_spec.endswith(".pth")):
        # PyTorch file path
        return True
    elif isinstance(model_spec, _torch.jit.ScriptModule):
        # PyTorch object
        return True
    return False


def _torchscript_from_spec(model_spec: Union[str, RecursiveScriptModule]) -> RecursiveScriptModule:
    if isinstance(model_spec, str) and (model_spec.endswith(".pt") or model_spec.endswith(".pth")):
        filename = _os_path.abspath(model_spec)
        try:
            return _torch.jit.load(filename)
        except Exception as e:
            logger.error("\n\nERROR - Could not load the PyTorch model. Got the following error:\n")
            raise e

    elif isinstance(model_spec, _torch.jit.ScriptModule):
        return model_spec
    elif _HAS_TORCH_EXPORT_API and isinstance(model_spec, ExportedProgram):
        return model_spec
    else:
        raise TypeError(
            "A PyTorch model must either be a .pt or .pth file, or a TorchScript object. Received: {}".format(
                type(model_spec)
            )
        )


def _perform_torch_convert(converter: TorchConverter, debug: bool) -> Program:
    try:
        prog = converter.convert()
    except RuntimeError as e:
        if debug and "convert function" in str(e):
            implemented, missing = converter.check_ops()
            print("the following model ops are IMPLEMENTED:")
            print("\n".join(["  " + str(x) for x in sorted(implemented)]))
            print("the following model ops are MISSING:")
            print("\n".join(["  " + str(x) for x in sorted(missing)]))
        raise e

    return prog
