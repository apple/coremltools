#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os.path as _os_path

import torch as _torch

from coremltools import _logger as logger
from coremltools.converters.mil.input_types import InputType, TensorType

from .converter import TorchConverter, torch_to_mil_types


def load(model_spec, inputs, specification_version,
         debug=False, outputs=None, cut_at_symbols=None,
         **kwargs):
    """
    Convert PyTorch model to mil CoreML format.

    Parameters
    ----------
    model_spec: String path to .pt file, or a TorchScript object representing
        the model to convert.
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
    """
    torchscript = _torchscript_from_model(model_spec)

    if hasattr(torchscript, 'training') and torchscript.training:
        logger.warning("Model is not in eval mode. "
                         "Consider calling '.eval()' on your model prior to conversion")
    if type(torchscript) == _torch.jit._script.RecursiveScriptModule:
        logger.warning("Support for converting Torch Script Models is experimental. "
                         "If possible you should use a traced model for conversion.")

    inputs = _convert_to_torch_inputtype(inputs)
    converter = TorchConverter(torchscript, inputs, outputs, cut_at_symbols, specification_version)
    return _perform_torch_convert(converter, debug)


def _torchscript_from_model(model_spec):
    if isinstance(model_spec, str) and (model_spec.endswith(".pt") or model_spec.endswith(".pth")):
        filename = _os_path.abspath(model_spec)
        return _torch.jit.load(filename)
    elif isinstance(model_spec, _torch.jit.ScriptModule):
        return model_spec
    else:
        raise TypeError(
            "@model must either be a PyTorch .pt or .pth file or a TorchScript object, received: {}".format(
                type(model_spec)
            )
        )

def _convert_to_torch_inputtype(inputs):
    input_type = []
    for _input in inputs:
        if isinstance(_input, (list, tuple)):
            input_type.append(_convert_to_torch_inputtype(_input))
        elif isinstance(_input, InputType):
            if _input.shape is None:
                raise ValueError("'shape' must be provided in the 'inputs' argument for pytorch conversion")
            input_type.append(_input)
        elif isinstance(_input, _torch.Tensor):
            input_type.append(
                TensorType(
                    shape=_input.shape, dtype=torch_to_mil_types[_input.dtype]
                )
            )
        else:
            raise ValueError(
                "Unknown type {} for conversion to InputType.".format(type(_input))
            )
    return input_type

def _perform_torch_convert(converter, debug):
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
