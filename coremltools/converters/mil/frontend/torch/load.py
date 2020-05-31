from __future__ import print_function as _

import logging
import os.path

import torch

from .converter import TorchConverter
from coremltools.converters.mil.mil import Program


def load(model_spec, debug=False, **kwargs):
    """
    Convert Pytorch .pt file to mil CoreML format.

    Parameters
    ----------
    model_spec: String path to .pt file, or a TorchScript object representing
        the model to convert.
    debug: bool, optional. Defaults to False.
        This flag should generally be False except for debugging purposes
        for diagnosing conversion errors. Setting this flag to True will
        print the list of supported and unsupported ops found in the model
        if conversion fails due to an unsupported op.
    example_inputs: Can be a singular element or list of elements of the following form
        1. tuple of size
        2. torch.Tensor (only shape will be used)
        3. tuple of (name, (1. or 2.))
        Inputs are parsed in the flattened order that the model accepts them. 
        If names are not specified: input keys for calling predict on the converted model
        will be internal symbols of the input to the graph. 
        User can specify a subset of names.    
        TODO: Allow @example_inputs to describe variable size inputs.
    outputs (optional): List of output name strings. If specified: keys of output dictionary 
        will be these names in order of flattened returned outputs. If not specified:
        output dictionary keys will be the internal output symbols in the graph. 
        User can specify a subset of names.
    cut_at_symbols (optional): List of internal symbol name strings. Graph conversion will
        terminate once these symbols have been generated. For debugging use
        only.
    """

    torchscript = _torchscript_from_model(model_spec)

    inputs = kwargs["example_inputs"]
    outputs = kwargs.get("outputs", None)
    cut_at_symbols = kwargs.get("cut_at_symbols", None)
    converter = TorchConverter(torchscript, inputs, outputs, cut_at_symbols)

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
    except Exception as e:
        raise e

    return prog


def _torchscript_from_model(model_spec):

    if isinstance(model_spec, str) and model_spec.endswith(".pt"):
        filename = os.path.abspath(model_spec)
        return torch.jit.load(filename)
    elif isinstance(model_spec, torch.jit.ScriptModule):
        return model_spec
    else:
        raise TypeError(
            "@model must either be a PyTorch .pt file or a TorchScript object, received: {}".format(
                type(model_spec)
            )
        )
