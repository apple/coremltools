from __future__ import print_function as _

import logging
import os.path

import torch

from .converter import TorchConverter
from coremltools.converters.nnv2.nnv2_program.program import SsaProgram


def load(model_spec, debug=False, **kwargs):
    """
    Convert Pytorch .pt file to nnv2 CoreML format.

    Parameters
    ----------
    model_spec: String path to .pt file, or a TorchScript object representing
        the model to convert.
    debug: bool, optional. Defaults to False.
        This flag should generally be False except for debugging purposes
        for diagnosing conversion errors. Setting this flag to True will
        print the list of supported and unsupported ops found in the model
        if conversion fails due to an unsupported op.
    example_inputs: List of torch.Tensor inputs to the model.
        TODO: Allow @example_inputs to describe variable size inputs.
    cut_output_names: List of output name strings. Graph conversion will
        terminate once these symbols have been generated. For debugging use
        only.
    """

    torchscript = _torchscript_from_model(model_spec)

    inputs = kwargs["example_inputs"]
    cut_outputs = kwargs.get("cut_output_names", None)
    converter = TorchConverter(torchscript, inputs, cut_outputs)

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
