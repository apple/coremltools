# Copyright (c) 2020, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging
import os.path

import torch
from six import string_types
from ...models import MLModel
from ..mil import converter as MIL_converter
from ..mil.frontend.torch.converter import TorchConverter


def convert(model_spec, inputs, check_only=False):
    """
    Convert Pytorch .pt file to mil CoreML format.

    model_spec: String path to .pt file, or a TorchScript object representing
        the model to convert.
    inputs: List of torch.Tensor inputs to the model.
        TODO: Allow @inputs to describe variable size inputs.
    check_only: If False (default), will convert the model as normal. If True,
        instead of converting the model this will print which ops in the model
        are implemented and which aren't.
    """

    logging.warning(
        "This API is deprecated. Please use coremltools.converters.convert() instead."
    )

    if isinstance(model_spec, string_types):
        filename = os.path.abspath(model_spec)
        torchscript = torch.jit.load(filename)
    elif isinstance(model_spec, torch.jit.ScriptModule):
        torchscript = model_spec
    else:
        raise TypeError(
            "@model_spec must either be a PyTorch .pt file or a TorchScript object, received: {}".format(
                type(model_spec)
            )
        )

    converter = TorchConverter(torchscript, inputs)
    if check_only:
        implemented, missing = converter.check_ops()
        all_ops = implemented.union(missing)
        if "loop" in all_ops or "if" in all_ops:
            print("Warning: control flow ops present, results will be incomplete")
        print("the following model ops are IMPLEMENTED:")
        print("\n".join(["  " + str(x) for x in sorted(implemented)]))
        print("the following model ops are MISSING:")
        print("\n".join(["  " + str(x) for x in sorted(missing)]))
    else:
        prog = converter.convert()
        proto = MIL_converter.convert(
            prog, convert_from="NitroSSA", convert_to="nn_proto"
        )
        return MLModel(proto, useCPUOnly=True)
