# Copyright (c) 2020, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os.path

import torch

from ...models import MLModel
from ..nnv2 import converter as NNV2_converter
from ..nnv2.frontend.torch.converter import TorchConverter


def convert(model_spec, inputs):
    """
    Convert Pytorch .pt file to nnv2 CoreML format.

    model_spec: String path to .pt file, or a TorchScript object representing the model to convert.
    inputs: List of torch.Tensor inputs to the model.
        TODO: Allow @inputs to describe variable size inputs.
    """

    if isinstance(model_spec, str):
        filename = os.path.abspath(model_spec)
        torchir = torch.jit.load(filename)
    elif isinstance(model_spec, torch.jit.ScriptModule):
        torchir = model_spec
    else:
        raise TypeError(
            "@model_spec must either be a PyTorch .pt file or a TorchScript object, received: {}".format(
                type(model_spec)
            )
        )

    converter = TorchConverter(torchir, inputs)
    prog = converter.convert()
    proto = NNV2_converter.convert(
        prog, convert_from="NitroSSA", convert_to="nnv1_proto"
    )
    return MLModel(proto, useCPUOnly=True)
