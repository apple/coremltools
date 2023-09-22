#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from torch import Tensor, dtype

import coremltools as ct

TYPE_MAP = {
    "torch.float32": np.float32,
    "torch.float16": np.float16,
    "torch.int64": np.int64,
}


def to_coreml_tensor_type(name: str, tensor: Tensor):
    dtype = str(tensor.dtype)
    # TODO: rdar://115845948 ([Executorch] Handle inputs of shapes with dynamic dimensions)
    return ct.TensorType(name=name, dtype=TYPE_MAP[dtype], shape=tensor.shape)


def extract_inputs_from_edge_program(exported_program):

    module = exported_program.graph_module
    inputs_to_parameters = exported_program.graph_signature.inputs_to_parameters
    inputs_to_buffers = exported_program.graph_signature.inputs_to_buffers
    inputs = []
    for node in module.graph.nodes:
        if node.op == "placeholder" and node.meta is not None and "val" in node.meta:
            if isinstance(node.meta["val"], Tensor):
                if node.name not in inputs_to_parameters and node.name not in inputs_to_buffers:
                    inputs.append(to_coreml_tensor_type(node.name, node.meta["val"]))
            else:
                raise NotImplementedError("Only Tensor inputs handled yet")
    return inputs
