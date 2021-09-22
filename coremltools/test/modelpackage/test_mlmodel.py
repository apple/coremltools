# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import os
import pytest
import shutil
import torch

import coremltools as ct
from coremltools._deps import _IS_MACOS
from coremltools.converters.mil.mil.builder import Builder as mb
from coremltools.models.model import MLModel
from coremltools.models.utils import _macos_version


def test_mlmodel_demo(tmpdir):
    NUM_TOKENS = 3
    EMBEDDING_SIZE = 5

    class TestModule(torch.nn.Module):
        def __init__(self):
            super(TestModule, self).__init__()
            self.embedding = torch.nn.Embedding(NUM_TOKENS, EMBEDDING_SIZE)

        def forward(self, x):
            return self.embedding(x)

    model = TestModule()
    model.eval()

    example_input = torch.randint(high=NUM_TOKENS, size=(2,),
        dtype=torch.int64)
    traced_model = torch.jit.trace(model, example_input)
    mlmodel = ct.convert(
        traced_model,
        source='pytorch',
        convert_to='mlprogram',
        inputs=[
            ct.TensorType(
                name="input",
                shape=example_input.shape,
                dtype=example_input.numpy().dtype,
            )
        ],
        compute_precision=ct.precision.FLOAT32
    )
    # `coremltools_internal.convert` returns
    # `coremltools_internal.models.MLModel` for `mlprogram` and `neuralnetwork`
    # backend
    assert isinstance(mlmodel, MLModel)

    # mlpackage_path is a model package
    mlpackage_path = os.path.join(str(tmpdir), 'mymodel.mlpackage')
    mlmodel.save(mlpackage_path)

    # Read back the saved bundle and compile
    mlmodel2 = MLModel(mlpackage_path)

    if not _IS_MACOS or _macos_version() < (12, 0):
        # Can not get predictions unless on macOS 12 or higher.
        shutil.rmtree(mlpackage_path)
        return

    result = mlmodel2.predict(
        {"input": example_input.cpu().detach().numpy().astype(np.float32)},
        useCPUOnly=True,
    )

    # Verify outputs
    expected = model(example_input)
    name = list(result.keys())[0]
    np.testing.assert_allclose(result[name], expected.cpu().detach().numpy())

    # Cleanup package
    shutil.rmtree(mlpackage_path)
