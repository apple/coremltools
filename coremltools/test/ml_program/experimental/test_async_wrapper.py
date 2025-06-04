#  Copyright (c) 2025, Apple Inc. All rights reserved.
import gc
import os

import coremltools as ct
from coremltools.converters.mil import Builder as mb
import pytest

from coremltools.models.ml_program.experimental.async_wrapper import (
    MLModelAsyncWrapper,
)

class TestMLModelAsyncWrapper:
    @staticmethod
    def get_test_model():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 2, 3, 4)),
            ]
        )
        def prog(x):
            y = mb.const(val=1.2, name="y")
            x = mb.add(x=x, y=y, name="add")
            return x

        return prog
    
    @staticmethod
    @pytest.mark.asyncio
    async def test_source_model_remains_intact():
        prog = TestMLModelAsyncWrapper.get_test_model()
        mlmodel = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32)
        async_wrapper = MLModelAsyncWrapper.from_spec_or_path(
            spec_or_path=mlmodel.package_path, 
            weights_dir=mlmodel.weights_dir
        )

        await async_wrapper.load()
        del(async_wrapper)
        gc.collect()

        assert os.path.isdir(mlmodel.package_path) and len(os.listdir(mlmodel.package_path)) > 0, (
            f"Model resources unexpectedly deleted from {mlmodel.package_path}\n"
            "Expected: Source model directory remains intact after wrapper destruction\n")