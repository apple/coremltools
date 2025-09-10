#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at
#  https://opensource.org/licenses/BSD-3-Clause
#
#  Test suite for RMSNorm feature (Issue #2585)

import pytest
from coremltools._deps import _HAS_TORCH
import numpy as np


# Check if pytorch module is installed
# Also, check if pytorch and coremltools' versions are compatible for this test
if _HAS_TORCH:
    import torch
    import coremltools as ct

    # get package versions
    torch_major = int(torch.__version__.split('.')[0])
    ct_version_parts = ct.__version__.split('.')
    ct_major = int(ct_version_parts[0])

    # Run only on PyTorch 2.x and coremltools >= 8.x
    _TORCH_COMPATIBLE = torch_major >= 2
    _CT_COMPATIBLE = ct_major >= 8
    _VERSIONS_COMPATIBLE = _TORCH_COMPATIBLE and _CT_COMPATIBLE
else:
    _VERSIONS_COMPATIBLE = False


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not found")
@pytest.mark.skipif(not _VERSIONS_COMPATIBLE, reason="Incompatible versions")
class TestRMSNorm:
    """
    Test RMSNorm conversion from PyTorch to CoreML
    """

    @staticmethod
    @pytest.mark.parametrize(
        "input_shape, normalized_shape, elementwise_affine, eps, test_name",
        [
            # Standard tests
            ((2, 10, 768), 768, True, 1e-5, "standard_3d"),
            ((32, 128, 1024), 1024, True, 1e-5, "large_batch"),
            ((5, 512), 512, True, 1e-5, "2d_input"),
            ((1, 1, 256), 256, True, 1e-5, "singleton_dims"),

            # Without learnable parameters
            ((10, 512), 512, False, 1e-5, "no_weight"),
            ((2, 4, 512), 512, False, 1e-5, "no_weight_3d"),

            # Different epsilon values
            ((8, 256), 256, True, 1e-8, "small_epsilon"),
            ((8, 256), 256, True, 1e-3, "large_epsilon"),

            # Multiple axes normalization
            ((4, 8, 16, 32), (16, 32), True, 1e-5, "multi_axis"),
        ]
    )
    def test_rms_norm_conversion(
        input_shape,
        normalized_shape,
        elementwise_affine,
        eps,
        test_name
    ):
        """
        Test RMSNorm conversion with various configurations
        """
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.RMSNorm(
                    normalized_shape,
                    eps=eps,
                    elementwise_affine=elementwise_affine
                )

            def forward(self, x):
                return self.norm(x)

        model = TestModel()
        model.eval()

        example = torch.randn(input_shape)
        torch_out = model(example)
        traced = torch.jit.trace(model, example)
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(
                shape=input_shape,
                dtype=np.float32,
                name="input"
            )],
            outputs=[ct.TensorType(name="output", dtype=np.float32)],
            convert_to="mlprogram"
        )

        result = mlmodel.predict({"input": example.numpy()})
        coreml_out = result["output"]

        # Compare outputs
        np.testing.assert_allclose(
            torch_out.detach().numpy(),
            coreml_out,
            rtol=1e-2,  # 0.01 relative tolerance
            atol=1e-3,  # 0.001 absolute tolerance
            err_msg=f"Test '{test_name}' failed: outputs don't match"
        )

        # Verify no NaN or Inf are present in any tensor
        assert not np.isnan(coreml_out).any(), \
            f"Test '{test_name}' produced NaN values"
        assert not np.isinf(coreml_out).any(), \
            f"Test '{test_name}' produced Inf values"

    @staticmethod
    def test_edge_cases():
        """
        Test edge cases like zero inputs, very small values
        """
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.RMSNorm(512)

            def forward(self, x):
                return self.norm(x)

        model = TestModel()
        model.eval()

        # Test with zeros
        zeros = torch.zeros(2, 512)
        out_zeros = model(zeros)
        assert not torch.isnan(out_zeros).any(), \
            "RMSNorm produced NaN with zero input"

        # Test with very small values
        small = torch.full((2, 512), 1e-10)
        out_small = model(small)
        assert not torch.isinf(out_small).any(), \
            "RMSNorm produced Inf with small input"

    @staticmethod
    def test_dynamic_shapes():
        """
        Test RMSNorm with dynamic input shapes
        """
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.RMSNorm(768)

            def forward(self, x):
                return self.norm(x)

        model = TestModel()
        example = torch.randn(1, 10, 768)
        traced = torch.jit.trace(model, example)

        # Convert with flexible batch and sequence dimensions
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(
                shape=ct.EnumeratedShapes(
                    shapes=[[1, 10, 768], [2, 20, 768], [4, 50, 768]],
                    default=[1, 10, 768],
                ),
                dtype=np.float32,
                name="input"
            )],
            outputs=[ct.TensorType(name="output", dtype=np.float32)],
            convert_to="mlprogram"
        )

        # Test different shapes
        for shape in [(1, 10, 768), (2, 20, 768), (4, 50, 768)]:
            test_input = torch.randn(shape)
            torch_out = model(test_input)
            coreml_out = mlmodel.predict({
                "input": test_input.numpy()
            })["output"]

            np.testing.assert_allclose(
                torch_out.detach().numpy(),
                coreml_out,
                rtol=1e-2,
                atol=1e-3
            )
