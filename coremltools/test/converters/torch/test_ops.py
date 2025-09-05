#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at
#  https://opensource.org/licenses/BSD-3-Clause
#
#  Test suite for dynamic padding conversion (Issue #2583)
#  These tests verify the fix for converting PyTorch pad operations with
#  runtime-determined padding values to Core ML.
#  The issue occurred in _translate_torch_args() when handling
#  dynamic padding values like (1, x.size(-1)).

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
class TestPadDynamicFix:
    """
    Test dynamic padding fix for Issue #2583 - torch.nn.functional.pad
    with x.size(-1)
    """

    @staticmethod
    @pytest.mark.parametrize(
        "input_size, pad_fn, expected_size, test_name",
        [
            # Dynamic padding tests
            (3, lambda x: (1, x.size(-1)), 7, "dynamic_right"),
            (5, lambda x: (0, x.size(-1)), 10, "dynamic_right_only"),
            (4, lambda x: (x.size(-1), 0), 8, "dynamic_left_only"),
            (2, lambda x: (x.size(-1), x.size(-1)), 6, "both_dynamic"),
        ]
    )
    def test_dynamic_padding(input_size, pad_fn, expected_size, test_name):
        """
        Test dynamic padding cases where pad values depend on input size
        """
        class TestModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.pad(x, pad_fn(x))

        model = TestModel()
        example = torch.rand(input_size)
        traced = torch.jit.trace(model, example)

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(
                shape=ct.EnumeratedShapes(
                    shapes=[[2], [3], [4], [5], [input_size]],
                    default=[input_size],
                ),
                dtype=np.float32,
                name="input"
            )],
            outputs=[ct.TensorType(name="output", dtype=np.float32)],
            convert_to="mlprogram"
        )

        result = mlmodel.predict({"input": example.numpy()})
        assert result["output"].shape[0] == expected_size, \
            f"Test '{test_name}' failed: expected shape ({expected_size},)," \
            f"got {result['output'].shape}"

    @staticmethod
    @pytest.mark.parametrize(
        "input_size,pad_fn,expected_size,test_name",
        [
            # Constant padding tests (regression test)
            (3, lambda x: (1, 2), 6, "both_constant"),
            (4, lambda x: (0, 3), 7, "constant_right_only"),
            (5, lambda x: (2, 0), 7, "constant_left_only"),
            (2, lambda x: (3, 4), 9, "large_constants"),
        ]
    )
    def test_constant_padding(input_size, pad_fn, expected_size, test_name):
        """
        Test constant padding cases - regression test
        """
        class TestModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.pad(x, pad_fn(x))

        model = TestModel()
        example = torch.rand(input_size)
        traced = torch.jit.trace(model, example)

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(
                shape=ct.EnumeratedShapes(
                    shapes=[[2], [3], [4], [5], [input_size]],
                    default=[input_size],
                ),
                dtype=np.float32,
                name="input"
            )],
            outputs=[ct.TensorType(name="output", dtype=np.float32)],
            convert_to="mlprogram"
        )

        result = mlmodel.predict({"input": example.numpy()})
        output = result["output"]

        # Verify shape
        assert output.shape[0] == expected_size, \
            f"Test '{test_name}' failed: expected shape ({expected_size},)," \
            f"got {output.shape}"

        # Verify padding values are zeros
        pad_config = pad_fn(example)
        left_pad, right_pad = pad_config

        if left_pad > 0:
            assert np.allclose(output[:left_pad], 0.0), \
                f"Test '{test_name}' failed: left padding should be zeros"

        assert np.allclose(
                output[left_pad:left_pad+input_size], example.numpy()
            ), \
            f"Test '{test_name}' failed: original values not preserved"

        if right_pad > 0:
            assert np.allclose(output[-right_pad:], 0.0), \
                f"Test '{test_name}' failed: right padding should be zeros"

    @staticmethod
    @pytest.mark.parametrize(
        "input_size,pad_fn,expected_size,test_name",
        [
            # Mixed padding tests
            (3, lambda x: (2, x.size(-1)), 8, "constant_left_dynamic_right"),
            (4, lambda x: (x.size(-1), 3), 11, "dynamic_left_constant_right"),
        ]
    )
    def test_mixed_padding(input_size, pad_fn, expected_size, test_name):
        """
        Test mixed padding cases with both constant and dynamic values
        """
        class TestModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.pad(x, pad_fn(x))

        model = TestModel()
        example = torch.rand(input_size)
        traced = torch.jit.trace(model, example)

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(
                shape=ct.EnumeratedShapes(
                    shapes=[[2], [3], [4], [5], [input_size]],
                    default=[input_size],
                ),
                dtype=np.float32,
                name="input"
            )],
            outputs=[ct.TensorType(name="output", dtype=np.float32)],
            convert_to="mlprogram"
        )

        result = mlmodel.predict({"input": example.numpy()})
        assert result["output"].shape[0] == expected_size, \
            f"Test '{test_name}' failed: expected shape ({expected_size},)," \
            f"got {result['output'].shape}"
