# Copyright (c) 2025, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
from collections import defaultdict

import numpy as np
import torch
from io import StringIO
import sys

import coremltools as ct
from coremltools.optimize.coreml.experimental._post_training_quantization import (
    _get_activation_calibration_stats,
)
from coremltools.test.optimize.coreml.test_passes import TestCompressionPasses
from coremltools.optimize.coreml.experimental._model_debugger import ModelDebugger
import coremltools.optimize as cto



class TestActivationQuantization:
    @staticmethod
    def _get_test_mlmodel_conv_relu():
        """A mlmodel with conv, relu"""

        # Prepare torch model.
        inputs = [ct.TensorType(name="data", shape=(5, 10, 4, 4))]
        input_data = [torch.rand(*i.shape.to_list()) for i in inputs]
        m = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=4),
            torch.nn.ReLU(),
        )
        torchmodel = torch.jit.trace(m, input_data)

        # Convert to mlmodel.
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_ONLY,
            compute_precision=ct.precision.FLOAT16,
        )

        return mlmodel

    @staticmethod
    def _get_test_mlmodel_boolean_type():
        """A mlmodel with boolean type intermediate tensor"""

        # Prepare torch model.
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear1 = torch.nn.Linear(28 * 28, 100)
                self.linear2 = torch.nn.Linear(28 * 28, 100)

            def forward(self, img):  # convert + flatten
                y1 = self.linear1(img)
                y2 = self.linear2(img)
                y = torch.logical_and(y1, y2)
                return y

        model = Net()
        inputs = [ct.TensorType(name="data", shape=(1, 28 * 28))]
        input_data = [torch.rand(*i.shape.to_list()) for i in inputs]
        torchmodel = torch.jit.trace(model, input_data)

        # Convert to mlmodel.
        mlmodel = ct.convert(
            torchmodel,
            inputs=inputs,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_ONLY,
            compute_precision=ct.precision.FLOAT16,
        )

        return mlmodel


class TestGetActivationStats(TestActivationQuantization):

    def test_activation_quantization(self):
        """
        Test the usage of linear_quantize_activations.
        """
        sample_data = []
        for _ in range(3):
            input_data = np.random.rand(5, 10, 4, 4)
            sample_data.append({"data": input_data})

        mlmodel = self._get_test_mlmodel_conv_relu()
        activation_quant_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.experimental.OpActivationLinearQuantizerConfig(
                mode="linear_symmetric", weight_threshold=10
            )
        )

        def _run_quantization_with_group_size(group_size, expected_batch_size):
            buffer = StringIO()
            original_stderr = sys.stderr
            sys.stderr = buffer
            mlmodel_activation_quantized = cto.coreml.experimental.linear_quantize_activations(
                mlmodel,
                activation_quant_config,
                sample_data,
                calibration_op_group_size=group_size,
            )
            sys.stderr = original_stderr
            output = buffer.getvalue()
            assert f"tensors batch-by-batch: {expected_batch_size} batches" in output
        
        # when setting group size to -1, all intermediate outputs are in the same batch,
        # hence we will only get 1 batch
        _run_quantization_with_group_size(-1, 1)
        # when setting group size to 1, all intermediate outputs are split into different batches,
        # hence there will be 3 batches
        _run_quantization_with_group_size(1, 3)


    def test_get_activation_calibration_stats_basic(self):
        """
        Calibration a floating point model with sample data.
        """
        sample_data = []
        for _ in range(3):
            input_data = np.random.rand(5, 10, 4, 4)
            sample_data.append({"data": input_data})

        mlmodel = self._get_test_mlmodel_conv_relu()
        _get_activation_calibration_stats(mlmodel, sample_data)

    def test_get_activation_calibration_stats_skip_invalid_ops(self):
        """
        Calibration a floating point model with sample data.
        rdar://130623705 A unit test for model with boolean type intermediate tensor.
        """

        # Prepare sample data
        sample_data = []
        for _ in range(3):
            input_data = np.random.rand(1, 28 * 28)
            sample_data.append({"data": input_data})

        mlmodel = self._get_test_mlmodel_boolean_type()
        _get_activation_calibration_stats(mlmodel, sample_data)

    def test_get_activation_calibration_stats_concat_surrounding_ops(self):
        """
        Calibration a floating point model with sample data.
        rdar://132017374 A unit test for model with concat would be surrounded by quantize/dequantize pairs after activation quantization.
        The activation_stats of concat surrounding nodes should be the same, so quantize/dequantize pairs could share same scale/zp.
        """

        # Prepare sample data
        sample_data = []
        for _ in range(3):
            input_data = np.random.rand(5, 10, 4, 4)
            sample_data.append({"data_0": input_data})

        # Loading a floating point mlmodel
        mlmodel = TestCompressionPasses._get_test_mlmodel_conv_concat()

        activation_stats = _get_activation_calibration_stats(mlmodel, sample_data)

        activation_stats_unique = set()
        for value in activation_stats.values():
            activation_stats_unique.add((value["rmin"], value["rmax"]))

        # Since mlmodel has a concat with 2 inputs and 1 output, we should see at least 3 rmin/rmax pairs are identical in activation_stats.
        # If we dedup rmin/rmax pairs with identical values, the length of unique values should at least reduced by 2 compared with original one.
        assert len(activation_stats) - len(activation_stats_unique) >= 2

    def test_calibration_stats_accumulate_across_samples(self):
        """
        Regression test: activation stats from an earlier sample must not be overwritten
        by a later sample with a narrower activation range.

        The large-input sample is placed FIRST so the bug (unconditionally overwriting
        rmin/rmax on every call instead of accumulating the running min/max) would discard
        it, leaving only the near-zero activations from the last sample.
        """
        mlmodel = self._get_test_mlmodel_conv_relu()

        # Large inputs produce activations with a wide range after Conv2d.
        # Near-zero inputs produce activations close to zero (ReLU clips negatives).
        sample_wide   = {"data": np.ones((5, 10, 4, 4), dtype=np.float32) * 100.0}
        sample_narrow = {"data": np.zeros((5, 10, 4, 4), dtype=np.float32)}
        stats_wide_only = _get_activation_calibration_stats(mlmodel, [sample_wide])
        stats_combined = _get_activation_calibration_stats(mlmodel, [sample_wide, sample_narrow])

        for key in stats_wide_only:
            assert stats_combined[key]["rmax"] >= stats_wide_only[key]["rmax"] - 1e-5, (
                f"rmax for '{key}' was overwritten by the later narrow sample: "
                f"combined={stats_combined[key]['rmax']:.4f}, "
                f"wide-only={stats_wide_only[key]['rmax']:.4f}"
            )



class TestRecordIntermediateOutput:
    """Regression tests for ModelDebugger.record_intermediate_output in _model_debugger.py."""

    def _make_stats(self):
        return defaultdict(dict)

    def test_second_call_narrower_does_not_overwrite(self):
        stats = self._make_stats()
        ModelDebugger.record_intermediate_output(np.array([0.0, 10.0]), "t", stats)
        ModelDebugger.record_intermediate_output(np.array([2.0, 5.0]), "t", stats)
        assert stats["t"]["rmin"] == 0.0, "rmin was overwritten by a narrower batch"
        assert stats["t"]["rmax"] == 10.0, "rmax was overwritten by a narrower batch"

    def test_rmax_from_first_call_not_overwritten_by_narrower_second_call(self):
        # Call 1 establishes rmax=10.0. Call 2 is narrower ([2, 5]).
        # Buggy code overwrites stats with the last call → rmax becomes 5.0.
        # Fixed code keeps the running max → rmax stays 10.0.
        stats = self._make_stats()
        ModelDebugger.record_intermediate_output(np.array([0.0, 10.0]), "t", stats)
        ModelDebugger.record_intermediate_output(np.array([2.0, 5.0]), "t", stats)
        assert stats["t"]["rmax"] == 10.0, "rmax was overwritten by a narrower second call"

    def test_rmin_from_first_call_not_overwritten_by_narrower_second_call(self):
        # Call 1 establishes rmin=-1.0. Call 2 is narrower ([2, 5]).
        # Buggy code overwrites stats with the last call → rmin becomes 2.0.
        # Fixed code keeps the running min → rmin stays -1.0.
        stats = self._make_stats()
        ModelDebugger.record_intermediate_output(np.array([-1.0, 8.0]), "t", stats)
        ModelDebugger.record_intermediate_output(np.array([2.0, 5.0]), "t", stats)
        assert stats["t"]["rmin"] == -1.0, "rmin was overwritten by a narrower second call"

        
    def test_many_calls_accumulate_global_extremes(self):
        stats = self._make_stats()
        batches = [
            np.array([3.0, 6.0]),
            np.array([1.0, 5.0]),
            np.array([4.0, 9.0]),
            np.array([2.0, 7.0]),
        ]
        for b in batches:
            ModelDebugger.record_intermediate_output(b, "t", stats)
        assert stats["t"]["rmin"] == 1.0
        assert stats["t"]["rmax"] == 9.0