# Copyright (c) 2025, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import torch

import coremltools as ct
from coremltools.optimize.coreml.experimental._post_training_quantization import (
    _get_activation_calibration_stats,
)


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
        mlmodel = self._get_test_mlmodel_conv_concat()

        activation_stats = _get_activation_calibration_stats(mlmodel, sample_data)

        activation_stats_unique = set()
        for value in activation_stats.values():
            activation_stats_unique.add((value["rmin"], value["rmax"]))

        # Since mlmodel has a concat with 2 inputs and 1 output, we should see at least 3 rmin/rmax pairs are identical in activation_stats.
        # If we dedup rmin/rmax pairs with identical values, the length of unique values should at least reduced by 2 compared with original one.
        assert len(activation_stats) - len(activation_stats_unique) >= 2
