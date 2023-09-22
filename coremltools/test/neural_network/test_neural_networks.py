# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import unittest

import coremltools
from coremltools.models.utils import (_get_custom_layer_names,
                                      _replace_custom_layer_name)
from coremltools.proto import Model_pb2


class CustomLayerUtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        spec = Model_pb2.Model()
        spec.specificationVersion = coremltools.SPECIFICATION_VERSION

        features = ["feature_1", "feature_2"]
        output = "output"
        for f in features:
            input_ = spec.description.input.add()
            input_.name = f
            input_.type.doubleType.MergeFromString(b"")

        output_ = spec.description.output.add()
        output_.name = output
        output_.type.doubleType.MergeFromString(b"")

        layer = spec.neuralNetwork.layers.add()
        layer.name = "custom1"
        layer.input.append("input")
        layer.output.append("temp1")
        layer.custom.className = "name1"

        layer2 = spec.neuralNetwork.layers.add()
        layer2.name = "custom2"
        layer2.input.append("temp1")
        layer2.output.append("temp2")
        layer2.custom.className = "name2"

        layer3 = spec.neuralNetwork.layers.add()
        layer3.name = "custom3"
        layer3.input.append("temp2")
        layer3.output.append("output")
        layer3.custom.className = "name1"

        self.spec = spec

    def test_get_custom_names(self):
        names = _get_custom_layer_names(self.spec)
        self.assertEqual(names, {"name1", "name2"})

    def test_change_custom_name(self):
        _replace_custom_layer_name(self.spec, "name1", "notname1")
        names = _get_custom_layer_names(self.spec)
        self.assertEqual(names, {"notname1", "name2"})
        # set it back for future tests
        _replace_custom_layer_name(self.spec, "notname1", "name1")
