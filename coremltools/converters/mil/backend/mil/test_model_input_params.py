#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import coremltools as ct
import pytest
import numpy as np

from coremltools._deps import _IS_MACOS
from coremltools.converters.mil.mil.builder import Builder as mb

class TestMILFlexibleShapes:
    from coremltools.converters.mil.mil.program import Symbol
    @mb.program(
        input_specs = [
            mb.TensorSpec(shape=[1, 3, Symbol("H"), Symbol("W")])
        ])
    def basic_network(x):
        return mb.relu(x=x)

    def test_mil_enumerated_multiarray(self):
        enumerated_shapes = tuple([(1, 3, 10, 10), (1, 3, 10, 20), (1, 3, 10, 30)])
        input_shape = [ct.TensorType(name="x", shape=ct.EnumeratedShapes(shapes=enumerated_shapes))]
        mlmodel = ct.convert(self.basic_network, source="milinternal", convert_to="mlprogram", inputs=input_shape)
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 1, "1 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(input_spec[0].name)
        assert input_spec[0].type.WhichOneof("Type") == "multiArrayType", "Expected multiArrayType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert input_spec[0].type.multiArrayType.WhichOneof("ShapeFlexibility") == "enumeratedShapes", "Expected enumeratedShapes in ShapeFlexibility"

        spec_default_shape = [s for s in input_spec[0].type.multiArrayType.shape]
        spec_enumerated_shapes = set()
        for enumerated in input_spec[0].type.multiArrayType.enumeratedShapes.shapes:
            spec_enumerated_shapes.add(tuple([s for s in enumerated.shape]))
        assert spec_default_shape == [1, 3, 10, 10], "Expected default shape to be [1, 3, 10, 10], got {} instead".format(str(spec_default_shape))
        assert spec_enumerated_shapes == set(enumerated_shapes), "Enumerated shape mismatch"

    def test_mil_enumerated_multiarray_with_default(self):
        enumerated_shapes = tuple([(1, 3, 10, 10), (1, 3, 10, 20), (1, 3, 10, 30)])
        input_shape = [ct.TensorType(name="x", shape=ct.EnumeratedShapes(shapes=enumerated_shapes, default=(1, 3, 10, 30)))]
        mlmodel = ct.convert(self.basic_network, source="milinternal", convert_to="mlprogram", inputs=input_shape)
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 1, "1 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(input_spec[0].name)
        assert input_spec[0].type.WhichOneof("Type") == "multiArrayType", "Expected multiArrayType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert input_spec[0].type.multiArrayType.WhichOneof("ShapeFlexibility") == "enumeratedShapes", "Expected enumeratedShapes in ShapeFlexibility"

        spec_default_shape = [s for s in input_spec[0].type.multiArrayType.shape]
        spec_enumerated_shapes = set()
        for enumerated in input_spec[0].type.multiArrayType.enumeratedShapes.shapes:
            spec_enumerated_shapes.add(tuple([s for s in enumerated.shape]))
        assert spec_default_shape == [1, 3, 10, 30], "Expected default shape to be [1, 3, 10, 10], got {} instead".format(str(spec_default_shape))
        assert spec_enumerated_shapes == set(enumerated_shapes), "Enumerated shape mismatch"

    def test_mil_enumerated_image(self):
        enumerated_shapes = tuple([(1, 3, 10, 10), (1, 3, 10, 20), (1, 3, 10, 30)])
        input_shape = [ct.ImageType(name="x", shape=ct.EnumeratedShapes(shapes=enumerated_shapes))]
        mlmodel = ct.convert(self.basic_network, source="milinternal", convert_to="mlprogram", inputs=input_shape)
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 1, "1 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(input_spec[0].name)
        assert input_spec[0].type.WhichOneof("Type") == "imageType", "Expected imageType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert input_spec[0].type.imageType.WhichOneof("SizeFlexibility") == "enumeratedSizes", "Expected enumeratedShapes in ShapeFlexibility"

        spec_H = input_spec[0].type.imageType.height
        spec_W = input_spec[0].type.imageType.width
        assert spec_H == 10 and spec_W == 10, "expected [H, W] == [10, 10], got [{}, {}] instead".format(spec_H, spec_W)

        spec_enumerated_shapes = set()
        for enumerated in input_spec[0].type.imageType.enumeratedSizes.sizes:
            spec_enumerated_shapes.add(tuple([1, 3, enumerated.height, enumerated.width]))
        assert spec_enumerated_shapes == set(enumerated_shapes), "Enumerated shape mismatch"

    def test_mil_enumerated_image_with_default(self):
        enumerated_shapes = tuple([(1, 3, 10, 10), (1, 3, 10, 20), (1, 3, 10, 30)])
        input_shape = [ct.ImageType(name="x", shape=ct.EnumeratedShapes(shapes=enumerated_shapes, default=(1, 3, 10, 30)))]
        mlmodel = ct.convert(self.basic_network, source="milinternal", convert_to="mlprogram", inputs=input_shape)
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 1, "1 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(input_spec[0].name)
        assert input_spec[0].type.WhichOneof("Type") == "imageType", "Expected imageType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert input_spec[0].type.imageType.WhichOneof("SizeFlexibility") == "enumeratedSizes", "Expected enumeratedShapes in ShapeFlexibility"

        spec_H = input_spec[0].type.imageType.height
        spec_W = input_spec[0].type.imageType.width
        assert spec_H == 10 and spec_W == 30, "expected [H, W] == [10, 30], got [{}, {}] instead".format(spec_H, spec_W)

        spec_enumerated_shapes = set()
        for enumerated in input_spec[0].type.imageType.enumeratedSizes.sizes:
            spec_enumerated_shapes.add(tuple([1, 3, enumerated.height, enumerated.width]))
        assert spec_enumerated_shapes == set(enumerated_shapes), "Enumerated shape mismatch"

    def test_mil_ranged_multiarray(self):
        input_shape = [ct.TensorType(name="x", shape=(1, 3, 10, ct.RangeDim(10, 30)))]
        mlmodel = ct.convert(self.basic_network, source="milinternal", convert_to="mlprogram", inputs=input_shape)
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 1, "1 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(input_spec[0].name)
        assert input_spec[0].type.WhichOneof("Type") == "multiArrayType", "Expected multiArrayType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert input_spec[0].type.multiArrayType.WhichOneof("ShapeFlexibility") == "shapeRange", "Expected shapeRange in ShapeFlexibility"

        spec_default_shape = [s for s in input_spec[0].type.multiArrayType.shape]
        ranged_shapes = [(1, 1), (3, 3), (10, 10), (10, 30)]
        spec_ranged_shapes = []
        for range_dim in input_spec[0].type.multiArrayType.shapeRange.sizeRanges:
            spec_ranged_shapes.append(tuple([range_dim.lowerBound, range_dim.upperBound]))
        assert spec_default_shape == [1, 3, 10, 10], "Expected default shape to be [1, 3, 10, 10], got {} instead".format(str(spec_default_shape))
        assert spec_ranged_shapes == ranged_shapes, "Enumerated shape mismatch"

    def test_mil_ranged_multiarray_with_default(self):
        input_shape = [ct.TensorType(name="x", shape=(1, 3, 10, ct.RangeDim(10, 30, default=20)))]
        mlmodel = ct.convert(self.basic_network, source="milinternal", convert_to="mlprogram", inputs=input_shape)
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 1, "1 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(input_spec[0].name)
        assert input_spec[0].type.WhichOneof("Type") == "multiArrayType", "Expected multiArrayType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert input_spec[0].type.multiArrayType.WhichOneof("ShapeFlexibility") == "shapeRange", "Expected shapeRange in ShapeFlexibility"

        spec_default_shape = [s for s in input_spec[0].type.multiArrayType.shape]
        ranged_shapes = [(1, 1), (3, 3), (10, 10), (10, 30)]
        spec_ranged_shapes = []
        for range_dim in input_spec[0].type.multiArrayType.shapeRange.sizeRanges:
            spec_ranged_shapes.append(tuple([range_dim.lowerBound, range_dim.upperBound]))
        assert spec_default_shape == [1, 3, 10, 20], "Expected default shape to be [1, 3, 10, 20], got {} instead".format(str(spec_default_shape))
        assert spec_ranged_shapes == ranged_shapes, "Enumerated shape mismatch"

    def test_mil_ranged_image(self):
        input_shape = [ct.ImageType(name="x", shape=(1, 3, 10, ct.RangeDim(10, 30)))]
        mlmodel = ct.convert(self.basic_network, source="milinternal", convert_to="mlprogram", inputs=input_shape)
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 1, "1 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(input_spec[0].name)
        assert input_spec[0].type.WhichOneof("Type") == "imageType", "Expected imageType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert input_spec[0].type.imageType.WhichOneof("SizeFlexibility") == "imageSizeRange", "Expected imageSizeRange in ShapeFlexibility"

        spec_H = input_spec[0].type.imageType.height
        spec_W = input_spec[0].type.imageType.width
        assert spec_H == 10 and spec_W == 10, "expected [H, W] == [10, 10], got [{}, {}] instead".format(spec_H, spec_W)

        spec_H_range = [input_spec[0].type.imageType.imageSizeRange.heightRange.lowerBound, input_spec[0].type.imageType.imageSizeRange.heightRange.upperBound]
        spec_W_range = [input_spec[0].type.imageType.imageSizeRange.widthRange.lowerBound, input_spec[0].type.imageType.imageSizeRange.widthRange.upperBound]
        assert spec_H_range == [10, 10], "Ranged height mismatch"
        assert spec_W_range == [10, 30], "Ranged width mismatch"

    def test_mil_ranged_image_with_default(self):
        input_shape = [ct.ImageType(name="x", shape=(1, 3, 10, ct.RangeDim(10, 30, default=20)))]
        mlmodel = ct.convert(self.basic_network, source="milinternal", convert_to="mlprogram", inputs=input_shape)
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 1, "1 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(input_spec[0].name)
        assert input_spec[0].type.WhichOneof("Type") == "imageType", "Expected imageType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert input_spec[0].type.imageType.WhichOneof("SizeFlexibility") == "imageSizeRange", "Expected imageSizeRange in ShapeFlexibility"

        spec_H = input_spec[0].type.imageType.height
        spec_W = input_spec[0].type.imageType.width
        assert spec_H == 10 and spec_W == 20, "expected [H, W] == [10, 20], got [{}, {}] instead".format(spec_H, spec_W)

        spec_H_range = [input_spec[0].type.imageType.imageSizeRange.heightRange.lowerBound, input_spec[0].type.imageType.imageSizeRange.heightRange.upperBound]
        spec_W_range = [input_spec[0].type.imageType.imageSizeRange.widthRange.lowerBound, input_spec[0].type.imageType.imageSizeRange.widthRange.upperBound]
        assert spec_H_range == [10, 10], "Ranged height mismatch"
        assert spec_W_range == [10, 30], "Ranged width mismatch"

class TestMILDefaultValues:
    @mb.program(
        input_specs = [
            mb.TensorSpec(shape=[1]),
            mb.TensorSpec(shape=[1])
        ])
    def basic_network(x, y):
        return mb.add(x=x, y=y, name="output")

    def test_mil_default_value_to_proto(self):
        program_input_spec = [ct.TensorType(name="x", shape=[1], default_value=np.array([1.0]).astype(np.float32)), ct.TensorType(name="y", shape=[1])]
        mlmodel = ct.convert(self.basic_network, convert_to="mlprogram", inputs=program_input_spec)
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 2, "2 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(input_spec[0].name)
        assert input_spec[0].type.WhichOneof("Type") == "multiArrayType", "Expected multiArrayType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert input_spec[0].type.multiArrayType.WhichOneof("defaultOptionalValue") == "floatDefaultValue", "Expected floatDefaultValue, got {} instead".format(input_spec[0].type.multiArrayType.WhichOneof("defaultOptionalValue"))
        assert input_spec[0].type.multiArrayType.floatDefaultValue == 1.0

    def test_mil_default_value_runtime(self):
        program_input_spec = [ct.TensorType(name="x", shape=[1], default_value=np.array([1.0]).astype(np.float32)), ct.TensorType(name="y", shape=[1])]
        mlmodel = ct.convert(self.basic_network, convert_to="mlprogram", inputs=program_input_spec)

        if not _IS_MACOS:
            # Can not get predictions unless on macOS.
            return

        res = mlmodel.predict({"x": np.array([3.]), "y": np.array([2.])})
        assert res["output"][0] == 5.0

        res = mlmodel.predict({"y": np.array([2.])})
        assert res["output"][0] == 3.0
