#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import math
import os
import platform
import shutil
import tempfile
from typing import List, Union

import numpy as np
import pytest

import coremltools as ct
from coremltools import _SPECIFICATION_VERSION_IOS_18, proto
from coremltools.converters.mil import mil
from coremltools.converters.mil.converter import mil_convert as _mil_convert
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.mil.builder import Builder as mb
from coremltools.converters.mil.mil.ops.tests.iOS18.test_compression import (
    TestConstexprLut as _TestConstexprLut,
)
from coremltools.converters.mil.mil.program import Symbol
from coremltools.converters.mil.mil.types.type_mapping import string_to_nptype
from coremltools.models.utils import _macos_version


class TestWeightFileSerialization:
    @staticmethod
    @pytest.mark.parametrize(
        "dtype, opset_version",
        itertools.product(
            ["fp16", "fp32", "uint8", "int8", "uint16", "int16", "int32", "uint32"],
            [ct.target.iOS16, ct.target.iOS18],
        ),
    )
    def test_weight_serialization(dtype, opset_version):
        if dtype == "uint32":
            # There is a pass that casts the output to CoreML supported dtype.
            # uint32 will fail because `cast` op doesn't accept such input type.
            pytest.skip("uint32 is not supported in `cast` op.")
        if dtype in ["uint8", "int8", "uint16", "int16"] and opset_version == ct.target.iOS16:
            # iOS16 doesn't support the above dtype either
            pytest.skip("dtype not support in iOS16")

        if dtype in ["fp16", "fp32", "uint8", "int8"]:
            should_serialize_weight = True
        else:
            should_serialize_weight = opset_version >= ct.target.iOS18

        @mb.program(input_specs=[mb.TensorSpec((1,))], opset_version=opset_version)
        def prog(x):
            val = np.random.rand(1000).astype(string_to_nptype(dtype))
            return mb.const(val=val), mb.add(x=x, y=1.0)

        # we don't want the const to be constant folding after casting
        pipeline = ct.PassPipeline()
        pipeline.set_options("common::const_elimination", {"skip_const_by_size": "-1"})
        mlmodel = ct.convert(
            prog,
            minimum_deployment_target=opset_version,
            pass_pipeline=pipeline,
        )

        # check the weights are serialized as file value
        if ct.utils._macos_version() >= (15, 0):
            mil_file = open(os.path.join(mlmodel.get_compiled_model_path(), "model.mil"))
            mil_txt = mil_file.read()
            if should_serialize_weight:
                assert f"tensor<{dtype}, [1000]>(BLOBFILE" in mil_txt
            else:
                assert f"tensor<{dtype}, [1000]>(BLOBFILE" not in mil_txt


class TestMILFlexibleShapes:
    @mb.program(input_specs=[mb.TensorSpec(shape=[1, 3, Symbol("H"), Symbol("W")])])
    def basic_network(x):
        return mb.relu(x=x)

    def test_mil_enumerated_multiarray(self):
        enumerated_shapes = tuple([(1, 3, 10, 10), (1, 3, 10, 20), (1, 3, 10, 30)])
        input_shape = [ct.TensorType(name="x", shape=ct.EnumeratedShapes(shapes=enumerated_shapes))]
        mlmodel = ct.convert(
            self.basic_network, source="milinternal", convert_to="mlprogram", inputs=input_shape
        )
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 1, "1 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(
            input_spec[0].name
        )
        assert (
            input_spec[0].type.WhichOneof("Type") == "multiArrayType"
        ), "Expected multiArrayType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert (
            input_spec[0].type.multiArrayType.WhichOneof("ShapeFlexibility") == "enumeratedShapes"
        ), "Expected enumeratedShapes in ShapeFlexibility"

        spec_default_shape = [s for s in input_spec[0].type.multiArrayType.shape]
        spec_enumerated_shapes = set()
        for enumerated in input_spec[0].type.multiArrayType.enumeratedShapes.shapes:
            spec_enumerated_shapes.add(tuple([s for s in enumerated.shape]))
        assert spec_default_shape == [
            1,
            3,
            10,
            10,
        ], "Expected default shape to be [1, 3, 10, 10], got {} instead".format(
            str(spec_default_shape)
        )
        assert spec_enumerated_shapes == set(enumerated_shapes), "Enumerated shape mismatch"

    def test_mil_enumerated_multiarray_with_default(self):
        enumerated_shapes = tuple([(1, 3, 10, 10), (1, 3, 10, 20), (1, 3, 10, 30)])
        input_shape = [
            ct.TensorType(
                name="x",
                shape=ct.EnumeratedShapes(shapes=enumerated_shapes, default=(1, 3, 10, 30)),
            )
        ]
        mlmodel = ct.convert(
            self.basic_network, source="milinternal", convert_to="mlprogram", inputs=input_shape
        )
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 1, "1 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(
            input_spec[0].name
        )
        assert (
            input_spec[0].type.WhichOneof("Type") == "multiArrayType"
        ), "Expected multiArrayType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert (
            input_spec[0].type.multiArrayType.WhichOneof("ShapeFlexibility") == "enumeratedShapes"
        ), "Expected enumeratedShapes in ShapeFlexibility"

        spec_default_shape = [s for s in input_spec[0].type.multiArrayType.shape]
        spec_enumerated_shapes = set()
        for enumerated in input_spec[0].type.multiArrayType.enumeratedShapes.shapes:
            spec_enumerated_shapes.add(tuple([s for s in enumerated.shape]))
        assert spec_default_shape == [
            1,
            3,
            10,
            30,
        ], "Expected default shape to be [1, 3, 10, 10], got {} instead".format(
            str(spec_default_shape)
        )
        assert spec_enumerated_shapes == set(enumerated_shapes), "Enumerated shape mismatch"

    def test_mil_enumerated_image(self):
        enumerated_shapes = tuple([(1, 3, 10, 10), (1, 3, 10, 20), (1, 3, 10, 30)])
        input_shape = [ct.ImageType(name="x", shape=ct.EnumeratedShapes(shapes=enumerated_shapes))]
        mlmodel = ct.convert(
            self.basic_network, source="milinternal", convert_to="mlprogram", inputs=input_shape
        )
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 1, "1 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(
            input_spec[0].name
        )
        assert (
            input_spec[0].type.WhichOneof("Type") == "imageType"
        ), "Expected imageType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert (
            input_spec[0].type.imageType.WhichOneof("SizeFlexibility") == "enumeratedSizes"
        ), "Expected enumeratedShapes in ShapeFlexibility"

        spec_H = input_spec[0].type.imageType.height
        spec_W = input_spec[0].type.imageType.width
        assert (
            spec_H == 10 and spec_W == 10
        ), "expected [H, W] == [10, 10], got [{}, {}] instead".format(spec_H, spec_W)

        spec_enumerated_shapes = set()
        for enumerated in input_spec[0].type.imageType.enumeratedSizes.sizes:
            spec_enumerated_shapes.add(tuple([1, 3, enumerated.height, enumerated.width]))
        assert spec_enumerated_shapes == set(enumerated_shapes), "Enumerated shape mismatch"

    def test_mil_enumerated_image_with_default(self):
        enumerated_shapes = tuple([(1, 3, 10, 10), (1, 3, 10, 20), (1, 3, 10, 30)])
        input_shape = [
            ct.ImageType(
                name="x",
                shape=ct.EnumeratedShapes(shapes=enumerated_shapes, default=(1, 3, 10, 30)),
            )
        ]
        mlmodel = ct.convert(
            self.basic_network, source="milinternal", convert_to="mlprogram", inputs=input_shape
        )
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 1, "1 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(
            input_spec[0].name
        )
        assert (
            input_spec[0].type.WhichOneof("Type") == "imageType"
        ), "Expected imageType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert (
            input_spec[0].type.imageType.WhichOneof("SizeFlexibility") == "enumeratedSizes"
        ), "Expected enumeratedShapes in ShapeFlexibility"

        spec_H = input_spec[0].type.imageType.height
        spec_W = input_spec[0].type.imageType.width
        assert (
            spec_H == 10 and spec_W == 30
        ), "expected [H, W] == [10, 30], got [{}, {}] instead".format(spec_H, spec_W)

        spec_enumerated_shapes = set()
        for enumerated in input_spec[0].type.imageType.enumeratedSizes.sizes:
            spec_enumerated_shapes.add(tuple([1, 3, enumerated.height, enumerated.width]))
        assert spec_enumerated_shapes == set(enumerated_shapes), "Enumerated shape mismatch"

    def test_mil_ranged_multiarray(self):
        input_shape = [ct.TensorType(name="x", shape=(1, 3, 10, ct.RangeDim(10, 30)))]
        mlmodel = ct.convert(
            self.basic_network, source="milinternal", convert_to="mlprogram", inputs=input_shape
        )
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 1, "1 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(
            input_spec[0].name
        )
        assert (
            input_spec[0].type.WhichOneof("Type") == "multiArrayType"
        ), "Expected multiArrayType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert (
            input_spec[0].type.multiArrayType.WhichOneof("ShapeFlexibility") == "shapeRange"
        ), "Expected shapeRange in ShapeFlexibility"

        spec_default_shape = [s for s in input_spec[0].type.multiArrayType.shape]
        ranged_shapes = [(1, 1), (3, 3), (10, 10), (10, 30)]
        spec_ranged_shapes = []
        for range_dim in input_spec[0].type.multiArrayType.shapeRange.sizeRanges:
            spec_ranged_shapes.append(tuple([range_dim.lowerBound, range_dim.upperBound]))
        assert spec_default_shape == [
            1,
            3,
            10,
            10,
        ], "Expected default shape to be [1, 3, 10, 10], got {} instead".format(
            str(spec_default_shape)
        )
        assert spec_ranged_shapes == ranged_shapes, "Enumerated shape mismatch"

    def test_mil_ranged_multiarray_with_default(self):
        input_shape = [ct.TensorType(name="x", shape=(1, 3, 10, ct.RangeDim(10, 30, default=20)))]
        mlmodel = ct.convert(
            self.basic_network, source="milinternal", convert_to="mlprogram", inputs=input_shape
        )
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 1, "1 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(
            input_spec[0].name
        )
        assert (
            input_spec[0].type.WhichOneof("Type") == "multiArrayType"
        ), "Expected multiArrayType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert (
            input_spec[0].type.multiArrayType.WhichOneof("ShapeFlexibility") == "shapeRange"
        ), "Expected shapeRange in ShapeFlexibility"

        spec_default_shape = [s for s in input_spec[0].type.multiArrayType.shape]
        ranged_shapes = [(1, 1), (3, 3), (10, 10), (10, 30)]
        spec_ranged_shapes = []
        for range_dim in input_spec[0].type.multiArrayType.shapeRange.sizeRanges:
            spec_ranged_shapes.append(tuple([range_dim.lowerBound, range_dim.upperBound]))
        assert spec_default_shape == [
            1,
            3,
            10,
            20,
        ], "Expected default shape to be [1, 3, 10, 20], got {} instead".format(
            str(spec_default_shape)
        )
        assert spec_ranged_shapes == ranged_shapes, "Enumerated shape mismatch"

    def test_mil_ranged_image(self):
        input_shape = [ct.ImageType(name="x", shape=(1, 3, 10, ct.RangeDim(10, 30)))]
        mlmodel = ct.convert(
            self.basic_network, source="milinternal", convert_to="mlprogram", inputs=input_shape
        )
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 1, "1 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(
            input_spec[0].name
        )
        assert (
            input_spec[0].type.WhichOneof("Type") == "imageType"
        ), "Expected imageType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert (
            input_spec[0].type.imageType.WhichOneof("SizeFlexibility") == "imageSizeRange"
        ), "Expected imageSizeRange in ShapeFlexibility"

        spec_H = input_spec[0].type.imageType.height
        spec_W = input_spec[0].type.imageType.width
        assert (
            spec_H == 10 and spec_W == 10
        ), "expected [H, W] == [10, 10], got [{}, {}] instead".format(spec_H, spec_W)

        spec_H_range = [
            input_spec[0].type.imageType.imageSizeRange.heightRange.lowerBound,
            input_spec[0].type.imageType.imageSizeRange.heightRange.upperBound,
        ]
        spec_W_range = [
            input_spec[0].type.imageType.imageSizeRange.widthRange.lowerBound,
            input_spec[0].type.imageType.imageSizeRange.widthRange.upperBound,
        ]
        assert spec_H_range == [10, 10], "Ranged height mismatch"
        assert spec_W_range == [10, 30], "Ranged width mismatch"

    def test_mil_ranged_image_with_default(self):
        input_shape = [ct.ImageType(name="x", shape=(1, 3, 10, ct.RangeDim(10, 30, default=20)))]
        mlmodel = ct.convert(
            self.basic_network, source="milinternal", convert_to="mlprogram", inputs=input_shape
        )
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 1, "1 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(
            input_spec[0].name
        )
        assert (
            input_spec[0].type.WhichOneof("Type") == "imageType"
        ), "Expected imageType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert (
            input_spec[0].type.imageType.WhichOneof("SizeFlexibility") == "imageSizeRange"
        ), "Expected imageSizeRange in ShapeFlexibility"

        spec_H = input_spec[0].type.imageType.height
        spec_W = input_spec[0].type.imageType.width
        assert (
            spec_H == 10 and spec_W == 20
        ), "expected [H, W] == [10, 20], got [{}, {}] instead".format(spec_H, spec_W)

        spec_H_range = [
            input_spec[0].type.imageType.imageSizeRange.heightRange.lowerBound,
            input_spec[0].type.imageType.imageSizeRange.heightRange.upperBound,
        ]
        spec_W_range = [
            input_spec[0].type.imageType.imageSizeRange.widthRange.lowerBound,
            input_spec[0].type.imageType.imageSizeRange.widthRange.upperBound,
        ]
        assert spec_H_range == [10, 10], "Ranged height mismatch"
        assert spec_W_range == [10, 30], "Ranged width mismatch"


class TestMILDefaultValues:
    @mb.program(input_specs=[mb.TensorSpec(shape=[1]), mb.TensorSpec(shape=[1])])
    def basic_network(x, y):
        return mb.add(x=x, y=y, name="output")

    def test_mil_default_value_to_proto(self):
        program_input_spec = [
            ct.TensorType(name="x", shape=[1], default_value=np.array([1.0]).astype(np.float32)),
            ct.TensorType(name="y", shape=[1]),
        ]
        mlmodel = ct.convert(self.basic_network, convert_to="mlprogram", inputs=program_input_spec)
        input_spec = mlmodel.get_spec().description.input
        assert len(input_spec) == 2, "2 input expected, got {} instead".format(len(input_spec))
        assert input_spec[0].name == "x", "input name in MLModel is {}, 'x' is expected".format(
            input_spec[0].name
        )
        assert (
            input_spec[0].type.WhichOneof("Type") == "multiArrayType"
        ), "Expected multiArrayType, got {}".format(input_spec[0].type.WhichOneof("Type"))
        assert (
            input_spec[0].type.multiArrayType.WhichOneof("defaultOptionalValue")
            == "floatDefaultValue"
        ), "Expected floatDefaultValue, got {} instead".format(
            input_spec[0].type.multiArrayType.WhichOneof("defaultOptionalValue")
        )
        assert input_spec[0].type.multiArrayType.floatDefaultValue == 1.0

    def test_mil_default_value_runtime(self):
        program_input_spec = [
            ct.TensorType(name="x", shape=[1], default_value=np.array([1.0]).astype(np.float32)),
            ct.TensorType(name="y", shape=[1]),
        ]
        mlmodel = ct.convert(self.basic_network, convert_to="mlprogram", inputs=program_input_spec)

        if _macos_version() < (12, 0):
            # Can only get predictions for ml program on macOS 12+
            return

        res = mlmodel.predict({"x": np.array([3.0]), "y": np.array([2.0])})
        assert res["output"][0] == 5.0

        res = mlmodel.predict({"y": np.array([2.0])})
        assert res["output"][0] == 3.0


class TestMILProtoLoad:
    """Verify that the MIL Proto in mlmodel is correctly loaded in iOS18+."""

    @staticmethod
    @pytest.mark.parametrize("opset_version", [ct.target.iOS17, ct.target.iOS18])
    def test_constexpr_use_inputs_instead_of_attributes(opset_version):
        """Test the constexpr uses inputs instead of attributes starting from iOS18."""

        @mb.program(input_specs=[], opset_version=ct.target.iOS17)
        def prog_ios17():
            return mb.constexpr_lut_to_dense(
                lut=np.array([1.0, 2.0, 3.0, 4.0]),
                indices=np.array([10, 4]).astype(np.uint8),
                shape=np.array([5]).astype(np.uint32),
            )

        @mb.program(input_specs=[], opset_version=ct.target.iOS18)
        def prog_ios18():
            return mb.constexpr_lut_to_dense(
                indices=np.array([4, 8, 10, 13, 24, 5, 6, 9, 13, 31, 17, 7, 2, 8, 3, 1])
                .reshape((2, 4, 2))
                .astype(np.uint8),
                lut=_TestConstexprLut._generate_lut(shape=(1, 2, 1, 256, 3)),
                vector_axis=1,
            )

        mlmodel = ct.convert(
            prog_ios17 if opset_version == ct.target.iOS17 else prog_ios18,
            convert_to="mlprogram",
            minimum_deployment_target=opset_version,
        )

        # Iterates the milproto in mlmodel to make sure lut op uses inputs instead of attributes.
        mil = mlmodel.get_spec().mlProgram
        for function in mil.functions.values():
            for block in function.block_specializations.values():
                for op in block.operations:
                    if op.type == "constexpr_lut_to_dense":
                        # The "attributes" field has at least one value for "name".
                        expected_attributes_num = 1
                        expected_inputs_num = 0
                        if opset_version >= ct.target.iOS18:
                            # Since iOS18, constexpr ops use inputs instead of attributes in milproto.
                            expected_inputs_num += 3
                        else:
                            expected_attributes_num += 3

                        assert len(op.attributes.values()) == expected_attributes_num
                        assert len(op.inputs.values()) == expected_inputs_num

    @staticmethod
    def test_constexpr_multiple_outputs():
        """Starting from iOS18 there are constexpr ops that have multiple outputs."""

        @mb.program(input_specs=[], opset_version=ct.target.iOS18)
        def prog():
            return mb.constexpr_sparse_blockwise_shift_scale(
                data_mask=np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]]).astype(
                    types.np_uint1_dtype
                ),
                nonzero_data=np.array([10, 11, 3, 4, 5, 6, 7, 8, 9]).astype(np.int8),
                scale=np.array([[0.1, 0.2, 0.3, 0.4]]),
                offset=np.array([[1, 2, 3, 4]]).astype(np.int8),
            )[1]

        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
        )

        mil = mlmodel.get_spec().mlProgram
        for function in mil.functions.values():
            for block in function.block_specializations.values():
                for op in block.operations:
                    if op.type == "constexpr_sparse_blockwise_shift_scale":
                        assert len(op.outputs) == 2

    @staticmethod
    def test_sub_byte_immediate_value():
        """
        Test the sub-byte immediate value tensor is exported as packed bytes.

        The sub-byte file value is tested in `coremltools/test/blob/test_weights.py` which
        is not in the scope of this test.
        """

        @mb.program(input_specs=[], opset_version=ct.target.iOS18)
        def prog():
            return mb.constexpr_blockwise_shift_scale(
                data=np.array([-8, 7]).reshape((1, 2, 1)).astype(types.np_int4_dtype),
                scale=np.array([4]).reshape((1, 1, 1)).astype(np.float16),
                offset=np.array([4]).reshape((1, 1, 1)).astype(types.np_int4_dtype),
            )

        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
        )

        mil = mlmodel.get_spec().mlProgram
        for function in mil.functions.values():
            for block in function.block_specializations.values():
                for op in block.operations:
                    if op.type == "constexpr_blockwise_shift_scale":
                        bytes_val = (
                            op.inputs["data"].arguments[0].value.immediateValue.tensor.bytes.values
                        )
                        # The two 4-bit values should be packed into a single byte.
                        assert len(bytes_val) == 1

    @staticmethod
    def check_functions_description(
        mlmodel: ct.models.MLModel,
        expect_function_names: List[str],
        expected_default_function_name: str,
    ) -> None:
        spec = mlmodel.get_spec()
        desc = spec.description
        assert len(desc.functions) == len(expect_function_names)
        for i in range(len(expect_function_names)):
            assert desc.functions[i].name == expect_function_names[i]
        assert desc.defaultFunctionName == expected_default_function_name

    @staticmethod
    def convert_and_save(prog: mil.Program) -> str:
        prog.export_as_multifunction = True
        mlmodel = _mil_convert(
            prog,
            convert_to="mlprogram",
            convert_from="milinternal",
            specification_version=_SPECIFICATION_VERSION_IOS_18,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            skip_model_load=True,
        )
        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        mlmodel.save(package_path)
        return package_path

    @staticmethod
    def check_relu(model: Union[ct.models.MLModel, ct.models.CompiledMLModel]) -> None:
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        y_relu = [0, 0, 1]
        y = model.predict({"x": x})
        assert all(y["relu_0"] == y_relu)

    @staticmethod
    def check_sin(model: Union[ct.models.MLModel, ct.models.CompiledMLModel]) -> None:
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        y_sin = list(map(math.sin, x))
        y = model.predict({"x": x})
        np.testing.assert_allclose(y["sin_0"], y_sin, rtol=5e-04, atol=5e-04)

    @staticmethod
    def check_cos(model: Union[ct.models.MLModel, ct.models.CompiledMLModel]) -> None:
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        y_sin = list(map(math.cos, x))
        y = model.predict({"x": x})
        np.testing.assert_allclose(y["cos_0"], y_sin, rtol=5e-04, atol=5e-04)


    @pytest.mark.skipif(ct.utils._macos_version() < (15, 0),
                    reason="Multi-function only supported on macOS 15+")
    def test_multi_functions(self):
        """
        Test multi-functions program can be exported into multi-functions Core ML proto.
        """

        @mb.function(
            input_specs=[mb.TensorSpec((3,))],
            opset_version=ct.target.iOS18,
        )
        def func(x):
            return mb.relu(x=x)

        @mb.function(
            input_specs=[mb.TensorSpec((3,))],
            opset_version=ct.target.iOS18,
        )
        def func_1(x):
            return mb.sin(x=x)

        @mb.function(
            input_specs=[mb.TensorSpec((3,))],
            opset_version=ct.target.iOS18,
        )
        def func_2(x):
            return mb.cos(x=x)

        prog = mil.Program()
        prog.add_function("main", func)
        prog.add_function("sin", func_1)
        prog.add_function("cos", func_2)

        package_path = self.convert_and_save(prog)

        # Test the proto can be loaded back and validate the spec
        mlmodel = ct.models.MLModel(package_path, function_name="main")
        self.check_functions_description(
            mlmodel,
            expect_function_names=["main", "sin", "cos"],
            expected_default_function_name="main",
        )

        # Validate MLModel predictions for all three functions
        self.check_relu(mlmodel)
        self.check_sin(
            ct.models.MLModel(
                package_path, function_name="sin", compute_units=ct.ComputeUnit.CPU_ONLY
            )
        )
        self.check_cos(
            ct.models.MLModel(
                package_path, function_name="cos", compute_units=ct.ComputeUnit.CPU_ONLY
            )
        )

        # Validate MLModel function_name property
        assert mlmodel.function_name == "main"
        assert ct.models.MLModel(package_path, function_name="sin").function_name == "sin"
        assert ct.models.MLModel(package_path, function_name="cos").function_name == "cos"

        # Invalid function_name
        with pytest.raises(ValueError, match="function_name invalid not found in the model"):
            mlmodel = ct.models.MLModel(package_path, function_name="invalid")

        # Validate CompiledMLModel predictions for all three functions
        compiled_path = mlmodel.get_compiled_model_path()
        self.check_relu(ct.models.CompiledMLModel(compiled_path, function_name="main"))
        self.check_sin(ct.models.CompiledMLModel(compiled_path, function_name="sin"))
        self.check_cos(ct.models.CompiledMLModel(compiled_path, function_name="cos"))

        # clean up
        shutil.rmtree(package_path)


    @pytest.mark.skipif(ct.utils._macos_version() < (15, 0),
                    reason="Multi-function only supported on macOS 15+")
    def test_multi_functions_default_function(self):
        """
        Test if no function_name passes to MLModel, default function name will be picked up.
        """

        @mb.function(
            input_specs=[mb.TensorSpec((3,))],
            opset_version=ct.target.iOS18,
        )
        def func(x):
            return mb.relu(x=x)

        @mb.function(
            input_specs=[mb.TensorSpec((3,))],
            opset_version=ct.target.iOS18,
        )
        def func_1(x):
            return mb.sin(x=x)

        prog = mil.Program()
        prog.add_function("main_1", func)
        prog.add_function("sin", func_1)
        prog.default_function_name = "main_1"

        package_path = self.convert_and_save(prog)

        # With no function_name passed, mlmodel.function_name defaults to defaultFunctionName
        mlmodel = ct.models.MLModel(package_path)
        self.check_functions_description(
            mlmodel,
            expect_function_names=["main_1", "sin"],
            expected_default_function_name="main_1",
        )
        assert mlmodel.function_name == "main_1"

        # Validate the prediction runs on default function
        self.check_relu(mlmodel)

        # Validate CompiledMLModel predictions for default function
        compiled_path = mlmodel.get_compiled_model_path()
        self.check_relu(ct.models.CompiledMLModel(compiled_path))

        # clean up
        shutil.rmtree(package_path)


    @pytest.mark.skipif(ct.utils._macos_version() < (15, 0),
                        reason="Multi-function only supported on macOS 15+")
    def test_single_function_in_multifunction_format(self):
        @mb.function(
            input_specs=[mb.TensorSpec((3,))],
            opset_version=ct.target.iOS18,
        )
        def func(x):
            return mb.relu(x=x)

        prog = mil.Program()
        prog.add_function("main_1", func)
        prog.default_function_name = "main_1"

        package_path = self.convert_and_save(prog)

        # No function_name is passed, default function name is picked up
        mlmodel = ct.models.MLModel(package_path)
        self.check_functions_description(
            mlmodel,
            expect_function_names=["main_1"],
            expected_default_function_name="main_1",
        )

        # Validate MLModel predictions
        self.check_relu(mlmodel)
        self.check_relu(ct.models.MLModel(package_path, function_name="main_1"))

        # Validate CompiledMLModel predictions
        compiled_path = mlmodel.get_compiled_model_path()
        self.check_relu(ct.models.CompiledMLModel(compiled_path))
        self.check_relu(ct.models.CompiledMLModel(compiled_path, function_name="main_1"))

        # clean up
        shutil.rmtree(package_path)


    @pytest.mark.skipif(ct.utils._macos_version() < (15, 0),
                    reason="Multi-function only supported on macOS 15+")
    def test_multi_functions_backward_compatibility(self):
        # Test the new MLModel class can load pre-iOS17 single function model
        @mb.program(input_specs=[mb.TensorSpec((3,))], opset_version=ct.target.iOS16)
        def prog(x):
            return mb.relu(x=x)

        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS16,
        )

        # Test the proto can be saved and loaded back
        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        mlmodel.save(package_path)

        # Validate the MLModel predictions
        self.check_relu(ct.models.MLModel(package_path))
        self.check_relu(ct.models.MLModel(package_path, function_name="main"))

        # Validate the MLModel function_name property
        assert ct.models.MLModel(package_path).function_name is None
        assert ct.models.MLModel(package_path, function_name="main").function_name == "main"

        # Other function_name will error out
        with pytest.raises(
            ValueError, match='function_name must be "main" for non multifunction model'
        ):
            mlmodel = ct.models.MLModel(package_path, function_name="invalid")

        # Validate the CompiledMLModel predictions
        compiled_path = mlmodel.get_compiled_model_path()
        self.check_relu(ct.models.CompiledMLModel(compiled_path))
        self.check_relu(ct.models.CompiledMLModel(compiled_path, function_name="main"))

        # invalid function error at runtime
        with pytest.raises(RuntimeError):
            compiled_model = ct.models.CompiledMLModel(compiled_path, function_name="invalid")

        # clean up
        shutil.rmtree(package_path)


class TestMILProtoExporter:
    @pytest.mark.skipif(
        ct.utils._macos_version() < (15, 0), reason="Tests are for deployment target iOS18/macos15"
    )
    def test_const_share_blob_value(self):
        """For the same const in the function, they should share the same blob value."""
        mask_val = np.random.randint(low=0, high=2, size=(32, 32)).astype(types.np_uint1_dtype)
        nonzero_data_val = np.random.randint(low=1, high=100, size=(np.sum(mask_val),)).astype(
            np.int8
        )

        @mb.program(input_specs=[], opset_version=ct.target.iOS18)
        def prog():
            mask_const = mb.const(val=mask_val)
            mask, nonzero_data = mb.constexpr_sparse_blockwise_shift_scale(
                data_mask=mask_const,
                nonzero_data=nonzero_data_val,
                scale=np.array([[0.1]]),
                offset=np.array([[1]]).astype(np.int8),
            )
            return mb.constexpr_sparse_to_dense(
                nonzero_data=nonzero_data,
                mask=mask_const,
            )

        mlmodel = ct.convert(prog, minimum_deployment_target=ct.target.iOS18)
        saved_package_path = tempfile.mkdtemp(suffix=".mlpackage")
        mlmodel.save(saved_package_path)

        with tempfile.TemporaryDirectory() as serialize_dir:
            mil_file = open(os.path.join(mlmodel.get_compiled_model_path(), "model.mil"))
            mil_txt = mil_file.read()
            # The `data_mask` and `mask` should share the same offset.
            assert (
                'constexpr_sparse_blockwise_shift_scale(data_mask = tensor<uint1, [32, 32]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))'
                in mil_txt
            )
            assert (
                'constexpr_sparse_to_dense(mask = tensor<uint1, [32, 32]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))'
                in mil_txt
            )

        shutil.rmtree(saved_package_path)


@pytest.mark.skipif(
    ct.utils._macos_version() < (15, 0), reason="Tests are for deployment target iOS18/macos15"
)
class TestStateModelLoad:
    """
    Verify stateful model can be loaded via milproto.
    """

    @staticmethod
    def verify_stateful_model(mlmodel, expected_output, input=None):
        def verify_numerical(mlmodel, state, expected_output, input=None):
            if input is None:
                input_dict = {}
            else:
                input_dict = {"y": input}
            output = mlmodel.predict(input_dict, state=state)["output"]
            np.testing.assert_allclose(expected_output, output, rtol=5e-04, atol=5e-04)

        # verify the model can be ran
        state_1 = mlmodel.make_state()
        verify_numerical(mlmodel, state_1, expected_output, input)
        verify_numerical(mlmodel, state_1, expected_output, input)

        # create a new state, and make sure the model can run prediction on both old and new state
        state_2 = mlmodel.make_state()
        verify_numerical(mlmodel, state_2, expected_output, input)
        verify_numerical(mlmodel, state_1, expected_output, input)

    def test_export_state_input_feature(self):
        """
        Test milproto can export model with state type.
        """

        @mb.program(
            input_specs=[
                mb.StateTensorSpec((2, 3), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            return mb.read_state(input=x, name="output")

        # verify the model can be converted
        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )

        # verify the state feature
        spec = mlmodel.get_spec()
        state = spec.description.state
        assert len(state) == 1
        assert state[0].name == "x"
        assert state[0].type.WhichOneof("Type") == "stateType"
        assert state[0].type.stateType.WhichOneof("Type") == "arrayType"

        array_type = state[0].type.stateType.arrayType
        assert array_type.shape == [2, 3]
        assert array_type.dataType == proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16

        # verify the model
        expected_output = np.zeros((2, 3))
        self.verify_stateful_model(mlmodel, expected_output)

    def test_export_mixed_state_input_features(self):
        """
        Test milproto can export model with states and inputs.
        """

        @mb.program(
            input_specs=[
                mb.StateTensorSpec((2, 3), dtype=types.fp16),
                mb.TensorSpec((2, 3), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(x, y):
            x = mb.read_state(input=x)
            return mb.add(x=x, y=y, name="output")

        # verify the model can be converted
        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )

        # verify the state feature
        spec = mlmodel.get_spec()
        state = spec.description.state
        assert len(state) == 1
        assert state[0].name == "x"
        assert state[0].type.WhichOneof("Type") == "stateType"
        assert state[0].type.stateType.WhichOneof("Type") == "arrayType"

        array_type = state[0].type.stateType.arrayType
        assert array_type.shape == [2, 3]
        assert array_type.dataType == proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16

        # verify the input
        input = spec.description.input
        assert len(input) == 1
        assert input[0].name == "y"
        assert input[0].type.WhichOneof("Type") == "multiArrayType"

        array_type = input[0].type.multiArrayType
        assert array_type.shape == [2, 3]
        assert array_type.dataType == proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16

        # verify the model
        input = np.random.rand(2, 3)
        self.verify_stateful_model(mlmodel, input, input)


    def test_multi_functions_state_model(self):
        """
        Make sure multi-functions Core ML models support state.
        """

        @mb.function(
            input_specs=[mb.StateTensorSpec((3,), dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def func(x):
            return mb.read_state(input=x, name="output")

        @mb.function(
            input_specs=[mb.StateTensorSpec((2,), dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def func_1(y):
            return mb.read_state(input=y, name="output")

        prog = mil.Program()
        prog.add_function("main", func)
        prog.add_function("func_1", func_1)
        prog.export_as_multifunction = True

        mlmodel = _mil_convert(
            prog,
            convert_to="mlprogram",
            convert_from="milinternal",
            specification_version=_SPECIFICATION_VERSION_IOS_18,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )

        spec = mlmodel.get_spec()
        desc = spec.description
        assert len(desc.functions) == 2
        assert desc.functions[0].name == "main"
        assert len(desc.functions[0].state) == 1
        assert desc.functions[0].state[0].name == "x"
        assert desc.functions[1].name == "func_1"
        assert len(desc.functions[1].state) == 1
        assert desc.functions[1].state[0].name == "y"

        # main function is the default function
        self.verify_stateful_model(mlmodel, np.zeros((3,)))

        # save the mlmodel on disk, and load "main" and "func_1" seperately
        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        mlmodel.save(package_path)

        # test "main" function
        mlmodel_main = ct.models.MLModel(
            package_path, compute_units=ct.ComputeUnit.CPU_ONLY, function_name="main"
        )
        self.verify_stateful_model(mlmodel_main, np.zeros((3,)))

        # test "func_1" function
        mlmodel_func_1 = ct.models.MLModel(
            package_path, compute_units=ct.ComputeUnit.CPU_ONLY, function_name="func_1"
        )
        self.verify_stateful_model(mlmodel_func_1, np.zeros((2,)))

        # cleanup mlpackage
        shutil.rmtree(package_path)

    def test_export_coreml_update_state(self):
        """
        The ``coreml_update_state`` dialect op is decomposed into:
            write_state -> read_state
        """

        @mb.program(
            input_specs=[
                mb.StateTensorSpec((2, 3), dtype=types.fp16),
                mb.TensorSpec((2, 3), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(x, y):
            return mb.coreml_update_state(state=x, value=y, name="output")

        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )

        mil = mlmodel.get_spec().mlProgram
        for function in mil.functions.values():
            for block in function.block_specializations.values():
                ops = list(block.operations)
                assert ops[0].type == "write_state"
                assert len(ops[0].outputs) == 0
                assert ops[1].type == "read_state"

        # verify the model
        input = np.random.rand(2, 3)
        self.verify_stateful_model(mlmodel, input, input)


    @staticmethod
    def test_invalid_state_input():
        """
        Test unsupported input state modes.
        """
        # state only supports fp16
        @mb.program(
            input_specs=[
                mb.StateTensorSpec((2, 3), dtype=types.fp32),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            return mb.read_state(input=x)

        with pytest.raises(
            ValueError,
            match="State only support fp16 dtype. Got input var x with dtype fp32.",
        ):
            mlmodel = ct.convert(
                prog,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS18,
            )

        # state doesn't support flexible shape
        @mb.program(
            input_specs=[
                mb.StateTensorSpec((2, get_new_symbol()), dtype=types.fp32),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            return mb.read_state(input=x)

        with pytest.raises(ValueError, match="Flexible shape model states are not supported!"):
            mlmodel = ct.convert(
                prog,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS18,
            )

    @staticmethod
    def test_coreml_update_state_lowering():
        """
        If the output of coreml_update_state is not a block output and
        it is not fed into any other ops, the op should be translated into
        a single write_state.
        """

        @mb.program(
            input_specs=[
                mb.StateTensorSpec((1,), dtype=types.fp16),
                mb.TensorSpec((1,), dtype=types.fp16),
                mb.TensorSpec((1,), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(state, x, y):
            mb.coreml_update_state(state=state, value=x)
            mb.coreml_update_state(state=state, value=y)
            return x, mb.coreml_update_state(state=state, value=y)

        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
            skip_model_load=True,
        )

        mil = mlmodel.get_spec().mlProgram

        for function in mil.functions.values():
            for block in function.block_specializations.values():
                ops = list(block.operations)
                expected_ops = [
                    "write_state",
                    "write_state",
                    "write_state",
                    "read_state",
                ]
                assert [val.type for val in ops] == expected_ops

    @staticmethod
    def test_coreml_update_state_lowering_with_prefer_state_in_downstream():
        @mb.program(
            input_specs=[
                mb.StateTensorSpec((1,), dtype=types.fp16),
                mb.TensorSpec((1,), dtype=types.fp16),
                mb.TensorSpec((1,), dtype=types.fp16),
                mb.TensorSpec((1,), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(state, x, y, z):
            # Although seemingly not used, graph pass prefer_state_in_downstream will
            # make its output as identiy.x
            mb.coreml_update_state(state=state, value=x)
            # If value only feeds into coreml_update_state,
            # the prefer_state_in_downstream has no affects
            mb.coreml_update_state(state=state, value=y)
            # This is the one that really is not used
            mb.coreml_update_state(state=state, value=z)
            return mb.identity(x=x), mb.coreml_update_state(state=state, value=y)

        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
            skip_model_load=True,
        )

        mil = mlmodel.get_spec().mlProgram
        for function in mil.functions.values():
            for block in function.block_specializations.values():
                ops = list(block.operations)
                expected_ops = [
                    "write_state",
                    "read_state",
                    "write_state",
                    "write_state",
                    "identity",
                    "write_state",
                    "read_state",
                ]
                assert [val.type for val in ops] == expected_ops

    @staticmethod
    @pytest.mark.skipif(ct.utils._macos_version() < (15, 0),
                    reason="State only supported on macOS 15+")
    def test_prediction_state():
        """
        Test prediction from a stateful model
        """

        def extract_value(y):
            return list(y.values())[0][0]

        def test_state_model(mlmodel, multiplier):
            # Using first state
            state1 = mlmodel.make_state()
            for i in range(1, 5):
                y = mlmodel.predict({}, state=state1)
                assert extract_value(y) == multiplier * i

            # Use a new state
            state2 = mlmodel.make_state()
            for i in range(1, 5):
                y = mlmodel.predict({}, state=state2)
                assert extract_value(y) == multiplier * i

            # Go back to using the first state
            for i in range(5, 10):
                y = mlmodel.predict({}, state=state1)
                assert extract_value(y) == multiplier * i

        @mb.program(
            input_specs=[
                mb.StateTensorSpec((1,), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def increment(x):
            # Read
            y = mb.read_state(input=x)
            # Update
            y = mb.add(x=y, y=np.array([1.0]).astype("float16"))
            # Write
            y = mb.coreml_update_state(state=x, value=y)
            # Return
            return y

        @mb.program(
            input_specs=[
                mb.StateTensorSpec((1,), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def increment_by_2(x):
            # Read
            y = mb.read_state(input=x)
            # Update
            y = mb.add(x=y, y=np.array([1.0]).astype("float16"))
            # Write
            y = mb.coreml_update_state(state=x, value=y)
            # Update
            y = mb.add(x=y, y=np.array([1.0]).astype("float16"))
            # Write
            mb.coreml_update_state(state=x, value=y)
            # Return
            return y

        for model, multiplier in [(increment, 1), (increment_by_2, 2)]:
            mlmodel = ct.convert(
                model,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS18,
            )

            # The test is failing on x86_64 machines
            # rdar://126957030 ([State][Bug][Intel] Stateful model prediction is wrong on Intel laptop)
            if platform.machine() == "arm64":
                test_state_model(mlmodel, multiplier)

            # save the model and load it back
            package_path = tempfile.mkdtemp(suffix=".mlpackage")
            mlmodel.save(package_path)

            # Load with CPU
            test_state_model(
                ct.models.MLModel(package_path, compute_units=ct.ComputeUnit.CPU_ONLY), multiplier
            )

            # Load with ALL
            if platform.machine() == "arm64":
                test_state_model(ct.models.MLModel(package_path), multiplier)

            shutil.rmtree(package_path)
