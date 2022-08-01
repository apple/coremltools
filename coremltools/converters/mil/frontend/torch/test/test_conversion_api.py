#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from PIL import Image
import pytest

import coremltools as ct
from coremltools.converters.mil.testing_utils import (
    assert_cast_ops_count,
    assert_input_dtype,
    assert_ops_in_mil_program,
    assert_output_dtype,
    assert_prog_input_type,
    assert_prog_output_type,
    assert_spec_input_image_type,
    assert_spec_output_image_type,
    verify_prediction,
)
from coremltools.proto import FeatureTypes_pb2 as ft

torch = pytest.importorskip("torch")


@pytest.fixture
def int32_input_model():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 5
    example_input = torch.randint(0, 100, (10, 20), dtype=torch.int32)
    return torch.jit.trace(Model().eval(), example_input)

@pytest.fixture
def int64_input_model():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 5
    example_input = torch.randint(0, 100, (10, 20), dtype=torch.int64)
    return torch.jit.trace(Model().eval(), example_input)

@pytest.fixture
def float32_input_model_add_op():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 5.5
    example_input = torch.randint(0, 100, (10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), example_input)

@pytest.fixture
def float32_input_model_relu_ops():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = torch.nn.ReLU()(x)
            return torch.nn.ReLU()(x)
    example_input = torch.randint(0, 100, (10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), example_input)

@pytest.fixture
def float32_two_input_model():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y
    example_input = torch.randint(0, 100, (10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), [example_input, example_input])

@pytest.fixture
def float32_two_output_model():
    class Model(torch.nn.Module):
        def forward(self, x):
            y = torch.nn.ReLU()(x)
            out1 = torch.nn.ReLU()(y)
            out2 = torch.nn.ReLU6()(x)
            return out1, out2
    example_input = torch.randint(0, 100, (10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), example_input)

@pytest.fixture
def rank3_input_model():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 5.5
    example_input = torch.randint(0, 100, (1, 10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), example_input)

@pytest.fixture
def rank4_input_model():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 5.5
    example_input = torch.randint(0, 100, (1, 3, 10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), example_input)

@pytest.fixture
def rank4_grayscale_input_model():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 10
    example_input = torch.randint(0, 100, (1, 1, 10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), example_input)

@pytest.fixture
def linear_model():
    # this model will test the fuse_linear_bias pass
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 15, bias=False)
            self.constant_tensor = torch.ones((15,), dtype=torch.float32)

        def forward(self, x):
            x = self.linear(x)
            x = x - self.constant_tensor
            x = torch.nn.ReLU()(x)
            return x
    example_input = torch.randint(0, 10, (1, 10), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), example_input)


@pytest.mark.skipif(ct.utils._macos_version() < (13, 0), reason='Tests are for deployment target ios16/macos13')
class TestInputOutputConversionAPI:

    def test_input_dtype_default(self, int32_input_model):
        #if dtype is not provided it defaults to float32
        mlmodel = ct.convert(int32_input_model,
                             inputs=[ct.TensorType(shape=(10, 20))],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

    def test_input_shape_missing_error(self, float32_input_model_add_op):
        with pytest.raises(ValueError,
                           match="'shape' must be provided in the 'inputs' argument for pytorch conversion"):
            mlmodel = ct.convert(float32_input_model_add_op,
                                 inputs=[ct.TensorType(dtype=np.int32)],
                                 minimum_deployment_target=ct.target.macOS12)

    def test_unsupported_input_dtype_in_torch_model(self, int64_input_model):
        # test that no error is raised when no dtype is provided by the user,
        # and the Torch model's input dtype is not supported.
        # In this case, it will be mapped to the default dtype which is float32
        mlmodel = ct.convert(int64_input_model,
                             inputs=[ct.TensorType(shape=(10, 20))],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

    def test_input_dtype_user_provided(self, float32_input_model_add_op):
        # test that provided dtype in the api is applied
        mlmodel = ct.convert(float32_input_model_add_op,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.int32)],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="int32")
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

    def test_invalid_input_dtype(self, int32_input_model):
        with pytest.raises(TypeError,
                           match="is unsupported for inputs/outputs of the model"
                           ):
            mlmodel = ct.convert(int32_input_model,
                                 inputs=[ct.TensorType(dtype=np.int16)],
                                 minimum_deployment_target=ct.target.macOS12)

        with pytest.raises(TypeError,
                           match="float16 dtype for inputs is only supported for deployment target >= iOS16/macOS13"
                           ):
            mlmodel = ct.convert(int32_input_model,
                                 inputs=[ct.TensorType(dtype=np.float16)],
                                 minimum_deployment_target=ct.target.macOS12)

    def test_fp16_input_dtype(self, float32_input_model_add_op, float32_input_model_relu_ops, int32_input_model):
        """
        Test that providing fp16 input dtype works with macOS13.
        """
        mlmodel = ct.convert(float32_input_model_add_op,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add", "cast"])
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

        mlmodel = ct.convert(float32_input_model_relu_ops,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["relu", "relu", "cast"])
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

        mlmodel = ct.convert(int32_input_model,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add", "cast"])
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

    def test_fp16_input_dtype_fp32_precision(self, float32_input_model_add_op, float32_input_model_relu_ops,
                                             int32_input_model):
        """
        Same test as test_fp16_input_dtype, but with Float32 precision
        """
        mlmodel = ct.convert(float32_input_model_add_op,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             compute_precision=ct.precision.FLOAT32,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "add"])
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

        """
        Although no FP16ComputePrecision is applied, the float16 input propagates through the network
        """
        mlmodel = ct.convert(float32_input_model_relu_ops,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             compute_precision=ct.precision.FLOAT32,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "relu", "relu"])
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp32")

    def test_input_name_specified_by_user(self, float32_input_model_relu_ops,
                                          float32_two_input_model):
        mlmodel = ct.convert(float32_input_model_relu_ops,
                             inputs=[ct.TensorType(shape=(10, 20), name="my_custom_input_name")],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="fp32", expected_name="my_custom_input_name")

        mlmodel = ct.convert(float32_two_input_model,
                             inputs=[ct.TensorType(shape=(10, 20), name="user_provided_name_1"),
                                     ct.TensorType(shape=(10, 20), name="user_provided_name_2")],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="fp32", expected_name="user_provided_name_1", index=0)
        assert_input_dtype(mlmodel, expected_type_str="fp32", expected_name="user_provided_name_2", index=1)

    def test_two_input_model(self, float32_two_input_model):
        # test that error is raised if only 1 input is provided
        with pytest.raises(ValueError):
            ct.convert(float32_two_input_model,
                       inputs=[ct.TensorType(shape=(10, 20), dtype=np.int32)],
                       minimum_deployment_target=ct.target.macOS12)


        # test forcing 1st input to type int32
        mlmodel = ct.convert(float32_two_input_model,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.int32),
                                     ct.TensorType(shape=(10, 20))],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="int32", index=0)
        assert_input_dtype(mlmodel, expected_type_str="fp32", index=1)
        assert_output_dtype(mlmodel, expected_type_str="fp32")

        # test forcing both inputs to be int32
        mlmodel = ct.convert(float32_two_input_model,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.int32),
                                     ct.TensorType(shape=(10, 20), dtype=np.int32),
                                     ],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="int32", index=0)
        assert_input_dtype(mlmodel, expected_type_str="int32", index=1)
        assert_output_dtype(mlmodel, expected_type_str="int32")

        # test forcing both inputs to be float16
        mlmodel = ct.convert(float32_two_input_model,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.float16),
                                     ct.TensorType(shape=(10, 20), dtype=np.float16),
                                     ],
                             minimum_deployment_target=ct.target.macOS13)
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add", "cast"])
        assert_input_dtype(mlmodel, expected_type_str="fp16", index=0)
        assert_input_dtype(mlmodel, expected_type_str="fp16", index=1)
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

    def test_output_name_specified_by_user(self, float32_input_model_relu_ops, float32_two_output_model):
        mlmodel = ct.convert(float32_input_model_relu_ops,
                             inputs=[ct.TensorType(shape=(10, 20), name="custom_input_name")],
                             outputs=[ct.TensorType(name="custom_output_name")],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="fp32", expected_name="custom_input_name")
        assert_output_dtype(mlmodel, expected_type_str="fp32", expected_name="custom_output_name")

        mlmodel = ct.convert(float32_two_output_model,
                             inputs=[ct.TensorType(shape=(10, 20), name="custom_input_name")],
                             outputs=[ct.TensorType(name="custom_output1_name"),
                                      ct.TensorType(name="custom_output2_name")],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="fp32", expected_name="custom_input_name")
        assert_output_dtype(mlmodel, expected_type_str="fp32", expected_name="custom_output1_name", index=0)
        assert_output_dtype(mlmodel, expected_type_str="fp32", expected_name="custom_output2_name", index=1)

    def test_single_output_model(self, int32_input_model, float32_input_model_relu_ops):
        # test output type: if not provided, it should be the default which is float32
        mlmodel = ct.convert(int32_input_model,
                             inputs=[ct.TensorType(shape=(10, 20))],
                             minimum_deployment_target=ct.target.macOS12)
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "add", "cast"])
        assert_input_dtype(mlmodel, expected_type_str="fp32")
        assert_output_dtype(mlmodel, expected_type_str="fp32")

        # test that the output dtype provided by the user is applied during conversion
        mlmodel = ct.convert(float32_input_model_relu_ops,
                             inputs=[ct.TensorType(shape=(10, 20))],
                             outputs=[ct.TensorType(dtype=np.int32)],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="fp32")
        assert_output_dtype(mlmodel, expected_type_str="int32")
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "relu", "relu", "cast", "cast"])

        # test that an error is raised when shape is provided for the output
        with pytest.raises(ValueError):
            mlmodel = ct.convert(int32_input_model,
                                 inputs=[ct.TensorType(shape=(10, 20))],
                                 outputs=[ct.TensorType(dtype=np.float32, shape=(10, 20))],
                                 minimum_deployment_target=ct.target.macOS12)

        # test that output dtype of float16 is rejected when deployment target is low
        with pytest.raises(TypeError,
                           match="float16 dtype for outputs is only supported for deployment target >= iOS16/macOS13"
                           ):
            ct.convert(float32_input_model_relu_ops,
                       inputs=[ct.TensorType(shape=(10, 20))],
                       outputs=[ct.TensorType(dtype=np.float16)],
                       minimum_deployment_target=ct.target.macOS12,
                       )

        # test that output type float16 is applied correctly
        mlmodel = ct.convert(float32_input_model_relu_ops,
                             inputs=[ct.TensorType(shape=(10, 20))],
                             outputs=[ct.TensorType(dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_output_dtype(mlmodel, expected_type_str="fp16")
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "relu", "relu"])

        # test that input and output types float16 are applied correctly
        mlmodel = ct.convert(float32_input_model_relu_ops,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.float16)],
                             outputs=[ct.TensorType(dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp16")
        assert_ops_in_mil_program(mlmodel, expected_op_list=["relu", "relu"])
        verify_prediction(mlmodel)

    def test_multi_output_model(self, float32_two_output_model):
        # check that error is raised when only 1 output provided
        with pytest.raises(ValueError, match="Number of outputs provided, 1, "
                                        "do not match the number of outputs detected in the model, 2"):
            ct.convert(float32_two_output_model,
                       inputs=[ct.TensorType(shape=(10, 20))],
                       outputs=[ct.TensorType()],
                       minimum_deployment_target=ct.target.macOS12)

        # set 1 output to float16 and the other to float32
        mlmodel = ct.convert(float32_two_output_model,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.float16)],
                             outputs=[ct.TensorType(name="out1", dtype=np.float16),
                                      ct.TensorType(name="out2", dtype=np.float32)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_cast_ops_count(mlmodel, expected_count=1)
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp16", expected_name="out1" ,index=0)
        assert_output_dtype(mlmodel, expected_type_str="fp32", expected_name="out2", index=1)
        verify_prediction(mlmodel)

    def test_color_input(self, rank4_input_model, rank3_input_model):
        mlmodel = ct.convert(rank4_input_model,
                             inputs=[ct.ImageType(shape=(1, 3, 10, 20), color_layout=ct.colorlayout.RGB)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "add", "cast"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.RGB)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp32")
        verify_prediction(mlmodel)

        with pytest.raises(ValueError, match="must have rank 4"):
            mlmodel = ct.convert(rank3_input_model,
                                 inputs=[ct.ImageType(shape=(1, 10, 20), color_layout=ct.colorlayout.RGB)],
                                 minimum_deployment_target=ct.target.macOS12,
                                 )

    def test_grayscale_input(self, rank4_input_model, rank3_input_model, rank4_grayscale_input_model):
        with pytest.raises(ValueError, match="must have rank 4"):
            ct.convert(rank3_input_model,
                       inputs=[ct.ImageType(shape=(1, 10, 20), color_layout=ct.colorlayout.GRAYSCALE)],
                       minimum_deployment_target=ct.target.macOS13,
                      )

        # invalid shape
        with pytest.raises(ValueError):
            ct.convert(rank4_input_model,
                       inputs=[ct.ImageType(shape=(1, 3, 10, 20), color_layout=ct.colorlayout.GRAYSCALE)],
                       minimum_deployment_target=ct.target.macOS13,
                       )

        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.ImageType(shape=(1, 1, 10, 20), color_layout=ct.colorlayout.GRAYSCALE)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "add", "cast"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp32")
        verify_prediction(mlmodel)

        with pytest.raises(TypeError, match="float16 dtype for inputs is only supported for deployment target >= iOS16/macOS13"):
            ct.convert(rank4_grayscale_input_model,
                       inputs=[ct.ImageType(shape=(1, 1, 10, 20),
                                            color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                       minimum_deployment_target=ct.target.macOS12,
                       )

        # test that grayscale_16 raises error when used with neural network
        with pytest.raises(TypeError, match="float16 dtype for inputs is only supported for deployment target >= iOS16/macOS13"):
            ct.convert(rank4_grayscale_input_model,
                       inputs=[ct.ImageType(shape=(1, 1, 10, 20),
                                            color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                      )

        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.ImageType(shape=(1, 1, 10, 20),
                                                  color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                             outputs=[ct.TensorType(dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp16")
        verify_prediction(mlmodel)

    def test_color_output(self, rank4_input_model, float32_input_model_add_op):
        # check that an error is raised if the output shape is not of form (1, 3, H, W)
        with pytest.raises(ValueError, match="must have rank 4. Instead it has rank 2"):
            ct.convert(float32_input_model_add_op,
                       inputs=[ct.TensorType(shape=(10, 20))],
                       outputs=[ct.ImageType(color_layout=ct.colorlayout.RGB)],
                       minimum_deployment_target=ct.target.macOS13)

        mlmodel = ct.convert(rank4_input_model,
                             inputs=[ct.ImageType(shape=(1, 3, 10, 20),
                                                  color_layout=ct.colorlayout.BGR)],
                             outputs=[ct.ImageType(color_layout=ct.colorlayout.RGB)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "add", "cast"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.BGR)
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.RGB)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp32")
        verify_prediction(mlmodel)

        # check neural network conversion
        mlmodel = ct.convert(rank4_input_model,
                             inputs=[ct.ImageType(shape=(1, 3, 10, 20),
                                                  color_layout=ct.colorlayout.RGB)],
                             outputs=[ct.ImageType(color_layout=ct.colorlayout.BGR)],
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.RGB)
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.BGR)
        verify_prediction(mlmodel)

    def test_grayscale_output(self, rank4_grayscale_input_model):
        with pytest.raises(TypeError, match="float16 dtype for outputs is only supported for deployment target >= iOS16/macOS13"):
            ct.convert(rank4_grayscale_input_model,
                       inputs=[ct.TensorType(shape=(1, 1, 10, 20))],
                       outputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                       minimum_deployment_target=ct.target.macOS12,
                      )

        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.ImageType(shape=(1, 1, 10, 20),
                                                  color_layout=ct.colorlayout.GRAYSCALE)],
                             outputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE)],
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE)
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE)
        verify_prediction(mlmodel)

        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.ImageType(shape=(1, 1, 10, 20),
                                                  color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                             outputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16)
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp16")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp16")
        verify_prediction(mlmodel)

        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.ImageType(shape=(1, 1, 10, 20),
                                                  color_layout=ct.colorlayout.GRAYSCALE)],
                             outputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "add"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE)
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp16")
        verify_prediction(mlmodel)

    def test_linear_model(self, linear_model):
        # this will test the fuse_linear_bias pass, when the inputs are of type float16
        mlmodel = ct.convert(linear_model,
                             inputs=[ct.TensorType(shape=(1, 10), dtype=np.float16)],
                             outputs=[ct.TensorType(dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp16")
        assert_ops_in_mil_program(mlmodel, ["linear", "relu"])
        verify_prediction(mlmodel)


    def test_classifier(self):
        torch_model = torch.nn.ReLU().eval()
        traced_model = torch.jit.trace(torch_model, torch.rand(3,))
        model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=(3,), dtype=np.float16)],
            outputs=[ct.TensorType(dtype=np.float16)],
            classifier_config = ct.ClassifierConfig(['a', 'b', 'c']),
            convert_to='mlprogram',
            minimum_deployment_target=ct.target.macOS13,
        )
        assert_input_dtype(model, expected_type_str="fp16")
        assert_ops_in_mil_program(model, ["relu", "cast", "classify"])
        spec = model.get_spec()
        input_name = spec.description.input[0].name
        out_dict = model.predict({input_name : np.array([1.0, 2.0, 3.0])})
        assert 'classLabel' in out_dict
        assert out_dict['classLabel'] == 'c'
        assert len(spec.description.output) == 2
        assert "classLabel_probs" in out_dict
        assert isinstance(out_dict["classLabel_probs"], dict)

    def test_prediction_with_fp16_io(self):
        torch_model = torch.nn.Linear(30, 5).eval()
        traced_model = torch.jit.trace(torch_model, torch.rand(1, 30))
        mlmodel = ct.convert(traced_model,
                             inputs=[ct.TensorType(name="input", shape=(1, 30), dtype=np.float32)],
                             outputs=[ct.TensorType(dtype=np.float32)],
                             minimum_deployment_target=ct.target.macOS13,
                             compute_units=ct.ComputeUnit.CPU_ONLY,
                             )
        # test prediction
        sample_input = np.random.rand(1, 30).astype(np.float32) * 10
        model_output = mlmodel.predict({"input": sample_input})[mlmodel._spec.description.output[0].name]
        reference_output = traced_model(torch.from_numpy(sample_input)).detach().numpy()
        np.testing.assert_allclose(reference_output, model_output, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(ct.utils._macos_version() < (13, 0), reason='Tests are for deployment target ios16/macos13')
class TestGrayscaleImagePredictions:

    def test_grayscale_input_image(self, rank4_grayscale_input_model):
        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.ImageType(name="input_image",
                                                  shape=(1, 1, 10, 20),
                                                  color_layout=ct.colorlayout.GRAYSCALE)],
                             outputs=[ct.TensorType(name="output")],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        sample_input = np.random.randint(low=0, high=246, size=(1, 1, 10, 20))
        img_input = Image.fromarray(sample_input[0, 0, :, :].astype(np.uint8), 'L')
        model_output = mlmodel.predict({"input_image": img_input})['output']
        reference_output = rank4_grayscale_input_model(torch.from_numpy(sample_input.astype(np.float32))).detach().numpy()
        np.testing.assert_allclose(reference_output, model_output, rtol=1e-2, atol=1e-2)

    def test_grayscale_fp16_input_image(self, rank4_grayscale_input_model):
        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.ImageType(name="input_image",
                                                  shape=(1, 1, 10, 20),
                                                  color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                             outputs=[ct.TensorType(name="output")],
                             minimum_deployment_target=ct.target.macOS13,
                             )

        # incorrect way to do prediction
        with pytest.raises(TypeError,
                           match="must be of type PIL.Image.Image with mode=='F'",
                           ):
            sample_input = np.random.randint(low=0, high=246, size=(1, 1, 10, 20))
            img_input = Image.fromarray(sample_input[0, 0, :, :].astype(np.uint8), 'L')
            mlmodel.predict({"input_image": img_input})

        # correct way to do prediction
        sample_input = np.random.rand(1, 1, 10, 20) # in between [0, 1]
        img_input = Image.fromarray(sample_input[0, 0, :, :].astype(np.float32), 'F')
        model_output = mlmodel.predict({"input_image": img_input})['output']
        reference_output = rank4_grayscale_input_model(torch.from_numpy(sample_input.astype(np.float32))).detach().numpy()
        np.testing.assert_allclose(reference_output, model_output, rtol=1e-2, atol=1e-2)

    def test_grayscale_output_image(self, rank4_grayscale_input_model):
        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.TensorType(name="input",
                                                  shape=(1, 1, 10, 20))],
                             outputs=[ct.ImageType(name="output_image",
                                                   color_layout=ct.colorlayout.GRAYSCALE)],
                             minimum_deployment_target=ct.target.macOS13,
                             compute_precision=ct.precision.FLOAT32,
                             )
        sample_input = np.random.randint(low=0, high=200, size=(1, 1, 10, 20)).astype(np.float32)
        model_output_pil_image = mlmodel.predict({"input": sample_input})['output_image']
        assert isinstance(model_output_pil_image, Image.Image)
        assert model_output_pil_image.mode == "L"
        model_output_as_numpy = np.array(model_output_pil_image)
        reference_output = rank4_grayscale_input_model(torch.from_numpy(sample_input)).detach().numpy()
        reference_output = np.squeeze(reference_output)
        np.testing.assert_allclose(reference_output, model_output_as_numpy, rtol=1e-2, atol=1e-2)

    def test_grayscale_fp16_output_image(self, rank4_grayscale_input_model):
        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.TensorType(name="input",
                                                  shape=(1, 1, 10, 20))],
                             outputs=[ct.ImageType(name="output_image",
                                                   color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                             minimum_deployment_target=ct.target.macOS13,
                             compute_precision=ct.precision.FLOAT32,
                             )
        sample_input = np.random.randint(low=0, high=200, size=(1, 1, 10, 20)).astype(np.float32)
        model_output_pil_image = mlmodel.predict({"input": sample_input})['output_image']
        assert isinstance(model_output_pil_image, Image.Image)
        assert model_output_pil_image.mode == "F"
        model_output_as_numpy = np.array(model_output_pil_image)
        reference_output = rank4_grayscale_input_model(torch.from_numpy(sample_input)).detach().numpy()
        reference_output = np.squeeze(reference_output)
        np.testing.assert_allclose(reference_output, model_output_as_numpy, rtol=1e-2, atol=1e-2)


