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
from coremltools._deps import _HAS_TF_2

tf = pytest.importorskip("tensorflow")

@pytest.fixture
def int32_input_model():
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.int32, shape=[10, 20], name="input")
        out = tf.add(x, tf.constant(5, dtype=tf.int32), name="output")
    return graph

@pytest.fixture
def float32_input_model_add_op():
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.float32, shape=[10, 20], name="input")
        out = tf.add(x, tf.constant(5.5, dtype=tf.float32), name="output")
    return graph

@pytest.fixture
def float32_input_model_relu_ops():
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.float32, shape=[10, 20], name="input")
        x1 = tf.nn.relu(x)
        out = tf.nn.relu(x1, name="output")
    return graph

@pytest.fixture
def int64_input_model():
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.int64, shape=[10, 20], name="input")
        out = tf.add(x, tf.constant(5, dtype=tf.int64), name="output")
    return graph

@pytest.fixture
def float32_two_input_model():
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.float32, shape=[10, 20], name="input1")
        y = tf.placeholder(tf.float32, shape=[10, 20], name="input2")
        out = tf.add(x, y, name="output")
    return graph

@pytest.fixture
def float32_two_output_model():
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.float32, shape=[10, 20], name="input")
        y = tf.nn.relu(x)
        out2 = tf.nn.relu6(x, name="output2")
        out1 = tf.nn.relu(y, name="output1")
    return graph

@pytest.fixture
def rank3_input_model():
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.float32, shape=[1, 10, 20], name="input")
        out = tf.add(x, tf.constant(5, dtype=tf.float32), name="output")
    return graph

@pytest.fixture
def rank4_input_model():
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.float32, shape=[1, 10, 20, 3], name="input")
        out = tf.add(x, tf.constant(5, dtype=tf.float32), name="output")
    return graph

@pytest.fixture
def rank4_input_model_with_channel_first_output():
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.float32, shape=[1, 10, 20, 3], name="input")
        y = tf.add(x, tf.constant(5, dtype=tf.float32))
        out = tf.transpose(y, perm=[0, 3, 1, 2], name="output")
    return graph

@pytest.fixture
def rank4_grayscale_input_model():
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.float32, shape=[1, 10, 20, 1], name="input")
        out = tf.add(x, tf.constant(5, dtype=tf.float32), name="output")
    return graph

@pytest.fixture
def rank4_grayscale_input_model_with_channel_first_output():
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.float32, shape=[1, 10, 20, 1], name="input")
        y = tf.add(x, tf.constant(5, dtype=tf.float32))
        out = tf.transpose(y, perm=[0, 3, 1, 2], name="output")
    return graph

@pytest.fixture
def linear_model():
    # this model will test the fuse_matmul_weight_bias pass
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.float32, shape=[1, 2], name="input")
        y = tf.matmul(x, tf.constant([1, 2], shape=(2, 4), dtype=tf.float32))
        y = tf.add(y, tf.constant([1, 2, 3, 4], shape=(4,), dtype=tf.float32))
        out = tf.nn.relu(y)
    return graph


@pytest.mark.skipif(ct.utils._macos_version() < (13, 0), reason='Tests are for deployment target ios16/macos13')
class TestInputOutputConversionAPI:

    def test_input_dtype_inferred(self, int32_input_model):
        # test that the input dtype is picked up from TF correctly
        mlmodel = ct.convert(int32_input_model,
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="int32")
        verify_prediction(mlmodel)

    def test_unsupported_input_dtype_in_tf_graph(self, int64_input_model):
        # test that no error is raised when no dtype is provided by the user,
        # and the TF graph's input dtype is not supported.
        # In this case, it will be mapped to the closest supported dtype
        mlmodel = ct.convert(int64_input_model,
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="int32")
        verify_prediction(mlmodel)

    def test_input_dtype_user_provided(self, int32_input_model):
        # test that provided dtype in the api overrides the input dtype in the TF model
        mlmodel = ct.convert(int32_input_model,
                             inputs=[ct.TensorType(dtype=np.float32)],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="fp32")
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

    def test_invalid_input_dtype(self, int32_input_model):
        # error should be raised if a dtype is provided by the user that is not supported
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
                             inputs=[ct.TensorType(dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add", "cast"])
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

        mlmodel = ct.convert(float32_input_model_relu_ops,
                             inputs=[ct.TensorType(dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["relu", "relu", "cast"])
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

        mlmodel = ct.convert(int32_input_model,
                             inputs=[ct.TensorType(dtype=np.float16)],
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
                             inputs=[ct.TensorType(dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             compute_precision=ct.precision.FLOAT32,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "add"])
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

        mlmodel = ct.convert(float32_input_model_relu_ops,
                             inputs=[ct.TensorType(dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             compute_precision=ct.precision.FLOAT32,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "relu", "relu"])
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp32")

    def test_two_input_model(self, float32_two_input_model):
        # test forcing input type of "input1" to be int32
        mlmodel = ct.convert(float32_two_input_model,
                             inputs=[ct.TensorType(name="input1", dtype=np.int32)],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="int32", expected_name="input1")
        assert_input_dtype(mlmodel, expected_type_str="fp32", expected_name="input2")
        assert_output_dtype(mlmodel, expected_type_str="fp32")

        # test forcing both inputs to be int32
        mlmodel = ct.convert(float32_two_input_model,
                             inputs=[ct.TensorType(name="input1", dtype=np.int32),
                                     ct.TensorType(name="input2", dtype=np.int32),
                                     ],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="int32", expected_name="input1")
        assert_input_dtype(mlmodel, expected_type_str="int32", expected_name="input2")
        assert_output_dtype(mlmodel, expected_type_str="int32")

        # if names are not provided an error should be raised
        with pytest.raises(ValueError):
            mlmodel = ct.convert(float32_two_input_model,
                                 inputs=[ct.TensorType(dtype=np.int32),
                                         ct.TensorType(dtype=np.int32),
                                         ],
                                 minimum_deployment_target=ct.target.macOS12)

        # test forcing both inputs to be float16
        mlmodel = ct.convert(float32_two_input_model,
                             inputs=[ct.TensorType(name="input1", dtype=np.float16),
                                     ct.TensorType(name="input2", dtype=np.float16),
                                     ],
                             minimum_deployment_target=ct.target.macOS13)
        assert_input_dtype(mlmodel, expected_type_str="fp16", expected_name="input1")
        assert_input_dtype(mlmodel, expected_type_str="fp16", expected_name="input2")
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        assert_cast_ops_count(mlmodel, expected_count=1)
        verify_prediction(mlmodel)

    def test_single_output_model(self, int32_input_model, float32_input_model_relu_ops):
        # test output type
        mlmodel = ct.convert(int32_input_model,
                             minimum_deployment_target=ct.target.macOS12)
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add"])
        assert_output_dtype(mlmodel, expected_type_str="int32")

        # test that error is raised when an output of unknown name is provided
        with pytest.raises(Exception):
            # output name does not exist in the model
            mlmodel = ct.convert(int32_input_model,
                                 outputs=["z"],
                                 minimum_deployment_target=ct.target.macOS12)

        # test that error is raised when two outputs are provided without names
        with pytest.raises(ValueError, match=", does not have names"):
            mlmodel = ct.convert(int32_input_model,
                                 outputs=[ct.TensorType(dtype=np.float32), ct.TensorType(dtype=np.float32)],
                                 minimum_deployment_target=ct.target.macOS12)

        # test that an error is raised when shape is provided for the output
        with pytest.raises(ValueError):
            mlmodel = ct.convert(int32_input_model,
                                 outputs=[ct.TensorType(dtype=np.float32, shape=(10, 20))],
                                 minimum_deployment_target=ct.target.macOS12)

        # test that the output dtype provided by the user is applied during conversion
        mlmodel = ct.convert(int32_input_model,
                             outputs=[ct.TensorType(dtype=np.float32)],
                             minimum_deployment_target=ct.target.macOS12)
        assert_output_dtype(mlmodel, expected_type_str="fp32", expected_name="Identity" if _HAS_TF_2 else "output")
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add", "cast"])

        # test that output dtype of float16 is rejected when deployment target is low
        with pytest.raises(TypeError,
                           match="float16 dtype for outputs is only supported for deployment target >= iOS16/macOS13"
                           ):
            ct.convert(float32_input_model_relu_ops,
                       outputs=[ct.TensorType(dtype=np.float16)],
                       minimum_deployment_target=ct.target.macOS12,
                       )

        # test that output type float16 is applied correctly
        mlmodel = ct.convert(float32_input_model_relu_ops,
                             outputs=[ct.TensorType(dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_output_dtype(mlmodel, expected_type_str="fp16", expected_name="Identity" if _HAS_TF_2 else "output")
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "relu", "relu"])

        # test that input and output types float16 are applied correctly
        mlmodel = ct.convert(float32_input_model_relu_ops,
                             inputs=[ct.TensorType(dtype=np.float16)],
                             outputs=[ct.TensorType(dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp16", expected_name="Identity" if _HAS_TF_2 else "output")
        assert_ops_in_mil_program(mlmodel, expected_op_list=["relu", "relu"])
        verify_prediction(mlmodel)

    def test_multi_output_model(self, float32_two_output_model):
        # check that error is raised when only 1 output provided
        with pytest.raises(ValueError, match="please provide names for each of the outputs"):
            mlmodel = ct.convert(float32_two_output_model,
                                 outputs=[ct.TensorType(dtype=np.float16)],
                                 minimum_deployment_target=ct.target.macOS13,
                                 )

        # check that error is raised when multiple outputs are provided without names
        with pytest.raises(ValueError, match="please provide names for each of the outputs"):
            mlmodel = ct.convert(float32_two_output_model,
                                 outputs=[ct.TensorType(dtype=np.float16), ct.TensorType(dtype=np.float32)],
                                 minimum_deployment_target=ct.target.macOS13,
                                 )

        # set 1 output to float16 and the other to float32
        output1_name = "Identity" if _HAS_TF_2 else "output1"
        output2_name = "Identity_1" if _HAS_TF_2 else "output2"
        mlmodel = ct.convert(float32_two_output_model,
                             inputs=[ct.TensorType(dtype=np.float16)],
                             outputs=[ct.TensorType(name=output2_name, dtype=np.float16),
                                      ct.TensorType(name=output1_name, dtype=np.float32)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_cast_ops_count(mlmodel, expected_count=1)
        assert_output_dtype(mlmodel, expected_type_str="fp16", expected_name=output2_name, index=0)
        assert_output_dtype(mlmodel, expected_type_str="fp32", expected_name=output1_name, index=1)
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        verify_prediction(mlmodel)

        # in this case only the single output will be selected
        mlmodel = ct.convert(float32_two_output_model,
                             inputs=[ct.TensorType(dtype=np.float16)],
                             outputs=[ct.TensorType(name=output2_name, dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_cast_ops_count(mlmodel, expected_count=0)
        assert_output_dtype(mlmodel, expected_type_str="fp16", expected_name=output2_name, index=0)
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        verify_prediction(mlmodel)

    def test_color_input(self, rank4_input_model, rank3_input_model):
        mlmodel = ct.convert(rank4_input_model,
                             inputs=[ct.ImageType(color_layout=ct.colorlayout.RGB)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "transpose", "add", "cast"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.RGB)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp32")
        verify_prediction(mlmodel)

        with pytest.raises(ValueError, match="must have rank 4"):
            mlmodel = ct.convert(rank3_input_model,
                                 inputs=[ct.ImageType(color_layout=ct.colorlayout.RGB)],
                                 minimum_deployment_target=ct.target.macOS12,
                                 )

    def test_grayscale_input(self, rank4_input_model, rank3_input_model, rank4_grayscale_input_model):
        with pytest.raises(ValueError, match="must have rank 4"):
            mlmodel = ct.convert(rank3_input_model,
                                 inputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE)],
                                 minimum_deployment_target=ct.target.macOS13,
                                 )

        # invalid shape
        with pytest.raises(ValueError):
            mlmodel = ct.convert(rank4_input_model,
                                 inputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE)],
                                 minimum_deployment_target=ct.target.macOS13,
                                 )

        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "transpose", "add", "cast"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp32")
        verify_prediction(mlmodel)

        with pytest.raises(TypeError, match="float16 dtype for inputs is only supported for deployment target >= iOS16/macOS13"):
            mlmodel = ct.convert(rank4_grayscale_input_model,
                                 inputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                                 minimum_deployment_target=ct.target.macOS12,
                                 )

        # test that grayscale_16 raises error when used with neural network
        with pytest.raises(TypeError, match="float16 dtype for inputs is only supported for deployment target >= iOS16/macOS13"):
            mlmodel = ct.convert(rank4_grayscale_input_model,
                                 inputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                                 )

        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                             outputs=[ct.TensorType(dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["transpose", "add"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp16")
        verify_prediction(mlmodel)

    def test_color_output(self, rank4_input_model, rank4_input_model_with_channel_first_output):
        # check that an error is raised if the output shape is not of form (1, 3, H, W)
        with pytest.raises(ValueError, match="Shape of the RGB/BGR image output,"):
            mlmodel = ct.convert(rank4_input_model,
                                 inputs=[ct.ImageType(color_layout=ct.colorlayout.RGB)],
                                 outputs=[ct.ImageType(color_layout=ct.colorlayout.RGB)],
                                 minimum_deployment_target=ct.target.macOS13,
                                 )

        mlmodel = ct.convert(rank4_input_model_with_channel_first_output,
                             inputs=[ct.ImageType(color_layout=ct.colorlayout.BGR)],
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
        mlmodel = ct.convert(rank4_input_model_with_channel_first_output,
                             inputs=[ct.ImageType(color_layout=ct.colorlayout.RGB)],
                             outputs=[ct.ImageType(color_layout=ct.colorlayout.BGR)],
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.RGB)
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.BGR)
        verify_prediction(mlmodel)

    def test_grayscale_output(self, rank4_grayscale_input_model, rank4_grayscale_input_model_with_channel_first_output):
        # check that an error is raised if the output shape is not of form (1, 1, H, W)
        with pytest.raises(ValueError, match="Shape of the Grayscale image output,"):
            mlmodel = ct.convert(rank4_grayscale_input_model,
                                 inputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE)],
                                 outputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE)],
                                 )

        with pytest.raises(TypeError, match="float16 dtype for outputs is only supported for deployment target >= iOS16/macOS13"):
            mlmodel = ct.convert(rank4_grayscale_input_model_with_channel_first_output,
                                 outputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                                 minimum_deployment_target=ct.target.macOS12,
                                 )

        mlmodel = ct.convert(rank4_grayscale_input_model_with_channel_first_output,
                             inputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE)],
                             outputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE)],
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE)
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE)
        verify_prediction(mlmodel)

        mlmodel = ct.convert(rank4_grayscale_input_model_with_channel_first_output,
                             inputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                             outputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_cast_ops_count(mlmodel, expected_count=0)
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16)
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp16")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp16")
        verify_prediction(mlmodel)

        mlmodel = ct.convert(rank4_grayscale_input_model_with_channel_first_output,
                             inputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE)],
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
        # this will test the fuse_matmul_weight_bias pass, when the inputs are of type float16
        mlmodel = ct.convert(linear_model,
                             inputs=[ct.TensorType(dtype=np.float16)],
                             outputs=[ct.TensorType(dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp16")
        assert_ops_in_mil_program(mlmodel, ["linear", "relu"])
        verify_prediction(mlmodel)
