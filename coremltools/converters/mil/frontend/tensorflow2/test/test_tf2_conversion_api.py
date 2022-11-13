#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os
import platform
import urllib
from io import BytesIO
from os import chdir, getcwd
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
import pytest
import requests
from PIL import Image

import coremltools as ct
from coremltools.converters.mil.mil import types

tf = pytest.importorskip("tensorflow", minversion="2.1.0")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@pytest.fixture
def int32_input_model():
    x = tf.keras.Input(batch_input_shape=(10, 20), name="input", dtype=tf.int32)
    out = tf.add(x, tf.constant(5, dtype=tf.int32), name="output")
    return tf.keras.Model(inputs=x, outputs=out)

@pytest.fixture
def float32_input_model_add_op():
    x = tf.keras.Input(batch_input_shape=(10, 20), name="input", dtype=tf.float32)
    out = tf.add(x, tf.constant(5.5, dtype=tf.float32), name="output")
    return tf.keras.Model(inputs=x, outputs=out)

@pytest.fixture
def float32_input_model_relu_ops():
    x = tf.keras.Input(batch_input_shape=(10, 20), name="input", dtype=tf.float32)
    x1 = tf.keras.layers.ReLU()(x)
    out = tf.keras.layers.ReLU(name="output")(x1)
    return tf.keras.Model(inputs=x, outputs=out)

@pytest.fixture
def int64_input_model():
    x = tf.keras.Input(batch_input_shape=(10, 20), name="input", dtype=tf.int64)
    out = tf.add(x, tf.constant(5, dtype=tf.int64), name="output")
    return tf.keras.Model(inputs=x, outputs=out)

@pytest.fixture
def float32_two_input_model():
    x = tf.keras.Input(batch_input_shape=(10, 20), name="input1", dtype=tf.float32)
    y = tf.keras.Input(batch_input_shape=(10, 20), name="input2", dtype=tf.float32)
    out = tf.add(x, y, name="output")
    return tf.keras.Model(inputs=[x, y], outputs=out)

@pytest.fixture
def float32_two_output_model():
    x = tf.keras.Input(batch_input_shape=(10, 20), name="input", dtype=tf.float32)
    y = tf.nn.relu(x)
    out2 = tf.nn.relu6(x, name="output2")
    out1 = tf.nn.relu(y, name="output1")
    return tf.keras.Model(inputs=x, outputs=[out1, out2])

@pytest.fixture
def rank3_input_model():
    x = tf.keras.Input(batch_input_shape=(1, 10, 20), name="input", dtype=tf.float32)
    out = tf.add(x, tf.constant(5, dtype=tf.float32), name="output")
    return tf.keras.Model(inputs=x, outputs=out)

@pytest.fixture
def rank4_input_model():
    x = tf.keras.Input(batch_input_shape=(1, 10, 20, 3), name="input", dtype=tf.float32)
    out = tf.add(x, tf.constant(5, dtype=tf.float32), name="output")
    return tf.keras.Model(inputs=x, outputs=out)

@pytest.fixture
def rank4_input_model_with_channel_first_output():
    x = tf.keras.Input(batch_input_shape=(1, 10, 20, 3), name="input", dtype=tf.float32)
    y = tf.add(x, tf.constant(5, dtype=tf.float32))
    out = tf.transpose(y, perm=[0, 3, 1, 2], name="output")
    return tf.keras.Model(inputs=x, outputs=out)

@pytest.fixture
def rank4_grayscale_input_model():
    x = tf.keras.Input(batch_input_shape=(1, 10, 20, 1), name="input", dtype=tf.float32)
    out = tf.add(x, tf.constant(5, dtype=tf.float32), name="output")
    return tf.keras.Model(inputs=x, outputs=out)

@pytest.fixture
def rank4_grayscale_input_model_with_channel_first_output():
    x = tf.keras.Input(batch_input_shape=(1, 10, 20, 1), name="input", dtype=tf.float32)
    y = tf.add(x, tf.constant(5, dtype=tf.float32))
    out = tf.transpose(y, perm=[0, 3, 1, 2], name="output")
    return tf.keras.Model(inputs=x, outputs=out)

@pytest.fixture
def linear_model():
    # this model will test the fuse_matmul_weight_bias pass
    x = tf.keras.Input(batch_input_shape=(1, 10), name="input", dtype=tf.float32)
    y = tf.keras.layers.Dense(4)(x)
    y = tf.add(y, tf.constant([1, 2, 3, 4], shape=(4,), dtype=tf.float32))
    out = tf.nn.relu(y)
    return tf.keras.Model(inputs=x, outputs=out)



#################################################################################
# Note: all tests are also used as examples in https://coremltools.readme.io/docs
# as a reference.
# Whenever any of the following test fails, we should update API documentations
#################################################################################

class TestTensorFlow2ConverterExamples:
    def setup_class(self):
        self._cwd = getcwd()
        self._temp_dir = mkdtemp()
        # step into temp directory as working directory
        # to make the user-facing examples cleaner
        chdir(self._temp_dir)

        # create toy models for conversion examples
        # write a toy tf.keras HDF5 model
        tf_keras_model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation=tf.nn.relu),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax),
            ]
        )
        tf_keras_model.save("./tf_keras_model.h5")

        # write a toy SavedModel directory
        tf_keras_model.save("./saved_model", save_format="tf")

    def teardown_class(self):
        chdir(self._cwd)
        if os.path.exists(self._temp_dir):
            rmtree(self._temp_dir)
    
    @staticmethod
    def test_convert_tf_keras_h5_file():
        if platform.machine() == "arm64":
            pytest.xfail("rdar://101162740 ([CI] [TF] The tf_keras_h5_file API testing is failing on M1 with new OS)")
            
        for file_extension in ("h5", "hdf5"):
            x = tf.keras.Input(shape=(32,), name="input")
            y = tf.keras.layers.Dense(16, activation="softmax")(x)
            keras_model = tf.keras.Model(x, y)
            temp_dir = mkdtemp()
            save_dir = str(temp_dir)
            path = os.path.join(save_dir, "tf_keras_model." + file_extension)
            keras_model.save(path)
            mlmodel = ct.convert(path)

            test_input = np.random.rand(2, 32)
            expected_val = keras_model(test_input)
            results = mlmodel.predict({"input": test_input})
            np.testing.assert_allclose(results["Identity"], expected_val, rtol=1e-4)

    @staticmethod
    def test_convert_tf_keras_model():
        x = tf.keras.Input(shape=(32,), name="input")
        y = tf.keras.layers.Dense(16, activation="softmax")(x)
        keras_model = tf.keras.Model(x, y)

        mlmodel = ct.convert(keras_model)

        test_input = np.random.rand(2, 32)
        expected_val = keras_model(test_input)
        results = mlmodel.predict({"input": test_input})
        np.testing.assert_allclose(results["Identity"], expected_val, rtol=1e-4)

    @staticmethod
    @pytest.mark.parametrize(
            "dtype", ['default', 'mil_type', 'np type'])
    def test_convert_tf_keras_applications_model(dtype):
        tf_keras_model = tf.keras.applications.MobileNet(
            weights="imagenet", input_shape=(224, 224, 3)
        )

        # inputs / outputs are optional, we can get from tf.keras model
        # this can be extremely helpful when we want to extract sub-graphs
        input_name = tf_keras_model.inputs[0].name.split(":")[0]

        if dtype == 'default':
            dtype = None
        elif dtype == 'mil_type':
            dtype = types.fp32
        else:
            dtype = np.float32

        mlmodel = ct.convert(
            tf_keras_model,
            inputs=[ct.TensorType(shape=(1, 224, 224, 3), dtype=dtype)],
        )
        mlmodel.save("./mobilenet.mlmodel")

    @staticmethod
    def test_convert_from_saved_model_dir():
        # SavedModel directory generated by TensorFlow 2.x
        mlmodel = ct.convert("./saved_model")
        mlmodel.save("./model.mlmodel")


    @staticmethod
    def test_keras_custom_layer_model():
        # testing : https://coremltools.readme.io/docs/tensorflow-2#conversion-from-user-defined-models

        class CustomDense(layers.Layer):
            def __init__(self, units=32):
                super(CustomDense, self).__init__()
                self.units = units

            def build(self, input_shape):
                self.w = self.add_weight(
                    shape=(input_shape[-1], self.units),
                    initializer="random_normal",
                    trainable=True,
                )
                self.b = self.add_weight(
                    shape=(self.units,), initializer="random_normal", trainable=True
                )

            def call(self, inputs):
                return tf.matmul(inputs, self.w) + self.b

        inputs = keras.Input((4,))
        outputs = CustomDense(10)(inputs)
        model = keras.Model(inputs, outputs)
        ct.convert(model)

    @staticmethod
    def test_concrete_function_conversion():
        # testing : https://coremltools.readme.io/docs/tensorflow-2#conversion-from-user-defined-models

        @tf.function(input_signature=[tf.TensorSpec(shape=(6,), dtype=tf.float32)])
        def gelu_tanh_activation(x):
            a = (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))
            y = 0.5 * (1.0 + tf.tanh(a))
            return x * y

        conc_func = gelu_tanh_activation.get_concrete_function()
        mlmodel = ct.convert([conc_func])

    @staticmethod
    def test_convert_tf2_keras():
        x = tf.keras.Input(shape=(32,), name="input")
        y = tf.keras.layers.Dense(16, activation="softmax")(x)
        keras_model = tf.keras.Model(x, y)
        model = ct.convert(keras_model, convert_to='milinternal')
        assert isinstance(model, ct.converters.mil.Program)


class TestTF2FlexibleInput:
    # Test examples in https://coremltools.readme.io/docs/flexible-inputs
    @staticmethod
    @pytest.mark.parametrize("use_symbol", [True, False])
    def test_tf2keras_shared_range_dim(use_symbol):
        input_dim = 3
        # None denotes seq_len dimension
        x1 = tf.keras.Input(shape=(None,input_dim), name="seq1")
        x2 = tf.keras.Input(shape=(None,input_dim), name="seq2")
        y = x1 + x2
        keras_model = tf.keras.Model(inputs=[x1, x2], outputs=[y])

        # One RangeDim shared by two inputs
        if use_symbol:
            seq_len_dim = ct.RangeDim(symbol='seq_len')
        else:
            # symbol is optional
            seq_len_dim = ct.RangeDim()
        seq1_input = ct.TensorType(name="seq1", shape=(1, seq_len_dim, input_dim))
        seq2_input = ct.TensorType(name="seq2", shape=(1, seq_len_dim, input_dim))
        mlmodel = ct.convert(keras_model,
                inputs=[seq1_input, seq2_input])

        batch = 1
        seq_len = 5
        test_input_x1 = np.random.rand(batch, seq_len, input_dim).astype(np.float32)
        test_input_x2 = np.random.rand(batch, seq_len, input_dim).astype(np.float32)
        expected_val = keras_model([test_input_x1, test_input_x2])
        if ct.utils._is_macos():
            results = mlmodel.predict({
                "seq1": test_input_x1,
                "seq2": test_input_x2})
            np.testing.assert_allclose(results["Identity"], expected_val,
                rtol=1e-2, atol=1e-2)


    @staticmethod
    def test_tf2keras_incorrect_range_dim():
        input_dim = 3
        # None denotes seq_len dimension
        x1 = tf.keras.Input(shape=(None,input_dim), name="seq1")
        y = x1 + 1
        keras_model = tf.keras.Model(inputs=[x1], outputs=[y])

        # Incorrectly using -1 instead of ct.RangeDim
        # One RangeDim shared by two inputs
        with pytest.raises(ValueError,
            match=r"Can\'t convert to CoreML shaping"):
            seq1_input = ct.TensorType(name="seq1", shape=(1, -1, input_dim))
            mlmodel = ct.convert(keras_model, inputs=[seq1_input])

    @staticmethod
    @pytest.mark.parametrize("use_symbol", [True, False])
    def test_tf2keras_outofbound_range_dim(use_symbol):
        input_dim = 3
        # None denotes seq_len dimension
        x = tf.keras.Input(shape=(None,input_dim), name="seq")
        y = x * 2
        keras_model = tf.keras.Model(inputs=[x], outputs=[y])

        if use_symbol:
            seq_len_dim = ct.RangeDim(symbol='sequence_len', lower_bound=3,
                    upper_bound=5)
        else:
            seq_len_dim = ct.RangeDim(lower_bound=3, upper_bound=5)
        seq_input = ct.TensorType(name="seq", shape=(1, seq_len_dim, input_dim))
        mlmodel = ct.convert(keras_model, inputs=[seq_input])

        # seq_len is within bound
        batch = 1
        seq_len = 3
        test_input_x = np.random.rand(batch, seq_len, input_dim).astype(np.float32)
        expected_val = keras_model([test_input_x])
        if ct.utils._is_macos():
            results = mlmodel.predict({"seq": test_input_x})
            np.testing.assert_allclose(results["Identity"], expected_val,
                rtol=1e-4, atol=1e-3)

            # seq_len below/above lower_bound/upper_bound
            with pytest.raises(RuntimeError,
                    match=r"Size \(2\) of dimension \(1\) is not in allowed range \(3\.\.5\)"):
                seq_len = 2
                test_input_x = np.random.rand(batch, seq_len,
                        input_dim).astype(np.float32)
                results = mlmodel.predict({"seq": test_input_x})

            with pytest.raises(RuntimeError,
                    match=r"Size \(6\) of dimension \(1\) is not in allowed range \(3\.\.5\)"):
                seq_len = 6
                test_input_x = np.random.rand(batch, seq_len,
                        input_dim).astype(np.float32)
                results = mlmodel.predict({"seq": test_input_x})

    @staticmethod
    def test_tf2_image_enumerated_shapes():
        keras_model = tf.keras.applications.MobileNetV2(
            input_shape=(None, None, 3,),
            classes=1000,
            include_top=False,
        )
        input_shapes = ct.EnumeratedShapes(shapes=[(1, 192, 192, 3), (1, 224, 224, 3)])
        image_input = ct.ImageType(shape=input_shapes,
                                   bias=[-1,-1,-1], scale=1/127)
        model = ct.convert(keras_model, inputs=[image_input])
        assert model is not None
        spec = model.get_spec()
        assert len(spec.description.input[0].type.imageType.enumeratedSizes.sizes) == 2

    @staticmethod
    def test_tf2keras_enumerated_shapes():
        input_shape = (28, 28, 3)
        # None denotes seq_len dimension
        x = tf.keras.Input(shape=input_shape, name="input")
        C_out = 2
        kHkW = 3
        y = tf.keras.layers.Conv2D(C_out, kHkW, activation='relu',
                input_shape=input_shape)(x)
        keras_model = tf.keras.Model(inputs=[x], outputs=[y])

        # One RangeDim shared by two inputs
        shapes = [(1, 28, 28, 3), (1, 56, 56, 3)]
        enumerated_shapes = ct.EnumeratedShapes(shapes=shapes)
        tensor_input = ct.TensorType(name="input", shape=enumerated_shapes)
        mlmodel = ct.convert(keras_model, inputs=[tensor_input])

        # Test (1, 28, 28, 3) shape
        test_input_x = np.random.rand(*shapes[0]).astype(np.float32)
        expected_val = keras_model([test_input_x])
        if ct.utils._is_macos():
            results = mlmodel.predict({
                "input": test_input_x})
            # rdar://101303143 ([CI] test_tf2keras_enumerated_shapes is getting some stochastic numerical issues on intel machines)
            # The tolerance is set a little bit big here. Need to investigate this issue if possible and lower the threshold down.
            np.testing.assert_allclose(results["Identity"],
                    expected_val, atol=1e-2, rtol=3)

            # Test (1, 56, 56, 3) shape (can't verify numerical parity with Keras
            # which doesn't support enumerated shape)
            test_input_x = np.random.rand(*shapes[1]).astype(np.float32)
            results = mlmodel.predict({
                "input": test_input_x})

            # Test with a wrong shape
            with pytest.raises(RuntimeError,
                    match=r"MultiArray Shape \(1 x 29 x 29 x 3\) was not in enumerated set of allowed shapes"):
                test_input_x = np.random.rand(1, 29, 29, 3).astype(np.float32)
                results = mlmodel.predict({
                    "input": test_input_x})

    @staticmethod
    def test_tf2keras_optional_input():
        input_dim = 3
        # None denotes seq_len dimension
        x1 = tf.keras.Input(shape=(None,input_dim), name="optional_input")
        x2 = tf.keras.Input(shape=(None,input_dim), name="required_input")
        y = x1 + x2
        keras_model = tf.keras.Model(inputs=[x1, x2], outputs=[y])

        seq_len_dim = ct.RangeDim()
        default_value = np.ones((1, 2, input_dim)).astype(np.float32)
        optional_input = ct.TensorType(
            name="optional_input",
            shape=(1, seq_len_dim, input_dim),
            default_value=default_value,
          )
        required_input = ct.TensorType(
            name="required_input",
            shape=(1, seq_len_dim, input_dim),
          )
        mlmodel = ct.convert(keras_model,
                inputs=[optional_input, required_input])

        batch = 1
        seq_len = 2
        test_input_x2 = np.random.rand(batch, seq_len, input_dim).astype(np.float32)
        expected_val = keras_model([default_value, test_input_x2])
        if ct.utils._is_macos():
            results = mlmodel.predict({"required_input": test_input_x2})
            np.testing.assert_allclose(results["Identity"], expected_val, rtol=1e-2)
