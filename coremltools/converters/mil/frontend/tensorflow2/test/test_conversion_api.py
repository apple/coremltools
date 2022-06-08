#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest

tf = pytest.importorskip("tensorflow", minversion="2.1.0")

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


from coremltools.converters.mil.frontend.tensorflow.test.test_conversion_api import TestInputOutputConversionAPI