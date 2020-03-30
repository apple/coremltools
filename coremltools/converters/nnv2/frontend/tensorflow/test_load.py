import os
import pytest
import shutil
import tempfile
import numpy as np
from coremltools.converters.nnv2._deps import (
    HAS_TF1, HAS_TF2, MSG_TF1_NOT_FOUND, MSG_TF2_NOT_FOUND
)

from coremltools.models import MLModel
from coremltools.converters.nnv2 import converter

if HAS_TF1 or HAS_TF2:
    import tensorflow as tf


def get_keras_io_names(model):
    input_name = model.inputs[0].name.split(':')[0]
    output_name = model.outputs[0].name.split(':')[0].split('/')[-1]
    return input_name, output_name


@pytest.mark.skipif(not HAS_TF2, reason=MSG_TF2_NOT_FOUND)
class TestKerasFashionMnist:

    def setup(self):
        self.input_shape = (1, 28, 28)
        self.saved_model_dir = tempfile.mkdtemp()
        _, self.model_path_h5 = tempfile.mkstemp(
            suffix='.h5', prefix=self.saved_model_dir)

    def teardown(self):
        if os.path.exists(self.saved_model_dir):
            shutil.rmtree(self.saved_model_dir)

    @staticmethod
    def _build_model_sequential():
        keras_model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        return keras_model

    @staticmethod
    def _build_model_functional():
        inputs = tf.keras.Input(shape=(28, 28))
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)
        outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x)
        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return keras_model

    def _test_conversion_prediction(self, keras_model, model_path, inputs, outputs):
        # convert and validate
        proto = converter.convert(
            model_path,
            inputs=inputs,
            outputs=outputs
        )
        model = MLModel(proto)

        # verify numeric correctness of predictions
        inputs = np.random.rand(*self.input_shape)
        keras_prediction = keras_model.predict(inputs)
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        prediction = model.predict({keras_model.inputs[0].name.split(':')[0]: inputs})[output_name]
        np.testing.assert_array_equal(keras_prediction.shape, prediction.shape)
        np.testing.assert_almost_equal(keras_prediction.flatten(), prediction.flatten(), decimal=4)

    def test_sequential_builder_keras_model_format(self):
        keras_model = self._build_model_sequential()
        # save model as Keras hdf5 .h5 model file
        keras_model.save(self.model_path_h5)
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]

        self._test_conversion_prediction(
            keras_model=keras_model,
            model_path=self.model_path_h5,
            inputs={input_name: self.input_shape},
            outputs=[output_name]
        )

    def test_sequential_builder_saved_model_format(self):
        keras_model = self._build_model_sequential()
        # save model as SavedModel directory
        keras_model.save(self.saved_model_dir, save_format='tf')
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_conversion_prediction(
            keras_model=keras_model,
            model_path=self.saved_model_dir,
            inputs={input_name: self.input_shape},
            outputs=[output_name]
        )

    def test_functional_builder(self):
        keras_model = self._build_model_functional()
        # save model as Keras hdf5 .h5 model file
        keras_model.save(self.model_path_h5)
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_conversion_prediction(
            keras_model=keras_model,
            model_path=self.model_path_h5,
            inputs={input_name: self.input_shape},
            outputs=[output_name]
        )


class TestModelInputsOutputs:

    def setup(self):
        self.saved_model_dir = tempfile.mkdtemp()
        _, self.model_path_h5 = tempfile.mkstemp(
            suffix='.h5', prefix=self.saved_model_dir)
        _, self.model_path_pb = tempfile.mkstemp(
            suffix='.pb', prefix=self.saved_model_dir)

    def teardown(self):
        if os.path.exists(self.saved_model_dir):
            shutil.rmtree(self.saved_model_dir)

    def test_infer_inputs(self):
        if HAS_TF2:
            m = tf.keras.Sequential()
            m.add(tf.keras.layers.ReLU(input_shape=(4, 5), batch_size=3))
            m.save(self.model_path_h5)
            model = self.model_path_h5
            _, output_name = get_keras_io_names(m)
        elif HAS_TF1:
            with tf.Graph().as_default() as model:
                out = tf.nn.relu(tf.placeholder(tf.float32, shape=(3, 4, 5)))
            output_name = out.op.name
        proto = converter.convert(model, outputs=[output_name])
        assert MLModel(proto) is not None

    @pytest.mark.skip('Not yet implemented.')
    def test_infer_outputs(self):
        if HAS_TF2:
            m = tf.keras.Sequential()
            m.add(tf.keras.layers.ReLU(input_shape=(4, 5), batch_size=3))
            m.save(self.model_path_h5)
            model = self.model_path_h5
            input_name, _ = get_keras_io_names(m)
        elif HAS_TF1:
            with tf.Graph().as_default() as model:
                x = tf.placeholder(tf.float32, shape=(3, 4, 5))
                tf.nn.relu(x)
            input_name = x.op.name
        proto = converter.convert(model, inputs={input_name: (3, 4, 5)})
        assert MLModel(proto) is not None

    def test_infer_inputs_and_outputs(self):
        if HAS_TF2:
            m = tf.keras.Sequential()
            m.add(tf.keras.layers.ReLU(input_shape=(4, 5), batch_size=3))
            m.save(self.model_path_h5)
            model = self.model_path_h5
        elif HAS_TF1:
            with tf.Graph().as_default() as model:
                tf.nn.relu(tf.placeholder(tf.float32, shape=(3, 4, 5)))
        proto = converter.convert(model)
        assert MLModel(proto) is not None

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    def test_op_as_outputs(self):
        with tf.Graph().as_default() as model:
            out = tf.nn.relu(tf.placeholder(tf.float32, shape=(3, 4, 5)))
        proto = converter.convert(model, outputs=out)
        assert MLModel(proto) is not None

    def test_negative_input_shapes(self):
        # Respect negative value as default and also set it to flexible
        if HAS_TF2:
            m = tf.keras.Sequential()
            m.add(tf.keras.layers.ReLU(input_shape=(4, 5), batch_size=3))
            m.save(self.model_path_h5)
            model = self.model_path_h5
            input_name, output_name = get_keras_io_names(m)
        elif HAS_TF1:
            with tf.Graph().as_default() as model:
                x = tf.placeholder(tf.float32, shape=(None, 4, 5))
                output = tf.nn.relu(x)
            input_name, output_name = x.op.name, output.op.name
        proto = converter.convert(
            model, inputs={input_name: (-3, 4, 5)}, outputs=[output_name])
        assert MLModel(proto) is not None

    def test_invalid_input_names(self):
        if HAS_TF2:
            m = tf.keras.Sequential()
            m.add(tf.keras.layers.ReLU(input_shape=(4, 5), batch_size=3))
            m.save(self.model_path_h5)
            model = self.model_path_h5
        elif HAS_TF1:
            with tf.Graph().as_default() as model:
                tf.nn.relu(tf.placeholder(tf.float32, shape=(3, 4, 5)))
        with pytest.raises(KeyError) as e:
            converter.convert(model, inputs={'invalid_name': (3, 4, 5)})
        e.match(r'Input node name .* does not exist')

    def test_invalid_output_names(self):
        if HAS_TF2:
            m = tf.keras.Sequential()
            m.add(tf.keras.layers.ReLU(input_shape=(4, 5), batch_size=3))
            m.save(self.model_path_h5)
            model = self.model_path_h5
        elif HAS_TF1:
            with tf.Graph().as_default() as model:
                tf.nn.relu(tf.placeholder(tf.float32, shape=(3, 4, 5)))
        with pytest.raises(KeyError) as e:
            converter.convert(model, outputs=['invalid_name'])
        e.match(r'Output node name .* does exist')


class TestModelFormats:

    def setup(self):
        self.saved_model_dir = tempfile.mkdtemp()
        _, self.model_path_h5 = tempfile.mkstemp(
            suffix='.h5', prefix=self.saved_model_dir)
        _, self.model_path_pb = tempfile.mkstemp(
            suffix='.pb', prefix=self.saved_model_dir)

    def teardown(self):
        if os.path.exists(self.saved_model_dir):
            shutil.rmtree(self.saved_model_dir)

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    def test_graph_def(self):
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=(3, 4, 5))
            out = tf.nn.relu(x)
        proto = converter.convert(
            graph, inputs={x.op.name: (3, 4, 5)}, outputs=[out.op.name])
        assert MLModel(proto) is not None

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    def test_graph_def_file(self):
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=(3, 4, 5))
            out = tf.nn.relu(x)
        tf.io.write_graph(
            graph, self.saved_model_dir,
            self.model_path_pb, as_text=False)
        proto = converter.convert(
            self.model_path_pb,
            inputs={x.op.name: (3, 4, 5)},
            outputs=[out.op.name])
        assert MLModel(proto) is not None

    @pytest.mark.skipif(not HAS_TF2, reason=MSG_TF2_NOT_FOUND)
    def test_keras_saved_model_file(self):
        keras_model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28), batch_size=1),
            tf.keras.layers.Dense(10, activation=tf.nn.relu),
        ])
        input_name, output_name = get_keras_io_names(keras_model)
        keras_model.save(self.saved_model_dir, save_format='tf')
        proto = converter.convert(self.saved_model_dir, outputs='Identity')
        assert MLModel(proto) is not None
        # m = MLModel(proto)
        # p = m.predict({input_name: np.random.rand(1, 28, 28)})
        # print(p)

    @pytest.mark.skipif(not HAS_TF2, reason=MSG_TF2_NOT_FOUND)
    def test_keras_h5_file(self):
        keras_model = tf.keras.Sequential([
            tf.keras.layers.ReLU(input_shape=(4, 5), batch_size=3)
        ])
        input_name, output_name = get_keras_io_names(keras_model)
        keras_model.save(self.model_path_h5, save_format='h5')
        proto = converter.convert(
            self.model_path_h5,
            inputs={input_name: (3, 4, 5)},
            outputs=['Identity'])
        assert MLModel(proto) is not None
        # m = MLModel(proto)
        # p = m.predict({input_name: np.random.rand(3, 4, 5)})
        # print(p)

    @pytest.mark.skipif(not HAS_TF2, reason=MSG_TF2_NOT_FOUND)
    def test_concrete_function_list_from_tf_low_level_api(self):
        root = tf.train.Checkpoint()
        root.v1 = tf.Variable(3.)
        root.v2 = tf.Variable(2.)
        root.f = tf.function(lambda x: root.v1 * root.v2 * x)

        input_data = tf.constant(1., shape=[1, 1])
        to_save = root.f.get_concrete_function(input_data)
        tf.saved_model.save(root, self.saved_model_dir, to_save)

        tf_model = tf.saved_model.load(self.saved_model_dir)
        concrete_func = tf_model.signatures[
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        proto = converter.convert([concrete_func], outputs='Identity')
        assert MLModel(proto) is not None

    @pytest.mark.skipif(not HAS_TF2, reason=MSG_TF2_NOT_FOUND)
    def test_saved_model_list_from_tf_function(self):
        class build_model(tf.Module):
            @tf.function(input_signature=[
                tf.TensorSpec(shape=[3, 4, 5], dtype=tf.float32)])
            def __call__(self, x):
                return tf.nn.relu(x)

        model = build_model()
        tf.saved_model.save(model, self.saved_model_dir)
        proto = converter.convert(
            self.saved_model_dir, outputs=['Identity'])
        assert MLModel(proto) is not None

    @pytest.mark.skipif(not HAS_TF2, reason=MSG_TF2_NOT_FOUND)
    def test_concrete_function_list_from_tf_function(self):
        class build_model(tf.Module):
            @tf.function(input_signature=[
                tf.TensorSpec(shape=[3, 4, 5], dtype=tf.float32)])
            def __call__(self, x):
                return tf.nn.relu(x)

        model = build_model()
        concrete_func = model.__call__.get_concrete_function()
        proto = converter.convert(
            [concrete_func], outputs=['Identity'])
        assert MLModel(proto) is not None

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    def test_invalid_format_tf1_none(self):
        with pytest.raises(NotImplementedError) as e:
            converter.convert(None)
        e.match(r'Expected model format: .* .pb')

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    def test_invalid_format_tf1_h5(self):
        # we do not support tf.keras model defined in TF 1.x
        with pytest.raises(NotImplementedError) as e:
            converter.convert(self.model_path_h5)
        e.match(r'Expected model format: .* .pb')

    @pytest.mark.skipif(not HAS_TF1, reason=MSG_TF1_NOT_FOUND)
    def test_invalid_format_tf1_invalid_extension(self):
        _, invalid_filename = tempfile.mkstemp(
            suffix='.invalid', prefix=self.saved_model_dir)
        with pytest.raises(NotImplementedError) as e:
            converter.convert(invalid_filename)
        e.match(r'Expected model format: .* .pb')

    @pytest.mark.skipif(not HAS_TF2, reason=MSG_TF2_NOT_FOUND)
    def test_invalid_format_tf2_none(self):
        with pytest.raises(NotImplementedError) as e:
            converter.convert(None)
        e.match(r'Expected model format: .* .h5')

    @pytest.mark.skipif(not HAS_TF2, reason=MSG_TF2_NOT_FOUND)
    def test_invalid_format_tf2_invalid_extension(self):
        _, invalid_filename = tempfile.mkstemp(
            suffix='.invalid', prefix=self.saved_model_dir)
        with pytest.raises(NotImplementedError) as e:
            converter.convert(invalid_filename)
        e.match(r'Expected model format: .* .h5')

    def test_invalid_converter_type(self):
        with pytest.raises(NotImplementedError) as e:
            converter.convert(None, convert_from='invalid')
        e.match(r'Frontend converter .* not implemented')

        with pytest.raises(NotImplementedError) as e:
            converter.convert(None, convert_to='invalid')
        e.match(r'Backend converter .* not implemented')

    def test_invalid_format_non_exist(self):
        non_exist_filename = self.model_path_pb.replace('.pb', '_non_exist.pb')
        with pytest.raises(ValueError) as e:
            converter.convert(non_exist_filename)
        e.match(r'Input model .* does not exist')

    @pytest.mark.skipif(not HAS_TF2, reason=MSG_TF2_NOT_FOUND)
    def test_invalid_format_multiple_concrete_functions(self):
        class build_model(tf.Module):
            @tf.function(input_signature=[
                tf.TensorSpec(shape=[3, 4, 5], dtype=tf.float32)])
            def __call__(self, x):
                return tf.nn.relu(x)

        model = build_model()
        cf = model.__call__.get_concrete_function()
        with pytest.raises(NotImplementedError) as e:
            converter.convert([cf, cf, cf])
        e.match(r'Only a single concrete function is supported')
