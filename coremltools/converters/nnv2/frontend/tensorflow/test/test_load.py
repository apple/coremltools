import os
import pytest
import shutil
import tempfile
import coremltools.converters as converter
from coremltools.converters.nnv2.testing_utils import random_gen
from coremltools.converters.nnv2.frontend.tensorflow.test.testing_utils import (
    frontend, make_tf_graph, run_compare_tf
)
from coremltools.models import MLModel

tf = pytest.importorskip('tensorflow')


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
        x_shape = (3, 4, 5)

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.nn.relu(x)

        model, inputs, outputs = build_model
        mlmodel = converter.convert(
            model, outputs=outputs)
        assert mlmodel is not None

        input_values = [random_gen(x_shape, -10., 10.)]
        input_dict = dict(zip(inputs, input_values))
        run_compare_tf(model, input_dict, outputs)

    def test_infer_outputs(self):
        x_shape = (3, 4, 5)

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.nn.relu(x)

        model, inputs, outputs = build_model
        input_name = inputs[0] if isinstance(inputs[0], str) else inputs[0].op.name
        mlmodel = converter.convert(
            model, inputs=[(input_name, (3, 4, 5))])
        assert mlmodel is not None

        input_values = [random_gen(x_shape, -10., 10.)]
        input_dict = dict(zip(inputs, input_values))
        run_compare_tf(model, input_dict, outputs)

    def test_infer_inputs_and_outputs(self):
        x_shape = (3, 4, 5)

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.nn.relu(x)

        model, inputs, outputs = build_model
        mlmodel = converter.convert(
            model)
        assert mlmodel is not None

        input_values = [random_gen(x_shape, -10., 10.)]
        input_dict = dict(zip(inputs, input_values))
        run_compare_tf(model, input_dict, outputs)

    def test_invalid_input_names(self):
        x_shape = (3, 4, 5)

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.nn.relu(x)

        model, inputs, outputs = build_model

        with pytest.raises(KeyError) as e:
            converter.convert(
                model, inputs=[('invalid_name', x_shape)])
        e.match(r'TF var .* not found in context')

    def test_invalid_output_names(self):
        x_shape = (3, 4, 5)

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.nn.relu(x)

        model, inputs, outputs = build_model

        with pytest.raises(KeyError) as e:
            converter.convert(
                model, outputs=['invalid_name'])
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

    def test_graph_def(self):
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=(3, 4, 5))
            out = tf.nn.relu(x)
        mlmodel = converter.convert(
            graph, inputs=[(x.op.name, (3, 4, 5))], outputs=[out.op.name])
        assert mlmodel is not None

    def test_graph_def_file(self):
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=(3, 4, 5))
            out = tf.nn.relu(x)
        tf.io.write_graph(
            graph, self.saved_model_dir,
            self.model_path_pb, as_text=False)
        mlmodel = converter.convert(
            self.model_path_pb,
            inputs=[(x.op.name, (3, 4, 5))],
            outputs=[out.op.name])
        assert mlmodel is not None

    def test_invalid_format_none(self):
        with pytest.raises(NotImplementedError) as e:
            converter.convert(None, source='tensorflow')
        e.match(r'Expected model format: .* .pb')

    def test_invalid_format_h5(self):
        # we do not support tf.keras model defined in TF 1.x
        with pytest.raises(NotImplementedError) as e:
            converter.convert(self.model_path_h5, source='tensorflow')
        e.match(r'Expected model format: .* .pb')

    def test_invalid_format_invalid_extension(self):
        _, invalid_filename = tempfile.mkstemp(
            suffix='.invalid', prefix=self.saved_model_dir)
        with pytest.raises(NotImplementedError) as e:
            converter.convert(invalid_filename, source='tensorflow')
        e.match(r'Expected model format: .* .pb')

    def test_invalid_converter_source(self):
        with pytest.raises(ValueError) as e:
            converter.convert(None, source='invalid')
        expected_msg = "Unrecognized value of argument \"source\": invalid. " \
                        "It must be one of \"auto\", \"tensorflow\", \"pytorch\"."
        e.match(expected_msg)

    def test_invalid_converter_target(self):
        with pytest.raises(NotImplementedError) as e:
            converter.convert(None, convert_to='invalid', source='tensorflow')
        e.match(r'Backend converter .* not implemented')

    def test_invalid_format_non_exist(self):
        non_exist_filename = self.model_path_pb.replace('.pb', '_non_exist.pb')
        with pytest.raises(ValueError) as e:
            converter.convert(non_exist_filename, source='tensorflow')
        e.match(r'Input model .* does not exist')
