#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os
import shutil
import tempfile

import numpy as np
import pytest

import coremltools as ct
import coremltools.converters as converter
import coremltools.proto.FeatureTypes_pb2 as ft
from coremltools import EnumeratedShapes, ImageType, RangeDim, TensorType
from coremltools._deps import _HAS_TF_1, _IS_MACOS, MSG_TF1_NOT_FOUND
from coremltools.converters.mil.frontend.tensorflow.converter import TFConverter
from coremltools.converters.mil.frontend.tensorflow.test.testing_utils import (
    TensorFlowBaseTest, get_tf_keras_io_names, make_tf_graph)
from coremltools.converters.mil.testing_reqs import backends
from coremltools.converters.mil.testing_utils import random_gen

tf = pytest.importorskip("tensorflow")
frontend = "tensorflow"

class TestTfModelInputsOutputs(TensorFlowBaseTest):
    def setup(self):
        self.saved_model_dir = tempfile.mkdtemp()
        _, self.model_path_h5 = tempfile.mkstemp(
            suffix=".h5", prefix=self.saved_model_dir
        )
        _, self.model_path_pb = tempfile.mkstemp(
            suffix=".pb", prefix=self.saved_model_dir
        )

    def teardown(self):
        if os.path.exists(self.saved_model_dir):
            shutil.rmtree(self.saved_model_dir)

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_infer_inputs(self, backend):
        x_shape = (3, 4, 5)

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.nn.relu(x)

        model, inputs, outputs = build_model
        if not isinstance(outputs, (tuple, list)):
            outputs = [outputs]

        output_names = [j if isinstance(j, str) else j.op.name for j in outputs]
        mlmodel = converter.convert(model, outputs=output_names, convert_to=backend[0])
        assert mlmodel is not None

        input_values = [random_gen(x_shape, -10.0, 10.0)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(model, input_dict, outputs)

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_infer_outputs(self, backend):
        x_shape = (3, 4, 5)

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.nn.relu(x)

        model, inputs, outputs = build_model
        input_name = inputs[0] if isinstance(inputs[0], str) else inputs[0].op.name
        mlmodel = converter.convert(
            model, inputs=[TensorType(input_name, (3, 4, 5))], convert_to=backend[0]
        )
        assert mlmodel is not None

        input_values = [random_gen(x_shape, -10.0, 10.0)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(model, input_dict, outputs)

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_infer_inputs_and_outputs(self, backend):
        x_shape = (3, 4, 5)

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.nn.relu(x)

        model, inputs, outputs = build_model
        mlmodel = converter.convert(model, convert_to=backend[0])
        assert mlmodel is not None

        input_values = [random_gen(x_shape, -10.0, 10.0)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(model, input_dict, outputs)

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_extract_sub_model(self, backend):
        x_shape = (3, 4, 5)
        y_shape = (3, 4, 5)

        @make_tf_graph([x_shape, y_shape])
        def build_model(x, y):
            return tf.nn.relu(x), tf.math.add(x, y)

        model, inputs, outputs = build_model
        if isinstance(outputs[0], str):
            first_output_name = outputs[0]
        else:
            first_output_name = outputs[0].name.split(":")[0]
        mlmodel = converter.convert(model, outputs=[first_output_name], convert_to=backend[0])
        assert mlmodel is not None

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_auto_image_nhwc_input_names(self, backend):
        x_shape = (4, 5, 3) if backend[0] == "neuralnetwork" else (1, 4, 5, 3)

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.nn.relu(x)

        model, inputs, outputs = build_model

        mlmodel = converter.convert(model, inputs=[ImageType()], convert_to=backend[0])
        assert mlmodel is not None

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_auto_image_nchw_input_names(self, backend):
        x_shape = (3, 4, 5) if backend[0] == "neuralnetwork" else (1, 3, 4, 5)

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.nn.relu(x)

        model, inputs, outputs = build_model

        mlmodel = converter.convert(
            model, inputs=[ImageType(channel_first=True)], convert_to=backend[0]
        )
        assert mlmodel is not None

    @pytest.mark.parametrize(
        "target",
        [ct.target.iOS13, ct.target.macOS10_15, ct.target.watchOS6, ct.target.tvOS13],
    )
    def test_invalid_deployment_target_cumsum(self, target):
        x_shape = (3, 4, 5)

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.math.cumsum(x, axis=-1, reverse=False, exclusive=False)

        model, inputs, outputs = build_model

        with pytest.raises(ValueError) as e:
            converter.convert(model, minimum_deployment_target=target)
            e.match(
                r"Provided minimum deployment target requires model to be of version 4 but converted model "
                r"uses following features which are available from version 5 onwards. "
                r"Please use a higher minimum deployment target to convert. \n    1. Cumsum operation\n"
            )

    @pytest.mark.parametrize(
        "target",
        [ct.target.iOS14, ct.target.macOS10_16, ct.target.watchOS7, ct.target.tvOS14],
    )
    def test_valid_deployment_target_cumsum(self, target):
        x_shape = (3, 4, 5)

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.math.cumsum(x, axis=-1, reverse=False, exclusive=False)

        model, inputs, outputs = build_model

        # successful conversion
        converter.convert(model, minimum_deployment_target=target)

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_invalid_output_names(self, backend):
        x_shape = (3, 4, 5)

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.nn.relu(x)

        model, inputs, outputs = build_model
        with pytest.raises(AssertionError) as e:
            converter.convert(
                model, source=frontend, outputs=["invalid_name"], convert_to=backend[0]
            )
        e.match(r".* is not in graph")

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_missing_placeholder_shape(self, backend):
        x_shape = None  # Missing Placeholder shape

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.nn.relu(x)

        model, inputs, outputs = build_model
        with pytest.raises(ValueError) as e:
            converter.convert(model, source=frontend, convert_to=backend[0])
            e.match(r"Unable to determine the shape of input .*")

        mlmodel = converter.convert(
            model, source=frontend, inputs=[ct.TensorType(shape=(1,))], convert_to=backend[0]
        )
        assert mlmodel is not None

    @pytest.mark.skip(reason="Rank-0 input is not supported")
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_scalar_placeholder_shape(self, backend):
        x_shape = ()  # Scalar Placeholder Shape

        @make_tf_graph([x_shape])
        def build_model(x):
            return tf.nn.relu(x)

        model, inputs, outputs = build_model
        mlmodel = converter.convert(model, source=frontend, convert_to=backend[0])
        assert mlmodel is not None

        input_values = [random_gen(x_shape, -10.0, 10.0)]
        input_dict = dict(zip(inputs, input_values))
        TensorFlowBaseTest.run_compare_tf(model, input_dict, outputs)

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_shaping_utils(self, backend):
        @make_tf_graph([(None, 4, 5)])
        def build_flexible_model(x):
            return tf.nn.relu(x)

        model, inputs, outputs = build_flexible_model
        input_name = TFConverter._get_tensor_name(inputs[0])
        output_name = TFConverter._get_tensor_name(outputs[0])

        # static-Flexible shape
        if backend[0] == "neuralnetwork":
            inputs = [
                # Use TF's input shapes (None, 4, 5)
                TensorType(name=input_name)
            ]
        else:
            inputs = [TensorType(name=input_name, shape=(RangeDim(upper_bound=3), 4, 5))]

        mlmodel = converter.convert(
            model, inputs=inputs, outputs=[output_name], convert_to=backend[0]
        )
        assert mlmodel is not None
        input_values = [random_gen((3, 4, 5), -10.0, 10.0)]
        input_dict = {input_name: input_values[0]}
        if _IS_MACOS:
            ret = mlmodel.predict(input_dict)
            np.allclose(ret[output_name], np.maximum(input_values[0], 0.0))

        # Enumerate shape
        inputs_shape = [TensorType(input_name, EnumeratedShapes(shapes=[(3, 4, 5), (4, 4, 5)]))]
        mlmodel = converter.convert(
            model, inputs=inputs_shape, outputs=[output_name], convert_to=backend[0]
        )
        assert mlmodel is not None
        input_values = [random_gen((3, 4, 5), -10.0, 10.0)]
        input_dict = {input_name: input_values[0]}
        if _IS_MACOS:
            ret = mlmodel.predict(input_dict)
            np.allclose(ret[output_name], np.maximum(input_values[0], 0.0))

        input_values = [random_gen((4, 4, 5), -10.0, 10.0)]
        input_dict = {input_name: input_values[0]}
        if _IS_MACOS:
            ret = mlmodel.predict(input_dict)
            np.allclose(ret[output_name], np.maximum(input_values[0], 0.0))

        if _IS_MACOS:
            with pytest.raises(RuntimeError):
                input_values = [random_gen((5, 4, 5), -10.0, 10.0)]
                input_dict = {input_name: input_values[0]}
                ret = mlmodel.predict(input_dict)

        # Ranged shape
        inputs_shape = [TensorType(input_name, [RangeDim(3, 5), 4, 5])]
        mlmodel = converter.convert(
            model, inputs=inputs_shape, outputs=[output_name], convert_to=backend[0]
        )
        assert mlmodel is not None
        input_values = [random_gen((3, 4, 5), -10.0, 10.0)]
        input_dict = {input_name: input_values[0]}
        if _IS_MACOS:
            ret = mlmodel.predict(input_dict)
            np.allclose(ret[output_name], np.maximum(input_values[0], 0.0))

        input_values = [random_gen((4, 4, 5), -10.0, 10.0)]
        input_dict = {input_name: input_values[0]}
        if _IS_MACOS:
            ret = mlmodel.predict(input_dict)
            np.allclose(ret[output_name], np.maximum(input_values[0], 0.0))

        if _IS_MACOS:
            with pytest.raises(RuntimeError):
                input_values = [random_gen((2, 4, 5), -10.0, 10.0)]
                input_dict = {input_name: input_values[0]}
                ret = mlmodel.predict(input_dict)

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_default_data_types(self, backend):
        @make_tf_graph([(2, 2)])
        def build_model(x):
            return tf.nn.relu(x)

        model, inputs, outputs = build_model
        mlmodel = converter.convert(model, convert_to=backend[0])
        assert mlmodel is not None
        spec = mlmodel.get_spec()

        # Defaults should be FLOAT32 instead of DOUBLE
        it = spec.description.input[0].type.multiArrayType.dataType
        assert it == ft.ArrayFeatureType.ArrayDataType.Value("FLOAT32")
        ot = spec.description.output[0].type.multiArrayType.dataType
        assert ot == ft.ArrayFeatureType.ArrayDataType.Value("FLOAT32")


@pytest.mark.skipif(not _HAS_TF_1, reason=MSG_TF1_NOT_FOUND)
class TestTf1ModelFormats:
    def setup(self):
        self.saved_model_dir = tempfile.mkdtemp()
        _, self.model_path_h5 = tempfile.mkstemp(
            suffix=".h5", prefix=self.saved_model_dir
        )
        _, self.model_path_pb = tempfile.mkstemp(
            suffix=".pb", prefix=self.saved_model_dir
        )

    def teardown(self):
        if os.path.exists(self.saved_model_dir):
            shutil.rmtree(self.saved_model_dir)

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_graph_def(self, backend):
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=(3, 4, 5))
            out = tf.nn.relu(x)
            mlmodel = converter.convert(
                graph,
                inputs=[TensorType(x.op.name, (3, 4, 5))],
                outputs=[out.op.name],
                convert_to=backend[0],
            )
            assert mlmodel is not None

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_graph_def_file(self, backend):
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=(3, 4, 5))
            out = tf.nn.relu(x)
            tf.io.write_graph(
                graph, self.saved_model_dir, self.model_path_pb, as_text=False
            )
        mlmodel = converter.convert(
            self.model_path_pb,
            inputs=[TensorType(x.op.name, (3, 4, 5))],
            outputs=[out.op.name],
            convert_to=backend[0],
        )
        assert mlmodel is not None

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_saved_model_from_simple_save(self, backend):
        with tf.compat.v1.Session() as sess:
            x = tf.placeholder(shape=(1, 3, 5), dtype=tf.float32)
            y = tf.nn.relu(x)
            inputs = {"x": x}
            outputs = {"y": y}
            tf.compat.v1.saved_model.simple_save(sess, self.saved_model_dir, inputs, outputs)
        mlmodel = converter.convert(self.saved_model_dir, convert_to=backend[0])
        assert mlmodel is not None

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_tf_keras(self, backend):
        keras_model = tf.keras.Sequential([tf.keras.layers.ReLU(input_shape=(4, 5), batch_size=3)])
        input_names, output_names = get_tf_keras_io_names(keras_model)
        mlmodel = converter.convert(
            keras_model,
            inputs=[TensorType(input_names[0], (3, 4, 5))],
            outputs=["Identity"],
            source=frontend,
            convert_to=backend[0],
        )
        assert mlmodel is not None

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_tf_keras_hdf5_file(self, backend):
        keras_model = tf.keras.Sequential([tf.keras.layers.ReLU(input_shape=(4, 5), batch_size=3)])
        keras_model.save(self.model_path_h5)
        input_names, output_names = get_tf_keras_io_names(keras_model)
        mlmodel = converter.convert(
            self.model_path_h5,
            inputs=[TensorType(input_names[0], (3, 4, 5))],
            outputs=["Identity"],
            source=frontend,
            convert_to=backend[0],
        )
        assert mlmodel is not None

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_model_metadata(self, backend):
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=(3, 4, 5))
            out = tf.nn.relu(x)
            mlmodel = converter.convert(
                graph,
                inputs=[TensorType(x.op.name, (3, 4, 5))],
                outputs=[out.op.name],
                convert_to=backend[0],
            )
            metadata_keys = mlmodel.get_spec().description.metadata.userDefined
            assert "com.github.apple.coremltools.version" in metadata_keys
            assert "com.github.apple.coremltools.source" in metadata_keys
            assert "tensorflow==1." in metadata_keys["com.github.apple.coremltools.source"]

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_invalid_format_none(self, backend):
        with pytest.raises(NotImplementedError) as e:
            converter.convert(None, source="tensorflow", convert_to=backend[0])
            e.match(r"Expected model format: .* .pb")

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_invalid_format_invalid_extension(self, backend):
        _, invalid_filename = tempfile.mkstemp(suffix=".invalid", prefix=self.saved_model_dir)
        with pytest.raises(NotImplementedError) as e:
            converter.convert(invalid_filename, source="tensorflow", convert_to=backend[0])
            e.match(r"Expected model format: .* .pb")

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_invalid_converter_source(self, backend):
        with pytest.raises(ValueError) as e:
            converter.convert(None, source="invalid", convert_to=backend[0])
            expected_msg = r'Unrecognized value of argument "source": .*'
            e.match(expected_msg)

    def test_invalid_converter_minimum_deployment_flag(self):
        with pytest.raises(TypeError) as e:
            converter.convert(
                None, source="tensorflow", minimum_deployment_target="iOs14"
            )
            expected_msg = (
                "Unrecognized value of argument 'minimum_deployment_target': iOs14. "
                "It needs to be a member of 'coremltools.target' enumeration"
            )

            e.match(expected_msg)

    def test_invalid_converter_target(self):
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=(3, 4, 5))
        with pytest.raises(NotImplementedError) as e:
            converter.convert(graph, convert_to="invalid", source="tensorflow")
            e.match(r"Backend converter .* not implemented")

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_invalid_format_non_exist(self, backend):
        non_exist_filename = self.model_path_pb.replace(".pb", "_non_exist.pb")
        with pytest.raises(ValueError) as e:
            converter.convert(non_exist_filename, source="tensorflow", convert_to=backend[0])
            e.match(r"Input model .* does not exist")
