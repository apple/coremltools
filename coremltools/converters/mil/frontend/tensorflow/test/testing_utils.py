#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import numpy as np
import os
import pytest
import tempfile

from coremltools import TensorType
import coremltools.models.utils as coremltoolsutils
from coremltools.converters.mil.testing_utils import compare_shapes, \
    compare_backend, run_core_ml_predict, ct_convert
from coremltools.converters.mil.testing_reqs import ct

tf = pytest.importorskip("tensorflow", minversion="1.14.0")

from tensorflow.python.framework import dtypes
from tensorflow.python.tools.freeze_graph import freeze_graph as freeze_g
from tensorflow.python.keras.saving import saving_utils as _saving_utils

frontend = "tensorflow"


def make_tf_graph(input_types):
    """
    Decorator to help construct TensorFlow 1.x model.

    Parameters
    ----------
    input_types: list of tuple or list of list
        List of input types. E.g. [(3, 224, 224, tf.int32)] represent 1 input,
        with shape (3, 224, 224), and the expected data type is tf.int32. The
        dtype is optional, in case it's missing, tf.float32 will be used.

    Returns
    -------
    tf.Graph, list of str, list of str
    """

    def wrapper(ops):
        with tf.Graph().as_default() as model:
            inputs = []
            for input_type in input_types:
                input_type = tuple(input_type) if input_type is not None else None
                if input_type is not None and len(input_type) > 0 and isinstance(input_type[-1], dtypes.DType):
                    shape, dtype = input_type[:-1], input_type[-1]
                else:
                    shape, dtype = input_type, tf.float32
                inputs.append(tf.placeholder(shape=shape, dtype=dtype))

            outputs = ops(*inputs)
        return model, inputs, outputs

    return wrapper


def get_tf_keras_io_names(model):
    """
    Utility function to get tf.keras inputs/outputs names from a tf.keras model.

    Parameter
    ---------
    model: tf.keras.Model
    """
    input_names, output_names = [], []
    try:
        # The order of outputs in conc_func.structured_outputs is the same order
        # that Keras predicts in, which can be different from model.outputs
        input_signature = _saving_utils.model_input_signature(
            model, keep_original_batch_size=True
        )
        fn = _saving_utils.trace_model_call(model, input_signature)
        conc_func = fn.get_concrete_function()
        for key in conc_func.structured_outputs:
            output_names.append(conc_func.structured_outputs[key].name.split(":")[0])
    except:
        for o in model.outputs:
            output_names.append(o.name.split(":")[0].split("/")[-1])
    for name in model.input_names:
        input_names.append(name.split(":")[0])
    return input_names, output_names


def get_tf_node_names(tf_nodes, mode="inputs"):
    """
    Inputs:
        - tf_nodes: list[str]. Names of target placeholders or output variable.
        - mode: str. When mode == inputs, do the stripe for the input names, for
                instance 'placeholder:0' could become 'placeholder'.
                when model == 'outputs', we keep the origin suffix number, like
                'bn:0' will still be 'bn:0'.
    Return a list of names from given list of TensorFlow nodes. Tensor name's
    postfix is eliminated if there's no ambiguity. Otherwise, postfix is kept
    """
    if not isinstance(tf_nodes, list):
        tf_nodes = [tf_nodes]
    names = list()
    for n in tf_nodes:
        tensor_name = n if isinstance(n, str) else n.name
        if mode == "outputs":
            names.append(tensor_name)
            continue
        name = tensor_name.split(":")[0]
        if name in names:
            # keep postfix notation for multiple inputs/outputs
            names[names.index(name)] = name + ":" + str(names.count(name) - 1)
            names.append(tensor_name)
        else:
            names.append(name)
    return names


def tf_graph_to_mlmodel(
    graph, feed_dict, output_nodes, frontend="tensorflow", backend=("neuralnetwork", "fp32"),
    use_cpu_for_conversion=False,
):
    """
    Parameters
    ----------
    graph: tf.Graph
        TensorFlow 1.x model in tf.Graph format.
    feed_dict: dict of {tf.placeholder -> np.array or python primitive)
        Dict of placeholder and value pairs representing inputs.
    output_nodes: tf.node or list[tf.node]
        List of names representing outputs.
    frontend: str
        Frontend to convert from.
    backend: str
        Backend to convert to.
    use_cpu_for_conversion: bool
        Argument which is passed as is to the unified converter API.
        That is, "ct.convert(...., useCPUOnly=use_cpu_for_conversion)"
        It forces the model to be loaded on the CPU context, post conversion.
    -----------
    Returns MLModel, Input Values, Output Names
    """
    if isinstance(output_nodes, tuple):
        output_nodes = list(output_nodes)
    if not isinstance(output_nodes, list):
        output_nodes = [output_nodes]

    # Convert TF graph.
    input_names = get_tf_node_names(list(feed_dict.keys()), mode="inputs")
    output_names = get_tf_node_names(output_nodes, mode="outputs")
    input_values = {name: val for name, val in zip(input_names, feed_dict.values())}

    mlmodel = ct_convert(
        graph, inputs=None, outputs=output_names, source=frontend, convert_to=backend,
        useCPUOnly=use_cpu_for_conversion,
    )

    return mlmodel, input_values, output_names, output_nodes


def load_tf_pb(pb_file):
    """
    Loads a pb file to tf.Graph
    """
    # We load the protobuf file from the disk and parse it to retrieve the
    # unsterilized graph_def
    with tf.io.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="")
    return graph


def run_compare_tf(
    graph,
    feed_dict,
    output_nodes,
    use_cpu_only=False,
    use_cpu_for_conversion=False,
    frontend_only=False,
    frontend="tensorflow",
    backend=("neuralnetwork", "fp32"),
    atol=1e-04,
    rtol=1e-05,
    validate_shapes_only=False,
    freeze_graph=False,
    tf_outputs=None,
):
    """
    Utility function to convert and compare a given TensorFlow 1.x model.

    Parameters
    ----------
    graph: tf.Graph
        TensorFlow 1.x model in tf.Graph format.
    feed_dict: dict of (tf.placeholder, np.array)
        Dict of placeholder and value pairs representing inputs.
    output_nodes: tf.node or list[tf.node]
        List of names representing outputs.
    use_cpu_only: bool
        If true, use CPU only for prediction, otherwise, use GPU also.
    use_cpu_for_conversion: bool
        If true, the converter is invoked using "ct.convert(...., useCPUOnly=True)",
        which in turn forces the model to be loaded with the CPU context, which happens
        when the converter loads the ML model object from the proto spec
        using "ct.models.MLModel(proto_spec, useCPUOnly=True)".
        The other argument, i.e., "use_cpu_only" on the other hand refers to only the compute engine
        for prediction purposes. For a model that is loaded on a non-CPU context, it can still be forced
        to execute on the CPU at the time of prediction. Hence,
        "use_cpu_for_conversion = False && use_cpu_only = True" is valid and results in a case when a model is
        loaded for GPU but executed on the CPU.
        The scenario, "use_cpu_for_conversion = True && use_cpu_only = False" is invalid though,
        since once a model is loaded on a CPU context its context cannot be changed to a non CPU device
        at the time of prediction.
    frontend_only: bool
        If true, skip the prediction call, only validate conversion.
    frontend: str
        Frontend to convert from.
    backend: str
        Backend to convert to.
    atol: float
        The absolute tolerance parameter.
    rtol: float
        The relative tolerance parameter.
    validate_shapes_only: bool
        If true, skip element-wise value comparision.
    tf_outputs: float or list[float]
        If present, use it as TensorFlow predictions

    Return:
        Proto, mlmodel, input dictionay, prediction(if possible)
    """
    if use_cpu_for_conversion and not use_cpu_only:
        # use_cpu_for_conversion = True && use_cpu_only = False
        raise ValueError("use_cpu_for_conversion = True && use_cpu_only = False is an invalid test case")

    mlmodel, input_key_values, output_names, output_nodes = tf_graph_to_mlmodel(
        graph, feed_dict, output_nodes, frontend, backend, use_cpu_for_conversion=use_cpu_for_conversion,
    )

    if frontend_only or coremltoolsutils._macos_version() < (10, 13) \
       or (mlmodel.is_package and coremltoolsutils._macos_version() < (12, 0)):
        return mlmodel._spec, mlmodel, input_key_values, None

    if not isinstance(output_nodes, (tuple, list)):
        output_nodes = [output_nodes]

    if freeze_graph:
        model_dir = tempfile.mkdtemp()
        graph_def_file = os.path.join(model_dir, "tf_graph.pb")
        checkpoint_file = os.path.join(model_dir, "tf_model.ckpt")
        static_model_file = os.path.join(model_dir, "tf_static.pb")
        coreml_model_file = os.path.join(model_dir, "coreml_model.mlmodel")

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            tf_outputs = sess.run(output_nodes, feed_dict=feed_dict)

            tf.train.write_graph(sess.graph, model_dir, graph_def_file, as_text=False)
            saver = tf.train.Saver()
            saver.save(sess, checkpoint_file)
            freeze_g(
                input_graph=graph_def_file,
                input_saver="",
                input_binary=True,
                input_checkpoint=checkpoint_file,
                output_node_names=",".join([n.op.name for n in output_nodes]),
                restore_op_name="save/restore_all",
                filename_tensor_name="save/Const:0",
                output_graph=static_model_file,
                clear_devices=True,
                initializer_nodes="",
            )
        graph = load_tf_pb(static_model_file)

        # Need to convert again using frozen graph
        mlmodel, input_key_values, output_names, output_nodes = tf_graph_to_mlmodel(
            graph, feed_dict, output_nodes, frontend, backend
        )
    else:
        if not tf_outputs:
            with tf.Session(graph=graph) as sess:
                sess.run(tf.global_variables_initializer())
                tf_outputs = sess.run(output_nodes, feed_dict=feed_dict)
    expected_outputs = {name: val for name, val in zip(output_names, tf_outputs)}

    for k,v in input_key_values.items():
        if isinstance(v, np.ndarray) and issubclass(v.dtype.type, np.integer):
            input_key_values[k] = v.astype(np.float) # Core ML only accepts floats

    if validate_shapes_only:
        compare_shapes(mlmodel, input_key_values, expected_outputs, use_cpu_only)
    else:
        compare_backend(
            mlmodel,
            input_key_values,
            expected_outputs,
            use_cpu_only,
            atol=atol,
            rtol=rtol,
            also_compare_shapes=True,
            dtype=backend[1],
        )
    pred=None
    if not coremltoolsutils._has_custom_layer(mlmodel.get_spec()):
        pred = run_core_ml_predict(mlmodel, input_key_values, use_cpu_only)
    else:
        print('Skipping model prediction as it has a custom nn layer!')
    return mlmodel._spec, mlmodel, input_key_values, pred


def layer_counts(spec, layer_type):
    spec_type_map = {
        "neuralNetworkClassifier": spec.neuralNetworkClassifier,
        "neuralNetwork": spec.neuralNetwork,
        "neuralNetworkRegressor": spec.neuralNetworkRegressor,
    }
    nn_spec = spec_type_map.get(spec.WhichOneof("Type"))
    if nn_spec is None:
        raise ValueError("MLModel must have a neural network")

    n = 0
    for layer in nn_spec.layers:
        if layer.WhichOneof("layer") == layer_type:
            n += 1
    return n


class TensorFlowBaseTest(object):
    testclassname=''
    testmodelname=''
    @pytest.fixture(autouse=True)
    def store_testname_with_args(self, request):
        TensorFlowBaseTest.testclassname = type(self).__name__
        TensorFlowBaseTest.testmodelname = request.node.name

    def teardown_method(self, method):
        pass

    @staticmethod
    def run_compare_tf(graph, feed_dict, output_nodes, use_cpu_only=False,
                       use_cpu_for_conversion=False,
                       frontend_only=False, frontend="tensorflow",
                       backend=("neuralnetwork", "fp32"), atol=1e-04, rtol=1e-05,
                       validate_shapes_only=False, freeze_graph=False,
                       tf_outputs=None):
        res = run_compare_tf(graph, feed_dict, output_nodes,
                        use_cpu_only=use_cpu_only,
                        use_cpu_for_conversion=use_cpu_for_conversion,
                       frontend_only=frontend_only, frontend=frontend,
                       backend=backend, atol=atol,
                       rtol=rtol,
                       validate_shapes_only=validate_shapes_only,
                       freeze_graph=freeze_graph, tf_outputs=tf_outputs)
        alist = []
        if res is not None:
            alist = list(res)
        alist.append(TensorFlowBaseTest.testclassname)
        alist.append(TensorFlowBaseTest.testmodelname)
        return tuple(alist)
