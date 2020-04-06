import six

import coremltools.converters.nnv2.converter as converter
from coremltools.converters.nnv2.testing_utils import compare_shapes, compare_backend
import tensorflow as tf

frontend = 'tensorflow'


def make_tf_graph(input_spec):
    def wrapper(ops):
        with tf.Graph().as_default() as model:
            inputs = [tf.placeholder(tf.float32, shape=s, name=n)
                      for n, s in input_spec.items()]
            outputs = ops(*inputs)
        return model, inputs, outputs

    return wrapper


def get_tf_keras_io_name(model):
    """
    Utility function to get tf.keras inputs/outputs names from a keras model.

    Parameter
    ---------
    model: tf.keras.Model
    """
    input_name = model.inputs[0].name.split(':')[0]
    output_name = model.outputs[0].name.split(':')[0].split('/')[-1]
    return input_name, output_name


def get_tf_node_names(tf_nodes, mode='inputs'):
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
        tensor_name = n if isinstance(n, six.string_types) else n.name
        if mode == 'outputs':
            names.append(tensor_name)
            continue
        name = tensor_name.split(':')[0]
        if name in names:
            # keep postfix notation for multiple inputs/outputs
            names[names.index(name)] = name + ':' + str(names.count(name) - 1)
            names.append(tensor_name)
        else:
            names.append(name)
    return names

def tf_graph_to_proto(graph, feed_dict, output_nodes, backend='nnv1_proto', add_custom_layer=False):
    """
    Parameters
    ----------
    graph: tf.Graph
        TensorFlow 1.x model in tf.Graph format.
    feed_dict: dict of (tf.placeholder, np.array)
        Dict of placeholder and value pairs representing inputs.
    output_nodes: tf.node or list[tf.node]
        List of names representing outputs.
    backend: str
        Backend to convert to.
    -----------
    Returns Proto, Input Values, Output Names
    """
    if isinstance(output_nodes, tuple):
        output_nodes = list(output_nodes)
    if not isinstance(output_nodes, list):
        output_nodes = [output_nodes]

    # Convert TF graph.
    input_names = get_tf_node_names(list(feed_dict.keys()), mode='inputs')
    output_names = get_tf_node_names(output_nodes, mode='outputs')
    input_values = {name: val for name, val in zip(input_names, feed_dict.values())}

    proto = converter.convert(graph, convert_from='tensorflow',
                              convert_to=backend,
                              inputs=input_names,
                              outputs=output_names,
                              add_custom_layer=add_custom_layer)
    return proto, input_values, output_names, output_nodes


def run_compare_tf(
        graph, feed_dict, output_nodes,
        use_cpu_only=False, frontend_only=False,
        frontend='tensorflow', backend='nnv1_proto',
        atol=1e-04, rtol=1e-05, validate_shapes_only=False):
    """
    Utility function to convert and compare a given TensorFlow 2.x model.

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
    """
    proto, input_key_values, output_names, output_nodes = tf_graph_to_proto(graph, feed_dict, output_nodes, backend)

    if frontend_only:
        return

    tf_outputs = tf.Session(graph=graph).run(output_nodes, feed_dict=feed_dict)
    expected_outputs = {name: val for name, val in zip(output_names, tf_outputs)}

    if validate_shapes_only:
        compare_shapes(proto, input_key_values, expected_outputs, use_cpu_only)
    else:
        compare_backend(proto, input_key_values, expected_outputs,
                        use_cpu_only, atol=atol, rtol=rtol)
