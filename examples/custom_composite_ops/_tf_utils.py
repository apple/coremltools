# Temporary utility helper routines
# TODO: Should be removed and user should be able to use .convert API on graph directly
import six
import numpy as np
import tensorflow as tf
import coremltools
from tensorflow.python.framework import dtypes

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


def make_tf_graph(input_spec):
    def wrapper(ops):
        with tf.Graph().as_default() as model:
            inputs = [tf.placeholder(tf.float32, shape=s, name=n)
                      for n, s in input_spec.items()]
            outputs = ops(*inputs)
        return model, inputs, outputs

    return wrapper

def make_tf2_graph(input_spec):
    def wrapper(ops):
        class TensorFlowModule(tf.Module):
            @tf.function(input_signature=[
                tf.TensorSpec(shape=s, dtype=tf.float32, name=n)
                for n, s in input_spec.items()])
            def __call__(self, *args):
                return ops(*args)

        module = TensorFlowModule()
        concrete_func = module.__call__.get_concrete_function()
        inputs = get_tf_node_names(
            [t.name for t in concrete_func.inputs if t.dtype != dtypes.resource], mode='input')
        outputs = get_tf_node_names(
            [t.name for t in concrete_func.outputs], mode='output')
        return [concrete_func], inputs, outputs

    return wrapper

def compare_backend(
        proto, input_key_values, expected_outputs,
        use_cpu_only=False, atol=1e-04, rtol=1e-05):
    """
    Inputs:
        - proto: MLModel proto.

        - input_key_values: str -> np.array. Keys must match those in
          input_placeholders.

        - expected_outputs: dict[str, np.array]. Required iff
          frontend_only == False

        - use_cpu_only: True/False.
    """
    model = coremltools.models.MLModel(proto, useCPUOnly=use_cpu_only)
    input_key_values = dict(
        [(k, v.astype(np.float32) if not np.isscalar(v) and not v.shape == () else np.array([v], dtype=np.float32)) for k, v in
         input_key_values.items()])
    pred = model.predict(input_key_values, useCPUOnly=use_cpu_only)
    if not use_cpu_only:
        atol = min(atol * 100., 1e-1)
        rtol = min(rtol * 100., 1e-2)
    for o, expected in expected_outputs.items():
        msg = 'Output {} differs. useCPUOnly={}.\nInput={}, ' + \
              'Expected={}, Output={}\n'
        np.testing.assert_almost_equal(pred[o], expected, decimal=3,
              err_msg=msg.format(o, use_cpu_only, input_key_values, expected, pred[o]))
    print('Output match!')

def convert_tf1(graph, feed_dict, output_nodes):
    if isinstance(output_nodes, tuple):
        output_nodes = list(output_nodes)
    if not isinstance(output_nodes, list):
        output_nodes = [output_nodes]
        
    input_names = get_tf_node_names(list(feed_dict.keys()), mode='inputs')
    output_names = get_tf_node_names(output_nodes, mode='outputs')
    input_values = {name: val for name, val in zip(input_names, feed_dict.values())}

    spec = coremltools.converter.convert(graph,
                                         inputs=input_names,
                                         outputs=output_names)
    return spec

def convert_tf2(model, output_names):
    inputs = {}
    cf_inputs = [t for t in model[0].inputs if t.dtype != dtypes.resource]
    for t in cf_inputs:
        name = get_tf_node_names(t.name)[0]
        inputs[name] = list(t.get_shape())
    outputs = []
    for t in output_names:
        name = get_tf_node_names(t)[0]
        outputs.append(name)

    proto = coremltools.converter.convert(model, inputs=inputs, outputs=outputs)
    return proto
    
def compare_results(proto, tf_model, input_dict, output_names, atol=1e-04, rtol=1e-05):
    """
    Parameters
    ----------
    proto: MLModel spec
        NNv1 Spec.
    model: list of tf.ConcreteFunction
        List of TensorFlow 2.x concrete functions.
    input_dict: dict of (str, np.array)
        Dict of name and value pairs representing inputs.
    output_names: list of str
        List of output node names.
    atol: float
        The absolute tolerance parameter.
    rtol: float
        The relative tolerance parameter.
    """
    # get TensorFlow 2.x output as reference and run comparision
    tf_input_values = [tf.constant(t) for t in input_dict.values()]
    ref = [tf_model[0](*tf_input_values).numpy()]
    outputs = []
    for t in output_names:
        name = get_tf_node_names(t)[0]
        outputs.append(name)
    expected_outputs = {n: v for n, v in zip(outputs, ref)}
    compare_backend(
           proto, input_dict, expected_outputs, atol=atol, rtol=rtol)
