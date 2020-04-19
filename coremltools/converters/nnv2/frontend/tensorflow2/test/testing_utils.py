import coremltools.converters.nnv2.converter as converter
import coremltools.converters.nnv2.frontend.tensorflow.test.testing_utils as tf_testing_utils
import tensorflow as tf
from coremltools.converters.nnv2.frontend.tensorflow.test.testing_utils import get_tf_node_names
from coremltools.converters.nnv2.testing_utils import compare_shapes, compare_backend
from tensorflow.python.framework import dtypes


def make_tf2_graph(input_types):
    """
    Decorator to help construct TensorFlow 2.x model.

    Parameters
    ----------
    input_types: list of tuple
        List of input types. E.g. [(3, 224, 224, tf.int32)] represent 1 input,
        with shape (3, 224, 224), and the expected data type is tf.int32. The
        dtype is optional, in case it's missing, tf.float32 will be used.

    Returns
    -------
    list of ConcreteFunction, list of str, list of str
    """
    def wrapper(ops):
        class TensorFlowModule(tf.Module):
            input_signature = []
            for input_type in input_types:
                if isinstance(input_type[-1], dtypes.DType):
                    shape, dtype = input_type[:-1], input_type[-1]
                else:
                    shape, dtype = input_type, tf.float32
                input_signature.append(tf.TensorSpec(shape=shape, dtype=dtype))

            @tf.function(input_signature=input_signature)
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


def run_compare_tf2(
        model, input_dict, output_names,
        use_cpu_only=False, frontend_only=False,
        frontend='tensorflow2', backend='nnv1_proto', atol=1e-04, rtol=1e-05):
    """
    Parameters
    ----------
    model: list of tf.ConcreteFunction
        List of TensorFlow 2.x concrete functions.
    input_dict: dict of (str, np.array)
        Dict of name and value pairs representing inputs.
    output_names: list of str
        List of output node names.
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
    """
    inputs = {}
    cf_inputs = [t for t in model[0].inputs if t.dtype != dtypes.resource]
    for t in cf_inputs:
        name = get_tf_node_names(t.name)[0]
        inputs[name] = list(t.get_shape())
    outputs = []
    for t in output_names:
        name = get_tf_node_names(t)[0]
        outputs.append(name)

    proto = converter.convert(
        model, convert_from=frontend, convert_to=backend,
        inputs=inputs, outputs=outputs)

    if frontend_only:
        return

    # get TensorFlow 2.x output as reference and run comparision
    tf_input_values = [tf.constant(t) for t in input_dict.values()]
    ref = [model[0](*tf_input_values).numpy()]
    expected_outputs = {n: v for n, v in zip(outputs, ref)}

    compare_shapes(
        proto, input_dict, expected_outputs, use_cpu_only)
    compare_backend(
        proto, input_dict, expected_outputs, use_cpu_only, atol=atol, rtol=rtol)


def run_compare_tf_keras(
        model, input_values, use_cpu_only=False, frontend_only=False,
        frontend='tensorflow2', backend='nnv1_proto', atol=1e-04, rtol=1e-05):
    """
    Parameters
    ----------
    model: TensorFlow 2.x model
        TensorFlow 2.x model annotated with @tf.function.
    input_values: list of np.array
        List of input values in the same order as the input signature.
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
    """
    # construct model and convert
    tf_keras_model = tf.function(lambda x: model(x))

    input_tensor_spec = []
    for i in range(len(model.inputs)):
        input_tensor_spec.append(
            tf.TensorSpec(model.inputs[i].shape, model.inputs[i].dtype))

    cf = tf_keras_model.get_concrete_function(*input_tensor_spec)

    inputs = {}
    cf_inputs = [t for t in cf.inputs if t.dtype != dtypes.resource]
    for t in cf_inputs:
        name = get_tf_node_names(t.name)[0]
        inputs[name] = list(t.get_shape())
    outputs = []
    for t in cf.outputs:
        name = get_tf_node_names(t.name)[0]
        outputs.append(name)

    proto = converter.convert(
        [cf], convert_from=frontend, convert_to=backend,
        inputs=inputs, outputs=outputs)

    if frontend_only:
        return

    # get tf.keras model output as reference and run comparision
    ref = [model(*input_values).numpy()]
    expected_outputs = {n: v for n, v in zip(outputs, ref)}
    input_key_values = {n: v for n, v in zip(inputs.keys(), input_values)}
    compare_shapes(
        proto, input_key_values, expected_outputs, use_cpu_only)
    compare_backend(
        proto, input_key_values, expected_outputs,
        use_cpu_only, atol=atol, rtol=rtol)
