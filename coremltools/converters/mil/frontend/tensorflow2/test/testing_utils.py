#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.testing_reqs import ct
import pytest
import numpy as np

tf = pytest.importorskip("tensorflow", minversion="2.1.0")
from coremltools.converters.mil.frontend.tensorflow.test.testing_utils import (
    get_tf_node_names
)

from coremltools.converters.mil.input_types import TensorType, RangeDim
from coremltools.converters.mil.testing_utils import compare_shapes, compare_backend
from tensorflow.python.framework import dtypes


def make_tf2_graph(input_types):
    """
    Decorator to help construct TensorFlow 2.x model.

    Parameters
    ----------
    input_types: list of tuple or list of list
        List of input types. E.g. [(3, 224, 224, tf.int32)] represent 1 input,
        with shape (3, 224, 224), and the expected data type is tf.int32. The
        dtype is optional, in case it's missing, tf.float32 will be used.

    Returns
    -------
    list of ConcreteFunction, list of str, list of str
    """

    def wrapper(ops):
        input_signature = []
        for input_type in input_types:
            if input_type is not None and len(input_type) > 0 and isinstance(input_type[-1], dtypes.DType):
                shape, dtype = input_type[:-1], input_type[-1]
            else:
                shape, dtype = input_type, tf.float32
            input_signature.append(tf.TensorSpec(shape=shape, dtype=dtype))

        @tf.function(input_signature=input_signature)
        def tf2_model(*args):
            return ops(*args)

        concrete_func = tf2_model.get_concrete_function()
        inputs = get_tf_node_names(
            [t.name for t in concrete_func.inputs if t.dtype != dtypes.resource],
            mode="input",
        )
        outputs = get_tf_node_names(
            [t.name for t in concrete_func.outputs], mode="output"
        )
        return [concrete_func], inputs, outputs

    return wrapper


def run_compare_tf2(
    model,
    input_dict,
    output_names,
    use_cpu_only=False,
    frontend_only=False,
    frontend="tensorflow",
    backend="nn_proto",
    debug=False,
    atol=1e-04,
    rtol=1e-05,
):
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
    debug: bool
        If true, print verbose information and plot intermediate graphs.
    atol: float
        The absolute tolerance parameter.
    rtol: float
        The relative tolerance parameter.
    """
    inputs = []
    cf_inputs = [t for t in model[0].inputs if t.dtype != dtypes.resource]
    for t in cf_inputs:
        name = get_tf_node_names(t.name)[0]
        shape = [RangeDim() if s is None or s == -1 else s \
                for s in list(t.get_shape())]
        inputs.append(TensorType(name=name, shape=shape,
            dtype=t.dtype.as_numpy_dtype))
    outputs = []
    for t in output_names:
        name = get_tf_node_names(t)[0]
        outputs.append(name)

    # get TensorFlow 2.x output as reference and run comparison
    tf_input_values = [tf.constant(t) for t in input_dict.values()]
    tf_outputs = model[0](*tf_input_values)
    if isinstance(tf_outputs, (tuple, list)):
        ref = [t.numpy() for t in tf_outputs]
    else:
        ref = [tf_outputs.numpy()]
    expected_outputs = {n: v for n, v in zip(outputs, ref)}

    mlmodel = ct.convert(
        model,
        source=frontend,
        inputs=inputs,
        outputs=outputs,
        convert_to=backend,
        debug=debug,
    )

    if frontend_only:
        return

    for k,v in input_dict.items():
        if isinstance(v, np.ndarray) and issubclass(v.dtype.type, np.integer):
            input_dict[k] = v.astype(np.float) # Core ML only accepts floats

    compare_backend(
        mlmodel,
        input_dict,
        expected_outputs,
        use_cpu_only,
        atol=atol,
        rtol=rtol,
        also_compare_shapes=True,
    )

    return mlmodel.get_spec()


def run_compare_tf_keras(
    model,
    input_values,
    use_cpu_only=False,
    frontend_only=False,
    frontend="tensorflow",
    backend="nn_proto",
    atol=1e-04,
    rtol=1e-05,
):
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

    mlmodel = ct.convert(model, source=frontend, convert_to=backend)

    # assumes conversion preserve the i/o names
    proto = mlmodel.get_spec()
    inputs = sorted([str(i.name) for i in proto.description.input])
    outputs = [str(o.name) for o in proto.description.output]

    if frontend_only:
        return

    # get tf.keras model output as reference and run comparison
    keras_outputs = model(input_values)
    if not isinstance(keras_outputs, list):
        keras_outputs = [keras_outputs]
    ref = [output.numpy() for output in keras_outputs]
    expected_outputs = {n: v for n, v in zip(outputs, ref)}
    input_key_values = {n: v for n, v in zip(inputs, input_values)}
    compare_backend(
        mlmodel,
        input_key_values,
        expected_outputs,
        use_cpu_only,
        atol=atol,
        rtol=rtol,
        also_compare_shapes=True,
    )

    return proto
