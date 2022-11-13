#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", minversion="2.1.0")
from tensorflow.python.framework import dtypes

import coremltools as ct
import coremltools.models.utils as coremltoolsutils
from coremltools.converters.mil.frontend.tensorflow.test.testing_utils import (
    TensorFlowBaseTest, get_tf_node_names)
from coremltools.converters.mil.input_types import RangeDim, TensorType
from coremltools.converters.mil.testing_utils import (compare_backend,
                                                      ct_convert)
from coremltools.models.utils import _macos_version


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
    inputs_for_conversion=None,
    compute_unit=ct.ComputeUnit.CPU_ONLY,
    frontend_only=False,
    frontend="tensorflow",
    backend=("neuralnetwork", "fp32"),
    debug=False,
    atol=1e-04,
    rtol=1e-05,
    minimum_deployment_target=None,
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
    inputs_for_conversion: list of coremltools.TensorType() or coremltools.ImageType() objects
        Defaults to None. It is passed as is to the "inputs" argument of the converter.
    compute_unit: Enum[ct.ComputeUnit]
        Compute unit for the coreml model
    frontend_only: bool
        If True, skip the prediction call, only validate conversion.
    frontend: str
        Frontend to convert from.
    backend: str
        Backend to convert to.
    debug: bool
        If True, print verbose information and plot intermediate graphs.
    atol: float
        The absolute tolerance parameter.
    rtol: float
        The relative tolerance parameter.
    minimum_deployment_target: coremltools.target enumeration
        The spec version for the mlmodel
    """
    inputs = []
    if inputs_for_conversion is None:
        cf_inputs = [t for t in model[0].inputs if t.dtype != dtypes.resource]
        for t in cf_inputs:
            name = get_tf_node_names(t.name)[0]
            shape = [RangeDim() if s is None or s == -1 else s \
                    for s in list(t.get_shape())]
            inputs.append(TensorType(name=name, shape=shape,
                                     dtype=t.dtype.as_numpy_dtype))
    else:
        inputs = inputs_for_conversion
        
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
    
    mlmodel = ct_convert(
        model,
        source=frontend,
        inputs=inputs,
        outputs=outputs,
        convert_to=backend,
        debug=debug,
        compute_units=compute_unit,
        minimum_deployment_target=minimum_deployment_target,
    )

    for k,v in input_dict.items():
        if isinstance(v, np.ndarray) and issubclass(v.dtype.type, np.integer):
            input_dict[k] = v.astype(np.float) # Core ML only accepts floats

    if frontend_only or _macos_version() < (10, 13) \
       or (mlmodel.is_package and _macos_version() < (12, 0)):
        return mlmodel._spec, mlmodel, input_dict, None

    pred = None
    if not coremltoolsutils._has_custom_layer(mlmodel._spec):
        pred = compare_backend(
                mlmodel,
                input_dict,
                expected_outputs,
                atol=atol,
                rtol=rtol,
                also_compare_shapes=True,
                dtype=backend[1],
        )
    else:
        print('Skipping model prediction as it has a custom nn layer!')
    return mlmodel._spec, mlmodel, input_dict, pred


def run_compare_tf_keras(
    model,
    input_values,
    inputs_for_conversion=None,
    compute_unit=ct.ComputeUnit.CPU_ONLY,
    frontend_only=False,
    frontend="tensorflow",
    backend=("neuralnetwork", "fp32"),
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
    inputs_for_conversion: list of coremltools.TensorType() or coremltools.ImageType() objects
        Defaults to None. It is passed as is to the "inputs" argument of the converter.
    compute_unit: Enum[ct.ComputeUnit]
        Compute unit for the coreml model
    frontend_only: bool
        If True, skip the prediction call, only validate conversion.
    frontend: str
        Frontend to convert from.
    backend: str
        Backend to convert to.
    atol: float
        The absolute tolerance parameter.
    rtol: float
        The relative tolerance parameter.
    """
    mlmodel = ct_convert(model, inputs=inputs_for_conversion, source=frontend, convert_to=backend,
                         compute_units=compute_unit)

    # assumes conversion preserve the i/o names
    proto = mlmodel._spec
    inputs = [i.name.split(":")[0].strip() for i in model.inputs]
    outputs = [str(o.name) for o in proto.description.output]

    # get tf.keras model output as reference and run comparison
    keras_outputs = model(input_values)
    if not isinstance(keras_outputs, list):
        keras_outputs = [keras_outputs]
    ref = [output.numpy() for output in keras_outputs]
    expected_outputs = {n: v for n, v in zip(outputs, ref)}
    input_key_values = {n: v for n, v in zip(inputs, input_values)}

    if frontend_only or _macos_version() < (10, 13) \
       or (mlmodel.is_package and _macos_version() < (12, 0)):
        return proto, mlmodel, input_key_values, None

    pred = None
    if not coremltoolsutils._has_custom_layer(proto):
        pred = compare_backend(
                mlmodel,
                input_key_values,
                expected_outputs,
                atol=atol,
                rtol=rtol,
                also_compare_shapes=True,
                dtype=backend[1]
        )
    else:
        print('Skipping model prediction as it has a custom nn layer!')
    return proto, mlmodel, input_key_values, pred


class TensorFlow2BaseTest(TensorFlowBaseTest):

    @staticmethod
    def run_compare_tf2(model,
                        input_dict,
                        output_names,
                        inputs_for_conversion=None,
                        compute_unit=ct.ComputeUnit.CPU_ONLY,
                        frontend_only=False,
                        frontend="tensorflow",
                        backend=("neuralnetwork", "fp32"),
                        debug=False,
                        atol=1e-04,
                        rtol=1e-05,
                        minimum_deployment_target=None,):
        res = run_compare_tf2(model,
                              input_dict,
                              output_names,
                              inputs_for_conversion=inputs_for_conversion,
                              compute_unit=compute_unit,
                              frontend_only=frontend_only,
                              frontend=frontend,
                              backend=backend,
                              debug=debug,
                              atol=atol,
                              rtol=rtol,
                              minimum_deployment_target=minimum_deployment_target,)
        alist = list(res)
        alist.append(TensorFlow2BaseTest.testclassname)
        alist.append(TensorFlow2BaseTest.testmodelname)
        return tuple(alist)

    @staticmethod
    def run_compare_tf_keras(
            model,
            input_values,
            inputs_for_conversion=None,
            compute_unit=ct.ComputeUnit.CPU_ONLY,
            frontend_only=False,
            frontend="tensorflow",
            backend=("neuralnetwork", "fp32"),
            atol=1e-04,
            rtol=1e-05
        ):
        res = run_compare_tf_keras(model, input_values,
                                   inputs_for_conversion=inputs_for_conversion,
                                   compute_unit=compute_unit,
                                   frontend_only=frontend_only,
                                   frontend=frontend,
                                   backend=backend, atol=atol, rtol=rtol)
        alist = list(res)
        alist.append(TensorFlow2BaseTest.testclassname)
        alist.append(TensorFlow2BaseTest.testmodelname)
        return tuple(alist)
